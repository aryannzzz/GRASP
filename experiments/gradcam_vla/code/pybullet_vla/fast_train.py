"""
Feature Pre-Extraction and Fast Training for RefinedVLA

Since the CLIP backbone is frozen during training, we pre-compute all
patch features and text features once, then train solely on the cached
features. This eliminates the ViT forward pass (~5s/batch) and makes
training ~50-100x faster (~0.05s/batch).

This is a standard optimization for frozen-backbone training.
No architectural changes.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List
import time
import os

from vla_gradcam.refined_vla import RefinedVLA, load_refined_vla
from vla_gradcam.losses import VLALoss


def extract_features(
    data_dir: str,
    split: str,
    processor,
    model: RefinedVLA,
    device: str = "cpu",
    batch_size: int = 32,
) -> Dict[str, torch.Tensor]:
    """
    Pre-extract CLIP patch features and text features for a split.

    Returns dict of tensors:
        patches: [N, 196, 768]
        text_features: [N, 512]
        actions: [N, 7]
        object_labels: [N]
        target_bboxes: [N, 4]
    """
    data_dir = Path(data_dir)
    split_dir = data_dir / split
    if split_dir.exists() and (split_dir / "demonstrations.json").exists():
        demos_path = split_dir / "demonstrations.json"
    else:
        demos_path = data_dir / "demonstrations.json"

    with open(demos_path) as f:
        demonstrations = json.load(f)

    print(f"[{split}] Extracting features from {len(demonstrations)} demos...")

    all_patches = []
    all_text_feats = []
    all_actions = []
    all_labels = []
    all_bboxes = []

    model.eval()

    # Process in batches
    for i in tqdm(range(0, len(demonstrations), batch_size), desc=f"Extract {split}"):
        batch_demos = demonstrations[i:i+batch_size]

        # Load and process images
        images = []
        texts = []
        actions = []
        labels = []
        bboxes = []

        for demo in batch_demos:
            img_path = data_dir / demo["image_path"]
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            images.append(image)
            texts.append(demo["instruction"])
            actions.append(demo["action"])
            labels.append(demo.get("object_label", 0))

            target_bbox = demo.get("target_bbox", None)
            if target_bbox is not None and isinstance(target_bbox, dict):
                bbox = target_bbox.get("bbox", [0, 0, 1, 1])
            else:
                bbox = [0, 0, 1, 1]
            bboxes.append(bbox)

        if not images:
            continue

        # Process images
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)

        # Process text
        text_inputs = processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            # Extract patch features
            patches = model.encode_image_patches(pixel_values)[:, 1:, :]  # [B, 196, 768]
            # Extract text features
            text_feat = model.encode_text(text_inputs)  # [B, 512]

        all_patches.append(patches.cpu())
        all_text_feats.append(text_feat.cpu())
        all_actions.append(torch.tensor(actions, dtype=torch.float32))
        all_labels.append(torch.tensor(labels, dtype=torch.long))
        all_bboxes.append(torch.tensor(bboxes, dtype=torch.float32))

    result = {
        "patches": torch.cat(all_patches, dim=0),
        "text_features": torch.cat(all_text_feats, dim=0),
        "actions": torch.cat(all_actions, dim=0),
        "object_labels": torch.cat(all_labels, dim=0),
        "target_bboxes": torch.cat(all_bboxes, dim=0),
    }

    print(f"[{split}] Extracted: {result['patches'].shape[0]} samples, "
          f"patches={result['patches'].shape}, text={result['text_features'].shape}")

    return result


class PrecomputedDataset(Dataset):
    """Dataset from pre-extracted features."""

    def __init__(self, features: Dict[str, torch.Tensor]):
        self.patches = features["patches"]
        self.text_features = features["text_features"]
        self.actions = features["actions"]
        self.object_labels = features["object_labels"]
        self.target_bboxes = features["target_bboxes"]

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        return {
            "patches": self.patches[idx],
            "text_features": self.text_features[idx],
            "actions": self.actions[idx],
            "object_labels": self.object_labels[idx],
            "target_bboxes": self.target_bboxes[idx],
        }


def compute_attention_iou_batch(attn_weights, target_bboxes, n_patches_side=14):
    """Compute IoU between attention heatmap and target bboxes."""
    B = attn_weights.shape[0]
    ious = []

    for b in range(B):
        attn = attn_weights[b].detach().cpu().numpy()
        bbox = target_bboxes[b].detach().cpu().numpy()

        grid = attn[:n_patches_side**2].reshape(n_patches_side, n_patches_side)
        threshold = np.percentile(grid, 70)
        attn_mask = (grid >= threshold).astype(float)

        bbox_mask = np.zeros((n_patches_side, n_patches_side), dtype=float)
        x1 = int(np.clip(bbox[0] * n_patches_side, 0, n_patches_side - 1))
        y1 = int(np.clip(bbox[1] * n_patches_side, 0, n_patches_side - 1))
        x2 = int(np.clip(bbox[2] * n_patches_side, 0, n_patches_side))
        y2 = int(np.clip(bbox[3] * n_patches_side, 0, n_patches_side))
        bbox_mask[y1:y2, x1:x2] = 1.0

        intersection = (attn_mask * bbox_mask).sum()
        union = ((attn_mask + bbox_mask) > 0).sum()
        iou = intersection / (union + 1e-8)
        ious.append(iou)

    return np.mean(ious)


class FastVLATrainer:
    """
    Fast trainer using pre-extracted features.
    Only trains attention_pool + action_head + object_classifier.
    """

    def __init__(
        self,
        model: RefinedVLA,
        train_features: Dict[str, torch.Tensor],
        val_features: Dict[str, torch.Tensor],
        test_features: Optional[Dict[str, torch.Tensor]] = None,
        device: str = "cpu",
        lr: float = 1e-4,
        batch_size: int = 64,
        output_dir: str = "outputs/training",
        action_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        coverage_weight: float = 0.1,
        diversity_weight: float = 0.1,
    ):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Datasets
        self.train_loader = DataLoader(
            PrecomputedDataset(train_features), batch_size=batch_size,
            shuffle=True, num_workers=0, pin_memory=False,
        )
        self.val_loader = DataLoader(
            PrecomputedDataset(val_features), batch_size=batch_size,
            shuffle=False, num_workers=0,
        )
        self.test_loader = None
        if test_features is not None:
            self.test_loader = DataLoader(
                PrecomputedDataset(test_features), batch_size=batch_size,
                shuffle=False, num_workers=0,
            )

        # Loss
        self.criterion = VLALoss(
            action_weight=action_weight,
            contrastive_weight=contrastive_weight,
            coverage_weight=coverage_weight,
            diversity_weight=diversity_weight,
        )

        # Optimizer (only trainable params: attention_pool + action_head + object_classifier)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Batch size: {batch_size}")
        print(f"Batches/epoch: {len(self.train_loader)}")

        self.history = {
            "train_loss": [], "val_loss": [], "test_loss": [],
            "train_iou": [], "val_iou": [], "test_iou": [],
            "train_action_loss": [], "val_action_loss": [],
            "train_contrastive_loss": [], "val_contrastive_loss": [],
            "train_coverage_loss": [], "train_diversity_loss": [],
        }

    def _forward_batch(self, batch):
        """Forward pass from pre-computed features."""
        patches = batch["patches"].to(self.device)
        text_feat = batch["text_features"].to(self.device)
        actions_gt = batch["actions"].to(self.device)
        obj_labels = batch["object_labels"].to(self.device)
        target_bboxes = batch["target_bboxes"].to(self.device)

        # Forward from patches (skips CLIP ViT entirely)
        action, attn, obj_logits = self.model.forward_from_patches(patches, text_feat)

        return action, attn, obj_logits, text_feat, actions_gt, obj_labels, target_bboxes

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        totals = {"loss": 0, "action": 0, "contrastive": 0, "coverage": 0, "diversity": 0, "iou": 0}
        n = 0

        for batch in self.train_loader:
            action, attn, obj_logits, text_feat, actions_gt, obj_labels, bboxes = self._forward_batch(batch)

            losses = self.criterion(
                pred_action=action,
                target_action=actions_gt,
                attn_weights=attn,
                text_features=text_feat,
                obj_logits=obj_logits,
                obj_labels=obj_labels,
            )

            self.optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0
            )
            self.optimizer.step()

            iou = compute_attention_iou_batch(attn, bboxes)
            for k in ["loss", "action", "contrastive", "coverage", "diversity"]:
                key = k if k == "loss" else k
                totals[k] += losses.get(k if k != "loss" else "total").item()
            totals["iou"] += iou
            n += 1

        # Fix: recalculate loss properly
        totals_out = {}
        totals_out["loss"] = totals["loss"] / max(n, 1)
        totals_out["action"] = totals["action"] / max(n, 1)
        totals_out["contrastive"] = totals["contrastive"] / max(n, 1)
        totals_out["coverage"] = totals["coverage"] / max(n, 1)
        totals_out["diversity"] = totals["diversity"] / max(n, 1)
        totals_out["iou"] = totals["iou"] / max(n, 1)

        return totals_out

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        totals = {"loss": 0, "action": 0, "contrastive": 0, "iou": 0}
        n = 0

        for batch in loader:
            action, attn, obj_logits, text_feat, actions_gt, obj_labels, bboxes = self._forward_batch(batch)

            losses = self.criterion(
                pred_action=action,
                target_action=actions_gt,
                attn_weights=attn,
                text_features=text_feat,
                obj_logits=obj_logits,
                obj_labels=obj_labels,
            )

            iou = compute_attention_iou_batch(attn, bboxes)
            totals["loss"] += losses["total"].item()
            totals["action"] += losses["action"].item()
            totals["contrastive"] += losses["contrastive"].item()
            totals["iou"] += iou
            n += 1

        return {k: v / max(n, 1) for k, v in totals.items()}

    def train(self, n_epochs: int = 100, patience: int = 20):
        print(f"\nTraining for up to {n_epochs} epochs (patience={patience})")
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, n_epochs + 1):
            train_m = self.train_epoch()
            self.scheduler.step()

            val_m = self.evaluate(self.val_loader)
            test_m = self.evaluate(self.test_loader) if self.test_loader else {}

            # Log
            self.history["train_loss"].append(train_m["loss"])
            self.history["val_loss"].append(val_m["loss"])
            self.history["train_iou"].append(train_m["iou"])
            self.history["val_iou"].append(val_m["iou"])
            self.history["train_action_loss"].append(train_m["action"])
            self.history["val_action_loss"].append(val_m["action"])
            self.history["train_contrastive_loss"].append(train_m["contrastive"])
            self.history["val_contrastive_loss"].append(val_m["contrastive"])
            self.history["train_coverage_loss"].append(train_m["coverage"])
            self.history["train_diversity_loss"].append(train_m["diversity"])
            if test_m:
                self.history["test_loss"].append(test_m["loss"])
                self.history["test_iou"].append(test_m["iou"])

            elapsed = time.time() - start_time
            test_str = f" | Test L={test_m['loss']:.4f} IoU={test_m['iou']:.3f}" if test_m else ""

            print(
                f"E{epoch:3d} [{elapsed/60:.1f}m] "
                f"Train L={train_m['loss']:.4f} IoU={train_m['iou']:.3f} A={train_m['action']:.4f} C={train_m['contrastive']:.4f} "
                f"| Val L={val_m['loss']:.4f} IoU={val_m['iou']:.3f}"
                f"{test_str}"
            )

            if val_m["loss"] < best_val_loss:
                best_val_loss = val_m["loss"]
                patience_counter = 0
                self._save_checkpoint(epoch, val_m, "best_model.pt")
            else:
                patience_counter += 1

            if epoch % 25 == 0:
                self._save_checkpoint(epoch, val_m, f"checkpoint_ep{epoch}.pt")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Save history
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
        if self.history["test_iou"]:
            best_epoch = np.argmin(self.history["val_loss"])
            print(f"Test IoU at best epoch: {self.history['test_iou'][best_epoch]:.3f}")

        return self.history

    def _save_checkpoint(self, epoch, metrics, filename):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }, self.output_dir / filename)


def run_fast_training(
    data_dir: str,
    output_dir: str,
    device: str = "cpu",
    seed: int = 42,
    epochs: int = 100,
    patience: int = 20,
    batch_size: int = 64,
    lr: float = 1e-4,
    action_weight: float = 1.0,
    contrastive_weight: float = 0.5,
    coverage_weight: float = 0.1,
    diversity_weight: float = 0.1,
    clip_model_name: str = "openai/clip-vit-base-patch16",
):
    """Run fast training with feature pre-extraction."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print(f"FAST TRAINING (seed={seed})")
    print("=" * 60)

    # Load model
    model, processor = load_refined_vla(clip_model_name, device=device)

    # Check for cached features
    cache_dir = Path(data_dir) / "feature_cache"
    cache_dir.mkdir(exist_ok=True)

    splits = {}
    for split in ["train", "val", "test"]:
        cache_path = cache_dir / f"{split}_features.pt"
        if cache_path.exists():
            print(f"Loading cached {split} features...")
            splits[split] = torch.load(cache_path, map_location="cpu")
        else:
            print(f"Extracting {split} features...")
            splits[split] = extract_features(
                data_dir, split, processor, model, device=device, batch_size=16
            )
            torch.save(splits[split], cache_path)
            print(f"Cached to {cache_path}")

    # Train
    trainer = FastVLATrainer(
        model=model,
        train_features=splits["train"],
        val_features=splits["val"],
        test_features=splits["test"],
        device=device,
        lr=lr,
        batch_size=batch_size,
        output_dir=output_dir,
        action_weight=action_weight,
        contrastive_weight=contrastive_weight,
        coverage_weight=coverage_weight,
        diversity_weight=diversity_weight,
    )

    history = trainer.train(n_epochs=epochs, patience=patience)
    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/training_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.5)
    parser.add_argument("--coverage-weight", type=float, default=0.1)
    parser.add_argument("--diversity-weight", type=float, default=0.1)
    args = parser.parse_args()

    run_fast_training(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        action_weight=args.action_weight,
        contrastive_weight=args.contrastive_weight,
        coverage_weight=args.coverage_weight,
        diversity_weight=args.diversity_weight,
    )
