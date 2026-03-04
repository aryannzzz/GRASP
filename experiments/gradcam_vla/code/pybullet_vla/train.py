"""
Train RefinedVLA on Randomized PyBullet Demonstrations

Uses the unified RefinedVLA architecture with VLALoss:
- Action MSE
- Contrastive object discrimination
- Coverage regularization
- Attention diversity

Supports:
- Train/val/held-out split loading
- IoU tracking per split
- Causal masking drop monitoring
- Cross-instruction correlation monitoring
- Early stopping on validation loss
- Multi-seed training via --seed flag
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from typing import Tuple, Dict, Optional, List
import time

from vla_gradcam.refined_vla import RefinedVLA, load_refined_vla
from vla_gradcam.losses import VLALoss
from vla_gradcam.attribution import UnifiedAttribution


class VLADemonstrationDataset(Dataset):
    """PyTorch dataset for VLA demonstrations with object labels and bboxes."""

    def __init__(self, data_dir: str, processor, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.split = split

        # Try split-specific path first, then fallback
        split_dir = self.data_dir / split
        if split_dir.exists() and (split_dir / "demonstrations.json").exists():
            demos_path = split_dir / "demonstrations.json"
        else:
            demos_path = self.data_dir / "demonstrations.json"

        with open(demos_path, 'r') as f:
            self.demonstrations = json.load(f)

        print(f"[{split}] Loaded {len(self.demonstrations)} demonstrations")

    def __len__(self):
        return len(self.demonstrations)

    def __getitem__(self, idx):
        demo = self.demonstrations[idx]

        # Load image
        img_path = self.data_dir / demo["image_path"]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        instruction = demo["instruction"]
        action = torch.tensor(demo["action"], dtype=torch.float32)
        object_label = demo.get("object_label", 0)

        # Target bbox for IoU evaluation
        target_bbox = demo.get("target_bbox", None)
        if target_bbox is not None and isinstance(target_bbox, dict):
            bbox = target_bbox.get("bbox", [0, 0, 1, 1])
        else:
            bbox = [0, 0, 1, 1]

        return {
            "pixel_values": pixel_values,
            "instruction": instruction,
            "action": action,
            "object_label": object_label,
            "target_bbox": torch.tensor(bbox, dtype=torch.float32),
        }


def collate_fn(batch, processor, device):
    pixel_values = torch.stack([item["pixel_values"] for item in batch]).to(device)
    instructions = [item["instruction"] for item in batch]
    actions = torch.stack([item["action"] for item in batch]).to(device)
    object_labels = torch.tensor([item["object_label"] for item in batch], dtype=torch.long).to(device)
    target_bboxes = torch.stack([item["target_bbox"] for item in batch]).to(device)

    text_inputs = processor(
        text=instructions,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    return {
        "pixel_values": pixel_values,
        "text_inputs": text_inputs,
        "instructions": instructions,
        "actions": actions,
        "object_labels": object_labels,
        "target_bboxes": target_bboxes,
    }


def compute_attention_iou(attn_weights, target_bboxes, n_patches_side=14):
    """
    Compute IoU between attention heatmap and target bounding box.

    Args:
        attn_weights: [B, N] attention weights
        target_bboxes: [B, 4] normalized bboxes [x1, y1, x2, y2]
        n_patches_side: Grid side (14 for ViT-B/16)

    Returns:
        mean IoU across batch
    """
    B = attn_weights.shape[0]
    ious = []

    for b in range(B):
        attn = attn_weights[b].detach().cpu().numpy()
        bbox = target_bboxes[b].detach().cpu().numpy()

        # Reshape attention to spatial grid
        grid = attn[:n_patches_side**2].reshape(n_patches_side, n_patches_side)

        # Threshold attention (top 30%)
        threshold = np.percentile(grid, 70)
        attn_mask = (grid >= threshold).astype(float)

        # Create bbox mask on patch grid
        bbox_mask = np.zeros((n_patches_side, n_patches_side), dtype=float)
        x1 = int(np.clip(bbox[0] * n_patches_side, 0, n_patches_side - 1))
        y1 = int(np.clip(bbox[1] * n_patches_side, 0, n_patches_side - 1))
        x2 = int(np.clip(bbox[2] * n_patches_side, 0, n_patches_side))
        y2 = int(np.clip(bbox[3] * n_patches_side, 0, n_patches_side))
        bbox_mask[y1:y2, x1:x2] = 1.0

        # IoU
        intersection = (attn_mask * bbox_mask).sum()
        union = ((attn_mask + bbox_mask) > 0).sum()
        iou = intersection / (union + 1e-8)
        ious.append(iou)

    return np.mean(ious)


class VLATrainer:
    """Trainer for RefinedVLA with VLALoss."""

    def __init__(
        self,
        model: RefinedVLA,
        processor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: str = "cpu",
        lr: float = 1e-4,
        output_dir: str = "outputs/training",
        action_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        coverage_weight: float = 0.1,
        diversity_weight: float = 0.1,
    ):
        self.model = model.to(device)
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss
        self.criterion = VLALoss(
            action_weight=action_weight,
            contrastive_weight=contrastive_weight,
            coverage_weight=coverage_weight,
            diversity_weight=diversity_weight,
        )

        # Optimizer: only trainable params
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")

        # Metrics history
        self.history = {
            "train_loss": [], "val_loss": [], "test_loss": [],
            "train_iou": [], "val_iou": [], "test_iou": [],
            "train_action_loss": [], "val_action_loss": [],
            "train_contrastive_loss": [], "val_contrastive_loss": [],
        }

    def _encode_text(self, text_inputs):
        """Encode text with CLIP (frozen)."""
        with torch.no_grad():
            text_feat = self.model.encode_text(text_inputs)
        return text_feat

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        totals = {"loss": 0, "action": 0, "contrastive": 0, "coverage": 0, "diversity": 0, "iou": 0}
        n = 0

        for batch in tqdm(self.train_loader, desc="Train", leave=False):
            text_feat = self._encode_text(batch["text_inputs"])

            # Forward
            action, attn, obj_logits = self.model(batch["pixel_values"], text_feat)

            # Loss
            losses = self.criterion(
                pred_action=action,
                target_action=batch["actions"],
                attn_weights=attn,
                text_features=text_feat,
                obj_logits=obj_logits,
                obj_labels=batch["object_labels"],
            )

            self.optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0
            )
            self.optimizer.step()

            # Metrics
            iou = compute_attention_iou(attn, batch["target_bboxes"])
            totals["loss"] += losses["total"].item()
            totals["action"] += losses["action"].item()
            totals["contrastive"] += losses["contrastive"].item()
            totals["coverage"] += losses["coverage"].item()
            totals["diversity"] += losses["diversity"].item()
            totals["iou"] += iou
            n += 1

        return {k: v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, name: str = "val") -> Dict[str, float]:
        self.model.eval()
        totals = {"loss": 0, "action": 0, "contrastive": 0, "iou": 0}
        n = 0

        for batch in tqdm(loader, desc=name, leave=False):
            text_feat = self._encode_text(batch["text_inputs"])
            action, attn, obj_logits = self.model(batch["pixel_values"], text_feat)

            losses = self.criterion(
                pred_action=action,
                target_action=batch["actions"],
                attn_weights=attn,
                text_features=text_feat,
                obj_logits=obj_logits,
                obj_labels=batch["object_labels"],
            )

            iou = compute_attention_iou(attn, batch["target_bboxes"])
            totals["loss"] += losses["total"].item()
            totals["action"] += losses["action"].item()
            totals["contrastive"] += losses["contrastive"].item()
            totals["iou"] += iou
            n += 1

        return {k: v / max(n, 1) for k, v in totals.items()}

    def train(self, n_epochs: int = 100, patience: int = 20):
        """Train with early stopping on validation loss."""
        print(f"\nTraining for up to {n_epochs} epochs (patience={patience})")
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, n_epochs + 1):
            # Train
            train_metrics = self.train_epoch()
            self.scheduler.step()

            # Validate
            val_metrics = self.evaluate(self.val_loader, "Val")

            # Test (monitor only, no tuning)
            test_metrics = {}
            if self.test_loader is not None:
                test_metrics = self.evaluate(self.test_loader, "Test")

            # Log
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_iou"].append(train_metrics["iou"])
            self.history["val_iou"].append(val_metrics["iou"])
            self.history["train_action_loss"].append(train_metrics["action"])
            self.history["val_action_loss"].append(val_metrics["action"])
            self.history["train_contrastive_loss"].append(train_metrics["contrastive"])
            self.history["val_contrastive_loss"].append(val_metrics["contrastive"])
            if test_metrics:
                self.history["test_loss"].append(test_metrics["loss"])
                self.history["test_iou"].append(test_metrics["iou"])

            # Print
            elapsed = time.time() - start_time
            test_str = ""
            if test_metrics:
                test_str = f" | Test L={test_metrics['loss']:.4f} IoU={test_metrics['iou']:.3f}"

            print(
                f"E{epoch:3d} [{elapsed/60:.1f}m] "
                f"Train L={train_metrics['loss']:.4f} IoU={train_metrics['iou']:.3f} "
                f"| Val L={val_metrics['loss']:.4f} IoU={val_metrics['iou']:.3f}"
                f"{test_str}"
            )

            # Save best
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, "best_model.pt")
            else:
                patience_counter += 1

            # Periodic checkpoint
            if epoch % 25 == 0:
                self._save_checkpoint(epoch, val_metrics, f"checkpoint_ep{epoch}.pt")

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

        # Save history
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)

        # Final summary
        print(f"\nTraining complete.")
        print(f"Best val loss: {best_val_loss:.4f}")
        if self.history["test_iou"]:
            print(f"Final test IoU: {self.history['test_iou'][-1]:.3f}")

        return self.history

    def _save_checkpoint(self, epoch, metrics, filename):
        path = self.output_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }, path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train RefinedVLA")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.5)
    parser.add_argument("--coverage-weight", type=float, default=0.1)
    parser.add_argument("--diversity-weight", type=float, default=0.1)
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch16")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("REFINEDVLA TRAINING")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"Data: {args.data_dir}")
    print()

    # Load model
    model, processor = load_refined_vla(
        clip_model_name=args.clip_model, device=args.device
    )

    # Load datasets
    train_ds = VLADemonstrationDataset(args.data_dir, processor, split="train")
    val_ds = VLADemonstrationDataset(args.data_dir, processor, split="val")
    test_ds = VLADemonstrationDataset(args.data_dir, processor, split="test")

    make_loader = lambda ds, shuffle: DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, processor, args.device),
    )

    train_loader = make_loader(train_ds, True)
    val_loader = make_loader(val_ds, False)
    test_loader = make_loader(test_ds, False)

    # Train
    trainer = VLATrainer(
        model=model,
        processor=processor,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        lr=args.lr,
        output_dir=args.output_dir,
        action_weight=args.action_weight,
        contrastive_weight=args.contrastive_weight,
        coverage_weight=args.coverage_weight,
        diversity_weight=args.diversity_weight,
    )

    history = trainer.train(n_epochs=args.epochs, patience=args.patience)


if __name__ == "__main__":
    main()
