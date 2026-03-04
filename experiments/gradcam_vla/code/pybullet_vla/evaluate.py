"""
Held-Out Evaluation Pipeline for RefinedVLA

Covers Phases 4-7 of the Tier 1 Reconstruction:

Phase 4: Held-Out Attribution Benchmark
  - Attention-only IoU
  - GradCAM IoU
  - Integrated Gradients IoU
  - Occlusion-based importance IoU
  - Counterfactual displacement
  - Cross-instruction correlation
  - Pointing accuracy

Phase 5: Causality Validation
  - Mask top-k salient patches
  - Mask bottom-k patches
  - Random masking baseline
  - Causal Ratio = Top-k Drop / Bottom-k Drop

Phase 6: Multi-Seed Stability
  - Mean, std, worst-case IoU across seeds
  - Mean, worst-case causal drop

Phase 7: Structured Ablation Table
  - Full model vs ablated variants
  - All metrics on held-out data only

IMPORTANT: No evaluation on training layouts in final results.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import time

from vla_gradcam.refined_vla import RefinedVLA, load_refined_vla
from vla_gradcam.attribution import UnifiedAttribution
from pybullet_vla.train import (
    VLADemonstrationDataset, collate_fn, compute_attention_iou
)


# ======================================================================
#  PHASE 4: HELD-OUT ATTRIBUTION BENCHMARK
# ======================================================================

def compute_heatmap_iou(heatmap: np.ndarray, bbox: np.ndarray, n_side: int = 14) -> float:
    """Compute IoU between a heatmap and bounding box on a patch grid."""
    grid = heatmap[:n_side**2].reshape(n_side, n_side)
    threshold = np.percentile(grid, 70)
    attn_mask = (grid >= threshold).astype(float)

    bbox_mask = np.zeros((n_side, n_side), dtype=float)
    x1 = int(np.clip(bbox[0] * n_side, 0, n_side - 1))
    y1 = int(np.clip(bbox[1] * n_side, 0, n_side - 1))
    x2 = int(np.clip(bbox[2] * n_side, 0, n_side))
    y2 = int(np.clip(bbox[3] * n_side, 0, n_side))
    bbox_mask[y1:y2, x1:x2] = 1.0

    intersection = (attn_mask * bbox_mask).sum()
    union = ((attn_mask + bbox_mask) > 0).sum()
    return intersection / (union + 1e-8)


def compute_pointing_accuracy(heatmap: np.ndarray, bbox: np.ndarray, n_side: int = 14) -> float:
    """Check if the peak attention patch falls within the target bbox."""
    grid = heatmap[:n_side**2].reshape(n_side, n_side)
    peak_idx = np.unravel_index(grid.argmax(), grid.shape)
    peak_y, peak_x = peak_idx

    # Normalize to [0, 1]
    peak_x_norm = (peak_x + 0.5) / n_side
    peak_y_norm = (peak_y + 0.5) / n_side

    x1, y1, x2, y2 = bbox
    inside = (x1 <= peak_x_norm <= x2) and (y1 <= peak_y_norm <= y2)
    return 1.0 if inside else 0.0


def compute_occlusion_importance(
    model: RefinedVLA,
    pixel_values: torch.Tensor,
    text_features: torch.Tensor,
    n_patches_side: int = 14,
    occlusion_value: float = 0.0,
) -> np.ndarray:
    """
    Compute occlusion-based patch importance.

    For each patch, zero it out and measure action change.
    Higher change = more important patch.
    """
    model.eval()
    device = pixel_values.device

    with torch.no_grad():
        patches_full = model.get_patch_features(pixel_values)  # [1, N, D]
        action_full, _, _ = model.forward_from_patches(patches_full, text_features)

    N = patches_full.shape[1]
    importance = np.zeros(N)

    for i in range(N):
        patches_occ = patches_full.clone()
        patches_occ[0, i, :] = occlusion_value

        with torch.no_grad():
            action_occ, _, _ = model.forward_from_patches(patches_occ, text_features)

        diff = (action_full - action_occ).abs().sum().item()
        importance[i] = diff

    # Normalize
    if importance.max() > importance.min():
        importance = (importance - importance.min()) / (importance.max() - importance.min())

    return importance


def run_attribution_benchmark(
    model: RefinedVLA,
    processor,
    test_loader: DataLoader,
    device: str = "cpu",
    ig_steps: int = 30,
    max_samples: int = 100,
) -> Dict:
    """
    Phase 4: Compute all attribution metrics on held-out data.
    """
    model.eval()
    attr = UnifiedAttribution(model)

    results = {
        "attention_iou": [], "gradcam_iou": [], "ig_iou": [], "occlusion_iou": [],
        "attention_pointing": [], "gradcam_pointing": [], "ig_pointing": [], "occlusion_pointing": [],
        "cross_instruction_corr_attention": [], "cross_instruction_corr_gradcam": [],
        "counterfactual_displacement": [],
    }

    n_side = model.n_patches_side
    sample_count = 0

    for batch in tqdm(test_loader, desc="Attribution benchmark"):
        B = batch["pixel_values"].shape[0]

        for b in range(B):
            if sample_count >= max_samples:
                break

            pv = batch["pixel_values"][b:b+1].to(device)
            bbox = batch["target_bboxes"][b].cpu().numpy()

            # Encode text
            single_text = {k: v[b:b+1] for k, v in batch["text_inputs"].items()}
            with torch.no_grad():
                tf = model.encode_text(single_text)

            # --- Attention ---
            attn_result = attr.compute_attention(pv, tf)
            results["attention_iou"].append(compute_heatmap_iou(attn_result.patch_scores, bbox, n_side))
            results["attention_pointing"].append(compute_pointing_accuracy(attn_result.patch_scores, bbox, n_side))

            # --- GradCAM ---
            gc_result = attr.compute_gradcam(pv, tf)
            results["gradcam_iou"].append(compute_heatmap_iou(gc_result.patch_scores, bbox, n_side))
            results["gradcam_pointing"].append(compute_pointing_accuracy(gc_result.patch_scores, bbox, n_side))

            # --- Integrated Gradients ---
            ig_result = attr.compute_integrated_gradients(pv, tf, n_steps=ig_steps)
            results["ig_iou"].append(compute_heatmap_iou(ig_result.patch_scores, bbox, n_side))
            results["ig_pointing"].append(compute_pointing_accuracy(ig_result.patch_scores, bbox, n_side))

            # --- Occlusion (expensive, subsample) ---
            if sample_count < 50:  # Limit expensive occlusion computation
                occ_scores = compute_occlusion_importance(model, pv, tf, n_side)
                results["occlusion_iou"].append(compute_heatmap_iou(occ_scores, bbox, n_side))
                results["occlusion_pointing"].append(compute_pointing_accuracy(occ_scores, bbox, n_side))

            # --- Cross-instruction correlation ---
            if b + 1 < B:
                single_text_b = {k: v[b+1:b+2] for k, v in batch["text_inputs"].items()}
                with torch.no_grad():
                    tf_b = model.encode_text(single_text_b)

                corr_attn = attr.compute_instruction_correlation(pv, tf, tf_b, method="attention")
                corr_gc = attr.compute_instruction_correlation(pv, tf, tf_b, method="gradcam")
                results["cross_instruction_corr_attention"].append(corr_attn)
                results["cross_instruction_corr_gradcam"].append(corr_gc)

            # --- Counterfactual displacement ---
            with torch.no_grad():
                action_orig = model(pv, tf)[0][0].cpu().numpy()

            # Counterfactual: swap instruction (use generic different instruction)
            cf_text = processor(text=["do nothing"], return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                tf_cf = model.encode_text(cf_text)
                action_cf = model(pv, tf_cf)[0][0].cpu().numpy()

            cf_disp = np.linalg.norm(action_orig - action_cf)
            results["counterfactual_displacement"].append(float(cf_disp))

            sample_count += 1

        if sample_count >= max_samples:
            break

    # Aggregate
    summary = {}
    for key, values in results.items():
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))

    attr.remove_hooks()
    return summary


# ======================================================================
#  PHASE 5: CAUSALITY VALIDATION
# ======================================================================

def causal_masking_test(
    model: RefinedVLA,
    pixel_values: torch.Tensor,
    text_features: torch.Tensor,
    method_scores: np.ndarray,
    k_frac: float = 0.15,
) -> Dict[str, float]:
    """
    Mask top-k and bottom-k salient patches and measure action degradation.

    Args:
        model: RefinedVLA
        pixel_values: [1, 3, H, W]
        text_features: [1, D]
        method_scores: [N] attribution scores
        k_frac: Fraction of patches to mask

    Returns:
        Dict with top_k_drop, bottom_k_drop, random_drop, causal_ratio
    """
    model.eval()
    device = pixel_values.device
    N = len(method_scores)
    k = max(1, int(N * k_frac))

    with torch.no_grad():
        patches = model.get_patch_features(pixel_values)
        action_full, _, _ = model.forward_from_patches(patches, text_features)
        action_full = action_full[0].cpu().numpy()

    # Sort patches by importance
    sorted_idx = np.argsort(method_scores)[::-1]  # highest first
    top_k_idx = sorted_idx[:k]
    bottom_k_idx = sorted_idx[-k:]
    random_idx = np.random.choice(N, k, replace=False)

    def mask_and_measure(indices):
        patches_masked = patches.clone()
        for idx in indices:
            patches_masked[0, idx, :] = 0.0
        with torch.no_grad():
            action_masked, _, _ = model.forward_from_patches(patches_masked, text_features)
        return np.linalg.norm(action_full - action_masked[0].cpu().numpy())

    top_k_drop = mask_and_measure(top_k_idx)
    bottom_k_drop = mask_and_measure(bottom_k_idx)
    random_drop = mask_and_measure(random_idx)

    causal_ratio = top_k_drop / (bottom_k_drop + 1e-8)

    return {
        "top_k_drop": float(top_k_drop),
        "bottom_k_drop": float(bottom_k_drop),
        "random_drop": float(random_drop),
        "causal_ratio": float(causal_ratio),
    }


def run_causality_validation(
    model: RefinedVLA,
    processor,
    test_loader: DataLoader,
    device: str = "cpu",
    max_samples: int = 100,
    ig_steps: int = 30,
) -> Dict:
    """
    Phase 5: Causality validation on held-out layouts.
    """
    model.eval()
    attr = UnifiedAttribution(model)

    methods = ["attention", "gradcam", "integrated_gradients"]
    causal_results = {m: {"top_k_drop": [], "bottom_k_drop": [], "random_drop": [], "causal_ratio": []} for m in methods}

    sample_count = 0

    for batch in tqdm(test_loader, desc="Causality validation"):
        B = batch["pixel_values"].shape[0]
        for b in range(B):
            if sample_count >= max_samples:
                break

            pv = batch["pixel_values"][b:b+1].to(device)
            single_text = {k: v[b:b+1] for k, v in batch["text_inputs"].items()}
            with torch.no_grad():
                tf = model.encode_text(single_text)

            # Compute attributions
            attn_result = attr.compute_attention(pv, tf)
            gc_result = attr.compute_gradcam(pv, tf)
            ig_result = attr.compute_integrated_gradients(pv, tf, n_steps=ig_steps)

            scores_map = {
                "attention": attn_result.patch_scores,
                "gradcam": gc_result.patch_scores,
                "integrated_gradients": ig_result.patch_scores,
            }

            for method_name, scores in scores_map.items():
                cm = causal_masking_test(model, pv, tf, scores)
                for key in causal_results[method_name]:
                    causal_results[method_name][key].append(cm[key])

            sample_count += 1

        if sample_count >= max_samples:
            break

    # Aggregate
    summary = {}
    for method_name in methods:
        for metric_name, values in causal_results[method_name].items():
            summary[f"{method_name}_{metric_name}_mean"] = float(np.mean(values))
            summary[f"{method_name}_{metric_name}_std"] = float(np.std(values))

    attr.remove_hooks()
    return summary


# ======================================================================
#  PHASE 6: MULTI-SEED STABILITY
# ======================================================================

def run_multi_seed_evaluation(
    data_dir: str,
    model_dirs: List[str],
    device: str = "cpu",
    clip_model_name: str = "openai/clip-vit-base-patch16",
    max_samples: int = 50,
    ig_steps: int = 20,
) -> Dict:
    """
    Phase 6: Train and evaluate across multiple seeds.

    Args:
        data_dir: Dataset directory
        model_dirs: List of checkpoint directories (one per seed)
        device: Device
        clip_model_name: CLIP model name
        max_samples: Max samples for evaluation per seed

    Returns:
        Multi-seed stability metrics
    """
    seed_ious = {"attention": [], "gradcam": [], "ig": []}
    seed_causal_ratios = {"attention": [], "gradcam": [], "ig": []}
    seed_pointings = {"attention": [], "gradcam": [], "ig": []}

    for model_dir in model_dirs:
        ckpt_path = Path(model_dir) / "best_model.pt"
        if not ckpt_path.exists():
            print(f"Skipping {model_dir}: no checkpoint")
            continue

        # Load model
        model, processor = load_refined_vla(clip_model_name, device=device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Load test data
        test_ds = VLADemonstrationDataset(data_dir, processor, split="test")
        test_loader = DataLoader(
            test_ds, batch_size=8, shuffle=False, num_workers=0,
            collate_fn=lambda batch: collate_fn(batch, processor, device),
        )

        # Attribution benchmark
        attr_results = run_attribution_benchmark(
            model, processor, test_loader, device,
            ig_steps=ig_steps, max_samples=max_samples
        )

        # Causality validation
        causal_results = run_causality_validation(
            model, processor, test_loader, device,
            max_samples=max_samples // 2, ig_steps=ig_steps
        )

        # Collect per-seed metrics
        seed_ious["attention"].append(attr_results.get("attention_iou_mean", 0))
        seed_ious["gradcam"].append(attr_results.get("gradcam_iou_mean", 0))
        seed_ious["ig"].append(attr_results.get("ig_iou_mean", 0))

        seed_causal_ratios["attention"].append(causal_results.get("attention_causal_ratio_mean", 0))
        seed_causal_ratios["gradcam"].append(causal_results.get("gradcam_causal_ratio_mean", 0))
        seed_causal_ratios["ig"].append(causal_results.get("integrated_gradients_causal_ratio_mean", 0))

        seed_pointings["attention"].append(attr_results.get("attention_pointing_mean", 0))
        seed_pointings["gradcam"].append(attr_results.get("gradcam_pointing_mean", 0))
        seed_pointings["ig"].append(attr_results.get("ig_pointing_mean", 0))

    summary = {}
    for method in ["attention", "gradcam", "ig"]:
        ious = seed_ious[method]
        crs = seed_causal_ratios[method]
        pts = seed_pointings[method]

        if ious:
            summary[f"{method}_iou_mean"] = float(np.mean(ious))
            summary[f"{method}_iou_std"] = float(np.std(ious))
            summary[f"{method}_iou_worst"] = float(np.min(ious))
            summary[f"{method}_causal_ratio_mean"] = float(np.mean(crs))
            summary[f"{method}_causal_ratio_worst"] = float(np.min(crs))
            summary[f"{method}_pointing_mean"] = float(np.mean(pts))
            summary[f"{method}_pointing_worst"] = float(np.min(pts))

    return summary


# ======================================================================
#  PHASE 7: STRUCTURED ABLATION TABLE
# ======================================================================

def run_single_ablation(
    model: RefinedVLA,
    processor,
    test_loader: DataLoader,
    device: str,
    variant_name: str,
    max_samples: int = 50,
    ig_steps: int = 20,
) -> Dict:
    """Run full evaluation for one ablation variant."""
    model.eval()

    # Attribution benchmark
    attr_results = run_attribution_benchmark(
        model, processor, test_loader, device,
        ig_steps=ig_steps, max_samples=max_samples
    )

    # Causality validation
    causal_results = run_causality_validation(
        model, processor, test_loader, device,
        max_samples=max_samples // 2, ig_steps=ig_steps
    )

    return {
        "variant": variant_name,
        "heldout_attn_iou": attr_results.get("attention_iou_mean", 0),
        "heldout_gc_iou": attr_results.get("gradcam_iou_mean", 0),
        "heldout_ig_iou": attr_results.get("ig_iou_mean", 0),
        "heldout_occ_iou": attr_results.get("occlusion_iou_mean", 0),
        "cross_instr_corr": attr_results.get("cross_instruction_corr_gradcam_mean", 0),
        "cf_displacement": attr_results.get("counterfactual_displacement_mean", 0),
        "attn_pointing": attr_results.get("attention_pointing_mean", 0),
        "gc_pointing": attr_results.get("gradcam_pointing_mean", 0),
        "ig_pointing": attr_results.get("ig_pointing_mean", 0),
        "attn_causal_ratio": causal_results.get("attention_causal_ratio_mean", 0),
        "gc_causal_ratio": causal_results.get("gradcam_causal_ratio_mean", 0),
        "ig_causal_ratio": causal_results.get("integrated_gradients_causal_ratio_mean", 0),
    }


def format_ablation_table(rows: List[Dict]) -> str:
    """Format results as a markdown table."""
    headers = [
        "Variant", "HeldOut IoU (Attn)", "HeldOut IoU (GC)", "HeldOut IoU (IG)",
        "Corr", "CF Disp", "Pointing (GC)", "Causal Ratio (GC)"
    ]

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        vals = [
            row["variant"],
            f"{row['heldout_attn_iou']:.3f}",
            f"{row['heldout_gc_iou']:.3f}",
            f"{row['heldout_ig_iou']:.3f}",
            f"{row['cross_instr_corr']:.3f}",
            f"{row['cf_displacement']:.3f}",
            f"{row['gc_pointing']:.3f}",
            f"{row['gc_causal_ratio']:.2f}",
        ]
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


# ======================================================================
#  FULL PIPELINE
# ======================================================================

def run_full_evaluation(
    model_path: str,
    data_dir: str,
    output_dir: str,
    device: str = "cpu",
    clip_model_name: str = "openai/clip-vit-base-patch16",
    max_samples: int = 100,
    ig_steps: int = 30,
):
    """
    Run Phases 4 and 5 evaluation on a single trained model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, processor = load_refined_vla(clip_model_name, device=device)
    if Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {model_path}")
    model.eval()

    # Load test data
    test_ds = VLADemonstrationDataset(data_dir, processor, split="test")
    test_loader = DataLoader(
        test_ds, batch_size=8, shuffle=False, num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, processor, device),
    )

    # Phase 4
    print("\n=== PHASE 4: Attribution Benchmark ===")
    attr_results = run_attribution_benchmark(
        model, processor, test_loader, device,
        ig_steps=ig_steps, max_samples=max_samples
    )
    print("\nAttribution Results (Held-Out):")
    for k, v in sorted(attr_results.items()):
        print(f"  {k}: {v:.4f}")

    # Phase 5
    print("\n=== PHASE 5: Causality Validation ===")
    causal_results = run_causality_validation(
        model, processor, test_loader, device,
        max_samples=max_samples // 2, ig_steps=ig_steps
    )
    print("\nCausality Results (Held-Out):")
    for k, v in sorted(causal_results.items()):
        print(f"  {k}: {v:.4f}")

    # Combine and save
    all_results = {
        "phase4_attribution": attr_results,
        "phase5_causality": causal_results,
        "model_path": model_path,
        "data_dir": data_dir,
        "max_samples": max_samples,
        "ig_steps": ig_steps,
    }

    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Final verdict
    heldout_iou = attr_results.get("gradcam_iou_mean", 0)
    heldout_pointing = attr_results.get("gradcam_pointing_mean", 0)
    causal_ratio = causal_results.get("gradcam_causal_ratio_mean", 0)
    gradcam_delta = attr_results.get("gradcam_iou_mean", 0) - attr_results.get("attention_iou_mean", 0)

    print("\n" + "=" * 60)
    print("TIER 1 VERDICT (Single Seed)")
    print("=" * 60)
    print(f"HeldOut_IoU (GradCAM):   {heldout_iou:.3f}  {'PASS' if heldout_iou > 0.5 else 'FAIL'}")
    print(f"HeldOut_Pointing (GC):   {heldout_pointing:.3f}  {'PASS' if heldout_pointing > 0.8 else 'FAIL'}")
    print(f"Causal_Ratio (GC):       {causal_ratio:.2f}  {'PASS' if causal_ratio >= 5 else 'FAIL'}")
    print(f"GradCAM_vs_Attention:    {gradcam_delta:+.3f}")
    print("=" * 60)

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RefinedVLA (Phases 4-7)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--ig-steps", type=int, default=30)
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch16")

    # Multi-seed mode
    parser.add_argument("--multi-seed", action="store_true", help="Run Phase 6 multi-seed evaluation")
    parser.add_argument("--seed-dirs", nargs="+", help="Checkpoint directories for each seed")

    # Ablation mode
    parser.add_argument("--ablation", action="store_true", help="Run Phase 7 ablation")
    parser.add_argument("--ablation-dirs", nargs="+", help="Checkpoint directories for ablation variants")
    parser.add_argument("--ablation-names", nargs="+", help="Names for ablation variants")

    args = parser.parse_args()

    if args.multi_seed and args.seed_dirs:
        print("=== PHASE 6: Multi-Seed Stability ===")
        results = run_multi_seed_evaluation(
            data_dir=args.data_dir,
            model_dirs=args.seed_dirs,
            device=args.device,
            clip_model_name=args.clip_model,
            max_samples=args.max_samples,
        )
        print("\nMulti-Seed Results:")
        for k, v in sorted(results.items()):
            print(f"  {k}: {v:.4f}")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "multi_seed_results.json", 'w') as f:
            json.dump(results, f, indent=2)

    elif args.ablation and args.ablation_dirs:
        print("=== PHASE 7: Ablation Table ===")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ablation_rows = []
        names = args.ablation_names or [f"variant_{i}" for i in range(len(args.ablation_dirs))]

        for model_dir, name in zip(args.ablation_dirs, names):
            ckpt_path = Path(model_dir) / "best_model.pt"
            model, processor = load_refined_vla(args.clip_model, device=args.device)
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=args.device)
                model.load_state_dict(ckpt["model_state_dict"])

            test_ds = VLADemonstrationDataset(args.data_dir, processor, split="test")
            test_loader = DataLoader(
                test_ds, batch_size=8, shuffle=False, num_workers=0,
                collate_fn=lambda batch: collate_fn(batch, processor, args.device),
            )

            row = run_single_ablation(
                model, processor, test_loader, args.device,
                variant_name=name,
                max_samples=args.max_samples,
            )
            ablation_rows.append(row)
            print(f"  {name}: IoU={row['heldout_gc_iou']:.3f}, CR={row['gc_causal_ratio']:.2f}")

        # Format and save
        table = format_ablation_table(ablation_rows)
        print("\n" + table)

        with open(output_dir / "ablation_table.md", 'w') as f:
            f.write(table)

        with open(output_dir / "ablation_results.json", 'w') as f:
            json.dump(ablation_rows, f, indent=2)

    else:
        # Single model evaluation (Phases 4+5)
        run_full_evaluation(
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            clip_model_name=args.clip_model,
            max_samples=args.max_samples,
            ig_steps=args.ig_steps,
        )


if __name__ == "__main__":
    main()
