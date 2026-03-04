"""
Master Pipeline: End-to-end Tier 1 Reconstruction

Orchestrates all 7 phases:
  Phase 1: Architectural unification (already done)
  Phase 2: Randomized dataset generation
  Phase 3: Training
  Phase 4: Held-out attribution benchmark
  Phase 5: Causality validation
  Phase 6: Multi-seed stability
  Phase 7: Structured ablation table

Usage:
  # Full pipeline (all phases):
  python pybullet_vla/run_pipeline.py --all --data-dir outputs/dataset --output-dir outputs/

  # Generate dataset only:
  python pybullet_vla/run_pipeline.py --phase 2 --data-dir outputs/dataset

  # Train only:
  python pybullet_vla/run_pipeline.py --phase 3 --data-dir outputs/dataset --output-dir outputs/

  # Evaluate only (Phases 4+5):
  python pybullet_vla/run_pipeline.py --phase 45 --data-dir outputs/dataset --model-path outputs/training/best_model.pt

  # Multi-seed (Phase 6):
  python pybullet_vla/run_pipeline.py --phase 6 --data-dir outputs/dataset --output-dir outputs/ --n-seeds 5

  # Ablation (Phase 7):
  python pybullet_vla/run_pipeline.py --phase 7 --data-dir outputs/dataset --output-dir outputs/
"""

import sys
import os
import json
from pathlib import Path
import argparse
import time
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_phase2(args):
    """Phase 2: Generate randomized dataset."""
    print("\n" + "=" * 60)
    print("PHASE 2: RANDOMIZED DATASET GENERATION")
    print("=" * 60)

    from pybullet_vla.data_collector import DemonstrationCollector

    collector = DemonstrationCollector(
        output_dir=args.data_dir,
        image_size=224,
    )

    meta = collector.collect_with_splits(
        n_episodes=args.n_episodes,
        n_objects=3,
        scale_range=(0.8, 1.2),
        camera_jitter_std=0.05,
        n_distractors=0,
        base_seed=42,
    )

    print(f"\nPhase 2 complete. Dataset: {args.data_dir}")
    return meta


def run_phase3(args, seed=42):
    """Phase 3: Train one general model."""
    print("\n" + "=" * 60)
    print(f"PHASE 3: TRAINING (seed={seed})")
    print("=" * 60)

    from vla_gradcam.refined_vla import load_refined_vla
    from pybullet_vla.train import (
        VLADemonstrationDataset, VLATrainer, collate_fn
    )
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(args.output_dir) / f"training_seed{seed}"

    model, processor = load_refined_vla(device=args.device)

    train_ds = VLADemonstrationDataset(args.data_dir, processor, split="train")
    val_ds = VLADemonstrationDataset(args.data_dir, processor, split="val")
    test_ds = VLADemonstrationDataset(args.data_dir, processor, split="test")

    make_loader = lambda ds, shuffle: DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, processor, args.device),
    )

    trainer = VLATrainer(
        model=model,
        processor=processor,
        train_loader=make_loader(train_ds, True),
        val_loader=make_loader(val_ds, False),
        test_loader=make_loader(test_ds, False),
        device=args.device,
        lr=args.lr,
        output_dir=str(output_dir),
    )

    history = trainer.train(n_epochs=args.epochs, patience=args.patience)
    print(f"\nPhase 3 complete. Checkpoint: {output_dir}/best_model.pt")
    return history


def run_phase45(args, model_path=None):
    """Phases 4+5: Attribution benchmark + Causality validation."""
    print("\n" + "=" * 60)
    print("PHASES 4+5: ATTRIBUTION BENCHMARK + CAUSALITY VALIDATION")
    print("=" * 60)

    from pybullet_vla.evaluate import run_full_evaluation

    if model_path is None:
        model_path = str(Path(args.output_dir) / "training_seed42" / "best_model.pt")

    results = run_full_evaluation(
        model_path=model_path,
        data_dir=args.data_dir,
        output_dir=str(Path(args.output_dir) / "evaluation"),
        device=args.device,
        max_samples=args.max_eval_samples,
        ig_steps=args.ig_steps,
    )

    return results


def run_phase6(args):
    """Phase 6: Multi-seed stability."""
    print("\n" + "=" * 60)
    print("PHASE 6: MULTI-SEED STABILITY")
    print("=" * 60)

    seeds = list(range(42, 42 + args.n_seeds))
    model_dirs = []

    # Train each seed
    for seed in seeds:
        output_dir = Path(args.output_dir) / f"training_seed{seed}"
        if (output_dir / "best_model.pt").exists():
            print(f"Seed {seed}: checkpoint already exists, skipping training")
        else:
            run_phase3(args, seed=seed)
        model_dirs.append(str(output_dir))

    # Evaluate
    from pybullet_vla.evaluate import run_multi_seed_evaluation

    results = run_multi_seed_evaluation(
        data_dir=args.data_dir,
        model_dirs=model_dirs,
        device=args.device,
        max_samples=args.max_eval_samples,
        ig_steps=args.ig_steps,
    )

    output_dir = Path(args.output_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "multi_seed_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\nPhase 6 Results:")
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.4f}")

    return results


def run_phase7(args):
    """Phase 7: Structured ablation table."""
    print("\n" + "=" * 60)
    print("PHASE 7: STRUCTURED ABLATION TABLE")
    print("=" * 60)

    from vla_gradcam.refined_vla import load_refined_vla
    from pybullet_vla.train import VLADemonstrationDataset, VLATrainer, collate_fn
    from pybullet_vla.evaluate import run_single_ablation, format_ablation_table
    from torch.utils.data import DataLoader

    # Define ablation variants
    ablation_configs = [
        {"name": "Full model", "action_w": 1.0, "contrastive_w": 0.5, "coverage_w": 0.1, "diversity_w": 0.1},
        {"name": "No contrastive", "action_w": 1.0, "contrastive_w": 0.0, "coverage_w": 0.1, "diversity_w": 0.1},
        {"name": "No coverage", "action_w": 1.0, "contrastive_w": 0.5, "coverage_w": 0.0, "diversity_w": 0.1},
        {"name": "No diversity", "action_w": 1.0, "contrastive_w": 0.5, "coverage_w": 0.1, "diversity_w": 0.0},
        {"name": "Action only", "action_w": 1.0, "contrastive_w": 0.0, "coverage_w": 0.0, "diversity_w": 0.0},
    ]

    ablation_rows = []

    for config in ablation_configs:
        name = config["name"]
        variant_dir = Path(args.output_dir) / f"ablation_{name.replace(' ', '_').lower()}"

        # Train if needed
        if not (variant_dir / "best_model.pt").exists():
            print(f"\nTraining ablation variant: {name}")
            torch.manual_seed(42)
            np.random.seed(42)

            model, processor = load_refined_vla(device=args.device)
            train_ds = VLADemonstrationDataset(args.data_dir, processor, split="train")
            val_ds = VLADemonstrationDataset(args.data_dir, processor, split="val")
            test_ds = VLADemonstrationDataset(args.data_dir, processor, split="test")

            make_loader = lambda ds, shuffle: DataLoader(
                ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0,
                collate_fn=lambda batch: collate_fn(batch, processor, args.device),
            )

            trainer = VLATrainer(
                model=model, processor=processor,
                train_loader=make_loader(train_ds, True),
                val_loader=make_loader(val_ds, False),
                test_loader=make_loader(test_ds, False),
                device=args.device, lr=args.lr,
                output_dir=str(variant_dir),
                action_weight=config["action_w"],
                contrastive_weight=config["contrastive_w"],
                coverage_weight=config["coverage_w"],
                diversity_weight=config["diversity_w"],
            )
            trainer.train(n_epochs=args.epochs, patience=args.patience)

        # Evaluate
        model, processor = load_refined_vla(device=args.device)
        ckpt = torch.load(variant_dir / "best_model.pt", map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        test_ds = VLADemonstrationDataset(args.data_dir, processor, split="test")
        test_loader = DataLoader(
            test_ds, batch_size=8, shuffle=False, num_workers=0,
            collate_fn=lambda batch: collate_fn(batch, processor, args.device),
        )

        row = run_single_ablation(
            model, processor, test_loader, args.device,
            variant_name=name, max_samples=args.max_eval_samples,
        )
        ablation_rows.append(row)
        print(f"  {name}: IoU(GC)={row['heldout_gc_iou']:.3f}, Pointing={row['gc_pointing']:.3f}, CR={row['gc_causal_ratio']:.2f}")

    # Format table
    table = format_ablation_table(ablation_rows)
    print("\n" + table)

    output_dir = Path(args.output_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ablation_table.md", 'w') as f:
        f.write(table)
    with open(output_dir / "ablation_results.json", 'w') as f:
        json.dump(ablation_rows, f, indent=2)

    return ablation_rows


def generate_final_verdict(args):
    """Generate final Tier 1 verdict from all evaluation results."""
    eval_dir = Path(args.output_dir) / "evaluation"

    # Load results
    results = {}
    for filename in ["evaluation_results.json", "multi_seed_results.json", "ablation_results.json"]:
        path = eval_dir / filename
        if path.exists():
            with open(path) as f:
                results[filename] = json.load(f)

    # Extract key metrics
    single = results.get("evaluation_results.json", {})
    multi = results.get("multi_seed_results.json", {})

    p4 = single.get("phase4_attribution", {})
    p5 = single.get("phase5_causality", {})

    heldout_iou = p4.get("gradcam_iou_mean", 0)
    heldout_pointing = p4.get("gradcam_pointing_mean", 0)
    causal_ratio = p5.get("gradcam_causal_ratio_mean", 0)
    iou_std = multi.get("gradcam_iou_std", float("nan"))
    gc_delta = p4.get("gradcam_iou_mean", 0) - p4.get("attention_iou_mean", 0)

    # Check pass/fail
    blocking = []
    if heldout_iou <= 0.5:
        blocking.append(f"HeldOut_IoU={heldout_iou:.3f} <= 0.5")
    if heldout_pointing <= 0.8:
        blocking.append(f"HeldOut_Pointing={heldout_pointing:.3f} <= 0.8")
    if causal_ratio < 5:
        blocking.append(f"Causal_Ratio={causal_ratio:.2f} < 5")
    if not np.isnan(iou_std) and iou_std >= 0.05:
        blocking.append(f"IoU_Std={iou_std:.3f} >= 0.05")

    ready = len(blocking) == 0

    verdict = {
        "HeldOut_IoU": round(heldout_iou, 3),
        "HeldOut_Pointing": round(heldout_pointing, 3),
        "Causal_Ratio": round(causal_ratio, 2),
        "IoU_Std": round(iou_std, 3) if not np.isnan(iou_std) else "N/A",
        "GradCAM_vs_Attention_Delta": round(gc_delta, 3),
        "Ready_For_Arxiv": "YES" if ready else "NO",
        "Blocking_Issues": blocking if blocking else "None",
    }

    print("\n" + "=" * 60)
    print("TIER 1 FINAL VERDICT")
    print("=" * 60)
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    with open(eval_dir / "tier1_verdict.json", 'w') as f:
        json.dump(verdict, f, indent=2)

    return verdict


def main():
    parser = argparse.ArgumentParser(description="Tier 1 Reconstruction Pipeline")
    parser.add_argument("--data-dir", type=str, default="outputs/dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Phase selection
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--phase", type=str, help="Run specific phase: 2, 3, 45, 6, 7")

    # Dataset params
    parser.add_argument("--n-episodes", type=int, default=500)

    # Training params
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)

    # Eval params
    parser.add_argument("--max-eval-samples", type=int, default=100)
    parser.add_argument("--ig-steps", type=int, default=30)
    parser.add_argument("--model-path", type=str, default=None)

    # Multi-seed
    parser.add_argument("--n-seeds", type=int, default=5)

    args = parser.parse_args()

    start = time.time()

    if args.all:
        run_phase2(args)
        run_phase3(args, seed=42)
        run_phase45(args)
        run_phase6(args)
        run_phase7(args)
        generate_final_verdict(args)

    elif args.phase == "2":
        run_phase2(args)

    elif args.phase == "3":
        run_phase3(args, seed=42)

    elif args.phase == "45":
        run_phase45(args, model_path=args.model_path)

    elif args.phase == "6":
        run_phase6(args)

    elif args.phase == "7":
        run_phase7(args)

    elif args.phase == "verdict":
        generate_final_verdict(args)

    else:
        print("Specify --all or --phase {2|3|45|6|7|verdict}")

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
