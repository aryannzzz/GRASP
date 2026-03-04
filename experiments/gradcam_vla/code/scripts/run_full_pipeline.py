#!/usr/bin/env python3
"""
VLA-GradCAM: Full Pipeline End-to-End Test

Runs the complete pipeline from scratch:
1. Load CLIP-VLA with attention pooling
2. Create scenes (robot, kitchen, office)
3. Train on each scene
4. Compute GradCAM saliency
5. Generate all visualizations + videos
6. Compute validation metrics
7. Save everything to output1/

This is the definitive end-to-end test script.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict

from vla_gradcam.clip_vla_attn import load_clip_vla_attn, CLIPVLA_AttnPool
from vla_gradcam.gradcam_engine import VLAGradCAMEngine
from vla_gradcam.visualizer import VLAGradCAMVisualizer


# ============================================================
# Scene creation functions
# ============================================================

def create_robot_scene() -> np.ndarray:
    """Robot manipulation scene with red cup, blue block, green ball."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 235
    img[310:, :] = [180, 150, 110]
    cv2.line(img, (0, 310), (640, 310), [140, 110, 70], 2)

    # RED CUP (left, x~130)
    cup_cx, cup_cy = 130, 275
    pts = np.array([[cup_cx-30, cup_cy-50], [cup_cx+30, cup_cy-50],
                    [cup_cx+25, cup_cy+20], [cup_cx-25, cup_cy+20]], np.int32)
    cv2.fillPoly(img, [pts], [60, 50, 200])
    cv2.ellipse(img, (cup_cx, cup_cy-50), (30, 10), 0, 0, 360, [80, 60, 220], -1)

    # BLUE BLOCK (center, x~330)
    block_cx, block_cy = 330, 275
    cv2.rectangle(img, (block_cx-35, block_cy-35), (block_cx+35, block_cy+25), [200, 120, 50], -1)
    pts_top = np.array([[block_cx-35, block_cy-35], [block_cx-20, block_cy-50],
                        [block_cx+50, block_cy-50], [block_cx+35, block_cy-35]], np.int32)
    cv2.fillPoly(img, [pts_top], [220, 150, 80])

    # GREEN BALL (right, x~520)
    ball_cx, ball_cy = 520, 280
    cv2.circle(img, (ball_cx, ball_cy), 35, [50, 180, 50], -1)
    cv2.circle(img, (ball_cx-10, ball_cy-12), 10, [100, 230, 100], -1)

    # Robot gripper
    cv2.rectangle(img, (290, 0), (350, 130), [80, 80, 90], -1)
    cv2.rectangle(img, (275, 130), (365, 175), [90, 90, 100], -1)
    cv2.rectangle(img, (280, 175), (310, 250), [70, 70, 80], -1)
    cv2.rectangle(img, (330, 175), (360, 250), [70, 70, 80], -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "red cup", (90, 320), font, 0.45, [60, 50, 200], 1)
    cv2.putText(img, "blue block", (290, 320), font, 0.45, [200, 120, 50], 1)
    cv2.putText(img, "green ball", (480, 320), font, 0.45, [50, 180, 50], 1)

    return img


def create_kitchen_scene() -> np.ndarray:
    """Kitchen counter with banana, orange, knife."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 220
    img[340:, :] = [160, 140, 120]
    cv2.line(img, (0, 340), (640, 340), [120, 100, 80], 2)

    # Yellow banana (left)
    pts = np.array([[80, 280], [180, 260], [190, 270], [90, 300]], np.int32)
    cv2.fillPoly(img, [pts], [40, 210, 240])

    # Orange (center)
    cv2.circle(img, (320, 290), 40, [30, 130, 240], -1)
    cv2.circle(img, (310, 275), 8, [60, 160, 255], -1)

    # Silver knife (right)
    cv2.rectangle(img, (470, 250), (490, 330), [180, 180, 180], -1)
    cv2.rectangle(img, (475, 220), (485, 250), [140, 140, 140], -1)

    # Robot gripper
    cv2.rectangle(img, (290, 0), (350, 120), [80, 80, 90], -1)
    cv2.rectangle(img, (275, 120), (365, 160), [90, 90, 100], -1)
    cv2.rectangle(img, (280, 160), (310, 230), [70, 70, 80], -1)
    cv2.rectangle(img, (330, 160), (360, 230), [70, 70, 80], -1)

    return img


def create_office_scene() -> np.ndarray:
    """Office desk with laptop, phone, notebook."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 245
    img[300:, :] = [200, 180, 170]
    cv2.line(img, (0, 300), (640, 300), [160, 140, 130], 2)

    # Laptop (left, gray/silver)
    cv2.rectangle(img, (80, 220), (220, 280), [140, 140, 145], -1)
    cv2.rectangle(img, (85, 225), (215, 275), [40, 40, 50], -1)  # screen

    # Phone (center, black)
    cv2.rectangle(img, (300, 240), (340, 295), [30, 30, 35], -1)
    cv2.rectangle(img, (305, 245), (335, 290), [60, 60, 70], -1)  # screen

    # Notebook (right, blue)
    cv2.rectangle(img, (460, 230), (570, 295), [200, 110, 50], -1)
    for i in range(5):
        y = 245 + i*10
        cv2.line(img, (470, y), (560, y), [180, 100, 45], 1)

    return img


# ============================================================
# Training
# ============================================================

def train_on_scene(model, scene, instructions_actions, device="cpu", n_epochs=300, lr=5e-4):
    """Train action head + attention pool on a given scene."""
    pil_img = PILImage.fromarray(scene)
    pixel_values = model.processor(images=pil_img, return_tensors="pt")["pixel_values"].to(device)

    data = []
    for inst, target in instructions_actions:
        text_inputs = model.processor(text=[inst], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_out = model.clip.text_model(**text_inputs)
            text_feat = model.clip.text_projection(text_out.pooler_output)
            text_feat = F.normalize(text_feat, dim=-1)
        target_tensor = torch.tensor([target], dtype=torch.float32, device=device)
        data.append((pixel_values, text_feat, target_tensor, inst))

    # Unfreeze trainable parts
    for param in model.parameters():
        param.requires_grad = False
    for param in model.action_head.parameters():
        param.requires_grad = True
    for param in model.attention_pool.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        list(model.action_head.parameters()) + list(model.attention_pool.parameters()),
        lr=lr,
    )
    loss_fn = nn.MSELoss()

    model.train()
    losses = []
    for epoch in range(n_epochs):
        total_loss = 0.0
        for pv, tf, target, inst in data:
            optimizer.zero_grad()
            pred = model.forward_for_gradcam(pv, tf)
            loss = loss_fn(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.action_head.parameters()) + list(model.attention_pool.parameters()),
                1.0,
            )
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data)
        losses.append(avg_loss)
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"    Epoch {epoch:3d}/{n_epochs}: loss = {avg_loss:.6f}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Print final predictions
    print("    Final predictions:")
    for pv, tf, target, inst in data:
        with torch.no_grad():
            pred = model.forward_for_gradcam(pv, tf)
        pred_np = pred[0].cpu().numpy()
        tgt_np = target[0].cpu().numpy()
        print(f"      '{inst}'")
        print(f"        pred: [{', '.join(f'{v:.3f}' for v in pred_np)}]")
        print(f"        tgt:  [{', '.join(f'{v:.1f}' for v in tgt_np)}]")

    return losses[-1]


# ============================================================
# Evaluation
# ============================================================

def evaluate_scene(model, scene, instructions, scene_name, output_dir, device="cpu"):
    """Evaluate GradCAM on a scene, generate all outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = VLAGradCAMEngine(model, target_layer_idx=-1)
    viz = VLAGradCAMVisualizer(figsize_scale=1.0)

    # Save scene image
    cv2.imwrite(str(output_dir / f"{scene_name}_scene.png"),
                cv2.cvtColor(scene, cv2.COLOR_RGB2BGR))

    # Compute GradCAM for all instructions
    results = {}
    for inst in instructions:
        print(f"    Computing saliency: \"{inst}\"")
        result = engine.compute_full(scene, inst, action_dims=[0, 1, 2, 6])
        results[inst] = result
        action_str = ", ".join(f"{v:.3f}" for v in result.predicted_action)
        print(f"      Action: [{action_str}]")

    # Compute correlation matrix
    result_list = list(results.values())
    correlations = []
    for i in range(len(result_list)):
        for j in range(i+1, len(result_list)):
            sal_i = result_list[i].combined_saliency
            sal_j = result_list[j].combined_saliency
            corr = np.corrcoef(sal_i.flatten(), sal_j.flatten())[0, 1]
            correlations.append(corr)
            print(f"    Correlation: '{result_list[i].instruction}' vs '{result_list[j].instruction}': {corr:.3f}")

    # Per-action-dim correlations
    per_dim_corrs = {}
    for dim_name in ['delta_x', 'delta_y', 'delta_z', 'gripper']:
        dim_corrs = []
        for i in range(len(result_list)):
            for j in range(i+1, len(result_list)):
                s_i = result_list[i].saliency_maps.get(dim_name)
                s_j = result_list[j].saliency_maps.get(dim_name)
                if s_i is not None and s_j is not None:
                    c = np.corrcoef(s_i.flatten(), s_j.flatten())[0, 1]
                    dim_corrs.append(c)
        if dim_corrs:
            per_dim_corrs[dim_name] = float(np.mean(dim_corrs))

    # Generate per-instruction saliency images
    for inst, result in results.items():
        safe_name = inst.replace(" ", "_")[:30]
        fig = viz.plot_per_action_saliency(result, save_path=str(output_dir / f"per_action_{safe_name}.png"))
        plt.close(fig)

    # Instruction comparison
    fig = viz.plot_instruction_comparison(
        list(results.values()), action_dim_name="delta_x",
        save_path=str(output_dir / f"{scene_name}_comparison_dx.png"),
    )
    plt.close(fig)

    # Combined overview for first instruction
    first_result = list(results.values())[0]
    fig = viz.plot_combined_overview(
        first_result,
        save_path=str(output_dir / f"{scene_name}_overview.png"),
    )
    plt.close(fig)

    # Generate video
    video_path = str(output_dir / f"{scene_name}_demo.mp4")
    viz.generate_full_demo_video(results, video_path, fps=2, hold_frames=4)

    engine.remove_hooks()

    metrics = {
        'scene_name': scene_name,
        'avg_cross_instruction_corr': float(np.mean(correlations)),
        'min_cross_instruction_corr': float(np.min(correlations)),
        'max_cross_instruction_corr': float(np.max(correlations)),
        'training_loss': 0.0,  # filled by caller
        'per_action_dim_correlations': per_dim_corrs,
        'instructions': [inst for inst in instructions],
        'predictions': {inst: result.predicted_action.tolist() for inst, result in results.items()},
    }

    return metrics


# ============================================================
# Main pipeline
# ============================================================

def main():
    start_time = time.time()

    print("=" * 80)
    print("VLA-GradCAM: FULL END-TO-END PIPELINE")
    print("=" * 80)
    print(f"Output directory: output1/")

    device = "cpu"
    output_base = PROJECT_ROOT / "output1"

    # ===============================================================
    # Step 1: Load model
    # ===============================================================
    print("\n[Step 1/5] Loading CLIP-VLA with attention pooling...")
    model, processor = load_clip_vla_attn(device=device)
    print("  Model loaded successfully.")

    all_metrics = []

    # ===============================================================
    # Step 2: Robot Scene
    # ===============================================================
    print("\n" + "=" * 80)
    print("[Step 2/5] SCENE 1: Robot Manipulation")
    print("=" * 80)

    scene1 = create_robot_scene()
    instructions1 = [
        ("pick up the red cup", [-0.5, 0.1, -0.3, 0.0, 0.0, 0.0, 0.6]),
        ("push the blue block", [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -0.2]),
        ("grasp the green ball", [0.5, 0.1, -0.3, 0.0, 0.0, 0.0, 0.6]),
        ("move the robot gripper down", [0.0, 0.0, -0.6, 0.0, 0.0, 0.0, 0.0]),
    ]

    print("  Training...")
    loss1 = train_on_scene(model, scene1, instructions1, device)
    print(f"  Final loss: {loss1:.6f}")

    print("  Evaluating...")
    metrics1 = evaluate_scene(
        model, scene1, [inst for inst, _ in instructions1],
        "robot_scene", output_base / "robot_scene", device
    )
    metrics1['training_loss'] = loss1
    all_metrics.append(metrics1)

    print(f"\n  Results: avg_corr={metrics1['avg_cross_instruction_corr']:.3f}, "
          f"min_corr={metrics1['min_cross_instruction_corr']:.3f}")

    # ===============================================================
    # Step 3: Kitchen Scene
    # ===============================================================
    print("\n" + "=" * 80)
    print("[Step 3/5] SCENE 2: Kitchen")
    print("=" * 80)

    scene2 = create_kitchen_scene()
    instructions2 = [
        ("pick up the banana", [-0.6, 0.1, -0.3, 0.0, 0.0, 0.0, 0.7]),
        ("grasp the orange", [0.0, 0.2, -0.4, 0.0, 0.0, 0.0, 0.7]),
        ("grab the knife", [0.5, 0.1, -0.3, 0.0, 0.0, 0.0, 0.5]),
    ]

    print("  Training...")
    loss2 = train_on_scene(model, scene2, instructions2, device)
    print(f"  Final loss: {loss2:.6f}")

    print("  Evaluating...")
    metrics2 = evaluate_scene(
        model, scene2, [inst for inst, _ in instructions2],
        "kitchen_scene", output_base / "kitchen_scene", device
    )
    metrics2['training_loss'] = loss2
    all_metrics.append(metrics2)

    print(f"\n  Results: avg_corr={metrics2['avg_cross_instruction_corr']:.3f}, "
          f"min_corr={metrics2['min_cross_instruction_corr']:.3f}")

    # ===============================================================
    # Step 4: Office Scene
    # ===============================================================
    print("\n" + "=" * 80)
    print("[Step 4/5] SCENE 3: Office")
    print("=" * 80)

    scene3 = create_office_scene()
    instructions3 = [
        ("pick up the laptop", [-0.5, 0.0, -0.2, 0.0, 0.0, 0.0, 0.6]),
        ("grasp the phone", [0.0, 0.1, -0.3, 0.0, 0.0, 0.0, 0.5]),
        ("grab the notebook", [0.6, 0.0, -0.2, 0.0, 0.0, 0.0, 0.6]),
    ]

    print("  Training...")
    loss3 = train_on_scene(model, scene3, instructions3, device)
    print(f"  Final loss: {loss3:.6f}")

    print("  Evaluating...")
    metrics3 = evaluate_scene(
        model, scene3, [inst for inst, _ in instructions3],
        "office_scene", output_base / "office_scene", device
    )
    metrics3['training_loss'] = loss3
    all_metrics.append(metrics3)

    print(f"\n  Results: avg_corr={metrics3['avg_cross_instruction_corr']:.3f}, "
          f"min_corr={metrics3['min_cross_instruction_corr']:.3f}")

    # ===============================================================
    # Step 5: Summary & Save
    # ===============================================================
    print("\n" + "=" * 80)
    print("[Step 5/5] SUMMARY")
    print("=" * 80)

    avg_corrs = [m['avg_cross_instruction_corr'] for m in all_metrics]
    min_corrs = [m['min_cross_instruction_corr'] for m in all_metrics]
    losses = [m['training_loss'] for m in all_metrics]

    print(f"\n  Cross-Scene Statistics (N={len(all_metrics)} scenes):")
    print(f"    Mean avg correlation: {np.mean(avg_corrs):.3f} +/- {np.std(avg_corrs):.3f}")
    print(f"    Mean min correlation: {np.mean(min_corrs):.3f}")
    print(f"    Mean training loss:   {np.mean(losses):.6f}")

    for m in all_metrics:
        print(f"\n    {m['scene_name']}:")
        print(f"      Avg corr: {m['avg_cross_instruction_corr']:.3f}")
        print(f"      Min corr: {m['min_cross_instruction_corr']:.3f}")
        print(f"      Loss: {m['training_loss']:.6f}")

    # Success criteria
    print("\n  Success Criteria:")
    avg_c = np.mean(avg_corrs)
    print(f"    Avg correlation < 0.5: {avg_c:.3f} {'PASS' if avg_c < 0.5 else 'FAIL'}")
    min_c = np.min(min_corrs)
    print(f"    Min correlation < 0.3: {min_c:.3f} {'PASS' if min_c < 0.3 else 'FAIL'}")
    avg_l = np.mean(losses)
    print(f"    Training loss < 0.001: {avg_l:.6f} {'PASS' if avg_l < 0.001 else 'FAIL'}")

    # Save metrics
    metrics_file = output_base / "validation_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Metrics saved to: {metrics_file}")

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

    # List all output files
    print(f"\nAll outputs in {output_base}:")
    for scene_dir in sorted(output_base.iterdir()):
        if scene_dir.is_dir():
            print(f"\n  {scene_dir.name}/")
            for f in sorted(scene_dir.iterdir()):
                size_kb = f.stat().st_size / 1024
                print(f"    {f.name:50s} {size_kb:8.1f} KB")
        elif scene_dir.is_file():
            size_kb = scene_dir.stat().st_size / 1024
            print(f"  {scene_dir.name:50s} {size_kb:8.1f} KB")


if __name__ == "__main__":
    main()
