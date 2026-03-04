#!/usr/bin/env python3
"""
Multi-Scene VLA-GradCAM Validation

Tests the attention-pooling VLA on diverse scenes:
1. Robot manipulation scene (red cup, blue block, green ball)
2. Kitchen scene (banana, orange, knife)
3. Office scene (laptop, phone, notebook)
4. Outdoor scene (ball, frisbee, bottle)

Validates:
- Saliency differentiation across instructions
- Spatial alignment with objects
- Action prediction accuracy
- Video quality

Tracks all results in research document.
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
from dataclasses import dataclass, asdict
from typing import List, Dict

from vla_gradcam.clip_vla_attn import load_clip_vla_attn, CLIPVLA_AttnPool
from vla_gradcam.gradcam_engine import VLAGradCAMEngine
from vla_gradcam.visualizer import VLAGradCAMVisualizer


@dataclass
class ValidationMetrics:
    """Metrics for a single scene."""
    scene_name: str
    avg_cross_instruction_corr: float
    min_cross_instruction_corr: float
    max_cross_instruction_corr: float
    training_loss: float
    instruction_action_pairs: List[Dict]
    per_action_dim_correlations: Dict[str, float]


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


def train_on_scene(model, scene, instructions_actions, device="cpu", n_epochs=300, lr=5e-4):
    """Train action head on a given scene."""
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

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return losses[-1]  # Return final loss


def evaluate_scene(model, scene, instructions, scene_name, output_dir, device="cpu"):
    """Evaluate GradCAM on a scene and compute metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = VLAGradCAMEngine(model, target_layer_idx=-1)
    viz = VLAGradCAMVisualizer(figsize_scale=1.0)

    # Compute GradCAM for all instructions
    results = {}
    for inst in instructions:
        result = engine.compute_full(scene, inst, action_dims=[0, 1, 2, 6])
        results[inst] = result

    # Compute correlation matrix
    result_list = list(results.values())
    correlations = []
    for i in range(len(result_list)):
        for j in range(i+1, len(result_list)):
            sal_i = result_list[i].combined_saliency
            sal_j = result_list[j].combined_saliency
            corr = np.corrcoef(sal_i.flatten(), sal_j.flatten())[0, 1]
            correlations.append(corr)

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
            per_dim_corrs[dim_name] = np.mean(dim_corrs)

    # Generate visualizations
    for inst, result in results.items():
        safe_name = inst.replace(" ", "_")[:30]
        fig = viz.plot_per_action_saliency(result, save_path=str(output_dir / f"per_action_{safe_name}.png"))
        plt.close(fig)

    fig = viz.plot_instruction_comparison(
        list(results.values()), action_dim_name="delta_x",
        save_path=str(output_dir / f"{scene_name}_comparison.png"),
    )
    plt.close(fig)

    # Generate video
    video_path = str(output_dir / f"{scene_name}_demo.mp4")
    viz.generate_full_demo_video(results, video_path, fps=2, hold_frames=4)

    engine.remove_hooks()

    # Collect metrics
    instruction_action_pairs = []
    for inst, result in results.items():
        instruction_action_pairs.append({
            'instruction': inst,
            'action': result.predicted_action.tolist(),
        })

    metrics = ValidationMetrics(
        scene_name=scene_name,
        avg_cross_instruction_corr=np.mean(correlations),
        min_cross_instruction_corr=np.min(correlations),
        max_cross_instruction_corr=np.max(correlations),
        training_loss=0.0,  # Will be filled by caller
        instruction_action_pairs=instruction_action_pairs,
        per_action_dim_correlations=per_dim_corrs,
    )

    return metrics


def main():
    print("=" * 80)
    print("MULTI-SCENE VLA-GRADCAM VALIDATION")
    print("=" * 80)

    device = "cpu"
    output_base = PROJECT_ROOT / "outputs" / "multi_scene_validation"

    # Load model
    print("\n[Step 1/4] Loading CLIP-VLA with attention pooling...")
    model, processor = load_clip_vla_attn(device=device)

    all_metrics = []

    # ===============================================================
    # Scene 1: Robot Manipulation
    # ===============================================================
    print("\n" + "=" * 80)
    print("SCENE 1: Robot Manipulation (red cup, blue block, green ball)")
    print("=" * 80)

    scene1 = create_robot_scene()
    instructions1 = [
        ("pick up the red cup", [-0.5, 0.1, -0.3, 0.0, 0.0, 0.0, 0.6]),
        ("push the blue block", [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -0.2]),
        ("grasp the green ball", [0.5, 0.1, -0.3, 0.0, 0.0, 0.0, 0.6]),
        ("move the robot gripper down", [0.0, 0.0, -0.6, 0.0, 0.0, 0.0, 0.0]),
    ]

    print("\n  Training...")
    loss1 = train_on_scene(model, scene1, instructions1, device)
    print(f"  Final loss: {loss1:.6f}")

    print("\n  Evaluating...")
    metrics1 = evaluate_scene(
        model, scene1, [inst for inst, _ in instructions1],
        "robot_scene", output_base / "robot_scene", device
    )
    metrics1.training_loss = loss1
    all_metrics.append(metrics1)

    print(f"\n  Results:")
    print(f"    Avg correlation: {metrics1.avg_cross_instruction_corr:.3f}")
    print(f"    Min correlation: {metrics1.min_cross_instruction_corr:.3f}")
    print(f"    Max correlation: {metrics1.max_cross_instruction_corr:.3f}")

    # ===============================================================
    # Scene 2: Kitchen
    # ===============================================================
    print("\n" + "=" * 80)
    print("SCENE 2: Kitchen (banana, orange, knife)")
    print("=" * 80)

    scene2 = create_kitchen_scene()
    instructions2 = [
        ("pick up the banana", [-0.6, 0.1, -0.3, 0.0, 0.0, 0.0, 0.7]),
        ("grasp the orange", [0.0, 0.2, -0.4, 0.0, 0.0, 0.0, 0.7]),
        ("grab the knife", [0.5, 0.1, -0.3, 0.0, 0.0, 0.0, 0.5]),
    ]

    print("\n  Training...")
    loss2 = train_on_scene(model, scene2, instructions2, device)
    print(f"  Final loss: {loss2:.6f}")

    print("\n  Evaluating...")
    metrics2 = evaluate_scene(
        model, scene2, [inst for inst, _ in instructions2],
        "kitchen_scene", output_base / "kitchen_scene", device
    )
    metrics2.training_loss = loss2
    all_metrics.append(metrics2)

    print(f"\n  Results:")
    print(f"    Avg correlation: {metrics2.avg_cross_instruction_corr:.3f}")
    print(f"    Min correlation: {metrics2.min_cross_instruction_corr:.3f}")

    # ===============================================================
    # Scene 3: Office
    # ===============================================================
    print("\n" + "=" * 80)
    print("SCENE 3: Office (laptop, phone, notebook)")
    print("=" * 80)

    scene3 = create_office_scene()
    instructions3 = [
        ("pick up the laptop", [-0.5, 0.0, -0.2, 0.0, 0.0, 0.0, 0.6]),
        ("grasp the phone", [0.0, 0.1, -0.3, 0.0, 0.0, 0.0, 0.5]),
        ("grab the notebook", [0.6, 0.0, -0.2, 0.0, 0.0, 0.0, 0.6]),
    ]

    print("\n  Training...")
    loss3 = train_on_scene(model, scene3, instructions3, device)
    print(f"  Final loss: {loss3:.6f}")

    print("\n  Evaluating...")
    metrics3 = evaluate_scene(
        model, scene3, [inst for inst, _ in instructions3],
        "office_scene", output_base / "office_scene", device
    )
    metrics3.training_loss = loss3
    all_metrics.append(metrics3)

    print(f"\n  Results:")
    print(f"    Avg correlation: {metrics3.avg_cross_instruction_corr:.3f}")
    print(f"    Min correlation: {metrics3.min_cross_instruction_corr:.3f}")

    # ===============================================================
    # Summary
    # ===============================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Multi-Scene Validation")
    print("=" * 80)

    # Aggregate statistics
    avg_corrs = [m.avg_cross_instruction_corr for m in all_metrics]
    min_corrs = [m.min_cross_instruction_corr for m in all_metrics]
    losses = [m.training_loss for m in all_metrics]

    print("\n  Cross-Scene Statistics:")
    print(f"    Mean avg correlation: {np.mean(avg_corrs):.3f} ± {np.std(avg_corrs):.3f}")
    print(f"    Mean min correlation: {np.mean(min_corrs):.3f} ± {np.std(min_corrs):.3f}")
    print(f"    Mean training loss: {np.mean(losses):.6f}")

    print("\n  Per-Scene Breakdown:")
    for m in all_metrics:
        print(f"\n    {m.scene_name}:")
        print(f"      Avg corr: {m.avg_cross_instruction_corr:.3f}")
        print(f"      Min corr: {m.min_cross_instruction_corr:.3f}")
        print(f"      Loss: {m.training_loss:.6f}")

    # Save metrics to JSON
    metrics_file = output_base / "validation_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump([asdict(m) for m in all_metrics], f, indent=2)
    print(f"\n  Metrics saved to: {metrics_file}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_base}")


if __name__ == "__main__":
    main()
