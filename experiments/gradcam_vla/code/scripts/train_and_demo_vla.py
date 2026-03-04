#!/usr/bin/env python3
"""
VLA-GradCAM: Train Action Head + Full Demo

This script:
  1. Loads CLIP-based VLA
  2. Trains the action head on synthetic instruction-object pairs
     so that different instructions produce genuinely different actions
     (e.g., 'pick up red cup' -> move left, 'grasp green ball' -> move right)
  3. Runs GradCAM on the trained model
  4. Generates publication-quality visualizations and demo video
  5. Validates language-conditioned saliency differentiation

The training makes GradCAM saliency maps meaningful: the model must
attend to the correct object for each instruction, so gradients
reveal *where* it looks for each action dimension.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from vla_gradcam.clip_vla import load_clip_vla, CLIPVLA
from vla_gradcam.gradcam_engine import VLAGradCAMEngine
from vla_gradcam.visualizer import VLAGradCAMVisualizer, overlay_heatmap


# ============================================================
# Scene creation
# ============================================================

def create_robot_scene() -> np.ndarray:
    """
    Create a synthetic robot manipulation scene with clearly
    distinct, spatially separated objects.
    Returns [480, 640, 3] uint8 RGB.
    """
    img = np.ones((480, 640, 3), dtype=np.uint8) * 235

    # Wooden table surface
    img[310:, :] = [180, 150, 110]
    cv2.line(img, (0, 310), (640, 310), [140, 110, 70], 2)

    # 1. RED CUP (left side, x ~ 130)
    cup_cx, cup_cy = 130, 275
    pts = np.array([
        [cup_cx - 30, cup_cy - 50],
        [cup_cx + 30, cup_cy - 50],
        [cup_cx + 25, cup_cy + 20],
        [cup_cx - 25, cup_cy + 20],
    ], np.int32)
    cv2.fillPoly(img, [pts], [60, 50, 200])
    cv2.ellipse(img, (cup_cx, cup_cy - 50), (30, 10), 0, 0, 360, [80, 60, 220], -1)
    cv2.ellipse(img, (cup_cx + 32, cup_cy - 20), (10, 18), 0, -90, 90, [70, 55, 210], 3)

    # 2. BLUE BLOCK (center, x ~ 330)
    block_cx, block_cy = 330, 275
    cv2.rectangle(img, (block_cx - 35, block_cy - 35),
                  (block_cx + 35, block_cy + 25), [200, 120, 50], -1)
    pts_top = np.array([
        [block_cx - 35, block_cy - 35],
        [block_cx - 20, block_cy - 50],
        [block_cx + 50, block_cy - 50],
        [block_cx + 35, block_cy - 35],
    ], np.int32)
    cv2.fillPoly(img, [pts_top], [220, 150, 80])
    pts_right = np.array([
        [block_cx + 35, block_cy - 35],
        [block_cx + 50, block_cy - 50],
        [block_cx + 50, block_cy + 10],
        [block_cx + 35, block_cy + 25],
    ], np.int32)
    cv2.fillPoly(img, [pts_right], [160, 90, 40])

    # 3. GREEN BALL (right side, x ~ 520)
    ball_cx, ball_cy = 520, 280
    cv2.circle(img, (ball_cx, ball_cy), 35, [50, 180, 50], -1)
    cv2.circle(img, (ball_cx - 10, ball_cy - 12), 10, [100, 230, 100], -1)
    cv2.ellipse(img, (ball_cx, ball_cy + 37), (28, 6), 0, 0, 360, [120, 100, 70], -1)

    # Robot arm + gripper (top center)
    cv2.rectangle(img, (290, 0), (350, 130), [80, 80, 90], -1)
    cv2.rectangle(img, (275, 130), (365, 175), [90, 90, 100], -1)
    cv2.rectangle(img, (280, 175), (310, 250), [70, 70, 80], -1)
    cv2.rectangle(img, (330, 175), (360, 250), [70, 70, 80], -1)
    cv2.circle(img, (295, 175), 6, [100, 100, 110], -1)
    cv2.circle(img, (345, 175), 6, [100, 100, 110], -1)

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "red cup", (90, 320), font, 0.45, [60, 50, 200], 1)
    cv2.putText(img, "blue block", (290, 320), font, 0.45, [200, 120, 50], 1)
    cv2.putText(img, "green ball", (480, 320), font, 0.45, [50, 180, 50], 1)

    return img


def create_kitchen_scene() -> np.ndarray:
    """Kitchen counter scene."""
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


# ============================================================
# Training: fine-tune action head
# ============================================================

def build_training_data(model: CLIPVLA, scene: np.ndarray, device: str):
    """
    Build (pixel_values, text_features, target_action) tuples.

    We define target actions that make spatial/semantic sense:
    - "pick up the red cup"   -> move LEFT  (negative dx), DOWN (neg dz), CLOSE gripper
    - "push the blue block"   -> move FORWARD (pos dy), small dx, OPEN gripper
    - "grasp the green ball"  -> move RIGHT (pos dx), DOWN (neg dz), CLOSE gripper
    - "move gripper up"       -> move UP (pos dz), no lateral, OPEN gripper

    This teaches the model to produce distinct actions per instruction,
    so GradCAM gradients become instruction-specific.
    """
    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(scene)
    pixel_values = model.processor(images=pil_img, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device)

    # Targets kept moderate (well within [-1,1] Tanh range) to avoid saturation
    instructions_and_actions = [
        # (instruction, target [dx, dy, dz, droll, dpitch, dyaw, gripper])
        ("pick up the red cup",
         [-0.5,  0.1, -0.3,  0.0,  0.0,  0.0,  0.6]),
        ("push the blue block",
         [ 0.0,  0.5,  0.0,  0.0,  0.0,  0.0, -0.2]),
        ("grasp the green ball",
         [ 0.5,  0.1, -0.3,  0.0,  0.0,  0.0,  0.6]),
        ("move the robot gripper down",
         [ 0.0,  0.0, -0.6,  0.0,  0.0,  0.0,  0.0]),
        ("move the robot gripper up",
         [ 0.0,  0.0,  0.6,  0.0,  0.0,  0.0,  0.0]),
        ("lift the red cup",
         [-0.4,  0.0,  0.4,  0.0,  0.0,  0.0,  0.6]),
        ("push the green ball to the left",
         [ 0.4,  0.0,  0.0,  0.0,  0.0, -0.2, -0.1]),
        ("place the blue block down",
         [ 0.0,  0.1, -0.5,  0.0,  0.0,  0.0, -0.5]),
    ]

    data = []
    for inst, target in instructions_and_actions:
        # Pre-compute text features (frozen)
        text_inputs = model.processor(
            text=[inst], return_tensors="pt", padding=True, truncation=True,
        ).to(device)
        with torch.no_grad():
            text_out = model.clip.text_model(**text_inputs)
            text_feat = model.clip.text_projection(text_out.pooler_output)
            text_feat = F.normalize(text_feat, dim=-1)

        target_tensor = torch.tensor([target], dtype=torch.float32, device=device)
        data.append((pixel_values, text_feat, target_tensor, inst))

    return data


def train_action_head(
    model: CLIPVLA,
    scene: np.ndarray,
    device: str = "cpu",
    n_epochs: int = 500,
    lr: float = 5e-4,
):
    """
    Train only the action head (CLIP frozen) to map
    (image, instruction) -> distinct target actions.

    Two-phase training:
      Phase 1: train action_head only (vision_proj frozen)
      Phase 2: fine-tune both action_head + vision_proj at lower LR
    """
    print("\n--- Training Action Head ---")

    data = build_training_data(model, scene, device)
    loss_fn = nn.MSELoss()

    # Phase 1: Only action head (more stable gradient flow)
    print("  Phase 1: training action_head only...")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.action_head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.action_head.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for pv, tf, target, inst in data:
            optimizer.zero_grad()
            pred = model.forward_for_gradcam(pv, tf)
            loss = loss_fn(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.action_head.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data)
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}/{n_epochs}: loss = {avg_loss:.6f}")

    # Phase 2: Joint fine-tune action_head + vision_proj at lower LR
    print("\n  Phase 2: fine-tuning action_head + vision_proj...")
    for param in model.vision_proj.parameters():
        param.requires_grad = True

    optimizer2 = torch.optim.Adam([
        {'params': model.action_head.parameters(), 'lr': lr * 0.2},
        {'params': model.vision_proj.parameters(), 'lr': lr * 0.1},
    ])

    n_phase2 = 200
    for epoch in range(n_phase2):
        total_loss = 0.0
        for pv, tf, target, inst in data:
            optimizer2.zero_grad()
            pred = model.forward_for_gradcam(pv, tf)
            loss = loss_fn(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.action_head.parameters()) + list(model.vision_proj.parameters()),
                1.0,
            )
            optimizer2.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data)
        if epoch % 50 == 0 or epoch == n_phase2 - 1:
            print(f"  Epoch {epoch:3d}/{n_phase2}: loss = {avg_loss:.6f}")

    model.eval()

    # Re-freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Print final predictions
    print("\n  Final predictions:")
    for pv, tf, target, inst in data:
        with torch.no_grad():
            pred = model.forward_for_gradcam(pv, tf)
        pred_str = ", ".join(f"{v:.3f}" for v in pred[0].cpu().numpy())
        tgt_str = ", ".join(f"{v:.1f}" for v in target[0].cpu().numpy())
        print(f"    '{inst}'")
        print(f"      pred: [{pred_str}]")
        print(f"      tgt:  [{tgt_str}]")

    return model


# ============================================================
# Main demo
# ============================================================

def main():
    print("=" * 70)
    print("VLA-GradCAM: Train + Demo (End-to-End)")
    print("=" * 70)

    output_dir = PROJECT_ROOT / "outputs" / "vla_gradcam_trained"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 1. Load CLIP-based VLA
    # ----------------------------------------------------------
    print("\n[1/7] Loading CLIP-based VLA model...")
    device = "cpu"
    model, processor = load_clip_vla(device=device)

    # ----------------------------------------------------------
    # 2. Create scene
    # ----------------------------------------------------------
    print("\n[2/7] Creating robot manipulation scene...")
    scene = create_robot_scene()
    cv2.imwrite(str(output_dir / "scene.png"), cv2.cvtColor(scene, cv2.COLOR_RGB2BGR))

    # ----------------------------------------------------------
    # 3. Train action head
    # ----------------------------------------------------------
    print("\n[3/7] Training action head on scene-instruction pairs...")
    model = train_action_head(model, scene, device=device, n_epochs=500, lr=5e-4)

    # Save trained weights
    ckpt_path = output_dir / "vla_action_head.pt"
    torch.save({
        'action_head': model.action_head.state_dict(),
        'vision_proj': model.vision_proj.state_dict(),
    }, str(ckpt_path))
    print(f"  Saved checkpoint: {ckpt_path}")

    # ----------------------------------------------------------
    # 4. Create GradCAM engine
    # ----------------------------------------------------------
    print("\n[4/7] Initializing VLA-GradCAM engine...")
    engine = VLAGradCAMEngine(model, target_layer_idx=-1)
    viz = VLAGradCAMVisualizer(figsize_scale=1.0)

    # ----------------------------------------------------------
    # 5. Compute saliency for all instructions
    # ----------------------------------------------------------
    instructions = [
        "pick up the red cup",
        "push the blue block",
        "grasp the green ball",
        "move the robot gripper down",
    ]

    print("\n[5/7] Computing GradCAM saliency maps (trained model)...")
    results = {}
    for inst in instructions:
        print(f"  Processing: \"{inst}\" ...")
        result = engine.compute_full(scene, inst, action_dims=[0, 1, 2, 6])
        results[inst] = result
        action_str = ", ".join(f"{v:.3f}" for v in result.predicted_action)
        print(f"    Action: [{action_str}]")

    # ----------------------------------------------------------
    # 6. Generate visualizations
    # ----------------------------------------------------------
    print("\n[6/7] Generating visualizations...")

    # 6a. Per-action saliency
    for inst, result in results.items():
        safe_name = inst.replace(" ", "_")[:30]
        fig = viz.plot_per_action_saliency(
            result,
            save_path=str(output_dir / f"per_action_{safe_name}.png"),
        )
        plt.close(fig)
        print(f"  Saved per_action_{safe_name}.png")

    # 6b. Overview
    first_result = list(results.values())[0]
    fig = viz.plot_combined_overview(
        first_result,
        save_path=str(output_dir / "overview.png"),
    )
    plt.close(fig)
    print("  Saved overview.png")

    # 6c. Instruction comparison
    fig = viz.plot_instruction_comparison(
        list(results.values()),
        action_dim_name="delta_x",
        save_path=str(output_dir / "instruction_comparison_dx.png"),
    )
    plt.close(fig)
    print("  Saved instruction_comparison_dx.png")

    fig = viz.plot_instruction_comparison(
        list(results.values()),
        action_dim_name="gripper",
        save_path=str(output_dir / "instruction_comparison_gripper.png"),
    )
    plt.close(fig)
    print("  Saved instruction_comparison_gripper.png")

    # 6d. Action contrast (delta_x vs gripper)
    contrast = engine.compute_action_contrast(scene, "pick up the red cup", dim_a=0, dim_b=6)
    fig = viz.plot_action_contrast(
        scene, contrast, "delta_x", "gripper",
        save_path=str(output_dir / "action_contrast_x_vs_gripper.png"),
    )
    plt.close(fig)
    print("  Saved action_contrast_x_vs_gripper.png")

    # 6e. Instruction contrast (red cup vs green ball)
    inst_contrast = engine.compute_instruction_contrast(
        scene, "pick up the red cup", "grasp the green ball",
    )
    fig = viz.plot_action_contrast(
        scene, inst_contrast, "red cup", "green ball",
        save_path=str(output_dir / "instruction_contrast_red_vs_green.png"),
    )
    plt.close(fig)
    print("  Saved instruction_contrast_red_vs_green.png")

    # ----------------------------------------------------------
    # 7. Generate demo videos
    # ----------------------------------------------------------
    print("\n[7/7] Generating demo videos...")

    # Full demo video
    video_path = str(output_dir / "vla_gradcam_demo.mp4")
    viz.generate_full_demo_video(results, video_path, fps=2, hold_frames=4)

    # Instruction-cycling video
    short_video = str(output_dir / "instruction_cycle.mp4")
    viz.generate_instruction_video(
        list(results.values()), short_video,
        action_dim_name="delta_x", fps=1, hold_frames=3,
    )

    # Per-action-dim video
    action_video = str(output_dir / "action_dims.mp4")
    viz.generate_action_dim_video(first_result, action_video, fps=1, hold_frames=3)

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("VALIDATION: Language-Conditioned Saliency")
    print("=" * 70)

    result_list = list(results.values())
    correlations = []
    mads = []
    for i in range(len(result_list)):
        for j in range(i + 1, len(result_list)):
            sal_i = result_list[i].combined_saliency
            sal_j = result_list[j].combined_saliency
            corr = np.corrcoef(sal_i.flatten(), sal_j.flatten())[0, 1]
            mad = np.mean(np.abs(sal_i - sal_j))
            correlations.append(corr)
            mads.append(mad)
            print(
                f"  '{result_list[i].instruction}' vs "
                f"'{result_list[j].instruction}': "
                f"corr={corr:.3f}, MAD={mad:.4f}"
            )

    avg_corr = np.mean(correlations)
    avg_mad = np.mean(mads)
    print(f"\n  Average correlation: {avg_corr:.3f}")
    print(f"  Average MAD: {avg_mad:.4f}")

    # Per-action saliency differentiation
    print("\n  Per-action-dim saliency differentiation:")
    for dim_name in ['delta_x', 'delta_y', 'delta_z', 'gripper']:
        dim_corrs = []
        for i in range(len(result_list)):
            for j in range(i + 1, len(result_list)):
                s_i = result_list[i].saliency_maps.get(dim_name)
                s_j = result_list[j].saliency_maps.get(dim_name)
                if s_i is not None and s_j is not None:
                    c = np.corrcoef(s_i.flatten(), s_j.flatten())[0, 1]
                    dim_corrs.append(c)
        if dim_corrs:
            print(f"    {dim_name}: avg cross-instruction corr = {np.mean(dim_corrs):.3f}")

    # ----------------------------------------------------------
    # Kitchen scene bonus
    # ----------------------------------------------------------
    print("\n\n--- Bonus: Kitchen scene (using same model) ---")
    kitchen = create_kitchen_scene()
    kitchen_instructions = [
        "pick up the banana",
        "grasp the orange",
        "grab the knife",
    ]
    kitchen_results = {}
    for inst in kitchen_instructions:
        result = engine.compute_full(kitchen, inst, action_dims=[0, 1, 2, 6])
        kitchen_results[inst] = result
        action_str = ", ".join(f"{v:.3f}" for v in result.predicted_action)
        print(f"  '{inst}': [{action_str}]")

    fig = viz.plot_instruction_comparison(
        list(kitchen_results.values()),
        action_dim_name="delta_x",
        save_path=str(output_dir / "kitchen_comparison.png"),
    )
    plt.close(fig)

    kitchen_video = str(output_dir / "kitchen_demo.mp4")
    viz.generate_full_demo_video(kitchen_results, kitchen_video, fps=2, hold_frames=3)
    print("  Saved kitchen outputs")

    # Cleanup
    engine.remove_hooks()

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:50s} {size_kb:8.1f} KB")


if __name__ == "__main__":
    main()
