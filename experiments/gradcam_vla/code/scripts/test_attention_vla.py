#!/usr/bin/env python3
"""
VLA-GradCAM with Attention Pooling: Full Test & Validation

Tests the fixed architecture end-to-end:
1. Verify gradient flow is NOT uniform
2. Train on single scene
3. Compute GradCAM
4. Generate visualizations
5. Validate saliency quality
6. Update research document
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

from vla_gradcam.clip_vla_attn import load_clip_vla_attn, CLIPVLA_AttnPool
from vla_gradcam.gradcam_engine import VLAGradCAMEngine
from vla_gradcam.visualizer import VLAGradCAMVisualizer

# Import scene creators
from train_and_demo_vla import create_robot_scene, create_kitchen_scene


def test_gradient_flow(model, scene, device="cpu"):
    """Test that gradients are NOT uniform with attention pooling."""
    print("\n" + "=" * 70)
    print("TEST 1: Gradient Flow with Attention Pooling")
    print("=" * 70)

    pil_img = PILImage.fromarray(scene)
    pixel_values = model.processor(images=pil_img, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device)

    # Get text features
    text_inputs = model.processor(text=["pick up the red cup"], return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_out = model.clip.text_model(**text_inputs)
        text_feat = model.clip.text_projection(text_out.pooler_output)
        text_feat = F.normalize(text_feat, dim=-1)

    # Enable gradients
    for p in model.clip.vision_model.parameters():
        p.requires_grad_(True)
    for p in model.attention_pool.parameters():
        p.requires_grad_(True)

    # Forward with gradient tracking
    model.zero_grad()
    patch_features = model.encode_image_patches(pixel_values)
    patches = patch_features[:, 1:, :]
    patches.retain_grad()  # CRITICAL: retain grad on intermediate tensor

    vision_feat, attn_weights = model.attention_pool(patches, text_feat)
    combined = torch.cat([vision_feat, text_feat], dim=-1)
    action = model.action_head(combined)

    # Backprop from delta_x
    action[0, 0].backward()

    # Check gradients
    if patches.grad is not None:
        grad_norms = patches.grad.norm(dim=-1).squeeze()  # [196]
        print(f"\n  Gradient norms per patch:")
        print(f"    Mean: {grad_norms.mean():.8f}")
        print(f"    Std:  {grad_norms.std():.8f}")
        print(f"    CoV (std/mean): {grad_norms.std() / grad_norms.mean():.6f}")
        print(f"    Min: {grad_norms.min():.8f}")
        print(f"    Max: {grad_norms.max():.8f}")

        if grad_norms.std() / grad_norms.mean() > 0.05:
            print("\n  ✅ SUCCESS: Gradients show spatial variation!")
            print("  Attention pooling preserves spatial information in gradient flow.")
        else:
            print("\n  ❌ FAILED: Gradients still uniform. Architecture issue persists.")

        # Show attention weights
        attn_np = attn_weights.detach().cpu().numpy().squeeze()  # [196]
        attn_grid = attn_np.reshape(14, 14)
        print(f"\n  Attention weights (14x14 grid):")
        print(f"    Mean: {attn_np.mean():.6f} (should be ~1/196 = 0.0051)")
        print(f"    Std:  {attn_np.std():.6f}")
        print(f"    Min:  {attn_np.min():.6f}")
        print(f"    Max:  {attn_np.max():.6f}")

        if attn_np.max() / attn_np.mean() > 2.0:
            print("  ✅ Attention is selective (max >> mean)")
        else:
            print("  ⚠️ Attention is somewhat uniform")

    else:
        print("  ❌ ERROR: No gradients computed!")

    # Re-freeze
    for p in model.clip.vision_model.parameters():
        p.requires_grad_(False)
    for p in model.attention_pool.parameters():
        p.requires_grad_(False)


def train_action_head(model, scene, device="cpu", n_epochs=300, lr=5e-4):
    """Train action head with attention pooling."""
    print("\n" + "=" * 70)
    print("TEST 2: Training Action Head")
    print("=" * 70)

    pil_img = PILImage.fromarray(scene)
    pixel_values = model.processor(images=pil_img, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device)

    # Training data
    instructions_and_actions = [
        ("pick up the red cup", [-0.5, 0.1, -0.3, 0.0, 0.0, 0.0, 0.6]),
        ("push the blue block", [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -0.2]),
        ("grasp the green ball", [0.5, 0.1, -0.3, 0.0, 0.0, 0.0, 0.6]),
        ("move the robot gripper down", [0.0, 0.0, -0.6, 0.0, 0.0, 0.0, 0.0]),
    ]

    data = []
    for inst, target in instructions_and_actions:
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
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}/{n_epochs}: loss = {avg_loss:.6f}")

    model.eval()

    # Re-freeze
    for param in model.parameters():
        param.requires_grad = False

    # Print predictions
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


def compute_gradcam(model, scene, device="cpu"):
    """Compute GradCAM and validate quality."""
    print("\n" + "=" * 70)
    print("TEST 3: GradCAM Computation & Validation")
    print("=" * 70)

    engine = VLAGradCAMEngine(model, target_layer_idx=-1)

    instructions = [
        "pick up the red cup",
        "push the blue block",
        "grasp the green ball",
        "move the robot gripper down",
    ]

    results = {}
    for inst in instructions:
        print(f"\n  Computing saliency for: \"{inst}\"")
        result = engine.compute_full(scene, inst, action_dims=[0, 1, 2, 6])
        results[inst] = result

        action_str = ", ".join(f"{v:.3f}" for v in result.predicted_action)
        print(f"    Action: [{action_str}]")

        # Check saliency quality
        for dim_name, sal in result.saliency_maps.items():
            sal_range = sal.max() - sal.min()
            sal_mean = sal.mean()
            print(f"    {dim_name}: range={sal_range:.4f}, mean={sal_mean:.4f}")

    print("\n  Cross-instruction saliency correlations:")
    result_list = list(results.values())
    correlations = []
    for i in range(len(result_list)):
        for j in range(i + 1, len(result_list)):
            sal_i = result_list[i].combined_saliency
            sal_j = result_list[j].combined_saliency
            corr = np.corrcoef(sal_i.flatten(), sal_j.flatten())[0, 1]
            correlations.append(corr)
            inst_i = result_list[i].instruction
            inst_j = result_list[j].instruction
            print(f"    '{inst_i}' vs '{inst_j}': {corr:.3f}")

    avg_corr = np.mean(correlations)
    print(f"\n  Average correlation: {avg_corr:.3f}")

    if avg_corr < 0.5:
        print("  ✅ SUCCESS: Saliency maps are well-differentiated!")
    elif avg_corr < 0.7:
        print("  ⚠️ PARTIAL: Some differentiation, but could be better")
    else:
        print("  ❌ FAILED: Saliency maps too similar")

    engine.remove_hooks()
    return results


def generate_visualizations(results, output_dir):
    """Generate all visualizations."""
    print("\n" + "=" * 70)
    print("TEST 4: Generating Visualizations")
    print("=" * 70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    viz = VLAGradCAMVisualizer(figsize_scale=1.0)

    # Per-action saliency
    for inst, result in results.items():
        safe_name = inst.replace(" ", "_")[:30]
        fig = viz.plot_per_action_saliency(
            result,
            save_path=str(output_dir / f"per_action_{safe_name}.png"),
        )
        plt.close(fig)
        print(f"  Saved per_action_{safe_name}.png")

    # Instruction comparison
    fig = viz.plot_instruction_comparison(
        list(results.values()),
        action_dim_name="delta_x",
        save_path=str(output_dir / "instruction_comparison_dx.png"),
    )
    plt.close(fig)
    print("  Saved instruction_comparison_dx.png")

    # Overview
    first_result = list(results.values())[0]
    fig = viz.plot_combined_overview(
        first_result,
        save_path=str(output_dir / "overview.png"),
    )
    plt.close(fig)
    print("  Saved overview.png")

    # Generate videos
    print("\n  Generating videos...")
    video_path = str(output_dir / "vla_gradcam_demo.mp4")
    viz.generate_full_demo_video(results, video_path, fps=2, hold_frames=4)

    print(f"\n  All outputs saved to: {output_dir}")


def main():
    print("=" * 70)
    print("VLA-GradCAM with ATTENTION POOLING: Full Validation")
    print("=" * 70)

    output_dir = PROJECT_ROOT / "outputs" / "vla_gradcam_attention"
    device = "cpu"

    # Load model
    print("\n[Step 1/5] Loading CLIP-VLA with attention pooling...")
    model, processor = load_clip_vla_attn(device=device)

    # Create scene
    print("\n[Step 2/5] Creating test scene...")
    scene = create_robot_scene()

    # Test gradient flow
    print("\n[Step 3/5] Testing gradient flow...")
    test_gradient_flow(model, scene, device)

    # Train
    print("\n[Step 4/5] Training action head...")
    model = train_action_head(model, scene, device, n_epochs=300, lr=5e-4)

    # Compute GradCAM
    print("\n[Step 5/5] Computing GradCAM & generating outputs...")
    results = compute_gradcam(model, scene, device)

    # Visualize
    generate_visualizations(results, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nNext: Review visualizations, update research doc, test on more scenes")


if __name__ == "__main__":
    main()
