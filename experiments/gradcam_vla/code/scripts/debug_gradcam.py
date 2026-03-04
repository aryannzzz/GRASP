#!/usr/bin/env python3
"""
VLA-GradCAM Diagnostic Script

Traces every step of the pipeline to identify exactly where
saliency breaks down.

KNOWN SUSPECTED ISSUES (to verify):
1. The architecture pools patches via mean() BEFORE the action head.
   => All spatial information is destroyed before the gradient target.
   => Backpropagating from action[dim] through mean-pool gives UNIFORM
      gradients across all patches, because d(mean)/d(patch_i) = 1/N for all i.
   => The hook captures activations & grads at the ViT layer output,
      and grads at that point are uniform => GradCAM is uniform noise.

2. Training on a SINGLE image means the action head learns to distinguish
   instructions purely from text features. The vision features are identical
   for every training example (same image). So the model doesn't need to
   attend to specific image regions at all.

3. The CLIP-similarity saliency (compute_similarity_saliency) is independent
   of the trained action head - it just uses raw CLIP embeddings. This one
   should show some language conditioning, but it has nothing to do with
   action prediction.

PLAN: Verify each issue step by step, then redesign the architecture.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vla_gradcam.clip_vla import load_clip_vla, CLIPVLA
from vla_gradcam.gradcam_engine import VLAGradCAMEngine

# Import scene creator from the demo
from train_and_demo_vla import create_robot_scene, train_action_head


def main():
    print("=" * 70)
    print("VLA-GradCAM DIAGNOSTIC")
    print("=" * 70)

    device = "cpu"
    model, processor = load_clip_vla(device=device)
    scene = create_robot_scene()

    # Train the action head (same as before)
    print("\n--- Training action head ---")
    model = train_action_head(model, scene, device=device, n_epochs=300, lr=5e-4)

    # ================================================================
    # DIAGNOSTIC 1: Check gradient uniformity through mean-pool
    # ================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1: Gradient Flow Through Mean-Pool")
    print("=" * 70)

    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(scene)
    pixel_values = model.processor(images=pil_img, return_tensors="pt")["pixel_values"]

    # Encode text
    text_inputs = model.processor(
        text=["pick up the red cup"], return_tensors="pt", padding=True, truncation=True,
    )
    with torch.no_grad():
        text_out = model.clip.text_model(**text_inputs)
        text_feat = model.clip.text_projection(text_out.pooler_output)
        text_feat = F.normalize(text_feat, dim=-1)

    # Enable grad on vision model
    for p in model.clip.vision_model.parameters():
        p.requires_grad_(True)

    # Forward pass - capture intermediate values
    model.zero_grad()
    pv = pixel_values.clone().detach().requires_grad_(True)
    patch_features = model.encode_image_patches(pv)

    print(f"  patch_features shape: {patch_features.shape}")  # [1, 197, 768]

    patches = patch_features[:, 1:, :]  # [1, 196, 768]
    pooled_vision = patches.mean(dim=1)  # [1, 768]
    vision_feat = model.vision_proj(pooled_vision)  # [1, 256]
    combined = torch.cat([vision_feat, text_feat], dim=-1)
    action = model.action_head(combined)

    print(f"  action: {action.detach().numpy()}")

    # Backprop from delta_x (dim 0)
    action[0, 0].backward(retain_graph=True)

    # Check gradient at patch_features level
    if patch_features.grad is not None:
        patch_grads = patch_features.grad[:, 1:, :]  # [1, 196, 768]
        # Compute per-patch gradient magnitude
        patch_grad_norms = patch_grads.norm(dim=-1).squeeze()  # [196]
        print(f"\n  Per-patch gradient norms (N=196 patches):")
        print(f"    mean: {patch_grad_norms.mean():.8f}")
        print(f"    std:  {patch_grad_norms.std():.8f}")
        print(f"    min:  {patch_grad_norms.min():.8f}")
        print(f"    max:  {patch_grad_norms.max():.8f}")
        print(f"    CoV (std/mean): {(patch_grad_norms.std()/patch_grad_norms.mean()):.6f}")

        # Reshape to 14x14 grid and show
        grid = patch_grad_norms.reshape(14, 14).detach().numpy()
        print(f"\n  14x14 gradient norm grid (should vary if spatial info preserved):")
        print(f"    Range: [{grid.min():.8f}, {grid.max():.8f}]")

        if grid.max() - grid.min() < 1e-6:
            print("  >>> UNIFORM GRADIENTS! Mean-pool destroys spatial info. <<<")
        else:
            print(f"  Gradient variation ratio: {(grid.max()-grid.min())/grid.mean():.4f}")
    else:
        print("  !!! patch_features.grad is None !!!")

    # Re-freeze
    for p in model.clip.vision_model.parameters():
        p.requires_grad_(False)

    # ================================================================
    # DIAGNOSTIC 2: Check hook activations & gradients
    # ================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: Hook-captured Activations & Gradients")
    print("=" * 70)

    engine = VLAGradCAMEngine(model, target_layer_idx=-1)

    # Enable grad, forward, backward
    for p in model.clip.vision_model.parameters():
        p.requires_grad_(True)

    model.zero_grad()
    engine._activations = None
    engine._gradients = None

    pv2 = pixel_values.clone().detach().requires_grad_(True)
    action2 = model.forward_for_gradcam(pv2, text_feat.detach())
    action2[0, 0].backward()

    if engine._activations is not None:
        act = engine._activations
        grad = engine._gradients
        print(f"  Activation shape: {act.shape}")
        print(f"  Gradient shape: {grad.shape}")

        # Patch-level analysis (skip CLS)
        act_patches = act[:, 1:, :]  # [1, 196, 768]
        grad_patches = grad[:, 1:, :]

        # Per-patch activation norms
        act_norms = act_patches.norm(dim=-1).squeeze()
        grad_norms = grad_patches.norm(dim=-1).squeeze()

        print(f"\n  Activation norms per patch:")
        print(f"    mean={act_norms.mean():.4f}, std={act_norms.std():.4f}, "
              f"CoV={act_norms.std()/act_norms.mean():.4f}")

        print(f"\n  Gradient norms per patch:")
        print(f"    mean={grad_norms.mean():.8f}, std={grad_norms.std():.8f}, "
              f"CoV={(grad_norms.std()/grad_norms.mean()):.6f}")

        # GradCAM = sum(grad * act, dim=-1)
        cam = (grad_patches * act_patches).sum(dim=-1).squeeze()  # [196]
        cam_abs = cam.abs()
        cam_grid = cam_abs.reshape(14, 14).detach().numpy()

        print(f"\n  GradCAM values per patch (grad*act summed over features):")
        print(f"    mean={cam_abs.mean():.8f}, std={cam_abs.std():.8f}, "
              f"CoV={(cam_abs.std()/cam_abs.mean()):.6f}")
        print(f"    Range: [{cam_grid.min():.8f}, {cam_grid.max():.8f}]")

        if cam_grid.max() - cam_grid.min() < cam_grid.mean() * 0.01:
            print("  >>> NEAR-UNIFORM CAM! Gradient signal doesn't discriminate patches. <<<")
        else:
            print(f"  Variation ratio: {(cam_grid.max()-cam_grid.min())/cam_grid.mean():.4f}")

        # Check different instructions
        print("\n  Comparing CAM across instructions:")
        instructions = ["pick up the red cup", "push the blue block", "grasp the green ball"]
        cams = {}
        for inst in instructions:
            model.zero_grad()
            engine._activations = None
            engine._gradients = None

            tf = engine._encode_text(inst)
            pv3 = pixel_values.clone().detach().requires_grad_(True)
            a = model.forward_for_gradcam(pv3, tf)
            a[0, 0].backward()

            if engine._activations is not None and engine._gradients is not None:
                ap = engine._activations[:, 1:, :]
                gp = engine._gradients[:, 1:, :]
                c = (gp * ap).sum(dim=-1).squeeze().abs()
                cams[inst] = c.reshape(14, 14).detach().numpy()

        if len(cams) >= 2:
            keys = list(cams.keys())
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    corr = np.corrcoef(cams[keys[i]].flatten(), cams[keys[j]].flatten())[0, 1]
                    print(f"    '{keys[i]}' vs '{keys[j]}': corr={corr:.4f}")

    for p in model.clip.vision_model.parameters():
        p.requires_grad_(False)

    # ================================================================
    # DIAGNOSTIC 3: Does CLIP-similarity saliency actually work?
    # ================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3: CLIP-Similarity Saliency")
    print("=" * 70)

    instructions_3 = ["pick up the red cup", "push the blue block", "grasp the green ball"]
    sim_maps = {}
    for inst in instructions_3:
        sal = engine.compute_similarity_saliency(scene, inst)
        sim_maps[inst] = sal
        print(f"  '{inst}': range=[{sal.min():.4f}, {sal.max():.4f}], mean={sal.mean():.4f}")

    print("\n  Cross-instruction correlations (CLIP-similarity):")
    keys = list(sim_maps.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            c = np.corrcoef(sim_maps[keys[i]].flatten(), sim_maps[keys[j]].flatten())[0, 1]
            print(f"    '{keys[i]}' vs '{keys[j]}': corr={c:.4f}")

    # ================================================================
    # DIAGNOSTIC 4: Multi-image training check
    # ================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 4: Architecture Bottleneck Analysis")
    print("=" * 70)
    print("""
  CRITICAL FINDING: The current architecture has a fundamental issue.

  Data flow:
    pixel_values -> CLIP ViT -> [1, 197, 768] patches
                                        |
                                patches[:, 1:, :].mean(dim=1)  <-- SPATIAL INFO DESTROYED
                                        |
                                [1, 768] pooled_vision
                                        |
                                vision_proj -> [1, 256]
                                        |
                              cat(vision, text) -> [1, 768]
                                        |
                                  action_head -> [1, 7]

  When we backpropagate from action[dim] through mean-pool:
    d(action[dim])/d(patch_i) = d(action[dim])/d(pooled) * d(pooled)/d(patch_i)
                               = d(action[dim])/d(pooled) * (1/N)  for ALL i

  The gradient is IDENTICAL for every patch! The model cannot tell which
  patch contributed to the action because mean-pooling erases that info.

  GradCAM = grad * activation. Since grad is uniform, the CAM pattern
  is entirely determined by activation magnitudes (which come from CLIP's
  pretrained ViT, not from action prediction). Different instructions
  give the same image -> same activations -> same GradCAM pattern.

  SOLUTION OPTIONS:
  A) Use attention-weighted pooling instead of mean pooling
     - Query = text features, Key/Value = patches
     - This preserves spatial selectivity in the gradient path
  B) Use per-patch action prediction (predict from each patch, aggregate)
  C) Skip pooling entirely, use cross-attention fusion
    """)

    engine.remove_hooks()

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
