"""
Debug script to check if attention pooling is working correctly.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from pybullet_vla.env import setup_env
from vla_gradcam.clip_vla_attn import load_clip_vla_attn

def check_attention_conditioning():
    """Check if attention weights change with different instructions."""

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint_path = "outputs/pybullet_vla_final/training/best_model.pt"
    print(f"\nLoading model from {checkpoint_path}...")

    # Load base model
    model, processor = load_clip_vla_attn(device=device)

    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Set up environment
    print("\nSetting up environment...")
    env, object_names = setup_env(n_objects=3, gui=False, image_size=224)
    print(f"Objects: {object_names}")

    # Capture image
    image = env.render(view="topdown")
    print(f"Image shape: {image.shape}")

    # Preprocess image
    from PIL import Image as PILImage
    pil_image = PILImage.fromarray(image)
    inputs = processor(images=pil_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # Test instructions
    instructions = [
        "pick up the duck",
        "pick up the sphere",
        "pick up the cube",
        "move to the top right corner",
    ]

    print("\n" + "="*60)
    print("CHECKING TEXT FEATURES AND ATTENTION WEIGHTS")
    print("="*60)

    text_features_list = []
    attention_weights_list = []
    actions_list = []

    for inst in instructions:
        # Encode text
        text_inputs = processor(
            text=[inst],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            text_outputs = model.clip.text_model(**text_inputs)
            text_feat = model.clip.text_projection(text_outputs.pooler_output)
            text_feat = torch.nn.functional.normalize(text_feat, dim=-1)

        # Get attention weights
        with torch.no_grad():
            patch_features = model.encode_image_patches(pixel_values)
            patches = patch_features[:, 1:, :]
            _, attn_weights = model.attention_pool(patches, text_feat)

            # Also get predicted action
            action = model.forward_for_gradcam(pixel_values, text_feat)

        text_features_list.append(text_feat.cpu().numpy())
        attention_weights_list.append(attn_weights.cpu().numpy())
        actions_list.append(action.cpu().numpy())

        print(f"\n'{inst}':")
        print(f"  Text feat norm: {torch.norm(text_feat).item():.4f}")
        print(f"  Text feat [:5]: {text_feat[0, :5].cpu().numpy()}")
        print(f"  Attention weights - min: {attn_weights.min().item():.6f}, "
              f"max: {attn_weights.max().item():.6f}, "
              f"std: {attn_weights.std().item():.6f}")
        print(f"  Top 5 patches: {torch.topk(attn_weights, 5).indices.cpu().numpy()[0]}")
        print(f"  Predicted action: {action[0].cpu().numpy()}")

    # Check if text features are different
    print("\n" + "="*60)
    print("TEXT FEATURE DIFFERENCES")
    print("="*60)
    for i, inst_i in enumerate(instructions):
        for j, inst_j in enumerate(instructions):
            if i < j:
                feat_i = text_features_list[i]
                feat_j = text_features_list[j]
                diff = np.linalg.norm(feat_i - feat_j)
                cosine_sim = np.dot(feat_i[0], feat_j[0]) / (np.linalg.norm(feat_i) * np.linalg.norm(feat_j))
                print(f"'{inst_i}' vs '{inst_j}':")
                print(f"  L2 distance: {diff:.4f}, Cosine similarity: {cosine_sim:.4f}")

    # Check if attention weights are different
    print("\n" + "="*60)
    print("ATTENTION WEIGHT DIFFERENCES")
    print("="*60)
    for i, inst_i in enumerate(instructions):
        for j, inst_j in enumerate(instructions):
            if i < j:
                attn_i = attention_weights_list[i][0]
                attn_j = attention_weights_list[j][0]
                diff = np.linalg.norm(attn_i - attn_j)
                cosine_sim = np.dot(attn_i, attn_j) / (np.linalg.norm(attn_i) * np.linalg.norm(attn_j))
                print(f"'{inst_i}' vs '{inst_j}':")
                print(f"  L2 distance: {diff:.6f}, Cosine similarity: {cosine_sim:.6f}")

    # Check if actions are different
    print("\n" + "="*60)
    print("ACTION PREDICTION DIFFERENCES")
    print("="*60)
    for i, inst_i in enumerate(instructions):
        for j, inst_j in enumerate(instructions):
            if i < j:
                action_i = actions_list[i][0]
                action_j = actions_list[j][0]
                diff = np.linalg.norm(action_i - action_j)
                print(f"'{inst_i}' vs '{inst_j}':")
                print(f"  L2 distance: {diff:.6f}")

    # Visualize attention weights
    print("\n" + "="*60)
    print("ATTENTION WEIGHT HEATMAPS")
    print("="*60)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(instructions), figsize=(16, 4))

    for i, (inst, attn) in enumerate(zip(instructions, attention_weights_list)):
        attn_2d = attn[0].reshape(14, 14)  # 14x14 patches for 224x224 image
        axes[i].imshow(attn_2d, cmap='jet')
        axes[i].set_title(inst, fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    output_path = "outputs/attention_debug.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved attention visualization to {output_path}")

    env.disconnect()

    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    print("""
If attention weights are nearly identical (cosine sim > 0.99) despite
different text features, it means:
  1. The attention pooling didn't learn to condition on text
  2. Need to debug training or increase attention capacity

If attention weights ARE different but GradCAM looks same:
  1. GradCAM implementation bug
  2. Need to check gradient flow through attention layer
    """)

if __name__ == "__main__":
    check_attention_conditioning()
