#!/usr/bin/env python3
"""
VLA GradCAM Demo with Robot Manipulation Images

Uses CLIP to demonstrate language-conditioned visual attention
on robot manipulation scenarios.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import requests
from io import BytesIO


def create_robot_scene():
    """Create a synthetic robot manipulation scene."""
    # Create a 480x480 RGB image
    img = np.ones((480, 480, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Table surface (brown)
    img[300:, :] = [139, 90, 43]  # Brown table
    
    # Red cup (target object)
    cv2.circle(img, (150, 280), 40, (200, 50, 50), -1)
    cv2.ellipse(img, (150, 240), (40, 15), 0, 0, 360, (180, 40, 40), -1)
    
    # Blue block
    cv2.rectangle(img, (300, 260), (360, 300), (50, 50, 200), -1)
    
    # Green cylinder
    cv2.ellipse(img, (420, 280), (30, 40), 0, 0, 360, (50, 180, 50), -1)
    cv2.ellipse(img, (420, 240), (30, 10), 0, 0, 360, (40, 150, 40), -1)
    
    # Robot gripper (gray)
    cv2.rectangle(img, (200, 100), (280, 180), (100, 100, 100), -1)
    # Gripper fingers
    cv2.rectangle(img, (205, 180), (225, 250), (80, 80, 80), -1)
    cv2.rectangle(img, (255, 180), (275, 250), (80, 80, 80), -1)
    
    # Robot arm (dark gray)
    cv2.rectangle(img, (220, 50), (260, 100), (60, 60, 60), -1)
    
    return img


def load_robot_image():
    """Try to load a real robot manipulation image, fall back to synthetic."""
    # Try some robot manipulation image URLs
    urls = [
        "https://ai.stanford.edu/~openvla/images/openvla_teaser.png",
        "https://rail.eecs.berkeley.edu/datasets/bridge_release/images/bridge.png",
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert('RGB')
                return np.array(img), "real"
        except:
            continue
    
    # Fall back to synthetic
    print("Using synthetic robot scene")
    return create_robot_scene(), "synthetic"


class CLIPGradCAM:
    """CLIP-based GradCAM for language-guided attention."""
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.activations = None
        self.gradients = None
        
        # Hook last vision encoder layer
        target = model.vision_model.encoder.layers[-1]
        target.register_forward_hook(self._fwd_hook)
        target.register_full_backward_hook(self._bwd_hook)
        
        self.n_patches = model.config.vision_config.image_size // model.config.vision_config.patch_size
    
    def _fwd_hook(self, m, i, o):
        self.activations = o[0].detach() if isinstance(o, tuple) else o.detach()
    
    def _bwd_hook(self, m, gi, go):
        self.gradients = go[0].detach() if isinstance(go, tuple) else go.detach()
    
    def __call__(self, image, texts):
        """Compute saliency for each text query."""
        inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True)
        
        results = {}
        for i, text in enumerate(texts):
            self.model.zero_grad()
            
            outputs = self.model(**inputs)
            img_emb = F.normalize(outputs.image_embeds, dim=-1)
            txt_emb = F.normalize(outputs.text_embeds, dim=-1)
            
            similarity = (img_emb @ txt_emb[i:i+1].T).squeeze()
            similarity.backward(retain_graph=True)
            
            if self.gradients is None:
                continue
            
            # Compute saliency from patches (skip CLS token)
            act = self.activations[:, 1:, :]
            grad = self.gradients[:, 1:, :]
            
            weights = grad.mean(dim=2, keepdim=True)
            sal = F.relu((act * weights).sum(dim=2))
            sal = sal.view(self.n_patches, self.n_patches).cpu().numpy()
            
            if sal.max() > 0:
                sal = (sal - sal.min()) / (sal.max() - sal.min())
            
            results[text] = sal
        
        # Also get similarity scores
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = (100 * outputs.logits_per_image).softmax(dim=1)[0]
        
        return results, {t: scores[i].item() for i, t in enumerate(texts)}


def main():
    print("=" * 70)
    print("VLA GradCAM Demo - Language-Guided Robot Attention")
    print("=" * 70)
    
    from transformers import CLIPModel, CLIPProcessor
    
    # Load CLIP
    print("\nLoading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model.eval()
    
    gradcam = CLIPGradCAM(model, processor)
    
    # Load/create robot image
    print("\nPreparing robot manipulation image...")
    img, img_type = load_robot_image()
    print(f"Image type: {img_type}, shape: {img.shape}")
    
    # Robot manipulation queries (like VLA instructions)
    queries = [
        "pick up the red cup",
        "push the blue block",
        "grasp the green object",
        "the robot gripper",
        "the table surface"
    ]
    
    print(f"\nComputing attention for {len(queries)} robot instructions...")
    saliency_maps, scores = gradcam(Image.fromarray(img), queries)
    
    # Print similarity scores
    print("\nInstruction-Image Relevance Scores:")
    for query, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  '{query}': {score:.1%}")
    
    # Visualize
    output_dir = Path("/home/aryannzzz/GRASP/GradCAM/outputs/vla_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n = len(saliency_maps)
    fig, axes = plt.subplots(2, n + 1, figsize=(4 * (n + 1), 8))
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Robot Scene", fontsize=12)
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Per-query saliency
    for i, (query, sal) in enumerate(saliency_maps.items()):
        sal_resized = cv2.resize(sal, (img.shape[1], img.shape[0]))
        
        # Saliency map
        axes[0, i + 1].imshow(sal_resized, cmap='jet')
        axes[0, i + 1].set_title(f'"{query}"', fontsize=10)
        axes[0, i + 1].axis('off')
        
        # Overlay
        axes[1, i + 1].imshow(img)
        axes[1, i + 1].imshow(sal_resized, cmap='jet', alpha=0.5)
        axes[1, i + 1].set_title(f"Score: {scores[query]:.1%}", fontsize=10)
        axes[1, i + 1].axis('off')
    
    plt.suptitle("VLA GradCAM: What does the model attend to for each instruction?", fontsize=14)
    plt.tight_layout()
    
    save_path = output_dir / "vla_demo_gradcam.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {save_path}")
    plt.close()
    
    # Create a focused comparison
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(img)
    axes[0].set_title("Original Scene")
    axes[0].axis('off')
    
    # Show three most relevant queries
    top_queries = sorted(scores.items(), key=lambda x: -x[1])[:3]
    for i, (query, score) in enumerate(top_queries):
        sal = saliency_maps[query]
        sal_resized = cv2.resize(sal, (img.shape[1], img.shape[0]))
        
        axes[i + 1].imshow(img)
        axes[i + 1].imshow(sal_resized, cmap='jet', alpha=0.5)
        axes[i + 1].set_title(f'"{query}"\n{score:.1%}', fontsize=11)
        axes[i + 1].axis('off')
    
    plt.suptitle("Language-Conditioned Attention: Different instructions → Different focus regions", fontsize=12)
    plt.tight_layout()
    
    save_path = output_dir / "vla_demo_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to: {save_path}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Insight: Different language instructions cause the model to")
    print("attend to different regions of the image - this is exactly what")
    print("VLA-GradCAM aims to reveal for robot action prediction!")


if __name__ == "__main__":
    main()
