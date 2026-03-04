"""
CLIP-based VLA with Attention-Weighted Pooling (Fixed Architecture)

This fixes the mean-pooling bottleneck by using text-conditioned attention
to aggregate patch features. Now gradients flow through attention weights,
enabling meaningful GradCAM visualization.

Key change:
  OLD: pooled = patches.mean(dim=1)  # Uniform gradients!
  NEW: pooled = attention_pool(patches, text_feat)  # Gradients via attention!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple
from PIL import Image
import math

try:
    from transformers import CLIPModel, CLIPProcessor
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


# 7-DOF robot action space
ACTION_NAMES = [
    'delta_x',      # End-effector X
    'delta_y',      # End-effector Y
    'delta_z',      # End-effector Z
    'delta_roll',   # Rotation X
    'delta_pitch',  # Rotation Y
    'delta_yaw',    # Rotation Z
    'gripper',      # Gripper open/close
]


class AttentionPooling(nn.Module):
    """
    Text-conditioned attention pooling for patch features.

    Given:
      - patches: [B, N_patches, D_vision]
      - text_feat: [B, D_text]

    Output:
      - pooled: [B, D_hidden]
      - attn_weights: [B, N_patches]

    The attention weights depend on text, so different instructions
    produce different attention patterns. Gradients flow through these
    attention weights, enabling GradCAM to see which patches matter.
    """

    def __init__(self, vision_dim: int, text_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project patches and text to same dimension for attention
        self.query_proj = nn.Linear(text_dim, hidden_dim)
        self.key_proj = nn.Linear(vision_dim, hidden_dim)
        self.value_proj = nn.Linear(vision_dim, hidden_dim)

    def forward(
        self,
        patches: torch.Tensor,
        text_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: [B, N, D_vision]
            text_feat: [B, D_text]

        Returns:
            output: [B, D_hidden]
            attn_weights: [B, N]
        """
        B, N, D = patches.shape

        # Compute attention: text as query, patches as key/value
        Q = self.query_proj(text_feat).unsqueeze(1)  # [B, 1, H]
        K = self.key_proj(patches)  # [B, N, H]
        V = self.value_proj(patches)  # [B, N, H]

        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.hidden_dim)  # [B, 1, N]
        attn_weights = F.softmax(scores, dim=-1)  # [B, 1, N]

        # Aggregate patches
        output = (attn_weights @ V).squeeze(1)  # [B, H]

        return output, attn_weights.squeeze(1)  # [B, H], [B, N]


class CLIPActionHead(nn.Module):
    """MLP that maps CLIP features to robot actions."""

    def __init__(self, input_dim: int, action_dim: int = 7, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Actions in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CLIPVLA_AttnPool(nn.Module):
    """
    CLIP-based VLA with attention-weighted pooling.

    Architecture:
        Image → CLIP ViT → patches [B, N, D]
        Text → CLIP Text → text_feat [B, D_text]
        patches + text_feat → AttentionPooling → pooled [B, H]
        pooled + text_feat → ActionHead → action [B, 7]

    Key innovation: Attention pooling allows gradients to flow per-patch,
    enabling meaningful GradCAM visualization.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        clip_processor,
        action_dim: int = 7,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.clip = clip_model
        self.processor = clip_processor

        # CLIP config
        self.vision_config = clip_model.config.vision_config
        self.hidden_size = self.vision_config.hidden_size  # 768 for base
        self.patch_size = self.vision_config.patch_size      # 16 for base
        self.image_size = self.vision_config.image_size      # 224 for base
        self.n_patches = (self.image_size // self.patch_size)  # 14

        # Text-conditioned attention pooling (replaces mean pooling)
        self.attention_pool = AttentionPooling(
            vision_dim=self.hidden_size,
            text_dim=self.clip.config.projection_dim,
            hidden_dim=hidden_dim,
        )

        # Action head
        action_input_dim = hidden_dim + self.clip.config.projection_dim
        self.action_head = CLIPActionHead(action_input_dim, action_dim, hidden_dim)

        # Freeze CLIP backbone
        for param in self.clip.parameters():
            param.requires_grad = False

    def encode_image_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get vision transformer patch features (before projection)."""
        vision_outputs = self.clip.vision_model(pixel_values)
        return vision_outputs.last_hidden_state  # [B, 1+N, D]

    def forward(
        self,
        image: torch.Tensor,
        instruction: str,
    ) -> torch.Tensor:
        """
        Forward pass: image + instruction -> action.

        Args:
            image: [B, 3, 224, 224] preprocessed image tensor
            instruction: text instruction string

        Returns:
            action: [B, action_dim] predicted action
        """
        device = image.device

        # Get patch-level vision features
        patch_features = self.encode_image_patches(image)  # [B, 1+N, D]
        patches = patch_features[:, 1:, :]  # [B, N, D] (skip CLS)

        # Get text features
        text_inputs = self.processor(
            text=[instruction] * image.shape[0],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        text_outputs = self.clip.text_model(**text_inputs)
        text_feat = self.clip.text_projection(text_outputs.pooler_output)
        text_feat = F.normalize(text_feat, dim=-1)

        # Attention-pooling (text-conditioned)
        vision_feat, attn_weights = self.attention_pool(patches, text_feat)

        # Concatenate vision + text features
        combined = torch.cat([vision_feat, text_feat], dim=-1)

        # Predict action
        action = self.action_head(combined)
        return action

    def forward_for_gradcam(
        self,
        pixel_values: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with pre-computed text features.
        Used by GradCAM so we only backprop through vision.

        Args:
            pixel_values: [B, 3, 224, 224]
            text_features: [B, D_text] pre-computed

        Returns:
            action: [B, action_dim]
        """
        patch_features = self.encode_image_patches(pixel_values)
        patches = patch_features[:, 1:, :]

        # Attention-pool with text
        vision_feat, _ = self.attention_pool(patches, text_features)

        combined = torch.cat([vision_feat, text_features], dim=-1)
        action = self.action_head(combined)
        return action

    def get_attention_weights(
        self,
        image: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get attention weights for visualization.

        Returns:
            attn_weights: [B, N_patches]
        """
        patch_features = self.encode_image_patches(image)
        patches = patch_features[:, 1:, :]
        _, attn_weights = self.attention_pool(patches, text_features)
        return attn_weights


def load_clip_vla_attn(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    device: str = "cpu",
) -> Tuple[CLIPVLA_AttnPool, CLIPProcessor]:
    """
    Load CLIP-based VLA model with attention pooling.

    Args:
        clip_model_name: HuggingFace CLIP model name
        device: Device to load on

    Returns:
        (model, processor) tuple
    """
    if not HAS_CLIP:
        raise ImportError("transformers required: pip install transformers")

    print(f"Loading CLIP backbone: {clip_model_name}")
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    vla = CLIPVLA_AttnPool(clip_model, processor)
    vla = vla.to(device)
    vla.eval()

    n_params_total = sum(p.numel() for p in vla.parameters())
    n_params_trainable = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    print(f"Total params: {n_params_total/1e6:.1f}M, Trainable: {n_params_trainable/1e6:.2f}M")

    return vla, processor
