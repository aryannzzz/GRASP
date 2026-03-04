"""
CLIP-based Vision-Language-Action Model

A lightweight VLA that uses CLIP's vision-language encoder as the backbone
and adds an MLP action head to predict robot actions.

This is a genuine VLA that processes both image AND language instruction
to produce 7-DOF robot actions (dx, dy, dz, droll, dpitch, dyaw, gripper).

Architecture:
    Image ──→ CLIP Vision Encoder ──→ Vision Features [CLS, patches...]
                                            │
    Text  ──→ CLIP Text Encoder  ──→ Text Features [CLS]
                                            │
                               ┌────────────┘
                               ↓
                    Concat(vision_cls, text_cls)
                               ↓
                         MLP Action Head
                               ↓
                    Action [7-DOF: x,y,z,r,p,y,g]

The GradCAM hooks into CLIP's vision encoder transformer layers,
so saliency maps show which image patches drive each action dimension
CONDITIONED on the language instruction (because text affects the
features through the shared CLIP embedding space).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple
from PIL import Image

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


class CLIPVLA(nn.Module):
    """
    CLIP-based Vision-Language-Action model.

    Uses pretrained CLIP for vision-language understanding,
    adds an action head for robot control predictions.
    Language conditioning is inherent - different instructions
    produce different text embeddings that change the action output.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        clip_processor,
        action_dim: int = 7,
        hidden_dim: int = 256,
        use_patch_features: bool = True,
    ):
        super().__init__()
        self.clip = clip_model
        self.processor = clip_processor
        self.use_patch_features = use_patch_features

        # CLIP config
        self.vision_config = clip_model.config.vision_config
        self.hidden_size = self.vision_config.hidden_size  # 768 for base
        self.patch_size = self.vision_config.patch_size      # 16 for base
        self.image_size = self.vision_config.image_size      # 224 for base
        self.n_patches = (self.image_size // self.patch_size)  # 14

        # Vision projection: pool patch features into a single vector
        if use_patch_features:
            # Use attention-pooled patch features + text features
            self.vision_proj = nn.Linear(self.hidden_size, hidden_dim)
            action_input_dim = hidden_dim + self.clip.config.projection_dim
        else:
            # Just use CLIP's projected embeddings
            action_input_dim = self.clip.config.projection_dim * 2

        # Action head
        self.action_head = CLIPActionHead(action_input_dim, action_dim, hidden_dim)

        # Freeze CLIP backbone (we only train the action head)
        for param in self.clip.parameters():
            param.requires_grad = False

    def encode_image_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get vision transformer patch features (before projection)."""
        vision_outputs = self.clip.vision_model(pixel_values)
        # last_hidden_state: [B, 1+num_patches, hidden_size]
        return vision_outputs.last_hidden_state

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

        # Get patch-level vision features (keeps spatial info for GradCAM)
        patch_features = self.encode_image_patches(image)
        # patch_features: [B, 1+196, 768]

        # Pool patch features (skip CLS token at index 0)
        patches = patch_features[:, 1:, :]  # [B, 196, 768]
        pooled_vision = patches.mean(dim=1)  # [B, 768]

        if self.use_patch_features:
            vision_feat = self.vision_proj(pooled_vision)  # [B, hidden_dim]
        else:
            # Use CLIP's projected vision embedding
            vision_feat = self.clip.visual_projection(
                self.clip.vision_model(image).pooler_output
            )

        # Get text features via CLIP
        text_inputs = self.processor(
            text=[instruction] * image.shape[0],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        text_outputs = self.clip.text_model(**text_inputs)
        text_feat = self.clip.text_projection(text_outputs.pooler_output)
        text_feat = F.normalize(text_feat, dim=-1)

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
        This is used by GradCAM so we can backpropagate through vision only.

        Args:
            pixel_values: [B, 3, 224, 224] preprocessed image
            text_features: [B, text_dim] pre-computed text features

        Returns:
            action: [B, action_dim]
        """
        # Get patch features (this is what GradCAM hooks into)
        patch_features = self.encode_image_patches(pixel_values)
        patches = patch_features[:, 1:, :]
        pooled_vision = patches.mean(dim=1)

        if self.use_patch_features:
            vision_feat = self.vision_proj(pooled_vision)
        else:
            vision_feat = self.clip.visual_projection(
                self.clip.vision_model(pixel_values).pooler_output
            )

        combined = torch.cat([vision_feat, text_features], dim=-1)
        action = self.action_head(combined)
        return action


def load_clip_vla(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    device: str = "cpu",
) -> Tuple[CLIPVLA, CLIPProcessor]:
    """
    Load a CLIP-based VLA model.

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

    vla = CLIPVLA(clip_model, processor)
    vla = vla.to(device)
    vla.eval()

    n_params_total = sum(p.numel() for p in vla.parameters())
    n_params_trainable = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    print(f"Total params: {n_params_total/1e6:.1f}M, Trainable: {n_params_trainable/1e6:.2f}M")

    return vla, processor
