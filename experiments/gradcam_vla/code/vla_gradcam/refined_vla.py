"""
RefinedVLA: Unified Vision-Language-Action Model with Attribution Support

Architecture:
    Image -> CLIP ViT -> patch features [B, N, D]
    Text  -> CLIP Text -> text features [B, D_text]
    patches + text -> RefinedCrossAttention (multi-head, learnable temp) -> vision_feat [B, H]
    vision_feat -> ActionHead (vision-only, no text shortcut) -> action [B, 7]
    vision_feat -> ObjectClassifier -> object logits [B, n_classes]

Key design decisions:
    1. Vision-only action head forces the model to encode action-relevant info
       through the vision pathway (text-conditioned via attention), preventing
       text->action shortcuts that bypass visual grounding.
    2. Multi-head cross-attention with learnable per-head temperature allows
       both sharp (object-focused) and broad (context-aware) attention patterns.
    3. Positional encoding preserves spatial structure for attribution.
    4. Object classifier enables contrastive grounding loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

try:
    from transformers import CLIPModel, CLIPProcessor
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


ACTION_NAMES = [
    'delta_x', 'delta_y', 'delta_z',
    'delta_roll', 'delta_pitch', 'delta_yaw',
    'gripper',
]


class RefinedCrossAttention(nn.Module):
    """
    Multi-head cross-attention with learnable per-head temperature.

    Text features serve as query, vision patch features as key/value.
    Each head can learn a different temperature, allowing the model to
    maintain both sharp and diffuse attention patterns simultaneously.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_patches: int = 196,
        init_temperature: float = 0.2,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.hidden_dim = hidden_dim

        self.query_proj = nn.Linear(text_dim, hidden_dim)
        self.key_proj = nn.Linear(vision_dim, hidden_dim)
        self.value_proj = nn.Linear(vision_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Per-head learnable temperature
        self.temperature = nn.Parameter(
            torch.ones(n_heads) * init_temperature
        )

        # Learnable positional encoding for patches
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_patches, hidden_dim) * 0.02
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        patches: torch.Tensor,
        text_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: [B, N, D_vision] patch features
            text_feat: [B, D_text] text features

        Returns:
            output: [B, hidden_dim] aggregated vision features
            attn_weights: [B, N] mean attention weights across heads
        """
        B, N, D = patches.shape
        H = self.n_heads
        d = self.head_dim

        Q = self.query_proj(text_feat).view(B, 1, H, d).permute(0, 2, 1, 3)
        K = (self.key_proj(patches) + self.pos_encoding[:, :N, :]).view(B, N, H, d).permute(0, 2, 1, 3)
        V = self.value_proj(patches).view(B, N, H, d).permute(0, 2, 1, 3)

        temp = self.temperature.view(1, H, 1, 1).clamp(min=0.01)
        scores = (Q @ K.transpose(-2, -1)) / (math.sqrt(d) * temp)
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, 1, N]

        output = (attn_weights @ V).squeeze(2).reshape(B, self.hidden_dim)
        output = self.layer_norm(self.output_proj(output))

        # Mean attention across heads for visualization
        mean_attn = attn_weights.mean(dim=1).squeeze(1)  # [B, N]
        return output, mean_attn


class RefinedVLA(nn.Module):
    """
    Unified Vision-Language-Action model with attribution support.

    Combines CLIP vision/text encoders with multi-head cross-attention
    and a vision-only action head. Designed for clean GradCAM/IG
    attribution through the attention value projections.
    """

    def __init__(
        self,
        clip_model: 'CLIPModel',
        clip_processor: 'CLIPProcessor',
        action_dim: int = 7,
        hidden_dim: int = 256,
        n_heads: int = 8,
        finetune_vit_layers: int = 4,
        n_object_classes: int = 5,
        init_temperature: float = 0.2,
    ):
        """
        Args:
            clip_model: Pretrained CLIP model
            clip_processor: CLIP processor for tokenization/image preprocessing
            action_dim: Number of action dimensions (7 for 6-DOF + gripper)
            hidden_dim: Hidden dimension for cross-attention and action head
            n_heads: Number of attention heads
            finetune_vit_layers: Number of last ViT layers to fine-tune (0=frozen)
            n_object_classes: Number of object classes for contrastive classifier
            init_temperature: Initial attention temperature
        """
        super().__init__()
        self.clip = clip_model
        self.processor = clip_processor
        self.vision_config = clip_model.config.vision_config
        self.hidden_size = self.vision_config.hidden_size
        self.patch_size = self.vision_config.patch_size
        self.image_size = self.vision_config.image_size
        self.n_patches_side = self.image_size // self.patch_size
        self.n_patches = self.n_patches_side ** 2

        # Cross-attention pooling
        self.attention_pool = RefinedCrossAttention(
            vision_dim=self.hidden_size,
            text_dim=clip_model.config.projection_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_patches=self.n_patches,
            init_temperature=init_temperature,
        )

        # Vision-only action head (no text concatenation)
        # Forces the model to encode action-relevant info through
        # the vision pathway, ensuring visual grounding.
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, action_dim),
        )

        # Object classifier for contrastive grounding
        self.object_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_object_classes),
        )

        # Freeze CLIP backbone
        for param in self.clip.parameters():
            param.requires_grad = False

        # Optionally fine-tune last N ViT layers
        if finetune_vit_layers > 0:
            for layer in self.clip.vision_model.encoder.layers[-finetune_vit_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            if hasattr(self.clip.vision_model, 'post_layernorm'):
                for param in self.clip.vision_model.post_layernorm.parameters():
                    param.requires_grad = True

    def encode_image_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level features from CLIP ViT.

        Args:
            pixel_values: [B, 3, H, W] preprocessed images

        Returns:
            hidden_states: [B, 1+N, D] (CLS + patch tokens)
        """
        return self.clip.vision_model(pixel_values).last_hidden_state

    def encode_text(self, text_inputs: dict) -> torch.Tensor:
        """
        Encode text with CLIP and project to shared space.

        Args:
            text_inputs: Tokenized text inputs dict

        Returns:
            text_feat: [B, D_text] normalized text features
        """
        text_outputs = self.clip.text_model(**text_inputs)
        text_feat = self.clip.text_projection(text_outputs.pooler_output)
        return F.normalize(text_feat, dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            pixel_values: [B, 3, H, W] preprocessed images
            text_features: [B, D_text] precomputed text features

        Returns:
            action: [B, action_dim] predicted action clamped to [-1, 1]
            attn: [B, N] attention weights
            obj_logits: [B, n_classes] object classification logits
        """
        patches = self.encode_image_patches(pixel_values)[:, 1:, :]
        vis, attn = self.attention_pool(patches, text_features)
        action = self.action_head(vis).clamp(-1.0, 1.0)
        obj_logits = self.object_classifier(vis)
        return action, attn, obj_logits

    def forward_for_gradcam(
        self,
        pixel_values: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass returning only the action, for GradCAM backward.

        Args:
            pixel_values: [B, 3, H, W]
            text_features: [B, D_text]

        Returns:
            action: [B, action_dim]
        """
        return self.forward(pixel_values, text_features)[0]

    def get_attention_weights(
        self,
        pixel_values: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get attention weights without gradients.

        Returns:
            attn_weights: [B, N]
        """
        with torch.no_grad():
            patches = self.encode_image_patches(pixel_values)[:, 1:, :]
            _, attn = self.attention_pool(patches, text_features)
        return attn

    def get_patch_features(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get raw patch features (for integrated gradients baseline).

        Returns:
            patches: [B, N, D]
        """
        return self.encode_image_patches(pixel_values)[:, 1:, :]

    def forward_from_patches(
        self,
        patches: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass from pre-computed patch features.
        Used by Integrated Gradients to interpolate patch embeddings.

        Args:
            patches: [B, N, D] patch features (may have gradients attached)
            text_features: [B, D_text] precomputed text features

        Returns:
            action: [B, action_dim]
            attn: [B, N]
            obj_logits: [B, n_classes]
        """
        vis, attn = self.attention_pool(patches, text_features)
        action = self.action_head(vis).clamp(-1.0, 1.0)
        obj_logits = self.object_classifier(vis)
        return action, attn, obj_logits


def load_refined_vla(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    device: str = "cpu",
    **kwargs,
) -> Tuple['RefinedVLA', 'CLIPProcessor']:
    """
    Load RefinedVLA model with CLIP backbone.

    Args:
        clip_model_name: HuggingFace CLIP model name
        device: Device to load on
        **kwargs: Additional args passed to RefinedVLA

    Returns:
        (model, processor) tuple
    """
    if not HAS_CLIP:
        raise ImportError("transformers required: pip install transformers")

    clip_model = CLIPModel.from_pretrained(clip_model_name)
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    model = RefinedVLA(clip_model, processor, **kwargs)
    model = model.to(device)
    model.eval()

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"RefinedVLA loaded: {n_total/1e6:.1f}M params, {n_trainable/1e6:.2f}M trainable")

    return model, processor
