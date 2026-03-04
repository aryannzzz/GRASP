#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration for Modified ACT with images in VAE encoder."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("modified_act")
@dataclass
class ModifiedACTConfig(PreTrainedConfig):
    """Configuration class for the Modified Action Chunking Transformers policy.
    
    This is a modified version of ACT where the VAE encoder also takes images as input,
    not just joint states and actions. This helps prevent overfitting on joint states
    by conditioning the latent distribution on visual information as well.

    The key difference from standard ACT:
        - VAE encoder input: [cls, robot_state, *image_features, *action_sequence]
        - Standard ACT VAE encoder input: [cls, robot_state, *action_sequence]

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation.
        normalization_mapping: Dictionary mapping feature types to normalization modes.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final stride with dilation.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads for multi-head attention.
        dim_feedforward: The dimension for feed-forward layers.
        feedforward_activation: The activation function for feed-forward layers.
        n_encoder_layers: The number of transformer layers for the main encoder.
        n_decoder_layers: The number of transformer layers for the decoder.
        use_vae: Whether to use a variational objective during training.
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers for the VAE encoder.
        vae_encoder_use_images: Whether the VAE encoder should take images as input (the key modification).
        vae_encoder_image_pooling: How to pool image features for VAE encoder ('global_avg', 'spatial').
        temporal_ensemble_coeff: Coefficient for temporal ensembling. None disables it.
        dropout: Dropout rate for transformer layers.
        kl_weight: Weight for the KL-divergence component of the loss.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    
    # Modified ACT specific: VAE encoder with images
    vae_encoder_use_images: bool = True  # The key modification flag
    vae_encoder_image_pooling: str = "global_avg"  # 'global_avg' or 'spatial'

    # Inference.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        if self.vae_encoder_image_pooling not in ("global_avg", "spatial"):
            raise ValueError(
                f"`vae_encoder_image_pooling` must be 'global_avg' or 'spatial'. Got {self.vae_encoder_image_pooling}."
            )
        if self.vae_encoder_use_images and not self.use_vae:
            raise ValueError(
                "`vae_encoder_use_images` requires `use_vae=True`."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
        if self.vae_encoder_use_images and not self.image_features:
            raise ValueError(
                "`vae_encoder_use_images=True` requires at least one image feature to be defined."
            )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
