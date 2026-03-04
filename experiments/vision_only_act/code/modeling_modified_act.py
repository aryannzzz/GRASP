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
"""Modified Action Chunking Transformer Policy with Images in VAE Encoder.

This is a modified version of ACT where the VAE encoder takes images as input
in addition to joint states and actions. This helps prevent overfitting on
joint states by conditioning the latent distribution on visual information.

Key modification from standard ACT:
    - VAE encoder input: [cls, robot_state, *pooled_image_features, *action_sequence]
    - Standard ACT VAE encoder: [cls, robot_state, *action_sequence]
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

# Import the modified config
try:
    from lerobot.policies.modified_act.configuration_modified_act import ModifiedACTConfig
except ImportError:
    # For standalone usage
    from configuration_modified_act import ModifiedACTConfig


class ModifiedACTPolicy(PreTrainedPolicy):
    """
    Modified Action Chunking Transformer Policy with images in the VAE encoder.
    
    This modification conditions the latent distribution on visual information
    in addition to joint states, which can help prevent overfitting on proprioceptive
    information alone.
    """

    config_class = ModifiedACTConfig
    name = "modified_act"

    def __init__(
        self,
        config: ModifiedACTConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ModifiedACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class ACTTemporalEnsembler:
    """Temporal ensembling as described in Algorithm 2 of the ACT paper."""
    
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """Update temporal ensemble and return the next action."""
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ModifiedACT(nn.Module):
    """Modified Action Chunking Transformer with images in VAE encoder.

    Architecture diagram:
    
                                     Transformer
                                     Used alone for inference
                                     (acts as VAE decoder
                                      during training)
                                    ┌───────────────────────┐
                                    │             Outputs   │
                                    │                ▲      │
                                    │     ┌─────►┌───────┐  │
                       ┌──────┐     │     │      │Transf.│  │
                       │      │     │     ├─────►│decoder│  │
                  ┌────┴────┐ │     │     │      │       │  │
                  │ Modified│ │     │ ┌───┴───┬─►│       │  │
                  │ VAE     │ │     │ │       │  └───────┘  │
                  │ encoder │ │     │ │Transf.│             │
                  │ +IMAGES │ │     │ │encoder│             │
                  └───▲─────┘ │     │ │       │             │
                      │       │     │ └▲──▲─▲─┘             │
                      │       │     │  │  │ │               │
                    inputs    └─────┼──┘  │ image emb.      │
                    +images         │    state emb.         │
                                    └───────────────────────┘
    
    Key modification: The VAE encoder now takes images as input in addition to
    joint states and actions, helping prevent overfitting on proprioceptive info.
    """

    def __init__(self, config: ModifiedACTConfig):
        super().__init__()
        self.config = config

        # Backbone for image feature extraction (shared between VAE encoder and main encoder)
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
            self.backbone_out_channels = backbone_model.fc.in_features

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            
            # Projection for robot state
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            
            # Projection for action sequence
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            
            # === MODIFIED: Image projection for VAE encoder ===
            if self.config.vae_encoder_use_images and self.config.image_features:
                # Project backbone features to model dimension
                self.vae_encoder_img_proj = nn.Linear(self.backbone_out_channels, config.dim_model)
                
                # If using spatial features, we need 2D positional embeddings
                if config.vae_encoder_image_pooling == "spatial":
                    self.vae_encoder_img_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)
            
            # Projection to latent space
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            
            # Calculate number of input tokens for positional embedding
            # [cls, (robot_state), (*image_tokens), *actions]
            num_input_token_encoder = 1 + config.chunk_size  # cls + actions
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            
            # For image tokens in VAE encoder
            if self.config.vae_encoder_use_images and self.config.image_features:
                num_cameras = len(self.config.image_features)
                if config.vae_encoder_image_pooling == "global_avg":
                    # One token per camera (global average pooled)
                    num_input_token_encoder += num_cameras
                # For spatial pooling, positional embeddings are handled separately
            
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Main transformer encoder
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Encoder input projections
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                self.backbone_out_channels, config.dim_model, kernel_size=1
            )
        
        # Encoder positional embeddings
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Decoder positional embeddings
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Action head
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode_images_for_vae(self, images: list[Tensor]) -> tuple[Tensor, Tensor | None]:
        """Encode images for the VAE encoder.
        
        Args:
            images: List of (B, C, H, W) image tensors, one per camera.
            
        Returns:
            image_tokens: (num_tokens, B, D) tensor of image tokens
            image_pos_embed: Position embeddings if using spatial pooling, else None
        """
        batch_size = images[0].shape[0]
        image_tokens_list = []
        image_pos_embed_list = []
        
        for img in images:
            # Extract features using shared backbone
            cam_features = self.backbone(img)["feature_map"]  # (B, C, H', W')
            
            if self.config.vae_encoder_image_pooling == "global_avg":
                # Global average pooling -> one token per camera
                pooled = cam_features.mean(dim=[2, 3])  # (B, C)
                token = self.vae_encoder_img_proj(pooled)  # (B, D)
                image_tokens_list.append(token.unsqueeze(0))  # (1, B, D)
            else:
                # Spatial features -> multiple tokens per camera
                B, C, H, W = cam_features.shape
                # Flatten spatial dimensions
                features = cam_features.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
                tokens = self.vae_encoder_img_proj(features)  # (H*W, B, D)
                image_tokens_list.append(tokens)
                
                # Generate 2D positional embeddings
                pos_embed = self.vae_encoder_img_pos_embed(cam_features)
                pos_embed = einops.rearrange(pos_embed, "b c h w -> (h w) b c")
                image_pos_embed_list.append(pos_embed)
        
        image_tokens = torch.cat(image_tokens_list, dim=0)  # (total_tokens, B, D)
        
        if self.config.vae_encoder_image_pooling == "spatial" and image_pos_embed_list:
            image_pos_embed = torch.cat(image_pos_embed_list, dim=0)
        else:
            image_pos_embed = None
            
        return image_tokens, image_pos_embed

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """Forward pass through the Modified ACT model.

        Args:
            batch: Dictionary containing:
                - [robot_state_feature] (optional): (B, state_dim) robot states
                - [image_features]: (B, n_cameras, C, H, W) images OR list of (B, C, H, W)
                - [env_state_feature] (optional): (B, env_dim) environment states  
                - [action_feature] (optional, training only): (B, chunk_size, action_dim) actions

        Returns:
            actions: (B, chunk_size, action_dim) predicted action sequence
            latent_params: Tuple of (mu, log_sigma_x2) or (None, None)
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # === VAE Encoder (Modified to include images) ===
        if self.config.use_vae and ACTION in batch and self.training:
            # Build VAE encoder input: [cls, (robot_state), (*image_tokens), *actions]
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            
            vae_encoder_input_list = [cls_embed]
            
            # Robot state embedding
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
                vae_encoder_input_list.append(robot_state_embed)
            
            # === MODIFIED: Add image embeddings to VAE encoder ===
            vae_image_pos_embed = None
            if self.config.vae_encoder_use_images and OBS_IMAGES in batch:
                image_tokens, vae_image_pos_embed = self._encode_images_for_vae(batch[OBS_IMAGES])
                # image_tokens is (num_tokens, B, D), convert to (B, num_tokens, D)
                image_tokens = image_tokens.permute(1, 0, 2)
                vae_encoder_input_list.append(image_tokens)
            
            # Action sequence embedding
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)
            vae_encoder_input_list.append(action_embed)
            
            vae_encoder_input = torch.cat(vae_encoder_input_list, dim=1)  # (B, total_seq, D)

            # Prepare positional embedding
            if self.config.vae_encoder_image_pooling == "global_avg" or not self.config.vae_encoder_use_images:
                # Use pre-computed sinusoidal positional embeddings
                pos_embed = self.vae_encoder_pos_enc[:, :vae_encoder_input.shape[1], :].clone().detach()
            else:
                # For spatial image features, we need to handle positions specially
                # This is more complex - build position embeddings dynamically
                num_1d_tokens = 1  # cls
                if self.config.robot_state_feature:
                    num_1d_tokens += 1
                num_action_tokens = batch[ACTION].shape[1]
                
                # 1D positional embeddings for cls, robot_state, actions
                pos_1d = create_sinusoidal_pos_embedding(
                    num_1d_tokens + num_action_tokens, self.config.dim_model
                ).to(vae_encoder_input.device)
                
                # Split for before and after image tokens
                pos_before_images = pos_1d[:num_1d_tokens].unsqueeze(0)
                pos_actions = pos_1d[num_1d_tokens:].unsqueeze(0)
                
                # Image positional embeddings (already computed)
                if vae_image_pos_embed is not None:
                    pos_images = vae_image_pos_embed.permute(1, 0, 2)  # (B, num_img_tokens, D)
                    pos_embed = torch.cat([pos_before_images.expand(batch_size, -1, -1), 
                                          pos_images, 
                                          pos_actions.expand(batch_size, -1, -1)], dim=1)
                else:
                    pos_embed = self.vae_encoder_pos_enc[:, :vae_encoder_input.shape[1], :].clone().detach()

            # Prepare key padding mask
            num_non_action_tokens = vae_encoder_input.shape[1] - batch[ACTION].shape[1]
            cls_state_img_is_pad = torch.full(
                (batch_size, num_non_action_tokens),
                False,
                device=vae_encoder_input.device,
            )
            key_padding_mask = torch.cat([cls_state_img_is_pad, batch["action_is_pad"]], dim=1)

            # Forward pass through VAE encoder
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),  # (S, B, D)
                pos_embed=pos_embed.permute(1, 0, 2),  # (S, 1, D) or (S, B, D)
                key_padding_mask=key_padding_mask,
            )[0]  # Select the class token: (B, D)
            
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample latent using reparameterization trick
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # During inference, use zero latent (as per original ACT)
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device if OBS_STATE in batch else batch[OBS_IMAGES][0].device
            )

        # === Main Transformer Encoder (same as original ACT) ===
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features and OBS_IMAGES in batch:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)

        # Forward through transformer encoder and decoder
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        decoder_out = decoder_out.transpose(0, 1)  # (B, S, D)
        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


# === Helper modules (same as original ACT) ===

class ACTEncoder(nn.Module):
    """Transformer encoder with optional pre-normalization."""

    def __init__(self, config: ModifiedACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(self, config: ModifiedACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    """Transformer decoder."""
    
    def __init__(self, config: ModifiedACTConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention and cross-attention."""
    
    def __init__(self, config: ModifiedACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings."""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings for image features."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, :1])
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency

        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
