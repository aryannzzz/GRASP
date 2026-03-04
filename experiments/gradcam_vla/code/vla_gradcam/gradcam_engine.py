"""
VLA-GradCAM Engine: Gradient-weighted Saliency for Vision-Language-Action Models

This module computes action-conditioned, language-conditioned GradCAM saliency
maps for VLA models. It hooks into the vision encoder's transformer layers
and backpropagates from each action dimension to reveal which image patches
the model attends to when predicting that action component.

Key novelty over standard GradCAM:
  1. Language-conditioned: different instructions -> different saliency
  2. Per-action-dimension: separate map for x, y, z, gripper, etc.
  3. Action-contrast: highlight what distinguishes one action dim from another
  4. Works with ViT patch features, not just CNN feature maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field
import cv2


@dataclass
class VLAGradCAMResult:
    """Container for a single VLA-GradCAM computation."""
    image: np.ndarray                      # Original image [H, W, 3] uint8
    instruction: str                       # Language instruction
    predicted_action: np.ndarray           # [action_dim] predicted action
    saliency_maps: Dict[str, np.ndarray]   # {action_name: [H, W] saliency}
    combined_saliency: np.ndarray          # [H, W] mean of all maps
    action_names: List[str]                # Names for action dimensions
    raw_patch_saliency: Dict[str, np.ndarray] = field(default_factory=dict)


class VLAGradCAMEngine:
    """
    GradCAM engine for CLIP-based VLA models.

    Hooks into the last transformer layer of CLIP's vision encoder
    to capture patch-level activations and gradients. Then computes
    per-action-dimension saliency maps.

    Usage:
        model, processor = load_clip_vla()
        engine = VLAGradCAMEngine(model)
        result = engine.compute(image, "pick up the red cup")
        engine.visualize(result)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_idx: int = -1,
        action_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model: CLIPVLA model instance
            target_layer_idx: Which vision encoder layer to hook (-1 = last)
            action_names: Names for action dimensions
        """
        self.model = model
        self.action_names = action_names or [
            'delta_x', 'delta_y', 'delta_z',
            'delta_roll', 'delta_pitch', 'delta_yaw', 'gripper',
        ]

        # Vision encoder config from CLIP
        self.patch_size = model.vision_config.patch_size
        self.image_size = model.vision_config.image_size
        self.n_patches = model.n_patches  # Patches per side (14 for 224/16)

        # Hook storage
        self._activations = None
        self._gradients = None
        self._handles = []

        # Register hook on target vision transformer layer
        vision_layers = model.clip.vision_model.encoder.layers
        target_layer = vision_layers[target_layer_idx]
        self._register_hooks(target_layer)

    def _register_hooks(self, target_layer: nn.Module):
        """Register forward/backward hooks to capture activations and gradients."""

        def fwd_hook(module, inp, out):
            # ViT layer outputs: (hidden_states, ...) or just hidden_states
            if isinstance(out, tuple):
                self._activations = out[0]
            else:
                self._activations = out

        def bwd_hook(module, grad_in, grad_out):
            if isinstance(grad_out, tuple):
                self._gradients = grad_out[0]
            else:
                self._gradients = grad_out

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)
        self._handles = [h1, h2]

    def remove_hooks(self):
        """Clean up hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []

    def _preprocess_image(
        self,
        image: Union[np.ndarray, 'PIL.Image.Image'],
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for CLIP.

        Returns:
            pixel_values: [1, 3, 224, 224] tensor
            original_image: [H, W, 3] uint8 numpy array
        """
        from PIL import Image as PILImage

        if isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            original = image.copy()
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = image
            original = np.array(pil_image)

        # Use CLIP processor for proper normalization
        inputs = self.model.processor(
            images=pil_image,
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"]  # [1, 3, 224, 224]
        return pixel_values, original

    def _encode_text(self, instruction: str) -> torch.Tensor:
        """Encode instruction with CLIP text encoder (no grad needed)."""
        device = next(self.model.parameters()).device
        text_inputs = self.model.processor(
            text=[instruction],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            text_outputs = self.model.clip.text_model(**text_inputs)
            text_feat = self.model.clip.text_projection(text_outputs.pooler_output)
            text_feat = F.normalize(text_feat, dim=-1)

        return text_feat  # [1, projection_dim]

    def _pixel_grad_to_saliency(
        self,
        pixel_grad: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """
        Convert raw pixel-level gradient [1, 3, 224, 224] into a smooth
        saliency heatmap by aggregating into ViT patch grid (14x14)
        then bilinear-upsampling and applying mild Gaussian smoothing.

        This produces the smooth, interpretable heatmaps GradCAM is
        known for, instead of noisy pixel-level gradients.
        """
        # Absolute gradient, mean over channels -> [224, 224]
        grad = pixel_grad.detach().abs().squeeze(0).mean(dim=0)  # [224, 224]

        # Aggregate into ViT patch grid: average each 16x16 block -> [14, 14]
        ps = self.patch_size  # 16
        npatch = self.n_patches  # 14
        grad_2d = grad[:npatch * ps, :npatch * ps]  # trim to exact grid
        # Reshape [14*16, 14*16] -> [14, 16, 14, 16] -> mean over (16, 16) dims
        patch_grid = grad_2d.reshape(npatch, ps, npatch, ps).mean(dim=(1, 3))
        patch_np = patch_grid.cpu().numpy()  # [14, 14]

        # Bilinear upsample to target size (produces smooth heatmap)
        cam = cv2.resize(patch_np, (target_w, target_h),
                         interpolation=cv2.INTER_LINEAR)

        # Mild Gaussian blur for extra smoothness
        ksize = max(3, target_h // 30) | 1  # ensure odd
        cam = cv2.GaussianBlur(cam, (ksize, ksize), 0)

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam.astype(np.float32)

    def _compute_cam_from_activations_grads(
        self,
        act: torch.Tensor,
        grad: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Compute GradCAM saliency from activations and gradients.

        For ViT models, we use element-wise grad*activation per patch
        (summed over feature dim) rather than the CNN-style
        global-avg-pool-weights approach, because ViT gradients are
        already per-token and averaging over spatial dims loses signal.

        Args:
            act: [B, 1+N, D] activations from hooked layer
            grad: [B, 1+N, D] gradients from hooked layer
            target_size: (H, W) to resize cam to

        Returns:
            cam: [H, W] numpy array normalized to [0, 1]
        """
        # Remove CLS token, keep only patch tokens
        act_patches = act[:, 1:, :]    # [B, N, D]
        grad_patches = grad[:, 1:, :]  # [B, N, D]

        # Element-wise gradient * activation, sum over features
        # This is the ViT-adapted GradCAM approach
        cam = (grad_patches * act_patches).sum(dim=-1)  # [B, N]

        # Reshape to spatial grid
        cam = cam.view(1, self.n_patches, self.n_patches)  # [B, h, w]

        # For ViT models, use absolute value rather than ReLU.
        # Unlike CNNs where only positive activations are meaningful,
        # in ViTs both positive and negative grad*act indicate importance
        # (a strong negative signal means the patch actively drives the
        # action dimension in the opposite direction, which is still salient).
        cam = cam.abs()

        cam = cam.squeeze(0).detach().cpu().numpy()  # [h, w]

        # Upsample
        cam_up = cv2.resize(cam, (target_size[1], target_size[0]),
                            interpolation=cv2.INTER_LINEAR)

        # Normalize
        if cam_up.max() > cam_up.min():
            cam_up = (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min())

        return cam_up.astype(np.float32)

    def compute(
        self,
        image: Union[np.ndarray, 'PIL.Image.Image'],
        instruction: str,
        action_dims: Optional[List[int]] = None,
    ) -> VLAGradCAMResult:
        """
        Compute per-action-dimension saliency maps.

        Args:
            image: Input image (numpy HWC uint8 or PIL Image)
            instruction: Language instruction
            action_dims: Which action dims to analyze (None = all)

        Returns:
            VLAGradCAMResult with saliency maps
        """
        device = next(self.model.parameters()).device

        # Preprocess
        pixel_values, original_image = self._preprocess_image(image)
        pixel_values = pixel_values.to(device)

        # Pre-compute text features (frozen, no gradient needed)
        text_features = self._encode_text(instruction)

        # Enable gradients on vision encoder for backprop
        for param in self.model.clip.vision_model.parameters():
            param.requires_grad_(True)

        # Determine action dims
        if action_dims is None:
            action_dims = list(range(len(self.action_names)))

        # Get predicted action first (for display)
        with torch.no_grad():
            pv_copy = pixel_values.clone()
            predicted_action = self.model.forward_for_gradcam(
                pv_copy, text_features.clone()
            ).cpu().numpy().squeeze()

        # Compute saliency for each action dimension
        saliency_maps = {}
        raw_patch_saliency = {}

        for dim_idx in action_dims:
            self.model.zero_grad()
            self._activations = None
            self._gradients = None

            # Forward pass WITH gradients through vision encoder
            pixel_values_grad = pixel_values.clone().detach().requires_grad_(True)
            action = self.model.forward_for_gradcam(pixel_values_grad, text_features)

            # Backprop from specific action dimension
            target = action[0, dim_idx]
            target.backward(retain_graph=False)

            dim_name = self.action_names[dim_idx] if dim_idx < len(self.action_names) else f'dim_{dim_idx}'

            # Primary method: proper GradCAM from hooked layer activations
            if self._activations is not None and self._gradients is not None:
                cam_up = self._compute_cam_from_activations_grads(
                    self._activations, self._gradients,
                    (original_image.shape[0], original_image.shape[1]),
                )
                saliency_maps[dim_name] = cam_up
            elif pixel_values_grad.grad is not None:
                # Fallback: pixel-gradient saliency
                cam_up = self._pixel_grad_to_saliency(
                    pixel_values_grad.grad,
                    original_image.shape[0], original_image.shape[1],
                )
                saliency_maps[dim_name] = cam_up

        # Re-freeze vision encoder
        for param in self.model.clip.vision_model.parameters():
            param.requires_grad_(False)

        # Combined saliency (mean across all dims)
        if saliency_maps:
            combined = np.mean(list(saliency_maps.values()), axis=0)
            if combined.max() > combined.min():
                combined = (combined - combined.min()) / (combined.max() - combined.min())
        else:
            combined = np.zeros(
                (original_image.shape[0], original_image.shape[1]),
                dtype=np.float32,
            )

        return VLAGradCAMResult(
            image=original_image,
            instruction=instruction,
            predicted_action=predicted_action,
            saliency_maps=saliency_maps,
            combined_saliency=combined,
            action_names=self.action_names,
            raw_patch_saliency=raw_patch_saliency,
        )

    def compute_similarity_saliency(
        self,
        image: Union[np.ndarray, 'PIL.Image.Image'],
        instruction: str,
    ) -> np.ndarray:
        """
        Compute saliency by backpropagating from the CLIP image-text
        similarity score to the input pixels. This leverages CLIP's
        pretrained alignment for strongly language-conditioned saliency.

        Uses input-gradient saliency (grad w.r.t. pixel_values) which
        works reliably for ViT models where standard GradCAM has issues
        with gradient flow through the CLS-token pooler.

        Returns: [H, W] saliency map normalized to [0, 1]
        """
        device = next(self.model.parameters()).device
        pixel_values, original_image = self._preprocess_image(image)
        pixel_values = pixel_values.to(device).requires_grad_(True)

        # Enable gradients on vision encoder
        for param in self.model.clip.vision_model.parameters():
            param.requires_grad_(True)

        self.model.zero_grad()

        # Use full CLIP forward pass
        text_inputs = self.model.processor(
            text=[instruction],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        clip_inputs = {
            'pixel_values': pixel_values,
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
        }

        outputs = self.model.clip(**clip_inputs)

        image_embeds = F.normalize(outputs.image_embeds, dim=-1)
        text_embeds = F.normalize(outputs.text_embeds, dim=-1)

        # Compute similarity and backpropagate to pixels
        similarity = (image_embeds * text_embeds).sum()
        similarity.backward()

        if pixel_values.grad is None:
            for param in self.model.clip.vision_model.parameters():
                param.requires_grad_(False)
            return np.zeros(
                (original_image.shape[0], original_image.shape[1]), dtype=np.float32
            )

        # Patch-level saliency (smooth heatmap)
        cam_up = self._pixel_grad_to_saliency(
            pixel_values.grad,
            original_image.shape[0], original_image.shape[1],
        )

        # Re-freeze
        for param in self.model.clip.vision_model.parameters():
            param.requires_grad_(False)

        return cam_up

    def compute_full(
        self,
        image: Union[np.ndarray, 'PIL.Image.Image'],
        instruction: str,
        action_dims: Optional[List[int]] = None,
        sim_weight: float = 0.4,
    ) -> VLAGradCAMResult:
        """
        Compute saliency using both CLIP similarity (language-conditioned)
        and action-head gradients, blended for the combined map.

        This gives the best of both worlds:
        - CLIP similarity captures "what the instruction refers to"
        - Action gradients capture "what drives the predicted action"
        - Combined = weighted blend of both signals

        Args:
            sim_weight: Weight for CLIP-similarity map in the blend (0-1).
                       The action-gradient mean gets (1-sim_weight).
        """
        # CLIP-similarity based saliency (strongly language-conditioned)
        sim_saliency = self.compute_similarity_saliency(image, instruction)

        # Action-head based per-dim saliency
        action_result = self.compute(image, instruction, action_dims=action_dims)

        # Blend: combine CLIP-similarity with mean of per-action saliency maps
        if action_result.saliency_maps:
            action_mean = np.mean(list(action_result.saliency_maps.values()), axis=0)
            if action_mean.max() > action_mean.min():
                action_mean = (action_mean - action_mean.min()) / (action_mean.max() - action_mean.min())
            combined = sim_weight * sim_saliency + (1.0 - sim_weight) * action_mean
            if combined.max() > combined.min():
                combined = (combined - combined.min()) / (combined.max() - combined.min())
        else:
            combined = sim_saliency

        return VLAGradCAMResult(
            image=action_result.image,
            instruction=instruction,
            predicted_action=action_result.predicted_action,
            saliency_maps=action_result.saliency_maps,
            combined_saliency=combined,
            action_names=action_result.action_names,
            raw_patch_saliency=action_result.raw_patch_saliency,
        )

    def compute_action_contrast(
        self,
        image: Union[np.ndarray, 'PIL.Image.Image'],
        instruction: str,
        dim_a: int,
        dim_b: int,
    ) -> np.ndarray:
        """
        Compute contrast saliency: what image regions drive dim_a vs dim_b.

        Positive values = favors dim_a, negative = favors dim_b.
        """
        result_a = self.compute(image, instruction, action_dims=[dim_a])
        result_b = self.compute(image, instruction, action_dims=[dim_b])

        name_a = self.action_names[dim_a]
        name_b = self.action_names[dim_b]

        # Unnormalized difference
        cam_a = result_a.saliency_maps.get(name_a, np.zeros_like(result_a.combined_saliency))
        cam_b = result_b.saliency_maps.get(name_b, np.zeros_like(result_b.combined_saliency))

        contrast = cam_a - cam_b

        # Normalize to [-1, 1]
        max_abs = max(abs(contrast.min()), abs(contrast.max()), 1e-8)
        contrast = contrast / max_abs

        return contrast

    def compute_instruction_contrast(
        self,
        image: Union[np.ndarray, 'PIL.Image.Image'],
        instruction_a: str,
        instruction_b: str,
        action_dim: int = 0,
    ) -> np.ndarray:
        """
        Compute contrast between two instructions on the same image.

        Uses CLIP-similarity saliency for strong language conditioning.
        Shows what image regions are differentially attended
        under instruction A vs instruction B.
        """
        cam_a = self.compute_similarity_saliency(image, instruction_a)
        cam_b = self.compute_similarity_saliency(image, instruction_b)

        contrast = cam_a - cam_b
        max_abs = max(abs(contrast.min()), abs(contrast.max()), 1e-8)
        return contrast / max_abs
