"""
Unified Attribution Module for RefinedVLA

Provides three attribution methods under a single interface:
    1. Attention-only: Raw cross-attention weights (baseline)
    2. GradCAM: Gradient-weighted activation on cross-attention value projections
    3. Integrated Gradients: Path-integrated gradients over patch embeddings

All methods return per-patch attribution scores that can be reshaped to a
spatial grid (e.g. 14x14 for ViT-B/16) and upsampled to image resolution.

Design:
    GradCAM hooks into RefinedCrossAttention.value_proj to capture the
    value representations and their gradients. This is the correct hook
    point because V encodes "what each patch contributes" and grad_V
    encodes "how much each patch's contribution affects the action".

    Integrated Gradients interpolates from a zero baseline to the actual
    patch embeddings, accumulating gradients along the path. This provides
    axiomatic attribution (completeness, sensitivity) that GradCAM lacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class AttributionResult:
    """Container for attribution computation results."""
    method: str                                  # "attention", "gradcam", "integrated_gradients"
    patch_scores: np.ndarray                     # [N] per-patch attribution scores
    spatial_map: np.ndarray                      # [grid_h, grid_w] reshaped patch scores
    heatmap: np.ndarray                          # [H, W] upsampled to image resolution
    action_dim: Optional[int] = None             # Which action dim (None = combined)
    predicted_action: Optional[np.ndarray] = None


@dataclass
class MultiMethodResult:
    """Container for results from all attribution methods."""
    attention: AttributionResult
    gradcam: AttributionResult
    integrated_gradients: AttributionResult
    predicted_action: np.ndarray
    instruction: str


class UnifiedAttribution:
    """
    Unified attribution module for RefinedVLA.

    Computes per-patch importance scores using three methods:
    - attention: Direct attention weights from cross-attention
    - gradcam: grad * activation on value projections
    - integrated_gradients: path-integrated gradients over patches

    Usage:
        model, processor = load_refined_vla()
        attr = UnifiedAttribution(model)

        # Single method
        result = attr.compute_gradcam(pixel_values, text_features)

        # All methods
        results = attr.compute_all(pixel_values, text_features)

        # Clean up
        attr.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            model: RefinedVLA model instance
            target_size: (H, W) for upsampled heatmaps
        """
        self.model = model
        self.target_size = target_size

        # Spatial grid dimensions from model config
        self.n_patches_side = model.n_patches_side  # 14 for ViT-B/16
        self.n_patches = model.n_patches             # 196

        # Hook storage for GradCAM
        self._value_activation = None
        self._value_gradient = None
        self._hooks = []

        # Register hooks on cross-attention value projection
        self._register_gradcam_hooks()

    def _register_gradcam_hooks(self):
        """Register forward/backward hooks on the value projection in cross-attention."""
        value_proj = self.model.attention_pool.value_proj

        def fwd_hook(module, inp, out):
            self._value_activation = out

        def bwd_hook(module, grad_in, grad_out):
            # grad_out is a tuple; first element is gradient w.r.t. output
            self._value_gradient = grad_out[0] if isinstance(grad_out, tuple) else grad_out

        h1 = value_proj.register_forward_hook(fwd_hook)
        h2 = value_proj.register_full_backward_hook(bwd_hook)
        self._hooks = [h1, h2]

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _patches_to_heatmap(self, patch_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert per-patch scores to spatial grid and upsampled heatmap.

        Args:
            patch_scores: [N] per-patch importance scores

        Returns:
            spatial_map: [grid_h, grid_w] reshaped scores
            heatmap: [H, W] bilinear-upsampled and normalized
        """
        n = self.n_patches_side
        spatial = patch_scores[:n * n].reshape(n, n)

        # Bilinear upsample
        heatmap = cv2.resize(spatial, (self.target_size[1], self.target_size[0]),
                             interpolation=cv2.INTER_LINEAR)

        # Mild Gaussian blur for smoothness
        ksize = max(3, self.target_size[0] // 30) | 1
        heatmap = cv2.GaussianBlur(heatmap, (ksize, ksize), 0)

        # Normalize to [0, 1]
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        return spatial.astype(np.float32), heatmap.astype(np.float32)

    # ------------------------------------------------------------------
    # Method 1: Attention-only (baseline)
    # ------------------------------------------------------------------

    def compute_attention(
        self,
        pixel_values: torch.Tensor,
        text_features: torch.Tensor,
    ) -> AttributionResult:
        """
        Compute attention-only attribution (baseline).

        Returns the mean attention weights across heads from
        RefinedCrossAttention, reshaped as a spatial heatmap.

        Args:
            pixel_values: [B, 3, H, W] preprocessed images (B=1)
            text_features: [B, D_text] precomputed text features

        Returns:
            AttributionResult with attention-based heatmap
        """
        with torch.no_grad():
            action, attn, _ = self.model(pixel_values, text_features)

        # attn: [B, N] mean attention across heads
        patch_scores = attn[0].cpu().numpy()

        # Normalize
        if patch_scores.max() > patch_scores.min():
            patch_scores = (patch_scores - patch_scores.min()) / (patch_scores.max() - patch_scores.min())

        spatial, heatmap = self._patches_to_heatmap(patch_scores)

        return AttributionResult(
            method="attention",
            patch_scores=patch_scores,
            spatial_map=spatial,
            heatmap=heatmap,
            predicted_action=action[0].cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # Method 2: GradCAM on cross-attention value projections
    # ------------------------------------------------------------------

    def compute_gradcam(
        self,
        pixel_values: torch.Tensor,
        text_features: torch.Tensor,
        action_dim: Optional[int] = None,
    ) -> AttributionResult:
        """
        Compute GradCAM attribution via grad * activation on value projections.

        Hooks capture V = value_proj(patches) and its gradient.
        GradCAM score per patch = |sum_d(grad_V[n,d] * V[n,d])|

        If action_dim is None, backpropagates from the sum of all action dims.

        Args:
            pixel_values: [B, 3, H, W] (B=1)
            text_features: [B, D_text]
            action_dim: Specific action dimension to attribute, or None for all

        Returns:
            AttributionResult with GradCAM heatmap
        """
        self.model.zero_grad()
        self._value_activation = None
        self._value_gradient = None

        # Forward pass with gradients enabled
        # Temporarily enable grads on vision encoder for backprop
        vision_params_state = {}
        for name, param in self.model.clip.vision_model.named_parameters():
            vision_params_state[name] = param.requires_grad
            param.requires_grad_(True)

        action, attn, _ = self.model(pixel_values, text_features)

        # Backward from selected action dimension(s)
        if action_dim is not None:
            target = action[0, action_dim]
        else:
            target = action[0].sum()

        target.backward(retain_graph=False)

        # Restore vision encoder grad state
        for name, param in self.model.clip.vision_model.named_parameters():
            param.requires_grad_(vision_params_state[name])

        # Compute GradCAM from hooked activations and gradients
        if self._value_activation is None or self._value_gradient is None:
            # Fallback: uniform
            patch_scores = np.ones(self.n_patches, dtype=np.float32) / self.n_patches
        else:
            # V activation: [B, N, hidden_dim]
            act = self._value_activation[0].detach()  # [N, hidden_dim]
            grad = self._value_gradient[0].detach()    # [N, hidden_dim]

            # GradCAM: element-wise product, sum over feature dim
            cam = (grad * act).sum(dim=-1)  # [N]

            # Use absolute value (both positive and negative signals are important)
            cam = cam.abs()

            patch_scores = cam.cpu().numpy()

            # Normalize
            if patch_scores.max() > patch_scores.min():
                patch_scores = (patch_scores - patch_scores.min()) / (patch_scores.max() - patch_scores.min())

        spatial, heatmap = self._patches_to_heatmap(patch_scores)

        return AttributionResult(
            method="gradcam",
            patch_scores=patch_scores,
            spatial_map=spatial,
            heatmap=heatmap,
            action_dim=action_dim,
            predicted_action=action[0].detach().cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # Method 3: Integrated Gradients over patch embeddings
    # ------------------------------------------------------------------

    def compute_integrated_gradients(
        self,
        pixel_values: torch.Tensor,
        text_features: torch.Tensor,
        action_dim: Optional[int] = None,
        n_steps: int = 50,
        baseline: Optional[torch.Tensor] = None,
    ) -> AttributionResult:
        """
        Compute Integrated Gradients over patch embeddings.

        Interpolates from a zero baseline to actual patch features,
        accumulating gradients at each step. Satisfies completeness
        and sensitivity axioms (Sundararajan et al., 2017).

        Args:
            pixel_values: [B, 3, H, W] (B=1)
            text_features: [B, D_text]
            action_dim: Action dimension to attribute, or None for sum
            n_steps: Number of interpolation steps (higher = more accurate)
            baseline: Custom baseline [B, N, D]; defaults to zeros

        Returns:
            AttributionResult with IG heatmap
        """
        device = pixel_values.device

        # Get actual patch features (detached)
        with torch.no_grad():
            patches_actual = self.model.get_patch_features(pixel_values)  # [B, N, D]

        # Baseline: zeros by default
        if baseline is None:
            patches_baseline = torch.zeros_like(patches_actual)
        else:
            patches_baseline = baseline

        # Accumulate gradients along interpolation path
        integrated_grads = torch.zeros_like(patches_actual)  # [B, N, D]

        for step in range(n_steps + 1):
            alpha = step / n_steps

            # Interpolated patches
            patches_interp = patches_baseline + alpha * (patches_actual - patches_baseline)
            patches_interp = patches_interp.detach().requires_grad_(True)

            # Forward from interpolated patches
            action, _, _ = self.model.forward_from_patches(patches_interp, text_features)

            # Backward
            if action_dim is not None:
                target = action[0, action_dim]
            else:
                target = action[0].sum()

            self.model.zero_grad()
            target.backward(retain_graph=False)

            if patches_interp.grad is not None:
                integrated_grads += patches_interp.grad.detach()

        # Riemann sum approximation
        integrated_grads = integrated_grads * (patches_actual - patches_baseline) / (n_steps + 1)

        # Per-patch score: sum absolute IG over feature dimension
        patch_scores = integrated_grads[0].abs().sum(dim=-1).cpu().numpy()  # [N]

        # Normalize
        if patch_scores.max() > patch_scores.min():
            patch_scores = (patch_scores - patch_scores.min()) / (patch_scores.max() - patch_scores.min())

        spatial, heatmap = self._patches_to_heatmap(patch_scores)

        # Get predicted action for reference
        with torch.no_grad():
            action_pred, _, _ = self.model(pixel_values, text_features)

        return AttributionResult(
            method="integrated_gradients",
            patch_scores=patch_scores,
            spatial_map=spatial,
            heatmap=heatmap,
            action_dim=action_dim,
            predicted_action=action_pred[0].cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # Combined: all three methods
    # ------------------------------------------------------------------

    def compute_all(
        self,
        pixel_values: torch.Tensor,
        text_features: torch.Tensor,
        instruction: str = "",
        action_dim: Optional[int] = None,
        ig_steps: int = 50,
    ) -> MultiMethodResult:
        """
        Compute all three attribution methods for comparison.

        Args:
            pixel_values: [B, 3, H, W] (B=1)
            text_features: [B, D_text]
            instruction: Text instruction for metadata
            action_dim: Action dimension, or None for combined
            ig_steps: Steps for integrated gradients

        Returns:
            MultiMethodResult with all three method results
        """
        attn_result = self.compute_attention(pixel_values, text_features)
        gradcam_result = self.compute_gradcam(pixel_values, text_features, action_dim)
        ig_result = self.compute_integrated_gradients(
            pixel_values, text_features, action_dim, n_steps=ig_steps
        )

        return MultiMethodResult(
            attention=attn_result,
            gradcam=gradcam_result,
            integrated_gradients=ig_result,
            predicted_action=attn_result.predicted_action,
            instruction=instruction,
        )

    # ------------------------------------------------------------------
    # Diagnostic: Verify spatial gradient non-uniformity
    # ------------------------------------------------------------------

    def verify_gradient_nonuniformity(
        self,
        pixel_values: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Verify that GradCAM produces spatially non-uniform attributions.

        Checks:
        1. Coefficient of variation (std/mean) of patch scores
        2. Max/min ratio of patch scores
        3. Entropy relative to uniform distribution
        4. Gini coefficient

        A properly functioning model should produce non-uniform scores
        (CV > 0.3, entropy < 0.95 * max_entropy).

        Returns:
            Dict with diagnostic metrics
        """
        gradcam = self.compute_gradcam(pixel_values, text_features)
        scores = gradcam.patch_scores

        # Basic stats
        mean = scores.mean()
        std = scores.std()
        cv = std / (mean + 1e-8)

        # Max/min ratio
        max_min_ratio = (scores.max() + 1e-8) / (scores.min() + 1e-8)

        # Entropy (normalized scores as probability)
        probs = scores / (scores.sum() + 1e-8)
        probs = probs + 1e-10  # avoid log(0)
        entropy = -(probs * np.log(probs)).sum()
        max_entropy = np.log(len(scores))
        normalized_entropy = entropy / max_entropy

        # Gini coefficient
        sorted_scores = np.sort(scores)
        n = len(sorted_scores)
        cumsum = np.cumsum(sorted_scores)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_scores)) / (n * np.sum(sorted_scores) + 1e-8)) - (n + 1) / n

        is_nonuniform = cv > 0.3 and normalized_entropy < 0.95

        return {
            "coefficient_of_variation": float(cv),
            "max_min_ratio": float(max_min_ratio),
            "normalized_entropy": float(normalized_entropy),
            "gini_coefficient": float(gini),
            "is_nonuniform": bool(is_nonuniform),
            "mean": float(mean),
            "std": float(std),
            "n_patches": int(len(scores)),
        }

    # ------------------------------------------------------------------
    # Cross-instruction comparison
    # ------------------------------------------------------------------

    def compute_instruction_correlation(
        self,
        pixel_values: torch.Tensor,
        text_features_a: torch.Tensor,
        text_features_b: torch.Tensor,
        method: str = "gradcam",
    ) -> float:
        """
        Compute correlation between attributions for two different instructions
        on the same image. Low correlation indicates instruction-sensitivity.

        Args:
            pixel_values: [B, 3, H, W]
            text_features_a: [B, D_text] first instruction
            text_features_b: [B, D_text] second instruction
            method: "attention", "gradcam", or "integrated_gradients"

        Returns:
            Pearson correlation coefficient (-1 to 1)
        """
        compute_fn = {
            "attention": self.compute_attention,
            "gradcam": self.compute_gradcam,
            "integrated_gradients": self.compute_integrated_gradients,
        }[method]

        result_a = compute_fn(pixel_values, text_features_a)
        result_b = compute_fn(pixel_values, text_features_b)

        scores_a = result_a.patch_scores
        scores_b = result_b.patch_scores

        # Pearson correlation
        corr = np.corrcoef(scores_a, scores_b)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
