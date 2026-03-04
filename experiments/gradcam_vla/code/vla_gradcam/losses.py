"""
Unified Loss Module for RefinedVLA Training

Loss components:
    1. Action MSE: Primary action prediction loss
    2. Contrastive Object Discrimination: Forces the model to distinguish
       objects via the object classifier head
    3. Coverage Regularization: Encourages attention to cover a reasonable
       spatial extent (prevents degenerate single-patch attention)
    4. Attention Diversity: Penalizes identical attention patterns for
       semantically different instructions

All components are combined with configurable weights under VLALoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class VLALoss(nn.Module):
    """
    Unified loss for RefinedVLA training.

    Combines action prediction, contrastive grounding, coverage
    regularization, and attention diversity losses.
    """

    def __init__(
        self,
        action_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        coverage_weight: float = 0.1,
        diversity_weight: float = 0.1,
    ):
        """
        Args:
            action_weight: Weight for action MSE loss
            contrastive_weight: Weight for contrastive object discrimination
            coverage_weight: Weight for attention coverage regularization
            diversity_weight: Weight for attention diversity loss
        """
        super().__init__()
        self.action_weight = action_weight
        self.contrastive_weight = contrastive_weight
        self.coverage_weight = coverage_weight
        self.diversity_weight = diversity_weight

        self.action_criterion = nn.MSELoss()
        self.contrastive_criterion = nn.CrossEntropyLoss()

    def action_loss(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Action prediction loss (MSE).

        Args:
            pred_action: [B, action_dim] predicted actions
            target_action: [B, action_dim] expert actions

        Returns:
            Scalar MSE loss
        """
        return self.action_criterion(pred_action, target_action)

    def contrastive_loss(
        self,
        obj_logits: torch.Tensor,
        obj_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Contrastive object discrimination loss.

        Forces the model to correctly identify which object the instruction
        refers to, strengthening the vision-language binding.

        Args:
            obj_logits: [B, n_classes] object classification logits
            obj_labels: [B] integer class labels

        Returns:
            Scalar cross-entropy loss
        """
        return self.contrastive_criterion(obj_logits, obj_labels)

    def coverage_loss(
        self,
        attn_weights: torch.Tensor,
        target_coverage: float = 0.1,
    ) -> torch.Tensor:
        """
        Attention coverage regularization.

        Encourages attention to cover at least target_coverage fraction
        of patches, preventing degenerate single-patch attention.

        Uses soft coverage: sum of attn values above a threshold.

        Args:
            attn_weights: [B, N] attention weights (sum to 1)
            target_coverage: Desired fraction of patches receiving attention

        Returns:
            Scalar coverage penalty
        """
        B, N = attn_weights.shape

        # Entropy-based: higher entropy = more spread
        # Maximize entropy up to a point
        eps = 1e-8
        entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(float(N), device=attn_weights.device))

        # Penalize when entropy is too low (attention too concentrated)
        # Target: entropy should be at least target_coverage * max_entropy
        target_entropy = target_coverage * max_entropy
        penalty = F.relu(target_entropy - entropy).mean()

        return penalty

    def diversity_loss(
        self,
        attn_weights: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attention diversity loss.

        For pairs with dissimilar text features (different instructions),
        penalizes similar attention patterns. This forces the model to
        attend differently based on instruction content.

        Args:
            attn_weights: [B, N] attention weights
            text_features: [B, D] text feature vectors

        Returns:
            Scalar diversity penalty
        """
        B = attn_weights.size(0)
        if B < 2:
            return torch.tensor(0.0, device=attn_weights.device)

        # Pairwise text similarity
        text_sim = torch.mm(
            F.normalize(text_features, dim=-1),
            F.normalize(text_features, dim=-1).t()
        )
        text_sim.fill_diagonal_(0)

        # Pairwise attention similarity
        attn_norm = F.normalize(attn_weights, p=2, dim=1)
        attn_sim = torch.mm(attn_norm, attn_norm.t())
        attn_sim.fill_diagonal_(0)

        # Penalize high attention similarity for dissimilar text pairs
        dissimilar_mask = (text_sim < 0.9).float()
        loss = (dissimilar_mask * attn_sim).sum() / (dissimilar_mask.sum() + 1e-8)

        return loss

    def forward(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        attn_weights: torch.Tensor,
        text_features: torch.Tensor,
        obj_logits: Optional[torch.Tensor] = None,
        obj_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components and weighted total.

        Args:
            pred_action: [B, action_dim] predicted actions
            target_action: [B, action_dim] expert actions
            attn_weights: [B, N] attention weights
            text_features: [B, D] text features
            obj_logits: [B, n_classes] object logits (optional)
            obj_labels: [B] object labels (optional)

        Returns:
            Dict with 'total', 'action', 'contrastive', 'coverage', 'diversity'
        """
        losses = {}

        # Action MSE (always present)
        losses["action"] = self.action_loss(pred_action, target_action)

        # Contrastive (only if labels provided)
        if obj_logits is not None and obj_labels is not None and self.contrastive_weight > 0:
            losses["contrastive"] = self.contrastive_loss(obj_logits, obj_labels)
        else:
            losses["contrastive"] = torch.tensor(0.0, device=pred_action.device)

        # Coverage regularization
        if self.coverage_weight > 0:
            losses["coverage"] = self.coverage_loss(attn_weights)
        else:
            losses["coverage"] = torch.tensor(0.0, device=pred_action.device)

        # Diversity
        if self.diversity_weight > 0:
            losses["diversity"] = self.diversity_loss(attn_weights, text_features)
        else:
            losses["diversity"] = torch.tensor(0.0, device=pred_action.device)

        # Weighted total
        losses["total"] = (
            self.action_weight * losses["action"]
            + self.contrastive_weight * losses["contrastive"]
            + self.coverage_weight * losses["coverage"]
            + self.diversity_weight * losses["diversity"]
        )

        return losses
