"""
VLA-GradCAM: Gradient-weighted Attribution for Vision-Language-Action Models

Primary architecture: RefinedVLA with multi-head cross-attention.
Attribution: Unified module supporting attention, GradCAM, and Integrated Gradients.

Key Components:
- RefinedVLA: Primary VLA model with cross-attention and vision-only action head
- UnifiedAttribution: Attribution module with 3 methods
- VLALoss: Unified training loss (action + contrastive + coverage + diversity)
- VLAGradCAMEngine: Lower-level GradCAM engine (ViT layer hooks)
"""

# Primary model
from .refined_vla import RefinedVLA, RefinedCrossAttention, load_refined_vla, ACTION_NAMES

# Attribution
from .attribution import UnifiedAttribution, AttributionResult, MultiMethodResult

# Losses
from .losses import VLALoss

# GradCAM engine (lower-level, hooks into ViT layers)
from .gradcam_engine import VLAGradCAMEngine, VLAGradCAMResult

# Visualization
from .visualizer import VLAGradCAMVisualizer, overlay_heatmap

__all__ = [
    # Primary model
    'RefinedVLA',
    'RefinedCrossAttention',
    'load_refined_vla',
    'ACTION_NAMES',
    # Attribution
    'UnifiedAttribution',
    'AttributionResult',
    'MultiMethodResult',
    # Losses
    'VLALoss',
    # Engine
    'VLAGradCAMEngine',
    'VLAGradCAMResult',
    # Visualization
    'VLAGradCAMVisualizer',
    'overlay_heatmap',
]

__version__ = '1.0.0'
