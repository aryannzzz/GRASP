#!/usr/bin/env python3
"""
ACT (Action Chunking Transformer) Integration for VLA-GradCAM

This module provides GradCAM saliency analysis for ACT models trained
on MetaWorld tasks in the GRASP environment.

ACT Architecture:
- Vision Encoder: ResNet18 → [B, 512, H/32, W/32] → projected to [B, N_patches, 512]
- CVAE Encoder: (joints, actions) → latent z
- Decoder: Cross-attention between queries and (image_tokens, joint_token, latent_token)
- Output: Action chunk [B, chunk_size, action_dim]

For GradCAM, we target the ResNet18's final conv layer (layer4).
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import cv2

# Add ACT-modification to path
ACT_PATH = Path("/home/aryannzzz/GRASP/ACT-modification")
if str(ACT_PATH) not in sys.path:
    sys.path.insert(0, str(ACT_PATH))

try:
    from models.standard_act import StandardACT, ResNetEncoder
    from models.modified_act import ModifiedACT
    HAS_ACT = True
except ImportError as e:
    print(f"Warning: Could not import ACT models: {e}")
    HAS_ACT = False


@dataclass
class ACTSaliencyResult:
    """Container for ACT saliency analysis results."""
    image: np.ndarray                      # Original image [H, W, C]
    joints: np.ndarray                     # Joint state
    predicted_actions: np.ndarray          # Predicted action chunk [chunk_size, action_dim]
    saliency_maps: Dict[str, np.ndarray]   # Per-action-dim or per-timestep saliency
    combined_saliency: np.ndarray          # Combined saliency map
    action_names: List[str]                # Names of action dimensions
    timestep: int                          # Which timestep in chunk was analyzed


# MetaWorld action space (typical 4-DOF + gripper)
METAWORLD_ACTION_NAMES = [
    'delta_x',      # End-effector X movement
    'delta_y',      # End-effector Y movement
    'delta_z',      # End-effector Z movement
    'gripper',      # Gripper command
]

# Full 8-DOF action space if using full joint control
METAWORLD_FULL_ACTION_NAMES = [
    'joint_0', 'joint_1', 'joint_2', 'joint_3',
    'joint_4', 'joint_5', 'joint_6', 'gripper'
]


class ACTGradCAM:
    """
    GradCAM for ACT (Action Chunking Transformer) models.
    
    Targets the ResNet18 vision encoder's final convolutional layer.
    Computes saliency for specific action dimensions at specific timesteps.
    
    Usage:
        model = load_act_model("checkpoints/best_model.pth")
        gradcam = ACTGradCAM(model)
        result = gradcam.compute_saliency(image, joints, action_dim=0, timestep=0)
        gradcam.visualize(result)
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: str = 'decoder.resnet.features.7',  # ResNet layer4
        action_names: List[str] = None,
        image_size: Tuple[int, int] = (128, 128),
    ):
        """
        Args:
            model: ACT model (StandardACT or ModifiedACT)
            target_layer: Path to target layer for GradCAM
            action_names: Names for action dimensions
            image_size: Expected input image size
        """
        self.model = model
        self.target_layer_name = target_layer
        self.action_names = action_names or METAWORLD_FULL_ACTION_NAMES
        self.image_size = image_size
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Find and hook the target layer
        self.target_layer = self._get_target_layer()
        self._register_hooks()
        
    def _get_target_layer(self) -> nn.Module:
        """Navigate to the target layer."""
        layer = self.model
        for name in self.target_layer_name.split('.'):
            if name.isdigit():
                layer = layer[int(name)]
            else:
                layer = getattr(layer, name)
        return layer
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.clone()
            self.activations.requires_grad_(True)
            self.activations.retain_grad()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].clone() if isinstance(grad_output, tuple) else grad_output.clone()
        
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
    
    def remove_hooks(self):
        """Remove registered hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def compute_saliency(
        self,
        image: Union[torch.Tensor, np.ndarray],
        joints: Union[torch.Tensor, np.ndarray],
        action_dims: Optional[List[int]] = None,
        timestep: int = 0,
        normalize: bool = True,
    ) -> ACTSaliencyResult:
        """
        Compute saliency for ACT model.
        
        Args:
            image: Input image [H, W, C] or [C, H, W]
            joints: Joint state [joint_dim]
            action_dims: Which action dimensions to analyze (None = all)
            timestep: Which timestep in the action chunk to analyze
            normalize: Whether to normalize saliency maps
            
        Returns:
            ACTSaliencyResult containing saliency maps
        """
        self.model.eval()
        
        # Prepare image
        if isinstance(image, np.ndarray):
            original_image = image.copy()
            if image.ndim == 3 and image.shape[-1] == 3:
                # HWC -> CHW
                image = torch.from_numpy(image).permute(2, 0, 1).float()
            else:
                image = torch.from_numpy(image).float()
            if image.max() > 1.0:
                image = image / 255.0
        else:
            original_image = image.permute(1, 2, 0).cpu().numpy()
            
        # Prepare joints
        if isinstance(joints, np.ndarray):
            original_joints = joints.copy()
            joints = torch.from_numpy(joints).float()
        else:
            original_joints = joints.cpu().numpy()
        
        # Add batch dimensions
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if joints.dim() == 1:
            joints = joints.unsqueeze(0)
            
        # Move to model device
        device = next(self.model.parameters()).device
        image = image.to(device)
        joints = joints.to(device)
        image.requires_grad = True
        
        # Create images dict (ACT expects dict)
        images = {'corner': image}
        
        # Determine action dims to analyze
        action_dim = self.model.decoder.action_head.out_features
        if action_dims is None:
            action_dims = list(range(min(action_dim, len(self.action_names))))
        
        # Forward pass (inference mode)
        with torch.enable_grad():
            pred_actions = self.model(images, joints, training=False)
            
        predicted_actions = pred_actions.detach().cpu().numpy().squeeze()
        
        # Compute saliency for each action dimension
        saliency_maps = {}
        
        for dim_idx in action_dims:
            self.model.zero_grad()
            if image.grad is not None:
                image.grad.zero_()
            
            # Forward again for fresh graph
            pred_actions = self.model(images, joints, training=False)
            
            # Target specific action at specific timestep
            target = pred_actions[0, timestep, dim_idx]
            
            # Backward
            target.backward(retain_graph=True)
            
            if self.gradients is None:
                print(f"Warning: No gradients captured for dim {dim_idx}")
                continue
            
            # GradCAM computation
            # Global average pool of gradients: [B, C, H, W] -> [B, C, 1, 1]
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            
            # Weighted combination of feature maps
            saliency = (self.activations * weights).sum(dim=1)  # [B, H, W]
            
            # ReLU (only positive contributions)
            saliency = F.relu(saliency)
            
            # Upsample to image size
            saliency = F.interpolate(
                saliency.unsqueeze(1),
                size=self.image_size,
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            saliency = saliency.detach().cpu().numpy()
            
            if normalize and saliency.max() > 0:
                saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            
            dim_name = self.action_names[dim_idx] if dim_idx < len(self.action_names) else f'dim_{dim_idx}'
            saliency_maps[dim_name] = saliency
        
        # Combined saliency
        if saliency_maps:
            combined = np.mean(list(saliency_maps.values()), axis=0)
            if normalize and combined.max() > 0:
                combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
        else:
            combined = np.zeros(self.image_size)
        
        return ACTSaliencyResult(
            image=original_image,
            joints=original_joints,
            predicted_actions=predicted_actions,
            saliency_maps=saliency_maps,
            combined_saliency=combined,
            action_names=self.action_names,
            timestep=timestep
        )
    
    def compute_temporal_saliency(
        self,
        image: Union[torch.Tensor, np.ndarray],
        joints: Union[torch.Tensor, np.ndarray],
        action_dim: int = 0,
        timesteps: Optional[List[int]] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Compute saliency across multiple timesteps in the action chunk.
        
        Useful to see how attention shifts over the predicted trajectory.
        """
        if timesteps is None:
            chunk_size = self.model.decoder.chunk_size
            timesteps = [0, chunk_size // 4, chunk_size // 2, 3 * chunk_size // 4, chunk_size - 1]
        
        temporal_saliency = {}
        for t in timesteps:
            result = self.compute_saliency(image, joints, action_dims=[action_dim], timestep=t)
            dim_name = self.action_names[action_dim] if action_dim < len(self.action_names) else f'dim_{action_dim}'
            temporal_saliency[t] = result.saliency_maps.get(dim_name, np.zeros(self.image_size))
        
        return temporal_saliency
    
    def visualize(
        self,
        result: ACTSaliencyResult,
        show_individual: bool = True,
        figsize: Tuple[int, int] = (15, 5),
        cmap: str = 'jet',
        alpha: float = 0.5,
        save_path: Optional[str] = None,
    ):
        """Visualize saliency results."""
        import matplotlib.pyplot as plt
        
        n_maps = len(result.saliency_maps) + 1 if show_individual else 1
        n_cols = min(4, n_maps)
        n_rows = (n_maps + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_maps == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        # Prepare image
        image = result.image
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Combined saliency
        ax = axes[0]
        ax.imshow(image)
        heatmap = cv2.resize(result.combined_saliency, (image.shape[1], image.shape[0]))
        ax.imshow(heatmap, cmap=cmap, alpha=alpha)
        ax.set_title(f'Combined (t={result.timestep})')
        ax.axis('off')
        
        # Individual action dims
        if show_individual:
            for idx, (name, saliency) in enumerate(result.saliency_maps.items()):
                if idx + 1 >= len(axes):
                    break
                ax = axes[idx + 1]
                ax.imshow(image)
                heatmap = cv2.resize(saliency, (image.shape[1], image.shape[0]))
                ax.imshow(heatmap, cmap=cmap, alpha=alpha)
                
                # Get action value for this dim
                dim_idx = list(result.saliency_maps.keys()).index(name)
                if dim_idx < result.predicted_actions.shape[-1]:
                    action_val = result.predicted_actions[result.timestep, dim_idx]
                    ax.set_title(f'{name}: {action_val:.3f}')
                else:
                    ax.set_title(name)
                ax.axis('off')
        
        # Hide unused axes
        for idx in range(n_maps, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        return fig


def load_act_model(
    checkpoint_path: str,
    model_type: str = 'standard',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **model_kwargs
) -> nn.Module:
    """
    Load a trained ACT model.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_type: 'standard' or 'modified'
        device: Device to load on
        **model_kwargs: Additional model arguments
        
    Returns:
        Loaded ACT model
    """
    if not HAS_ACT:
        raise ImportError("ACT models not available. Check ACT-modification path.")
    
    # Load checkpoint first to infer config
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Infer dimensions from checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    joint_dim = state_dict['encoder.joint_proj.weight'].shape[1]
    action_dim = state_dict['decoder.action_head.weight'].shape[0]
    
    print(f"Detected model config: joint_dim={joint_dim}, action_dim={action_dim}")
    
    # Default model config
    config = {
        'joint_dim': joint_dim,
        'action_dim': action_dim,
        'hidden_dim': 512,
        'latent_dim': 32,
        'n_encoder_layers': 4,
        'n_decoder_layers': 7,
        'n_heads': 8,
        'feedforward_dim': 3200,
        'chunk_size': 100,
        'n_cameras': 1,
        'dropout': 0.1,
    }
    config.update(model_kwargs)
    
    # Create model
    if model_type == 'standard':
        model = StandardACT(**config)
    else:
        model = ModifiedACT(**config)
    
    # Load checkpoint
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    print(f"Loaded {model_type} ACT from {checkpoint_path}")
    return model, config


def test_act_gradcam():
    """Test ACT-GradCAM with loaded model."""
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("Testing ACT-GradCAM with Real Model")
    print("=" * 60)
    
    # Load model
    checkpoint_path = "/home/aryannzzz/GRASP/ACT-modification/checkpoints_proper/standard/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    model, config = load_act_model(checkpoint_path, model_type='standard')
    
    # Action names for 4-DOF
    action_names = ['delta_x', 'delta_y', 'delta_z', 'gripper']
    
    # Create GradCAM
    gradcam = ACTGradCAM(
        model=model,
        target_layer='decoder.resnet.features.7',
        action_names=action_names,
        image_size=(128, 128)
    )
    
    # Create test inputs with correct dimensions
    test_image = np.random.rand(128, 128, 3).astype(np.float32)
    test_joints = np.random.rand(config['joint_dim']).astype(np.float32) * 0.1
    
    print(f"Input image shape: {test_image.shape}")
    print(f"Input joints shape: {test_joints.shape}")
    
    # Compute saliency
    result = gradcam.compute_saliency(
        image=test_image,
        joints=test_joints,
        action_dims=[0, 1, 2, 3],  # x, y, z, gripper
        timestep=0
    )
    
    print(f"\nResults:")
    print(f"  Predicted actions shape: {result.predicted_actions.shape}")
    print(f"  Saliency maps: {list(result.saliency_maps.keys())}")
    print(f"  First action: {result.predicted_actions[0]}")
    
    # Visualize
    output_dir = Path("/home/aryannzzz/GRASP/GradCAM/outputs/act_gradcam")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = gradcam.visualize(
        result,
        show_individual=True,
        save_path=str(output_dir / "test_act_saliency.png")
    )
    plt.close(fig)
    
    gradcam.remove_hooks()
    
    print("\n✓ ACT-GradCAM test passed!")
    return result


if __name__ == "__main__":
    test_act_gradcam()
