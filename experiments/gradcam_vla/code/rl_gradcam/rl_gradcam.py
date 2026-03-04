"""
RL-GradCAM: Action-Conditioned Gradient-weighted Class Activation Mapping for RL

The key novelty: Instead of explaining "why this class?", we explain "why this action?"

For each action the agent can take, we generate a separate attention heatmap showing
which parts of the observation influenced that action's Q-value. This enables:

1. **Action Comparison**: See different attention for different actions
2. **Decision Understanding**: Why did agent choose action A over action B?
3. **Failure Analysis**: Did agent attend to wrong features?
4. **Risk Assessment**: Does agent "see" dangerous tiles?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import cv2


class RLGradCAM:
    """
    GradCAM adapted for Reinforcement Learning policy interpretation.
    
    Key differences from standard GradCAM:
    1. Target is Q-value for specific action, not class probability
    2. Generates CAM for ALL actions simultaneously for comparison
    3. Tracks how attention changes with different actions
    4. Designed for sequential decision-making context
    
    Usage:
        gradcam = RLGradCAM(policy_network, target_layer)
        
        # Get attention for single action
        cam = gradcam.compute_cam(observation, action=2)
        
        # Get attention for ALL actions (for comparison)
        all_cams = gradcam.compute_all_action_cams(observation)
        
        # Visualize
        viz = gradcam.visualize(observation, action=2)
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: str = 'cpu'
    ):
        """
        Args:
            model: Policy network (must output Q-values)
            target_layer: Conv layer to extract activations from
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.target_layer = target_layer
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Storage for activations and gradients
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def _preprocess_input(self, observation: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert observation to proper tensor format."""
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        
        if observation.dim() == 3:
            observation = observation.unsqueeze(0)  # Add batch dim
        
        return observation.to(self.device)
    
    def compute_cam(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        action: int,
        normalize: bool = True,
        relu: bool = True
    ) -> np.ndarray:
        """
        Compute GradCAM heatmap for a specific action.
        
        The CAM shows which spatial locations in the observation
        most influenced the Q-value for the given action.
        
        Args:
            observation: (C, H, W) or (B, C, H, W) visual observation
            action: Action index to explain
            normalize: Scale CAM to [0, 1]
            relu: Apply ReLU (only positive contributions)
            
        Returns:
            cam: (H, W) heatmap, same spatial size as conv feature map
        """
        # Prepare input
        obs_tensor = self._preprocess_input(observation)
        obs_tensor.requires_grad_(True)
        
        # Forward pass
        self.model.zero_grad()
        q_values = self.model(obs_tensor)
        
        # Backward pass for specific action's Q-value
        target_q = q_values[0, action]
        target_q.backward()
        
        # Get activations and gradients
        activations = self.activations  # (B, C, H, W)
        gradients = self.gradients      # (B, C, H, W)
        
        # Compute weights: global average pooling of gradients
        # This gives importance of each channel for the action
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Apply ReLU (focus on positive influences)
        if relu:
            cam = F.relu(cam)
        
        # Remove batch and channel dims
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        if normalize and cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam
    
    def compute_all_action_cams(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        action_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute GradCAM for ALL actions simultaneously.
        
        This is the key novelty for RL interpretability:
        Compare attention patterns across different actions to understand
        why the agent prefers one action over another.
        
        Args:
            observation: Visual observation
            action_names: Optional names for actions (e.g., ['Left', 'Down', 'Right', 'Up'])
            
        Returns:
            Dictionary mapping action (name or index) to CAM heatmap
        """
        obs_tensor = self._preprocess_input(observation)
        
        # Get Q-values to know number of actions
        with torch.no_grad():
            q_values = self.model(obs_tensor)
        n_actions = q_values.shape[1]
        
        # Default action names
        if action_names is None:
            action_names = [f"action_{i}" for i in range(n_actions)]
        
        # Compute CAM for each action
        cams = {}
        for action_idx in range(n_actions):
            cam = self.compute_cam(observation, action_idx)
            action_key = action_names[action_idx] if action_idx < len(action_names) else f"action_{action_idx}"
            cams[action_key] = cam
        
        return cams
    
    def compute_action_contrast_cam(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        action_a: int,
        action_b: int
    ) -> np.ndarray:
        """
        Compute CONTRAST attention: what differentiates action A from action B?
        
        This shows regions that specifically favor action A over action B,
        helping understand the decision boundary between actions.
        
        Args:
            observation: Visual observation
            action_a: First action
            action_b: Second action
            
        Returns:
            Contrast CAM (positive = favors A, negative = favors B)
        """
        cam_a = self.compute_cam(observation, action_a, normalize=False, relu=False)
        cam_b = self.compute_cam(observation, action_b, normalize=False, relu=False)
        
        # Contrast: regions that differentiate the two actions
        contrast = cam_a - cam_b
        
        # Normalize to [-1, 1]
        max_abs = max(abs(contrast.min()), abs(contrast.max()))
        if max_abs > 0:
            contrast = contrast / max_abs
        
        return contrast
    
    def get_q_values(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Get Q-values for all actions (useful for analysis)."""
        obs_tensor = self._preprocess_input(observation)
        with torch.no_grad():
            q_values = self.model(obs_tensor)
        return q_values.cpu().numpy()[0]
    
    def resize_cam_to_observation(
        self,
        cam: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Resize CAM to match observation size."""
        return cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR)
    
    def create_heatmap_overlay(
        self,
        observation: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Create visualization overlay of CAM on observation.
        
        Args:
            observation: (C, H, W) normalized observation
            cam: (h, w) CAM heatmap
            alpha: Blending factor
            colormap: OpenCV colormap
            
        Returns:
            overlay: (H, W, 3) RGB overlay image
        """
        # Convert observation to (H, W, C) uint8
        if observation.ndim == 3 and observation.shape[0] == 3:
            obs_img = observation.transpose(1, 2, 0)
        else:
            obs_img = observation
        
        if obs_img.max() <= 1.0:
            obs_img = (obs_img * 255).astype(np.uint8)
        else:
            obs_img = obs_img.astype(np.uint8)
        
        # Resize CAM to match observation
        cam_resized = self.resize_cam_to_observation(
            cam, (obs_img.shape[1], obs_img.shape[0])
        )
        
        # Create colored heatmap
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8),
            colormap
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend with observation
        overlay = cv2.addWeighted(obs_img, 1 - alpha, heatmap, alpha, 0)
        
        return overlay
    
    def visualize_all_actions(
        self,
        observation: np.ndarray,
        action_names: Optional[List[str]] = None,
        chosen_action: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Create visualization for all actions.
        
        Args:
            observation: Visual observation
            action_names: Names for each action
            chosen_action: Highlight which action was taken
            
        Returns:
            Dictionary with overlays for each action + comparison info
        """
        # Compute all CAMs
        cams = self.compute_all_action_cams(observation, action_names)
        q_values = self.get_q_values(observation)
        
        # Create overlays
        visualizations = {}
        for action_key, cam in cams.items():
            overlay = self.create_heatmap_overlay(observation, cam)
            visualizations[action_key] = {
                'cam': cam,
                'overlay': overlay,
            }
        
        # Add Q-values
        if action_names:
            for i, name in enumerate(action_names):
                if name in visualizations:
                    visualizations[name]['q_value'] = q_values[i]
        
        # Mark chosen action
        if chosen_action is not None and action_names:
            chosen_name = action_names[chosen_action]
            visualizations[chosen_name]['chosen'] = True
        
        return visualizations


class ActionAttentionAnalyzer:
    """
    Higher-level analysis of action attention patterns.
    
    Provides insights like:
    - Which action has most focused attention?
    - Do different actions attend to different regions?
    - Does attention correlate with Q-values?
    """
    
    def __init__(self, gradcam: RLGradCAM):
        self.gradcam = gradcam
    
    def compute_attention_entropy(self, cam: np.ndarray) -> float:
        """
        Compute entropy of attention distribution.
        Lower entropy = more focused attention.
        """
        # Normalize to probability distribution
        cam_flat = cam.flatten()
        cam_flat = cam_flat / (cam_flat.sum() + 1e-8)
        
        # Compute entropy
        entropy = -np.sum(cam_flat * np.log(cam_flat + 1e-8))
        return entropy
    
    def compute_attention_focus(self, cam: np.ndarray) -> Dict[str, float]:
        """
        Compute attention focus metrics.
        """
        return {
            'entropy': self.compute_attention_entropy(cam),
            'max_activation': float(cam.max()),
            'mean_activation': float(cam.mean()),
            'std_activation': float(cam.std()),
            'sparsity': float((cam > 0.5).sum() / cam.size),  # Fraction of "hot" pixels
        }
    
    def compare_action_attention(
        self,
        observation: np.ndarray,
        action_names: List[str]
    ) -> Dict[str, Dict]:
        """
        Compare attention patterns across all actions.
        """
        cams = self.gradcam.compute_all_action_cams(observation, action_names)
        q_values = self.gradcam.get_q_values(observation)
        
        comparison = {}
        for i, (action_name, cam) in enumerate(cams.items()):
            comparison[action_name] = {
                'focus_metrics': self.compute_attention_focus(cam),
                'q_value': q_values[i],
            }
        
        # Add overall analysis
        entropies = [comparison[name]['focus_metrics']['entropy'] for name in action_names]
        comparison['_analysis'] = {
            'most_focused_action': action_names[np.argmin(entropies)],
            'least_focused_action': action_names[np.argmax(entropies)],
            'best_action': action_names[np.argmax(q_values)],
            'q_value_range': float(q_values.max() - q_values.min()),
        }
        
        return comparison
    
    def compute_attention_overlap(
        self,
        cam_a: np.ndarray,
        cam_b: np.ndarray
    ) -> float:
        """
        Compute overlap between two attention maps.
        High overlap = actions attend to similar regions.
        """
        # Normalize
        cam_a_norm = cam_a / (cam_a.sum() + 1e-8)
        cam_b_norm = cam_b / (cam_b.sum() + 1e-8)
        
        # Compute overlap (intersection over union style)
        intersection = np.minimum(cam_a_norm, cam_b_norm).sum()
        union = np.maximum(cam_a_norm, cam_b_norm).sum()
        
        return intersection / (union + 1e-8)


# Quick test
if __name__ == "__main__":
    from cnn_policy import RLPolicyNetwork
    
    print("Testing RLGradCAM...")
    
    # Create network
    net = RLPolicyNetwork(
        input_channels=3,
        input_size=(64, 64),
        n_actions=4
    )
    
    # Create GradCAM
    gradcam = RLGradCAM(
        model=net,
        target_layer=net.gradcam_target_layer,
        device='cpu'
    )
    
    # Test with random observation
    obs = np.random.rand(3, 64, 64).astype(np.float32)
    
    # Compute single action CAM
    cam = gradcam.compute_cam(obs, action=2)
    print(f"Single action CAM shape: {cam.shape}")
    print(f"CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
    
    # Compute all action CAMs
    action_names = ['Left', 'Down', 'Right', 'Up']
    all_cams = gradcam.compute_all_action_cams(obs, action_names)
    print(f"\nAll action CAMs:")
    for name, cam in all_cams.items():
        print(f"  {name}: shape={cam.shape}, range=[{cam.min():.3f}, {cam.max():.3f}]")
    
    # Compute contrast CAM
    contrast = gradcam.compute_action_contrast_cam(obs, action_a=0, action_b=2)
    print(f"\nContrast CAM (Left vs Right): range=[{contrast.min():.3f}, {contrast.max():.3f}]")
    
    # Get Q-values
    q_values = gradcam.get_q_values(obs)
    print(f"\nQ-values: {q_values}")
    
    # Test analyzer
    analyzer = ActionAttentionAnalyzer(gradcam)
    comparison = analyzer.compare_action_attention(obs, action_names)
    print(f"\nAttention analysis:")
    print(f"  Most focused: {comparison['_analysis']['most_focused_action']}")
    print(f"  Best action: {comparison['_analysis']['best_action']}")
    
    print("\n✓ RLGradCAM test passed!")
