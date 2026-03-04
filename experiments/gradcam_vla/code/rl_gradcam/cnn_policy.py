"""
CNN Policy Network for RL-GradCAM

Designed specifically for GradCAM interpretability:
- Clear convolutional layers as GradCAM targets
- Separate Q-value heads for action-conditioned analysis
- Exposed intermediate activations for visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict


class ConvBlock(nn.Module):
    """Convolutional block with optional batch norm."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batchnorm: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=False)  # IMPORTANT: inplace=False for GradCAM compatibility
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class RLPolicyNetwork(nn.Module):
    """
    CNN-based policy network for visual RL environments.
    
    Architecture designed for GradCAM interpretability:
    - Multiple conv layers with increasing channels
    - Clear target layers for attention visualization
    - Outputs Q-values for all actions (DQN-style)
    
    For a 64x64 input (4x4 grid with 16px tiles):
    - Conv1: 64x64 -> 64x64 (same size, 32 channels)
    - Pool1: 64x64 -> 32x32
    - Conv2: 32x32 -> 32x32 (64 channels)
    - Pool2: 32x32 -> 16x16
    - Conv3: 16x16 -> 16x16 (128 channels) <- Main GradCAM target
    - Pool3: 16x16 -> 8x8
    - Flatten + FC layers -> Q-values
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        input_size: Tuple[int, int] = (64, 64),
        n_actions: int = 4,
        hidden_dims: List[int] = [32, 64, 128],
        fc_dim: int = 256,
        use_batchnorm: bool = False
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_size = input_size
        self.n_actions = n_actions
        
        # Convolutional feature extraction
        self.conv1 = ConvBlock(input_channels, hidden_dims[0], use_batchnorm=use_batchnorm)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = ConvBlock(hidden_dims[0], hidden_dims[1], use_batchnorm=use_batchnorm)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = ConvBlock(hidden_dims[1], hidden_dims[2], use_batchnorm=use_batchnorm)
        self.pool3 = nn.MaxPool2d(2)
        
        # Calculate flattened size
        # Input 64x64 -> after 3 pools of 2x2 -> 8x8
        self.feature_size = hidden_dims[2] * (input_size[0] // 8) * (input_size[1] // 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, fc_dim)
        self.fc2 = nn.Linear(fc_dim, n_actions)
        
        # Store references to target layers for GradCAM
        # The last conv layer typically gives best spatial resolution
        self.gradcam_target_layer = self.conv3.conv
        
        # For storing intermediate activations (useful for analysis)
        self._features = {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning Q-values for all actions.
        
        Args:
            x: (B, C, H, W) visual observation
            
        Returns:
            q_values: (B, n_actions) Q-value for each action
        """
        # Convolutional layers
        x = self.conv1(x)
        x = self.pool1(x)
        self._features['conv1'] = x
        
        x = self.conv2(x)
        x = self.pool2(x)
        self._features['conv2'] = x
        
        x = self.conv3(x)
        self._features['conv3_pre_pool'] = x  # Before pooling - best for GradCAM
        x = self.pool3(x)
        self._features['conv3'] = x
        
        # Flatten and FC layers
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        self._features['fc1'] = x
        
        q_values = self.fc2(x)
        
        return q_values
    
    def get_action(self, x: torch.Tensor, epsilon: float = 0.0) -> Tuple[int, torch.Tensor]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            x: (B, C, H, W) or (C, H, W) visual observation
            epsilon: Exploration probability
            
        Returns:
            action: Selected action index
            q_values: Q-values for all actions
        """
        # Add batch dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.forward(x)
        
        # Epsilon-greedy
        if torch.rand(1).item() < epsilon:
            action = torch.randint(0, self.n_actions, (1,)).item()
        else:
            action = q_values.argmax(dim=1).item()
        
        return action, q_values
    
    def get_features(self, layer_name: str) -> Optional[torch.Tensor]:
        """Get intermediate features from last forward pass."""
        return self._features.get(layer_name)
    
    def get_gradcam_target_layers(self) -> List[nn.Module]:
        """Return list of layers suitable for GradCAM."""
        return [
            self.conv3.conv,  # Best resolution
            self.conv2.conv,  # Higher level features
            self.conv1.conv,  # Lower level features
        ]


class DuelingRLPolicyNetwork(nn.Module):
    """
    Dueling DQN architecture with separate value and advantage streams.
    
    Advantage for interpretability:
    - Value stream: "How good is this state overall?"
    - Advantage stream: "How much better is each action than average?"
    
    This separation can help interpret action-specific attention.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        input_size: Tuple[int, int] = (64, 64),
        n_actions: int = 4,
        hidden_dims: List[int] = [32, 64, 128],
        fc_dim: int = 256
    ):
        super().__init__()
        
        self.n_actions = n_actions
        
        # Shared convolutional backbone
        self.conv1 = ConvBlock(input_channels, hidden_dims[0])
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = ConvBlock(hidden_dims[0], hidden_dims[1])
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = ConvBlock(hidden_dims[1], hidden_dims[2])
        self.pool3 = nn.MaxPool2d(2)
        
        # Calculate flattened size
        self.feature_size = hidden_dims[2] * (input_size[0] // 8) * (input_size[1] // 8)
        
        # Value stream (state value)
        self.value_fc1 = nn.Linear(self.feature_size, fc_dim // 2)
        self.value_fc2 = nn.Linear(fc_dim // 2, 1)
        
        # Advantage stream (action advantages)
        self.advantage_fc1 = nn.Linear(self.feature_size, fc_dim // 2)
        self.advantage_fc2 = nn.Linear(fc_dim // 2, n_actions)
        
        # GradCAM target
        self.gradcam_target_layer = self.conv3.conv
        
        self._features = {}
        self._value = None
        self._advantage = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dueling architecture."""
        # Shared convolutions
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        self._features['conv3_pre_pool'] = x
        x = self.pool3(x)
        
        # Flatten
        x = x.flatten(1)
        
        # Value stream
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)
        self._value = value
        
        # Advantage stream
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)
        self._advantage = advantage
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_value_and_advantage(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get value and advantage from last forward pass."""
        return self._value, self._advantage
    
    def get_gradcam_target_layers(self) -> List[nn.Module]:
        """Return layers suitable for GradCAM."""
        return [self.conv3.conv, self.conv2.conv]


def create_policy_network(
    env,
    architecture: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Factory function to create policy network matching environment.
    
    Args:
        env: Visual environment (must have observation_space)
        architecture: "standard" or "dueling"
        **kwargs: Additional network arguments
        
    Returns:
        Policy network
    """
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    # Extract dimensions
    input_channels = obs_shape[0]
    input_size = (obs_shape[1], obs_shape[2])
    
    if architecture == "standard":
        return RLPolicyNetwork(
            input_channels=input_channels,
            input_size=input_size,
            n_actions=n_actions,
            **kwargs
        )
    elif architecture == "dueling":
        return DuelingRLPolicyNetwork(
            input_channels=input_channels,
            input_size=input_size,
            n_actions=n_actions,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# Quick test
if __name__ == "__main__":
    print("Testing RLPolicyNetwork...")
    
    # Create network
    net = RLPolicyNetwork(
        input_channels=3,
        input_size=(64, 64),
        n_actions=4
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 64, 64)
    q_values = net(x)
    print(f"Input shape: {x.shape}")
    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values}")
    
    # Test action selection
    action, qv = net.get_action(x[0], epsilon=0.0)
    print(f"\nSelected action: {action}")
    
    # Check features
    print(f"\nIntermediate features:")
    for name, feat in net._features.items():
        if feat is not None:
            print(f"  {name}: {feat.shape}")
    
    # Check GradCAM target
    print(f"\nGradCAM target layer: {net.gradcam_target_layer}")
    
    # Test gradient flow
    loss = q_values.mean()
    loss.backward()
    print(f"\nGradients flowing: ✓")
    
    print("\n✓ RLPolicyNetwork test passed!")
