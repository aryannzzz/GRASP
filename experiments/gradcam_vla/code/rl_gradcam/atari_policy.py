"""
Atari CNN Policy Network

Nature DQN architecture for Atari games:
- 3 convolutional layers
- 2 fully connected layers
- Designed for 84x84 stacked frame input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AtariCNN(nn.Module):
    """
    Nature DQN CNN architecture for Atari.
    
    Input: (batch, 4, 84, 84) - 4 stacked grayscale frames
    Output: (batch, n_actions) - Q-values for each action
    
    Architecture designed for GradCAM:
    - Conv layers use inplace=False for gradient hooks
    - Named layers for easy target selection
    """
    
    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()
        
        self.n_actions = n_actions
        
        # Convolutional layers (Nature DQN architecture)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions: 84 -> 20 -> 9 -> 7
        conv_out_size = 64 * 7 * 7
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
        # GradCAM target layer (last conv layer - highest spatial resolution with features)
        self.gradcam_target_layer = self.conv3
        
        # Alternative target layers for different granularity
        self.gradcam_layers = {
            'conv1': self.conv1,  # 20x20, coarse features
            'conv2': self.conv2,  # 9x9, mid-level features
            'conv3': self.conv3,  # 7x7, fine features (default)
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))  # (B, 32, 20, 20)
        x = F.relu(self.conv2(x))  # (B, 64, 9, 9)
        x = F.relu(self.conv3(x))  # (B, 64, 7, 7)
        
        # Flatten and FC layers
        x = x.reshape(x.size(0), -1)  # (B, 64*7*7)
        x = F.relu(self.fc1(x))       # (B, 512)
        x = self.fc2(x)               # (B, n_actions)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract convolutional features (for visualization)."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class DuelingAtariCNN(nn.Module):
    """
    Dueling DQN architecture for Atari.
    
    Separates value and advantage streams for better learning.
    """
    
    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()
        
        self.n_actions = n_actions
        
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        conv_out_size = 64 * 7 * 7
        
        # Value stream
        self.value_fc = nn.Linear(conv_out_size, 512)
        self.value = nn.Linear(512, 1)
        
        # Advantage stream
        self.advantage_fc = nn.Linear(conv_out_size, 512)
        self.advantage = nn.Linear(512, n_actions)
        
        self.gradcam_target_layer = self.conv3
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared features
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        
        # Value stream
        v = F.relu(self.value_fc(x))
        v = self.value(v)
        
        # Advantage stream
        a = F.relu(self.advantage_fc(x))
        a = self.advantage(a)
        
        # Combine: Q = V + (A - mean(A))
        q = v + (a - a.mean(dim=1, keepdim=True))
        
        return q


def create_atari_policy(env, dueling: bool = False) -> nn.Module:
    """Create CNN policy for Atari environment."""
    n_actions = env.action_space.n
    in_channels = env.observation_space.shape[0]  # Usually 4 (stacked frames)
    
    if dueling:
        return DuelingAtariCNN(n_actions, in_channels)
    else:
        return AtariCNN(n_actions, in_channels)
