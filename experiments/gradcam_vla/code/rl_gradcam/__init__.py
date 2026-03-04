"""
RL-GradCAM: Action-Conditioned Gradient-based Attention Visualization for RL

This package provides tools for understanding what visual features
influence RL agent decisions. Unlike standard GradCAM for classification,
RL-GradCAM focuses on explaining action selection.

Key Components:
- VisualFrozenLakeWrapper: Converts discrete RL states to visual observations
- RLPolicyNetwork: CNN policy designed for GradCAM interpretability  
- RLGradCAM: Action-conditioned attention visualization
- ActionAttentionAnalyzer: Quantitative attention analysis

Usage:
    from rl_gradcam import (
        make_visual_frozen_lake,
        create_policy_network,
        RLGradCAM,
        ActionAttentionAnalyzer
    )
    
    # Create visual environment
    env = make_visual_frozen_lake(is_slippery=False)
    
    # Create policy network
    policy = create_policy_network(env)
    
    # Create GradCAM analyzer
    gradcam = RLGradCAM(policy, policy.gradcam_target_layer)
    
    # Get attention for all actions
    obs, _ = env.reset()
    cams = gradcam.compute_all_action_cams(obs, env.ACTION_NAMES)
"""

from .visual_frozen_lake import (
    VisualFrozenLakeWrapper,
    make_visual_frozen_lake
)

from .cnn_policy import (
    RLPolicyNetwork,
    DuelingRLPolicyNetwork,
    create_policy_network
)

from .rl_gradcam import (
    RLGradCAM,
    ActionAttentionAnalyzer
)

from .visualizer import (
    RLGradCAMVisualizer
)

__all__ = [
    'VisualFrozenLakeWrapper',
    'make_visual_frozen_lake',
    'RLPolicyNetwork',
    'DuelingRLPolicyNetwork', 
    'create_policy_network',
    'RLGradCAM',
    'ActionAttentionAnalyzer',
    'RLGradCAMVisualizer',
]

__version__ = '0.1.0'
