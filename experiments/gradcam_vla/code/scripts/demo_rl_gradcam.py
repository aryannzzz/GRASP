#!/usr/bin/env python3
"""
RL-GradCAM Demo: Action-Conditioned Attention Visualization

This demo shows the core novelty: visualizing what the agent "looks at"
for EACH possible action, not just the chosen one.

Key insight: By comparing attention across actions, we can understand
WHY the agent prefers one action over another.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_gradcam import (
    make_visual_frozen_lake,
    create_policy_network,
    RLGradCAM,
    ActionAttentionAnalyzer
)
from rl_gradcam.visualizer import RLGradCAMVisualizer


def demo_action_conditioned_attention():
    """
    Demonstrate the key novelty: action-conditioned attention.
    
    Even with a random (untrained) policy, we can see:
    1. GradCAM works and produces spatial heatmaps
    2. Different actions produce different attention patterns
    3. We can compare and analyze these patterns
    """
    print("=" * 60)
    print("RL-GradCAM Demo: Action-Conditioned Attention")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating visual Frozen Lake environment...")
    env = make_visual_frozen_lake(
        map_name="4x4",
        is_slippery=False,  # Deterministic for easier interpretation
        tile_size=16,
        highlight_danger=True
    )
    
    print(f"   Grid: {env.grid_size}x{env.grid_size}")
    print(f"   Image size: {env.img_height}x{env.img_width}")
    print(f"   Actions: {env.ACTION_NAMES}")
    
    # Get grid info
    grid_info = env.get_grid_info()
    print(f"   Goal position: {grid_info['goal_position']}")
    print(f"   Holes: {grid_info['hole_positions']}")
    
    # Create policy network
    print("\n2. Creating CNN policy network...")
    policy = create_policy_network(env, architecture="standard")
    print(f"   Input: ({policy.input_channels}, {policy.input_size[0]}, {policy.input_size[1]})")
    print(f"   Output: {policy.n_actions} Q-values")
    print(f"   GradCAM target: {policy.gradcam_target_layer}")
    
    # Create GradCAM
    print("\n3. Creating RL-GradCAM analyzer...")
    gradcam = RLGradCAM(
        model=policy,
        target_layer=policy.gradcam_target_layer,
        device='cpu'
    )
    
    # Create visualizer
    visualizer = RLGradCAMVisualizer(action_names=env.ACTION_NAMES)
    
    # Reset environment and get observation
    print("\n4. Getting observation from environment...")
    obs, info = env.reset(seed=42)
    print(f"   State index: {info['state_index']}")
    print(f"   Agent position: {info['agent_position']}")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Compute attention for ALL actions
    print("\n5. Computing action-conditioned attention...")
    print("   (This is the KEY NOVELTY: separate attention per action)")
    
    all_cams = gradcam.compute_all_action_cams(obs, env.ACTION_NAMES)
    q_values = gradcam.get_q_values(obs)
    
    print("\n   Results:")
    for i, action_name in enumerate(env.ACTION_NAMES):
        cam = all_cams[action_name]
        print(f"   {action_name:6s}: CAM shape={cam.shape}, "
              f"range=[{cam.min():.3f}, {cam.max():.3f}], "
              f"Q={q_values[i]:.4f}")
    
    # Analyze attention patterns
    print("\n6. Analyzing attention patterns...")
    analyzer = ActionAttentionAnalyzer(gradcam)
    comparison = analyzer.compare_action_attention(obs, env.ACTION_NAMES)
    
    print(f"\n   Attention Analysis:")
    print(f"   - Most focused action: {comparison['_analysis']['most_focused_action']}")
    print(f"   - Least focused action: {comparison['_analysis']['least_focused_action']}")
    print(f"   - Best Q-value action: {comparison['_analysis']['best_action']}")
    
    for action_name in env.ACTION_NAMES:
        metrics = comparison[action_name]['focus_metrics']
        print(f"\n   {action_name}:")
        print(f"      Entropy: {metrics['entropy']:.3f} (lower = more focused)")
        print(f"      Max activation: {metrics['max_activation']:.3f}")
        print(f"      Sparsity: {metrics['sparsity']:.3f}")
    
    # Compute action contrast
    print("\n7. Computing action contrasts...")
    print("   (What differentiates one action from another?)")
    
    # Compare Left vs Right
    contrast_lr = gradcam.compute_action_contrast_cam(obs, action_a=0, action_b=2)
    print(f"   Left vs Right contrast range: [{contrast_lr.min():.3f}, {contrast_lr.max():.3f}]")
    
    # Compare Down vs Up
    contrast_du = gradcam.compute_action_contrast_cam(obs, action_a=1, action_b=3)
    print(f"   Down vs Up contrast range: [{contrast_du.min():.3f}, {contrast_du.max():.3f}]")
    
    # Create visualizations
    print("\n8. Creating visualizations...")
    
    # Determine best action (greedy)
    chosen_action = np.argmax(q_values)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'demo')
    os.makedirs(output_dir, exist_ok=True)
    
    # All actions comparison
    fig1 = visualizer.plot_all_actions_comparison(
        observation=obs,
        cams=all_cams,
        q_values=q_values,
        chosen_action=chosen_action,
        title="Action Attention Comparison (Random Policy)"
    )
    save_path1 = os.path.join(output_dir, 'action_comparison.png')
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"   Saved: {save_path1}")
    plt.close(fig1)
    
    # Decision explanation
    fig2 = visualizer.plot_decision_explanation(
        observation=obs,
        cams=all_cams,
        q_values=q_values,
        chosen_action=chosen_action
    )
    save_path2 = os.path.join(output_dir, 'decision_explanation.png')
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"   Saved: {save_path2}")
    plt.close(fig2)
    
    # Action contrast
    fig3 = visualizer.plot_action_contrast(
        observation=obs,
        contrast_cam=contrast_lr,
        action_a='Left',
        action_b='Right'
    )
    save_path3 = os.path.join(output_dir, 'action_contrast_lr.png')
    fig3.savefig(save_path3, dpi=150, bbox_inches='tight')
    print(f"   Saved: {save_path3}")
    plt.close(fig3)
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    
    print("\n📊 KEY OBSERVATIONS:")
    print("-" * 40)
    print("1. GradCAM produces spatial attention for each action")
    print("2. Different actions have DIFFERENT attention patterns")
    print("3. We can quantify and compare attention (entropy, focus)")
    print("4. Contrast maps show action differentiation")
    print("\n💡 INTERPRETATION:")
    print("-" * 40)
    print("• Even with random weights, gradients flow correctly")
    print("• After training, attention should focus on:")
    print("  - Holes (for avoidance)")
    print("  - Goal (for navigation)")
    print("  - Safe paths (for route planning)")
    print("\n🚀 NEXT STEPS:")
    print("-" * 40)
    print("1. Train the policy (DQN)")
    print("2. Compare attention before vs after training")
    print("3. Analyze failure cases")
    print("4. Track attention evolution over training")
    
    return {
        'env': env,
        'policy': policy,
        'gradcam': gradcam,
        'analyzer': analyzer,
        'visualizer': visualizer,
        'observation': obs,
        'cams': all_cams,
        'q_values': q_values
    }


def demo_multiple_states():
    """
    Show attention across multiple states to see how it changes.
    """
    print("\n" + "=" * 60)
    print("Multi-State Attention Analysis")
    print("=" * 60)
    
    # Create environment and policy
    env = make_visual_frozen_lake(map_name="4x4", is_slippery=False, tile_size=16)
    policy = create_policy_network(env)
    gradcam = RLGradCAM(policy, policy.gradcam_target_layer)
    visualizer = RLGradCAMVisualizer(action_names=env.ACTION_NAMES)
    
    # Sample several states
    test_states = [0, 1, 2, 4, 6, 9, 10, 14]  # Various positions on grid
    
    print("\nAnalyzing attention for different agent positions...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, state in enumerate(test_states):
        # Get visual observation for this state
        obs = env.get_state_image(state)
        
        # Get best action's attention
        q_values = gradcam.get_q_values(obs)
        best_action = np.argmax(q_values)
        cam = gradcam.compute_cam(obs, best_action)
        
        # Create overlay
        overlay = gradcam.create_heatmap_overlay(obs, cam)
        
        # Plot
        ax = axes[idx]
        ax.imshow(overlay)
        row, col = state // 4, state % 4
        ax.set_title(f'State {state} ({row},{col})\n'
                    f'Best: {env.ACTION_NAMES[best_action]}',
                    fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Attention Maps at Different States (Random Policy)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'demo')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'multi_state_attention.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    print("\n✓ Multi-state analysis complete")


if __name__ == "__main__":
    # Run main demo
    results = demo_action_conditioned_attention()
    
    # Run multi-state demo
    demo_multiple_states()
    
    print("\n" + "=" * 60)
    print("All demos completed successfully! 🎉")
    print("Check outputs/demo/ for visualizations")
    print("=" * 60)
