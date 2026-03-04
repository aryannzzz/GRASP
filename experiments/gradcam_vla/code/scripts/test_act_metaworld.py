#!/usr/bin/env python3
"""
Test ACT-GradCAM with Real MetaWorld Data

This script:
1. Loads a trained ACT model
2. Loads real demonstration images from MetaWorld
3. Computes GradCAM saliency maps
4. Visualizes where the model attends for each action dimension
"""

import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/home/aryannzzz/GRASP/ACT-modification")

import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2

from vla_gradcam.act_integration import ACTGradCAM, load_act_model


def load_metaworld_data(
    data_path: str = "/home/aryannzzz/GRASP/ACT-modification/data/diverse_demonstrations_with_images.hdf5",
    n_samples: int = 5
):
    """Load sample images and states from MetaWorld demonstrations."""
    print(f"Loading data from {data_path}")
    
    samples = []
    
    with h5py.File(data_path, 'r') as f:
        demo_keys = [k for k in f.keys() if k.startswith('demo_')]
        print(f"Found {len(demo_keys)} demonstrations")
        
        for i, demo_key in enumerate(demo_keys[:n_samples]):
            demo = f[demo_key]
            
            # Get image (first frame of episode)
            images = demo['images'][:]
            img = images[0]  # First frame
            
            # Get state
            states = demo['states'][:]
            state = states[0]  # First state
            
            # Get actions
            actions = demo['actions'][:]
            
            samples.append({
                'image': img,
                'state': state,
                'actions': actions,
                'episode': demo_key,
                'all_images': images,  # Keep all for video
                'all_states': states,
            })
            
            print(f"  {demo_key}: image={img.shape}, state={state.shape}, actions={actions.shape}")
    
    return samples


def test_with_metaworld():
    """Run full GradCAM analysis on MetaWorld data."""
    print("=" * 70)
    print("ACT-GradCAM Analysis on MetaWorld Data")
    print("=" * 70)
    
    # Load model
    checkpoint_path = "/home/aryannzzz/GRASP/ACT-modification/checkpoints_proper/standard/best_model.pth"
    model, config = load_act_model(checkpoint_path, model_type='standard')
    
    # Create GradCAM
    action_names = ['delta_x', 'delta_y', 'delta_z', 'gripper']
    gradcam = ACTGradCAM(
        model=model,
        target_layer='decoder.resnet.features.7',
        action_names=action_names,
        image_size=(128, 128)
    )
    
    # Load MetaWorld data
    samples = load_metaworld_data(n_samples=3)
    
    if not samples:
        print("No samples loaded! Creating synthetic test...")
        # Try with MetaWorld env directly
        try:
            import metaworld
            import metaworld.envs.mujoco.sawyer_xyz.v2 as sawyer_envs
            
            env = sawyer_envs.SawyerShelfPlaceEnvV2()
            env._partially_observable = False
            env._freeze_rand_vec = False
            env._set_task_called = True
            
            obs = env.reset()
            img = env.render(mode='rgb_array', width=128, height=128)
            
            samples = [{
                'image': img,
                'state': obs,
                'actions': np.zeros((100, 4)),
                'episode': 'live_env'
            }]
            env.close()
        except Exception as e:
            print(f"Could not create MetaWorld env: {e}")
            # Use random data as fallback
            samples = [{
                'image': np.random.rand(128, 128, 3).astype(np.float32),
                'state': np.random.rand(39).astype(np.float32),
                'actions': np.zeros((100, 4)),
                'episode': 'random'
            }]
    
    # Output directory
    output_dir = Path("/home/aryannzzz/GRASP/GradCAM/outputs/act_gradcam")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each sample
    for i, sample in enumerate(samples):
        print(f"\n{'='*50}")
        print(f"Analyzing Sample {i+1}: Episode {sample['episode']}")
        print(f"{'='*50}")
        
        image = sample['image']
        state = sample['state']
        
        # Ensure correct format
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if image.shape[:2] != (128, 128):
            image = cv2.resize(image, (128, 128))
        
        # Ensure state is correct dimension
        if len(state) != config['joint_dim']:
            print(f"  Adjusting state from {len(state)} to {config['joint_dim']}")
            if len(state) > config['joint_dim']:
                state = state[:config['joint_dim']]
            else:
                state = np.pad(state, (0, config['joint_dim'] - len(state)))
        
        # Compute saliency for all action dims
        result = gradcam.compute_saliency(
            image=image,
            joints=state.astype(np.float32),
            action_dims=[0, 1, 2, 3],
            timestep=0
        )
        
        print(f"  Predicted first action: {result.predicted_actions[0]}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        ax = axes[0, 0]
        ax.imshow(image)
        ax.set_title('Original Image')
        ax.axis('off')
        
        # Combined saliency
        ax = axes[0, 1]
        ax.imshow(image)
        heatmap = cv2.resize(result.combined_saliency, (image.shape[1], image.shape[0]))
        ax.imshow(heatmap, cmap='jet', alpha=0.5)
        ax.set_title('Combined Saliency')
        ax.axis('off')
        
        # Per-action saliency
        for j, (name, saliency) in enumerate(result.saliency_maps.items()):
            if j >= 4:
                break
            row = (j + 2) // 3
            col = (j + 2) % 3
            ax = axes[row, col]
            ax.imshow(image)
            heatmap = cv2.resize(saliency, (image.shape[1], image.shape[0]))
            ax.imshow(heatmap, cmap='jet', alpha=0.5)
            action_val = result.predicted_actions[0, j]
            ax.set_title(f'{name}: {action_val:.3f}')
            ax.axis('off')
        
        plt.suptitle(f'ACT-GradCAM Analysis - Episode: {sample["episode"]}', fontsize=14)
        plt.tight_layout()
        
        save_path = output_dir / f"metaworld_gradcam_ep{i}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
        
        # Also analyze different timesteps
        print(f"\n  Temporal saliency analysis (delta_x):")
        temporal = gradcam.compute_temporal_saliency(
            image=image,
            joints=state.astype(np.float32),
            action_dim=0,
            timesteps=[0, 25, 50, 75, 99]
        )
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for j, (t, saliency) in enumerate(temporal.items()):
            ax = axes[j]
            ax.imshow(image)
            heatmap = cv2.resize(saliency, (image.shape[1], image.shape[0]))
            ax.imshow(heatmap, cmap='jet', alpha=0.5)
            ax.set_title(f't={t}')
            ax.axis('off')
        
        plt.suptitle(f'Temporal Saliency (delta_x) - Episode: {sample["episode"]}')
        plt.tight_layout()
        
        save_path = output_dir / f"metaworld_temporal_ep{i}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved temporal: {save_path}")
        plt.close()
    
    gradcam.remove_hooks()
    
    print(f"\n{'='*70}")
    print("Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_with_metaworld()
