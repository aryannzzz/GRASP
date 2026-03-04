"""
Evaluate ACT model with scaled latent sampling to produce larger actions.
Tests if KL regularization was suppressing action magnitude.
"""

import torch
import numpy as np
import h5py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.standard_act import StandardACT
from models.modified_act import ModifiedACT
from envs.metaworld_wrapper import make_metaworld_env

def evaluate_with_scaling(model_type='standard', checkpoint_path=None, 
                         num_episodes=20, latent_scale=2.0):
    """
    Evaluate model with scaled latent sampling.
    latent_scale > 1.0 will produce more diverse/larger actions.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type.upper()} ACT with Latent Scale: {latent_scale}")
    print(f"{'='*60}\n")
    
    # Load normalization stats
    norm_path = Path(checkpoint_path).parent / 'norm_stats.npz'
    norm_data = np.load(norm_path)
    action_mean = norm_data['action_mean']
    action_std = norm_data['action_std']
    
    print(f"Action normalization:")
    print(f"  Mean: {action_mean}")
    print(f"  Std: {action_std}")
    
    # Create model
    if model_type == 'standard':
        model = StandardACT(
            obs_dim=39,
            action_dim=4,
            chunk_size=10,
            hidden_dim=512,
            dim_feedforward=3200,
            n_layer=8,
            n_head=8,
            latent_dim=32,
            dropout=0.1
        ).to(device)
    else:
        model = ModifiedACT(
            obs_dim=39,
            action_dim=4,
            chunk_size=10,
            hidden_dim=512,
            dim_feedforward=3200,
            n_layer=8,
            n_head=8,
            latent_dim=32,
            dropout=0.1
        ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create environment
    env = make_metaworld_env('shelf-place-v3', seed=42, camera_names=['corner3'])
    
    successes = []
    final_distances = []
    rewards = []
    
    for ep in range(num_episodes):
        obs_dict = env.reset()
        obs = obs_dict['observation']
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 500:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Sample latent with scaling
                z = torch.randn(1, model.latent_dim).to(device) * latent_scale
                
                # Get action sequence
                action_seq = model.decode_actions(obs_tensor, z)
                action_norm = action_seq[0, 0].cpu().numpy()
                
                # Denormalize
                action = action_norm * action_std + action_mean
                
                # MetaWorld will clip internally, but let's see what we're sending
                if step == 0:
                    print(f"\nEpisode {ep+1}, First action:")
                    print(f"  Normalized: {action_norm}")
                    print(f"  Denormalized: {action}")
            
            # Execute action
            obs_dict, reward, done, info = env.step(action)
            obs = obs_dict['observation']
            episode_reward += reward
            step += 1
        
        success = info.get('success', False)
        successes.append(success)
        
        # Get final distance to goal
        obj_pos = obs_dict['observation'][4:7]
        goal_pos = obs_dict['observation'][7:10]
        final_dist = np.linalg.norm(obj_pos - goal_pos)
        final_distances.append(final_dist)
        rewards.append(episode_reward)
        
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"Ep {ep+1:2d}: {status} | Dist: {final_dist:.4f}m | Reward: {episode_reward:.2f}")
    
    # Print summary
    success_rate = np.mean(successes) * 100
    print(f"\n{'='*60}")
    print(f"Results with Latent Scale {latent_scale}:")
    print(f"  Success Rate: {success_rate:.1f}% ({np.sum(successes)}/{num_episodes})")
    print(f"  Avg Final Distance: {np.mean(final_distances):.4f}m")
    print(f"  Avg Reward: {np.mean(rewards):.2f}")
    print(f"{'='*60}\n")
    
    env.close()
    return success_rate

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='standard', choices=['standard', 'modified'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--latent_scale', type=float, default=2.0,
                       help='Scale factor for latent sampling (>1.0 = larger actions)')
    args = parser.parse_args()
    
    evaluate_with_scaling(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        latent_scale=args.latent_scale
    )
