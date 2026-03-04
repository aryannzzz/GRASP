"""
Multi-task evaluation to test if model is learning properly.
Tests on increasingly complex tasks: reach → push → pick-place → shelf-place
"""

import os
import sys
import torch
import numpy as np
import metaworld
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.standard_act import StandardACT
from models.modified_act import ModifiedACT


def evaluate_on_task(model, norm_stats, task_name='reach-v3', num_episodes=10, device='cuda'):
    """
    Evaluate model on a specific MetaWorld task.
    """
    print(f"\n{'='*70}")
    print(f"Evaluating on: {task_name.upper()}")
    print(f"{'='*70}")
    
    # Create environment
    ml1 = metaworld.ML1(task_name, seed=42)
    env = ml1.train_classes[task_name]()
    task = ml1.train_tasks[0]
    env.set_task(task)
    
    # Get normalization stats
    action_mean = norm_stats['action_mean']
    action_std = norm_stats['action_std']
    
    successes = []
    distances = []
    rewards = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()  # MetaWorld v3 returns (obs, info)
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 500:
            # Prepare state
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Create dummy image (model trained without images working)
            dummy_image = torch.zeros(1, 3, 480, 480).to(device)
            images_dict = {'corner2': dummy_image}
            
            # Get action from model (deterministic: z=0)
            with torch.no_grad():
                pred_actions = model(images_dict, state_tensor, training=False)
                action_norm = pred_actions[0, 0].cpu().numpy()
            
            # Denormalize action
            action = action_norm * action_std + action_mean
            
            # Execute action (MetaWorld v3 returns 5 values)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
        
        # Get final results
        success = info.get('success', False)
        successes.append(success)
        rewards.append(episode_reward)
        
        # Calculate distance to goal
        obj_pos = obs[4:7]
        goal_pos = obs[7:10] if len(obs) > 7 else obs[4:7]
        distance = np.linalg.norm(obj_pos - goal_pos)
        distances.append(distance)
        
        status = "✓" if success else "✗"
        print(f"  Ep {ep+1:2d}/{num_episodes}: {status} | Dist: {distance:.3f}m | Reward: {episode_reward:.2f}")
    
    # Summary
    success_rate = np.mean(successes) * 100
    avg_distance = np.mean(distances)
    avg_reward = np.mean(rewards)
    
    print(f"\n{'-'*70}")
    print(f"Results for {task_name}:")
    print(f"  Success Rate: {success_rate:.1f}% ({int(np.sum(successes))}/{num_episodes})")
    print(f"  Avg Distance: {avg_distance:.4f}m")
    print(f"  Avg Reward:   {avg_reward:.2f}")
    print(f"{'-'*70}")
    
    return {
        'task': task_name,
        'success_rate': success_rate,
        'avg_distance': avg_distance,
        'avg_reward': avg_reward,
        'num_episodes': num_episodes
    }


def main(checkpoint_path, model_type='standard', episodes_per_task=10):
    """
    Evaluate model on multiple tasks from simple to complex.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"MULTI-TASK EVALUATION")
    print(f"{'='*70}")
    print(f"Model: {model_type.upper()}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes per task: {episodes_per_task}")
    print(f"Device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    norm_stats = checkpoint['norm_stats']
    
    print(f"\nCheckpoint Info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    
    # Create model
    if model_type == 'standard':
        model = StandardACT(
            joint_dim=39, action_dim=4, chunk_size=100,
            hidden_dim=512, feedforward_dim=3200, n_decoder_layers=7, n_heads=8,
            latent_dim=32, dropout=0.1
        ).to(device)
    else:
        model = ModifiedACT(
            joint_dim=39, action_dim=4, chunk_size=100,
            hidden_dim=512, feedforward_dim=3200, n_decoder_layers=7, n_heads=8,
            latent_dim=32, dropout=0.1
        ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test on increasingly complex tasks
    tasks = [
        'reach-v3',        # Easiest: just move gripper to target
        'push-v3',         # Easy: push object to target  
        'pick-place-v3',   # Medium: pick up and place object
        'shelf-place-v3',  # Current training task
    ]
    
    results = []
    for task in tasks:
        try:
            result = evaluate_on_task(
                model, norm_stats, task, 
                num_episodes=episodes_per_task,
                device=device
            )
            results.append(result)
        except Exception as e:
            print(f"\n⚠️  Error evaluating {task}: {e}")
            continue
    
    # Final Summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Task':<20} {'Success Rate':<15} {'Avg Distance':<15} {'Avg Reward':<15}")
    print(f"{'-'*70}")
    
    for result in results:
        task_display = result['task'].replace('-v3', '')
        print(f"{task_display:<20} {result['success_rate']:>6.1f}%{'':<8} "
              f"{result['avg_distance']:>8.4f}m{'':<6} {result['avg_reward']:>8.2f}")
    
    print(f"{'='*70}")
    
    # Analysis
    print(f"\n📊 Analysis:")
    if results:
        max_success = max(r['success_rate'] for r in results)
        if max_success > 0:
            print(f"  ✅ Model achieving {max_success:.1f}% success on some tasks!")
            best_task = [r for r in results if r['success_rate'] == max_success][0]['task']
            print(f"     Best performance on: {best_task}")
        else:
            print(f"  ⚠️  0% success on all tasks tested")
            print(f"     Model may need more training epochs")
            
            # Check if model is at least moving
            avg_dist = np.mean([r['avg_distance'] for r in results])
            if avg_dist < 0.5:
                print(f"  ✓ Average distance {avg_dist:.3f}m - robot is getting close!")
            else:
                print(f"  ⚠️  Average distance {avg_dist:.3f}m - robot not reaching goals")
    
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--model', type=str, default='standard', choices=['standard', 'modified'])
    parser.add_argument('--episodes', type=int, default=10, help='Episodes per task')
    args = parser.parse_args()
    
    main(args.checkpoint, args.model, args.episodes)
