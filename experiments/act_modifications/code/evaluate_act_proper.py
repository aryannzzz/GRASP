"""
Proper ACT evaluation following original implementation:
1. Load normalization stats
2. Normalize states with ImageNet normalization for images
3. Denormalize predicted actions
4. Use z=0 for deterministic inference (temporal ensembling can be added later)
"""

import os
import sys
import torch
import numpy as np
import metaworld
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.standard_act import StandardACT
from models.modified_act import ModifiedACT


def evaluate_model(model_type='standard', checkpoint_path=None, num_episodes=50, 
                   use_temporal_agg=False, query_frequency=100):
    """
    Evaluate model on MetaWorld task
    
    Args:
        model_type: 'standard' or 'modified'
        checkpoint_path: Path to model checkpoint
        num_episodes: Number of episodes to evaluate
        use_temporal_agg: Whether to use temporal aggregation (like original ACT)
        query_frequency: How often to query the policy (100 = use full chunk, 1 = query every step)
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type.upper()} ACT")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Temporal aggregation: {use_temporal_agg}")
    print(f"Query frequency: {query_frequency}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    norm_stats = checkpoint['norm_stats']
    
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    # Create model
    if model_type == 'standard':
        model = StandardACT(
            joint_dim=39,
            action_dim=4,
            chunk_size=100,
            hidden_dim=512,
            feedforward_dim=3200,
            n_encoder_layers=4,
            n_decoder_layers=7,
            n_heads=8,
            n_cameras=1
        )
    else:
        model = ModifiedACT(
            joint_dim=39,
            action_dim=4,
            chunk_size=100,
            hidden_dim=512,
            feedforward_dim=3200,
            n_encoder_layers=4,
            n_decoder_layers=7,
            n_heads=8,
            n_cameras=1
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Setup MetaWorld environment
    print("\nSetting up MetaWorld environment...")
    ml1 = metaworld.ML1('shelf-place-v3')
    env = ml1.train_classes['shelf-place-v3'](render_mode='rgb_array')
    task = [task for task in ml1.train_tasks if task.env_name == 'shelf-place-v3'][0]
    env.set_task(task)
    
    # Evaluation loop
    print(f"\nRunning {num_episodes} episodes...")
    successes = []
    episode_lengths = []
    final_distances = []
    total_rewards = []
    
    # Pre and post processing functions (like original ACT)
    pre_process = lambda s: (s - norm_stats['state_mean']) / norm_stats['state_std']
    post_process = lambda a: a * norm_stats['action_std'] + norm_stats['action_mean']
    
    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        reset_output = env.reset()
        obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        
        episode_reward = 0
        episode_length = 0
        
        # For temporal aggregation
        if use_temporal_agg:
            all_time_actions = torch.zeros([500, 500+100, 4]).to(device)
        
        with torch.no_grad():
            for step in range(500):
                # Normalize state
                state_norm = pre_process(obs)
                state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
                
                # Get and normalize image
                image = env.render()
                # MetaWorld returns image with negative strides, need to copy
                image = np.ascontiguousarray(image.copy())
                
                if image.shape != (480, 480, 3):
                    # Resize if needed
                    try:
                        import cv2
                        image = cv2.resize(image, (480, 480))
                    except:
                        # Fallback: crop/pad to 480x480
                        h, w = image.shape[:2]
                        if h > 480:
                            image = image[:480, :]
                        if w > 480:
                            image = image[:, :480]
                
                image_tensor = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
                # ImageNet normalization
                image_tensor = (image_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                              torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image_tensor = image_tensor.unsqueeze(0).to(device)
                
                images_dict = {'corner2': image_tensor}
                
                # Query policy at specified frequency
                if step % query_frequency == 0:
                    # Predict action chunk using model inference (deterministic: z=0)
                    pred_actions = model(images_dict, state_tensor, training=False)
                    
                    if use_temporal_agg:
                        # Store actions for temporal aggregation
                        all_time_actions[[step], step:step+100] = pred_actions[0]
                
                # Select action based on temporal aggregation or direct use
                if use_temporal_agg:
                    actions_for_curr_step = all_time_actions[:, step]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    
                    if len(actions_for_curr_step) > 0:
                        # Exponential weighting
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                        action_norm = (actions_for_curr_step * exp_weights).sum(dim=0).cpu().numpy()
                    else:
                        action_norm = pred_actions[0, 0].cpu().numpy()
                else:
                    # Use action from the predicted chunk (like original ACT)
                    # When query_freq=100, we use actions 0-99 from one prediction
                    action_norm = pred_actions[0, step % query_frequency].cpu().numpy()
                
                # Denormalize action
                action = post_process(action_norm)
                
                # Don't clip! MetaWorld env will clip internally
                # Training data has actions outside [-1,1] from expert policy
                # action = np.clip(action, -1.0, 1.0)
                
                # Step environment
                step_output = env.step(action)
                if len(step_output) == 5:
                    obs, reward, terminated, truncated, info = step_output
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_output
                
                episode_reward += reward
                episode_length += 1
                
                if done or info.get('success', False):
                    break
        
        # Record results
        success = info.get('success', False)
        successes.append(success)
        episode_lengths.append(episode_length)
        total_rewards.append(episode_reward)
        
        # Get final distance to goal
        try:
            # New MuJoCo API
            site_id = env.model.site('obj_site').id
            obj_pos = env.data.site_xpos[site_id]
        except:
            # Fallback - just use object position from observation
            obj_pos = obs[4:7]  # Object position is in observation
        goal_pos = env._target_pos
        final_distance = np.linalg.norm(obj_pos - goal_pos)
        final_distances.append(final_distance)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Success Rate: {np.mean(successes)*100:.1f}% ({np.sum(successes)}/{num_episodes})")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average Final Distance: {np.mean(final_distances):.4f}m ± {np.std(final_distances):.4f}m")
    print(f"Average Total Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"{'='*60}\n")
    
    return {
        'success_rate': np.mean(successes),
        'avg_length': np.mean(episode_lengths),
        'avg_distance': np.mean(final_distances),
        'avg_reward': np.mean(total_rewards)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='standard', choices=['standard', 'modified'])
    parser.add_argument('--checkpoint', type=str, default='checkpoints_proper/standard/best_model.pth')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--temporal_agg', action='store_true', help='Use temporal aggregation')
    parser.add_argument('--query_freq', type=int, default=100, help='Query frequency (100=full chunk, 1=every step)')
    args = parser.parse_args()
    
    evaluate_model(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        use_temporal_agg=args.temporal_agg,
        query_frequency=args.query_freq
    )
