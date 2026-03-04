"""
Record evaluation videos for ACT models
Saves videos showing model performance
"""

import os
import sys
import torch
import numpy as np
import metaworld
import argparse
from tqdm import tqdm
import imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.standard_act import StandardACT
from models.modified_act import ModifiedACT


def record_videos(model_type='standard', checkpoint_path=None, num_videos=10, 
                  output_dir='videos', query_freq=1, use_temporal_agg=False):
    """
    Record evaluation videos
    
    Args:
        model_type: 'standard' or 'modified'
        checkpoint_path: Path to model checkpoint
        num_videos: Number of videos to record
        output_dir: Directory to save videos
        query_freq: Query frequency
        use_temporal_agg: Use temporal aggregation
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Recording Videos for {model_type.upper()} ACT")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Number of videos: {num_videos}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Pre and post processing functions
    pre_process = lambda s: (s - norm_stats['state_mean']) / norm_stats['state_std']
    post_process = lambda a: a * norm_stats['action_std'] + norm_stats['action_mean']
    
    # Record videos
    print(f"\nRecording {num_videos} videos...")
    successes = 0
    
    for video_idx in tqdm(range(num_videos), desc="Recording"):
        reset_output = env.reset()
        obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        
        frames = []
        episode_reward = 0
        success = False
        
        # For temporal aggregation
        if use_temporal_agg:
            all_time_actions = torch.zeros([500, 500+100, 4]).to(device)
        
        with torch.no_grad():
            for step in range(500):
                # Render frame
                frame = env.render()
                frame = np.ascontiguousarray(frame.copy())
                frames.append(frame)
                
                # Normalize state
                state_norm = pre_process(obs)
                state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
                
                # Get and normalize image
                image = frame.copy()
                if image.shape[:2] != (480, 480):
                    # Resize if needed
                    try:
                        import cv2
                        image = cv2.resize(image, (480, 480))
                    except:
                        h, w = image.shape[:2]
                        new_img = np.zeros((480, 480, 3), dtype=np.uint8)
                        h_min = min(h, 480)
                        w_min = min(w, 480)
                        new_img[:h_min, :w_min] = image[:h_min, :w_min]
                        image = new_img
                
                image_tensor = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
                # ImageNet normalization
                image_tensor = (image_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                              torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image_tensor = image_tensor.unsqueeze(0).to(device)
                
                images_dict = {'corner2': image_tensor}
                
                # Query policy
                if step % query_freq == 0:
                    pred_actions = model(images_dict, state_tensor, training=False)
                    
                    if use_temporal_agg:
                        all_time_actions[[step], step:step+100] = pred_actions[0]
                
                # Select action
                if use_temporal_agg:
                    actions_for_curr_step = all_time_actions[:, step]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    
                    if len(actions_for_curr_step) > 0:
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                        action_norm = (actions_for_curr_step * exp_weights).sum(dim=0).cpu().numpy()
                    else:
                        action_norm = pred_actions[0, 0].cpu().numpy()
                else:
                    action_norm = pred_actions[0, step % query_freq].cpu().numpy()
                
                # Denormalize and clip action
                action = post_process(action_norm)
                action = np.clip(action, -1.0, 1.0)
                
                # Step environment
                step_output = env.step(action)
                if len(step_output) == 5:
                    obs, reward, terminated, truncated, info = step_output
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_output
                
                episode_reward += reward
                
                if info.get('success', False):
                    success = True
                
                if done or success:
                    break
        
        # Add text overlay to first frame
        first_frame = frames[0].copy()
        try:
            import cv2
            text = f"{model_type.upper()} ACT - Episode {video_idx+1}"
            cv2.putText(first_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)
            status = "SUCCESS" if success else "FAIL"
            color = (0, 255, 0) if success else (255, 0, 0)
            cv2.putText(first_frame, f"Status: {status}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            cv2.putText(first_frame, f"Steps: {len(frames)}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            frames[0] = first_frame
        except:
            pass
        
        # Save video
        video_path = os.path.join(output_dir, f'{model_type}_episode_{video_idx+1:03d}.mp4')
        imageio.mimsave(video_path, frames, fps=30)
        
        if success:
            successes += 1
    
    print(f"\n{'='*60}")
    print(f"Video Recording Complete!")
    print(f"{'='*60}")
    print(f"Videos saved to: {output_dir}")
    print(f"Success rate: {successes}/{num_videos} ({successes/num_videos*100:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='standard', choices=['standard', 'modified'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_videos', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='videos')
    parser.add_argument('--query_freq', type=int, default=100, help='Query frequency (100=full chunk)')
    parser.add_argument('--temporal_agg', action='store_true')
    args = parser.parse_args()
    
    record_videos(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        num_videos=args.num_videos,
        output_dir=args.output_dir,
        query_freq=args.query_freq,
        use_temporal_agg=args.temporal_agg
    )
