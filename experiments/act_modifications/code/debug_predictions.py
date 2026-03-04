"""
Debug script to visualize model predictions and check if they make sense.
"""

import torch
import numpy as np
import h5py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.standard_act import StandardACT
import metaworld

def debug_model_predictions():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load checkpoint
    ckpt_path = 'checkpoints_proper/standard/best_model.pth'
    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Epoch: {ckpt['epoch']}, Val Loss: {ckpt['val_loss']:.4f}")
    
    # Load normalization stats
    norm_stats = ckpt['norm_stats']
    print(f"\nNormalization stats:")
    print(f"  Action mean: {norm_stats['action_mean']}")
    print(f"  Action std: {norm_stats['action_std']}")
    
    # Create model
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
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\n✅ Model loaded successfully")
    
    # Set up environment
    ml1 = metaworld.ML1('shelf-place-v3')
    env = ml1.train_classes['shelf-place-v3']()
    task = ml1.train_tasks[0]
    env.set_task(task)
    
    # Run one episode and check predictions
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    print(f"\n=== EPISODE DIAGNOSTICS ===")
    print(f"Initial observation shape: {obs.shape}")
    print(f"Gripper pos: {obs[:3]}")
    print(f"Object pos: {obs[4:7]}")
    print(f"Goal pos: {obs[7:10] if len(obs) > 9 else 'N/A'}")
    
    # Normalize observation
    state_norm = (obs - norm_stats['state_mean']) / norm_stats['state_std']
    
    # Get image
    try:
        img = env.render(offscreen=True, camera_name='corner3', resolution=(480, 480))
        if img is None:
            img = env.sim.render(480, 480, camera_name='corner3')
    except:
        img = np.zeros((480, 480, 3), dtype=np.uint8)
    
    # Prepare inputs
    state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
    image_tensor = torch.FloatTensor(img).permute(2, 0, 1) / 255.0
    image_tensor = (image_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                   torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    images_dict = {'corner2': image_tensor}
    
    # Get model prediction
    with torch.no_grad():
        pred_actions = model(images_dict, state_tensor, training=False)
    
    pred_actions_np = pred_actions[0].cpu().numpy()
    
    print(f"\n=== MODEL PREDICTIONS (NORMALIZED) ===")
    print(f"Predicted actions shape: {pred_actions_np.shape}")
    print(f"First 5 actions (normalized):")
    for i in range(min(5, len(pred_actions_np))):
        print(f"  Action {i}: {pred_actions_np[i]}")
    
    # Denormalize actions
    actions_denorm = pred_actions_np * norm_stats['action_std'] + norm_stats['action_mean']
    
    print(f"\n=== MODEL PREDICTIONS (DENORMALIZED) ===")
    print(f"First 5 actions (denormalized):")
    for i in range(min(5, len(actions_denorm))):
        print(f"  Action {i}: {actions_denorm[i]}")
    
    # Check if actions are clipped
    print(f"\n=== ACTION STATISTICS ===")
    print(f"Denormalized action range: [{actions_denorm.min():.3f}, {actions_denorm.max():.3f}]")
    print(f"Actions need to be clipped to [-1, 1] for MetaWorld")
    
    # Clip and test first action
    action_clipped = np.clip(actions_denorm[0], -1, 1)
    print(f"\nFirst action after clipping: {action_clipped}")
    
    # Execute first action and see what happens
    print(f"\n=== EXECUTING FIRST ACTION ===")
    step_result = env.step(action_clipped)
    if len(step_result) == 5:
        next_obs, reward, terminated, truncated, info = step_result
    else:
        next_obs, reward, terminated, info = step_result
        truncated = False
    
    if isinstance(next_obs, tuple):
        next_obs = next_obs[0]
    
    print(f"Reward: {reward}")
    print(f"New gripper pos: {next_obs[:3]}")
    print(f"New object pos: {next_obs[4:7]}")
    print(f"Distance moved: {np.linalg.norm(next_obs[:3] - obs[:3]):.4f}")
    
    # Check training data for comparison
    print(f"\n=== TRAINING DATA COMPARISON ===")
    with h5py.File('data/diverse_demonstrations_with_images.hdf5', 'r') as f:
        demo_0 = f['demo_0']
        train_actions = demo_0['actions'][:]
        train_states = demo_0['states'][:]
        
        print(f"Training actions range: [{train_actions.min():.3f}, {train_actions.max():.3f}]")
        print(f"First training action: {train_actions[0]}")
        print(f"Training state range: [{train_states.min():.3f}, {train_states.max():.3f}]")
    
    env.close()
    
    print(f"\n{'='*60}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*60}")

if __name__ == '__main__':
    debug_model_predictions()
