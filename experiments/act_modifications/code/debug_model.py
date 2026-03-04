"""
Debug script to investigate why model gets 0% success despite good training loss.
"""
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.standard_act import StandardACT
import h5py

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load('checkpoints_proper/standard/best_model.pth', map_location=device, weights_only=False)

print("="*60)
print("MODEL CHECKPOINT ANALYSIS")
print("="*60)
print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Val Loss: {checkpoint.get('val_loss', 0):.4f}")

# Check normalization stats
if 'norm_stats' in checkpoint:
    stats = checkpoint['norm_stats']
    print(f"\nNormalization Stats:")
    print(f"  State mean: {stats['state_mean'][:5]}")
    print(f"  State std: {stats['state_std'][:5]}")
    print(f"  Action mean: {stats['action_mean']}")
    print(f"  Action std: {stats['action_std']}")
else:
    print("\n⚠️ No normalization stats in checkpoint!")

# Load a sample from training data
print("\n" + "="*60)
print("TRAINING DATA ANALYSIS")
print("="*60)

with h5py.File('data/single_task_demos_clipped.hdf5', 'r') as f:
    demo = f['demo_0']
    states = demo['states'][:]
    actions = demo['actions'][:]
    
    print(f"Demo 0:")
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  First state: {states[0][:10]}")
    print(f"  First action: {actions[0]}")
    print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    # Check if actions are clipped to [-1, 1]
    if actions.min() >= -1.0 and actions.max() <= 1.0:
        print("  ✓ Actions are properly clipped to [-1, 1]")
    else:
        print("  ✗ Actions are NOT clipped properly!")

# Load model and test inference
print("\n" + "="*60)
print("MODEL INFERENCE TEST")
print("="*60)

model = StandardACT(
    joint_dim=39,
    action_dim=4,
    chunk_size=100,
    hidden_dim=512,
    feedforward_dim=3200
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded successfully")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test with first state from training
with torch.no_grad():
    state_tensor = torch.FloatTensor(states[0]).unsqueeze(0).to(device)
    
    # Normalize state
    if 'norm_stats' in checkpoint:
        state_mean = torch.FloatTensor(checkpoint['norm_stats']['state_mean']).to(device)
        state_std = torch.FloatTensor(checkpoint['norm_stats']['state_std']).to(device)
        state_normalized = (state_tensor - state_mean) / (state_std + 1e-5)
    else:
        state_normalized = state_tensor
    
    # Get prediction
    pred_actions = model(state_normalized, None)
    
    print(f"\nInput state shape: {state_tensor.shape}")
    print(f"Predicted actions shape: {pred_actions.shape}")
    print(f"First predicted action: {pred_actions[0, 0].cpu().numpy()}")
    print(f"Predicted action range: [{pred_actions.min().item():.3f}, {pred_actions.max().item():.3f}]")
    
    # Denormalize action
    if 'norm_stats' in checkpoint:
        action_mean = torch.FloatTensor(checkpoint['norm_stats']['action_mean']).to(device)
        action_std = torch.FloatTensor(checkpoint['norm_stats']['action_std']).to(device)
        pred_denorm = pred_actions * action_std + action_mean
        print(f"\nDenormalized first action: {pred_denorm[0, 0].cpu().numpy()}")
        print(f"Denormalized range: [{pred_denorm.min().item():.3f}, {pred_denorm.max().item():.3f}]")
        
        # Check if denormalized actions are in valid range
        if pred_denorm.min().item() < -10 or pred_denorm.max().item() > 10:
            print("  ⚠️ WARNING: Denormalized actions seem too large!")
        
        # Re-clip to [-1, 1] for environment
        pred_clipped = torch.clamp(pred_denorm, -1, 1)
        print(f"After clipping to [-1,1]: {pred_clipped[0, 0].cpu().numpy()}")
    
    # Compare with ground truth
    gt_action = actions[0]
    print(f"\nGround truth action: {gt_action}")
    print(f"Predicted action (clipped): {pred_clipped[0, 0].cpu().numpy()}")
    print(f"Difference: {np.abs(pred_clipped[0, 0].cpu().numpy() - gt_action)}")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

# Key diagnostic questions
print("\n1. Are training actions properly clipped? Check above ☝️")
print("2. Are normalization stats correct? Check above ☝️")
print("3. Are predictions reasonable? Check above ☝️")
print("4. Is denormalization working? Check above ☝️")

print("\n" + "="*60)
