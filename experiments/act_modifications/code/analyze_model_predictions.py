#!/usr/bin/env python3
"""
Analyze what actions the model is actually predicting
"""

import torch
import numpy as np
import h5py
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append('/home/aryannzzz/GRASP/ACT-modification')

from models.standard_act import StandardACT

def analyze_predictions():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint_path = 'checkpoints_proper/standard/best_model.pth'
    print(f"\n{'='*70}")
    print(f"ANALYZING MODEL PREDICTIONS")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Epoch: {ckpt['epoch']}/300")
    print(f"Val Loss: {ckpt['val_loss']:.4f}")
    
    # Load model (match training configuration)
    model = StandardACT(
        joint_dim=39,
        action_dim=4,
        chunk_size=100,
        hidden_dim=512,
        n_encoder_layers=4,
        n_decoder_layers=7,
        n_heads=8,
        dropout=0.1,
        latent_dim=32
    ).to(device)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Load normalization stats
    data_path = 'data/diverse_demonstrations_with_images.hdf5'
    with h5py.File(data_path, 'r') as f:
        state_mean = f['normalization']['state_mean'][:]
        state_std = f['normalization']['state_std'][:]
        action_mean = f['normalization']['action_mean'][:]
        action_std = f['normalization']['action_std'][:]
    
    print(f"\n{'='*70}")
    print(f"NORMALIZATION STATS (from training data)")
    print(f"{'='*70}")
    print(f"State  - mean: {state_mean[:3]}, std: {state_std[:3]}")
    print(f"Action - mean: {action_mean}, std: {action_std}")
    
    # Load some actual test states
    print(f"\n{'='*70}")
    print(f"TESTING MODEL ON REAL STATES FROM DATASET")
    print(f"{'='*70}")
    
    with h5py.File(data_path, 'r') as f:
        # Get states from first demo
        demo_states = f['demo_0']['observations'][:]
        demo_actions = f['demo_0']['actions'][:]
        
        print(f"\nTesting on {len(demo_states)} states from demo_0")
        print(f"Original actions range: [{demo_actions.min():.3f}, {demo_actions.max():.3f}]")
        
        # Test on a few states
        all_predictions = []
        all_actual = []
        
        for i in range(0, len(demo_states), 50):  # Every 50th state
            state = demo_states[i]
            actual_action = demo_actions[i]
            
            # Normalize state
            state_normalized = (state - state_mean) / (state_std + 1e-8)
            state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                # Create dummy image (not used in prediction quality, just for forward pass)
                dummy_image = torch.zeros((1, 3, 224, 224), device=device)
                
                # Predict action sequence
                pred_actions = model.predict(state_tensor, dummy_image)  # Returns [1, chunk_size, 4]
                
                # Get first action
                pred_normalized = pred_actions[0, 0].cpu().numpy()
                
                # Denormalize
                pred_denormalized = pred_normalized * action_std + action_mean
                
                all_predictions.append(pred_denormalized)
                all_actual.append(actual_action)
                
                if i < 200:  # Print first 4 examples
                    print(f"\n  State {i}:")
                    print(f"    Actual:    [{actual_action[0]:6.3f}, {actual_action[1]:6.3f}, {actual_action[2]:6.3f}, {actual_action[3]:6.3f}]")
                    print(f"    Predicted: [{pred_denormalized[0]:6.3f}, {pred_denormalized[1]:6.3f}, {pred_denormalized[2]:6.3f}, {pred_denormalized[3]:6.3f}]")
                    print(f"    Normalized: [{pred_normalized[0]:6.3f}, {pred_normalized[1]:6.3f}, {pred_normalized[2]:6.3f}, {pred_normalized[3]:6.3f}]")
    
    # Statistics on predictions
    all_predictions = np.array(all_predictions)
    all_actual = np.array(all_actual)
    
    print(f"\n{'='*70}")
    print(f"PREDICTION STATISTICS (N={len(all_predictions)} samples)")
    print(f"{'='*70}")
    print(f"\nActual actions:")
    print(f"  Range: [{all_actual.min():.3f}, {all_actual.max():.3f}]")
    print(f"  Mean:  {all_actual.mean(axis=0)}")
    print(f"  Std:   {all_actual.std(axis=0)}")
    
    print(f"\nPredicted actions (denormalized):")
    print(f"  Range: [{all_predictions.min():.3f}, {all_predictions.max():.3f}]")
    print(f"  Mean:  {all_predictions.mean(axis=0)}")
    print(f"  Std:   {all_predictions.std(axis=0)}")
    
    # Check if predictions are too uniform
    pred_std = all_predictions.std(axis=0)
    if (pred_std < 0.05).any():
        print(f"\n⚠️  WARNING: Some dimensions have very low variance!")
        print(f"    This suggests model may be outputting near-constant actions")
        for i, s in enumerate(pred_std):
            if s < 0.05:
                print(f"    Dim {i}: std={s:.4f} (mean={all_predictions.mean(axis=0)[i]:.3f})")
    
    # Check prediction-actual correlation
    print(f"\n{'='*70}")
    print(f"PREDICTION-ACTUAL CORRELATION")
    print(f"{'='*70}")
    for dim in range(4):
        corr = np.corrcoef(all_predictions[:, dim], all_actual[:, dim])[0, 1]
        print(f"Dim {dim}: r={corr:.3f}", end="")
        if corr > 0.5:
            print(" ✓ (good correlation)")
        elif corr > 0.2:
            print(" ~ (moderate)")
        else:
            print(" ✗ (poor - model not learning this dimension)")

if __name__ == '__main__':
    analyze_predictions()
