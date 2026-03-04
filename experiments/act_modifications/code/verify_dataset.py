"""
Comprehensive dataset verification before training.
Checks data integrity, statistics, and training pipeline.
"""

import h5py
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

def verify_dataset(hdf5_path):
    """
    Verify dataset is correct and ready for training.
    """
    print(f"\n{'='*70}")
    print(f"DATASET VERIFICATION")
    print(f"{'='*70}")
    print(f"File: {hdf5_path}\n")
    
    checks_passed = []
    checks_failed = []
    
    with h5py.File(hdf5_path, 'r') as f:
        # Check 1: Number of demos
        n_demos = len(f.keys())
        print(f"✓ Check 1: Number of demos")
        print(f"  Found: {n_demos} demos")
        if n_demos >= 100:
            checks_passed.append("Demo count")
        else:
            checks_failed.append(f"Demo count (expected ≥100, got {n_demos})")
        
        # Check 2: Demo structure
        print(f"\n✓ Check 2: Demo structure")
        demo_name = list(f.keys())[0]
        demo = f[demo_name]
        required_keys = ['states', 'actions', 'images']
        
        for key in required_keys:
            if key in demo:
                print(f"  ✓ Has '{key}'")
            else:
                print(f"  ✗ Missing '{key}'")
                checks_failed.append(f"Missing {key}")
        
        if all(key in demo for key in required_keys):
            checks_passed.append("Demo structure")
        
        # Check 3: Data shapes
        print(f"\n✓ Check 3: Data shapes")
        obs = demo['states'][:]
        actions = demo['actions'][:]
        images = demo['images'][:]
        
        print(f"  Observations: {obs.shape}")
        print(f"  Actions: {actions.shape}")
        print(f"  Images: {images.shape}")
        
        if obs.shape[1] == 39 and actions.shape[1] == 4:
            checks_passed.append("Data shapes")
        else:
            checks_failed.append(f"Data shapes (obs={obs.shape[1]}, act={actions.shape[1]})")
        
        # Check 4: Action ranges (CRITICAL)
        print(f"\n✓ Check 4: Action ranges (CRITICAL)")
        all_actions = []
        for demo_name in f.keys():
            all_actions.append(f[demo_name]['actions'][:])
        all_actions = np.concatenate(all_actions, axis=0)
        
        print(f"  Total actions: {len(all_actions):,}")
        print(f"  Global range: [{all_actions.min():.3f}, {all_actions.max():.3f}]")
        
        action_ok = True
        for i in range(all_actions.shape[1]):
            dim_data = all_actions[:, i]
            dim_min, dim_max = dim_data.min(), dim_data.max()
            dim_mean, dim_std = dim_data.mean(), dim_data.std()
            
            status = "✓" if -1.0 <= dim_min <= dim_max <= 1.0 else "✗"
            print(f"  {status} Dim {i}: [{dim_min:6.3f}, {dim_max:6.3f}] "
                  f"mean={dim_mean:6.3f} std={dim_std:5.3f}")
            
            if not (-1.0 <= dim_min <= dim_max <= 1.0):
                action_ok = False
        
        if action_ok:
            checks_passed.append("Actions in [-1, 1]")
        else:
            checks_failed.append("Actions outside [-1, 1]")
        
        # Check 5: Normalization stats
        print(f"\n✓ Check 5: Recommended normalization stats")
        action_mean = all_actions.mean(axis=0)
        action_std = all_actions.std(axis=0)
        
        print(f"  Action mean: {action_mean}")
        print(f"  Action std:  {action_std}")
        
        if np.all(np.abs(action_mean) < 1.0) and np.all(action_std > 0.1) and np.all(action_std < 1.0):
            checks_passed.append("Normalization stats reasonable")
        else:
            checks_failed.append("Normalization stats unusual")
        
        # Check 6: Image data
        print(f"\n✓ Check 6: Image data")
        sample_image = demo['images'][0]
        print(f"  Image shape: {sample_image.shape}")
        print(f"  Image range: [{sample_image.min()}, {sample_image.max()}]")
        print(f"  Image dtype: {sample_image.dtype}")
        
        if sample_image.shape == (480, 480, 3) and sample_image.max() <= 255:
            checks_passed.append("Image data")
        else:
            checks_failed.append("Image data format")
        
        # Check 7: Data completeness
        print(f"\n✓ Check 7: Data completeness")
        total_steps = 0
        for demo_name in f.keys():
            demo_len = len(f[demo_name]['actions'])
            total_steps += demo_len
        
        print(f"  Total steps across all demos: {total_steps:,}")
        
        if total_steps >= 45000:  # At least 450 steps per demo on average
            checks_passed.append("Data completeness")
        else:
            checks_failed.append(f"Insufficient data (expected ≥45000, got {total_steps})")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Passed: {len(checks_passed)}/{len(checks_passed) + len(checks_failed)}")
    for check in checks_passed:
        print(f"  ✓ {check}")
    
    if checks_failed:
        print(f"\n✗ Failed: {len(checks_failed)}")
        for check in checks_failed:
            print(f"  ✗ {check}")
        print(f"\n⚠️  WARNING: Dataset has issues! Fix before training.")
        return False
    else:
        print(f"\n🎉 All checks passed! Dataset ready for training.")
        return True

def test_dataloader(hdf5_path):
    """
    Test that PyTorch DataLoader works correctly.
    """
    print(f"\n{'='*70}")
    print(f"DATALOADER TEST")
    print(f"{'='*70}\n")
    
    from scripts.train_act_proper import ACTDataset
    
    dataset = ACTDataset(hdf5_path)
    print(f"✓ Dataset created")
    print(f"  Total samples: {len(dataset)}")
    
    # Test loading a batch
    sample = dataset[0]
    print(f"\n✓ Sample loaded")
    print(f"  Sample keys: {sample.keys()}")
    if 'images' in sample:
        print(f"  Images keys: {sample['images'].keys()}")
    if 'image' in sample:
        print(f"  Image shape: {sample['image'].shape}")
    if 'state' in sample:
        print(f"  State shape: {sample['state'].shape}")
    if 'joint_states' in sample:
        print(f"  Joint states shape: {sample['joint_states'].shape}")
    print(f"  Actions shape: {sample['actions'].shape}")
    
    # Check action ranges
    actions = sample['actions'].numpy()
    print(f"\n✓ Action sample check")
    print(f"  Range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"  Mean: {actions.mean(axis=0)}")
    
    # Test DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    
    print(f"\n✓ DataLoader works")
    print(f"  Batch size: {batch['actions'].shape[0]}")
    if 'state' in batch:
        print(f"  State shape: {batch['state'].shape}")
    if 'joint_states' in batch:
        print(f"  Joint states shape: {batch['joint_states'].shape}")
    print(f"  Actions shape: {batch['actions'].shape}")
    
    print(f"\n🎉 DataLoader test passed!")
    return True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--test-loader', action='store_true', help='Also test DataLoader')
    args = parser.parse_args()
    
    # Verify dataset
    dataset_ok = verify_dataset(args.data)
    
    # Test dataloader if requested
    if args.test_loader and dataset_ok:
        dataloader_ok = test_dataloader(args.data)
        
        if dataset_ok and dataloader_ok:
            print(f"\n{'='*70}")
            print(f"✅ ALL VERIFICATIONS PASSED - READY FOR TRAINING")
            print(f"{'='*70}\n")
            sys.exit(0)
    elif dataset_ok:
        print(f"\n✅ Dataset verification passed. Use --test-loader to test DataLoader.")
        sys.exit(0)
    else:
        print(f"\n❌ Dataset verification failed. Fix issues before training.")
        sys.exit(1)
