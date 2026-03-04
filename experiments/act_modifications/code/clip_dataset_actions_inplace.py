"""
Clip actions in-place in HDF5 dataset (modifies original file).
More space-efficient than creating a copy.
"""

import h5py
import numpy as np
import shutil
from pathlib import Path

def clip_dataset_actions_inplace(file_path, create_backup=True):
    """
    Clip all actions in dataset to [-1, 1] in-place.
    
    Args:
        file_path: Path to HDF5 file (will be modified!)
        create_backup: If True, creates .backup copy first
    """
    file_path = Path(file_path)
    
    print(f"{'='*60}")
    print(f"Clipping actions IN-PLACE")
    print(f"{'='*60}")
    print(f"File: {file_path}")
    print(f"⚠️  WARNING: This will MODIFY the original file!")
    
    if create_backup:
        backup_path = file_path.with_suffix('.hdf5.backup')
        if backup_path.exists():
            print(f"\n⚠️  Backup already exists: {backup_path}")
            print(f"   Skipping backup creation")
        else:
            print(f"\n📦 Creating backup: {backup_path}")
            shutil.copy2(file_path, backup_path)
            backup_size = backup_path.stat().st_size / (1024**2)
            print(f"   ✓ Backup created ({backup_size:.1f} MB)")
    
    # Analyze and clip actions
    with h5py.File(file_path, 'r+') as f:
        print(f"\nDataset info:")
        print(f"  Demos: {len(f.keys())}")
        
        # First pass: analyze
        all_actions = []
        for demo_name in f.keys():
            actions = f[demo_name]['actions'][:]
            all_actions.append(actions)
        
        all_actions = np.concatenate(all_actions, axis=0)
        print(f"\nOriginal action statistics:")
        print(f"  Total actions: {len(all_actions)}")
        print(f"  Shape: {all_actions.shape}")
        print(f"  Range: [{all_actions.min():.3f}, {all_actions.max():.3f}]")
        
        outside_range = np.sum(np.abs(all_actions) > 1.0)
        pct = outside_range / all_actions.size * 100
        print(f"  Outside [-1,1]: {outside_range}/{all_actions.size} ({pct:.1f}%)")
        
        for i in range(all_actions.shape[1]):
            dim_data = all_actions[:, i]
            print(f"  Dim {i}: [{dim_data.min():7.3f}, {dim_data.max():7.3f}] "
                  f"mean={dim_data.mean():6.3f} std={dim_data.std():5.3f}")
        
        # Second pass: clip in-place
        print(f"\nClipping actions to [-1, 1]...")
        clipped_count = 0
        for demo_name in f.keys():
            actions = f[demo_name]['actions'][:]
            actions_clipped = np.clip(actions, -1.0, 1.0)
            
            # Count how many values changed
            changed = np.sum(actions != actions_clipped)
            clipped_count += changed
            
            # Write back in-place
            f[demo_name]['actions'][:] = actions_clipped
        
        print(f"  ✓ Clipped {clipped_count} values")
    
    # Verify
    print(f"\nVerifying clipped data...")
    with h5py.File(file_path, 'r') as f:
        all_actions_clipped = []
        for demo_name in f.keys():
            actions = f[demo_name]['actions'][:]
            all_actions_clipped.append(actions)
        
        all_actions_clipped = np.concatenate(all_actions_clipped, axis=0)
        print(f"  Clipped range: [{all_actions_clipped.min():.3f}, {all_actions_clipped.max():.3f}]")
        
        for i in range(all_actions_clipped.shape[1]):
            dim_data = all_actions_clipped[:, i]
            print(f"  Dim {i}: [{dim_data.min():7.3f}, {dim_data.max():7.3f}] "
                  f"mean={dim_data.mean():6.3f} std={dim_data.std():5.3f}")
        
        assert all_actions_clipped.min() >= -1.0, "ERROR: Actions below -1.0!"
        assert all_actions_clipped.max() <= 1.0, "ERROR: Actions above 1.0!"
        print(f"\n  ✓ All actions within [-1, 1]")
    
    file_size_mb = file_path.stat().st_size / (1024**2)
    print(f"\n✓ Successfully clipped actions in {file_path}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='HDF5 file to modify')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup creation')
    args = parser.parse_args()
    
    clip_dataset_actions_inplace(args.input, create_backup=not args.no_backup)
