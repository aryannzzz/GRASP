"""
Clip actions in existing HDF5 dataset to [-1, 1] range.
This fixes the action range mismatch between training data and model predictions.
"""

import h5py
import numpy as np
import shutil
from pathlib import Path

def clip_dataset_actions(input_path, output_path=None):
    """
    Clip all actions in dataset to [-1, 1] and save to new file.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path for output file (if None, creates *_clipped.hdf5)
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_clipped{input_path.suffix}"
    
    print(f"{'='*60}")
    print(f"Clipping actions in dataset")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    
    # Read input dataset
    with h5py.File(input_path, 'r') as f_in:
        print(f"\nInput dataset:")
        print(f"  Demos: {len(f_in.keys())}")
        
        # Analyze action ranges
        all_actions = []
        for demo_name in f_in.keys():
            actions = f_in[demo_name]['actions'][:]
            all_actions.append(actions)
        
        all_actions = np.concatenate(all_actions, axis=0)
        print(f"\nOriginal action statistics:")
        print(f"  Total actions: {len(all_actions)}")
        print(f"  Range: [{all_actions.min():.3f}, {all_actions.max():.3f}]")
        
        outside_range = np.sum(np.abs(all_actions) > 1.0)
        pct = outside_range / all_actions.size * 100
        print(f"  Outside [-1,1]: {outside_range}/{all_actions.size} ({pct:.1f}%)")
        
        for i in range(all_actions.shape[1]):
            print(f"  Dim {i}: [{all_actions[:, i].min():7.3f}, {all_actions[:, i].max():7.3f}]")
        
        # Create output dataset
        with h5py.File(output_path, 'w') as f_out:
            for demo_name in f_in.keys():
                demo_group = f_in[demo_name]
                new_group = f_out.create_group(demo_name)
                
                # Copy all datasets
                for key in demo_group.keys():
                    if key == 'actions':
                        # Clip actions
                        actions = demo_group['actions'][:]
                        actions_clipped = np.clip(actions, -1.0, 1.0)
                        new_group.create_dataset('actions', data=actions_clipped)
                    else:
                        # Copy other data as-is
                        data = demo_group[key][:]
                        new_group.create_dataset(key, data=data)
                
                # Copy attributes
                for attr_name, attr_val in demo_group.attrs.items():
                    new_group.attrs[attr_name] = attr_val
    
    # Verify output
    print(f"\nVerifying output...")
    with h5py.File(output_path, 'r') as f:
        all_actions_clipped = []
        for demo_name in f.keys():
            actions = f[demo_name]['actions'][:]
            all_actions_clipped.append(actions)
        
        all_actions_clipped = np.concatenate(all_actions_clipped, axis=0)
        print(f"  Clipped range: [{all_actions_clipped.min():.3f}, {all_actions_clipped.max():.3f}]")
        
        assert all_actions_clipped.min() >= -1.0, "ERROR: Actions below -1.0!"
        assert all_actions_clipped.max() <= 1.0, "ERROR: Actions above 1.0!"
        print(f"  ✓ All actions within [-1, 1]")
    
    file_size_mb = output_path.stat().st_size / (1024**2)
    print(f"\n✓ Created clipped dataset: {output_path}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"{'='*60}\n")
    
    return output_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input HDF5 file')
    parser.add_argument('--output', type=str, default=None, help='Output HDF5 file (optional)')
    args = parser.parse_args()
    
    clip_dataset_actions(args.input, args.output)
