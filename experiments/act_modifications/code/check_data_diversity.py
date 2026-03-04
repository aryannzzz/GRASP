"""Quick script to analyze dataset diversity"""
import h5py
import numpy as np

with h5py.File('data/single_task_demos_clipped.hdf5', 'r') as f:
    print(f"Number of demos: {len(f.keys())}")
    
    # Get first observation of each demo
    first_obs_list = []
    for demo_key in sorted(f.keys()):
        obs = f[demo_key]['states'][0]  # Use 'states' not 'observations'
        first_obs_list.append(obs)
    
    first_obs_array = np.array(first_obs_list)
    
    # Object positions (indices 4:7)
    obj_pos = first_obs_array[:, 4:7]
    
    print("\n=== Initial State Diversity ===")
    print(f"Object X range: [{obj_pos[:, 0].min():.4f}, {obj_pos[:, 0].max():.4f}]")
    print(f"Object Y range: [{obj_pos[:, 1].min():.4f}, {obj_pos[:, 1].max():.4f}]") 
    print(f"Object Z range: [{obj_pos[:, 2].min():.4f}, {obj_pos[:, 2].max():.4f}]")
    print(f"\nObject pos std: {obj_pos.std(axis=0)}")
    print(f"Object pos mean: {obj_pos.mean(axis=0)}")
    
    # Check if all identical
    if obj_pos.std(axis=0).sum() < 0.001:
        print("\n❌ DATA IS NOT DIVERSE - All demos have same initial state!")
    else:
        print("\n✅ DATA HAS DIVERSITY - Different initial states")
