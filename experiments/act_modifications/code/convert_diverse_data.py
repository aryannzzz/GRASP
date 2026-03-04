"""
Convert diverse demonstrations to the format expected by train_act_proper.py

The new diverse data has:
- observations: (T, 39) array with raw env observations
- actions: (T, 4)

The old format expects:
- states: (T, 39) for joint/object positions
- images: (T, H, W, 3) for camera images  
- actions: (T, 4)

Since MetaWorld doesn't provide images in the observation, we need to
render them from the environment.
"""

import h5py
import numpy as np
import metaworld
import random
from tqdm import tqdm

def convert_diverse_data_with_images(input_path, output_path):
    """
    Convert diverse demonstrations to include rendered images.
    """
    print("="*60)
    print("CONVERTING DIVERSE DATA WITH IMAGE RENDERING")
    print("="*60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print("="*60)
    
    # Initialize environment for rendering
    ml1 = metaworld.ML1('shelf-place-v3')
    env = ml1.train_classes['shelf-place-v3']()
    task = random.choice(ml1.train_tasks)
    env.set_task(task)
    
    # Load input data
    with h5py.File(input_path, 'r') as f_in:
        num_demos = f_in.attrs['num_demos']
        print(f"\nProcessing {num_demos} demonstrations...")
        
        # Create output file
        with h5py.File(output_path, 'w') as f_out:
            for demo_idx in tqdm(range(num_demos), desc="Converting demos"):
                demo_in = f_in[f'demo_{demo_idx}']
                
                observations = demo_in['observations'][:]
                actions = demo_in['actions'][:]
                rewards = demo_in['rewards'][:]
                
                # Render images by replaying the episode
                env.reset()
                
                # Set initial state from first observation
                # MetaWorld observations: [gripper_pos(3), gripper_state(1), obj_pos(3), obj_rel_pos(3), ...]
                images = []
                
                for step_idx in range(len(observations)):
                    # Render current state
                    img = env.render(mode='rgb_array', width=480, height=480, camera_name='corner3')
                    images.append(img)
                    
                    # Step to next state (to keep env in sync)
                    if step_idx < len(actions):
                        env.step(actions[step_idx])
                
                images = np.array(images)  # (T, H, W, 3)
                
                # Create demo group in output
                demo_out = f_out.create_group(f'demo_{demo_idx}')
                demo_out.create_dataset('states', data=observations, compression='gzip')
                demo_out.create_dataset('images', data=images, compression='gzip')
                demo_out.create_dataset('actions', data=actions, compression='gzip')
                demo_out.create_dataset('rewards', data=rewards, compression='gzip')
            
            # Copy metadata
            for key, value in f_in.attrs.items():
                f_out.attrs[key] = value
    
    env.close()
    
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE")
    print("="*60)
    print(f"Output: {output_path}")
    print(f"Demos: {num_demos}")
    print("Format: Compatible with train_act_proper.py")
    print("="*60)

if __name__ == '__main__':
    convert_diverse_data_with_images(
        input_path='data/diverse_demonstrations.hdf5',
        output_path='data/diverse_demonstrations_with_images.hdf5'
    )
