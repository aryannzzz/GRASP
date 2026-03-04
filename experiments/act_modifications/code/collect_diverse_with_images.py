"""
Collect diverse demonstrations using MetaWorld's expert policy WITH images.

This version saves data in the format expected by train_act_proper.py:
- states: (T, 39) joint/object positions
- images: (T, H, W, 3) rendered camera images
- actions: (T, 4)
"""

import numpy as np
import h5py
import os
from tqdm import tqdm
import metaworld
import random
from metaworld.policies import SawyerShelfPlaceV3Policy

def collect_diverse_with_images(num_demos=100, output_path='data/diverse_demonstrations_with_images.hdf5'):
    """
    Collect demonstrations with expert policy, random init states, and rendered images.
    """
    print("="*60)
    print("COLLECTING DIVERSE DEMONSTRATIONS WITH IMAGES")
    print("="*60)
    print(f"Number of demos: {num_demos}")
    print(f"Output path: {output_path}")
    print("="*60)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize environment
    ml1 = metaworld.ML1('shelf-place-v3')
    env = ml1.train_classes['shelf-place-v3']()
    expert_policy = SawyerShelfPlaceV3Policy()
    
    # Storage
    all_states = []
    all_images = []
    all_actions = []
    all_rewards = []
    
    initial_obj_pos_list = []
    successful_demos = 0
    pbar = tqdm(total=num_demos, desc="Successful demos")
    
    attempts = 0
    max_attempts = num_demos * 3
    
    while successful_demos < num_demos and attempts < max_attempts:
        attempts += 1
        
        # Random task for diverse initial state
        task = random.choice(ml1.train_tasks)
        env.set_task(task)
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        initial_obj_pos = np.array(obs[4:7])
        initial_obj_pos_list.append(initial_obj_pos)
        
        episode_states = []
        episode_images = []
        episode_actions = []
        episode_rewards = []
        
        done = False
        step = 0
        total_reward = 0
        max_steps = 500
        
        while not done and step < max_steps:
            # Render image
            try:
                img = env.render(offscreen=True, camera_name='corner3', resolution=(480, 480))
                if img is None:
                    # Fallback: try different render method
                    img = env.sim.render(480, 480, camera_name='corner3')
            except:
                # If rendering fails, create dummy image (should not happen)
                img = np.zeros((480, 480, 3), dtype=np.uint8)
            
            # Store state and image
            episode_states.append(obs)
            episode_images.append(img)
            
            # Get action from expert
            action = expert_policy.get_action(obs)
            episode_actions.append(action)
            
            # Execute
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result
            
            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            
            episode_rewards.append(reward)
            obs = next_obs
            total_reward += reward
            step += 1
        
        # Accept if reasonable reward
        if total_reward > 5.0:
            all_states.append(np.array(episode_states))
            all_images.append(np.array(episode_images))
            all_actions.append(np.array(episode_actions))
            all_rewards.append(np.array(episode_rewards))
            successful_demos += 1
            pbar.update(1)
    
    pbar.close()
    env.close()
    
    print("\n" + "="*60)
    print(f"Successful: {successful_demos}/{num_demos}")
    print(f"Total attempts: {attempts}")
    
    if successful_demos == 0:
        print("⚠️  No successful demos!")
        return None
    
    # Analyze diversity
    initial_obj_pos_array = np.array(initial_obj_pos_list[:successful_demos])
    print("\nDiversity:")
    print(f"  X range: [{initial_obj_pos_array[:, 0].min():.3f}, {initial_obj_pos_array[:, 0].max():.3f}]")
    print(f"  Y range: [{initial_obj_pos_array[:, 1].min():.3f}, {initial_obj_pos_array[:, 1].max():.3f}]")
    print(f"  Std: {initial_obj_pos_array.std(axis=0)}")
    
    # Save in correct format
    print(f"\nSaving to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        for i, (states, images, actions, rewards) in enumerate(zip(all_states, all_images, all_actions, all_rewards)):
            demo_group = f.create_group(f'demo_{i}')
            demo_group.create_dataset('states', data=states, compression='gzip')
            demo_group.create_dataset('images', data=images, compression='gzip')
            demo_group.create_dataset('actions', data=actions, compression='gzip')
            demo_group.create_dataset('rewards', data=rewards, compression='gzip')
        
        f.attrs['num_demos'] = successful_demos
        f.attrs['task'] = 'shelf-place-v3'
        f.attrs['diverse_initial_states'] = True
        f.attrs['collection_method'] = 'expert_with_images'
        f.attrs['obj_pos_mean'] = initial_obj_pos_array.mean(axis=0)
        f.attrs['obj_pos_std'] = initial_obj_pos_array.std(axis=0)
    
    print("="*60)
    print("✅ COMPLETE")
    print(f"File: {output_path}")
    print(f"Demos: {successful_demos}")
    print("="*60)
    
    return output_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_demos', type=int, default=100)
    parser.add_argument('--output', type=str, default='data/diverse_demonstrations_with_images.hdf5')
    args = parser.parse_args()
    
    collect_diverse_with_images(num_demos=args.num_demos, output_path=args.output)
