"""
Collect diverse demonstrations using MetaWorld's built-in SAC expert policy.

This script addresses the root cause of 0% success rate by collecting
demonstrations with RANDOMIZED initial states, enabling the model to
generalize beyond a single fixed configuration.
"""

import numpy as np
import h5py
import os
from tqdm import tqdm
import metaworld
import random
from metaworld.policies import SawyerShelfPlaceV3Policy

def collect_diverse_demos_with_expert(num_demos=150, output_path='data/diverse_demonstrations.hdf5'):
    """
    Collect demonstrations using MetaWorld's expert policy with randomized initial states.
    
    Args:
        num_demos: Number of demonstrations to collect
        output_path: Path to save the HDF5 file
    """
    print("="*60)
    print("COLLECTING DIVERSE DEMONSTRATIONS WITH EXPERT POLICY")
    print("="*60)
    print(f"Number of demos: {num_demos}")
    print(f"Output path: {output_path}")
    print(f"Task: shelf-place-v3")
    print(f"Initial states: RANDOMIZED (different each episode)")
    print(f"Policy: MetaWorld SAC Expert")
    print("="*60)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize MetaWorld environment with ML1
    ml1 = metaworld.ML1('shelf-place-v3')
    env = ml1.train_classes['shelf-place-v3']()
    
    # Get expert policy
    expert_policy = SawyerShelfPlaceV3Policy()
    
    # Storage for demonstrations
    all_observations = []
    all_actions = []
    all_rewards = []
    all_dones = []
    
    # Track initial state diversity
    initial_qpos_list = []
    initial_obj_pos_list = []
    
    successful_demos = 0
    failed_demos = 0
    
    print("\nCollecting demonstrations...")
    pbar = tqdm(total=num_demos, desc="Successful demos")
    
    attempts = 0
    max_attempts = num_demos * 3  # Allow up to 3x attempts
    
    while successful_demos < num_demos and attempts < max_attempts:
        attempts += 1
        
        # CRITICAL: Set a DIFFERENT random task for each episode
        # This ensures diverse initial states
        task = random.choice(ml1.train_tasks)
        env.set_task(task)
        
        # Reset with random seed
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # MetaWorld returns (obs, info) tuple
        
        # Record initial state for diversity tracking
        initial_obj_pos = np.array(obs[4:7])  # Object position
        initial_obj_pos_list.append(initial_obj_pos)
        
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        
        done = False
        step = 0
        total_reward = 0
        max_steps = 500
        
        # Run expert policy
        while not done and step < max_steps:
            # Get action from expert
            action = expert_policy.get_action(obs)
            
            # Execute action
            step_result = env.step(action)
            if len(step_result) == 5:  # (obs, reward, terminated, truncated, info)
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:  # (obs, reward, done, info)
                next_obs, reward, done, info = step_result
            
            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            
            # Store transition
            episode_obs.append(obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)
            
            obs = next_obs
            total_reward += reward
            step += 1
        
        # Check if demonstration is successful
        # MetaWorld reward is 10.0 for success
        if total_reward > 5.0:  # Accept if got reasonable reward
            all_observations.append(np.array(episode_obs))
            all_actions.append(np.array(episode_actions))
            all_rewards.append(np.array(episode_rewards))
            all_dones.append(np.array(episode_dones))
            successful_demos += 1
            pbar.update(1)
        else:
            failed_demos += 1
            if attempts % 10 == 0:
                tqdm.write(f"Attempt {attempts}: {successful_demos}/{num_demos} successful, {failed_demos} failed")
    
    pbar.close()
    env.close()
    
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    print(f"Successful demos: {successful_demos}/{num_demos}")
    print(f"Failed demos: {failed_demos}")
    print(f"Total attempts: {attempts}")
    print(f"Success rate: {successful_demos/attempts*100:.1f}%")
    
    if successful_demos == 0:
        print("\n⚠️  WARNING: No successful demonstrations collected!")
        print("This might be due to:")
        print("1. Expert policy not working with this MetaWorld version")
        print("2. Task configuration issues")
        print("3. Reward threshold too high")
        return None
    
    # Analyze initial state diversity
    initial_obj_pos_array = np.array(initial_obj_pos_list[:successful_demos])
    
    print("\nInitial State Diversity:")
    print(f"Object X range: [{initial_obj_pos_array[:, 0].min():.3f}, {initial_obj_pos_array[:, 0].max():.3f}]")
    print(f"Object Y range: [{initial_obj_pos_array[:, 1].min():.3f}, {initial_obj_pos_array[:, 1].max():.3f}]")
    print(f"Object Z range: [{initial_obj_pos_array[:, 2].min():.3f}, {initial_obj_pos_array[:, 2].max():.3f}]")
    print(f"Mean object pos: {initial_obj_pos_array.mean(axis=0)}")
    print(f"Std object pos: {initial_obj_pos_array.std(axis=0)}")
    
    # Verify diversity
    std_values = initial_obj_pos_array.std(axis=0)
    if np.all(std_values < 0.001):
        print("\n⚠️  WARNING: Very low diversity detected!")
        print("Standard deviations are near zero, suggesting fixed initial states.")
    else:
        print("\n✅ Good diversity detected! Std values show variation.")
    
    # Save to HDF5
    print(f"\nSaving to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        # Store demonstrations
        for i, (obs, act, rew, done) in enumerate(zip(all_observations, all_actions, all_rewards, all_dones)):
            demo_group = f.create_group(f'demo_{i}')
            demo_group.create_dataset('observations', data=obs, compression='gzip')
            demo_group.create_dataset('actions', data=act, compression='gzip')
            demo_group.create_dataset('rewards', data=rew, compression='gzip')
            demo_group.create_dataset('dones', data=done, compression='gzip')
        
        # Store metadata
        f.attrs['num_demos'] = successful_demos
        f.attrs['task'] = 'shelf-place-v3'
        f.attrs['diverse_initial_states'] = True
        f.attrs['collection_method'] = 'expert_policy_with_random_init'
        
        # Store diversity statistics
        f.create_dataset('initial_object_positions', data=initial_obj_pos_array, compression='gzip')
        f.attrs['obj_pos_mean'] = initial_obj_pos_array.mean(axis=0)
        f.attrs['obj_pos_std'] = initial_obj_pos_array.std(axis=0)
    
    print("="*60)
    print("✅ DATA COLLECTION COMPLETE")
    print("="*60)
    print(f"File saved: {output_path}")
    print(f"Total demos: {successful_demos}")
    print(f"Ready for training!")
    print("="*60)
    
    return output_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Collect diverse demonstrations with expert policy')
    parser.add_argument('--num_demos', type=int, default=150, 
                        help='Number of successful demonstrations to collect')
    parser.add_argument('--output', type=str, default='data/diverse_demonstrations.hdf5',
                        help='Output HDF5 file path')
    args = parser.parse_args()
    
    collect_diverse_demos_with_expert(num_demos=args.num_demos, output_path=args.output)
