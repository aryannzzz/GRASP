"""
Collect diverse demonstrations with RANDOMIZED initial states.

Key improvements over previous data collection:
1. Random initial states for each episode (no fixed seed)
2. Proper state diversity for generalization
3. Clear logging of initial state variations
4. Quality checks for successful demonstrations
"""

import numpy as np
import h5py
import os
from tqdm import tqdm
import metaworld
import random

def collect_diverse_demos(num_demos=150, output_path='data/diverse_demonstrations.hdf5'):
    """
    Collect demonstrations with randomized initial states.
    
    Args:
        num_demos: Number of demonstrations to collect (default 150)
        output_path: Path to save the HDF5 file
    """
    print("="*60)
    print("COLLECTING DIVERSE DEMONSTRATIONS")
    print("="*60)
    print(f"Number of demos: {num_demos}")
    print(f"Output path: {output_path}")
    print(f"Task: shelf-place-v3")
    print(f"Initial states: RANDOMIZED (different each episode)")
    print("="*60)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize MetaWorld environment
    ml1 = metaworld.ML1('shelf-place-v3')
    env = ml1.train_classes['shelf-place-v3']()
    
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
    for demo_idx in tqdm(range(num_demos), desc="Demos"):
        # CRITICAL: Set a DIFFERENT random seed for each episode
        # This ensures diverse initial states
        task = random.choice(ml1.train_tasks)
        env.set_task(task)
        
        # Reset with random seed
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # MetaWorld returns (obs, info) tuple
        
        # Record initial state for diversity tracking
        # Object position from observation
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
        
        # Use scripted policy to generate demonstration
        while not done and step < max_steps:
            # Get object and goal positions from observation
            gripper_pos = obs[:3]
            obj_pos = obs[4:7]  # Object position
            goal_pos = obs[7:10] if len(obs) > 9 else obs[4:7] + np.array([0, 0, 0.2])  # Goal/shelf position
            
            # Simple scripted policy: reach -> grasp -> lift -> place
            action = np.zeros(4)
            
            # Phase 1: Move to object
            if np.linalg.norm(gripper_pos - obj_pos) > 0.02:
                action[:3] = (obj_pos - gripper_pos) * 10
                action[3] = -1  # Open gripper
            # Phase 2: Grasp
            elif obj_pos[2] < 0.15:  # Object still on table
                action[:3] = [0, 0, 0]
                action[3] = 1  # Close gripper
            # Phase 3: Lift
            elif gripper_pos[2] < goal_pos[2]:
                action[:3] = [0, 0, 1]  # Lift up
                action[3] = 1  # Keep gripper closed
            # Phase 4: Move to shelf
            else:
                action[:3] = (goal_pos - gripper_pos) * 5
                action[3] = 1  # Keep gripper closed
            
            # Clip actions
            action = np.clip(action, -1, 1)
            
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
        
        # Check if demonstration is successful (at least some progress)
        # Accept demos that get reasonable reward (not just failures)
        if total_reward > 0.5:  # Lowered threshold to accept more diverse demos
            all_observations.append(np.array(episode_obs))
            all_actions.append(np.array(episode_actions))
            all_rewards.append(np.array(episode_rewards))
            all_dones.append(np.array(episode_dones))
            successful_demos += 1
        else:
            failed_demos += 1
            tqdm.write(f"Demo {demo_idx}: Failed (reward={total_reward:.2f}), skipping...")
    
    env.close()
    
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    print(f"Successful demos: {successful_demos}/{num_demos}")
    print(f"Failed demos: {failed_demos}/{num_demos}")
    
    # Analyze initial state diversity
    initial_obj_pos_array = np.array(initial_obj_pos_list)
    
    print("\nInitial State Diversity:")
    print(f"Object X range: [{initial_obj_pos_array[:, 0].min():.3f}, {initial_obj_pos_array[:, 0].max():.3f}]")
    print(f"Object Y range: [{initial_obj_pos_array[:, 1].min():.3f}, {initial_obj_pos_array[:, 1].max():.3f}]")
    print(f"Object Z range: [{initial_obj_pos_array[:, 2].min():.3f}, {initial_obj_pos_array[:, 2].max():.3f}]")
    print(f"Mean object pos: {initial_obj_pos_array.mean(axis=0)}")
    print(f"Std object pos: {initial_obj_pos_array.std(axis=0)}")
    
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
        f.attrs['collection_method'] = 'scripted_policy_with_random_init'
        
        # Store diversity statistics
        f.create_dataset('initial_object_positions', data=initial_obj_pos_array[:successful_demos], compression='gzip')
        f.attrs['obj_pos_mean'] = initial_obj_pos_array[:successful_demos].mean(axis=0)
        f.attrs['obj_pos_std'] = initial_obj_pos_array[:successful_demos].std(axis=0)
    
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
    parser = argparse.ArgumentParser(description='Collect diverse demonstrations')
    parser.add_argument('--num_demos', type=int, default=150, 
                        help='Number of demonstrations to collect')
    parser.add_argument('--output', type=str, default='data/diverse_demonstrations.hdf5',
                        help='Output HDF5 file path')
    args = parser.parse_args()
    
    collect_diverse_demos(num_demos=args.num_demos, output_path=args.output)
