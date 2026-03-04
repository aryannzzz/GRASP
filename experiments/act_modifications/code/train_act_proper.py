"""
Proper ACT training following the original implementation:
1. Normalize actions and states using dataset statistics
2. Sample random timesteps from episodes (not just start)
3. Use ImageNet normalization for images
4. Use is_pad mask in loss calculation
"""

import os
import sys
import h5py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.standard_act import StandardACT
from models.modified_act import ModifiedACT


class ACTDataset(Dataset):
    """Dataset following original ACT implementation"""
    def __init__(self, hdf5_path, chunk_size=100, norm_stats=None):
        self.hdf5_path = hdf5_path
        self.chunk_size = chunk_size
        self.norm_stats = norm_stats
        
        with h5py.File(hdf5_path, 'r') as f:
            self.num_demos = f.attrs['num_demos']
            
            # Check data format - support both old and new format
            demo_0 = f['demo_0']
            if 'observations' in demo_0:
                # ACT format: demo_0/observations/qpos
                self.data_format = 'act'
                self.state_dim = demo_0['observations']['qpos'].shape[1]
                self.action_dim = demo_0['action'].shape[1]
            else:
                # Simple format: demo_0/states, demo_0/actions
                self.data_format = 'simple'
                self.state_dim = demo_0['states'].shape[1]
                self.action_dim = demo_0['actions'].shape[1]
            
            # Get episode lengths
            self.episode_lengths = []
            for i in range(self.num_demos):
                if self.data_format == 'act':
                    self.episode_lengths.append(len(f[f'demo_{i}']['observations']['qpos']))
                else:
                    self.episode_lengths.append(len(f[f'demo_{i}']['states']))
            
        print(f"Dataset: {self.num_demos} demos (format: {self.data_format})")
        print(f"  State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"  Chunk size: {chunk_size}")
        if norm_stats:
            print(f"  Action mean: {norm_stats['action_mean']}")
            print(f"  Action std: {norm_stats['action_std']}")
    
    def __len__(self):
        return self.num_demos
    
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as f:
            demo = f[f'demo_{idx}']
            
            # Load data based on format
            if self.data_format == 'act':
                states = demo['observations']['qpos'][:]
                actions = demo['action'][:]
                images = demo['observations']['images']['corner2'][:]
            else:
                states = demo['states'][:]
                actions = demo['actions'][:]
                images = demo['images'][:]
            
            # Sample random starting timestep (key difference from our old code!)
            episode_len = len(states)
            start_ts = np.random.randint(0, episode_len)
            
            # Get observation at start_ts
            curr_state = states[start_ts]
            curr_image = images[start_ts]
            
            # Get all actions from start_ts onwards
            future_actions = actions[start_ts:]
            action_len = len(future_actions)
            
            # Pad to chunk_size (with proper bounds checking)
            copy_len = min(action_len, self.chunk_size)
            padded_actions = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
            padded_actions[:copy_len] = future_actions[:copy_len]
            
            # Create padding mask
            is_pad = np.zeros(self.chunk_size, dtype=bool)
            is_pad[copy_len:] = True
            
            # Normalize
            if self.norm_stats is not None:
                curr_state = (curr_state - self.norm_stats['state_mean']) / self.norm_stats['state_std']
                padded_actions = (padded_actions - self.norm_stats['action_mean']) / self.norm_stats['action_std']
            
            # Convert image to tensor and normalize with ImageNet stats
            image_tensor = torch.FloatTensor(curr_image).permute(2, 0, 1) / 255.0
            # ImageNet normalization
            image_tensor = (image_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                          torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            return {
                'state': torch.FloatTensor(curr_state),
                'image': image_tensor,
                'actions': torch.FloatTensor(padded_actions),
                'is_pad': torch.BoolTensor(is_pad)
            }


def compute_normalization_stats(hdf5_path):
    """Compute mean and std for states and actions"""
    print("Computing normalization statistics...")
    
    all_states = []
    all_actions = []
    
    with h5py.File(hdf5_path, 'r') as f:
        num_demos = f.attrs['num_demos']
        
        # Check format
        demo_0 = f['demo_0']
        if 'observations' in demo_0:
            data_format = 'act'
        else:
            data_format = 'simple'
        
        for i in range(num_demos):
            demo = f[f'demo_{i}']
            if data_format == 'act':
                all_states.append(demo['observations']['qpos'][:])
                all_actions.append(demo['action'][:])
            else:
                all_states.append(demo['states'][:])
                all_actions.append(demo['actions'][:])
    
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    
    state_mean = all_states.mean(axis=0)
    state_std = all_states.std(axis=0)
    state_std = np.clip(state_std, 1e-2, np.inf)  # Avoid division by zero
    
    action_mean = all_actions.mean(axis=0)
    action_std = all_actions.std(axis=0)
    action_std = np.clip(action_std, 1e-2, np.inf)
    
    stats = {
        'state_mean': state_mean,
        'state_std': state_std,
        'action_mean': action_mean,
        'action_std': action_std
    }
    
    print(f"State - mean: {state_mean[:5]}... std: {state_std[:5]}...")
    print(f"Action - mean: {action_mean} std: {action_std}")
    
    return stats


def train_model(model_type='standard', data_path='data/single_task_demos_clipped.hdf5',
                checkpoint_dir='checkpoints_proper', epochs=100, batch_size=8,
                lr=1e-4, kl_weight=10.0):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} ACT (PROPER)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Data: {data_path}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"KL weight: {kl_weight}")
    
    # Compute normalization stats
    norm_stats = compute_normalization_stats(data_path)
    
    # Create dataset
    dataset = ACTDataset(data_path, chunk_size=100, norm_stats=norm_stats)
    
    # Split into train and val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    if model_type == 'standard':
        model = StandardACT(
            joint_dim=39,
            action_dim=4,
            chunk_size=100,
            hidden_dim=512,
            feedforward_dim=3200,
            n_encoder_layers=4,
            n_decoder_layers=7,
            n_heads=8,
            n_cameras=1
        )
    else:  # modified
        model = ModifiedACT(
            joint_dim=39,
            action_dim=4,
            chunk_size=100,
            hidden_dim=512,
            feedforward_dim=3200,
            n_encoder_layers=4,
            n_decoder_layers=7,
            n_heads=8,
            n_cameras=1
        )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Create checkpoint directory
    ckpt_dir = os.path.join(checkpoint_dir, model_type)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Save normalization stats
    np.savez(os.path.join(ckpt_dir, 'norm_stats.npz'), **norm_stats)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_l1_losses = []
        epoch_kl_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            states = batch['state'].to(device)
            images = batch['image'].to(device)
            actions = batch['actions'].to(device)
            is_pad = batch['is_pad'].to(device)
            
            # Prepare images dict
            images_dict = {'corner2': images}
            
            # Forward pass
            pred_actions, z_mean, z_logvar = model(images_dict, states, actions, training=True)
            
            # Compute losses with padding mask
            # L1 loss (only on non-padded actions)
            all_l1 = nn.L1Loss(reduction='none')(pred_actions, actions)
            # Mask out padded actions
            l1_loss = (all_l1 * ~is_pad.unsqueeze(-1)).sum() / (~is_pad).sum() / actions.shape[-1]
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
            kl_loss = kl_loss / states.size(0)
            
            # Total loss
            loss = l1_loss + kl_weight * kl_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_l1_losses.append(l1_loss.item())
            epoch_kl_losses.append(kl_loss.item())
            
            pbar.set_postfix({
                'L1': f'{l1_loss.item():.4f}',
                'KL': f'{kl_loss.item():.4f}',
                'Total': f'{loss.item():.4f}'
            })
        
        avg_train_loss = np.mean(epoch_l1_losses) + kl_weight * np.mean(epoch_kl_losses)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_l1_losses = []
        val_kl_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                states = batch['state'].to(device)
                images = batch['image'].to(device)
                actions = batch['actions'].to(device)
                is_pad = batch['is_pad'].to(device)
                
                images_dict = {'corner2': images}
                pred_actions, z_mean, z_logvar = model(images_dict, states, actions, training=True)
                
                all_l1 = nn.L1Loss(reduction='none')(pred_actions, actions)
                l1_loss = (all_l1 * ~is_pad.unsqueeze(-1)).sum() / (~is_pad).sum() / actions.shape[-1]
                
                kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
                kl_loss = kl_loss / states.size(0)
                
                val_l1_losses.append(l1_loss.item())
                val_kl_losses.append(kl_loss.item())
        
        avg_val_loss = np.mean(val_l1_losses) + kl_weight * np.mean(val_kl_losses)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train - L1: {np.mean(epoch_l1_losses):.4f}, KL: {np.mean(epoch_kl_losses):.4f}, Total: {avg_train_loss:.4f}")
        print(f"  Val   - L1: {np.mean(val_l1_losses):.4f}, KL: {np.mean(val_kl_losses):.4f}, Total: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'norm_stats': norm_stats
            }, os.path.join(ckpt_dir, 'best_model.pth'))
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
    
    print(f"\n{'='*60}")
    print(f"✅ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {ckpt_dir}/best_model.pth")
    print(f"{'='*60}\n")
    
    return train_losses, val_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='standard', choices=['standard', 'modified'])
    parser.add_argument('--data', type=str, default='data/single_task_demos_clipped.hdf5')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--kl_weight', type=float, default=10.0)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_proper')
    args = parser.parse_args()
    
    train_model(
        model_type=args.model,
        data_path=args.data,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        kl_weight=args.kl_weight
    )
