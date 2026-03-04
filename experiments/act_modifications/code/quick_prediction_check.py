"""
Quick check of model predictions with clipped data.
"""
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.standard_act import StandardACT
import metaworld

# Load checkpoint and stats
checkpoint_path = 'checkpoints_proper/standard/best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
norm_stats = checkpoint['norm_stats']

print(f"{'='*60}")
print(f"MODEL PREDICTION ANALYSIS (Clipped Data)")
print(f"{'='*60}")
print(f"Checkpoint: Epoch {checkpoint['epoch']}, Val Loss {checkpoint['val_loss']:.4f}\n")

# Show normalization stats
action_mean = norm_stats['action_mean']
action_std = norm_stats['action_std']
print("Normalization stats (from clipped data):")
print(f"  Mean: {action_mean}")
print(f"  Std:  {action_std}")

# Load model (use same config as training)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StandardACT(
    joint_dim=39, action_dim=4, chunk_size=100,  # Training uses 100
    hidden_dim=512, feedforward_dim=3200, n_decoder_layers=7, n_heads=8,
    latent_dim=32, dropout=0.1
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create environment and get a state
ml1 = metaworld.ML1('shelf-place-v3', seed=42)
env = ml1.train_classes['shelf-place-v3']()
env.set_task(ml1.train_tasks[0])
obs = env.reset()

# Get state and create dummy image
state = torch.FloatTensor(obs).unsqueeze(0).to(device)
dummy_image = torch.zeros(1, 3, 480, 480).to(device)
images_dict = {'corner2': dummy_image}

# Get predictions
with torch.no_grad():
    pred_actions = model(images_dict, state, training=False)
    action_norm = pred_actions[0, :5].cpu().numpy()  # First 5 actions

# Denormalize
actions_denorm = action_norm * action_std + action_mean

print(f"\n{'='*60}")
print("Model Predictions (First 5 actions of sequence):")
print(f"{'='*60}")

for i in range(5):
    print(f"\nAction {i}:")
    print(f"  Normalized:   {action_norm[i]}")
    print(f"  Denormalized: {actions_denorm[i]}")
    print(f"  Range: [{actions_denorm[i].min():.3f}, {actions_denorm[i].max():.3f}]")

print(f"\n{'='*60}")
print("Overall Statistics:")
print(f"{'='*60}")
print(f"Normalized range:   [{action_norm.min():.3f}, {action_norm.max():.3f}]")
print(f"Denormalized range: [{actions_denorm.min():.3f}, {actions_denorm.max():.3f}]")

# Check if actions are reasonable
if actions_denorm.max() <= 1.0 and actions_denorm.min() >= -1.0:
    print(f"\n✅ Actions are within [-1, 1] - Good!")
else:
    print(f"\n⚠️  Actions exceed [-1, 1] - May be clipped by environment")

# Estimate movement
movement_estimate = np.abs(actions_denorm[:, :3]).mean()  # First 3 dims (x,y,z)
print(f"\nEstimated movement per action: {movement_estimate:.4f}")
if movement_estimate < 0.1:
    print("⚠️  Movement seems small - robot may move slowly")
elif movement_estimate > 0.5:
    print("⚠️  Movement seems large - robot may move erratically")
else:
    print("✓ Movement magnitude looks reasonable")

print(f"\n{'='*60}\n")
