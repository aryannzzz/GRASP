# Modified ACT: Images in VAE Encoder

This is a modified version of the **Action Chunking Transformer (ACT)** policy from LeRobot that includes images in the VAE encoder to help prevent overfitting on joint states.

## The Problem

In the standard ACT architecture, the VAE encoder only receives:
- CLS token
- Robot state (joint positions)
- Action sequence

This means the latent distribution is learned **only from proprioceptive information**, which can lead to overfitting on joint states and poor generalization.

## The Solution

In Modified ACT, the VAE encoder receives:
- CLS token
- Robot state (joint positions)
- **Image features (pooled from backbone)** ← NEW
- Action sequence

By conditioning the latent distribution on visual information, the model learns a more robust representation that considers both what the robot is doing (actions) and what it sees (images).

## Architecture Comparison

```
Standard ACT VAE Encoder:
┌─────────────────────────────────────┐
│  [CLS] [State] [Action_1...Action_T] │
└─────────────────────────────────────┘

Modified ACT VAE Encoder:
┌───────────────────────────────────────────────────────┐
│  [CLS] [State] [Img_1...Img_N] [Action_1...Action_T]  │
└───────────────────────────────────────────────────────┘
```

## Installation

Copy the `modified_act` folder to your LeRobot policies directory:

```bash
cp -r modified_act /path/to/lerobot/lerobot/policies/
```

## Configuration Options

The key new configuration options are:

```python
from lerobot.policies.modified_act import ModifiedACTConfig

config = ModifiedACTConfig(
    # Standard ACT options
    chunk_size=100,
    n_action_steps=100,
    use_vae=True,
    latent_dim=32,
    
    # NEW: Modified ACT specific options
    vae_encoder_use_images=True,  # Enable images in VAE encoder
    vae_encoder_image_pooling="global_avg",  # Pooling method
)
```

### `vae_encoder_use_images` (bool, default: `True`)
Whether to include image features in the VAE encoder input. Set to `False` to revert to standard ACT behavior.

### `vae_encoder_image_pooling` (str, default: `"global_avg"`)
How to process image features for the VAE encoder:

- **`"global_avg"`**: Global average pooling → 1 token per camera. More efficient, recommended for most cases.
- **`"spatial"`**: Keep spatial structure → H'×W' tokens per camera. More expressive but significantly more tokens.

## Usage Examples

### Basic Training

```python
from lerobot.policies.modified_act import ModifiedACTConfig, ModifiedACTPolicy

# Create config
config = ModifiedACTConfig(
    input_features={
        "observation.state": ...,
        "observation.images.top": ...,
    },
    output_features={
        "action": ...,
    },
    vae_encoder_use_images=True,
    vae_encoder_image_pooling="global_avg",
)

# Create policy
policy = ModifiedACTPolicy(config)

# Training loop
for batch in dataloader:
    loss, loss_dict = policy(batch)
    loss.backward()
    optimizer.step()
```

### With LeRobot Training Script

Update your training config YAML:

```yaml
policy:
  _target_: lerobot.policies.modified_act.ModifiedACTPolicy
  config:
    _target_: lerobot.policies.modified_act.ModifiedACTConfig
    chunk_size: 100
    n_action_steps: 100
    use_vae: true
    vae_encoder_use_images: true
    vae_encoder_image_pooling: global_avg
```

### Inference

```python
# Load trained policy
policy = ModifiedACTPolicy.from_pretrained("path/to/checkpoint")
policy.eval()

# Get action
with torch.no_grad():
    action = policy.select_action(observation)
```

## Implementation Details

### Shared Backbone
The image backbone (ResNet18 by default) is **shared** between:
1. The VAE encoder (for latent distribution)
2. The main transformer encoder (for action prediction)

This keeps the model size reasonable and ensures consistent visual representations.

### Image Token Processing

For `global_avg` pooling:
```python
# Each camera produces one token
cam_features = backbone(image)  # [B, 512, H', W']
pooled = cam_features.mean(dim=[2, 3])  # [B, 512]
token = projection(pooled)  # [B, dim_model]
```

For `spatial` pooling:
```python
# Each camera produces H'×W' tokens
cam_features = backbone(image)  # [B, 512, H', W']
tokens = cam_features.flatten(2).permute(0, 2, 1)  # [B, H'×W', 512]
tokens = projection(tokens)  # [B, H'×W', dim_model]
```

### Positional Embeddings

- **Global avg pooling**: Pre-computed sinusoidal embeddings for the full sequence
- **Spatial pooling**: 2D sinusoidal embeddings for image tokens + 1D for other tokens

## Comparison with Original ACT

| Aspect | Standard ACT | Modified ACT |
|--------|--------------|--------------|
| VAE encoder input | State + Actions | State + Images + Actions |
| Latent conditioning | Proprioceptive only | Proprioceptive + Visual |
| Overfitting risk | Higher on joint states | Lower (visual regularization) |
| Computational cost | Lower | Slightly higher |
| Model complexity | Simpler | More complex |

## Tips for Training

1. **Start with `global_avg` pooling**: It's faster and usually works well.

2. **Monitor KL divergence**: If KL loss collapses to 0, the VAE might be ignoring the latent. Try:
   - Reducing `kl_weight`
   - Using KL annealing

3. **Learning rate for backbone**: You may want to use a lower learning rate for the backbone:
   ```python
   config.optimizer_lr_backbone = 1e-6  # Lower than main lr
   ```

4. **Gradual enabling**: Start training with `vae_encoder_use_images=False`, then fine-tune with it enabled.

## File Structure

```
modified_act/
├── __init__.py                    # Package exports
├── configuration_modified_act.py  # Config class with new options
├── modeling_modified_act.py       # Model implementation
├── processor_modified_act.py      # Pre/post processing pipelines
└── README.md                      # This file
```

## Citation

If you use this modification, please cite the original ACT paper:

```bibtex
@article{zhao2023learning,
  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={arXiv preprint arXiv:2304.13705},
  year={2023}
}
```

## License

Apache 2.0, same as LeRobot.
