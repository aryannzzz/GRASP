# Experiment 2: Vision-Only ACT (Modified LeRobot)

**Goal**: Remove joint angle inputs from ACT entirely, forcing the policy to learn visually-grounded representations rather than relying on proprioceptive trajectory memorization.

---

## Motivation

### The Proprioceptive Bias Problem

Standard ACT (and most imitation learning methods) use both **image observations** and **joint state** (proprioception) as inputs. Joint states provide a strong, low-noise signal about the robot's current configuration, making it easy for the model to condition its outputs on where the robot is geometrically.

However, this creates a subtle problem:

> A policy with access to joint states can achieve near-zero training loss by memorizing **joint-angle trajectories** without learning anything about the visual scene.

This is because joint states in a fixed-task demonstration are highly repetitive and structured. The policy may essentially learn:
- "When joint state is near X₀, execute action sequence A₀"
- "When joint state is near X₁, execute action sequence A₁"
- ...

This is a form of state memorization rather than policy learning. The resulting model fails to generalize because:
1. New initial joint states → no memorized matching sequence
2. Visual variation is simply ignored by the learning process

### The Vision-Only Hypothesis

If we **remove all joint angle inputs**, the model has no choice but to reason from pixels. This:
- Forces the model to use vision to understand where the robot and objects are
- Eliminates the shortcut of proprioceptive trajectory memorization
- Should produce policies that are more robust to varied initial conditions

The intuition is that humans doing teleoperation use visual feedback, not explicit joint angle readout, to control the robot. A policy that imitates this process should benefit from the same visual grounding.

---

## Implementation

This experiment was implemented as a **direct modification to the LeRobot codebase** (HuggingFace's robot learning framework), rather than a standalone script.

### Why LeRobot?

LeRobot provides:
- A standardized `PreTrainedPolicy` interface compatible with HuggingFace Hub
- Efficient data loading with HuggingFace Datasets
- Reproducible training across robot platforms
- Easy model sharing and deployment

By integrating the modification into LeRobot's architecture, we get all these benefits for free.

### What Was Changed

**Modified files from standard LeRobot ACT**:

1. `configuration_modified_act.py` — Extended `ACTConfig` with new parameters:
   - `vae_encoder_use_images: bool` — Whether to include image features in the VAE encoder
   - `vae_encoder_image_pooling: str` — How to pool image features (`'global_avg'` or `'spatial'`)

2. `modeling_modified_act.py` — Modified `ACTPolicy` and internal `ACT` class:
   - `input_keys` now excludes `OBS_STATE` (no joint angle inputs)
   - VAE encoder conditionally receives pooled image features when `vae_encoder_use_images=True`
   - New `_pool_image_features()` method for extracting image tokens for the VAE encoder

3. `processor_modified_act.py` — `ModifiedACTImageProcessor` handles image preprocessing only (no state normalization needed)

### Architecture Difference

```
Standard ACT input:
  observations = {images: [B, C, H, W], state: [B, joint_dim]}
                                                  ↑
                                           joint angles included

Vision-Only ACT input:
  observations = {images: [B, C, H, W]}
                     ↑
            only pixels — no joint angles
```

The decoder architecture is unchanged — it still produces action chunks of the same dimensionality.

---

## File Structure

```
experiments/vision_only_act/
├── README.md                         # This file
└── code/
    ├── configuration_modified_act.py # Extended ACTConfig
    ├── modeling_modified_act.py      # Modified ACTPolicy (vision-only)
    ├── processor_modified_act.py     # Image processor module
    └── __init__.py
```

---

## Usage

### Integration into LeRobot

Copy these files into your LeRobot installation:

```bash
cp code/ /path/to/lerobot/lerobot/policies/modified_act/
```

Then register the policy in LeRobot's policy registry (add to `lerobot/policies/__init__.py`):

```python
from lerobot.policies.modified_act import ModifiedACTPolicy, ModifiedACTConfig
```

### Training (LeRobot CLI)

```bash
python lerobot/scripts/train.py \
  policy=modified_act \
  env=metaworld \
  dataset_repo_id=your_dataset \
  training.num_epochs=500
```

Or with a custom config:

```python
from code.configuration_modified_act import ModifiedACTConfig
from code.modeling_modified_act import ModifiedACTPolicy

config = ModifiedACTConfig(
    input_features={
        "observation.images.top": ImageFeature(shape=(3, 480, 480)),
        # NOTE: No "observation.state" entry — vision only
    },
    output_features={
        "action": ActionFeature(shape=(7,), type=FeatureType.ACTION),
    },
    chunk_size=100,
    vae_encoder_use_images=True,       # Key modification
    vae_encoder_image_pooling='global_avg',
)

policy = ModifiedACTPolicy(config)
```

### Inference

```python
observation = {
    "observation.images.top": image_tensor,  # [1, 3, 480, 480]
    # No joint state required
}

with torch.no_grad():
    action = policy.select_action(observation)
```

---

## Connection to ACT Modifications Experiment

This experiment and Experiment 1 (ACT Architecture Modifications) are complementary:

| Aspect | ACT Modifications | Vision-Only ACT |
|---|---|---|
| Joint state input | ✅ Included | ❌ Removed |
| Images in VAE encoder | ✅ Added | ✅ Added |
| Framework | Custom PyTorch | LeRobot (HuggingFace) |
| Key hypothesis | Better latent via images | Remove proprioceptive bias |

Together, they explore two different angles on the same problem: making the policy rely on visual information rather than proprioceptive shortcuts.

---

## HuggingFace Compatibility

The `ModifiedACTPolicy` class inherits from `PreTrainedPolicy` and is fully compatible with:

```python
# Push to HuggingFace Hub
policy.push_to_hub("your_username/vision-only-act")

# Load from Hub
policy = ModifiedACTPolicy.from_pretrained("your_username/vision-only-act")
```

---

## References

- Zhao et al. (2023). *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.* [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
- LeRobot: [https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
