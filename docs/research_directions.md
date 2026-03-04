# Research Directions

This document summarizes each experimental thread in the GRASP project, its motivation, approach, and current status.

---

## 1. ACT Architecture Modifications

**Folder**: `experiments/act_modifications/`

**Motivation**

The standard ACT (Action Chunking Transformer) architecture uses a CVAE (Conditional Variational Autoencoder) encoder that takes only joint states and action sequences as input. The latent code `z` therefore reflects only the robot's proprioceptive state — not what the robot sees.

This means two visually different scenes that happen to have the same joint angles will produce the same latent code, which can limit the policy's ability to adapt to visual variation.

**Approach**

Modify the CVAE encoder to additionally receive image observations (via a ResNet18 feature extractor), so the latent code is conditioned on both what the robot sees and where it is.

Key changes:
- Add `ResNetEncoder` module to encode camera observations
- Concatenate image patch features with joint and action embeddings in the CVAE encoder
- Everything else (decoder, policy) remains identical to standard ACT

**Key Results**

| Model | Val Loss | Parameters | Notes |
|---|---|---|---|
| Standard ACT | 0.1289 | 61.9M | VAE encoder: (joints + actions) |
| Modified ACT | 0.0931 | 73.3M | **27.8% lower** — VAE encoder: (images + joints + actions) |

**Key Insight**

Despite better representation quality, both models had 0% evaluation success when training data was collected from a single fixed initial state. This revealed a data diversity problem more fundamental than architecture choice.

**Status**: Code complete. Training converged. Data diversity root cause identified and documented.

---

## 2. Vision-Only ACT (Modified LeRobot)

**Folder**: `experiments/vision_only_act/`

**Motivation**

Joint angle inputs provide a very strong signal that is highly specific to the exact start state. A model with access to joint angles can "cheat" by memorizing joint-angle trajectories rather than learning visually-grounded behavior.

By removing joint angle inputs entirely, we force the policy to learn from visual observations alone — closer to how a generalizable policy should work.

**Approach**

This experiment is implemented as a **direct modification to the LeRobot codebase** — the HuggingFace framework for robot learning. The modification is architecture-compatible with the LeRobot training pipeline, enabling:
- Training with LeRobot's data tooling
- Evaluation with LeRobot's standard evaluation scripts
- Model sharing via HuggingFace Hub

Key changes from standard LeRobot ACT:
- `ACTConfig`: Added `vae_encoder_use_images` and `vae_encoder_image_pooling` parameters
- `ACTPolicy`: Modified forward pass to conditionally include image features in VAE encoder
- `configuration_modified_act.py` and `modeling_modified_act.py`: Standalone HuggingFace-compatible module

**Status**: Implementation complete. HuggingFace-compatible module created.

---

## 3. Classical Pipeline Experiment

**Folder**: `experiments/classical_pipeline/`

**Motivation**

As a structured baseline, we implemented a classical computer vision pipeline instead of end-to-end learning. The hypothesis is that explicit scene understanding (object detection + geometric reasoning) followed by a simple policy can match or exceed end-to-end models on simple pick-and-place tasks, while being more interpretable and sample-efficient.

**Approach**

Pipeline:
```
Camera image
    → Open-vocabulary object detection (language-prompted)
    → Object pose estimation
    → Grasp point calculation
    → Robot arm control
```

Uses open-vocabulary detection to identify objects from natural language descriptions (e.g., "red block"), computes grasp parameters geometrically, and sends joint commands to the robot.

Also includes full **camera calibration** (intrinsics + extrinsics) for pixel-to-world coordinate mapping required for real-robot deployment.

**Status**: Validated in simulation. Partial deployment on real robot arm. See `notebooks/` for full walkthrough.

---

## 4. Grad-CAM for Vision-Language-Action Models

**Folder**: `experiments/gradcam_vla/`

**Motivation**

When a robot policy fails, it is hard to determine *why* without being able to inspect what the model is attending to. Is it looking at the target object? At the background? At an irrelevant object?

Gradient-weighted Class Activation Mapping (GradCAM) was developed for image classification to highlight which image regions drove a particular prediction. Adapting it to robot policies that take **both images and language** as input requires addressing new challenges:
- Outputs are continuous multi-dimensional actions, not discrete class scores
- Architecture uses Vision Transformers (patches, not conv feature maps)
- Attribution should be language-conditioned: "pick up red cup" and "push blue block" should highlight different regions

**Approach**

**Phase 1**: Implement GradCAM for RL policies (DQN on Atari, CNN policies).

**Phase 2**: Extend to VLA models (CLIP-based vision-language-action).

**Key Architecture Innovation**:

The naive approach (mean-pooling patch features before the action head) fails because mean-pooling gives every patch an equal gradient:
```
∂(mean_pool(patches)) / ∂(patch_k) = 1/N  for ALL k
```
This produces uniform saliency maps that look random.

**Fix**: Replace mean-pooling with **cross-attention pooling**, where text features (from the language instruction) attend over visual patches:
```python
# Text attends to visual patches → instruction-specific pooling
attn_scores = softmax((text @ patch_keys.T) / sqrt(d))  # [196] weights
vision_feat = (attn_scores.unsqueeze(-1) * patches).sum(0)  # weighted sum
```
Now gradients flow through instruction-specific attention weights, enabling language-conditioned saliency.

**Key Results**

| Metric | Before fix | After fix | Target |
|---|---|---|---|
| Cross-instruction correlation | 0.896 | **0.134** | < 0.5 |
| Best differentiation | +0.89 | **-0.564** (anticorr) | < 0 |
| Training loss | 0.000000 | 0.000012 | < 0.001 |

85% improvement in instruction-specific saliency differentiation.

**Status**: RL GradCAM complete. VLA-GradCAM with attention pooling complete and validated on 3 diverse scenes.

---

## 5. RoboPrompt — In-Context Learning for Robots

**Folder**: `experiments/roboprompt/`

**Motivation**

Large language models (LLMs) have demonstrated remarkable generalization through **in-context learning** (ICL): given a few examples in the prompt, they can perform new tasks without any gradient updates.

RoboPrompt applies this idea to robotics: use an LLM to specify robot behaviors from natural language task descriptions, generating structured action plans that a robot can execute.

**Approach**

Integration of:
1. **LLM task specification**: Language description → structured action plan
2. **LeRobot execution**: Trained low-level policy executes planned actions
3. **PyRep simulation**: CoppeliaSim-based simulation for testing
4. **Real robot bridge**: SO100/SO101 robot arm integration via LeRobot

Key components:
- `roboprompt_lerobot_bridge.py`: Converts LLM-generated discretized actions to continuous joint commands via IK
- `form_icl_demonstrations.py`: Constructs in-context prompts from existing demonstrations
- `inference_pipeline.py`: Full inference pipeline from language instruction to robot action
- `integrated_roboprompt_agent.py`: Agent that combines LLM reasoning with policy execution
- `perception_system.py`: Computer vision for object detection (ArUco markers, depth-based segmentation)

**Status**: Integration pipeline complete. Tested in simulation and partially on real robot.

---

## 6. Multi-task ACT Experiments

**Folder**: `experiments/multitask_act/`

**Motivation**

Training a separate policy for each task is inefficient. Multi-task learning — training a single policy on data from many tasks — offers:
- Better data utilization (shared representations across tasks)
- Potential for zero-shot generalization to new tasks
- More scalable deployment (one model for many behaviors)

**Approach**

Implemented multi-task data collection and training using MetaWorld tasks:

Tasks studied:
- `pick-place`: Pick an object and place it at a target location
- `handle-pull`: Pull a handle toward the robot
- `reach`: Move end-effector to target position
- `push`: Push object to target
- `drawer-open/close`: Operate drawers
- `door-open`: Open doors

Pipeline:
1. Collect expert demonstrations for each individual task (notebooks 01-04)
2. Combine datasets and train a unified ACT policy (notebook 05)
3. Analyze multi-task policy behavior

**Status**: Per-task data collection and training complete (notebooks 01-04). Multi-task unified policy training explored (notebook 05). Full convergence to multi-task success not yet achieved — experiment ongoing.

---

## Common Threads

All experiments share:
- **MetaWorld** or **PyBullet** as simulation environment
- **ACT** (Action Chunking Transformer) as the base policy architecture
- **LeRobot** for training infrastructure where applicable
- The central theme of learning generalizable policies under data constraints

The experiments are **complementary**: GradCAM validates what the modified ACT architecture actually attends to, the classical pipeline provides a baseline, RoboPrompt addresses goal specification, and multi-task learning addresses data efficiency scaling.
