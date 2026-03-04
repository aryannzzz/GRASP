<div align="center">

# GRASP
## Generalizable Robotic Action Models for Sensorimotor Policies

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![LeRobot](https://img.shields.io/badge/LeRobot-HuggingFace-yellow.svg)](https://github.com/huggingface/lerobot)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-orange.svg)]()

*Can robots learn generalizable manipulation skills from a handful of demonstrations?*

[Project Overview](docs/project_overview.md) · [Motivation](docs/grasp_motivation.md) · [Research Directions](docs/research_directions.md) · [Timeline](docs/project_timeline.md)

</div>

---

## Overview

**GRASP** is a research project investigating how robots can learn to manipulate objects from **limited human demonstrations** and generalize these skills beyond the exact conditions under which they were trained.

Classical robotics requires accurate system models, controlled environments, and deterministic pipelines. These systems fail in unstructured real-world settings where object positions, lighting, backgrounds, and scene clutter vary unpredictably. Robot learning through imitation offers flexibility — but faces a fundamental constraint: **data scarcity**. Unlike language or vision, there are no internet-scale robot datasets.

This project explores six complementary experimental threads, each probing a different angle on the same core question:

> **How do we train sensorimotor policies that generalize from fewer demonstrations?**

---

## Motivation

Human beings generalize manipulation skills effortlessly. We:
- Build **object-centric representations** that abstract away irrelevant scene structure
- Learn **invariant features** robust to changes in lighting, background, and viewpoint
- Maintain **world models** that support planning and prediction under uncertainty

Standard end-to-end pixel-to-action models struggle with generalization because they learn **spurious correlations** — background colors, camera angles, irrelevant nearby objects — rather than the true causal structure of the manipulation task. Overcoming this without large datasets requires deliberate architectural and methodological choices.

We investigate **structural inductive biases** as the lever: modifications to the representation, architecture, perception pipeline, and policy conditioning that encode the right prior knowledge without requiring thousands of demonstrations.

> **Full motivation:** [docs/grasp_motivation.md](docs/grasp_motivation.md)

---

## Key Results

<div align="center">

| Experiment | Key Finding | Metric |
|:---|:---|:---|
| [ACT Architecture Modification](#1-act-architecture-modifications) | Image-conditioned CVAE encoder improves representation quality | **27.8% lower** validation loss vs baseline |
| [Vision-Only ACT](#2-vision-only-act-lerobot) | Removing joint inputs forces visual grounding | HuggingFace-compatible module, comparable task performance |
| [GradCAM for VLAs](#4-grad-cam-for-vision-language-action-models) | Attention pooling yields instruction-specific saliency maps | **85% reduction** in cross-instruction correlation (0.896 → 0.134) |
| [Classical Pipeline](#3-classical-pipeline) | Open-vocabulary detection + geometric planning generalizes without demos | **~90% success** in simulation (new objects via text prompt) |
| [RoboPrompt Integration](#5-roboprompt--in-context-learning-for-robots) | LLM in-context learning bridges language specification to robot actions | End-to-end pipeline validated in simulation |
| [Multi-task ACT](#6-multi-task-act) | Task conditioning reduces negative transfer across manipulation tasks | **+10pp** success with task embedding vs unconditioned shared policy |

</div>

---

## Research Directions

<div align="center">

| Experiment | Folder | Goal | Status |
|:---|:---|:---|:---:|
| ACT Architecture Modifications | [`experiments/act_modifications/`](experiments/act_modifications/) | Image inputs to CVAE encoder | ✅ Complete |
| Vision-Only ACT (LeRobot) | [`experiments/vision_only_act/`](experiments/vision_only_act/) | Remove joint inputs, force visual learning | ✅ Complete |
| Classical Pipeline | [`experiments/classical_pipeline/`](experiments/classical_pipeline/) | Structured perception-to-action baseline | 🔄 Ongoing |
| Grad-CAM for VLAs | [`experiments/gradcam_vla/`](experiments/gradcam_vla/) | Interpretability for VLA models | ✅ Complete |
| RoboPrompt Integration | [`experiments/roboprompt/`](experiments/roboprompt/) | LLM-conditioned in-context robot learning | 🔄 Ongoing |
| Multi-task ACT | [`experiments/multitask_act/`](experiments/multitask_act/) | Single policy for multiple tasks | 🔄 Ongoing |

</div>

---

## Repository Structure

```
grasp-research-showcase/
│
├── README.md                              ← You are here
│
├── docs/
│   ├── project_overview.md                ← Full project description
│   ├── grasp_motivation.md                ← Scientific background & motivation
│   ├── research_directions.md             ← All experimental threads
│   └── project_timeline.md                ← How the project evolved
│
├── experiments/
│   ├── act_modifications/                 ← ACT with image-conditioned CVAE (Exp 1)
│   │   ├── README.md
│   │   ├── code/                          ← Models, training, evaluation, data scripts
│   │   └── configs/                       ← YAML hyperparameter configurations
│   │
│   ├── vision_only_act/                   ← LeRobot: vision-only policy (Exp 2)
│   │   ├── README.md
│   │   └── code/                          ← HuggingFace-compatible modules
│   │
│   ├── classical_pipeline/                ← Open-vocab detection + grasp planning (Exp 3)
│   │   ├── README.md
│   │   ├── camera_calibration/            ← Intrinsics, scripts, calibration images
│   │   └── notebooks/                     ← Full pipeline walkthrough notebooks
│   │
│   ├── gradcam_vla/                       ← GradCAM interpretability for VLAs (Exp 4)
│   │   ├── README.md
│   │   └── code/
│   │       ├── vla_gradcam/               ← Core GradCAM engine + CLIP-VLA models
│   │       ├── rl_gradcam/                ← GradCAM for RL (Atari, FrozenLake)
│   │       ├── pybullet_vla/              ← PyBullet integration
│   │       └── scripts/                   ← Validation, demo, analysis scripts
│   │
│   ├── roboprompt/                        ← LLM-conditioned robot learning (Exp 5)
│   │   ├── README.md
│   │   └── code/                          ← Bridge, pipeline, perception, ArUco
│   │
│   └── multitask_act/                     ← Multi-task ACT training (Exp 6)
│       ├── README.md
│       ├── code/                          ← Complete single-script pipeline
│       └── notebooks/                     ← Per-task and combined training notebooks
│
├── figures/
│   ├── architecture/                      ← Architecture diagrams (PNG)
│   └── *.png                              ← Result figures (GradCAM saliency, etc.)
│
└── assets/
    └── demos/                             ← Demo GIFs and evaluation videos
```

---

## Experiment Summaries

### 1. ACT Architecture Modifications

> *Can conditioning the CVAE encoder on visual observations improve policy representation quality?*

<div align="center">
<img src="figures/architecture/act_architecture.png" alt="ACT Architecture: Standard vs Modified" width="820"/>
<br><em>Standard ACT (left) vs Modified ACT with image-conditioned CVAE encoder (right)</em>
</div>

**What changed**: The standard ACT CVAE encoder processes only joint states and action sequences to produce a latent code `z`. Our modification adds ResNet18-encoded image features into the encoder, so `z` is conditioned on both *what the robot sees* and *where it is*.

**Result**: Modified ACT achieves **27.8% lower validation loss** (0.0931 vs 0.1289), demonstrating improved representation quality.

**Key insight**: Better architecture is necessary but not sufficient — training data diversity is the primary bottleneck. A 27.8% improvement in val loss coexisted with 0% evaluation success when data was collected from a single fixed initial state. After diversifying the dataset, evaluation performance improved substantially.

→ [Full experiment documentation](experiments/act_modifications/README.md)

---

### 2. Vision-Only ACT (LeRobot)

> *Does removing proprioceptive inputs force the policy to learn more generalizable visual representations?*

**What changed**: Removed all joint angle inputs from the ACT policy. The model receives only image observations — no end-effector positions, no joint angles. This prevents the policy from taking a shortcut by memorizing joint-angle trajectories.

**Implementation**: Delivered as a HuggingFace-compatible drop-in module for LeRobot (`ModifiedACTPolicy`) with a new `vae_encoder_use_images` configuration parameter.

**Why it matters**: A policy that relies on joint angles can achieve low training loss by trajectory memorization — a form of overfitting that collapses at test time when initial conditions vary. Visual grounding is a prerequisite for genuine generalization.

→ [Full experiment documentation](experiments/vision_only_act/README.md)

---

### 3. Classical Pipeline

> *How much can explicit structure — open-vocabulary detection + geometric planning — achieve without any demonstrations?*

<div align="center">
<img src="figures/architecture/classical_pipeline.png" alt="Classical Perception-to-Action Pipeline" width="820"/>
<br><em>Structured pipeline: object detection → 3D localization → grasp planning → execution</em>
</div>

**What it does**: Rather than learning end-to-end, this pipeline uses an open-vocabulary visual detector (prompted with natural language, e.g., "red block") to localize objects, computes 3D coordinates via camera intrinsics + depth, plans top-down grasps geometrically, and executes via IK.

**Results**: ~90% success in simulation on fixed objects. No demonstrations required — new objects are handled by changing the text prompt. Partial deployment on a real robot arm with calibrated camera.

**Takeaway**: Classical pipelines are highly interpretable and sample-efficient. As a baseline, this experiment quantifies exactly what structured prior knowledge alone can achieve before any learning enters the loop.

→ [Full experiment documentation](experiments/classical_pipeline/README.md)

---

### 4. Grad-CAM for Vision-Language-Action Models

> *Can we build attribution maps that show which image regions drive each action dimension, conditioned on the language instruction?*

<div align="center">
<img src="figures/architecture/gradcam_vla_pipeline.png" alt="VLA-GradCAM Pipeline" width="820"/>
<br><em>VLA-GradCAM: language-conditioned gradient attribution for robot policies</em>
</div>

**The problem with mean pooling**: Standard ViT-based VLAs pool patch features by averaging before the action head. This gives every patch the same gradient (`∂L/∂xi = 1/N` for all `i`), making GradCAM produce uniform, meaningless saliency maps regardless of the instruction.

**The fix — attention pooling**: We replace mean pooling with cross-attention, where the **text instruction** queries the visual patches. Now different instructions produce different attention distributions over patches, and GradCAM maps become instruction-specific.

**Result**: Cross-instruction saliency correlation reduced from **0.896 → 0.134** (85% improvement). For "pick up red cup" vs "push blue block" vs "grasp green ball", the saliency maps now focus distinctly on the correct object in the scene.

→ [Full experiment documentation](experiments/gradcam_vla/README.md)

---

### 5. RoboPrompt — In-Context Learning for Robots

> *Can LLMs specify complex robot behaviors via in-context learning, with only a handful of demonstration examples in the prompt?*

**Approach**: Use an LLM to convert a natural language task description into a sequence of discretized robot actions (position + rotation bins), via in-context examples formatted from recorded demonstrations. A bridge module converts discretized actions to continuous end-effector poses and then to joint commands via IK.

**Key components**:
- `roboprompt_lerobot_bridge.py` — Discretized action → continuous pose conversion
- `form_icl_demonstrations.py` — Format recorded demos for LLM prompts
- `integrated_roboprompt_agent.py` — Joint LLM planning + LeRobot execution
- `perception_system.py` — ArUco marker + depth-based object localization

**Why ICL?** Fine-tuning a robot policy requires large annotated datasets. In-context learning requires 3–10 examples in the prompt, generalizes to new tasks without gradient updates, and works with any LLM API.

→ [Full experiment documentation](experiments/roboprompt/README.md)

---

### 6. Multi-task ACT

> *Can a single ACT policy learn to perform multiple distinct manipulation tasks, and how do we mitigate negative task transfer?*

**Approach**: Train ACT on combined datasets from 7 MetaWorld tasks (pick-place, handle-pull, reach, push, drawer-open/close, door-open). Explore unconditioned shared policy, task-embedding-conditioned shared policy, and mixture-of-experts architectures.

**Results**: Unconditioned multi-task policy averaged ~45% success (vs ~65% for per-task oracles). Adding a task embedding raised this to ~55%. Contact-rich tasks (push, handle-pull) showed more task interference than simpler tasks (reach, pick-place).

**Takeaway**: Multi-task robot learning requires careful task conditioning and data balancing. The experiment establishes baselines for future curriculum and mixture-of-experts approaches.

→ [Full experiment documentation](experiments/multitask_act/README.md)

---

## Setup

### Environment Requirements

```bash
# Create environment
conda create -n grasp python=3.10
conda activate grasp

# Core ML
pip install torch>=2.0 torchvision transformers
pip install huggingface_hub datasets

# Robot learning framework
pip install lerobot

# Simulation
pip install gymnasium metaworld
pip install pybullet

# Computer vision
pip install opencv-python numpy scipy
pip install open-clip-torch  # For GradCAM experiments

# Visualization
pip install matplotlib seaborn grad-cam
```

### Per-Experiment Dependencies

| Experiment | Additional Requirements |
|:---|:---|
| ACT Modifications | `metaworld h5py tqdm` |
| Vision-Only ACT | `lerobot` (for LeRobot integration) |
| Classical Pipeline | `opencv-python transformers groundingdino` |
| GradCAM for VLAs | `open-clip-torch grad-cam` |
| RoboPrompt | `openai scipy lerobot pyrep` |
| Multi-task ACT | `metaworld h5py tqdm jupyter` |

---

## Running Experiments

### 1. ACT Training (MetaWorld shelf-place)

```bash
# Step 1: Collect diverse demonstrations with expert policy
python experiments/act_modifications/code/collect_diverse_with_expert.py \
    --task shelf-place-v3 --n_demos 100 --output data/demos.hdf5

# Step 2: Train standard ACT
python experiments/act_modifications/code/train_act_proper.py \
    --model standard \
    --config experiments/act_modifications/configs/standard_act.yaml \
    --data data/demos.hdf5

# Step 3: Train modified ACT (images in CVAE encoder)
python experiments/act_modifications/code/train_act_proper.py \
    --model modified \
    --config experiments/act_modifications/configs/production_config.yaml \
    --data data/demos.hdf5

# Step 4: Evaluate
python experiments/act_modifications/code/evaluate_act_proper.py \
    --model modified --checkpoint checkpoints/modified/best_model.pth --episodes 50
```

### 2. GradCAM Visualization

```bash
# Quick single-scene test (2 minutes)
python experiments/gradcam_vla/code/scripts/test_attention_vla.py

# Full multi-scene validation (5 minutes)
python experiments/gradcam_vla/code/scripts/multi_scene_validation.py

# Interactive visualization demo
python experiments/gradcam_vla/code/scripts/vla_gradcam_demo_v2.py
```

### 3. Classical Pipeline (Notebook)

```bash
jupyter notebook experiments/classical_pipeline/notebooks/
# Open: Final_OpenVocabPickPlace_Classical_v2_(5)_(4).ipynb
```

### 4. Multi-task ACT

```bash
# Complete pipeline: collect + train + evaluate all tasks
python experiments/multitask_act/code/metaworld_act_complete.py \
    --tasks pick-place handle-pull reach push \
    --n_demos 50 --epochs 300

# Or step-by-step via notebooks
jupyter notebook experiments/multitask_act/notebooks/
```

### 5. RoboPrompt (Simulation)

```python
from experiments.roboprompt.code.simple_roboprompt_runner import RoboPromptRunner

runner = RoboPromptRunner(use_sim=True)
result = runner.run_task("Pick up the red block and place it on the plate")
```

---

## Pre-trained Models

Models are hosted on HuggingFace Hub:

| Model | Task | Description | Link |
|:---|:---|:---|:---|
| `aryannzzz/act-metaworld-shelf-standard` | shelf-place | Standard ACT (joints + actions encoder) | [🤗 Hub](https://huggingface.co/aryannzzz/act-metaworld-shelf-standard) |
| `aryannzzz/act-metaworld-shelf-modified` | shelf-place | Modified ACT (images + joints + actions encoder) | [🤗 Hub](https://huggingface.co/aryannzzz/act-metaworld-shelf-modified) |

---

## Lessons Learned

### 1. Data Diversity is the Primary Bottleneck

Training ACT with demonstrations from a single fixed initial state yielded **0% evaluation success** despite excellent training and validation loss (0.0931). The model had perfectly memorized the training configuration. After re-collecting demonstrations from randomized initial states, evaluation success improved substantially.

> **Lesson**: In robot imitation learning, distributional shift between training and evaluation is often more limiting than model capacity or architecture. Always verify your training data covers the full evaluation distribution.

### 2. Architecture Improvements Are Necessary But Not Sufficient

The modified CVAE encoder (with images) produced a 27.8% improvement in validation loss, demonstrating that richer conditioning improves representation quality. But this architectural gain was masked by the data diversity problem. Both improvements are required together.

### 3. The Mean-Pooling Trap in VLA Interpretability

Our GradCAM experiments revealed that any mean-pooling operation before the prediction head makes gradient attribution meaningless — every patch receives the same gradient weight by mathematical necessity. This is not obvious and affects any interpretability method that relies on gradient flow, not just GradCAM. Attention-weighted pooling is a simple fix that restores spatial gradient information.

### 4. Classical Pipelines Quantify the Value of Demonstrations

The classical pipeline achieves strong performance on simple tasks without any demonstrations. This provides a direct baseline: if your learned policy cannot exceed 90% on the tasks the classical pipeline handles well, something is wrong with your training setup, not your model.

### 5. Multi-task Transfer Depends on Task Similarity

For multi-task ACT, tasks with similar motor patterns transferred positively (reach ↔ pick-place), while contact-rich tasks with different force profiles interfered (push ↔ door-open). Task conditioning mitigates this, but curriculum learning and meta-learning remain open directions.

---

## Future Work

- **Object-centric representations**: Replace dense image features with explicit object-level tokens (position, identity, pose) as policy inputs — a structural prior that directly encodes task-relevant abstraction
- **Foundation model integration**: Investigate using pre-trained visual foundation models (DINOv2, SAM) as frozen vision encoders for improved cross-domain generalization
- **Scaling GradCAM to large VLAs**: Extend the attention-pooling approach to full OpenVLA and RT-2-scale models with multi-layer, multi-scale attribution
- **World model-augmented planning**: Integrate a learned dynamics model for imagined rollouts, enabling model-based planning under distribution shift
- **Multi-step RoboPrompt tasks**: Scale language-conditioned in-context learning to longer-horizon tasks with sub-goal decomposition and execution monitoring
- **Curriculum learning for multi-task ACT**: Design principled curricula that reduce negative transfer, drawing on task similarity metrics derived from learned representations
- **Real-world evaluation**: Deploy the full ACT pipeline, GradCAM attribution, and RoboPrompt integration on the real robot arm for systematic evaluation under varied lighting and object configurations

---

## Documentation

| Document | Description |
|:---|:---|
| [docs/project_overview.md](docs/project_overview.md) | Full project description, goals, and technical stack |
| [docs/grasp_motivation.md](docs/grasp_motivation.md) | Scientific motivation, spurious correlations, human vs robot generalization |
| [docs/research_directions.md](docs/research_directions.md) | Detailed summary of each experimental thread |
| [docs/project_timeline.md](docs/project_timeline.md) | How the research evolved phase by phase |

---

## References

1. Zhao et al. (2023). *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.* RSS. [[arXiv:2304.13705]](https://arxiv.org/abs/2304.13705)
2. Chi et al. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS. [[arXiv:2303.04137]](https://arxiv.org/abs/2303.04137)
3. Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV. [[arXiv:1610.02391]](https://arxiv.org/abs/1610.02391)
4. Brohan et al. (2023). *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.* CoRL. [[arXiv:2307.15818]](https://arxiv.org/abs/2307.15818)
5. Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML. (CLIP)
6. Brown et al. (2020). *Language Models are Few-Shot Learners.* NeurIPS. (GPT-3 / ICL)
7. Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization.* ICLR. (AdamW)
8. Yu et al. (2020). *Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning.* CoRL. [[arXiv:1910.10897]](https://arxiv.org/abs/1910.10897)
9. **LeRobot**: [[github.com/huggingface/lerobot]](https://github.com/huggingface/lerobot)
10. **PyRep**: [[github.com/stepjam/PyRep]](https://github.com/stepjam/PyRep)

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{grasp_robot_learning_2026,
  title     = {{GRASP}: Generalizable Robotic Action Models for Sensorimotor Policies},
  author    = {Aryan},
  year      = {2026},
  note      = {Research repository. Experiments in ACT architecture modifications,
               VLA interpretability, classical perception pipelines, and
               language-conditioned robot learning.},
  url       = {https://github.com/aryannzzz/grasp-research-showcase}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

*Research conducted as part of the GRASP project on robot learning and generalization.*

*For questions, open an issue or see the [documentation](docs/).*

</div>
