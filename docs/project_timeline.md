# Project Timeline

**GRASP — Generalizable Robotic Action Models for Sensorimotor Policies**

This document traces how the research evolved, the decisions made at each phase, and the technical and conceptual transitions between experiments. It is intended to give readers a sense of the research process, not just the final results.

---

## Phase 1 — ACT Architecture Exploration

**Focus**: Understand and modify the ACT (Action Chunking Transformer) architecture

**Starting point**: The original ACT paper (Zhao et al., 2023) demonstrated impressive manipulation performance using a CVAE architecture. The key observation was that the CVAE encoder receives only joint states and action sequences — no visual information. The question was whether conditioning the latent code on vision would improve generalization.

**Work done**:
- Implemented standard ACT from scratch in PyTorch against the original codebase as reference
- Instrumented the training loop extensively to track gradients, loss components, and latent statistics
- Modified the CVAE encoder to accept ResNet18 image features in addition to joint states and action sequences
- Ran systematic ablations: standard vs modified, with varying KL weights and learning rates

**Critical bugs discovered and fixed**:
1. `z = randn()` at inference time — model was using random noise during evaluation, causing inconsistent actions. Fixed to `z = zeros()`.
2. `query_frequency = 1` — the action chunking mechanism was not being used effectively. Fixed to `query_frequency = 100`, which produced a 66% improvement in validation loss.
3. Array bounds error in action chunk copy — fixed with `min(action_len, chunk_size)`.

**Result**: Modified ACT achieved 27.8% lower validation loss than standard ACT. Both models showed 0% evaluation success — later traced to a data diversity problem (see Phase 2).

**Key learning**: Debugging ML systems requires instrumenting both the training and evaluation pipelines separately. The evaluation bug (random z sampling) was completely invisible in training metrics.

---

## Phase 2 — Data Diversity Investigation

**Focus**: Root-cause analysis of the 0% evaluation success rate

**Trigger**: Both the standard and modified ACT models trained to low validation loss but achieved 0% success in evaluation with randomized initial states. This was the central mystery of Phase 1.

**Investigation approach**:
1. Visualized training data joint angle distributions — found essentially zero variance across all 100 demonstrations (std ~1e-17)
2. Compared training initial states vs evaluation initial states — discovered they were drawn from completely different distributions
3. Designed controlled experiments: fixed-initial-state training vs randomized-initial-state training

**Root cause**: All training demonstrations were collected from a single fixed initial state. The expert policy was always called with the same random seed, so every episode started from the same configuration. The model memorized this configuration rather than learning a generalizable policy.

**Fix**: Re-implemented the demonstration collection script to explicitly randomize the initial object position using MetaWorld's built-in random seed mechanism. Collected 100 diverse demonstrations with object position std of 0.054 (vs. 1e-17 before).

**Impact**: This phase produced the most important empirical finding of the project: **data diversity, not model architecture, was the primary bottleneck**.

---

## Phase 3 — Classical Perception Pipeline

**Focus**: Structured baseline to quantify what explicit perception can achieve without any learned policy

**Motivation**: After the ACT experiments, the team wanted to understand the ceiling performance of fully structured (non-learned) systems on the same class of tasks, to establish a meaningful baseline.

**Work done**:
- Implemented open-vocabulary object detection pipeline using CLIP-based detectors
- Built full camera calibration system: checkerboard intrinsics calibration, extrinsics relative to robot base frame
- Implemented pixel-to-world 3D coordinate transform using camera intrinsics and depth estimates
- Built geometric grasp planner: top-down approach trajectory, descent, grasp configuration
- Validated complete pipeline in simulation on the pick-and-place benchmark
- Began real robot deployment: calibrated real camera, tested pipeline on physical arm

**Key notebooks**: `Final_OpenVocabPickPlace_Classical_v2_(5)_(4).ipynb` contains the complete documented walkthrough.

**Result**: ~90% success in simulation on fixed-position objects. New objects are handled by changing the text prompt — no re-training needed. Real robot deployment was partial (camera calibrated, pipeline tested on simple cases).

**Key learning**: The classical pipeline revealed that a 5mm error in camera calibration at 50cm range produces ~1cm placement error — well above the manipulation tolerance. Camera calibration quality is a critical bottleneck for real-robot deployment of any perception-based pipeline.

---

## Phase 4 — GradCAM Interpretability for VLAs

**Focus**: Develop attribution methods to inspect what vision-language-action models attend to

**Motivation**: As the project moved toward more complex VLA architectures, a fundamental question emerged: when a policy fails, is it because the model is attending to the wrong image regions? Without interpretability tools, this question cannot be answered. Attribution methods also help *verify* that architecture improvements (like image-conditioned CVAE) are actually improving the right things.

**Phase 4a — RL GradCAM**:
- Implemented GradCAM for CNN-based RL policies (DQN on Atari, custom CNN on FrozenLake)
- Validated that saliency maps correctly highlight task-relevant regions (ball position in Pong, ice vs hole in FrozenLake)
- Built visualization infrastructure reused in later phases

**Phase 4b — VLA GradCAM attempt (failure)**:
- Extended GradCAM to CLIP-based VLA model using mean pooling
- Discovered that mean pooling produces uniform gradients: every patch receives the same gradient weight (1/N)
- Correlation between saliency maps from different instructions: 0.896 (nearly identical — meaningless)

**Phase 4c — Architecture diagnosis**:
- Traced the failure to the mathematical structure of mean pooling: `∂(mean(patches))/∂(patch_k) = 1/N` for all k
- Confirmed via numerical diagnostics that gradient standard deviation across patches was essentially zero
- Identified cross-attention pooling as the natural fix

**Phase 4d — Attention pooling fix**:
- Implemented `AttentionPooling` module: text features query visual patches via learned Q/K/V projections
- Cross-instruction saliency correlation reduced from 0.896 to **0.134** (85% improvement)
- Validated across 3 diverse scenes (robot manipulation, kitchen, outdoor)

---

## Phase 5 — Vision-Only ACT (Modified LeRobot)

**Focus**: Remove proprioceptive inputs from ACT to force visual representation learning

**Motivation**: Building on the data diversity findings (Phase 2) and the interpretability work (Phase 4), the hypothesis emerged that joint angle inputs allow the model to form a shortcut: memorize joint-angle trajectories without learning to visually understand the scene. Removing these inputs would force visual grounding.

**Integration choice**: Rather than modifying the standalone codebase from Phase 1, this was implemented directly inside the LeRobot (HuggingFace) framework. This provided:
- Access to LeRobot's standardized data pipeline
- HuggingFace Hub model sharing
- Reproducible training with modern tooling
- A cleaner interface for future extensions

**Work done**:
- Created `ModifiedACTConfig` extending `ACTConfig` with `vae_encoder_use_images` parameter
- Modified `ACTPolicy.select_action()` and `ACTPolicy.forward()` to exclude `OBS_STATE` from inputs
- Modified the internal `ACT` model to optionally include image features in the VAE encoder sequence
- Created `ModifiedACTImageProcessor` for image-only preprocessing
- Validated HuggingFace Hub push/load workflow

---

## Phase 6 — RoboPrompt: Language-Conditioned Robot Learning

**Focus**: Use LLM in-context learning to specify robot task behaviors from natural language

**Motivation**: Even with strong policies, specifying *what task to do* without manual engineering is a separate challenge. LLMs offer a path: given a handful of in-context examples and a natural language description, can an LLM generate executable robot action sequences?

**Integration work**:
- Built `RoboPromptToLeRobotBridge` to convert discretized LLM-generated actions to continuous poses using IK
- Implemented in-context example formatting from recorded demonstrations
- Built `IntegratedRoboPromptAgent` combining LLM planning with LeRobot execution
- Implemented perception system using ArUco markers for real-robot object tracking
- Integrated with SO101 platform via LeRobot interface
- Validated pipeline in simulation (CoppeliaSim via PyRep)

**Design challenge**: LLMs output text tokens; robot actions are continuous floating-point vectors. The discretization scheme (100 bins per position dimension, 72 bins per rotation dimension) bridged this gap, enabling LLMs to generate action plans as discrete integer sequences parseable from text.

---

## Phase 7 — Multi-task ACT

**Focus**: Train a single ACT policy across multiple manipulation tasks

**Motivation**: Separate per-task policies do not scale. If robotics is to move toward versatile manipulation systems, policies must handle multiple tasks from shared representations. The multi-task experiments explored what combination of data mixing, task conditioning, and architecture design is needed first.

**Work done**:
- Built per-task demonstration collection for 7 MetaWorld tasks
- Trained and evaluated per-task ACT as baseline
- Combined datasets and trained unconditioned shared policy
- Added task ID as one-hot and learned embedding to the decoder
- Analyzed task interference patterns (which tasks help each other, which interfere)

**Finding**: Simple tasks (reach, pick-place) transfer positively. Contact-rich tasks with distinct force profiles (push, handle-pull) interfere in the shared representation. Task conditioning (+10pp) partially mitigates this but does not close the gap.

---

## Current Status and Next Steps

| Phase | Status |
|:---|:---|
| Phase 1: ACT architecture | ✅ Complete |
| Phase 2: Data diversity | ✅ Complete |
| Phase 3: Classical pipeline | 🔄 Simulation complete, real robot ongoing |
| Phase 4: GradCAM for VLAs | ✅ Complete |
| Phase 5: Vision-only ACT | ✅ Complete |
| Phase 6: RoboPrompt | 🔄 Simulation complete, real robot ongoing |
| Phase 7: Multi-task ACT | 🔄 Baselines established, full convergence ongoing |

**Next priorities**:
1. Object-centric representations (slot attention / object tokens)
2. Foundation model vision encoders (DINOv2, SAM) as frozen backbones
3. Real-robot evaluation of vision-only ACT and GradCAM pipeline
4. Multi-step language-conditioned planning via RoboPrompt

---

## Timeline Summary

```
Late 2024
  │
  ├── Phase 1: ACT from scratch, standard vs modified CVAE
  │   └── Discovered 3 critical bugs, fixed them all
  │
  ├── Phase 2: Data diversity root-cause analysis
  │   └── Identified fixed initial state as root cause
  │
Early 2025
  │
  ├── Phase 3: Classical perception pipeline
  │   └── Simulation validation, camera calibration, real robot start
  │
  ├── Phase 5: Vision-only ACT (LeRobot integration)
  │   └── HuggingFace-compatible module complete
  │
Early 2026
  │
  ├── Phase 4: GradCAM for VLAs
  │   ├── RL GradCAM baseline
  │   ├── Discovered mean-pooling failure
  │   └── Attention pooling fix, 85% correlation reduction
  │
  ├── Phase 6: RoboPrompt integration
  │   └── LLM-to-robot bridge in simulation
  │
  └── Phase 7: Multi-task ACT
      └── Baselines established, research ongoing
```
