# Project Overview: GRASP

**GRASP** — *Generalizable Robotic Action Models for Sensorimotor Policies*

> **Navigation**: [Main README](../README.md) · [Motivation](grasp_motivation.md) · [Research Directions](research_directions.md) · [Timeline](project_timeline.md)

---

## Abstract

Training robots to manipulate objects from limited demonstrations and generalize to new conditions is a fundamental challenge in robot learning. Standard end-to-end pixel-to-action models rely on large datasets to overcome spurious correlations — but robotics lacks internet-scale training data. This project investigates six complementary structural approaches that encode the right inductive biases without requiring thousands of demonstrations: image-conditioned latent representations (modified ACT), vision-only policies (LeRobot integration), explicit classical perception pipelines, gradient-based interpretability for VLAs (GradCAM), language-conditioned in-context learning (RoboPrompt), and multi-task policy training (multi-task ACT). Together, these experiments probe different angles on the core question of data-efficient robot generalization.

---

## What is GRASP?

GRASP is a research project investigating how robots can learn to manipulate objects from limited human demonstrations and generalize these skills to new environments and conditions.

The central research question is:

> How can we train sensorimotor policies that generalize from a small number of demonstrations, without relying on thousands of examples?

---

## The Problem

### Classical Robotics

Classical robotic systems depend on:
- Precise mathematical models of the robot and environment
- Controlled, structured environments where assumptions hold
- Deterministic planning pipelines (e.g., motion planning, grasp planners)

These systems work well in factories and structured industrial settings. They fail in unstructured environments because:
- Small model errors accumulate into large trajectory errors
- Unexpected objects, lighting, or clutter break pre-programmed logic
- Manual re-engineering is required for each new task

### Robot Learning (Imitation Learning)

Instead of hand-engineering behaviors, robot learning trains policies from **demonstrations**:

1. A human teleoperates the robot to perform a task
2. The observations (images, joint angles) and actions are recorded
3. A neural network is trained to predict actions from observations (behavioral cloning / imitation learning)

This is more flexible — the robot learns from experience, not rules. However, it introduces a new core challenge:

### The Data Scarcity Problem

Language models train on **trillions of tokens** scraped from the internet.
Vision models train on **billions of images** from ImageNet, LAION, and similar sources.
Robots have **neither** of these advantages.

- A demonstration requires physical execution by a human
- Setting up and executing demonstrations is slow, expensive, and environment-specific
- Demonstrations do not transfer easily across robots, tasks, or environments
- There is no "Internet of robot demonstrations" at scale

As a result:
- Robot learning datasets are typically 50–1000 demonstrations per task
- Models trained on small datasets tend to **overfit** to the specific conditions during data collection
- Generalization to new initial states, lighting, or object positions often fails

---

## The Research Goal

This project investigates approaches for **learning generalizable sensorimotor policies from fewer demonstrations**, without requiring internet-scale robotics data.

The hypothesis is that **structured inductive biases** can compensate for data scarcity:

1. **Object-centric representations**: Instead of treating all pixels equally, focus on task-relevant objects
2. **Visual conditioning of latent spaces**: Force the model to learn *what* the scene looks like, not just *where the robot is*
3. **Structured perception pipelines**: Use open-vocabulary detection before learning actions
4. **Interpretability methods**: Verify what the model actually attends to
5. **Language-conditioned policies**: Use LLMs to specify and condition robot behavior

---

## Experimental Scope

All experiments were conducted in:
- **MetaWorld** simulation (manipulation tasks: shelf-place, pick-place, handle-pull, etc.)
- **PyBullet** simulation (custom environments)
- **CoppeliaSim / PyRep** (robot simulation for RoboPrompt)
- **Real robot arm** (6-DOF arm with wrist camera for classical pipeline and RoboPrompt)

---

## Summary of Approaches

| Experiment | Core Idea | Outcome |
|---|---|---|
| ACT Architecture Modification | Add images to CVAE encoder | 27.8% better val loss |
| Vision-Only ACT (LeRobot) | Remove joint inputs entirely | Forced visual representation learning |
| Classical Pipeline | Perception → grasp planning | Validated in simulation, partially deployed on real robot |
| VLA-GradCAM | Attribution maps for language-conditioned policies | 85% improvement in instruction-specific saliency |
| RoboPrompt Integration | LLM-based task specification for real robot | Integration pipeline built and tested |
| Multi-task ACT | Single policy for multiple tasks | Multi-task data collected, training explored |

---

## Repository Contents

Each experiment has its own subfolder under `experiments/` with:
- A `README.md` explaining the experiment
- The source code used
- Configuration files where applicable
- Notebooks for interactive exploration where applicable

See [research_directions.md](research_directions.md) for an overview of each experimental thread.

---

## Technical Stack

| Component | Technology |
|---|---|
| Core framework | PyTorch 2.0+ |
| Robot learning framework | LeRobot (HuggingFace) |
| Simulation | MetaWorld, PyBullet, CoppeliaSim / PyRep |
| Vision models | ResNet18, CLIP ViT |
| Language models | GPT-4 (via API), CLIP text encoder |
| Data format | HDF5 (custom), HuggingFace Datasets |
| Visualization | GradCAM, custom saliency tools |
| Robot hardware | 6-DOF arm, SO100/SO101 (LeRobot platform) |
