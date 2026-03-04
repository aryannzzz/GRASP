# Experiment 6: Multi-task ACT

**Goal**: Train ACT policies for multiple manipulation tasks and explore whether a single shared policy can succeed across diverse tasks.

---

## Motivation

### Single-Task Limitation

Standard ACT (and most imitation learning approaches) train a **separate policy per task**. This has fundamental scalability issues:

- Each task requires its own demonstration dataset (50–100+ demos)
- Each task requires its own training run
- No knowledge transfer between tasks
- Deploying N tasks requires N separate policies

### Multi-task Learning Hypothesis

If multiple manipulation tasks share underlying structure (grasping, placement, motion primitives), a **shared policy** should be able to:

1. **Learn faster** — shared representations amortize learning across tasks
2. **Generalize better** — diversity in training data acts as implicit data augmentation
3. **Scale** — one policy serves many tasks

This mirrors how humans generalize: we do not learn to pick up a coffee cup and a pen with two completely separate skills. We use common motor primitives that generalize.

### MetaWorld as a Testbed

MetaWorld provides a standardized benchmark of **50 manipulation tasks** with expert policies, making it ideal for multi-task exploration:
- All tasks use the same Sawyer robot arm
- Shared action space (end-effector delta commands)
- Consistent image observations
- Expert policies available for automated demonstration collection

---

## Tasks Studied

| Task | Description | Expert Policy |
|---|---|---|
| `pick-place` | Pick an object and place it at a target location | ✅ Available |
| `handle-pull` | Pull a handle toward the robot | ✅ Available |
| `reach` | Move end-effector to a target position | ✅ Available |
| `push` | Push an object to a target position | ✅ Available |
| `drawer-open` | Open a drawer | ✅ Available |
| `drawer-close` | Close a drawer | ✅ Available |
| `door-open` | Open a hinged door | ✅ Available |

---

## Pipeline

### Step 1: Per-Task Dataset Collection

For each task, collect 50–100 expert demonstrations using MetaWorld's built-in expert policies:

```python
# Example for pick-place
env = metaworld.MT1('pick-place-v3')
policy = SawyerPickPlaceV3Policy()
collect_demonstrations(env, policy, n_demos=50)
```

Each demonstration records:
- `states`: (T, 39) — joint positions, object positions, goal positions
- `images`: (T, H, W, 3) — RGB observations
- `actions`: (T, 4) — end-effector delta commands

### Step 2: Per-Task Training

Train a dedicated ACT policy for each task:

```bash
# Pick-place
python train_act.py --task pick-place --data demos_pickplace.hdf5

# Handle-pull
python train_act.py --task handle-pull --data demos_handlepull.hdf5
```

### Step 3: Multi-task Data Combination

Merge datasets from all tasks, adding a **task label** as an additional input:

```python
# Concatenate datasets
dataset = MultiTaskDataset([
    ('pick-place',   demos_pickplace.hdf5),
    ('handle-pull',  demos_handlepull.hdf5),
    ('reach',        demos_reach.hdf5),
    # ...
])
```

### Step 4: Shared Policy Training

Train a single ACT policy on the combined dataset:

```python
# Optional: task embedding layer to condition on task identity
model = MultiTaskACT(
    n_tasks=7,
    task_embed_dim=64,
    # ... other ACT hyperparams
)
```

---

## Files

```
experiments/multitask_act/
├── README.md                        # This file
├── code/
│   └── metaworld_act_complete.py    # Complete single-script pipeline:
│                                    #   data collection, training, evaluation
└── notebooks/
    ├── 01_pickplace_dataset.ipynb   # Pick-place data collection
    ├── 02_pickplace_train.ipynb     # Pick-place ACT training
    ├── 03_handlepull_dataset.ipynb  # Handle-pull data collection
    ├── 04_handlepull_train.ipynb    # Handle-pull ACT training
    ├── 05_multitask_act.ipynb       # Combined multi-task training
    ├── act-dataset.ipynb            # General dataset creation notebook
    └── act-train.ipynb              # General ACT training notebook
```

---

## Notebook Walkthrough

### `01_pickplace_dataset.ipynb`

Creates a demonstration dataset for the `pick-place-v3` MetaWorld task:
- Initializes MetaWorld environment
- Runs expert policy for N episodes
- Records states, actions, images, and rewards
- Saves to HDF5 format

### `02_pickplace_train.ipynb`

Trains an ACT policy on the pick-place dataset:
- Loads HDF5 dataset
- Configures ACT model (hidden_dim, n_heads, chunk_size)
- Training loop with loss tracking
- Saves checkpoint

### `03_handlepull_dataset.ipynb` and `04_handlepull_train.ipynb`

Same pipeline for the `handle-pull-v3` task. Demonstrates that the same code structure works across tasks with minimal changes.

### `05_multitask_act.ipynb`

The core multi-task experiment:
- Loads datasets from multiple tasks
- Optionally adds task conditioning (one-hot or learned embedding)
- Trains unified policy on combined data
- Evaluates on each task separately
- Analyzes task interference and transfer

---

## `metaworld_act_complete.py`

A standalone Python script (no notebooks required) that runs the complete multi-task pipeline:

```bash
# Collect + train + evaluate all tasks
python code/metaworld_act_complete.py \
    --tasks pick-place handle-pull reach push \
    --n_demos 50 \
    --epochs 300 \
    --output_dir results/multitask/
```

---

## Results and Observations

### Per-Task Results

| Task | Val Loss | Success Rate | Notes |
|---|---|---|---|
| pick-place | 0.142 | ~60% | Good with diverse demos |
| handle-pull | 0.118 | ~70% | Simpler trajectory |
| reach | 0.089 | ~85% | Easiest task |
| push | 0.161 | ~45% | Contact-rich, harder |

### Multi-task Policy

| Setting | Average Success |
|---|---|
| Per-task policies (oracle) | ~65% |
| Multi-task policy (shared) | ~45% |
| Multi-task + task conditioning | ~55% |

**Observations**:

1. **Negative transfer**: The shared policy performs worse than per-task policies in early training. Tasks with very different dynamics (reach vs. push) conflict in gradient updates.

2. **Task conditioning helps**: Adding a task embedding or one-hot task label significantly reduces interference.

3. **Data balance matters**: Overrepresented tasks dominate training. Balanced sampling per-task is essential.

4. **Simple tasks transfer**: Reach and pick-place show positive transfer. Contact-rich tasks (push, handle-pull) interfere more.

### Current Status

The experiment demonstrated the feasibility of multi-task ACT but did not achieve per-task performance parity. Key remaining challenges:
- Handling task-specific dynamics without catastrophic forgetting
- Optimal task balancing / curriculum
- Scaling to more diverse task sets

---

## Architecture Options Explored

### Option 1: Shared Policy (No Task Conditioning)

```
observations → [Shared ACT Policy] → actions
```
Simplest approach. Works for related tasks, fails for diverse ones.

### Option 2: Task-Conditioned Shared Policy

```
[task_id → task embedding]
         ↓
observations + task_embed → [Shared ACT Policy] → actions
```
Added to the decoder as an additional input token. More flexible.

### Option 3: Mixture of Experts (Explored)

```
observations → [Task Router] → experts[task] → actions
```
Each task has a specialized expert, shared backbone. Higher capacity, better compartmentalization.

---

## Requirements

```bash
conda create -n grasp python=3.10
conda activate grasp
pip install torch torchvision h5py numpy tqdm
pip install metaworld  # MetaWorld simulation
pip install jupyter    # For notebooks
```
