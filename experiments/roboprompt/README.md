# Experiment 5: RoboPrompt — In-Context Learning for Robots

**Goal**: Integrate LLM-based task specification with robot execution using in-context learning, enabling natural language-driven robot control without task-specific fine-tuning.

---

## Motivation

### The Task Specification Problem

Even if a robot can execute manipulation tasks, specifying *what* task to execute and *how* to execute it is cumbersome in traditional robotics. You need to:
- Manually write task-specific code or reward functions
- Hard-code task parameters
- Re-engineer for every new task

**Large Language Models (LLMs)** offer an alternative: describe the task in natural language, and the LLM figures out the task structure.

### In-Context Learning (ICL)

LLMs can perform new tasks by seeing a few examples in the prompt — this is **in-context learning**. Given:

```
Example 1: "Pick up the cube and place it in the bin" → [action sequence A]
Example 2: "Move object to the left" → [action sequence B]

New task: "Put the red block on top of the blue block" → ?
```

The LLM can generalize from the in-context examples to the new task without any gradient updates.

**RoboPrompt** applies this idea to robotics: use ICL to condition robot behavior on natural language.

---

## Approach

### Architecture Overview

```
Natural Language Instruction
           ↓
    [LLM with ICL Prompt]       ← in-context robot demonstrations
           ↓
  Discretized Action Plan       ← (trans_x, trans_y, trans_z, rot_x, rot_y, rot_z, gripper)
  [position bins + rotation bins]
           ↓
  [RoboPrompt→LeRobot Bridge]   ← decode bins to continuous pose
           ↓
  Continuous End-Effector Pose  ← (x, y, z, quaternion, gripper)
           ↓
  [Inverse Kinematics Solver]   ← compute joint angles
           ↓
  Joint Angle Commands
           ↓
    [Robot Arm Execution]        ← SO100 / SO101 / simulation
```

### Key Components

#### 1. In-Context Demonstration Formatting (`form_icl_demonstrations.py`)

Converts recorded robot demonstrations into LLM-readable format:
- Each demo: `[instruction, key_frames, action_sequence]`
- Formatted as structured text with discretized actions

#### 2. Inference Pipeline (`inference_pipeline.py`)

Full pipeline from instruction to robot command:
1. Load in-context examples
2. Construct LLM prompt
3. Call LLM API (GPT-4 / local model)
4. Parse discretized action output
5. Convert to robot commands

#### 3. RoboPrompt→LeRobot Bridge (`roboprompt_lerobot_bridge.py`)

Converts RoboPrompt's discretized action format to continuous robot poses:

```python
class RoboPromptToLeRobotBridge:
    def discretized_to_continuous_pose(self, discretized_action):
        """Convert position bins to XYZ + quaternion."""
        trans_indices = discretized_action[:3]
        rot_indices   = discretized_action[3:6]
        gripper_state = discretized_action[6]

        position = bounds_min + (trans_indices / 99.0) * (bounds_max - bounds_min)
        euler    = rot_indices * (360 / rotation_resolution)
        quat     = Rotation.from_euler('xyz', euler).as_quat()
        return position, quat, gripper_state
```

#### 4. Integrated Agent (`integrated_roboprompt_agent.py`)

Combines LLM task planning with LeRobot policy execution:
- High-level: LLM decomposes task into subtasks
- Low-level: Trained LeRobot policy executes each subtask
- Feedback loop: Perception system monitors progress

#### 5. HuggingFace Integration (`hf_pipeline.py`, `hf_dataset_loader.py`)

- Load robot demonstration datasets from HuggingFace Hub
- Compatible with LeRobot's data format
- Enables sharing of ICL demonstration sets

#### 6. Perception System (Online6: `perception_system.py`)

Computer vision for scene understanding on real robot:
- **ArUco marker detection**: Fast, reliable object tracking
- **Color segmentation**: Object detection by color
- **Depth-based object localization**: 3D position from RGB-D

---

## Files

```
experiments/roboprompt/
├── README.md                          # This file
└── code/
    ├── roboprompt_lerobot_bridge.py   # Action format conversion + IK bridge
    ├── integrated_roboprompt_agent.py # Full LLM + policy agent
    ├── inference_pipeline.py          # End-to-end inference
    ├── form_icl_demonstrations.py     # Format demos for LLM context
    ├── hf_dataset_loader.py           # Load datasets from HuggingFace
    ├── hf_pipeline.py                 # HuggingFace-compatible pipeline
    ├── simple_roboprompt_runner.py    # Simplified runner for testing
    ├── simplified_bridge.py           # Lightweight bridge implementation
    ├── test_roboprompt_hf.py          # Integration tests
    └── utils.py                       # Shared utilities
```

---

## Quickstart

### Install

```bash
pip install torch transformers lerobot
pip install openai scipy numpy opencv-python
```

### Run simple demo (simulation)

```python
from code.simple_roboprompt_runner import RoboPromptRunner

runner = RoboPromptRunner(use_sim=True)
result = runner.run_task("Pick up the red block and place it on the plate")
print(f"Success: {result.success}, Steps: {result.n_steps}")
```

### Build ICL prompt from demonstrations

```python
from code.form_icl_demonstrations import build_icl_prompt

# Load existing demonstrations
demos = load_demonstrations("path/to/demos.hdf5")

# Build prompt with 3 in-context examples
prompt = build_icl_prompt(
    task="Move the cup to the right side",
    demos=demos[:3],
    include_images=True
)

# Send to LLM
response = llm_client.chat(prompt)
actions = parse_discretized_actions(response)
```

### Convert LLM output to robot commands

```python
from code.roboprompt_lerobot_bridge import RoboPromptToLeRobotBridge

bridge = RoboPromptToLeRobotBridge(
    scene_bounds=[-0.3, -0.5, 0.6, 0.7, 0.5, 1.6],
    rotation_resolution=72
)

# LLM output: list of discretized actions
for action in llm_actions:
    position, quaternion, gripper = bridge.discretized_to_continuous_pose(action)
    joint_angles = bridge.continuous_pose_to_joint_angles(position, quaternion)
    robot.set_joint_angles(joint_angles)
```

---

## Experiments

### Simulation (PyRep / CoppeliaSim)

The integration was first validated in simulation using PyRep:
- SO100 and SO101 robot models
- Language-specified pick-and-place tasks
- ICL examples from recorded demonstrations

### Real Robot

Experiments on a real robot arm (SO101 platform via LeRobot):
- ArUco marker-based object detection
- Camera-calibrated workspace
- GPT-4 for action generation
- LeRobot's ACT policy for low-level execution

---

## Design Decisions

### Why discretized actions?

RoboPrompt uses discretized (binned) actions because:
- LLMs work natively with discrete tokens
- Continuous floating point values are hard for LLMs to generate accurately
- Discretization to 100 bins per dimension gives sufficient resolution for tabletop manipulation

### Why in-context learning and not fine-tuning?

- Fine-tuning requires large annotated datasets
- ICL requires only a few demonstrations (3–10 examples)
- ICL generalizes to new tasks at test time without re-training
- Compatible with any LLM (proprietary or open-source)

### Simulation-first strategy

All components were validated in simulation before running on the real robot. This:
- Eliminates hardware damage risk during development
- Allows rapid iteration without physical setup
- Makes bugs reproducible

---

## Status

| Component | Status |
|---|---|
| Discretized action encoding/decoding | ✅ Complete |
| LLM prompt construction | ✅ Complete |
| LeRobot bridge (simulation) | ✅ Complete |
| HuggingFace dataset integration | ✅ Complete |
| ArUco perception system | ✅ Complete |
| Real robot deployment | 🔄 In progress |
| Multi-step task planning | 🔄 Tested in simulation |

---

## References

- RoboPrompt: *In-Context Imitation Learning via Next-Token Prediction* (inspiration)
- LeRobot: [https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
- Brown et al. (2020). *Language Models are Few-Shot Learners.* (GPT-3 / ICL)
- PyRep: [https://github.com/stepjam/PyRep](https://github.com/stepjam/PyRep)
