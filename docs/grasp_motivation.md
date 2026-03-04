# Scientific Motivation: Why Generalization in Robotics is Hard

---

## Human vs. Robot Generalization

Humans can pick up a cup they have never seen before, in a kitchen they have never been in, even if the cup is partially hidden behind a plate. We do this effortlessly. Robots cannot.

The reason is not just limited hardware. It is about **how learning works**.

---

## The Spurious Correlation Problem

When a neural network is trained on limited data, it uses **every available signal** to minimize prediction error — including signals that are *irrelevant* to the task.

### Example

A robot is trained to pick up a red block on a white table.

The table is always white in training data. The block is always red. The camera angle is always fixed.

During evaluation:
- Change the table color → 40% drop in success rate
- Change lighting → 30% drop
- Change the background → 25% drop

The robot did not learn "pick up the red block". It learned "produce this motion sequence when I see this specific combination of pixels."

This is the **spurious correlation problem**: models learn associations that happen to be statistically reliable in the training data but do not reflect the true causal structure of the task.

### Why This Is Worse in Robotics

In image classification, spurious correlations can be corrected by collecting more diverse data. In robotics:

- Each additional demonstration requires physical execution
- Physical execution is time-limited (robot time is expensive)
- Demonstrations must be re-collected for each new environment
- The space of possible conditions (lighting, object position, clutter) is exponentially large

Therefore, **we cannot solve the spurious correlation problem by just collecting more data**.

---

## What Humans Do Differently

Humans generalize because we use structured representations:

### 1. Object-Centric Representation

Humans do not think about a scene as a grid of pixels. We think in terms of **objects**: *the cup*, *the table*, *my hand*.

When we pick up a cup, our mental representation is:
- "The cup is at position X"
- "My gripper needs to approach from above"
- "The task is complete when my hand closes around the rim"

This representation is **invariant** to the table color, the background, the lighting. It works in any kitchen because it is grounded in objects, not pixels.

### 2. Invariant Representations

Human perception is robust to irrelevant changes. We recognize a face in bright light, dim light, partial occlusion, different angles. We do not relearn face recognition for each new lighting condition.

This is because our visual system builds representations that are **invariant to domain-irrelevant factors** — a property called **disentanglement**.

### 3. World Models

Humans have an internal model of the physical world. We can imagine what will happen *before* we act. "If I push the cup to the right, it will slide off the table."

This enables **planning under uncertainty** without exhaustive data collection.

---

## The Standard End-to-End Approach and Its Limits

Modern robot learning typically trains an end-to-end neural network:

```
pixels → [neural network] → action
```

This is:
- ✅ Flexible: no manual feature engineering
- ✅ Powerful: can represent complex policies
- ❌ Data hungry: requires thousands of demonstrations to beat spurious correlations
- ❌ Opaque: hard to know what the model learned
- ❌ Fragile: small domain shifts cause large performance drops

---

## Structural Approaches to Generalization

The GRASP project investigates inductive biases that encode structured knowledge about the world into the learning process:

### Approach 1: Better Latent Representations

If the model's internal representation reflects *visual structure* (not just joint trajectories), it should generalize better across visual conditions.

**Experiment**: Modify the ACT CVAE encoder to process images — forcing the latent code to reflect what the robot sees, not just where it is.

### Approach 2: Removing Proprioceptive Bias

Joint angle sequences are highly specific to the exact initial configuration. A policy that memorizes joint trajectories cannot generalize to different start states.

**Experiment**: Train a vision-only ACT policy that receives no joint angle inputs, forcing the model to learn visual control.

### Approach 3: Structured Perception + Explicit Reasoning

Instead of learning to extract object information from pixels implicitly, use an explicit perception module (open-vocabulary object detection) and plan over the detected objects.

**Experiment**: Build a classical pipeline: detect objects → plan grasps → execute actions.

### Approach 4: Interpretability as a Verification Tool

Even if a model generalizes, we do not know *why* unless we can inspect its internal attention. Attribution methods allow us to verify that the model attends to task-relevant regions.

**Experiment**: Adapt GradCAM to vision-language-action models to produce saliency maps per action dimension, per instruction.

### Approach 5: Language as Structured Specification

Language provides a structured, compact representation of task goals. An LLM can specify complex task behavior in a way that generalizes across surface-level variations.

**Experiment**: Integrate LLM-based task specification with a robot policy using in-context learning.

---

## The Data Diversity Experiment: A Case Study

One of the clearest lessons from this project came from the ACT training experiments:

1. We trained ACT on demonstrations collected from a **single fixed initial state**
2. Training converged beautifully — validation loss 0.0931
3. Evaluation on **random initial states** → **0% success rate**

This demonstrated the spurious correlation problem in practice:
- The model had perfect training performance
- It had learned to memorize the single initial configuration
- It failed completely when even slight variation was introduced

When we re-collected data with **diverse initial states** (randomized object positions), success rate improved — with the same architecture and code.

**Lesson**: Data diversity is often the primary bottleneck in robot imitation learning, not model architecture or code correctness.

---

## Conclusion

The core challenge in robot learning is not compute or model capacity. It is:

1. **Limited data** that forces models to overfit
2. **Non-diverse data** that produces spurious correlations
3. **Opaque learned representations** that are hard to interpret or correct

Structured inductive biases — object-centric representations, visual conditioning, explicit perception pipelines, interpretability tools, and language specification — offer complementary paths to more generalizable robot policies under data constraints.

This is the scientific foundation of the GRASP project.
