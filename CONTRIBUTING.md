# Contributing to GRASP

Thank you for your interest in contributing to the GRASP research project. This document outlines how to contribute effectively.

---

## How to Contribute

### 1. Reporting Issues

If you find a bug, documentation error, or reproducibility problem:

1. Search existing issues to avoid duplicates
2. Open a new issue with:
   - A clear title describing the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Your environment (OS, Python version, GPU)
   - Relevant error messages or log output

### 2. Suggesting Experiments

If you want to propose a new research direction or experimental variation:

1. Open a discussion issue with the tag `experiment-proposal`
2. Describe:
   - The hypothesis being tested
   - The connection to existing experiments in the repo
   - Expected implementation complexity
   - What success would look like

### 3. Improving Documentation

Documentation improvements are always welcome:
- Fix typos, grammar, or unclear phrasing
- Add missing setup steps
- Improve code comments
- Add usage examples

For documentation changes, submit a pull request directly without opening an issue first.

### 4. Submitting Code Changes

**Scope**: This repository contains research code. Contributions should stay within the research scope of the existing experiments.

**What is in scope**:
- Bug fixes in existing experiment code
- Improved evaluation scripts
- Additional analysis scripts
- Better visualization utilities
- New experiment subfolders following the existing structure

**What is out of scope**:
- Changing experiment results already documented in READMEs
- Replacing core model files with substantially different implementations
- Adding dependencies that conflict with existing requirements

---

## Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/grasp-research-showcase.git
cd grasp-research-showcase

# Create a development environment
conda create -n grasp-dev python=3.10
conda activate grasp-dev

# Install core dependencies
pip install torch>=2.0 torchvision transformers
pip install lerobot metaworld pybullet
pip install opencv-python numpy scipy matplotlib
```

---

## Code Style

This is a research repository. We follow these minimal conventions:

- **Python**: PEP 8 formatting (use `black` or `ruff` if you like, but it is not enforced)
- **Notebooks**: Clear markdown cells explaining each step; restart and run all before submitting
- **Scripts**: Include a `if __name__ == "__main__":` block and argparse for configurable parameters
- **Documentation**: Write in the same style as existing READMEs (professional, direct, results-focused)

---

## Pull Request Process

1. Fork the repository
2. Create a new branch: `git checkout -b fix/your-description` or `feat/your-description`
3. Make your changes
4. Verify existing experiments still work (at minimum, run the quickstart commands in each affected README)
5. Update the relevant README if your changes affect documented behavior
6. Submit the pull request with:
   - A clear description of the change
   - The motivation and context
   - Any relevant results or outputs

---

## Adding a New Experiment

If you add a new experimental direction, follow this structure:

```
experiments/your_experiment_name/
├── README.md            ← Required: overview, motivation, methodology, results, how to run
├── code/
│   └── *.py             ← Implementation scripts
├── configs/             ← Optional: YAML/JSON config files
└── notebooks/           ← Optional: Jupyter notebooks for interactive exploration
```

The README must follow the standard format:
- **Overview** and research question
- **Motivation** explaining why this experiment matters
- **Methodology** with architecture or pipeline description
- **Results** with quantitative and qualitative findings
- **Key Insights** summarizing what was learned
- **How to Run** with exact commands

Also add an entry to the main `README.md` Research Directions table and Experiment Summaries section.

---

## Questions

Open a GitHub Issue with the `question` label for any questions about the project or codebase.
