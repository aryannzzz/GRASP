"""
PyBullet VLA Module - Tier 1 Reconstruction Pipeline

Modules:
- env: Randomized TableTopEnv for simulated pick-and-place
- data_collector: Collect expert demonstrations with layout seed isolation
- train: Train RefinedVLA with VLALoss
- evaluate: Held-out attribution benchmark, causality validation
- run_pipeline: Master pipeline orchestrating all phases
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

__version__ = "1.0.0"
