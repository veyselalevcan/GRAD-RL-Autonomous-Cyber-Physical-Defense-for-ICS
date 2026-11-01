"""Repository-relative locations used by all GRAD-RL scripts and services.

Datasets are not shipped with the repository. Point GRADRL_DATA_DIR at the folder
that holds them (default: <repo>/data).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("GRADRL_DATA_DIR", ROOT / "data"))
MODEL_DIR = ROOT / "models"
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
