"""Helpers shared by all approaches: seeds, device, saving results."""

import json
import random
from pathlib import Path

import numpy as np
import torch
import csv

SEED = 42                     # seed for reproducibility

def set_seed(seed=SEED):
    """Fix the random seeds of Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                     # also seeds CUDA and MPS
    torch.backends.cudnn.deterministic = True   # deterministic algorithms on CUDA
    torch.backends.cudnn.benchmark = False


def get_device():
    """Return the best available device: cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def save_json(data, path):
    """Save a dictionary (hyperparameters, metrics) to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def save_csv(rows, path):
    """Save a list of dictionaries (one per row) to a CSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)