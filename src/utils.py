"""


Utility functions for:
- setting random seed
- loading CSV data
- saving/loading RL models
"""

import random
import numpy as np
import pandas as pd
import torch
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.read_csv(csv_path)
    return df

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}")

def load_model(cls, path, env):
    if not os.path.exists(path + ".zip"):
        raise FileNotFoundError(f"Model not found at {path}.zip")
    model = cls.load(path, env=env)
    print(f"Model loaded from {path}")
    return model
