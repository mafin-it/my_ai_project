import os
import random
import numpy as np
import torch
from config import SEED


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def clear_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def count_lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def count_chars(path):
    with open(path, encoding="utf-8") as f:
        return len(f.read())


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)