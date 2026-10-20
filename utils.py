"""Small helpers shared by every module: device choice, seeding, checkpoints, CSV."""
import csv
import os
import random
import sys

import numpy as np
import torch


class RunningMeanStd:
    """Keeps a running mean and variance of every observation feature so a network
    sees inputs of roughly mean 0, std 1 (neural nets learn much better that way)."""

    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = torch.tensor(1e-4, device=device)

    def update(self, x):
        self.update_from_moments(x.mean(0), x.var(0, unbiased=False), x.shape[0])

    def update_from_moments(self, b_mean, b_var, b_count):
        # Combine the old statistics with a new batch (parallel version of Welford's method).
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean = self.mean + delta * b_count / total
        m2 = self.var * self.count + b_var * b_count + delta.square() * self.count * b_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x, clip):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-clip, clip)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, s):
        self.mean, self.var, self.count = s["mean"], s["var"], s["count"]


def get_device(name="auto"):
    """Pick CUDA (NVIDIA), then MPS (Apple), then CPU, unless one is forced."""
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if name == "cpu":
        # 100 cars = small tensors: a few threads help, many threads just add overhead.
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    return torch.device(name)


def device_name(device):
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return device.type


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA


def utf8_stdout():
    """Windows consoles default to cp1252; corner names like 'Brünnchen' need UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def rng_state():
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save never corrupts the old checkpoint


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt_time(seconds):
    """420.5 -> '7:00.50'"""
    if seconds is None or not np.isfinite(seconds):
        return "--:--"
    m, s = divmod(float(seconds), 60.0)
    return f"{int(m)}:{s:05.2f}"
