"""Measure simulation speed on each available device.

    python tools/bench.py            # 100 cars
    python tools/bench.py --cars 1000

Small batches (100 cars, tiny networks) are dominated by per-operation overhead, so the
CPU is often faster than a GPU here; with thousands of cars the GPU pulls ahead.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from brain import PopulationBrain  # noqa: E402
from config import Config  # noqa: E402
from env import VecEnv  # noqa: E402
from track import Track  # noqa: E402
from utils import device_name, get_device  # noqa: E402


def bench(device, n_cars, steps):
    cfg = Config()
    track = Track(cfg.track, device)
    env = VecEnv(track, cfg, n_cars=n_cars, device=device, random_start=True)
    brain = PopulationBrain(n_cars, (13, 16, 16, 2), device)
    obs = env.reset_all()

    def run(n):
        nonlocal obs
        for _ in range(n):
            obs, *_ = env.step(brain(obs), auto_reset=True)
        if device.type == "cuda":
            torch.cuda.synchronize()

    run(30)
    t0 = time.perf_counter()
    run(steps)
    dt = time.perf_counter() - t0
    return steps / dt, steps * n_cars / dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cars", type=int, default=100)
    p.add_argument("--steps", type=int, default=500)
    args = p.parse_args()
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devices.append("mps")
    for d in devices:
        dev = get_device(d)
        sps, csps = bench(dev, args.cars, args.steps)
        print(f"{device_name(dev):40s} {args.cars:5d} cars: {sps:7.0f} steps/s = {csps:10,.0f} car-steps/s")


if __name__ == "__main__":
    main()
