"""Shared plumbing for the GA and PPO trainers.

Both trainers expose the same small interface, so the headless loop, the pygame
viewer and the comparison script can drive either one:
    step()          one environment step for all cars; returns True when a
                    generation (GA) or an update (PPO) just finished
    iteration       generation number / update number
    history         list of dicts, one per generation / update (the learning curve)
    display_state() what the renderer needs to draw
    save() / load() checkpoints
"""
import os
import time

from track import Track
from utils import load_checkpoint, rng_state, save_checkpoint, set_rng_state, set_seed


class BaseTrainer:
    algo = "base"

    def __init__(self, cfg, device, seed):
        self.cfg = cfg
        self.device = device
        self.seed = seed
        set_seed(seed)
        self.track = Track(cfg.track, device)
        self.history = []
        self.iteration = 0
        self._wall_start = time.time()
        self._wall_before = 0.0  # wall time accumulated before a resume

    # -- interface implemented by subclasses ------------------------------------------
    def step(self):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state):
        raise NotImplementedError

    def display_state(self):
        raise NotImplementedError

    # -- shared -------------------------------------------------------------------------
    @property
    def env_steps(self):
        return self.env.total_steps

    @property
    def wall_time(self):
        return self._wall_before + time.time() - self._wall_start

    def train_until(self, max_steps=None, max_iters=None, on_iter=None):
        """Run headless until a step budget or iteration count is reached."""
        while True:
            if self.step():
                if on_iter is not None:
                    on_iter(self)
                if max_steps is not None and self.env_steps >= max_steps:
                    return
                if max_iters is not None and self.iteration >= max_iters:
                    return

    def checkpoint_dir(self):
        return os.path.join(self.cfg.checkpoint_dir, f"{self.algo}_seed{self.seed}")

    def save(self, path=None, numbered=False):
        state = {
            "algo": self.algo,
            "config": self.cfg.to_dict(),
            "seed": self.seed,
            "iteration": self.iteration,
            "env_steps": self.env_steps,
            "wall_time": self.wall_time,
            "history": self.history,
            "rng": rng_state(),
            **self.state_dict(),
        }
        if path is None:
            path = os.path.join(self.checkpoint_dir(), "latest.pt")
            if numbered:
                save_checkpoint(state, os.path.join(self.checkpoint_dir(), f"{self.algo}_{self.iteration:05d}.pt"))
        save_checkpoint(state, path)
        return path

    def load(self, path):
        state = load_checkpoint(path, self.device)
        if state["algo"] != self.algo:
            raise ValueError(f"{path} is a {state['algo']} checkpoint, not {self.algo}")
        self.iteration = state["iteration"]
        self.history = state["history"]
        self.env.total_steps = state["env_steps"]
        self._wall_before = state["wall_time"]
        self._wall_start = time.time()
        self.load_state_dict(state)
        set_rng_state(state["rng"])
        return state
