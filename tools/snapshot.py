"""Render viewer frames off-screen and save them as PNGs (for checking the UI headlessly).

    python tools/snapshot.py --algo ga --load checkpoints/ga_seed0/latest.pt --steps 600
    python tools/snapshot.py --watch checkpoints/ga_seed0/latest.pt
Writes results/snapshot_<algo>_follow.png / _overview.png, or results/snapshot_watch.png.
"""
import argparse
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame  # noqa: E402

from config import Config  # noqa: E402
from main import build_trainer  # noqa: E402
from render import Renderer, draw_training_frame, run_watch  # noqa: E402
from utils import get_device, utf8_stdout  # noqa: E402


def snapshot_watch(path, device, out, frames):
    """Run the real watch loop and grab frame number `frames`, then quit."""
    flip = pygame.display.flip
    count = [0]

    def flip_and_capture():
        flip()
        count[0] += 1
        if count[0] == frames:
            pygame.image.save(pygame.display.get_surface(), os.path.join(out, "snapshot_watch.png"))
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    pygame.display.flip = flip_and_capture
    run_watch(path, device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ga", "ppo"], default="ga")
    p.add_argument("--load", default=None)
    p.add_argument("--watch", default=None, help="snapshot the --watch viewer for this checkpoint")
    p.add_argument("--steps", type=int, default=600, help="simulation steps (or watch frames) before the snapshot")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    utf8_stdout()
    os.makedirs(args.out, exist_ok=True)
    device = get_device(args.device)
    if args.watch:
        snapshot_watch(args.watch, device, args.out, args.steps)
        print(f"Saved {args.out}/snapshot_watch.png")
        return
    cfg = Config()
    tr = build_trainer(args.algo, cfg, device, 0)
    if args.load:
        tr.load(args.load)
    for _ in range(args.steps):
        tr.step()
    r = Renderer(tr.track, cfg.render, "snapshot")
    for _ in range(90):                      # let the smooth camera settle on the leader
        draw_training_frame(r, tr, "1x", 1 / 30)
    pygame.image.save(r.screen, os.path.join(args.out, f"snapshot_{args.algo}_follow.png"))
    r.overview = True
    draw_training_frame(r, tr, "1x", 1 / 30)
    pygame.image.save(r.screen, os.path.join(args.out, f"snapshot_{args.algo}_overview.png"))
    print(f"Saved snapshots to {args.out}/")


if __name__ == "__main__":
    main()
