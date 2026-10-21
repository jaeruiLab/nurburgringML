"""100 AI cars learn the Nürburgring Nordschleife.

Examples
  python main.py                                # GA with the pygame viewer
  python main.py --algo ppo                     # PPO with the viewer
  python main.py --algo ga --headless --generations 50
  python main.py --algo ppo --headless --max-steps 20e6
  python main.py --algo ga --load checkpoints/ga_seed0/latest.pt     # resume training
  python main.py --watch checkpoints/ppo_seed0/latest.pt             # watch the best driver
  python main.py --cars 1000 --device cuda      # a bigger population / more parallel cars
"""
import argparse
import math

from config import Config
from utils import device_name, fmt_time, get_device, load_checkpoint, utf8_stdout


def build_trainer(algo, cfg, device, seed):
    if algo == "ga":
        from ga import GATrainer
        return GATrainer(cfg, device, seed)
    from ppo import PPOTrainer
    return PPOTrainer(cfg, device, seed)


def progress_line(tr):
    h = tr.history[-1]
    L = tr.track.length
    lap = fmt_time(h["best_lap_time_ever"])
    if tr.algo == "ga":
        return (f"gen {h['iteration']:4d} | steps {h['env_steps'] / 1e6:7.2f}M | best {h['best_distance']:7.0f} m "
                f"({100 * h['best_distance'] / L:5.1f}%) | ever {h['best_distance_ever']:7.0f} m | "
                f"mean {h['mean_distance']:6.0f} m | laps {h['laps_completed']:3d} | best lap {lap} | "
                f"sigma {h['sigma']:.3f} | {h['wall_time']:7.1f} s")
    return (f"upd {h['iteration']:5d} | steps {h['env_steps'] / 1e6:7.2f}M | ret {h['mean_return']:8.1f} | "
            f"dist {h['mean_distance']:6.0f} m | ever {h['best_distance_ever']:7.0f} m "
            f"({100 * h['best_distance_ever'] / L:5.1f}%) | lap {lap} | pi {h['policy_loss']:+.3f} "
            f"v {h['value_loss']:8.2f} | ent {h['entropy']:+.2f} | kl {h['approx_kl']:.4f} | {h['wall_time']:7.1f} s")


def run_headless(tr, max_steps=None, max_iters=None, save_every=None, save_log_ratio=None, quiet=False):
    save_every = save_every or (tr.cfg.save_every_generations if tr.algo == "ga" else tr.cfg.save_every_updates)
    if tr.iteration == 0:
        tr.save(numbered=True)          # the untrained starting point (record.py replays it as generation 1)
    next_log_save = [max(1, tr.env_steps) * (save_log_ratio or 1)]

    def on_iter(t):
        if not quiet:
            print(progress_line(t), flush=True)
        if t.iteration % save_every == 0:
            t.save(numbered=True)
        elif save_log_ratio and t.env_steps >= next_log_save[0]:
            # Extra checkpoints at log-spaced step counts (x ratio each time), so record.py can
            # find a snapshot close to any env-step count without saving every update.
            t.save(numbered=True)
        if save_log_ratio and t.env_steps >= next_log_save[0]:
            next_log_save[0] = t.env_steps * save_log_ratio

    try:
        tr.train_until(max_steps=max_steps, max_iters=max_iters, on_iter=on_iter)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    path = tr.save()
    print(f"Saved checkpoint: {path}")


def load_policy(path, device):
    """Return (algo, policy_fn, checkpoint) where policy_fn(obs) -> deterministic actions."""
    state = load_checkpoint(path, device)
    cfg = Config.from_dict(state["config"])
    if state["algo"] == "ga":
        from ga import ga_policy_from_checkpoint
        return state["algo"], ga_policy_from_checkpoint(state, device), state, cfg
    from ppo import ppo_policy_from_checkpoint
    return state["algo"], ppo_policy_from_checkpoint(state, cfg, device), state, cfg


def watch_headless(path, device, laps=1):
    """Drive one car with the saved policy and print how far it gets / its lap time."""
    from env import VecEnv
    from track import Track
    algo, policy, state, cfg = load_policy(path, device)
    track = Track(cfg.track, device)
    env = VecEnv(track, cfg, n_cars=1, device=device)
    obs = env.reset_all()
    for _ in range(env.max_steps):
        obs, _, term, trunc, info = env.step(policy(obs))
        if bool(term[0] | trunc[0]):
            break
    lap = env.lap_time[0].item()
    print(f"{algo.upper()} policy from {path}: distance {env.best_dist[0].item():.0f} m "
          f"/ {track.length:.0f} m, lap time {fmt_time(lap) if math.isfinite(lap) else 'no lap'}")


def main():
    p = argparse.ArgumentParser(description="100 AI cars learn the Nürburgring Nordschleife (GA vs PPO).")
    p.add_argument("--algo", choices=["ga", "ppo"], default="ga")
    p.add_argument("--headless", action="store_true", help="train without a window (fastest)")
    p.add_argument("--load", metavar="PATH", help="resume training from a checkpoint")
    p.add_argument("--watch", metavar="PATH", help="watch a saved GA brain / PPO policy drive deterministically")
    p.add_argument("--seed", type=int, default=None, help="random seed (default 0, or the checkpoint's seed with --load)")
    p.add_argument("--cars", type=int, default=None,
                   help="number of cars (GA population / PPO parallel envs; default 100). "
                        "Above ~500 a GPU is usually faster than the CPU")
    p.add_argument("--device", default=None, help="auto | cpu | cuda | mps (default: config.device = auto)")
    p.add_argument("--max-steps", type=float, default=None, help="stop after this many env steps (headless)")
    p.add_argument("--generations", type=int, default=None, help="GA: stop after this many generations (headless)")
    p.add_argument("--updates", type=int, default=None, help="PPO: stop after this many updates (headless)")
    p.add_argument("--save-every", type=int, default=None, help="checkpoint every N generations / updates")
    p.add_argument("--save-log-ratio", type=float, default=None,
                   help="also checkpoint whenever env steps grow by this factor, e.g. 1.1 (used by record.py)")
    args = p.parse_args()
    utf8_stdout()

    cfg = Config()
    checkpoint = load_checkpoint(args.load, "cpu") if args.load else None
    if args.seed is None:
        # Resuming keeps the run's own seed, so it also keeps saving into the same checkpoint folder.
        args.seed = checkpoint.get("seed", 0) if checkpoint is not None else 0
    cfg.seed = args.seed
    if args.cars:
        cfg.env.n_cars = args.cars
    elif checkpoint is not None:
        cfg.env.n_cars = checkpoint["config"]["env"]["n_cars"]   # resume with the same number of cars
    device = get_device(args.device or cfg.device)
    print(f"Device: {device_name(device)}, {cfg.env.n_cars} cars")
    if device.type != "cpu" and not args.device and cfg.env.n_cars < 500:
        print(f"Tip: with {cfg.env.n_cars} cars the CPU is often faster than a GPU (per-operation overhead "
              "dominates; the GPU wins above ~500 cars). Compare with: python tools/bench.py --cars N")

    if args.watch:
        if args.headless:
            watch_headless(args.watch, device)
        else:
            from render import run_watch
            run_watch(args.watch, device)
        return

    if checkpoint is not None:
        algo = checkpoint["algo"]
        if algo != args.algo:
            print(f"Checkpoint is {algo}; switching --algo to {algo}.")
            args.algo = algo
    tr = build_trainer(args.algo, cfg, device, args.seed)
    if args.load:
        tr.load(args.load)
        print(f"Resumed {args.algo} from {args.load} at iteration {tr.iteration}, {tr.env_steps:,} env steps")

    if args.headless:
        max_iters = args.generations if args.algo == "ga" else args.updates
        run_headless(tr, max_steps=int(args.max_steps) if args.max_steps else None,
                     max_iters=max_iters, save_every=args.save_every, save_log_ratio=args.save_log_ratio)
    else:
        from render import run_training_viewer
        run_training_viewer(tr)


if __name__ == "__main__":
    main()
