"""Record replays of saved GA / PPO checkpoints for the browser viewer in web/.

    python record.py                                   # defaults below
    python record.py --ga checkpoints/ga_seed1 --ppo checkpoints/ppo_seed1 --gens 1,5,20,50,100,final

Make the checkpoints first (one per GA generation, log-spaced ones for PPO):
    python main.py --algo ga  --headless --seed 1 --generations 100 --save-every 1
    python main.py --algo ppo --headless --seed 1 --max-steps 150e6 --save-every 1000000000 --save-log-ratio 1.1

What it does
  * GA generation g: loads the population as it was at the start of generation g (checkpoint
    ga_{g-1}.pt) and drives all its cars from the start grid until every car has crashed or
    finished, exactly like training did (the replay is deterministic, and its best distance is
    checked against the training log).
  * PPO: for each GA replay, picks the PPO checkpoint trained on the closest number of env steps
    and drives the same number of cars from the start grid. PPO has ONE policy, so the cars differ
    only through the random exploration noise in their actions (seeded, so the replay is
    reproducible). The noise-free policy is driven once more on its own and reported separately.
  * Writes into web/data/: track.json, replays.json (index + per-car stats), one .i16.gz file per
    replay, and copies of the learning-curve plots from results/.

Replay file format (<id>.i16.gz, gzip of little-endian int16)
  Car i has end_frames[i] + 1 frames of (x, y, heading). The file holds three "planes": every car's
  x values one car after another, then every car's y values, then every car's headings (grouping
  similar numbers compresses ~20% better than interleaving them).
  x, y are in units of `quant` meters (0.2 m) and stored as second differences (velocity changes,
  almost always tiny numbers that compress very well); heading is stored as first differences
  of round(angle / pi * 32767) with int16 wrap-around. web/js/decode.js undoes this.
"""
import argparse
import csv
import gzip
import json
import math
import os
import re
import shutil

import numpy as np
import torch

from config import Config
from env import VecEnv
from ga import fitness
from ppo import ActorCritic
from track import Track
from utils import RunningMeanStd, get_device, load_checkpoint, utf8_stdout

QUANT = 0.2          # meters per position unit
HEADING_SCALE = 32767 / math.pi


# ----------------------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------------------
def wrap_i16(a):
    return ((a + 32768) % 65536) - 32768


def encode_car(xy_q, h_q):
    """xy_q: (F, 2) int positions, h_q: (F,) int headings -> (F, 3) int16 deltas."""
    d = np.zeros((len(h_q), 3), dtype=np.int64)
    v = np.diff(xy_q, axis=0, prepend=xy_q[:1] * 0)             # first differences (frame 0 = absolute)
    d[:, :2] = np.diff(v, axis=0, prepend=v[:1] * 0)              # second differences
    d[:, 2] = wrap_i16(np.diff(h_q, prepend=0))
    if np.abs(d[:, :2]).max(initial=0) > 32767:
        raise ValueError("position delta does not fit in int16")
    return d.astype(np.int16)


def decode_car(d):
    """Inverse of encode_car (Python twin of web/js/decode.js, used by the tests)."""
    d = d.astype(np.int64)
    xy = np.cumsum(np.cumsum(d[:, :2], axis=0), axis=0)
    h = wrap_i16(np.cumsum(d[:, 2]))
    return xy * QUANT, h / HEADING_SCALE


def encode_replay(frames, end_frames):
    """frames: (F, N, 3) float [x, y, yaw]; end_frames: (N,) last stored frame per car -> gzip bytes."""
    parts = []
    for i, end in enumerate(end_frames):
        f = frames[: end + 1, i]
        xy_q = np.round(f[:, :2] / QUANT).astype(np.int64)
        yaw = (f[:, 2] + math.pi) % (2 * math.pi) - math.pi
        h_q = np.round(yaw * HEADING_SCALE).astype(np.int64)
        parts.append(encode_car(xy_q, h_q))
    planes = np.ascontiguousarray(np.concatenate(parts).T)           # (3, total frames)
    return gzip.compress(planes.astype("<i2").tobytes(), compresslevel=9, mtime=0)


def decode_replay(data, end_frames):
    """Inverse of encode_replay -> list of (xy (F, 2) meters, heading (F,) radians) per car."""
    planes = np.frombuffer(gzip.decompress(data), dtype="<i2").reshape(3, -1)
    cars, o = [], 0
    for end in end_frames:
        n = end + 1
        cars.append(decode_car(planes[:, o:o + n].T))
        o += n
    return cars


# ----------------------------------------------------------------------------------
# Driving a recorded episode
# ----------------------------------------------------------------------------------
@torch.no_grad()
def drive(env, act_fn, fps, max_seconds):
    """Run every car from the grid until all are done; return frames and per-car results."""
    every = max(1, round(1.0 / (fps * env.dt)))                     # physics steps per recorded frame
    max_steps = min(env.max_steps, int(max_seconds / env.dt))
    obs = env.reset_all()
    frames = [torch.stack([env.x, env.y, env.yaw], 1).cpu()]
    alive = [env.alive.cpu()]
    for step in range(1, max_steps + 1):
        obs, *_ = env.step(act_fn(obs))
        if step % every == 0:
            frames.append(torch.stack([env.x, env.y, env.yaw], 1).cpu())
            alive.append(env.alive.cpu())
            if not bool(alive[-1].any()):
                break
    frames = torch.stack(frames).numpy().astype(np.float64)       # (F, N, 3)
    alive = torch.stack(alive).numpy()                            # (F, N)
    F = len(frames)
    # Last stored frame per car: the first frame where it is no longer driving (it is frozen
    # there after a crash / finish), or the final frame if it was still driving.
    dead = ~alive
    end = np.where(dead.any(0), dead.argmax(0), F - 1)
    return frames, end, (~alive[-1]).astype(bool)


def car_results(env, cfg, track):
    lap = env.lap_time.clone()
    score = fitness(env.best_dist, env.speed_sum, env.ep_steps, lap, cfg, track.length)
    dist = env.best_dist.clamp(0, track.length)
    lap = [round(float(t), 2) if math.isfinite(t) else None for t in lap.tolist()]
    return score.cpu().numpy(), dist.cpu().numpy(), lap


def replay_entry(rid, algo, title, iteration, env_steps, frames, end, done, score, dist, lap, fps, out_dir):
    data = encode_replay(frames, end)
    with open(os.path.join(out_dir, f"{rid}.i16.gz"), "wb") as f:
        f.write(data)
    best = int(np.argmax(score))
    laps = [t for t in lap if t is not None]
    finished = [t is not None for t in lap]
    return {
        "id": rid, "algo": algo, "title": title, "iteration": iteration, "env_steps": int(env_steps),
        "file": f"{rid}.i16.gz", "bytes": len(data), "n_cars": len(end), "n_frames": int(frames.shape[0]),
        "duration": round((frames.shape[0] - 1) / fps, 2),
        "end_frames": [int(e) for e in end],
        # "crashed" = stopped before the end without finishing a lap.
        "crashed": [bool(d and not fi) for d, fi in zip(done, finished)],
        "finished": finished,
        "score": [round(float(s), 1) for s in score],
        "distance": [round(float(d), 1) for d in dist],
        "lap_time": lap,
        "best": best,
        "best_distance": round(float(dist.max()), 1),
        "best_lap": min(laps) if laps else None,
        "laps_completed": len(laps),
    }


# ----------------------------------------------------------------------------------
# GA and PPO replays
# ----------------------------------------------------------------------------------
def numbered(dirpath, prefix):
    """{iteration: path} for checkpoints like ga_00012.pt."""
    out = {}
    for name in os.listdir(dirpath):
        m = re.fullmatch(rf"{prefix}_(\d+)\.pt", name)
        if m:
            out[int(m.group(1))] = os.path.join(dirpath, name)
    return dict(sorted(out.items()))


def record_ga(path, generation, track_cache, device, args, history_by_gen):
    state = load_checkpoint(path, device)
    cfg = Config.from_dict(state["config"])
    track = track_cache(cfg)
    genes = state["genes"].to(device)
    from brain import PopulationBrain
    brain = PopulationBrain(genes.shape[0], state["layer_sizes"], device, genes=genes)
    mean, std = (t.to(device) for t in state["norm"])
    clip = state.get("obs_clip")

    def act(obs):
        if clip is not None:
            obs = ((obs - mean) / std).clamp(-clip, clip)
        return brain(obs)

    env = VecEnv(track, cfg, n_cars=genes.shape[0], device=device)
    frames, end, done = drive(env, act, args.fps, args.max_seconds)
    score, dist, lap = car_results(env, cfg, track)
    entry = replay_entry(f"ga_g{generation:03d}", "ga", f"Generation {generation}", generation,
                         state["env_steps"], frames, end, done, score, dist, lap, args.fps, args.out)
    h = history_by_gen.get(generation)
    entry["training_best_distance"] = round(h["best_distance"], 1) if h else None
    return entry


def record_ppo(path, n_cars, track_cache, device, args, seed):
    state = load_checkpoint(path, device)
    cfg = Config.from_dict(state["config"])
    track = track_cache(cfg)
    model = ActorCritic(state.get("hidden", cfg.ppo.hidden)).to(device)
    model.load_state_dict(state["model"])
    rms = RunningMeanStd(13, device)
    rms.load_state_dict({k: v.to(device) for k, v in state["obs_rms"].items()})
    noise = torch.Generator(device=device).manual_seed(seed)

    def act(obs, stochastic=True):
        mean, _ = model(rms.normalize(obs, cfg.ppo.obs_clip))
        if not stochastic:
            return mean.clamp(-1, 1)
        eps = torch.randn(mean.shape, generator=noise, device=device)
        return (mean + model.log_std.exp() * eps).clamp(-1, 1)

    env = VecEnv(track, cfg, n_cars=n_cars, device=device)
    frames, end, done = drive(env, act, args.fps, args.max_seconds)
    score, dist, lap = car_results(env, cfg, track)
    upd = state["iteration"]
    entry = replay_entry(f"ppo_u{upd:04d}", "ppo", f"Update {upd}", upd, state["env_steps"],
                         frames, end, done, score, dist, lap, args.fps, args.out)
    # The noise-free policy (mean action), driven alone.
    solo = VecEnv(track, cfg, n_cars=1, device=device)
    obs = solo.reset_all()
    for _ in range(solo.max_steps):
        obs, _, term, trunc, _ = solo.step(act(obs, stochastic=False))
        if bool(term[0] | trunc[0]):
            break
    t = solo.lap_time[0].item()
    entry["greedy_distance"] = round(min(solo.best_dist[0].item(), track.length), 1)
    entry["greedy_lap"] = round(t, 2) if math.isfinite(t) else None
    entry["action_std"] = round(float(model.log_std.detach().exp().mean()), 3)
    return entry


# ----------------------------------------------------------------------------------
# Static exports
# ----------------------------------------------------------------------------------
def export_track(track, out_dir, spacing=4.0):
    step = max(1, round(spacing / track.ds))
    c = track.center.cpu().numpy()[::step]
    q = np.round(c / QUANT).astype(int)
    idx_scale = len(c) / track.M
    corners = [{"name": k["name"], "major": bool(k["major"]), "s": round(k["s"], 1),
                "i": int(round(k["index"] * idx_scale)) % len(c)} for k in track.corners]
    data = {
        "quant": QUANT, "length": round(track.length, 1), "width": track.width,
        "center": q.reshape(-1).tolist(), "corners": corners,
        "source": "Centerline © OpenStreetMap contributors (ODbL 1.0), route relation 38566",
    }
    with open(os.path.join(out_dir, "track.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def learning_curve(history, keys):
    return [[h["env_steps"]] + [None if (isinstance(h[k], float) and not math.isfinite(h[k])) else
                                round(h[k], 1) for k in keys] for h in history]


def read_summary(results_dir):
    path = os.path.join(results_dir, "compare_summary.csv")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    utf8_stdout()
    p = argparse.ArgumentParser(description="Record GA/PPO replays for the browser viewer.")
    p.add_argument("--ga", default="checkpoints/ga_seed1", help="folder with ga_00000.pt, ga_00001.pt, ...")
    p.add_argument("--ppo", default="checkpoints/ppo_seed1", help="folder with numbered PPO checkpoints")
    p.add_argument("--gens", default="1,5,20,50,100,final", help="GA generations to record ('final' = last)")
    p.add_argument("--out", default="web/data")
    p.add_argument("--results", default="results", help="where compare.py wrote its plots / summary")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--max-seconds", type=float, default=1200.0, help="cap on replay length (sim seconds)")
    p.add_argument("--seed", type=int, default=0, help="seed for the PPO cars' exploration noise")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    for name in os.listdir(args.out):                 # start clean (old replays would linger otherwise)
        if name.endswith(".i16.gz"):
            os.remove(os.path.join(args.out, name))

    tracks = {}

    def track_cache(cfg):
        key = cfg.track.path
        if key not in tracks:
            tracks[key] = Track(cfg.track, device)
        return tracks[key]

    ga_ckpts = numbered(args.ga, "ga")
    ppo_ckpts = numbered(args.ppo, "ppo")
    if not ga_ckpts or not ppo_ckpts:
        raise SystemExit(f"No numbered checkpoints found in {args.ga} / {args.ppo} (see the docstring).")
    last = max(ga_ckpts)
    gens = []
    for g in args.gens.split(","):
        g = last + 1 if g.strip() == "final" else int(g)
        if g - 1 in ga_ckpts and g not in gens:
            gens.append(g)
        elif g not in gens:
            print(f"skipping GA generation {g}: no checkpoint ga_{g - 1:05d}.pt")
    latest_ga = load_checkpoint(ga_ckpts[last], "cpu")
    history_by_gen = {h["iteration"]: h for h in latest_ga["history"]}
    ppo_steps = {it: load_checkpoint(path, "cpu")["env_steps"] for it, path in ppo_ckpts.items()}

    replays, pairs = {}, []
    for g in gens:
        ga = record_ga(ga_ckpts[g - 1], g, track_cache, device, args, history_by_gen)
        replays[ga["id"]] = ga
        # PPO checkpoint with the closest amount of experience (compared on a log scale).
        target = max(ga["env_steps"], 1)
        it = min(ppo_steps, key=lambda k: abs(math.log(max(ppo_steps[k], 1) / target)))
        pid = f"ppo_u{it:04d}"
        if pid not in replays:
            replays[pid] = record_ppo(ppo_ckpts[it], ga["n_cars"], track_cache, device, args, args.seed)
        pairs.append({"label": ga["title"], "ga": ga["id"], "ppo": pid})
        pr = replays[pid]
        check = ga["training_best_distance"]
        print(f"gen {g:4d} @ {ga['env_steps'] / 1e6:6.2f}M steps: best {ga['best_distance']:7.0f} m "
              f"(training log {check if check is not None else '-':>7}), laps {ga['laps_completed']:3d}, "
              f"{ga['duration']:6.1f} s, {ga['bytes'] / 1e6:5.2f} MB | PPO upd {it} @ {pr['env_steps'] / 1e6:6.2f}M: "
              f"best {pr['best_distance']:6.0f} m, noise-free {pr['greedy_distance']:6.0f} m, {pr['bytes'] / 1e6:5.2f} MB",
              flush=True)

    track = next(iter(tracks.values()))
    export_track(track, args.out)
    latest_ppo = load_checkpoint(ppo_ckpts[max(ppo_ckpts)], "cpu")
    plots = []
    os.makedirs(os.path.join(args.out, "plots"), exist_ok=True)
    for name, alt in (("compare_distance.png", "Best distance vs environment steps, GA vs PPO, mean ± std over 3 seeds"),
                      ("compare_laptime.png", "Best lap time vs environment steps, GA vs PPO")):
        src = os.path.join(args.results, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, "plots", name))
            plots.append({"src": f"data/plots/{name}", "alt": alt})
    manifest = {
        "version": 1, "fps": args.fps, "quant": QUANT, "heading_scale": HEADING_SCALE,
        "track_length": round(track.length, 1), "pairs": pairs, "replays": replays,
        "curves": {
            "ga": learning_curve(latest_ga["history"], ["best_distance", "best_lap_time_ever"]),
            "ppo": learning_curve(latest_ppo["history"], ["best_distance", "mean_distance"]),
        },
        "plots": plots, "summary": read_summary(args.results),
    }
    with open(os.path.join(args.out, "replays.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, separators=(",", ":"))
    total = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(args.out) for fn in fns)
    print(f"Wrote {len(replays)} replays to {args.out}/ - total {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
