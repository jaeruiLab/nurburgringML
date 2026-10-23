"""Record replays of saved GA / PPO checkpoints for the browser viewer in web/.

    python record.py                                   # every GA generation, 6 worker processes
    python record.py --gens 1,5,20,50,100,final        # just a few
    python record.py --ga checkpoints/ga_seed1 --ppo checkpoints/ppo_seed0 --workers 4

Make the checkpoints first (one per GA generation, log-spaced ones for PPO):
    python main.py --algo ga  --headless --seed 1 --generations 100 --save-every 1
    python main.py --algo ppo --headless --seed 0 --threads 1 --max-steps 125e6 --save-every 1000000000 --save-log-ratio 1.1

What it does
  * GA generation g: loads the population as it was at the start of generation g (checkpoint
    ga_{g-1}.pt) and drives all its cars from the start grid until every car has crashed or
    finished, exactly like training did (the replay is deterministic, and its best distance is
    checked against the training log).
  * PPO: for each GA replay, picks the PPO checkpoint trained on the closest number of env steps
    and drives the same number of cars from the start grid. PPO has ONE policy, so the cars differ
    only through the random exploration noise in their actions (seeded, so the replay is
    reproducible). The noise-free policy is driven once more on its own and reported separately.
  * Writes into web/data/: track.json, replays.json (a small index), and per replay a binary
    <id>.i16.gz with the motion plus <id>.json with per-car results (both loaded only when that
    replay is opened, so the page stays fast however many replays there are). The learning-curve
    plots from results/ are copied too.

Replay file format (<id>.i16.gz, gzip of little-endian int16)
  Car i has end_frames[i] + 1 frames (20 per second). Three "planes" follow each other:
    x: every car's x values, car after car     (second differences, units of `quant` = 0.2 m)
    y: the same for y
    h: every car's heading keyframes: one every `heading_stride` frames plus the last frame,
       as first differences of round(angle * heading_steps / 2 pi)
  Second differences of position (= changes in velocity) are almost always tiny numbers, which
  gzip squeezes very well. Heading is the part that does not compress (a car's angle jitters a
  little every frame), so it is stored 5 times a second at ~0.35 degree resolution and
  interpolated in the browser; that alone makes the files ~4x smaller. web/js/decode.js reads it.
"""
import argparse
import csv
import gzip
import json
import math
import multiprocessing as mp
import os
import re
import shutil

import numpy as np
import torch

QUANT = 0.2            # meters per position unit
HEADING_STRIDE = 4     # one heading keyframe every 4 frames (5 per second)
HEADING_STEPS = 1024   # heading units per full turn (~0.35 degrees)


# ----------------------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------------------
def heading_indices(n, stride=HEADING_STRIDE):
    """Frames that carry a heading keyframe: 0, stride, 2*stride, ... and always the last one."""
    idx = np.arange(0, n, stride)
    return idx if idx[-1] == n - 1 else np.append(idx, n - 1)


def encode_replay(frames, end_frames):
    """frames: (F, N, 3) float [x, y, yaw (unwrapped)]; end_frames: (N,) -> gzip bytes."""
    px, py, ph = [], [], []
    for i, end in enumerate(end_frames):
        f = frames[: end + 1, i]
        for axis, out in ((0, px), (1, py)):
            q = np.round(f[:, axis] / QUANT).astype(np.int64)
            v = np.diff(q, prepend=0)                             # first differences (frame 0 = absolute)
            out.append(np.diff(v, prepend=0))                     # second differences
        h = np.round(f[heading_indices(end + 1), 2] * HEADING_STEPS / (2 * math.pi)).astype(np.int64)
        ph.append(np.diff(h, prepend=0))
    planes = [np.concatenate(p) for p in (px, py, ph)]
    if max(np.abs(p).max(initial=0) for p in planes) > 32767:
        raise ValueError("a value does not fit in int16")
    return gzip.compress(np.concatenate(planes).astype("<i2").tobytes(), compresslevel=9, mtime=0)


def decode_replay(data, end_frames):
    """Inverse of encode_replay (Python twin of web/js/decode.js, used by the tests).
    Returns per car: (xy (F, 2) meters, heading keyframes (K,) radians, keyframe indices (K,))."""
    values = np.frombuffer(gzip.decompress(data), dtype="<i2").astype(np.int64)
    n_frames = sum(e + 1 for e in end_frames)
    xs, ys, hs = values[:n_frames], values[n_frames:2 * n_frames], values[2 * n_frames:]
    cars, o, oh = [], 0, 0
    for end in end_frames:
        n = end + 1
        idx = heading_indices(n)
        xy = np.stack([np.cumsum(np.cumsum(xs[o:o + n])), np.cumsum(np.cumsum(ys[o:o + n]))], 1) * QUANT
        h = np.cumsum(hs[oh:oh + len(idx)]) * (2 * math.pi / HEADING_STEPS)
        cars.append((xy, h, idx))
        o += n
        oh += len(idx)
    if oh != len(hs):
        raise ValueError("replay data does not match its index")
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
    from ga import fitness
    lap = env.lap_time.clone()
    score = fitness(env.best_dist, env.speed_sum, env.ep_steps, lap, cfg, track.length)
    dist = env.best_dist.clamp(0, track.length)
    lap = [round(float(t), 2) if math.isfinite(t) else None for t in lap.tolist()]
    return score.cpu().numpy(), dist.cpu().numpy(), lap


def save_replay(rid, algo, title, iteration, env_steps, frames, end, done, score, dist, lap, fps, out_dir):
    """Write <rid>.i16.gz (motion) and <rid>.json (per-car results); return the index entry."""
    data = encode_replay(frames, end)
    with open(os.path.join(out_dir, f"{rid}.i16.gz"), "wb") as f:
        f.write(data)
    laps = [t for t in lap if t is not None]
    finished = [t is not None for t in lap]
    cars = {
        "end_frames": [int(e) for e in end],
        # "crashed" = stopped before the end without finishing a lap.
        "crashed": [bool(d and not fi) for d, fi in zip(done, finished)],
        "finished": finished,
        "score": [round(float(s), 1) for s in score],
        "distance": [round(float(d), 1) for d in dist],
        "lap_time": lap,
    }
    with open(os.path.join(out_dir, f"{rid}.json"), "w", encoding="utf-8") as f:
        json.dump(cars, f, separators=(",", ":"))
    return {
        "id": rid, "algo": algo, "title": title, "iteration": iteration, "env_steps": int(env_steps),
        "file": f"{rid}.i16.gz", "cars": f"{rid}.json", "bytes": len(data),
        "n_cars": len(end), "n_frames": int(frames.shape[0]), "duration": round((frames.shape[0] - 1) / fps, 2),
        "best": int(np.argmax(score)),
        "best_distance": round(float(dist.max()), 1),
        "best_lap": min(laps) if laps else None,
        "laps_completed": len(laps),
    }


# ----------------------------------------------------------------------------------
# GA and PPO replays (each runs in a worker process)
# ----------------------------------------------------------------------------------
_tracks = {}


def _track(cfg, device):
    from track import Track
    if cfg.track.path not in _tracks:
        _tracks[cfg.track.path] = Track(cfg.track, device)
    return _tracks[cfg.track.path]


def record_ga(job):
    path, generation, opts = job
    from brain import PopulationBrain
    from config import Config
    from env import VecEnv
    from utils import load_checkpoint
    torch.set_num_threads(1)
    device = torch.device(opts["device"])
    state = load_checkpoint(path, device)
    cfg = Config.from_dict(state["config"])
    track = _track(cfg, device)
    genes = state["genes"].to(device)
    brain = PopulationBrain(genes.shape[0], state["layer_sizes"], device, genes=genes)
    mean, std = (t.to(device) for t in state["norm"])
    clip = state.get("obs_clip")

    def act(obs):
        if clip is not None:
            obs = ((obs - mean) / std).clamp(-clip, clip)
        return brain(obs)

    env = VecEnv(track, cfg, n_cars=genes.shape[0], device=device)
    frames, end, done = drive(env, act, opts["fps"], opts["max_seconds"])
    score, dist, lap = car_results(env, cfg, track)
    return save_replay(f"ga_g{generation:03d}", "ga", f"Generation {generation}", generation,
                       state["env_steps"], frames, end, done, score, dist, lap, opts["fps"], opts["out"])


def record_ppo(job):
    path, n_cars, opts = job
    from config import Config
    from env import VecEnv
    from ppo import ActorCritic
    from utils import RunningMeanStd, load_checkpoint
    torch.set_num_threads(1)
    device = torch.device(opts["device"])
    state = load_checkpoint(path, device)
    cfg = Config.from_dict(state["config"])
    track = _track(cfg, device)
    model = ActorCritic(state.get("hidden", cfg.ppo.hidden)).to(device)
    model.load_state_dict(state["model"])
    rms = RunningMeanStd(13, device)
    rms.load_state_dict({k: v.to(device) for k, v in state["obs_rms"].items()})
    noise = torch.Generator(device=device).manual_seed(opts["seed"])

    @torch.no_grad()
    def act(obs, stochastic=True):
        mean, _ = model(rms.normalize(obs, cfg.ppo.obs_clip))
        if not stochastic:
            return mean.clamp(-1, 1)
        eps = torch.randn(mean.shape, generator=noise, device=device)
        return (mean + model.log_std.exp() * eps).clamp(-1, 1)

    env = VecEnv(track, cfg, n_cars=n_cars, device=device)
    frames, end, done = drive(env, act, opts["fps"], opts["max_seconds"])
    score, dist, lap = car_results(env, cfg, track)
    upd = state["iteration"]
    entry = save_replay(f"ppo_u{upd:04d}", "ppo", f"Update {upd}", upd, state["env_steps"],
                        frames, end, done, score, dist, lap, opts["fps"], opts["out"])
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


def learning_curve(history, keys, max_points=400):
    """Training log for the viewer's chart, thinned to ~max_points spread evenly on a log scale."""
    if len(history) > max_points:
        steps = np.log10(np.maximum([h["env_steps"] for h in history], 1))
        marks = np.linspace(steps.min(), steps.max(), max_points)
        keep = sorted(set(np.searchsorted(steps, marks).clip(0, len(history) - 1).tolist()) | {len(history) - 1})
        # keep each kept row's running best so the "best so far" line is not lost
        best = {k: -math.inf for k in keys}
        rows = []
        for i, h in enumerate(history):
            for k in keys:
                if isinstance(h[k], (int, float)) and math.isfinite(h[k]):
                    best[k] = max(best[k], h[k]) if k.startswith("best_distance") else h[k]
            if i in keep:
                rows.append({"env_steps": h["env_steps"], **{k: best[k] for k in keys}})
        history = rows
    return [[h["env_steps"]] + [None if (isinstance(h[k], float) and not math.isfinite(h[k])) else
                                round(h[k], 1) for k in keys] for h in history]


def read_summary(results_dir):
    path = os.path.join(results_dir, "compare_summary.csv")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def numbered(dirpath, prefix):
    """{iteration: path} for checkpoints like ga_00012.pt."""
    out = {}
    for name in os.listdir(dirpath):
        m = re.fullmatch(rf"{prefix}_(\d+)\.pt", name)
        if m:
            out[int(m.group(1))] = os.path.join(dirpath, name)
    return dict(sorted(out.items()))


def run_jobs(fn, jobs, workers):
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
        return pool.map(fn, jobs, chunksize=1)


def main():
    from config import Config
    from track import Track
    from utils import load_checkpoint, utf8_stdout
    utf8_stdout()
    p = argparse.ArgumentParser(description="Record GA/PPO replays for the browser viewer.")
    p.add_argument("--ga", default="checkpoints/ga_seed1", help="folder with ga_00000.pt, ga_00001.pt, ...")
    p.add_argument("--ppo", default="checkpoints/ppo_seed0", help="folder with numbered PPO checkpoints")
    p.add_argument("--gens", default="all", help="'all', or a list like 1,5,20,50,100,final ('final' = last)")
    p.add_argument("--out", default="web/data")
    p.add_argument("--results", default="results", help="where compare.py wrote its plots / summary")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--max-seconds", type=float, default=1200.0, help="cap on replay length (sim seconds)")
    p.add_argument("--seed", type=int, default=0, help="seed for the PPO cars' exploration noise")
    p.add_argument("--workers", type=int, default=6, help="replays recorded in parallel")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    # Record into a temporary folder and swap it in at the end, so a viewer that is open (or deployed)
    # keeps working with the old replays until the new set is complete.
    final_out = args.out
    args.out = final_out.rstrip("/\\") + ".new"
    shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(args.out)

    ga_ckpts = numbered(args.ga, "ga")
    ppo_ckpts = numbered(args.ppo, "ppo")
    if not ga_ckpts or not ppo_ckpts:
        raise SystemExit(f"No numbered checkpoints found in {args.ga} / {args.ppo} (see the docstring).")
    last = max(ga_ckpts)
    wanted = range(1, last + 2) if args.gens.strip() == "all" else [
        last + 1 if g.strip() == "final" else int(g) for g in args.gens.split(",")]
    gens = []
    for g in wanted:
        if g - 1 in ga_ckpts and g not in gens:
            gens.append(g)
        elif g not in gens:
            print(f"skipping GA generation {g}: no checkpoint ga_{g - 1:05d}.pt")
    opts = {"device": args.device, "fps": args.fps, "max_seconds": args.max_seconds, "out": args.out, "seed": args.seed}

    print(f"Recording {len(gens)} GA generations with {args.workers} workers ...", flush=True)
    ga_entries = run_jobs(record_ga, [(ga_ckpts[g - 1], g, opts) for g in gens], args.workers)

    # Each GA replay is paired with the PPO checkpoint trained on the closest number of steps
    # (compared on a log scale); several GA generations may share one PPO replay.
    ppo_steps = {it: load_checkpoint(path, "cpu")["env_steps"] for it, path in ppo_ckpts.items()}
    match = {}
    for e in ga_entries:
        target = max(e["env_steps"], 1)
        match[e["id"]] = min(ppo_steps, key=lambda k: abs(math.log(max(ppo_steps[k], 1) / target)))
    ppo_its = sorted(set(match.values()))
    n_cars = ga_entries[0]["n_cars"]
    print(f"Recording {len(ppo_its)} PPO checkpoints ...", flush=True)
    ppo_entries = {e["iteration"]: e for e in run_jobs(record_ppo, [(ppo_ckpts[it], n_cars, opts) for it in ppo_its],
                                                       args.workers)}

    latest_ga = load_checkpoint(ga_ckpts[last], "cpu")
    history_by_gen = {h["iteration"]: h for h in latest_ga["history"]}
    replays, pairs, mismatches = {}, [], 0
    for e in ga_entries:
        h = history_by_gen.get(e["iteration"])
        e["training_best_distance"] = round(h["best_distance"], 1) if h else None
        if h and abs(e["best_distance"] - h["best_distance"]) > 1.0:
            mismatches += 1
        replays[e["id"]] = e
        pr = ppo_entries[match[e["id"]]]
        replays[pr["id"]] = pr
        pairs.append({"label": e["title"], "generation": e["iteration"], "ga": e["id"], "ppo": pr["id"]})
        print(f"gen {e['iteration']:4d} @ {e['env_steps'] / 1e6:7.2f}M: best {e['best_distance']:7.0f} m "
              f"(log {e['training_best_distance'] if h else '-':>7}), laps {e['laps_completed']:3d}, {e['bytes'] / 1e6:5.2f} MB"
              f" | PPO upd {pr['iteration']} @ {pr['env_steps'] / 1e6:7.2f}M: best {pr['best_distance']:6.0f} m, "
              f"laps {pr['laps_completed']:3d}, noise-free {pr['greedy_distance']:6.0f} m")
    print(f"Determinism check: {len(ga_entries) - mismatches} of {len(ga_entries)} GA replays match the training log"
          f"{'' if not mismatches else ' (mismatches above!)'}")

    track = Track(Config.from_dict(latest_ga["config"]).track)
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
        "version": 2, "fps": args.fps, "quant": QUANT,
        "heading_stride": HEADING_STRIDE, "heading_steps": HEADING_STEPS,
        "track_length": round(track.length, 1), "pairs": pairs, "replays": replays,
        "curves": {
            "ga": learning_curve(latest_ga["history"], ["best_distance", "best_lap_time_ever"]),
            "ppo": learning_curve(latest_ppo["history"], ["best_distance", "mean_distance"]),
        },
        "plots": plots, "summary": read_summary(args.results),
    }
    with open(os.path.join(args.out, "replays.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, separators=(",", ":"))
    shutil.rmtree(final_out, ignore_errors=True)                  # swap the finished set in
    os.replace(args.out, final_out)
    total = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(final_out) for fn in fns)
    biggest = max(e["bytes"] for e in replays.values())
    print(f"Wrote {len(replays)} replays to {final_out}/ - total {total / 1e6:.1f} MB, "
          f"largest single replay {biggest / 1e6:.2f} MB, index {os.path.getsize(os.path.join(final_out, 'replays.json')) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
