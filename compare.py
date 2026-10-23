"""Fair comparison: GA vs PPO on the same budget of environment steps.

Why environment steps? A GA "generation" and a PPO "update" are completely different
amounts of work, so plotting against them would be meaningless. One environment step
= one car simulated for 1/60 s. Both methods are charged for every step a car is
actually driving, so the x-axis measures how much experience each method needed.

Usage
  python compare.py                          # 50M steps x 3 seeds x {GA, PPO}
  python compare.py --budget 5e6 --seeds 0,1 # quick version
  python compare.py --plot-only              # re-draw plots from results/*.csv

Outputs (in results/)
  {algo}_seed{n}.csv        learning curve of each run
  compare_distance.png      best distance vs env steps (mean +- std over seeds)
  compare_laptime.png       best lap time vs env steps
  compare_summary.csv       final numbers per run and per method
"""
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import time
import warnings

import numpy as np

COLORS = {"ga": "#2a78d6", "ppo": "#eb6834", "ppo_spec": "#1baf7a"}   # categorical slots 1, 2, 3
LABELS = {"ga": "GA (neuroevolution)", "ppo": "PPO (tuned)", "ppo_spec": "PPO (as specified)"}
SHORT = {"ga": "GA", "ppo": "PPO", "ppo_spec": "PPO (spec)"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------------------------
# One training run (executed in a worker process)
# ----------------------------------------------------------------------------------
def evaluate_policy(policy, track, cfg, device):
    """Drive ONE car deterministically from the start grid: how far / how fast?"""
    from env import VecEnv
    env = VecEnv(track, cfg, n_cars=1, device=device)
    obs = env.reset_all()
    for _ in range(env.max_steps):
        obs, _, term, trunc, _ = env.step(policy(obs))
        if bool(term[0] | trunc[0]):
            break
    lap = env.lap_time[0].item()
    return min(env.best_dist[0].item(), track.length), (lap if math.isfinite(lap) else math.nan)


def run_one(job):
    algo, seed, budget, device_name, out_dir, threads, n_cars = job
    import torch
    from config import Config
    from main import build_trainer
    from utils import get_device, utf8_stdout, write_csv

    utf8_stdout()
    device = get_device(device_name)
    torch.set_num_threads(threads)
    cfg = Config()
    cfg.seed = seed
    cfg.env.n_cars = n_cars
    cfg.checkpoint_dir = os.path.join(out_dir, "checkpoints")
    tr = build_trainer(algo, cfg, device, seed)
    rows = []
    csv_path = os.path.join(out_dir, f"{algo}_seed{seed}.csv")
    last_print = [0.0]

    def on_iter(t):
        h = t.history[-1]
        rows.append({
            "env_steps": h["env_steps"],
            "wall_time": h["wall_time"],
            "iteration": h["iteration"],
            "best_distance": h["best_distance_ever"],
            "best_lap_time": h["best_lap_time_ever"] if math.isfinite(h["best_lap_time_ever"]) else math.nan,
        })
        if time.time() - last_print[0] > 60:        # progress line + partial CSV once a minute
            last_print[0] = time.time()
            lap = h["best_lap_time_ever"]
            print(f"[{algo} seed {seed}] {h['env_steps'] / 1e6:6.1f}M / {budget / 1e6:.0f}M steps | "
                  f"best {h['best_distance_ever']:7.0f} m | lap {lap:6.1f} s | {h['wall_time'] / 60:5.1f} min",
                  flush=True)
            write_csv(csv_path, rows)

    tr.train_until(max_steps=budget, on_iter=on_iter)
    write_csv(csv_path, rows)
    tr.save(os.path.join(cfg.checkpoint_dir, f"{algo}_seed{seed}.pt"))
    eval_dist, eval_lap = evaluate_policy(tr.policy(), tr.track, cfg, device)
    print(f"[{algo} seed {seed}] done in {tr.wall_time / 60:.1f} min; deterministic eval: "
          f"{eval_dist:.0f} m, lap {eval_lap:.1f} s", flush=True)
    result = {"algo": algo, "seed": seed, "wall_time_s": round(tr.wall_time, 1),
              "eval_distance": round(eval_dist, 1), "eval_lap_time": eval_lap}
    with open(os.path.join(out_dir, f"{algo}_seed{seed}_eval.json"), "w", encoding="utf-8") as f:
        json.dump(result, f)
    return result


# ----------------------------------------------------------------------------------
# Aggregation and plots
# ----------------------------------------------------------------------------------
def read_run(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    steps = np.array([float(r["env_steps"]) for r in rows])
    dist = np.array([float(r["best_distance"]) for r in rows])
    lap = np.array([float(r["best_lap_time"]) if r["best_lap_time"] not in ("", "nan") else np.nan for r in rows])
    wall = np.array([float(r["wall_time"]) for r in rows])
    return steps, dist, lap, wall


def at_steps(steps, values, grid, before=np.nan):
    """Value of a 'best so far' curve at each grid point (last logged value <= that step)."""
    idx = np.searchsorted(steps, grid, side="right") - 1
    out = np.where(idx >= 0, values[np.clip(idx, 0, None)], before)
    return out.astype(float)


def collect(out_dir, algos, seeds, budget, n_grid, extra=None):
    """extra: {series name: (folder, file prefix)}, e.g. a reference run kept in another folder."""
    grid = np.linspace(0, budget, n_grid)
    data = {}
    sources = [(a, out_dir, a) for a in algos] + [(k, d, p) for k, (d, p) in (extra or {}).items()]
    for algo, folder, prefix in sources:
        runs = []
        for s in seeds:
            path = os.path.join(folder, f"{prefix}_seed{s}.csv")
            if os.path.exists(path):
                runs.append((s, *read_run(path)))
        if runs:
            data[algo] = {
                "seeds": [r[0] for r in runs],
                "dist": np.stack([at_steps(r[1], r[2], grid, before=0.0) for r in runs]),
                "lap": np.stack([at_steps(r[1], r[3], grid) for r in runs]),
                "runs": runs,
            }
    return grid, data


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bdbcb6")
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def plot_curves(grid, data, key, ylabel, title, path, scale=1.0, ref=None, lower_is_better=False, clip=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    x = grid / 1e6
    no_laps = []
    for algo, d in data.items():
        vals = d[key] * scale
        if key == "lap" and not np.isfinite(vals).any():
            no_laps.append(LABELS[algo])           # nothing to draw: say so instead of an empty legend entry
            continue
        n = np.sum(np.isfinite(vals), axis=0)
        with warnings.catch_warnings():          # columns with no lap yet are all-NaN: fine
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.where(n > 0, np.nanmean(np.where(np.isfinite(vals), vals, np.nan), axis=0), np.nan)
            std = np.where(n > 1, np.nanstd(vals, axis=0), 0.0)
        c = COLORS[algo]
        reference = algo == "ppo_spec"            # drawn dashed, as a baseline
        lo, hi = mean - std, mean + std
        if clip is not None:                      # e.g. distance can't be < 0 or > one lap
            lo, hi = np.clip(lo, *clip), np.clip(hi, *clip)
        ax.fill_between(x, lo, hi, color=c, alpha=0.10 if reference else 0.16, linewidth=0)
        ax.plot(x, mean, color=c, linewidth=1.6 if reference else 2.0, linestyle=(0, (5, 3)) if reference else "-",
                label=f"{LABELS[algo]}, {len(d['seeds'])} seeds")
        last = np.where(np.isfinite(mean))[0]
        if len(last):
            i = last[-1]
            ax.plot([x[i]], [mean[i]], "o", color=c, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(SHORT[algo], (x[i], mean[i]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK, weight="bold")
            if key == "lap" and n[i] < len(d["seeds"]):
                ax.annotate(f"{n[i]}/{len(d['seeds'])} seeds have a lap", (x[i], mean[i]), xytext=(6, -12),
                            textcoords="offset points", fontsize=7.5, color=INK_2)
    if key == "lap":
        note = "Average over the seeds that have finished a lap so far."
        if no_laps:
            note += "  No lap yet: " + ", ".join(no_laps) + "."
        ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=8, color=INK_2)
    if ref is not None:
        ax.axhline(ref[0], color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(ref[1], (0, ref[0]), xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xlabel("Environment steps (millions)  -  one step = one car simulated for 1/60 s")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, x[-1] * 1.06)
    ax.set_title(title + ("  (lower is better)" if lower_is_better else ""), loc="left", fontsize=11,
                 color=INK, pad=26)
    # Legend in one row between the title and the plot, so it never covers a line.
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=max(1, len(data) - len(no_laps)), frameon=False,
                    fontsize=8.5,
                    borderaxespad=0.2, handlelength=1.8)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def summarise(grid, data, results, budget, track_length, out_dir):
    from utils import fmt_time, write_csv
    by_run = {(r["algo"], r["seed"]): r for r in results}
    rows, table = [], []
    for algo, d in data.items():
        finals_d, finals_l, first_laps, walls, evals = [], [], [], [], []
        for (seed, steps, dist, lap, wall) in d["runs"]:
            fd = at_steps(steps, dist, np.array([budget]), before=0.0)[0]
            fl = at_steps(steps, lap, np.array([budget]))[0]
            has = np.where(np.isfinite(lap) & (steps <= budget))[0]
            first = steps[has[0]] if len(has) else math.nan
            w = wall[-1] if len(wall) else math.nan
            ev = by_run.get((algo, seed), {})
            finals_d.append(fd)
            finals_l.append(fl)
            first_laps.append(first)
            walls.append(w)
            evals.append(ev.get("eval_lap_time", math.nan))
            rows.append({"algo": algo, "seed": seed, "final_best_distance_m": round(fd, 1),
                         "final_best_lap_s": fl, "steps_to_first_lap": first, "wall_time_s": w,
                         "eval_distance_m": ev.get("eval_distance", ""), "eval_lap_s": ev.get("eval_lap_time", "")})

        def ms(v, fmt, fmt_std=None):
            v = np.array(v, dtype=float)
            ok = v[np.isfinite(v)]
            if len(ok) == 0:
                return "never"
            s = fmt(ok.mean())
            if len(ok) > 1:
                s += f" ± {(fmt_std or fmt)(ok.std())}"
            if len(ok) < len(v):
                s += f" ({len(ok)}/{len(v)} seeds)"
            return s

        secs = lambda v: f"{v:.1f} s"

        table.append([LABELS[algo],
                      ms(finals_d, lambda v: f"{v:,.0f} m"),
                      ms(finals_l, fmt_time, secs),
                      ms(first_laps, lambda v: f"{v / 1e6:.1f}M"),
                      ms(walls, lambda v: f"{v / 60:.1f} min"),
                      ms(evals, fmt_time, secs)])
    write_csv(os.path.join(out_dir, "compare_summary.csv"), rows)

    head = ["Method", "Final best distance", "Final best lap", "Steps to first lap", "Wall-clock", "Deterministic eval lap"]
    widths = [max(len(str(r[i])) for r in [head] + table) for i in range(len(head))]
    line = "+".join("-" * (w + 2) for w in widths)
    print(f"\nBudget: {budget / 1e6:.0f}M env steps per run, track {track_length / 1000:.2f} km")
    print(line)
    print("|".join(f" {h:<{w}} " for h, w in zip(head, widths)))
    print(line)
    for r in table:
        print("|".join(f" {c:<{w}} " for c, w in zip(r, widths)))
    print(line)


def main():
    from config import Config
    from track import Track
    from utils import utf8_stdout
    utf8_stdout()
    cfg = Config()
    p = argparse.ArgumentParser(description="GA vs PPO on an equal environment-step budget.")
    p.add_argument("--budget", type=float, default=cfg.compare.budget, help="env steps per run")
    p.add_argument("--seeds", default=",".join(map(str, cfg.compare.seeds)))
    p.add_argument("--algos", default="ga,ppo")
    p.add_argument("--workers", type=int, default=None, help="parallel runs (default: all at once, up to cores/2)")
    p.add_argument("--cars", type=int, default=cfg.env.n_cars, help="cars per run (GA population / PPO envs)")
    p.add_argument("--device", default="cpu",
                   help="cpu (default: many small runs in parallel are fastest on CPU) | auto | cuda | mps")
    p.add_argument("--out", default=cfg.compare.out_dir)
    p.add_argument("--plot-only", action="store_true", help="skip training; re-plot existing CSVs")
    p.add_argument("--reference", default=os.path.join(cfg.compare.out_dir, "ppo_spec"),
                   help="folder with as-specified PPO runs to draw as a dashed baseline ('' = none)")
    args = p.parse_args()

    budget = int(args.budget)
    seeds = [int(s) for s in args.seeds.split(",")]
    algos = args.algos.split(",")
    os.makedirs(args.out, exist_ok=True)

    results = []
    if args.plot_only:
        for a in algos:
            for s in seeds:
                path = os.path.join(args.out, f"{a}_seed{s}_eval.json")
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as f:
                        results.append(json.load(f))
    else:
        jobs = [(a, s) for s in seeds for a in algos]
        workers = args.workers or max(1, min(len(jobs), (os.cpu_count() or 2) // 2))
        threads = max(1, (os.cpu_count() or 2) // (2 * workers))
        print(f"{len(jobs)} runs x {budget / 1e6:.0f}M steps, {args.cars} cars each, {workers} in parallel, "
              f"{threads} thread(s) each, device {args.device}")
        t0 = time.time()
        job_args = [(a, s, budget, args.device, args.out, threads, args.cars) for a, s in jobs]
        if workers == 1:
            results = [run_one(j) for j in job_args]
        else:
            with mp.get_context("spawn").Pool(workers) as pool:
                results = pool.map(run_one, job_args, chunksize=1)
        print(f"All runs finished in {(time.time() - t0) / 60:.1f} min")

    extra = {}
    if args.reference and os.path.isdir(args.reference):
        extra["ppo_spec"] = (args.reference, "ppo")
        for s in seeds:
            path = os.path.join(args.reference, f"ppo_seed{s}_eval.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    results.append({**json.load(f), "algo": "ppo_spec"})
    track_length = Track(cfg.track).length
    grid, data = collect(args.out, algos, seeds, budget, cfg.compare.grid_points, extra)
    if not data:
        print("No result CSVs found.")
        return
    plot_curves(grid, data, "dist", "Best distance in one attempt (km)",
                "Best distance driven vs experience used", os.path.join(args.out, "compare_distance.png"),
                scale=1e-3, ref=(track_length / 1000, f"full lap ({track_length / 1000:.2f} km)"),
                clip=(0, track_length / 1000))
    plot_curves(grid, data, "lap", "Best lap time (minutes)", "Best lap time vs experience used",
                os.path.join(args.out, "compare_laptime.png"), scale=1 / 60, lower_is_better=True)
    summarise(grid, data, results, budget, track_length, args.out)
    print(f"Plots and CSVs written to {args.out}/")


if __name__ == "__main__":
    main()
