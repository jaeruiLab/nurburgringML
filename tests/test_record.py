import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from record import HEADING_STEPS, HEADING_STRIDE, QUANT, decode_replay, encode_replay, heading_indices


def synthetic_replay(seed=0, n_cars=5, n_frames=200):
    """Cars driving curvy paths at up to ~80 m/s, sampled at 20 Hz, ending at different frames."""
    rng = np.random.default_rng(seed)
    frames = np.zeros((n_frames, n_cars, 3))
    for i in range(n_cars):
        yaw = rng.uniform(-math.pi, math.pi) + np.cumsum(rng.normal(0, 0.05, n_frames))
        speed = rng.uniform(0, 4.0, n_frames)                        # meters per frame
        frames[:, i, 0] = rng.uniform(-3000, 3000) + np.cumsum(speed * np.cos(yaw))
        frames[:, i, 1] = rng.uniform(-3000, 2000) + np.cumsum(speed * np.sin(yaw))
        frames[:, i, 2] = yaw                                        # unwrapped, like the simulator
    end = rng.integers(0, n_frames, n_cars)
    end[0] = n_frames - 1
    end[1] = 4 * 7                                                   # length that lands exactly on a keyframe
    return frames, end


def test_heading_keyframes():
    assert list(heading_indices(1)) == [0]
    assert list(heading_indices(9)) == [0, 4, 8]
    assert list(heading_indices(10)) == [0, 4, 8, 9]


def test_encode_decode_roundtrip():
    frames, end = synthetic_replay()
    cars = decode_replay(encode_replay(frames, end), end)
    for i, (xy, h, idx) in enumerate(cars):
        n = end[i] + 1
        assert xy.shape == (n, 2)
        assert np.abs(xy - frames[:n, i, :2]).max() <= QUANT / 2 + 1e-9
        assert list(idx) == list(heading_indices(n, HEADING_STRIDE))
        assert np.abs(h - frames[idx, i, 2]).max() <= math.pi / HEADING_STEPS + 1e-9


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_javascript_decoder_matches_python(tmp_path):
    """web/js/decode.js must read exactly what record.py writes."""
    frames, end = synthetic_replay(seed=1)
    data = encode_replay(frames, end)
    (tmp_path / "r.i16.gz").write_bytes(data)
    script = tmp_path / "check.mjs"
    decode_js = Path("web/js/decode.js").resolve().as_uri()
    script.write_text(f"""
import {{ readFileSync }} from "node:fs";
import {{ gunzip, decodePlanes }} from "{decode_js}";
const raw = await gunzip(readFileSync(process.argv[2]));
const cars = decodePlanes(new Int16Array(raw), {json.dumps([int(e) for e in end])}, {QUANT}, {HEADING_STRIDE}, {HEADING_STEPS});
console.log(JSON.stringify(cars.map(c => ({{ x: Array.from(c.x), y: Array.from(c.y), h: Array.from(c.h) }}))));
""")
    out = subprocess.run(["node", str(script), str(tmp_path / "r.i16.gz")], capture_output=True, text=True, check=True)
    for (xy, h, idx), js in zip(decode_replay(data, end), json.loads(out.stdout)):
        assert np.allclose(xy[:, 0], js["x"], atol=1e-3)
        assert np.allclose(xy[:, 1], js["y"], atol=1e-3)
        js_h = np.array(js["h"])
        assert np.allclose(js_h[idx], h, atol=1e-5)                   # keyframes exact
        assert len(js_h) == len(xy)                                  # every frame has a heading
        if len(idx) > 1:                                             # in between: linear interpolation
            assert np.allclose(js_h[1:idx[1]], np.interp(np.arange(1, idx[1]), idx[:2], h[:2]), atol=1e-5)


def test_record_end_to_end(cfg, tmp_path, monkeypatch):
    """Tiny GA + PPO runs -> numbered checkpoints -> record.py -> index, replay and per-car files."""
    import copy
    import sys

    import torch

    import record
    from ga import GATrainer
    from ppo import PPOTrainer

    c = copy.deepcopy(cfg)
    c.env.n_cars = 6
    c.ga.max_generation_time = 4.0
    c.ppo.rollout_steps = 8
    c.ppo.minibatch_size = 16
    ga = GATrainer(c, torch.device("cpu"), seed=0)
    ppo = PPOTrainer(c, torch.device("cpu"), seed=0)
    for tr, d in ((ga, "ga"), (ppo, "ppo")):
        for _ in range(3):
            tr.save(str(tmp_path / d / f"{d}_{tr.iteration:05d}.pt"))
            tr.train_until(max_iters=tr.iteration + 1)
    out = tmp_path / "web_data"
    monkeypatch.setattr(sys, "argv", ["record.py", "--ga", str(tmp_path / "ga"), "--ppo", str(tmp_path / "ppo"),
                                      "--out", str(out), "--max-seconds", "5", "--workers", "2"])
    record.main()
    manifest = json.loads((out / "replays.json").read_text())
    assert [p["generation"] for p in manifest["pairs"]] == [1, 2, 3]  # --gens defaults to all
    for p in manifest["pairs"]:
        g = manifest["replays"][p["ga"]]
        info = json.loads((out / g["cars"]).read_text())
        cars = decode_replay((out / g["file"]).read_bytes(), info["end_frames"])
        assert len(cars) == 6 and max(len(xy) for xy, _, _ in cars) == g["n_frames"]
        if g["training_best_distance"] is not None:                # the replay reproduces training exactly
            assert abs(g["best_distance"] - g["training_best_distance"]) < 0.2
        assert p["ppo"] in manifest["replays"] and (out / manifest["replays"][p["ppo"]]["cars"]).exists()
    assert "end_frames" not in json.dumps(manifest)                  # per-car data stays out of the index
    assert json.loads((out / "track.json").read_text())["corners"]
