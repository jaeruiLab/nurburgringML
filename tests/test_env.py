import math

import pytest
import torch

from env import OBS_SIZE, VecEnv, scripted_driver


def make(track, cfg, n=8, random_start=False):
    env = VecEnv(track, cfg, n_cars=n, device="cpu", random_start=random_start)
    obs = env.reset_all()
    return env, obs


def run(env, action, steps, auto_reset=False):
    out = None
    for _ in range(steps):
        out = env.step(action, auto_reset=auto_reset)
    return out


def test_observation_shape_and_range(track, cfg):
    env, obs = make(track, cfg)
    assert obs.shape == (8, OBS_SIZE)
    assert torch.isfinite(obs).all()
    rays = obs[:, :9]
    assert (rays > 0).all() and (rays <= 1).all()
    # Standing on the centerline facing forward: side rays see the edge ~5.5 m away.
    assert abs(env.last_ray_m[0, 0].item() - track.width / 2) < 1.0
    assert abs(env.last_ray_m[0, -1].item() - track.width / 2) < 1.0
    assert obs[:, 9].abs().max() == 0          # speed 0
    assert obs[:, 10].abs().max() < 0.05       # heading error ~0


def test_full_steer_drives_off_track(track, cfg):
    env, _ = make(track, cfg, n=1)
    act = torch.tensor([[1.0, 1.0]])
    crashed = False
    for _ in range(600):
        _, r, term, trunc, info = env.step(act)
        if term[0]:
            crashed = bool(info["crashed"][0])
            break
    assert crashed and not env.alive[0]
    assert r[0] < -5  # crash penalty applied


def test_standing_still_times_out_after_3s(track, cfg):
    env, _ = make(track, cfg, n=2)
    act = torch.zeros(2, 2)
    for i in range(1, 400):
        _, _, term, _, _ = env.step(act)
        if term.all():
            break
    assert i == env.no_progress_steps


def test_driving_backwards_terminates(track, cfg):
    env, _ = make(track, cfg, n=1)
    # Spawn mid-straight (Döttinger Höhe) facing the wrong way.
    i = int(track.index_at(19_000.0))
    env.seg_idx[:] = i
    env.s[:] = track.cum_s[i]
    env.x[:], env.y[:] = track.center[i, 0], track.center[i, 1]
    env.yaw[:] = track.heading[i] + math.pi
    act = torch.tensor([[0.0, 1.0]])
    for _ in range(600):
        _, _, term, _, info = env.step(act)
        if term[0]:
            break
    assert term[0] and info["crashed"][0]
    assert env.dist[0] < -cfg.env.backwards_tolerance + 1e-3


def test_progress_matches_distance_driven(track, cfg):
    env, _ = make(track, cfg, n=4)
    act = torch.stack([torch.zeros(4), torch.full((4,), 0.6)], dim=1)
    travelled = torch.zeros(4)
    total_r = torch.zeros(4)
    for _ in range(120):
        _, r, _, _, _ = env.step(act)
        travelled += env.v * env.dt
        total_r += r
    # Straight-ish start: centerline progress ~ distance driven.
    assert torch.allclose(env.dist, travelled, rtol=0.05)
    assert (total_r > env.dist).all()  # progress + positive speed term


def test_ga_mode_freezes_dead_cars_and_counts_alive_steps(track, cfg):
    env, _ = make(track, cfg, n=4)
    act = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    steps_alive = 0
    for _ in range(400):
        steps_alive += int(env.alive.sum())
        env.step(act)
    assert not env.alive.any()
    assert env.total_steps == steps_alive
    x = env.x.clone()
    env.step(act)
    assert torch.equal(env.x, x)  # frozen
    assert env.total_steps == steps_alive


def test_auto_reset_respawns(track, cfg):
    env, _ = make(track, cfg, n=6, random_start=True)
    act = torch.tensor([[1.0, 1.0]]).repeat(6, 1)
    respawned = False
    for k in range(600):
        obs, _, term, _, info = env.step(act, auto_reset=True)
        assert env.alive.all()
        if term.any():
            assert (env.dist[term] == 0).all() and (env.v[term] == 0).all()
            assert torch.isfinite(info["ep_distance"][term]).all()
            respawned = True
    assert respawned
    assert env.total_steps == 600 * 6


def test_truncation_returns_final_obs(track, cfg):
    env, _ = make(track, cfg, n=2, random_start=True)
    env.k.max_steps.fill_(30)
    act = torch.tensor([[0.0, 0.3], [0.0, 0.3]])
    for _ in range(29):
        env.step(act, auto_reset=True)
    _, _, term, trunc, info = env.step(act, auto_reset=True)
    assert trunc.all() and not term.any()
    assert info["final_obs"].shape == (2, OBS_SIZE)


@pytest.mark.slow
def test_scripted_driver_completes_a_lap(track, cfg):
    """The track is drivable with this physics, and lap detection works."""
    env, _ = make(track, cfg, n=1)
    for _ in range(int(900 / env.dt)):
        _, _, term, _, info = env.step(scripted_driver(env))
        if term[0]:
            break
    assert info["finished"][0], f"scripted driver stopped at {env.best_dist[0].item():.0f} m"
    lap = env.lap_time[0].item()
    print(f"scripted driver lap: {lap:.1f} s")
    assert 300 < lap < 700
