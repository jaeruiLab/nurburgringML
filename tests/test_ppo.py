import copy
import math

import torch

from ppo import PPOTrainer, ppo_policy_from_checkpoint
from utils import RunningMeanStd, load_checkpoint


def test_running_mean_std_matches_full_batch():
    torch.manual_seed(0)
    rms = RunningMeanStd(3, "cpu")
    batches = [torch.randn(50, 3) * 4 + 2 for _ in range(5)]
    for b in batches:
        rms.update(b)
    full = torch.cat(batches)
    assert torch.allclose(rms.mean, full.mean(0), atol=1e-3)
    assert torch.allclose(rms.var, full.var(0, unbiased=False), rtol=1e-3)


def small_trainer(cfg, T=16, n=8):
    c = copy.deepcopy(cfg)
    c.env.n_cars = n
    c.ppo.rollout_steps = T
    c.ppo.minibatch_size = 32
    c.checkpoint_dir = "checkpoints_test"
    return PPOTrainer(c, torch.device("cpu"), seed=0)


def test_gae_matches_naive_reference(cfg):
    tr = small_trainer(cfg)
    T, N = tr.p.rollout_steps, tr.N
    torch.manual_seed(1)
    tr.buf_rew = torch.randn(T, N)
    tr.buf_val = torch.randn(T, N)
    tr.buf_done = (torch.rand(T, N) < 0.2).float()
    adv, ret = tr.compute_gae()
    with torch.no_grad():
        _, last_v = tr.model(tr.obs_rms.normalize(tr.obs, tr.p.obs_clip))
    g, lam = tr.p.gamma, tr.p.gae_lambda
    for n in range(N):                     # plain Python, one car at a time
        running = 0.0
        for t in reversed(range(T)):
            nv = last_v[n] if t == T - 1 else tr.buf_val[t + 1, n]
            nd = 1.0 - tr.buf_done[t, n]
            delta = tr.buf_rew[t, n] + g * nv * nd - tr.buf_val[t, n]
            running = delta + g * lam * nd * running
            assert abs(adv[t, n] - running) < 1e-4
    assert torch.allclose(ret, adv + tr.buf_val)


def test_update_runs_and_changes_policy(cfg, tmp_path):
    tr = small_trainer(cfg)
    before = [p.detach().clone() for p in tr.model.parameters()]
    updated = [tr.step() for _ in range(tr.p.rollout_steps)]
    assert updated[-1] and not any(updated[:-1])
    h = tr.history[-1]
    assert h["iteration"] == 1 and h["env_steps"] == tr.p.rollout_steps * tr.N
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"):
        assert math.isfinite(h[k])
    assert any(not torch.equal(a, b) for a, b in zip(before, tr.model.parameters()))

    path = tr.save(str(tmp_path / "ppo.pt"))
    tr2 = small_trainer(cfg)
    tr2.load(path)
    for a, b in zip(tr.model.parameters(), tr2.model.parameters()):
        assert torch.equal(a, b)
    state = load_checkpoint(path)
    policy = ppo_policy_from_checkpoint(state, tr.cfg, "cpu")
    a = policy(torch.zeros(3, 13))
    assert a.shape == (3, 2) and a.abs().max() <= 1
