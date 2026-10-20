"""PPO (Proximal Policy Optimization), written from scratch.

One neural network drives all 100 cars. Each car is its own little "environment"
that produces experience; together they fill a big batch quickly.

The loop:
  1. Rollout:  every car drives for 256 steps using the current policy. Actions are
               sampled from a Gaussian (bell curve) around the network's chosen action,
               so the cars try slightly different things -> exploration.
  2. Advantage: for every step, was the outcome better or worse than the critic
               expected? (GAE, see compute_gae)
  3. Update:   nudge the policy to make better-than-expected actions more likely and
               worse-than-expected ones less likely, but never by too much at once
               (the PPO clip). The critic learns to predict returns more accurately.
  4. Repeat.

Network ("actor-critic", shared body):
    obs (13) -> 64 -> 64 (tanh) -+-> policy head -> mean steer / throttle (2)
                                 +-> value head  -> "how good is this situation?" (1)
    plus a learnable log_std (2): how much random exploration noise to add.
"""
import math

import torch
import torch.nn as nn

from env import ACT_SIZE, OBS_SIZE, VecEnv
from trainer import BaseTrainer
from utils import RunningMeanStd


def layer(i, o, gain):
    lin = nn.Linear(i, o)
    nn.init.orthogonal_(lin.weight, gain)
    nn.init.zeros_(lin.bias)
    return lin


class ActorCritic(nn.Module):
    def __init__(self, hidden=64, init_log_std=-0.5, init_throttle_bias=0.0):
        super().__init__()
        self.body = nn.Sequential(layer(OBS_SIZE, hidden, math.sqrt(2)), nn.Tanh(),
                                  layer(hidden, hidden, math.sqrt(2)), nn.Tanh())
        self.policy_head = layer(hidden, ACT_SIZE, 0.01)   # tiny init: start with ~zero actions
        with torch.no_grad():
            self.policy_head.bias[1] = init_throttle_bias   # optional: start out rolling forward
        self.value_head = layer(hidden, 1, 1.0)
        # State-independent exploration noise: one learnable std per action.
        self.log_std = nn.Parameter(torch.full((ACT_SIZE,), float(init_log_std)))

    def forward(self, obs):
        h = self.body(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def distribution(self, mean):
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))


class PPOTrainer(BaseTrainer):
    algo = "ppo"

    def __init__(self, cfg, device, seed=0):
        super().__init__(cfg, device, seed)
        p = self.p = cfg.ppo
        self.N = cfg.env.n_cars
        self.env = VecEnv(self.track, cfg, n_cars=self.N, device=device, random_start=p.random_start)
        self.model = ActorCritic(p.hidden, p.init_log_std, p.init_throttle_bias).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=p.lr, eps=1e-5)
        self.obs_rms = RunningMeanStd(OBS_SIZE, device)
        T, N = p.rollout_steps, self.N
        z = lambda *s: torch.zeros(*s, device=device)
        self.buf_obs, self.buf_act = z(T, N, OBS_SIZE), z(T, N, ACT_SIZE)
        self.buf_logp, self.buf_rew, self.buf_done, self.buf_val = z(T, N), z(T, N), z(T, N), z(T, N)
        self.t = 0
        self.best_dist_ever = 0.0
        self.best_lap_ever = math.inf
        self._reset_episode_stats()
        self.obs = self.env.reset_all()

    def _reset_episode_stats(self):
        # Running sums kept on the device, read once per update (no per-step GPU->CPU syncs).
        z = lambda: torch.zeros((), device=self.device)
        self.st_episodes, self.st_return, self.st_crashes, self.st_crash_dist = z(), z(), z(), z()
        self.st_laps, self.st_best_dist = z(), z()
        self.st_best_lap = torch.tensor(math.inf, device=self.device)

    # ------------------------------------------------------------------------------
    def step(self):
        """One env step for all cars; runs a PPO update every rollout_steps. True if updated."""
        with torch.no_grad():
            self.collect_step()
        if self.t == self.p.rollout_steps:
            self.update()
            self.t = 0
            return True
        return False

    @torch.no_grad()
    def collect_step(self):
        """Drive every car one step with the current (noisy) policy and store the experience."""
        p = self.p
        self.obs_rms.update(self.obs)
        obs_n = self.obs_rms.normalize(self.obs, p.obs_clip)
        mean, value = self.model(obs_n)
        dist = self.model.distribution(mean)
        action = dist.sample()                      # raw sample (kept for the log-probability)
        logp = dist.log_prob(action).sum(-1)
        next_obs, reward, term, trunc, info = self.env.step(action.clamp(-1, 1), auto_reset=True)

        r = reward * p.reward_scale
        if "final_obs" in info:
            # Time limit reached: the episode was cut short, not failed. Pretend it goes on by
            # adding the critic's estimate of what would have come next (bootstrapping).
            _, v_final = self.model(self.obs_rms.normalize(info["final_obs"], p.obs_clip))
            r = r + p.gamma * v_final * trunc
        done = (term | trunc).float()

        t = self.t
        self.buf_obs[t], self.buf_act[t], self.buf_logp[t] = obs_n, action, logp
        self.buf_rew[t], self.buf_done[t], self.buf_val[t] = r, done, value

        d = info["ep_done"]
        crashed = info["crashed"]
        self.st_episodes += d.sum()
        self.st_return += (info["ep_return"] * d).sum()
        self.st_crashes += crashed.sum()
        self.st_crash_dist += (info["ep_distance"] * crashed).sum()
        self.st_laps += info["finished"].sum()
        self.st_best_dist = torch.maximum(self.st_best_dist, (info["ep_distance"] * d).max())
        self.st_best_lap = torch.minimum(self.st_best_lap, info["ep_lap_time"].min())

        self.obs = next_obs
        self.t += 1

    # ------------------------------------------------------------------------------
    @torch.no_grad()
    def compute_gae(self):
        """Generalised Advantage Estimation.

        Advantage = "how much better did this action turn out than the critic expected?"
        One-step version (TD error):  delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        GAE blends the TD errors of the following steps, each weighted by (gamma*lambda)^k:
            A_t = delta_t + (gamma*lambda) * delta_{t+1} + (gamma*lambda)^2 * delta_{t+2} + ...
        computed backwards in one pass. lambda=0 trusts the critic fully (low variance, more
        bias), lambda=1 uses the actual rewards (no bias, noisy); 0.95 is a good middle.
        When a car's episode ends at step t (done=1), nothing after t belongs to that
        episode, so the chain is cut there (the car respawned into a new episode).
        """
        p = self.p
        _, last_value = self.model(self.obs_rms.normalize(self.obs, p.obs_clip))
        adv = torch.zeros_like(self.buf_rew)
        running = torch.zeros(self.N, device=self.device)
        for t in reversed(range(p.rollout_steps)):          # loop over time; all cars at once
            next_value = last_value if t == p.rollout_steps - 1 else self.buf_val[t + 1]
            not_done = 1.0 - self.buf_done[t]
            delta = self.buf_rew[t] + p.gamma * next_value * not_done - self.buf_val[t]
            running = delta + p.gamma * p.gae_lambda * not_done * running
            adv[t] = running
        returns = adv + self.buf_val                          # targets for the critic
        return adv, returns

    def update(self):
        p = self.p
        adv, returns = self.compute_gae()
        B = p.rollout_steps * self.N
        b_obs, b_act = self.buf_obs.reshape(B, OBS_SIZE), self.buf_act.reshape(B, ACT_SIZE)
        b_logp, b_adv, b_ret = self.buf_logp.reshape(B), adv.reshape(B), returns.reshape(B)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
        n = 0
        for _ in range(p.epochs):
            perm = torch.randperm(B, device=self.device)
            for start in range(0, B, p.minibatch_size):
                idx = perm[start:start + p.minibatch_size]
                mean, value = self.model(b_obs[idx])
                dist = self.model.distribution(mean)
                logp = dist.log_prob(b_act[idx]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                a = b_adv[idx]
                if p.norm_adv:
                    a = (a - a.mean()) / (a.std() + 1e-8)

                # PPO clipped objective. ratio = new probability / old probability of the action.
                # If the advantage is positive we want ratio > 1 (make the action more likely),
                # but only up to 1 + eps; if negative, ratio < 1 but only down to 1 - eps.
                # Taking the minimum (here: max of the negated terms) removes any incentive to
                # push the policy further than that in one update -> stable learning.
                ratio = torch.exp(logp - b_logp[idx])
                surr1 = ratio * a
                surr2 = ratio.clamp(1 - p.clip_eps, 1 + p.clip_eps) * a
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (value - b_ret[idx]).square().mean()
                loss = policy_loss + p.vf_coef * value_loss - p.ent_coef * entropy

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), p.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    log_ratio = logp - b_logp[idx]
                    stats["policy_loss"] += policy_loss.detach()
                    stats["value_loss"] += value_loss.detach()
                    stats["entropy"] += entropy.detach()
                    stats["approx_kl"] += ((ratio - 1) - log_ratio).mean()   # how far the policy moved
                    stats["clip_frac"] += ((ratio - 1).abs() > p.clip_eps).float().mean()
                n += 1
        stats = {k: float(v) / n for k, v in stats.items()}
        self._log_update(stats)

    def _log_update(self, stats):
        episodes = int(self.st_episodes)
        crashes = int(self.st_crashes)
        best_dist = min(float(self.st_best_dist), self.track.length)
        best_lap = float(self.st_best_lap)
        self.best_dist_ever = max(self.best_dist_ever, best_dist)
        self.best_lap_ever = min(self.best_lap_ever, best_lap)
        self.iteration += 1
        self.history.append({
            "iteration": self.iteration,
            "env_steps": self.env_steps,
            "wall_time": round(self.wall_time, 2),
            "episodes": episodes,
            "mean_return": float(self.st_return) / episodes if episodes else math.nan,
            "mean_distance": float(self.st_crash_dist) / crashes if crashes else math.nan,
            "best_distance": best_dist,
            "best_distance_ever": self.best_dist_ever,
            "best_lap_time": best_lap,
            "best_lap_time_ever": self.best_lap_ever,
            "laps_completed": int(self.st_laps),
            "action_std": float(self.model.log_std.detach().exp().mean()),
            **stats,
        })
        self._reset_episode_stats()

    # ------------------------------------------------------------------------------
    def reset_cars(self):
        """Put every car back on a (random) start position (R key)."""
        if self.t > 0:
            self.buf_done[self.t - 1] = 1.0   # the episodes in the rollout end here
        self.obs = self.env.reset_all()

    def display_state(self):
        e = self.env
        # Colour = progress in the current episode, relative to the car that has got furthest.
        frac = (e.dist / e.dist.max().clamp(min=1.0)).clamp(0, 1)
        best = int(e.dist.argmax())
        return {
            "x": e.x, "y": e.y, "yaw": e.yaw, "alive": e.alive,
            "color_value": frac,
            "best": best, "rays": e.ray_endpoints(best),
            "leader_speed": float(e.v[best]), "leader_dist": float(e.dist[best]),
            "hud": {
                "Algorithm": "PPO (reinforcement learning)",
                "Update": f"{self.iteration + 1}  (step {self.t}/{self.p.rollout_steps})",
                "Exploration std": f"{float(self.model.log_std.detach().exp().mean()):.3f}",
            },
            "curve": [(h["env_steps"], h["best_distance"]) for h in self.history],
            "best_dist_ever": max(self.best_dist_ever, float(e.best_dist.max())),
            "best_lap_ever": self.best_lap_ever,
        }

    def policy(self):
        return make_deterministic_policy(self.model, self.obs_rms, self.p.obs_clip)

    def state_dict(self):
        return {
            "model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
            "obs_rms": self.obs_rms.state_dict(), "hidden": self.p.hidden,
            "best_dist_ever": self.best_dist_ever, "best_lap_ever": self.best_lap_ever,
        }

    def load_state_dict(self, s):
        self.model.load_state_dict(s["model"])
        self.opt.load_state_dict(s["optimizer"])
        self.obs_rms.load_state_dict({k: v.to(self.device) for k, v in s["obs_rms"].items()})
        self.best_dist_ever = s["best_dist_ever"]
        self.best_lap_ever = s["best_lap_ever"]
        self.t = 0
        self._reset_episode_stats()
        self.reset_cars()


def make_deterministic_policy(model, obs_rms, obs_clip):
    """Drive with the mean action (no exploration noise)."""
    @torch.no_grad()
    def act(obs):
        mean, _ = model(obs_rms.normalize(obs, obs_clip))
        return mean.clamp(-1, 1)
    return act


def ppo_policy_from_checkpoint(state, cfg, device):
    model = ActorCritic(state.get("hidden", cfg.ppo.hidden)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    rms = RunningMeanStd(OBS_SIZE, device)
    rms.load_state_dict({k: v.to(device) for k, v in state["obs_rms"].items()})
    return make_deterministic_policy(model, rms, cfg.ppo.obs_clip)
