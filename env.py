"""Vectorised driving environment: N cars simulated at once with PyTorch tensors.

There is no `for car in cars:` loop anywhere. Every quantity is a tensor with one
entry per car (shape (N,) or (N, ...)), and each line of math updates all cars in
one go. That is what lets 100 cars run nearly as fast as one.

Per step:
  1. physics      - kinematic bicycle model with a grip limit (understeer)
  2. progress     - project each car onto the centerline -> distance travelled
  3. termination  - off track / backwards / stuck / lap finished / time limit
  4. reward       - progress + small speed term - crash penalty + lap bonus
  5. observation  - 9 raycasts + speed + heading error + upcoming curvature (13 numbers)

Two ways to use it:
  * GA:  reset_all() puts every car on the grid; dead cars freeze until the next reset_all().
  * PPO: step(..., auto_reset=True) respawns a dead car immediately (optionally at a random
         point on the track), so all 100 "environments" keep producing experience.
"""
import math
from types import SimpleNamespace

import torch

from track import wrap_angle

OBS_SIZE = 13
ACT_SIZE = 2


class VecEnv:
    # Every per-car state tensor (shape (N, ...)); used to shrink / regrow the batch.
    PER_CAR = ("x", "y", "yaw", "v", "seg_idx", "s", "dist", "best_dist", "prog_ref", "stall", "ep_steps",
               "ep_return", "speed_sum", "lap_time", "alive", "last_ray_m", "last_action")

    def __init__(self, track, cfg, n_cars=None, device="cpu", random_start=False, grid_start_prob=0.0):
        self.track = track
        self.cfg = cfg
        self.ecfg, self.car = cfg.env, cfg.car
        self.n_full = n_cars or cfg.env.n_cars
        self.device = torch.device(device)
        self.random_start = random_start
        self.grid_start_prob = grid_start_prob    # with random starts: chance a respawn uses the start line
        self.dt = self.ecfg.dt
        e, c = self.ecfg, self.car
        f32 = dict(dtype=torch.float32, device=self.device)

        # Ray directions relative to the car's heading, and the distances sampled along a ray
        # (spaced more densely near the car, where precision matters most).
        half_fov = math.radians(e.ray_fov_deg) / 2
        self.ray_angles = torch.linspace(-half_fov, half_fov, e.n_rays, **f32)
        frac = torch.arange(1, e.ray_samples + 1, **f32) / e.ray_samples
        self.ray_d = e.ray_max * frac ** e.ray_spacing_power                  # (K,) meters
        self.ray_max_t = torch.tensor(e.ray_max, **f32)
        self.win = torch.arange(-e.window_back, e.window_fwd + 1, device=self.device)
        self.idx_near = int(round(e.curv_ahead_near / track.ds))
        self.idx_far = int(round(e.curv_ahead_far / track.ds))
        self.no_progress_steps = int(round(e.no_progress_time / self.dt))
        self.max_steps = int(round(e.max_episode_time / self.dt))
        self.steer_k = c.steer_overdrive * c.wheelbase * c.grip

        # Every constant used in the per-step math, stored as a 0-dim tensor. (Multiplying a
        # tensor by a plain Python number makes PyTorch convert that number into a tensor
        # every single time; with ~40 constants per step that alone cost ~20% of the runtime.)
        T = lambda v, dtype=torch.float32: torch.tensor(v, dtype=dtype, device=self.device)
        self.k = SimpleNamespace(
            dt=T(self.dt), power=T(c.power_per_mass), max_brake=T(c.max_brake), drag=T(c.drag),
            rolling=T(c.rolling), steer_k=T(self.steer_k), wheelbase=T(c.wheelbase), grip=T(c.grip),
            scrub=T(c.scrub_decel), inv_v_max=T(1.0 / c.v_max),
            inv_cell=T(1.0 / track.grid_cell), grid_W=T(track.grid_W, torch.int32),
            ox=track.grid_origin[0].to(self.device), oy=track.grid_origin[1].to(self.device),
            ray_scale=T(1.0 / math.log1p(e.ray_max) if e.ray_log_scale else 1.0 / e.ray_max),
            heading_scale=T(1.0 / e.heading_err_scale), curv_scale=T(e.curv_scale),
            pi=T(math.pi), two_pi=T(2 * math.pi),
            L=T(track.length), half_L=T(track.length / 2), ds=T(track.ds), M=T(track.M, torch.long),
            idx_near=T(self.idx_near, torch.long), idx_far=T(self.idx_far, torch.long), one=T(1, torch.long),
            min_progress=T(e.min_progress), backwards=T(e.backwards_tolerance),
            no_progress=T(self.no_progress_steps, torch.long), max_steps=T(self.max_steps, torch.long),
            speed_coef=T(e.reward_speed_coef), crash=T(e.crash_penalty), lap_bonus=T(e.lap_bonus * e.ref_lap_time),
        )
        self._alloc(self.n_full)
        # Total experience consumed = number of (alive car, step) pairs. Kept on the device
        # so counting never forces the GPU to wait.
        self._steps = torch.zeros((), dtype=torch.long, device=self.device)

    def _alloc(self, N):
        """Create the per-car state tensors for N cars."""
        self.N = N
        e = self.ecfg
        f32 = dict(dtype=torch.float32, device=self.device)
        z = lambda: torch.zeros(N, **f32)
        self.x, self.y, self.yaw, self.v = z(), z(), z(), z()
        self.seg_idx = torch.zeros(N, dtype=torch.long, device=self.device)
        self.s = z()                 # arc length of the car's projection on the centerline
        self.dist = z()              # signed progress since spawn (m)
        self.best_dist = z()         # furthest progress reached this episode
        self.prog_ref = z()          # best_dist value when the stall timer was last reset
        self.stall = torch.zeros(N, dtype=torch.long, device=self.device)
        self.ep_steps = torch.zeros(N, dtype=torch.long, device=self.device)
        self.ep_return = z()
        self.speed_sum = z()
        self.lap_time = torch.full((N,), float("inf"), **f32)
        self.alive = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.last_ray_m = torch.full((N, e.n_rays), e.ray_max, **f32)   # ray hit distances (m), for drawing
        self.last_action = torch.zeros(N, ACT_SIZE, **f32)

    def select_cars(self, idx):
        """Keep only the cars at batch positions idx and stop simulating the rest.

        The GA uses this once most cars have crashed: 100 cars cost more per step than
        10, even if 90 of them are just standing still.
        """
        for name in self.PER_CAR:
            setattr(self, name, getattr(self, name)[idx])
        self.N = len(idx)

    # ------------------------------------------------------------------------------
    @property
    def total_steps(self):
        return int(self._steps.item())

    @total_steps.setter
    def total_steps(self, value):
        self._steps.fill_(int(value))

    def reset_all(self):
        """Everyone back on the start grid (GA: start of a generation)."""
        if self.N != self.n_full:
            self._alloc(self.n_full)
        self._spawn(torch.ones(self.N, dtype=torch.bool, device=self.device), random=self.random_start)
        return self.observe()

    @torch.no_grad()
    def _spawn(self, mask, random=False):
        """Reset the cars where mask is True: on the centerline, standing still, facing forward."""
        t = self.track
        if random:
            idx = torch.randint(0, t.M, (self.N,), device=self.device)
            if self.grid_start_prob > 0:
                at_grid = torch.rand(self.N, device=self.device) < self.grid_start_prob
                idx = torch.where(at_grid, torch.zeros_like(idx), idx)
        else:
            idx = torch.zeros(self.N, dtype=torch.long, device=self.device)
        w = lambda new, old: torch.where(mask, new, old)
        zero = torch.zeros_like(self.x)
        self.seg_idx = w(idx, self.seg_idx)
        self.x = w(t.center[idx, 0], self.x)
        self.y = w(t.center[idx, 1], self.y)
        self.yaw = w(t.heading[idx], self.yaw)
        self.s = w(t.cum_s[idx], self.s)
        for name in ("v", "dist", "best_dist", "prog_ref", "ep_return", "speed_sum"):
            setattr(self, name, w(zero, getattr(self, name)))
        self.stall = torch.where(mask, torch.zeros_like(self.stall), self.stall)
        self.ep_steps = torch.where(mask, torch.zeros_like(self.ep_steps), self.ep_steps)
        self.lap_time = w(torch.full_like(self.x, float("inf")), self.lap_time)
        self.alive = self.alive | mask

    # ------------------------------------------------------------------------------
    def _on_grid(self, gx, gy):
        """Occupancy-grid lookup for points given in grid cells (any shape).

        The grid has a border wider than the rays are long (and cars are removed the moment
        their center leaves the asphalt), so no bounds checks are needed; the clamp is only
        a safety net.
        """
        g = self.track.grid
        cell = (gy.int() * self.k.grid_W + gx.int()).clamp_(0, g.numel() - 1)
        return g[cell]

    @torch.no_grad()
    def observe(self):
        """Build the 13-number observation for every car.

        Batched raycasting: for every car (N), every ray (9) and every sample distance
        along the ray (K=128) we compute the grid cell at once -> an (N, 9, K) block of
        cell indices. One lookup in the occupancy grid says which samples are on asphalt;
        the first sample that is NOT on asphalt is where the ray hits the edge.
        """
        t, k = self.track, self.k
        gx0 = (self.x - k.ox) * k.inv_cell                                        # car position in cells
        gy0 = (self.y - k.oy) * k.inv_cell
        ang = self.yaw[:, None] + self.ray_angles                                 # (N, R)
        gx = (torch.cos(ang) * k.inv_cell)[:, :, None] * self.ray_d + gx0[:, None, None]  # (N, R, K)
        gy = (torch.sin(ang) * k.inv_cell)[:, :, None] * self.ray_d + gy0[:, None, None]
        on = self._on_grid(gx, gy)                                                # (N, R, K) bool
        first_off = on.view(torch.uint8).argmin(dim=2)     # first 0 (= off track) along each ray
        all_on = on.gather(2, first_off[..., None]).squeeze(2)   # no off-track sample at all
        hit_m = torch.where(all_on, self.ray_max_t, self.ray_d[first_off])        # (N, R) meters
        self.last_ray_m = hit_m
        # Scale to [0, 1]. A log scale gives near walls (the side rays see ~5 m) as much
        # resolution as far ones; with a plain d / 150 m they would all be squashed near 0.
        rays = (torch.log1p(hit_m) if self.ecfg.ray_log_scale else hit_m) * k.ray_scale

        speed = self.v * k.inv_v_max
        # Heading error wrapped into [-pi, pi], then scaled.
        heading_err = ((self.yaw - t.heading[self.seg_idx] + k.pi) % k.two_pi - k.pi) * k.heading_scale
        k_near = t.kappa_near[(self.seg_idx + k.idx_near) % k.M]
        k_far = t.kappa_far[(self.seg_idx + k.idx_far) % k.M]
        curv = (torch.stack([k_near, k_far], dim=1) * k.curv_scale).clamp(-1, 1)
        return torch.cat([rays, speed[:, None], heading_err[:, None], curv], dim=1)

    # ------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, action, auto_reset=False):
        """Advance every car by one timestep.

        action: (N, 2) tensor in [-1, 1]: [steer (+ = left), throttle (+) / brake (-)]
        Returns obs (N, 13), reward (N,), terminated (N,), truncated (N,), info (dict).
        """
        c, t, k = self.car, self.track, self.k
        alive = self.alive
        # In auto-reset mode every car is always alive, so no masking is needed.
        keep = (lambda new, old: new) if auto_reset else (lambda new, old: torch.where(alive, new, old))
        a = action.clamp(-1, 1)
        self.last_action = a
        steer, pedal = a[:, 0], a[:, 1]
        v = self.v

        # ---- 1. physics -----------------------------------------------------------
        # Longitudinal: engine push is limited by traction at low speed and by power
        # (a = P / v) at high speed; drag grows with v^2 and sets the top speed.
        traction = (k.power / v.clamp(min=1.0)).clamp(max=c.max_accel)
        acc = pedal.clamp(min=0) * traction + pedal.clamp(max=0) * k.max_brake - (k.drag * v * v + k.rolling)

        # Lateral: kinematic bicycle model. Steering angle delta makes the car turn at a
        # yaw rate of v * tan(delta) / wheelbase. Speed-sensitive steering (like most racing
        # games): at speed, full lock asks for `steer_overdrive` x the grip limit.
        max_delta = (k.steer_k / (v * v).clamp(min=1e-3)).clamp(max=c.max_steer)
        yaw_rate_cmd = v * torch.tan(steer * max_delta) / k.wheelbase
        # Grip limit: lateral acceleration (v * yaw_rate) cannot exceed mu*g. Asking for
        # more makes the front tyres slide: the car turns less than commanded (understeer)
        # and the sliding tyres scrub off speed.
        yaw_rate_max = k.grip / v.clamp(min=0.5)
        yaw_rate = torch.clamp(yaw_rate_cmd, -yaw_rate_max, yaw_rate_max)
        cmd_abs = yaw_rate_cmd.abs()
        slide = ((cmd_abs - yaw_rate_max) / cmd_abs.clamp(min=1e-6)).clamp(min=0)   # 0 = full grip
        acc = acc - k.scrub * slide

        v_new = (v + acc * k.dt).clamp(min=0.0)
        yaw_new = self.yaw + yaw_rate * k.dt
        step_len = v_new * k.dt
        # Dead cars (GA mode) stay frozen where they crashed.
        self.x = keep(torch.addcmul(self.x, step_len, torch.cos(yaw_new)), self.x)
        self.y = keep(torch.addcmul(self.y, step_len, torch.sin(yaw_new)), self.y)
        self.v = keep(v_new, v)
        self.yaw = keep(yaw_new, self.yaw)

        # ---- 2. progress along the centerline -------------------------------------
        # Each car only searches the ~21 segments around where it was last step, so this
        # is cheap: (N, 21) candidates instead of all ~20 000 segments.
        cand = (self.seg_idx[:, None] + self.win) % k.M                            # (N, W)
        a0 = t.center[cand]                                                        # (N, W, 2)
        d = t.seg_vec[cand]
        rel = torch.stack([self.x, self.y], dim=1)[:, None, :] - a0
        u = ((rel * d).sum(-1) / t.seg_len2[cand]).clamp(0.0, 1.0)                 # 0..1 along segment
        dist2 = (rel - u[..., None] * d).square().sum(-1)
        j = dist2.argmin(dim=1, keepdim=True)
        new_idx = cand.gather(1, j).squeeze(1)
        s_new = t.cum_s[new_idx] + u.gather(1, j).squeeze(1) * k.ds
        delta_s = (s_new - self.s + k.half_L) % k.L - k.half_L                    # wrap across start line
        delta_s = keep(delta_s, torch.zeros_like(delta_s))
        self.seg_idx = keep(new_idx, self.seg_idx)
        self.s = keep(s_new, self.s)
        self.dist = self.dist + delta_s
        self.best_dist = torch.maximum(self.best_dist, self.dist)

        # ---- 3. termination -------------------------------------------------------
        improved = self.best_dist >= self.prog_ref + k.min_progress
        self.prog_ref = torch.where(improved, self.best_dist, self.prog_ref)
        self.stall = torch.where(improved, 0, self.stall + k.one)
        self.ep_steps = self.ep_steps + alive
        self._steps += alive.sum()

        off_track = ~self._on_grid((self.x - k.ox) * k.inv_cell, (self.y - k.oy) * k.inv_cell)
        backwards = (self.best_dist - self.dist) > k.backwards
        stuck = self.stall >= k.no_progress
        finished = (self.dist >= k.L) & alive
        crashed = (off_track | backwards | stuck) & alive & ~finished
        terminated = crashed | finished
        truncated = (self.ep_steps >= k.max_steps) & alive & ~terminated
        self.lap_time = torch.where(finished, self.ep_steps * k.dt, self.lap_time)

        # ---- 4. reward (PPO uses it, the GA has its own fitness) -------------------
        lap_bonus = torch.where(finished, k.lap_bonus / self.lap_time, 0.0)
        reward = delta_s + k.speed_coef * self.v - crashed * k.crash + lap_bonus
        reward = keep(reward, torch.zeros_like(reward))
        self.ep_return = self.ep_return + reward
        self.speed_sum = self.speed_sum + keep(self.v, torch.zeros_like(self.v))

        done = terminated | truncated
        # State tensors are replaced (never modified in place), so these stay valid
        # after the respawn below.
        info = {
            "ep_done": done,
            "ep_distance": self.best_dist,
            "ep_return": self.ep_return,
            "ep_lap_time": self.lap_time,          # inf if no lap was completed
            "ep_len": self.ep_steps,
            "crashed": crashed,
            "finished": finished,
        }

        # ---- 5. reset / observe ---------------------------------------------------
        if auto_reset:
            if bool(done.any()):
                # A car that ran out of time did not fail; PPO still needs the value of the
                # state it was in, so grab that observation before respawning.
                if bool(truncated.any()):
                    info["final_obs"] = self.observe()
                self._spawn(done, random=self.random_start)
        else:
            self.alive = alive & ~done
        obs = self.observe()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------------------
    def ray_endpoints(self, i):
        """World coordinates of car i's ray hits (for drawing)."""
        ang = self.yaw[i] + self.ray_angles
        r = self.last_ray_m[i]
        return torch.stack([self.x[i] + torch.cos(ang) * r, self.y[i] + torch.sin(ang) * r], dim=1)


@torch.no_grad()
def scripted_driver(env, lookahead_brake=250.0, grip_margin=0.9):
    """A hand-written (not learned) driver that knows the track perfectly.

    Used to test that a full lap is drivable with this physics, and as a reference lap
    time. Steering: pure pursuit toward a point on the centerline a bit ahead.
    Speed: the slowest of 'corner speed limit at distance d, plus what braking over d
    can shed', for every d in the next `lookahead_brake` meters.
    """
    t, c = env.track, env.car
    v = env.v
    Ld = (6.0 + 0.4 * v).clamp(8.0, 40.0)
    target = t.center[(env.seg_idx + (Ld / t.ds).long()) % t.M]
    alpha = wrap_angle(torch.atan2(target[:, 1] - env.y, target[:, 0] - env.x) - env.yaw)
    delta = torch.atan(2 * torch.sin(alpha) / Ld * c.wheelbase)
    max_delta = (env.steer_k / (v * v).clamp(min=1e-3)).clamp(max=c.max_steer)
    steer = (delta / max_delta).clamp(-1, 1)

    d = torch.arange(0, int(lookahead_brake / t.ds), 2, device=env.device)        # (D,)
    kap = t.kappa_near[(env.seg_idx[:, None] + d) % t.M].abs().clamp(min=1e-4)
    v_corner = torch.sqrt(grip_margin * c.grip / kap)
    v_allow = torch.sqrt(v_corner ** 2 + 2 * 0.8 * c.max_brake * d * t.ds)
    pedal = ((v_allow.min(dim=1).values - v) * 0.5).clamp(-1, 1)
    return torch.stack([steer, pedal], dim=1)
