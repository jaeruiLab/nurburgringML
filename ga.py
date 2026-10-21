"""Neuroevolution: a genetic algorithm that breeds better driving brains.

The idea, generation by generation:
  1. Evaluate:  all 100 cars start on the grid at once, each driven by its own brain,
                until every car has crashed (or the time limit hits).
  2. Score:     fitness = how far along the track it got
                          + a small bonus for driving fast
                          + a big bonus for finishing a lap (more for a faster lap).
  3. Breed:     the 5 best brains (elites) are copied unchanged. The other 95 are
                children: pick two parents by tournament, mix their genes (crossover),
                then randomly nudge some genes (mutation).
  4. Repeat.    Nobody computes a gradient. Good brains simply have more children.

Input normalisation: the brains see each observation scaled to roughly mean 0 /
std 1, using statistics gathered from the cars' own experience. The statistics are
refreshed between generations and frozen during one, so every car in a generation
is judged on exactly the same inputs. (Without it, small-range inputs such as the
heading error barely register and evolution stalls at the first corner.)

All of the breeding is done with tensor operations on the (100, G) gene matrix, so
no Python loop over cars is needed anywhere.
"""
import math

import torch

from brain import PopulationBrain
from env import ACT_SIZE, OBS_SIZE, VecEnv
from trainer import BaseTrainer
from utils import RunningMeanStd

# Per-car results kept for the whole population (also for cars no longer simulated).
RESULT_KEYS = ("best_dist", "speed_sum", "ep_steps", "lap_time", "x", "y", "yaw", "alive")


def fitness(best_dist, speed_sum, ep_steps, lap_time, cfg, track_length):
    """GA fitness: distance + small bonus for average speed + big bonus for a (fast) lap."""
    g = cfg.ga
    dist = best_dist.clamp(0, track_length)
    avg_speed = speed_sum / ep_steps.clamp(min=1)
    lap_bonus = torch.where(torch.isfinite(lap_time), g.fitness_lap_bonus * cfg.env.ref_lap_time / lap_time, 0.0)
    return dist + g.fitness_speed_coef * avg_speed + lap_bonus


class GATrainer(BaseTrainer):
    algo = "ga"

    def __init__(self, cfg, device, seed=0):
        super().__init__(cfg, device, seed)
        self.ga = cfg.ga
        self.env = VecEnv(self.track, cfg, n_cars=cfg.env.n_cars, device=device, random_start=False)
        self.P = cfg.env.n_cars
        self.brain = PopulationBrain(self.P, (OBS_SIZE, *cfg.ga.hidden, ACT_SIZE), device)
        self.sigma = cfg.ga.sigma_init
        self.max_gen_steps = int(round(cfg.ga.max_generation_time / cfg.env.dt))
        self.gen_step = 0
        self.fitness = torch.zeros(self.P, device=device)       # last finished generation
        self.best_genes = self.brain.genes[0].clone()
        self.best_fitness = -math.inf
        self.best_dist_ever = 0.0
        self.best_lap_ever = math.inf
        # Input normalisation (frozen during a generation; see module docstring).
        self.obs_rms = RunningMeanStd(OBS_SIZE, device)
        self.norm_mean = torch.zeros(OBS_SIZE, device=device)
        self.norm_std = torch.ones(OBS_SIZE, device=device)
        self.best_norm = (self.norm_mean.clone(), self.norm_std.clone())
        self.new_generation()

    def new_generation(self):
        """Put the whole population on the start grid."""
        self.gen_step = 0
        self._reset_obs_moments()
        self.obs = self.env.reset_all()
        # car_ids[i] = which member of the population sits in batch slot i. At first every
        # car is simulated; once most have crashed the batch shrinks (see compact()).
        self.car_ids = torch.arange(self.P, device=self.device)
        self.batch_genes = self.brain.genes
        self.res = {k: getattr(self.env, k).clone() for k in RESULT_KEYS}

    def store_results(self):
        """Copy the stats of the cars in the batch into the full-population arrays."""
        for k in RESULT_KEYS:
            self.res[k][self.car_ids] = getattr(self.env, k)

    def compact(self):
        """Stop simulating crashed cars: keep only the alive ones in the batch."""
        self.store_results()
        keep = self.env.alive.nonzero().squeeze(1)
        self.env.select_cars(keep)
        self.obs = self.obs[keep]
        self.car_ids = self.car_ids[keep]
        self.batch_genes = self.brain.genes[self.car_ids]

    def _reset_obs_moments(self):
        z = lambda *s: torch.zeros(s, device=self.device)
        self.m_sum, self.m_sumsq, self.m_count = z(OBS_SIZE), z(OBS_SIZE), z()

    def normalized(self, obs):
        if not self.ga.normalize_obs:
            return obs
        return ((obs - self.norm_mean) / self.norm_std).clamp(-self.ga.obs_clip, self.ga.obs_clip)

    # ------------------------------------------------------------------------------
    def current_fitness(self):
        """Fitness of every car so far this generation (also used to colour cars live)."""
        self.store_results()
        r = self.res
        return fitness(r["best_dist"], r["speed_sum"], r["ep_steps"], r["lap_time"], self.cfg, self.track.length)

    def _step(self):
        """One simulation step for the whole population. Returns True if a generation ended."""
        if self.ga.normalize_obs:
            # Collect statistics of what the alive cars see (sums only: no GPU->CPU sync).
            w = self.env.alive.float()[:, None]
            self.m_sum += (self.obs * w).sum(0)
            self.m_sumsq += (self.obs.square() * w).sum(0)
            self.m_count += w.sum()
        actions = self.brain(self.normalized(self.obs), self.batch_genes)
        self.obs, _, _, _, _ = self.env.step(actions)
        self.gen_step += 1
        if self.gen_step >= self.max_gen_steps:
            self.end_generation()
            return True
        # Counting the survivors needs the CPU to read a GPU value, so only do it every 30 steps.
        if self.gen_step % 30 == 0:
            n_alive = int(self.env.alive.sum())
            if n_alive == 0:
                self.end_generation()
                return True
            if n_alive <= self.env.N // 2:
                self.compact()
        return False

    def end_generation(self):
        fit = self.current_fitness()
        r = self.res
        laps = torch.isfinite(r["lap_time"])
        best = int(fit.argmax())
        best_dist_gen = float(r["best_dist"].max().clamp(max=self.track.length))
        best_lap_gen = float(r["lap_time"].min()) if bool(laps.any()) else math.inf
        if float(fit[best]) > self.best_fitness:
            self.best_fitness = float(fit[best])
            self.best_genes = self.brain.genes[best].clone()
            self.best_norm = (self.norm_mean.clone(), self.norm_std.clone())
        self.best_dist_ever = max(self.best_dist_ever, best_dist_gen)
        self.best_lap_ever = min(self.best_lap_ever, best_lap_gen)
        self.fitness = fit
        self.iteration += 1
        self.history.append({
            "iteration": self.iteration,
            "env_steps": self.env_steps,
            "wall_time": round(self.wall_time, 2),
            "best_distance": best_dist_gen,
            "mean_distance": float(r["best_dist"].clamp(0, self.track.length).mean()),
            "best_distance_ever": self.best_dist_ever,
            "best_lap_time": best_lap_gen,
            "best_lap_time_ever": self.best_lap_ever,
            "laps_completed": int(laps.sum()),
            "best_fitness": float(fit[best]),
            "mean_fitness": float(fit.mean()),
            "sigma": self.sigma,
            "gen_seconds": self.gen_step * self.cfg.env.dt,
        })
        self.evolve(fit)
        self.sigma = max(self.ga.sigma_min, self.sigma * self.ga.sigma_decay)
        self.update_normalization()
        self.new_generation()

    def update_normalization(self):
        if not self.ga.normalize_obs or float(self.m_count) < 1:
            return
        mean = self.m_sum / self.m_count
        var = (self.m_sumsq / self.m_count - mean.square()).clamp(min=0)
        self.obs_rms.update_from_moments(mean, var, self.m_count)
        self.norm_mean = self.obs_rms.mean.clone()
        self.norm_std = torch.sqrt(self.obs_rms.var + 1e-8)
        self._reset_obs_moments()

    @torch.no_grad()
    def evolve(self, fitness):
        """Make the next generation's genes from this generation's genes and fitness."""
        g = self.ga
        genes = self.brain.genes
        P, G = genes.shape
        n_children = P - g.elites

        # Elitism: the best few brains survive unchanged, so good solutions are never lost
        # (their score can still shift a little when the input normalisation is refreshed).
        order = fitness.argsort(descending=True)
        elites = genes[order[:g.elites]]

        # Tournament selection: for each child, draw k random brains twice; the fittest of
        # each group of k becomes a parent. Fitter brains win more tournaments, so they
        # get more children, but weaker ones still get a chance (keeps diversity).
        contenders = torch.randint(0, P, (n_children, 2, g.tournament_k), device=genes.device)
        winner = fitness[contenders].argmax(dim=2, keepdim=True)
        parents = contenders.gather(2, winner).squeeze(2)                 # (n_children, 2)
        mom, dad = genes[parents[:, 0]], genes[parents[:, 1]]

        # Uniform crossover: every gene is copied from mom or dad by a coin flip.
        from_mom = torch.rand(n_children, G, device=genes.device) < g.crossover_prob
        children = torch.where(from_mom, mom, dad)

        # Gaussian mutation: genes (each with probability mutation_prob) get a random nudge
        # drawn from a bell curve with standard deviation sigma.
        # sigma shrinks slowly over the generations: big jumps early, fine-tuning later.
        mutate = torch.rand(n_children, G, device=genes.device) < g.mutation_prob
        children = children + mutate * torch.randn(n_children, G, device=genes.device) * self.sigma

        self.brain.genes = torch.cat([elites, children], dim=0)

    # ------------------------------------------------------------------------------
    def reset_cars(self):
        """Restart the current generation from the grid (R key)."""
        self.new_generation()

    def display_state(self):
        e, r = self.env, self.res
        fit = self.current_fitness()
        rank = torch.empty_like(fit)
        rank[fit.argsort(descending=True)] = torch.linspace(0, 1, self.P, device=fit.device)
        # Leader = fittest car still driving (or fittest overall once everyone crashed).
        if bool(e.alive.any()):
            slot = int(torch.where(e.alive, fit[self.car_ids], -math.inf).argmax())
            best, rays = int(self.car_ids[slot]), e.ray_endpoints(slot)
            speed, dist = float(e.v[slot]), float(e.dist[slot])
        else:
            best = int(fit.argmax())
            rays, speed, dist = None, 0.0, float(r["best_dist"][best])
        return {
            "x": r["x"], "y": r["y"], "yaw": r["yaw"], "alive": r["alive"],
            "color_value": 1.0 - rank,       # 1 = best rank
            "best": best, "rays": rays, "leader_speed": speed, "leader_dist": dist,
            "hud": {
                "Algorithm": "Genetic algorithm",
                "Generation": f"{self.iteration + 1}  (t = {self.gen_step * self.cfg.env.dt:5.1f} s)",
                "Mutation sigma": f"{self.sigma:.3f}",
            },
            "curve": [(h["env_steps"], h["best_distance"]) for h in self.history],
            "best_dist_ever": max(self.best_dist_ever, float(r["best_dist"].max())),
            "best_lap_ever": self.best_lap_ever,
        }

    def policy(self):
        """Deterministic driver from the best brain found so far (for --watch / evaluation)."""
        return make_ga_policy(self.best_genes, self.brain.sizes, self.best_norm,
                              self.ga.obs_clip if self.ga.normalize_obs else None, self.device)

    # ------------------------------------------------------------------------------
    def state_dict(self):
        return {
            "genes": self.brain.genes, "sigma": self.sigma, "fitness": self.fitness,
            "best_genes": self.best_genes, "best_fitness": self.best_fitness,
            "best_dist_ever": self.best_dist_ever, "best_lap_ever": self.best_lap_ever,
            "layer_sizes": self.brain.sizes, "obs_rms": self.obs_rms.state_dict(),
            "norm": (self.norm_mean, self.norm_std), "best_norm": self.best_norm,
            "obs_clip": self.ga.obs_clip if self.ga.normalize_obs else None,
        }

    def load_state_dict(self, s):
        genes = s["genes"].to(self.device)
        if genes.shape[0] != self.P:
            # Population size changed (--cars): fill the new population by cycling through
            # the saved brains (the first rows are the elites, best first).
            genes = genes[torch.arange(self.P, device=self.device) % genes.shape[0]]
        self.brain.genes = genes
        self.sigma = s["sigma"]
        self.fitness = torch.zeros(self.P, device=self.device)
        self.best_genes = s["best_genes"].to(self.device)
        self.best_fitness = s["best_fitness"]
        self.best_dist_ever = s["best_dist_ever"]
        self.best_lap_ever = s["best_lap_ever"]
        self.obs_rms.load_state_dict({k: v.to(self.device) for k, v in s["obs_rms"].items()})
        self.norm_mean, self.norm_std = (t.to(self.device) for t in s["norm"])
        self.best_norm = tuple(t.to(self.device) for t in s["best_norm"])
        self.reset_cars()


def make_ga_policy(genes, layer_sizes, norm, obs_clip, device):
    """A 1-car driver: one brain plus the input normalisation it was evaluated with."""
    brain = PopulationBrain(1, layer_sizes, device, genes=genes[None].to(device).clone())
    mean, std = (t.to(device) for t in norm)

    @torch.no_grad()
    def act(obs):
        if obs_clip is not None:
            obs = ((obs - mean) / std).clamp(-obs_clip, obs_clip)
        return brain(obs)
    return act


def ga_policy_from_checkpoint(state, device):
    """Build a 1-car deterministic driver from a saved GA checkpoint."""
    return make_ga_policy(state["best_genes"], state["layer_sizes"], state["best_norm"],
                          state.get("obs_clip"), device)
