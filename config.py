"""Every tunable number in the project lives here.

Each dataclass groups the settings for one part of the system. The README has a
table explaining what each hyperparameter does and what happens if you change it.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class TrackConfig:
    path: str = "data/nordschleife.geojson"  # centerline file (GeoJSON lon/lat, or x/y meters)
    width: float = 11.0               # track width in meters (real Nordschleife: ~8-12 m)
    spline_ds: float = 1.0            # spacing (m) of the dense, smoothed centerline
    grid_cell: float = 0.5            # occupancy-grid cell size (m) used for raycasts / off-track checks
    grid_margin: float = 160.0        # empty border (m) around the track; >= ray length so rays never leave the grid
    curvature_smooth_near: float = 10.0  # window (m) for smoothing curvature looked up 50 m ahead
    curvature_smooth_far: float = 40.0   # window (m) for smoothing curvature looked up 150 m ahead


@dataclass
class CarConfig:
    wheelbase: float = 2.6            # distance between axles (m) for the bicycle model
    max_steer: float = 0.45           # max front-wheel angle (rad) at low speed (~26 degrees)
    steer_overdrive: float = 1.5      # at speed, full steering asks for this many times the grip limit
    max_accel: float = 7.0            # traction-limited acceleration (m/s^2)
    power_per_mass: float = 300.0     # engine power / mass (W/kg); acceleration is P/v at speed
    max_brake: float = 12.0           # braking deceleration at full brake (m/s^2)
    drag: float = 0.00055             # aerodynamic drag: decel = drag * v^2  (sets top speed ~80 m/s)
    rolling: float = 0.2              # rolling resistance decel (m/s^2)
    grip: float = 14.0                # max lateral acceleration mu*g (m/s^2), ~1.4 g
    scrub_decel: float = 8.0          # max extra decel (m/s^2) when the front tyres slide (understeer)
    v_max: float = 80.0               # used only to normalise the speed observation


@dataclass
class EnvConfig:
    n_cars: int = 100
    dt: float = 1.0 / 60.0            # physics timestep (s)
    n_rays: int = 9                   # raycasts spread from -90 to +90 degrees
    ray_fov_deg: float = 180.0
    ray_max: float = 150.0            # ray length (m)
    ray_samples: int = 128            # points tested along each ray
    ray_spacing_power: float = 1.5    # >1 puts more samples close to the car
    ray_log_scale: bool = True        # ray obs = log(1+d)/log(1+max): side rays (~5 m) get a useful range
    heading_err_scale: float = 0.785  # heading error (rad) that maps to an observation of 1.0 (pi/4)
    curv_ahead_near: float = 50.0     # look-ahead distances (m) for the curvature observations
    curv_ahead_far: float = 150.0
    curv_scale: float = 30.0          # curvature * this = observation (radius 30 m -> 1.0), clipped to [-1, 1]
    window_back: int = 5              # centerline segments searched behind / ahead of the car each step
    window_fwd: int = 15
    no_progress_time: float = 3.0     # die if progress improves by < min_progress within this many seconds
    min_progress: float = 5.0         # (creeping slower than ~1.7 m/s counts as stuck)
    backwards_tolerance: float = 10.0  # die if the car drives this far (m) backwards
    max_episode_time: float = 1200.0  # time limit (s of sim time) per episode / generation
    # Per-step reward (used by PPO; the GA ignores it and uses its own fitness).
    reward_speed_coef: float = 0.002  # reward += coef * speed (m/s) every step
    crash_penalty: float = 10.0
    lap_bonus: float = 100.0          # reward += lap_bonus * ref_lap_time / lap_time on finishing a lap
    ref_lap_time: float = 500.0       # reference lap time (s) for lap bonuses (the scripted driver laps in ~512 s)


@dataclass
class GAConfig:
    hidden: tuple = (16, 16)          # network: 13 -> 16 -> 16 -> 2, tanh everywhere
    elites: int = 5                   # best brains copied unchanged into the next generation
    tournament_k: int = 3             # brains per tournament when picking each parent
    crossover_prob: float = 0.5       # chance each gene comes from parent A (uniform crossover)
    mutation_prob: float = 1.0        # chance each gene gets Gaussian noise (1.0 = every gene, a little)
    sigma_init: float = 0.1           # starting mutation strength (std of the noise)
    sigma_decay: float = 0.995        # sigma *= decay every generation
    sigma_min: float = 0.01
    fitness_speed_coef: float = 0.2   # fitness += coef * average speed (m/s); small, breaks ties
    fitness_lap_bonus: float = 20000.0  # fitness += bonus * ref_lap_time / lap_time for a finished lap
    max_generation_time: float = 1200.0  # seconds of sim time before a generation is cut off
    normalize_obs: bool = True        # scale brain inputs to mean 0 / std 1 (stats updated per generation)
    obs_clip: float = 5.0             # normalised inputs are clipped to +-this


@dataclass
class PPOConfig:
    hidden: int = 64                  # actor-critic: 13 -> 64 -> 64, then policy head and value head
    rollout_steps: int = 256          # steps per car collected before each update (x100 cars)
    gamma: float = 0.99               # discount: how much future reward counts
    gae_lambda: float = 0.95          # GAE smoothing between 1-step and Monte-Carlo advantage
    epochs: int = 4                   # passes over each rollout
    minibatch_size: int = 4096
    clip_eps: float = 0.2             # PPO clip range for the probability ratio
    vf_coef: float = 0.5              # weight of the value loss
    ent_coef: float = 0.01            # weight of the entropy bonus (encourages exploration)
    lr: float = 3e-4                  # Adam learning rate
    max_grad_norm: float = 0.5        # gradient clipping
    init_log_std: float = -0.5        # starting exploration noise (std = e^-0.5 ~ 0.61)
    init_throttle_bias: float = 0.5   # starting mean throttle; 0 = cars mostly sit still at first
    obs_clip: float = 10.0            # normalised observations are clipped to +-this
    reward_scale: float = 1.0         # rewards are multiplied by this before learning
    norm_adv: bool = True             # normalise advantages per minibatch
    random_start: bool = True         # respawn crashed cars at a random point on the track


@dataclass
class CompareConfig:
    budget: int = 50_000_000          # env steps per run
    seeds: tuple = (0, 1, 2)
    out_dir: str = "results"
    grid_points: int = 200            # x-axis resolution of the averaged curves


@dataclass
class RenderConfig:
    width: int = 1280
    height: int = 800
    fps: int = 60
    redraw_interval_max: float = 0.5  # seconds between redraws in "max speed" mode


@dataclass
class Config:
    track: TrackConfig = field(default_factory=TrackConfig)
    car: CarConfig = field(default_factory=CarConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    compare: CompareConfig = field(default_factory=CompareConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    seed: int = 0
    device: str = "auto"              # auto | cpu | cuda | mps
    checkpoint_dir: str = "checkpoints"
    save_every_generations: int = 10
    save_every_updates: int = 50

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        def build(cls, sub):
            fields = cls.__dataclass_fields__
            kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in sub.items() if k in fields}
            return cls(**kwargs)
        cfg = Config()
        for name, cls in [("track", TrackConfig), ("car", CarConfig), ("env", EnvConfig), ("ga", GAConfig),
                          ("ppo", PPOConfig), ("compare", CompareConfig), ("render", RenderConfig)]:
            if name in d:
                setattr(cfg, name, build(cls, d[name]))
        for k in ("seed", "device", "checkpoint_dir", "save_every_generations", "save_every_updates"):
            if k in d:
                setattr(cfg, k, d[k])
        return cfg
