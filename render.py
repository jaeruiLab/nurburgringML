"""pygame viewer: watch the cars learn (or watch a saved driver).

Keys
  Space   pause / resume
  1 2 3   simulation speed 1x / 5x / 20x (steps per frame)
  4       maximum speed (the screen only refreshes twice a second)
  Tab     toggle full-track overview / follow camera
  V       show / hide the leader's raycasts
  R       put all cars back on the start line
  S       save a checkpoint
  Esc     quit (a checkpoint is saved when training)

Everything here is drawing code: the simulation itself stays in batched PyTorch,
and positions are copied to NumPy once per frame for pygame.
"""
import math
import os
import time

import numpy as np
import pygame

from utils import device_name, fmt_time

GRASS = (58, 94, 52)
GRASS_DARK = (48, 80, 44)
ASPHALT = (78, 79, 82)
EDGE = (232, 232, 226)
PANEL = (18, 18, 20, 200)
TEXT = (238, 238, 232)
TEXT_DIM = (170, 170, 162)
RAY = (255, 214, 64)
DEAD = (95, 95, 95)
# Colour ramp for car colouring (worst -> best): red, amber, green.
RAMP = np.array([(227, 73, 72), (237, 161, 0), (27, 175, 122)], dtype=float)
SPEEDS = {pygame.K_1: 1, pygame.K_2: 5, pygame.K_3: 20, pygame.K_4: 0}  # 0 = max


def ramp_colors(values):
    v = np.clip(np.asarray(values, dtype=float), 0, 1) * (len(RAMP) - 1)
    i = np.minimum(v.astype(int), len(RAMP) - 2)
    f = (v - i)[:, None]
    return (RAMP[i] * (1 - f) + RAMP[i + 1] * f).astype(int)


class Camera:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.cx, self.cy, self.zoom = 0.0, 0.0, 3.0

    def follow(self, x, y, zoom, dt, rate=3.0):
        a = 1.0 - math.exp(-rate * dt)          # smooth: move a fraction of the way each frame
        self.cx += (x - self.cx) * a
        self.cy += (y - self.cy) * a
        self.zoom += (zoom - self.zoom) * a

    def snap(self, x, y, zoom):
        self.cx, self.cy, self.zoom = x, y, zoom

    def to_screen(self, xy):
        """World meters (..., 2) -> screen pixels (y axis flipped: north is up)."""
        xy = np.asarray(xy, dtype=float)
        sx = (xy[..., 0] - self.cx) * self.zoom + self.w / 2
        sy = self.h / 2 - (xy[..., 1] - self.cy) * self.zoom
        return np.stack([sx, sy], axis=-1)

    def view_box(self, pad=50.0):
        hw, hh = self.w / 2 / self.zoom + pad, self.h / 2 / self.zoom + pad
        return self.cx - hw, self.cy - hh, self.cx + hw, self.cy + hh


class Renderer:
    def __init__(self, track, rcfg, title):
        pygame.init()
        pygame.display.set_caption(title)
        self.W, self.H = rcfg.width, rcfg.height
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 15)
        self.font_small = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 12)
        self.font_big = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 20, bold=True)
        self.font_label = pygame.font.SysFont("arial,helvetica,dejavusans", 13, bold=True)
        self.track = track
        g = track.numpy_geometry()
        self.center, self.left, self.right, self.normal = g["center"], g["left"], g["right"], g["normal"]
        self.length = track.length
        self.cam = Camera(self.W, self.H)
        self.overview = False
        self.show_rays = True

        # Pre-cut the track into chunks so we only draw the part that is on screen.
        M, step = len(self.center), 40
        self.chunks = []
        for i0 in range(0, M, step):
            idx = np.arange(i0, min(i0 + step, M - 1) + 2) % M
            poly = np.concatenate([self.left[idx], self.right[idx][::-1]])
            self.chunks.append((idx, poly, poly.min(0), poly.max(0)))

        lo = np.minimum(self.left.min(0), self.right.min(0))
        hi = np.maximum(self.left.max(0), self.right.max(0))
        self.world_lo, self.world_hi = lo, hi
        span = hi - lo
        # The overview map sits to the right of the learning-curve panel so nothing covers it.
        ov_x0, ov_x1, ov_y0, ov_y1 = 360, self.W - 30, 30, self.H - 40
        self.overview_zoom = min((ov_x1 - ov_x0) / span[0], (ov_y1 - ov_y0) / span[1])
        mid = (lo + hi) / 2
        # Shift the camera center so the track's middle lands in the middle of that area.
        self.overview_center = mid - np.array([((ov_x0 + ov_x1) / 2 - self.W / 2) / self.overview_zoom,
                                               -((ov_y0 + ov_y1) / 2 - self.H / 2) / self.overview_zoom])
        self.overview_surf = self._render_full_track(self.W, self.H, self.overview_zoom, self.overview_center,
                                                     labels=True)
        self.minimap_center = mid
        mm_zoom = min(230 / span[0], 170 / span[1])
        self.mm_w, self.mm_h = int(span[0] * mm_zoom) + 16, int(span[1] * mm_zoom) + 16
        self.mm_zoom = mm_zoom
        self.minimap_surf = self._render_full_track(self.mm_w, self.mm_h, mm_zoom, self.minimap_center,
                                                    labels=False, minimap=True)
        self.message, self.message_until = "", 0.0

    # ------------------------------------------------------------------------------
    def _render_full_track(self, w, h, zoom, center, labels, minimap=False):
        cam = Camera(w, h)
        cam.snap(center[0], center[1], zoom)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        if minimap:
            surf.fill((18, 18, 20, 190))
            width = 3
        else:
            surf.fill(GRASS)
            width = max(3, int(self.track.width * zoom))
        pts = cam.to_screen(self.center)
        pygame.draw.lines(surf, ASPHALT if not minimap else (150, 150, 150), True, pts.tolist(), width)
        s = cam.to_screen(np.stack([self.left[0], self.right[0]]))
        pygame.draw.line(surf, (230, 60, 60), s[0], s[1], 3)
        if labels:
            for c in self.track.corners:
                if not c["major"]:
                    continue
                i = c["index"]
                p = cam.to_screen(self.center[i] + self.normal[i] * 160)
                self._label(surf, c["name"], p, self.font_label)
        return surf

    def _label(self, surf, text, pos, font, color=TEXT):
        img = font.render(text, True, color)
        shadow = font.render(text, True, (0, 0, 0))
        r = img.get_rect(center=(int(pos[0]), int(pos[1])))
        r.clamp_ip(surf.get_rect().inflate(-8, -8))      # keep labels on screen
        surf.blit(shadow, r.move(1, 1))
        surf.blit(img, r)

    def flash(self, text, seconds=2.5):
        self.message, self.message_until = text, time.time() + seconds

    # ------------------------------------------------------------------------------
    def update_camera(self, x, y, speed, dt):
        if self.overview:
            self.cam.snap(self.overview_center[0], self.overview_center[1], self.overview_zoom)
            return
        # Zoom out as the leader speeds up so you can see what's coming.
        target_zoom = 7.0 / (1.0 + speed / 30.0)
        if self.cam.zoom < 0.5:                       # coming back from the overview
            self.cam.snap(x, y, target_zoom)
        self.cam.follow(x, y, target_zoom, dt)

    def draw_world(self):
        if self.overview:
            self.screen.blit(self.overview_surf, (0, 0))
            return
        self.screen.fill(GRASS)
        x0, y0, x1, y1 = self.cam.view_box()
        z = self.cam.zoom
        # Faint 50 m grid on the grass gives a sense of speed.
        gs = 50.0
        for gx in np.arange(math.floor(x0 / gs) * gs, x1, gs):
            sx = self.cam.to_screen(np.array([gx, 0.0]))[0]
            pygame.draw.line(self.screen, GRASS_DARK, (sx, 0), (sx, self.H))
        for gy in np.arange(math.floor(y0 / gs) * gs, y1, gs):
            sy = self.cam.to_screen(np.array([0.0, gy]))[1]
            pygame.draw.line(self.screen, GRASS_DARK, (0, sy), (self.W, sy))
        edge_w = max(1, int(0.4 * z))
        for idx, poly, lo, hi in self.chunks:
            if hi[0] < x0 or lo[0] > x1 or hi[1] < y0 or lo[1] > y1:
                continue
            pygame.draw.polygon(self.screen, ASPHALT, self.cam.to_screen(poly).tolist())
            pygame.draw.lines(self.screen, EDGE, False, self.cam.to_screen(self.left[idx]).tolist(), edge_w)
            pygame.draw.lines(self.screen, EDGE, False, self.cam.to_screen(self.right[idx]).tolist(), edge_w)
        # Start / finish line (chequered look: thick red/white).
        s = self.cam.to_screen(np.stack([self.left[0], self.right[0]]))
        pygame.draw.line(self.screen, (240, 240, 240), s[0], s[1], max(2, int(1.5 * z)))
        for c in self.track.corners:
            p = self.center[c["index"]] + self.normal[c["index"]] * (self.track.width / 2 + 12)
            if x0 < p[0] < x1 and y0 < p[1] < y1:
                self._label(self.screen, c["name"], self.cam.to_screen(p), self.font_label)

    def draw_cars(self, x, y, yaw, colors, alive, best, rays=None):
        if self.overview:
            pts = self.cam.to_screen(np.stack([x, y], 1))
            order = np.argsort(alive)              # dead first, alive on top
            for i in order:
                pygame.draw.circle(self.screen, tuple(colors[i]) if alive[i] else DEAD, pts[i], 4 if i == best else 3)
            pygame.draw.circle(self.screen, (255, 255, 255), pts[best], 7, 2)
            return
        L, Wd = 4.6, 2.0                          # car size in meters
        corners = np.array([[L / 2, Wd / 2], [L / 2, -Wd / 2], [-L / 2, -Wd / 2], [-L / 2, Wd / 2]])
        c, s = np.cos(yaw), np.sin(yaw)
        wx = x[:, None] + corners[None, :, 0] * c[:, None] - corners[None, :, 1] * s[:, None]
        wy = y[:, None] + corners[None, :, 0] * s[:, None] + corners[None, :, 1] * c[:, None]
        pts = self.cam.to_screen(np.stack([wx, wy], -1))             # (n, 4, 2)
        if rays is not None and self.show_rays:
            p0 = self.cam.to_screen(np.array([x[best], y[best]]))
            for e in self.cam.to_screen(rays):
                pygame.draw.line(self.screen, RAY, p0, e, 1)
                pygame.draw.circle(self.screen, RAY, e, 3)
        order = [i for i in np.argsort(alive) if i != best] + [best]
        for i in order:
            col = tuple(colors[i]) if alive[i] else DEAD
            pygame.draw.polygon(self.screen, col, pts[i].tolist())
            if i == best:
                pygame.draw.polygon(self.screen, (255, 255, 255), pts[i].tolist(), 2)

    def draw_minimap(self, x, y, colors, alive, best):
        if self.overview:
            return                                   # the whole screen is already a map
        ox, oy = self.W - self.mm_w - 12, 12
        self.screen.blit(self.minimap_surf, (ox, oy))
        cam = Camera(self.mm_w, self.mm_h)
        cam.snap(self.minimap_center[0], self.minimap_center[1], self.mm_zoom)
        pts = cam.to_screen(np.stack([x, y], 1)) + np.array([ox, oy])
        for i in np.argsort(alive):
            pygame.draw.circle(self.screen, tuple(colors[i]) if alive[i] else DEAD, pts[i], 2)
        pygame.draw.circle(self.screen, (255, 255, 255), pts[best], 4, 1)

    def draw_panel(self, lines, pos, width):
        h = 10 + 20 * len(lines)
        panel = pygame.Surface((width, h), pygame.SRCALPHA)
        panel.fill(PANEL)
        self.screen.blit(panel, pos)
        for k, (label, value) in enumerate(lines):
            y = pos[1] + 6 + 20 * k
            self.screen.blit(self.font.render(label, True, TEXT_DIM), (pos[0] + 10, y))
            self.screen.blit(self.font.render(str(value), True, TEXT), (pos[0] + 150, y))

    def draw_curve(self, curve, title="best distance per generation"):
        w, h = 330, 150
        ox, oy = 12, self.H - h - 40
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill(PANEL)
        self.screen.blit(panel, (ox, oy))
        self.screen.blit(self.font_small.render(title, True, TEXT_DIM), (ox + 8, oy + 5))
        px0, py0, pw, ph = ox + 10, oy + 24, w - 20, h - 36
        # Full lap reference line.
        pygame.draw.line(self.screen, (120, 120, 120), (px0, py0), (px0 + pw, py0), 1)
        self.screen.blit(self.font_small.render("full lap", True, (140, 140, 140)), (px0 + pw - 52, py0 + 2))
        if len(curve) >= 2:
            steps = np.array([c[0] for c in curve], dtype=float)
            dist = np.array([c[1] for c in curve], dtype=float)
            sx = px0 + pw * steps / max(steps[-1], 1)
            sy = py0 + ph - ph * np.clip(dist / self.length, 0, 1)
            pygame.draw.lines(self.screen, (57, 135, 229), False, np.stack([sx, sy], 1).tolist(), 2)
            best = np.maximum.accumulate(dist)
            sy2 = py0 + ph - ph * np.clip(best / self.length, 0, 1)
            pygame.draw.lines(self.screen, (235, 104, 52), False, np.stack([sx, sy2], 1).tolist(), 1)
            self.screen.blit(self.font_small.render(f"{steps[-1] / 1e6:.2f}M steps", True, TEXT_DIM),
                             (px0, py0 + ph + 1))
            label = f"best {best[-1] / 1000:.2f} km"
            self.screen.blit(self.font_small.render(label, True, TEXT), (px0 + pw - 8 * len(label), py0 + ph + 1))

    def draw_footer(self, text):
        img = self.font_small.render(text, True, TEXT_DIM)
        self.screen.blit(img, (12, self.H - 22))
        if time.time() < self.message_until:
            self._label(self.screen, self.message, (self.W / 2, 40), self.font_big, (255, 230, 120))


# ----------------------------------------------------------------------------------
# Training viewer
# ----------------------------------------------------------------------------------
def draw_training_frame(r, trainer, speed_text, dt):
    st = trainer.display_state()
    x, y, yaw = (st[k].detach().cpu().numpy() for k in ("x", "y", "yaw"))
    alive = st["alive"].cpu().numpy()
    colors = ramp_colors(st["color_value"].detach().cpu().numpy())
    best = st["best"]
    v_best = st["leader_speed"]
    r.update_camera(x[best], y[best], v_best, min(dt, 0.1))
    r.draw_world()
    r.draw_cars(x, y, yaw, colors, alive, best, None if st["rays"] is None else st["rays"].cpu().numpy())
    r.draw_minimap(x, y, colors, alive, best)
    L = trainer.track.length
    hud = list(st["hud"].items()) + [
        ("Env steps", f"{trainer.env_steps:,}"),
        ("Alive", f"{int(alive.sum())} / {len(alive)}"),
        ("Best distance", f"{st['best_dist_ever']:,.0f} m ({100 * st['best_dist_ever'] / L:.1f}%)"),
        ("Best lap", fmt_time(st["best_lap_ever"])),
        ("Leader", f"{v_best * 3.6:5.0f} km/h, {st['leader_dist']:,.0f} m"),
        ("Device", device_name(trainer.device)),
        ("Sim speed", speed_text),
    ]
    r.draw_panel(hud, (12, 12), 470)
    r.draw_curve(st["curve"], "best distance per " + ("generation" if trainer.algo == "ga" else "update"))
    r.draw_footer("Space pause | 1/2/3/4 speed | Tab overview | V rays | R reset | S save | Esc quit")


def run_training_viewer(trainer):
    cfg = trainer.cfg
    r = Renderer(trainer.track, cfg.render, f"Nürburgring Nordschleife - {trainer.algo.upper()}")
    save_every = cfg.save_every_generations if trainer.algo == "ga" else cfg.save_every_updates
    speed, paused = 1, False
    last_draw, last_frame = 0.0, time.time()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key in SPEEDS:
                    speed = SPEEDS[ev.key]
                elif ev.key in (pygame.K_TAB, pygame.K_o):
                    r.overview = not r.overview
                elif ev.key == pygame.K_v:
                    r.show_rays = not r.show_rays
                elif ev.key == pygame.K_r:
                    trainer.reset_cars()
                    r.flash("Cars reset to the start line")
                elif ev.key == pygame.K_s:
                    r.flash(f"Saved {trainer.save(numbered=True)}")

        if not paused:
            if speed == 0:      # max speed: simulate for ~100 ms between event checks
                t_end = time.time() + 0.1
                while time.time() < t_end:
                    if trainer.step() and trainer.iteration % save_every == 0:
                        trainer.save(numbered=True)
            else:
                for _ in range(speed):
                    if trainer.step() and trainer.iteration % save_every == 0:
                        trainer.save(numbered=True)

        now = time.time()
        if speed == 0 and not paused and now - last_draw < cfg.render.redraw_interval_max:
            continue
        dt, last_frame = now - last_frame, now
        last_draw = now
        draw_training_frame(r, trainer, "PAUSED" if paused else ("MAX" if speed == 0 else f"{speed}x"), dt)
        pygame.display.flip()
        if speed != 0 or paused:
            r.clock.tick(cfg.render.fps)
    path = trainer.save()
    print(f"Saved checkpoint: {path}")
    pygame.quit()


# ----------------------------------------------------------------------------------
# Watch a saved driver
# ----------------------------------------------------------------------------------
def run_watch(path, device):
    from env import VecEnv
    from main import load_policy
    from track import Track

    algo, policy, state, cfg = load_policy(path, device)
    track = Track(cfg.track, device)
    env = VecEnv(track, cfg, n_cars=1, device=device)
    obs = env.reset_all()
    r = Renderer(track, cfg.render, f"Watching {algo.upper()} driver - {os.path.basename(path)}")
    speed, paused, laps = 1, False, []
    last_frame = time.time()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key in SPEEDS:
                    speed = SPEEDS[ev.key] or 100
                elif ev.key in (pygame.K_TAB, pygame.K_o):
                    r.overview = not r.overview
                elif ev.key == pygame.K_v:
                    r.show_rays = not r.show_rays
                elif ev.key == pygame.K_r:
                    obs = env.reset_all()
        if not paused:
            for _ in range(speed):
                obs, _, term, trunc, info = env.step(policy(obs))
                if bool(term[0] | trunc[0]):
                    if bool(info["finished"][0]):
                        laps.append(env.lap_time[0].item())
                        r.flash(f"Lap complete: {fmt_time(laps[-1])}", 4)
                    else:
                        r.flash(f"Crashed after {env.best_dist[0].item():,.0f} m", 4)
                    obs = env.reset_all()
                    break
        now = time.time()
        dt, last_frame = now - last_frame, now
        x, y, yaw = env.x.cpu().numpy(), env.y.cpu().numpy(), env.yaw.cpu().numpy()
        v = float(env.v[0])
        r.update_camera(x[0], y[0], v, min(dt, 0.1))
        r.draw_world()
        frac = np.array([float(env.dist[0]) / track.length])
        r.draw_cars(x, y, yaw, ramp_colors(frac), np.array([True]), 0, env.ray_endpoints(0).cpu().numpy())
        r.draw_minimap(x, y, ramp_colors(frac), np.array([True]), 0)
        hud = [
            ("Driver", f"{algo.upper()} (deterministic)"),
            ("Lap time", fmt_time(float(env.ep_steps[0]) * env.dt)),
            ("Distance", f"{float(env.dist[0]):,.0f} / {track.length:,.0f} m"),
            ("Speed", f"{v * 3.6:.0f} km/h"),
            ("Last laps", ", ".join(fmt_time(t) for t in laps[-3:]) or "-"),
            ("Trained for", f"{state['env_steps']:,} env steps"),
            ("Sim speed", "PAUSED" if paused else f"{speed}x"),
        ]
        r.draw_panel(hud, (12, 12), 470)
        r.draw_footer("Space pause | 1/2/3/4 speed | Tab overview | V rays | R restart | Esc quit")
        pygame.display.flip()
        r.clock.tick(cfg.render.fps)
    pygame.quit()
