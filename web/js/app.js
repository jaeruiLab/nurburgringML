// Replay viewer for the Nordschleife GA-vs-PPO project. Plain JS + Canvas, no build step.
import { loadReplay } from "./decode.js";

const RAMP = [[227, 73, 72], [237, 161, 0], [27, 175, 122]];      // car colours: worst -> best rank
const GRASS = "#3a5e34";
const ASPHALT = "#55575b";
const EDGE = "#ecebe4";
const STOPPED = "rgba(150, 150, 150, 0.75)";
const FINISHED = "#f2c14e";
const TRAIL_SECONDS = 12;
const CAR_LEN = 4.6, CAR_WID = 2.0;                                 // meters
const MODES = ["ga", "ppo", "split"];
const SPEEDS = [1, 4, 16];

const state = { mode: "split", pair: 0, t: 0, playing: false, speed: 4, camera: "follow" };
const cache = new Map();                 // replay id -> Promise<cars>
let manifest, track, views, stats;
let idle = 0;                            // frames drawn since the last change while paused

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const algoColor = (algo) => cssVar(algo === "ga" ? "--ga" : "--ppo");

// ------------------------------------------------------------------------------------
// formatting
// ------------------------------------------------------------------------------------
function fmtSteps(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 1 : 2)} M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} k`;
  return `${n}`;
}
const fmtDist = (m) => (m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${Math.round(m)} m`);
function fmtTime(s, decimals = 1) {
  if (s == null || !isFinite(s)) return "—";
  const p = 10 ** decimals;
  const total = Math.round(s * p) / p;                              // round first: never "1:60.0"
  const m = Math.floor(total / 60);
  return `${m}:${(total - 60 * m).toFixed(decimals).padStart(decimals ? 3 + decimals : 2, "0")}`;
}

// ------------------------------------------------------------------------------------
// track geometry (built once from data/track.json)
// ------------------------------------------------------------------------------------
function buildTrack(data) {
  const q = data.quant, c = data.center, n = c.length / 2;
  const x = new Float64Array(n), y = new Float64Array(n);
  for (let i = 0; i < n; i++) { x[i] = c[2 * i] * q; y[i] = c[2 * i + 1] * q; }
  const nx = new Float64Array(n), ny = new Float64Array(n);
  const lx = new Float64Array(n), ly = new Float64Array(n), rx = new Float64Array(n), ry = new Float64Array(n);
  const half = data.width / 2;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    const a = (i - 1 + n) % n, b = (i + 1) % n;
    const tx = x[b] - x[a], ty = y[b] - y[a], len = Math.hypot(tx, ty) || 1;
    nx[i] = -ty / len; ny[i] = tx / len;                          // normal pointing left
    lx[i] = x[i] + nx[i] * half; ly[i] = y[i] + ny[i] * half;
    rx[i] = x[i] - nx[i] * half; ry[i] = y[i] - ny[i] * half;
    minX = Math.min(minX, lx[i], rx[i]); maxX = Math.max(maxX, lx[i], rx[i]);
    minY = Math.min(minY, ly[i], ry[i]); maxY = Math.max(maxY, ly[i], ry[i]);
  }
  // Cut the track into chunks so the follow camera only draws what is on screen.
  const chunks = [];
  const size = 40;
  for (let i0 = 0; i0 < n; i0 += size) {
    const idx = [];
    for (let i = i0; i <= Math.min(i0 + size, n); i++) idx.push(i % n);
    let cx0 = Infinity, cy0 = Infinity, cx1 = -Infinity, cy1 = -Infinity;
    for (const i of idx) {
      cx0 = Math.min(cx0, lx[i], rx[i]); cx1 = Math.max(cx1, lx[i], rx[i]);
      cy0 = Math.min(cy0, ly[i], ry[i]); cy1 = Math.max(cy1, ly[i], ry[i]);
    }
    chunks.push({ idx, box: [cx0, cy0, cx1, cy1] });
  }
  const corners = data.corners.map((k) => ({ ...k, x: x[k.i], y: y[k.i], nx: nx[k.i], ny: ny[k.i] }));
  return {
    n, x, y, nx, ny, lx, ly, rx, ry, chunks, corners, width: data.width, length: data.length,
    bounds: { x0: minX, y0: minY, x1: maxX, y1: maxY, w: maxX - minX, h: maxY - minY },
  };
}

function trackPath(ctx, idxList) {
  // Adds one asphalt polygon (left edge forward, right edge backward) to the current path.
  idxList.forEach((i, k) => (k ? ctx.lineTo(track.lx[i], track.ly[i]) : ctx.moveTo(track.lx[i], track.ly[i])));
  for (let k = idxList.length - 1; k >= 0; k--) ctx.lineTo(track.rx[idxList[k]], track.ry[idxList[k]]);
  ctx.closePath();
}

function edgePath(ctx, idxList, xs, ys) {
  ctx.beginPath();
  idxList.forEach((i, k) => (k ? ctx.lineTo(xs[i], ys[i]) : ctx.moveTo(xs[i], ys[i])));
}

// ------------------------------------------------------------------------------------
// one replay view (canvas + camera)
// ------------------------------------------------------------------------------------
function rampColor(v) {
  const s = Math.max(0, Math.min(1, v)) * (RAMP.length - 1);
  const i = Math.min(Math.floor(s), RAMP.length - 2), f = s - i;
  const c = RAMP[i].map((a, k) => Math.round(a + (RAMP[i + 1][k] - a) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

class View {
  constructor(el) {
    this.el = el;
    this.algo = el.dataset.algo;
    this.canvas = $("canvas", el);
    this.ctx = this.canvas.getContext("2d");
    this.badge = $(".badge", el);
    this.overlay = $(".overlay", el);
    this.cam = { x: 0, y: 0, z: 1 };
    this.meta = null;
    this.cars = null;
    this.snap = true;
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
  }

  resize() {
    const r = this.el.getBoundingClientRect();
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = Math.max(2, Math.round(r.width * this.dpr));
    this.h = Math.max(2, Math.round(r.height * this.dpr));
    this.canvas.width = this.w;
    this.canvas.height = this.h;
    this.bg = null;
    this.snap = true;
    idle = 0;
  }

  setReplay(meta, loaded) {
    this.meta = { ...meta, ...loaded.info };                      // summary + per-car results
    this.cars = loaded.cars;
    meta = this.meta;
    const N = meta.n_cars;
    // Rank by score: colour green (best) -> red (worst); draw the best cars last (on top).
    const order = [...Array(N).keys()].sort((a, b) => meta.score[a] - meta.score[b]);
    this.order = order;
    this.colors = new Array(N);
    order.forEach((car, rank) => { this.colors[car] = rampColor(N > 1 ? rank / (N - 1) : 1); });
    this.stopped = meta.end_frames.map((e, i) => (meta.crashed[i] || meta.finished[i] ? e : Infinity));
    this.snap = true;
    this.setBadge(meta);
  }

  setBadge(meta) {
    this.badge.textContent = `${meta.algo.toUpperCase()} · ${meta.title} · ${fmtSteps(meta.env_steps)} steps`;
  }

  showMessage(text) {
    this.overlay.hidden = !text;
    this.overlay.textContent = text || "";
  }

  // Car i at fractional frame f: interpolated position / heading, and whether it stopped.
  carAt(i, f) {
    const c = this.cars[i];
    const last = c.n - 1;
    let x, y, h;
    if (f >= last) { x = c.x[last]; y = c.y[last]; h = c.h[last]; }
    else {
      const k = Math.max(0, Math.floor(f)), a = Math.max(0, f - k);
      x = c.x[k] + (c.x[k + 1] - c.x[k]) * a;
      y = c.y[k] + (c.y[k + 1] - c.y[k]) * a;
      let dh = c.h[k + 1] - c.h[k];
      if (dh > Math.PI) dh -= 2 * Math.PI; else if (dh < -Math.PI) dh += 2 * Math.PI;
      h = c.h[k] + dh * a;
    }
    return { x, y, h, stopped: f >= this.stopped[i] };
  }

  speedAt(i, f) {
    const c = this.cars[i];
    const k = Math.min(Math.max(0, Math.floor(f)), c.n - 2);
    if (k < 0 || f >= this.stopped[i]) return 0;
    return Math.hypot(c.x[k + 1] - c.x[k], c.y[k + 1] - c.y[k]) * manifest.fps;
  }

  overviewCam() {
    const b = track.bounds;
    const z = Math.min(this.w / (b.w * 1.08), this.h / (b.h * 1.08));
    return { x: (b.x0 + b.x1) / 2, y: (b.y0 + b.y1) / 2, z };
  }

  update(t, dt) {
    if (state.camera === "overview" || !this.cars) {
      Object.assign(this.cam, this.overviewCam());
      this.snap = true;
      return;
    }
    const f = t * manifest.fps, best = this.meta.best;
    const p = this.carAt(best, f);
    const base = Math.min(this.w, this.h) / 75;                     // ~75 m across when standing still
    const z = base / (1 + this.speedAt(best, f) / 30);             // zoom out at speed
    if (this.snap) { this.cam = { x: p.x, y: p.y, z }; this.snap = false; return; }
    const a = 1 - Math.exp(-dt * 4);
    this.cam.x += (p.x - this.cam.x) * a;
    this.cam.y += (p.y - this.cam.y) * a;
    this.cam.z += (z - this.cam.z) * Math.min(1, a * 0.6);
  }

  worldTransform(ctx) {
    const { x, y, z } = this.cam;
    ctx.setTransform(z, 0, 0, -z, this.w / 2 - x * z, this.h / 2 + y * z);    // y axis points up
  }

  toScreen(wx, wy) {
    return [(wx - this.cam.x) * this.cam.z + this.w / 2, this.h / 2 - (wy - this.cam.y) * this.cam.z];
  }

  label(ctx, text, sx, sy, size, placed = null) {
    ctx.font = `600 ${size}px system-ui, -apple-system, "Segoe UI", sans-serif`;
    const w = ctx.measureText(text).width;
    const m = 6 * this.dpr;
    sx = Math.max(m + w / 2, Math.min(this.w - m - w / 2, sx));
    sy = Math.max(m + size, Math.min(this.h - m, sy));
    if (placed) {
      // Skip labels that would overlap one already drawn (or the badge in the corner).
      const box = [sx - w / 2 - 3, sy - size - 2, sx + w / 2 + 3, sy + 4];
      if (placed.some((b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) return;
      placed.push(box);
    }
    ctx.textAlign = "center";
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillText(text, sx + this.dpr, sy + this.dpr);
    ctx.fillStyle = "#f4f4ef";
    ctx.fillText(text, sx, sy);
  }

  renderOverviewBackground() {
    // The full-track picture never changes, so draw it once per canvas size.
    const bg = document.createElement("canvas");
    bg.width = this.w; bg.height = this.h;
    const ctx = bg.getContext("2d");
    ctx.fillStyle = GRASS;
    ctx.fillRect(0, 0, this.w, this.h);
    this.worldTransform(ctx);
    const all = [...Array(track.n).keys(), 0];
    ctx.beginPath();
    trackPath(ctx, all);
    ctx.fillStyle = ASPHALT;
    ctx.fill();
    // At this zoom the 11 m road is only ~2 px wide, so draw it as a bold outlined line.
    edgePath(ctx, all, track.x, track.y);
    ctx.lineJoin = "round";
    ctx.lineWidth = Math.max(track.width, 6 * this.dpr / this.cam.z);
    ctx.strokeStyle = "#26272a";
    ctx.stroke();
    ctx.lineWidth = Math.max(track.width, 3.5 * this.dpr / this.cam.z);
    ctx.strokeStyle = "#7d7f84";
    ctx.stroke();
    this.startLine(ctx, 3 * this.dpr / this.cam.z);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const size = Math.round((this.w / this.dpr < 500 ? 10.5 : 12) * this.dpr);
    const badge = this.badge.getBoundingClientRect();
    const placed = [[0, 0, (badge.width + 16) * this.dpr, (badge.height + 14) * this.dpr]];
    for (const k of track.corners) {
      if (!k.major) continue;
      const off = 26 * this.dpr / this.cam.z;
      const [sx, sy] = this.toScreen(k.x + k.nx * off, k.y + k.ny * off);
      this.label(ctx, k.name, sx, sy, size, placed);
    }
    this.bg = bg;
  }

  startLine(ctx, width) {
    ctx.beginPath();
    ctx.moveTo(track.lx[0], track.ly[0]);
    ctx.lineTo(track.rx[0], track.ry[0]);
    ctx.lineWidth = width;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
  }

  drawFollowBackground(ctx) {
    const { z } = this.cam;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = GRASS;
    ctx.fillRect(0, 0, this.w, this.h);
    this.worldTransform(ctx);
    const pad = 30;
    const x0 = this.cam.x - this.w / 2 / z - pad, x1 = this.cam.x + this.w / 2 / z + pad;
    const y0 = this.cam.y - this.h / 2 / z - pad, y1 = this.cam.y + this.h / 2 / z + pad;
    // A faint 50 m grid on the grass gives a sense of speed.
    ctx.beginPath();
    for (let gx = Math.floor(x0 / 50) * 50; gx < x1; gx += 50) { ctx.moveTo(gx, y0); ctx.lineTo(gx, y1); }
    for (let gy = Math.floor(y0 / 50) * 50; gy < y1; gy += 50) { ctx.moveTo(x0, gy); ctx.lineTo(x1, gy); }
    ctx.lineWidth = this.dpr / z;
    ctx.strokeStyle = "rgba(0,0,0,0.12)";
    ctx.stroke();
    const visible = track.chunks.filter(({ box }) => box[2] > x0 && box[0] < x1 && box[3] > y0 && box[1] < y1);
    // All visible pieces in ONE path, filled once, so there are no seams between them.
    ctx.beginPath();
    for (const ch of visible) trackPath(ctx, ch.idx);
    ctx.fillStyle = ASPHALT;
    ctx.fill();
    ctx.strokeStyle = EDGE;
    ctx.lineWidth = Math.max(0.35, 1.2 * this.dpr / z);
    for (const ch of visible) {
      edgePath(ctx, ch.idx, track.lx, track.ly); ctx.stroke();
      edgePath(ctx, ch.idx, track.rx, track.ry); ctx.stroke();
    }
    this.startLine(ctx, 1.5);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const size = Math.round(12 * this.dpr);
    for (const k of track.corners) {
      const off = track.width / 2 + 16;
      const wx = k.x + k.nx * off, wy = k.y + k.ny * off;
      if (wx < x0 || wx > x1 || wy < y0 || wy > y1) continue;
      const [sx, sy] = this.toScreen(wx, wy);
      this.label(ctx, k.name, sx, sy, size);
    }
  }

  draw(t) {
    const ctx = this.ctx;
    const overview = state.camera === "overview" || !this.cars;     // full track while loading
    if (overview) {
      if (!this.bg) this.renderOverviewBackground();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.drawImage(this.bg, 0, 0);
    } else {
      this.drawFollowBackground(ctx);
    }
    if (!this.cars) return;
    const f = t * manifest.fps;
    const z = this.cam.z;
    const best = this.meta.best;

    // The best car's trail: where it was over the last few seconds.
    const bc = this.cars[best];
    const k1 = Math.min(Math.floor(f), bc.n - 1), k0 = Math.max(0, k1 - TRAIL_SECONDS * manifest.fps);
    this.worldTransform(ctx);
    if (k1 > k0) {
      ctx.beginPath();
      ctx.moveTo(bc.x[k0], bc.y[k0]);
      for (let k = k0 + 1; k <= k1; k++) ctx.lineTo(bc.x[k], bc.y[k]);
      const p = this.carAt(best, f);
      ctx.lineTo(p.x, p.y);
      ctx.lineWidth = (overview ? 3 : 2.5) * this.dpr / z;
      ctx.lineJoin = "round";
      ctx.strokeStyle = algoColor(this.algo);
      ctx.stroke();
    }

    const drawCar = (i, isBest) => {
      const p = this.carAt(i, f);
      const finished = this.meta.finished[i] && p.stopped;
      const fill = finished ? FINISHED : p.stopped ? STOPPED : this.colors[i];
      if (overview) {
        const r = (isBest ? 5.5 : 3) * this.dpr / z;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, 2 * Math.PI);
        ctx.fillStyle = fill;
        ctx.fill();
        if (isBest) { ctx.lineWidth = 2 * this.dpr / z; ctx.strokeStyle = "#fff"; ctx.stroke(); }
        return;
      }
      const s = isBest ? 1.25 : 1;
      const c = Math.cos(p.h), sn = Math.sin(p.h);
      const hl = CAR_LEN / 2 * s, hw = CAR_WID / 2 * s;
      ctx.beginPath();
      ctx.moveTo(p.x + c * hl - sn * hw, p.y + sn * hl + c * hw);
      ctx.lineTo(p.x + c * hl + sn * hw, p.y + sn * hl - c * hw);
      ctx.lineTo(p.x - c * hl + sn * hw, p.y - sn * hl - c * hw);
      ctx.lineTo(p.x - c * hl - sn * hw, p.y - sn * hl + c * hw);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
      if (isBest) { ctx.lineWidth = 2 * this.dpr / z; ctx.strokeStyle = "#fff"; ctx.stroke(); }
    };
    // Stopped cars first, then running cars (worst to best), the best car on top.
    for (const i of this.order) if (i !== best && this.carAt(i, f).stopped) drawCar(i, false);
    for (const i of this.order) if (i !== best && !this.carAt(i, f).stopped) drawCar(i, false);
    drawCar(best, true);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  aliveAt(frame) {
    let alive = 0;
    for (const s of this.stopped) if (frame < s) alive++;
    return alive;
  }
}

// ------------------------------------------------------------------------------------
// stats cards
// ------------------------------------------------------------------------------------
class StatsCard {
  constructor(el) { this.el = el; this.algo = el.dataset.algo; }

  set(meta, pairLabel) {
    this.meta = meta;
    const lapPct = Math.round(100 * meta.best_distance / manifest.track_length);
    const tag = `<span class="tag tag-${meta.algo}">${meta.algo.toUpperCase()}</span>`;
    const extra = meta.algo === "ppo"
      ? `<div class="tile"><div class="k">Noise-free driver</div><div class="v">${fmtDist(meta.greedy_distance)}</div>
           <div class="sub">${meta.greedy_lap ? `lap ${fmtTime(meta.greedy_lap)}` : "same brain, no randomness"}</div></div>`
      : "";
    const note = meta.algo === "ga"
      ? `100 different brains. Colour = rank this generation (green = best); grey = crashed, gold = finished a lap.`
      : `One brain drives every car; they differ only by exploration noise (std ${meta.action_std}).
         ${pairLabel ? `Closest PPO checkpoint to ${pairLabel}.` : ""}`;
    this.el.innerHTML = `
      <h3>${tag} ${meta.title} <small>· ${fmtSteps(meta.env_steps)} env steps of training</small></h3>
      <div class="tiles">
        <div class="tile"><div class="k">Best distance</div><div class="v">${fmtDist(meta.best_distance)}</div><div class="sub">${lapPct}% of the lap</div></div>
        <div class="tile"><div class="k">Best lap</div><div class="v">${fmtTime(meta.best_lap)}</div></div>
        <div class="tile"><div class="k">Laps finished</div><div class="v">${meta.laps_completed} <small>/ ${meta.n_cars}</small></div></div>
        <div class="tile"><div class="k">Still driving</div><div class="v"><span data-alive>–</span> <small>/ ${meta.n_cars}</small></div></div>
        ${extra}
      </div>
      <div class="spark"><div class="k">Cars still driving over time</div><canvas></canvas></div>
      <p class="note">${note}</p>`;
    this.aliveEl = $("[data-alive]", this.el);
    this.spark = $(".spark canvas", this.el);
  }

  update(view, t) {
    if (!this.meta || !view.cars) return;
    const frame = t * manifest.fps;
    this.aliveEl.textContent = view.aliveAt(frame);
    const cv = this.spark, dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(cv.clientWidth * dpr), h = Math.round(cv.clientHeight * dpr);
    if (!w || !h) return;
    if (cv.width !== w || cv.height !== h || this.sparkFor !== this.meta.id) {
      cv.width = w; cv.height = h;
      this.sparkFor = this.meta.id;
      // Pre-compute the alive count for every recorded frame.
      const n = this.meta.n_frames, N = this.meta.n_cars;
      const stops = new Int32Array(n + 1);
      for (const s of view.stopped) if (s <= n) stops[Math.min(s, n)]++;
      this.alive = new Float32Array(n);
      let gone = 0;
      for (let k = 0; k < n; k++) { gone += stops[k]; this.alive[k] = (N - gone) / N; }
    }
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    const n = this.alive.length, pad = 2 * dpr;
    const X = (k) => pad + (w - 2 * pad) * k / Math.max(1, n - 1);
    const Y = (v) => h - pad - (h - 2 * pad) * v;
    ctx.beginPath();
    ctx.moveTo(X(0), Y(0));
    const stride = Math.max(1, Math.floor(n / (w / dpr)));
    for (let k = 0; k < n; k += stride) ctx.lineTo(X(k), Y(this.alive[k]));
    ctx.lineTo(X(n - 1), Y(this.alive[n - 1]));
    ctx.lineTo(X(n - 1), Y(0));
    ctx.closePath();
    const color = algoColor(this.algo);
    ctx.fillStyle = color + "40";
    ctx.fill();
    ctx.beginPath();
    for (let k = 0; k < n; k += stride) (k ? ctx.lineTo(X(k), Y(this.alive[k])) : ctx.moveTo(X(k), Y(this.alive[k])));
    ctx.lineTo(X(n - 1), Y(this.alive[n - 1]));
    ctx.lineWidth = 1.5 * dpr;
    ctx.strokeStyle = color;
    ctx.stroke();
    const px = X(Math.min(frame, n - 1));                          // playhead
    ctx.fillStyle = cssVar("--text");
    ctx.fillRect(px - dpr, 0, 2 * dpr, h);
  }
}

// ------------------------------------------------------------------------------------
// state <-> UI
// ------------------------------------------------------------------------------------
function visible() {
  return state.mode === "split" ? [views.ga, views.ppo] : [views[state.mode]];
}

function currentPair() { return manifest.pairs[state.pair]; }

// Index of the replay pair for a GA generation number (the nearest one if it wasn't recorded).
function pairIndexFor(generation) {
  if (!manifest) return 0;
  let best = 0;
  manifest.pairs.forEach((p, k) => {
    if (Math.abs(p.generation - generation) < Math.abs(manifest.pairs[best].generation - generation)) best = k;
  });
  return best;
}

function duration() {
  const p = currentPair();
  const ids = state.mode === "split" ? [p.ga, p.ppo] : [p[state.mode]];
  return Math.max(...ids.map((id) => manifest.replays[id].duration));
}

function getReplay(id) {
  if (!cache.has(id)) {
    const meta = manifest.replays[id];
    cache.set(id, loadReplay("data/", meta, manifest));
  }
  return cache.get(id);
}

async function selectPair(index) {
  state.pair = Math.max(0, Math.min(manifest.pairs.length - 1, index));
  const p = currentPair();
  const gaMeta = manifest.replays[p.ga], ppoMeta = manifest.replays[p.ppo];
  $("#pair").value = state.pair;
  $("#gen-input").value = p.generation;
  $("#pair-label").textContent =
    `${p.label} of ${manifest.pairs[manifest.pairs.length - 1].generation} · ` +
    `GA ${fmtSteps(gaMeta.env_steps)} / PPO ${fmtSteps(ppoMeta.env_steps)} env steps of training`;
  stats.ga.set(gaMeta);
  stats.ppo.set(ppoMeta, `GA ${p.label.toLowerCase()} (${fmtSteps(gaMeta.env_steps)} steps)`);
  for (const [algo, meta] of [["ga", gaMeta], ["ppo", ppoMeta]]) {
    const view = views[algo];
    view.cars = null;
    view.setBadge(meta);
    view.showMessage("Loading replay…");
    getReplay(meta.id).then((loaded) => {
      if (currentPair()[algo] !== meta.id) return;                 // the user moved on meanwhile
      view.setReplay(meta, loaded);
      view.showMessage("");
      idle = 0;
    }).catch((err) => view.showMessage(err.message));
  }
  // Warm the cache for the neighbouring generations.
  for (const j of [state.pair + 1, state.pair - 1]) {
    const q = manifest.pairs[j];
    if (q) { getReplay(q.ga).catch(() => {}); getReplay(q.ppo).catch(() => {}); }
  }
  setTime(Math.min(state.t, duration()));
  if (curve.canvas) drawCurve();
  idle = 0;
}

function setTime(t) {
  state.t = Math.max(0, Math.min(t, duration()));
  idle = 0;
}

function setPlaying(on) {
  if (on && state.t >= duration() - 1e-6) state.t = 0;            // replay from the start
  state.playing = on;
  $("#play").setAttribute("aria-pressed", String(on));
  $("#play").setAttribute("aria-label", on ? "Pause" : "Play");
  idle = 0;
  if (!on) writeHash();
}

function setMode(mode) {
  state.mode = mode;
  $("#stage").dataset.mode = mode;
  $("#stats").dataset.mode = mode;
  for (const b of $$("button[data-mode]")) b.setAttribute("aria-pressed", String(b.dataset.mode === mode));
  for (const v of Object.values(views)) v.resize();
  setTime(state.t);
  writeHash();
}

function setCamera(cam) {
  state.camera = cam;
  for (const b of $$("button[data-camera]")) b.setAttribute("aria-pressed", String(b.dataset.camera === cam));
  for (const v of Object.values(views)) v.snap = true;
  idle = 0;
  writeHash();
}

function setSpeed(s) {
  state.speed = s;
  for (const b of $$("button[data-speed]")) b.setAttribute("aria-pressed", String(+b.dataset.speed === s));
  writeHash();
}

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (MODES.includes(h.get("mode"))) state.mode = h.get("mode");
  if (h.has("gen")) state.pair = pairIndexFor(parseInt(h.get("gen"), 10) || 1);
  if (h.has("t")) state.t = parseFloat(h.get("t")) || 0;
  if (["follow", "overview"].includes(h.get("cam"))) state.camera = h.get("cam");
  if (SPEEDS.includes(+h.get("speed"))) state.speed = +h.get("speed");
  return h.get("play") === "1";
}

function writeHash() {
  if (!manifest) return;
  const h = new URLSearchParams({
    mode: state.mode, gen: currentPair().generation, t: state.t.toFixed(1), cam: state.camera, speed: state.speed,
  });
  history.replaceState(null, "", `#${h}`);
}

// ------------------------------------------------------------------------------------
// training log of the replayed runs (log-scale x axis, one marker per replay)
// ------------------------------------------------------------------------------------
const curve = { canvas: null, x0: 4, x1: 8 };

function curveX(steps, w, pad) {
  const lx = Math.log10(Math.max(steps, 10 ** curve.x0));
  return pad + (w - 2 * pad) * (lx - curve.x0) / (curve.x1 - curve.x0);
}

function drawCurve() {
  const cv = curve.canvas, dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(cv.clientWidth * dpr), h = Math.round(cv.clientHeight * dpr);
  if (!w || !h) return;
  cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d");
  const pad = 8 * dpr, top = 16 * dpr, bottom = h - 16 * dpr;
  const L = manifest.track_length;
  const Y = (d) => bottom - (bottom - top) * Math.min(1, Math.max(0, d / L));
  ctx.font = `${11 * dpr}px system-ui, sans-serif`;
  ctx.fillStyle = cssVar("--muted");
  ctx.strokeStyle = cssVar("--border");
  ctx.lineWidth = dpr;
  ctx.beginPath(); ctx.moveTo(pad, Y(L)); ctx.lineTo(w - pad, Y(L)); ctx.stroke();
  ctx.textAlign = "left";
  ctx.fillText("full lap", pad, Y(L) - 4 * dpr);
  ctx.textAlign = "center";
  for (let e = Math.ceil(curve.x0); e <= Math.floor(curve.x1); e++) {
    const x = curveX(10 ** e, w, pad);
    ctx.fillText(e >= 6 ? `${10 ** (e - 6)}M` : `${10 ** (e - 3)}k`, x, h - 3 * dpr);
  }
  for (const algo of ["ppo", "ga"]) {
    const pts = manifest.curves[algo] || [];
    ctx.beginPath();
    let best = 0;
    pts.forEach(([s, d], k) => {
      best = Math.max(best, d || 0);                                // best distance so far
      const x = curveX(s, w, pad), y = Y(best);
      k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = algoColor(algo);
    ctx.lineWidth = 2 * dpr;
    ctx.stroke();
  }
  manifest.pairs.forEach((p, k) => {
    const meta = manifest.replays[p.ga];
    const x = curveX(meta.env_steps, w, pad);
    const current = k === state.pair;
    ctx.beginPath();
    ctx.arc(x, Y(meta.best_distance), (current ? 5 : manifest.pairs.length > 20 ? 2 : 3.5) * dpr, 0, 2 * Math.PI);
    ctx.fillStyle = current ? cssVar("--accent") : cssVar("--text-2");
    ctx.fill();
    if (current) { ctx.fillRect(x - dpr / 2, top, dpr, bottom - top); }
  });
}

function setupCurve() {
  curve.canvas = $("#curve");
  const steps = [...(manifest.curves.ga || []), ...(manifest.curves.ppo || [])].map((p) => p[0]);
  curve.x1 = Math.log10(Math.max(1e5, ...steps) * 1.15);
  new ResizeObserver(drawCurve).observe(curve.canvas);
  curve.canvas.addEventListener("click", (e) => {
    // Jump to the replay whose marker is closest to the click.
    const r = curve.canvas.getBoundingClientRect();
    const px = e.clientX - r.left;
    let bestK = 0, bestD = Infinity;
    manifest.pairs.forEach((p, k) => {
      const d = Math.abs(curveX(manifest.replays[p.ga].env_steps, r.width, 8) - px);
      if (d < bestD) { bestD = d; bestK = k; }
    });
    selectPair(bestK);
    writeHash();
  });
}

// ------------------------------------------------------------------------------------
// static sections
// ------------------------------------------------------------------------------------
function renderCurves() {
  $("#plots").innerHTML = (manifest.plots || [])
    .map((p) => `<figure><img src="${p.src}" alt="${p.alt}" loading="lazy" width="1275" height="690"></figure>`).join("");
  const rows = manifest.summary || [];
  if (!rows.length) { $("#summary").hidden = true; return; }
  const num = (v) => (v === "" || v == null ? NaN : +v);
  const body = rows.map((r) => {
    const name = { ga: "GA", ppo: "PPO tuned", ppo_spec: "PPO as specified" }[r.algo] || r.algo;
    const first = num(r.steps_to_first_lap);
    return `<tr><td><span class="tag tag-${r.algo}">${name}</span></td><td>${r.seed}</td>
      <td>${fmtDist(num(r.final_best_distance_m))}</td><td>${fmtTime(num(r.final_best_lap_s))}</td>
      <td>${isFinite(first) ? fmtSteps(first) : "never"}</td>
      <td>${(num(r.wall_time_s) / 60).toFixed(0)} min</td><td>${fmtTime(num(r.eval_lap_s))}</td></tr>`;
  }).join("");
  $("#summary").innerHTML = `<thead><tr><th>Method</th><th>Seed</th><th>Best distance</th><th>Best lap</th>
    <th>Steps to first lap</th><th>Training time</th><th>Noise-free lap</th></tr></thead><tbody>${body}</tbody>`;
}

// ------------------------------------------------------------------------------------
// main loop and wiring
// ------------------------------------------------------------------------------------
let lastTime = performance.now();
let lastHashWrite = 0;

function frame(now) {
  const dt = Math.min(0.1, (now - lastTime) / 1000);
  lastTime = now;
  if (state.playing) {
    state.t += dt * state.speed;
    if (state.t >= duration()) { state.t = duration(); setPlaying(false); }
    if (now - lastHashWrite > 1000) { writeHash(); lastHashWrite = now; }
  }
  // While paused, keep drawing briefly so the camera can settle, then rest.
  if (state.playing || idle < 90) {
    idle++;
    for (const v of visible()) { v.update(state.t, dt); v.draw(state.t); }
    for (const algo of ["ga", "ppo"]) if (state.mode === "split" || state.mode === algo) stats[algo].update(views[algo], state.t);
    const d = duration();
    const tl = $("#timeline");
    tl.max = d.toFixed(2);
    tl.value = state.t.toFixed(2);
    $("#time-label").textContent = `${fmtTime(state.t, 0)} / ${fmtTime(d, 0)}`;
  }
  requestAnimationFrame(frame);
}

function wire() {
  for (const b of $$("button[data-mode]")) b.addEventListener("click", () => setMode(b.dataset.mode));
  for (const b of $$("button[data-camera]")) b.addEventListener("click", () => setCamera(b.dataset.camera));
  for (const b of $$("button[data-speed]")) b.addEventListener("click", () => setSpeed(+b.dataset.speed));
  $("#play").addEventListener("click", () => setPlaying(!state.playing));
  $("#timeline").addEventListener("input", (e) => { setTime(+e.target.value); writeHash(); });
  $("#pair").addEventListener("input", (e) => { selectPair(+e.target.value); writeHash(); });
  // Type a generation number and press Enter (or leave the box) to jump straight to it.
  const genInput = $("#gen-input");
  const jump = () => {
    const g = parseInt(genInput.value, 10);
    if (Number.isFinite(g)) { selectPair(pairIndexFor(g)); writeHash(); }
    else genInput.value = currentPair().generation;
  };
  genInput.addEventListener("change", jump);
  genInput.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); jump(); genInput.select(); } });
  // Tap a view to play / pause (handy on phones).
  for (const v of Object.values(views)) v.canvas.addEventListener("click", () => setPlaying(!state.playing));
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input[type=number], textarea")) return;     // typing a generation number
    if (e.target.closest("input, textarea") && e.key !== " ") return;
    if (e.target.closest("button") && (e.key === " " || e.key === "Enter")) return;   // the button handles it
    const keys = {
      " ": () => setPlaying(!state.playing),
      ArrowLeft: () => setTime(state.t - 5),
      ArrowRight: () => setTime(state.t + 5),
      ArrowUp: () => selectPair(state.pair + 1),
      ArrowDown: () => selectPair(state.pair - 1),
      1: () => setSpeed(1), 2: () => setSpeed(4), 3: () => setSpeed(16),
      f: () => setCamera(state.camera === "follow" ? "overview" : "follow"),
    };
    const fn = keys[e.key];
    if (fn) { e.preventDefault(); fn(); writeHash(); }
  });
  window.addEventListener("hashchange", () => {
    const play = readHash();
    setMode(state.mode); setCamera(state.camera); setSpeed(state.speed);
    selectPair(state.pair);
    if (play) setPlaying(true);
  });
}

async function init() {
  try {
    const [m, t] = await Promise.all(["data/replays.json", "data/track.json"].map(async (u) => {
      const r = await fetch(u);
      if (!r.ok) throw new Error(`could not load ${u} (${r.status}) - run record.py first`);
      return r.json();
    }));
    manifest = m;
    track = buildTrack(t);
  } catch (err) {
    $("#viewer").insertAdjacentHTML("afterbegin", `<p class="error">${err.message}</p>`);
    return;
  }
  views = { ga: new View($("#view-ga")), ppo: new View($("#view-ppo")) };
  stats = { ga: new StatsCard($("#stats-ga")), ppo: new StatsCard($("#stats-ppo")) };
  const play = readHash();
  if (!new URLSearchParams(location.hash.slice(1)).has("gen")) state.pair = manifest.pairs.length - 1;
  $("#pair").max = manifest.pairs.length - 1;
  $("#gen-input").min = manifest.pairs[0].generation;
  $("#gen-input").max = manifest.pairs[manifest.pairs.length - 1].generation;
  setMode(state.mode);
  setCamera(state.camera);
  setSpeed(state.speed);
  wire();
  setupCurve();
  renderCurves();
  await selectPair(state.pair);
  if (play) setPlaying(true);
  requestAnimationFrame(frame);
  window.__viewerReady = true;          // used by automated screenshot tests
}

init();
