"""The race track: centerline, boundaries, arc length, curvature and occupancy grid.

Pipeline (all done once at start-up):
  data file (lon/lat or x/y)  ->  local meters  ->  Catmull-Rom spline  ->
  evenly spaced 1 m centerline  ->  tangents / normals / curvature / boundaries  ->
  occupancy grid (a big 2-D boolean image: True = asphalt)

The occupancy grid makes two things very fast for 100 cars at once:
  * "Is this car off the track?"  = look up one pixel
  * raycasts (the cars' "eyes")   = look up the pixels along each ray

Run `python track.py` to save a labelled map to results/track.png.
"""
import csv
import json
import math
import os

import torch

EARTH_RADIUS = 6371008.8  # meters


def latlon_to_local_m(lat, lon, lat0, lon0):
    """Convert lat/lon degrees to flat x (east) / y (north) meters around (lat0, lon0).

    Equirectangular projection: fine for a 6 km area (error well under 0.1%).
    Works on floats or tensors.
    """
    x = math.radians(1.0) * (lon - lon0) * EARTH_RADIUS * math.cos(math.radians(lat0))
    y = math.radians(1.0) * (lat - lat0) * EARTH_RADIUS
    return x, y


# ----------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------
def _stitch_lines(lines):
    """Join several polylines into one by matching nearest endpoints (for OSM exports
    where the track is split into many ways)."""
    lines = [list(l) for l in lines if len(l) > 1]
    chain = lines.pop(0)
    while lines:
        end = chain[-1]
        best, best_d, flip = 0, float("inf"), False
        for i, l in enumerate(lines):
            for f, p in ((False, l[0]), (True, l[-1])):
                d = (p[0] - end[0]) ** 2 + (p[1] - end[1]) ** 2
                if d < best_d:
                    best, best_d, flip = i, d, f
        l = lines.pop(best)
        chain += (l[::-1] if flip else l)[1:]
    return chain


def load_centerline(path):
    """Read a track file. Returns (points_xy, corners, meta).

    points_xy: list of (x, y) meters for a closed loop, in driving order, starting at
               the start/finish line.
    corners:   list of dicts {name, x, y, major}
    Supported: GeoJSON (LineString / MultiLineString / FeatureCollection, lon/lat),
               .json {"points": [[x, y], ...]} in meters, or .csv with x,y columns.
    """
    ext = os.path.splitext(path)[1].lower()
    corners, meta = [], {}
    if ext == ".csv":
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        pts = [(float(r["x"]), float(r["y"])) for r in rows]
        return pts, corners, meta

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "points" in data:  # plain x/y meters
        pts = [tuple(p) for p in data["points"]]
        corners = data.get("corners", [])
        return pts, corners, data.get("meta", {})

    # GeoJSON: collect line geometries and named points.
    features = data["features"] if data.get("type") == "FeatureCollection" else [
        {"type": "Feature", "properties": {}, "geometry": data.get("geometry", data)}]
    lines, points = [], []
    for ft in features:
        g, props = ft["geometry"], ft.get("properties") or {}
        if g["type"] == "LineString":
            lines.append(g["coordinates"])
            meta.update(props)
        elif g["type"] == "MultiLineString":
            lines.extend(g["coordinates"])
            meta.update(props)
        elif g["type"] == "Point" and props.get("name"):
            points.append((props["name"], g["coordinates"], bool(props.get("major", True))))
    lonlat = _stitch_lines(lines) if len(lines) > 1 else lines[0]
    if lonlat[0] == lonlat[-1]:
        lonlat = lonlat[:-1]
    lat0 = sum(p[1] for p in lonlat) / len(lonlat)
    lon0 = sum(p[0] for p in lonlat) / len(lonlat)
    pts = [latlon_to_local_m(p[1], p[0], lat0, lon0) for p in lonlat]
    for name, (lon, lat), major in points:
        x, y = latlon_to_local_m(lat, lon, lat0, lon0)
        corners.append({"name": name, "x": x, "y": y, "major": major})
    meta["origin_latlon"] = (lat0, lon0)
    return pts, corners, meta


# ----------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------
def catmull_rom_closed(P, samples_per_seg=64, alpha=0.5):
    """Smooth closed curve through control points P (M, 2) using centripetal Catmull-Rom.

    For each segment P[i] -> P[i+1] the curve also looks at the neighbours P[i-1] and
    P[i+2] so that the direction is continuous at every control point. 'Centripetal'
    (alpha = 0.5) spaces the spline parameter by sqrt(distance), which prevents loops
    and overshoot where short and long segments meet.
    Returns (M * samples_per_seg, 2) points.
    """
    P0, P1, P2, P3 = P.roll(1, 0), P, P.roll(-1, 0), P.roll(-2, 0)

    def knot(a, b):
        return ((b - a).norm(dim=1) ** alpha).clamp(min=1e-6)

    t0 = torch.zeros(len(P), dtype=P.dtype)
    t1 = t0 + knot(P0, P1)
    t2 = t1 + knot(P1, P2)
    t3 = t2 + knot(P2, P3)
    u = torch.linspace(0, 1, samples_per_seg + 1, dtype=P.dtype)[:-1]
    t = t1[:, None] + (t2 - t1)[:, None] * u[None, :]         # (M, S)

    def blend(A, B, ta, tb):
        # Linear blend between A (at knot ta) and B (at knot tb) for every sample t.
        w = ((t - ta[:, None]) / (tb - ta)[:, None])[..., None]  # (M, S, 1)
        A = A[:, None, :] if A.dim() == 2 else A
        B = B[:, None, :] if B.dim() == 2 else B
        return A * (1 - w) + B * w

    # Barry-Goldman pyramid: three linear blends, then two, then one.
    A1, A2, A3 = blend(P0, P1, t0, t1), blend(P1, P2, t1, t2), blend(P2, P3, t2, t3)
    B1, B2 = blend(A1, A2, t0, t2), blend(A2, A3, t1, t3)
    return blend(B1, B2, t1, t2).reshape(-1, 2)


def resample_closed(P, ds):
    """Resample a closed polyline to evenly spaced points every ~ds meters."""
    seg = (P.roll(-1, 0) - P).norm(dim=1)
    cum = torch.cat([torch.zeros(1, dtype=P.dtype), seg.cumsum(0)])  # (M+1,)
    L = cum[-1].item()
    n = max(8, round(L / ds))
    s = torch.linspace(0, L, n + 1, dtype=P.dtype)[:-1]
    i = (torch.searchsorted(cum, s, right=True) - 1).clamp(0, len(P) - 1)
    w = ((s - cum[i]) / seg[i].clamp(min=1e-9))[:, None]
    return P[i] * (1 - w) + P.roll(-1, 0)[i] * w


def circular_smooth(x, window_pts):
    """Moving average along a closed loop (wraps around the start/finish line)."""
    k = max(1, int(window_pts)) | 1  # odd
    if k == 1:
        return x
    pad = k // 2
    xp = torch.cat([x[-pad:], x, x[:pad]])
    return torch.nn.functional.avg_pool1d(xp[None, None], k, stride=1)[0, 0]


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----------------------------------------------------------------------------------
# Track
# ----------------------------------------------------------------------------------
class Track:
    """All track geometry as tensors on `device`.

    Attributes (M = number of 1 m centerline points, ~20 800):
      center (M,2), tangent (M,2), normal (M,2) [points left], heading (M,),
      left (M,2), right (M,2), cum_s (M,), seg_vec (M,2), seg_len2 (M,),
      kappa_near / kappa_far (M,) signed curvature (1/m, + = left turn),
      length (float, m), ds (float, m), grid (H*W,) bool, corners (list of dicts with 's').
    """

    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.width = cfg.width
        pts, corners, meta = load_centerline(cfg.path)
        self.meta = meta
        P = torch.tensor(pts, dtype=torch.float64)

        # 1) Smooth the ~450 control points with a spline, then 2) resample every ~1 m.
        seg_len = (P.roll(-1, 0) - P).norm(dim=1)
        sps = max(16, int(math.ceil(seg_len.max().item() / (cfg.spline_ds / 4))))
        dense = catmull_rom_closed(P, samples_per_seg=min(sps, 2048))
        C = resample_closed(dense, cfg.spline_ds)
        M = len(C)
        nxt, prv = C.roll(-1, 0), C.roll(1, 0)
        seg_vec = nxt - C
        self.length = seg_vec.norm(dim=1).sum().item()
        self.ds = self.length / M

        # 3) Direction of travel (tangent) and the "left" direction (normal).
        tangent = nxt - prv
        tangent = tangent / tangent.norm(dim=1, keepdim=True)
        normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=1)
        heading = torch.atan2(tangent[:, 1], tangent[:, 0])
        # 4) Curvature = how fast the heading turns per meter (1 / corner radius).
        dtheta = wrap_angle(heading.roll(-1) - heading.roll(1))
        kappa = dtheta / (2 * self.ds)
        kappa_near = circular_smooth(kappa, cfg.curvature_smooth_near / self.ds)
        kappa_far = circular_smooth(kappa, cfg.curvature_smooth_far / self.ds)

        half = cfg.width / 2
        f32 = dict(dtype=torch.float32, device=self.device)
        self.M = M
        self.center = C.to(**f32)
        self.tangent = tangent.to(**f32)
        self.normal = normal.to(**f32)
        self.heading = heading.to(**f32)
        self.left = (C + normal * half).to(**f32)
        self.right = (C - normal * half).to(**f32)
        self.cum_s = (torch.arange(M, dtype=torch.float64) * self.ds).to(**f32)
        self.seg_vec = seg_vec.to(**f32)
        self.seg_len2 = (seg_vec ** 2).sum(1).clamp(min=1e-9).to(**f32)
        self.kappa = kappa.to(**f32)
        self.kappa_near = kappa_near.to(**f32)
        self.kappa_far = kappa_far.to(**f32)

        self._build_grid(C, normal)

        # Corner labels: arc-length position of each named point.
        self.corners = []
        for c in corners:
            d2 = ((C - torch.tensor([c["x"], c["y"]], dtype=torch.float64)) ** 2).sum(1)
            i = int(d2.argmin())
            self.corners.append({**c, "s": i * self.ds, "index": i})
        self.corners.sort(key=lambda c: c["s"])

    # ------------------------------------------------------------------------------
    def _build_grid(self, C, normal):
        """Rasterise the track into a boolean image with cell size grid_cell.

        We 'paint' the asphalt by sampling points every quarter meter along the track
        and every quarter meter across its width, then mark the cell each point falls
        in. Sampling at half the cell size guarantees there are no holes.
        """
        cell, margin = self.cfg.grid_cell, self.cfg.grid_margin
        half = self.width / 2
        lo = torch.minimum((C + normal * half).min(0).values, (C - normal * half).min(0).values) - margin
        hi = torch.maximum((C + normal * half).max(0).values, (C - normal * half).max(0).values) + margin
        self.grid_origin = lo.to(torch.float32)
        self.grid_W = int(math.ceil((hi[0] - lo[0]).item() / cell))
        self.grid_H = int(math.ceil((hi[1] - lo[1]).item() / cell))
        self.grid_cell = cell

        step = cell / 2
        sub = max(1, int(math.ceil(self.ds / step)))
        w = torch.arange(sub, dtype=torch.float64) / sub
        # Points along the centerline every ~step meters (linear interpolation).
        along = (C[:, None, :] * (1 - w[None, :, None]) + C.roll(-1, 0)[:, None, :] * w[None, :, None]).reshape(-1, 2)
        nrm = (normal[:, None, :].expand(-1, sub, -1)).reshape(-1, 2)
        offsets = torch.linspace(-half, half, int(math.ceil(self.width / step)) + 1, dtype=torch.float64)

        grid = torch.zeros(self.grid_H * self.grid_W, dtype=torch.bool)
        chunk = 20000
        for i in range(0, len(along), chunk):
            p = along[i:i + chunk, None, :] + nrm[i:i + chunk, None, :] * offsets[None, :, None]
            ix = ((p[..., 0] - lo[0]) / cell).long().clamp(0, self.grid_W - 1)
            iy = ((p[..., 1] - lo[1]) / cell).long().clamp(0, self.grid_H - 1)
            grid[(iy * self.grid_W + ix).reshape(-1)] = True
        self.grid = grid.to(self.device)

    def on_track(self, x, y):
        """Boolean tensor (same shape as x): is the point (x, y) on asphalt?

        Works for any shape, e.g. (N,) car positions or (N, rays, samples) ray points.
        Points outside the grid count as off-track.
        """
        ix = torch.floor((x - self.grid_origin[0]) / self.grid_cell).long()
        iy = torch.floor((y - self.grid_origin[1]) / self.grid_cell).long()
        inside = (ix >= 0) & (ix < self.grid_W) & (iy >= 0) & (iy < self.grid_H)
        idx = iy.clamp(0, self.grid_H - 1) * self.grid_W + ix.clamp(0, self.grid_W - 1)
        return self.grid[idx] & inside

    def index_at(self, s):
        """Centerline index for arc length s (meters, wraps around)."""
        return (torch.as_tensor(s, device=self.device) / self.ds).round().long() % self.M

    def numpy_geometry(self):
        """CPU copies for drawing."""
        return {k: getattr(self, k).cpu().numpy() for k in ("center", "left", "right", "normal", "heading")}


def plot_track(track, path):
    """Save a labelled map of the track (used to eyeball the geometry)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = track.numpy_geometry()
    fig, ax = plt.subplots(figsize=(11, 9), dpi=130)
    for side in ("left", "right"):
        xy = g[side]
        ax.plot(list(xy[:, 0]) + [xy[0, 0]], list(xy[:, 1]) + [xy[0, 1]], color="#444", lw=0.6)
    ax.plot(g["center"][:, 0], g["center"][:, 1], color="#bbb", lw=0.4, ls="--")
    sx, sy = g["center"][0]
    ax.plot([g["left"][0, 0], g["right"][0, 0]], [g["left"][0, 1], g["right"][0, 1]], color="red", lw=2)
    ax.annotate("Start / Finish", (sx, sy), xytext=(10, -14), textcoords="offset points", color="red", fontsize=8)
    for c in track.corners:
        i = c["index"]
        nx, ny = g["normal"][i]
        off = 70 if c["major"] else 45
        ax.plot(*g["center"][i], "o", ms=2.5 if c["major"] else 1.5, color="#1f5fa8" if c["major"] else "#999")
        ax.annotate(c["name"], (g["center"][i, 0] + nx * off, g["center"][i, 1] + ny * off),
                    fontsize=8 if c["major"] else 6, ha="center", va="center",
                    color="#1f5fa8" if c["major"] else "#777", weight="bold" if c["major"] else "normal")
    ax.set_aspect("equal")
    ax.set_title(f"Nürburgring Nordschleife - {track.length / 1000:.2f} km, {track.M} centerline points "
                 f"(map data © OpenStreetMap contributors)", fontsize=9)
    ax.set_xlabel("x (m, east)")
    ax.set_ylabel("y (m, north)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    from config import TrackConfig
    from utils import utf8_stdout

    utf8_stdout()
    t = Track(TrackConfig())
    print(f"Track length {t.length:.1f} m, {t.M} points (ds={t.ds:.3f} m), "
          f"grid {t.grid_W}x{t.grid_H} cells ({t.grid.sum().item() * t.grid_cell ** 2 / 1e4:.1f} ha of asphalt)")
    print(f"Tightest corner radius: {1 / t.kappa_near.abs().max().item():.1f} m")
    plot_track(t, "results/track.png")
    print("Saved results/track.png")
