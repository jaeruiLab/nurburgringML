"""Download the Nordschleife centerline from OpenStreetMap and save it as GeoJSON.

Run from the project root:   python tools/fetch_osm.py
Writes:                       data/nordschleife.geojson

What it does
1. Downloads OSM route relation 38566 ("Nürburgring Nordschleife") with all its
   ways and nodes from the main OSM API (no API key needed).
2. Stitches the ways end-to-end into one closed loop. The relation lists its
   ways in driving order and every way is named after the corner it covers
   (Hatzenbach, Flugplatz, Karussell, ...), so we also get corner labels.
3. Rotates the loop so it starts at the start/finish line by the T13 grandstand
   (where official 20.832 km Nordschleife laps are timed).
4. Simplifies the ~1000 raw OSM nodes down to ~450 points with Douglas-Peucker
   (keeps points in corners, drops redundant points on straights).
5. Writes a GeoJSON FeatureCollection: one LineString (lon/lat) for the
   centerline plus one Point per named section for map labels.

Only the Python standard library is used, so this works before installing
anything. Map data (c) OpenStreetMap contributors, ODbL 1.0.
"""
import json
import math
import os
import sys
import urllib.request

RELATION_ID = 38566
START_SECTION = "T13"          # loop starts at the first node of this section
TARGET_POINTS = 450
# The corners the README / map highlights (others are still stored as labels).
MAJOR = {"Hatzenbach", "Flugplatz", "Fuchsröhre", "Adenauer Forst", "Bergwerk",
         "Karussell", "Brünnchen", "Pflanzgarten", "Schwalbenschwanz", "Döttinger Höhe"}
USER_AGENT = "nurburgringML-track-fetch"
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "nordschleife.geojson")


def fetch_relation(rel_id):
    url = f"https://api.openstreetmap.org/api/0.6/relation/{rel_id}/full.json"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def stitch(data, rel_id):
    """Return a list of (lat, lon, section_name) for the closed loop, in driving order."""
    nodes = {e["id"]: (e["lat"], e["lon"]) for e in data["elements"] if e["type"] == "node"}
    ways = {e["id"]: e for e in data["elements"] if e["type"] == "way"}
    rel = next(e for e in data["elements"] if e["type"] == "relation" and e["id"] == rel_id)
    chain, names = [], []
    last_name = None
    for m in rel["members"]:
        if m["type"] != "way":
            continue
        w = ways[m["ref"]]
        ids = list(w["nodes"])
        # Unnamed bits (bridges, short connectors) inherit the previous section's name.
        name = w.get("tags", {}).get("name")
        if name is None or name.startswith("Nürburgring"):
            name = last_name
        if chain:
            if ids[0] == chain[-1]:
                ids = ids[1:]
            elif ids[-1] == chain[-1]:
                ids = ids[::-1][1:]
            else:
                raise RuntimeError(f"way {m['ref']} does not connect to the previous way")
        chain += ids
        names += [name] * len(ids)
        last_name = name
    if chain[0] != chain[-1]:
        raise RuntimeError("stitched ways do not form a closed loop")
    chain, names = chain[:-1], names[:-1]
    # The first ways of the relation are unnamed parts of the Döttinger Höhe straight.
    names = [n if n is not None else "Döttinger Höhe" for n in names]
    return [(nodes[i][0], nodes[i][1], n) for i, n in zip(chain, names)]


def to_xy(pts):
    lat0 = sum(p[0] for p in pts) / len(pts)
    lon0 = sum(p[1] for p in pts) / len(pts)
    R = 6371008.8
    k = math.cos(math.radians(lat0))
    return [(math.radians(p[1] - lon0) * R * k, math.radians(p[0] - lat0) * R) for p in pts]


def douglas_peucker(xy, keep, eps):
    """Mark points to keep (iterative Douglas-Peucker on an open polyline)."""
    stack = [(0, len(xy) - 1)]
    while stack:
        a, b = stack.pop()
        ax, ay = xy[a]
        bx, by = xy[b]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy or 1e-12
        best, best_d = None, eps
        for i in range(a + 1, b):
            px, py = xy[i]
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
            d = math.hypot(px - ax - t * dx, py - ay - t * dy)
            if d > best_d:
                best, best_d = i, d
        if best is not None:
            keep[best] = True
            stack += [(a, best), (best, b)]


def simplify(xy, target):
    """Binary-search the DP tolerance so roughly `target` points survive."""
    lo, hi = 0.01, 50.0
    for _ in range(40):
        eps = (lo + hi) / 2
        keep = [False] * len(xy)
        keep[0] = keep[-1] = True
        # Split the loop in two halves so DP has two fixed anchors.
        mid = len(xy) // 2
        keep[mid] = True
        douglas_peucker(xy[: mid + 1], keep, eps)
        sub = [False] * (len(xy) - mid + 1)
        douglas_peucker(xy[mid:] + [xy[0]], sub, eps)
        for i, k in enumerate(sub[:-1]):
            keep[mid + i] = keep[mid + i] or k
        n = sum(keep)
        if n > target:
            lo = eps
        else:
            hi = eps
    return keep


def main():
    print(f"Downloading OSM relation {RELATION_ID} ...")
    data = fetch_relation(RELATION_ID)
    loop = stitch(data, RELATION_ID)
    start = next(i for i, p in enumerate(loop) if p[2] == START_SECTION)
    loop = loop[start:] + loop[:start]

    xy = to_xy(loop)
    seg = [math.hypot(xy[(i + 1) % len(xy)][0] - xy[i][0], xy[(i + 1) % len(xy)][1] - xy[i][1])
           for i in range(len(xy))]
    length = sum(seg)
    # Shoelace area: negative = clockwise when x points east and y points north.
    area = 0.5 * sum(xy[i][0] * xy[(i + 1) % len(xy)][1] - xy[(i + 1) % len(xy)][0] * xy[i][1]
                     for i in range(len(xy)))
    print(f"Raw loop: {len(loop)} nodes, {length:.0f} m, {'clockwise' if area < 0 else 'counter-clockwise'}")

    keep = simplify(xy, TARGET_POINTS)
    line = [[round(loop[i][1], 7), round(loop[i][0], 7)] for i in range(len(loop)) if keep[i]]
    print(f"Simplified to {len(line)} points")

    # One label per named section, placed at the section's middle (by arc length).
    cum = [0.0]
    for d in seg[:-1]:
        cum.append(cum[-1] + d)
    sections = {}
    for i, p in enumerate(loop):
        sections.setdefault(p[2], []).append(i)
    features = [{
        "type": "Feature",
        "properties": {
            "name": "Nürburgring Nordschleife",
            "role": "centerline",
            "closed": True,
            "start": "first coordinate is the start/finish line (T13)",
            "track_width_m": 11.0,
            "source": f"OpenStreetMap relation {RELATION_ID}",
            "license": "(c) OpenStreetMap contributors, ODbL 1.0",
        },
        "geometry": {"type": "LineString", "coordinates": line},
    }]
    for name, idx in sections.items():
        mid_s = (cum[idx[0]] + cum[idx[-1]]) / 2
        i = min(idx, key=lambda j: abs(cum[j] - mid_s))
        features.append({
            "type": "Feature",
            "properties": {"role": "corner", "name": name, "major": name in MAJOR},
            "geometry": {"type": "Point", "coordinates": [round(loop[i][1], 7), round(loop[i][0], 7)]},
        })
        print(f"  {name:22s} at {cum[i] / 1000:6.2f} km")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False, indent=1)
    print(f"Wrote {os.path.normpath(OUT_PATH)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
