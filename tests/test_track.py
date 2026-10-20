import math

import torch

from track import catmull_rom_closed, latlon_to_local_m

MAJOR = ["Hatzenbach", "Flugplatz", "Fuchsröhre", "Adenauer Forst", "Bergwerk",
         "Karussell", "Brünnchen", "Pflanzgarten", "Schwalbenschwanz", "Döttinger Höhe"]


def test_latlon_converter():
    x, y = latlon_to_local_m(51.0, 7.0, 50.0, 7.0)
    assert abs(y - 111_195) < 50 and abs(x) < 1e-6
    x, y = latlon_to_local_m(50.0, 8.0, 50.0, 7.0)
    assert abs(x - 111_195 * math.cos(math.radians(50))) < 50


def test_catmull_rom_passes_through_control_points():
    P = torch.tensor([[0, 0], [10, 0], [12, 8], [3, 11], [-4, 5]], dtype=torch.float64)
    C = catmull_rom_closed(P, samples_per_seg=10)
    assert torch.allclose(C[::10], P, atol=1e-9)


def test_length_close_to_real_nordschleife(track):
    # Real lap: 20.832 km (OSM centerline: ~20.75 km)
    assert 20_500 < track.length < 21_000
    assert abs(track.ds - 1.0) < 0.01


def test_major_corners_present_and_in_order(track):
    names = [c["name"] for c in track.corners]
    positions = [names.index(m) for m in MAJOR]
    assert positions == sorted(positions), "corners should appear in driving order"
    assert track.corners[0]["s"] < 500  # T13 right after the start line


def test_occupancy_grid(track):
    c = track.center
    assert track.on_track(c[:, 0], c[:, 1]).all(), "every centerline point must be on track"
    inner = track.width / 2 - 0.6
    for side in (+1, -1):
        p = c + side * inner * track.normal
        assert track.on_track(p[:, 0], p[:, 1]).float().mean() > 0.999
        q = c + side * (track.width / 2 + 2.0) * track.normal
        assert track.on_track(q[:, 0], q[:, 1]).float().mean() < 0.001


def test_curvature_sign(track):
    # Clockwise loop -> the total heading change is -2*pi (right turns dominate).
    total_turn = (track.kappa * track.ds).sum().item()
    assert abs(total_turn + 2 * math.pi) < 0.05
