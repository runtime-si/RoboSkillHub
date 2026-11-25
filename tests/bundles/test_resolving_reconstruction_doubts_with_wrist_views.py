"""resolving-reconstruction-doubts-with-wrist-views: the pure geometry on synthesized clouds,
then the skill on a fake arm whose wrist camera ray-casts a pen cap and a dark tray."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "resolving-reconstruction-doubts-with-wrist-views"
RNG = np.random.default_rng(7)

# The real YAM station's calibrated left wrist D405, camera-to-tool (2026-11-05 dual-arm fit).
TOOL_CAM = np.array(
    [
        [0.0058, 0.701, 0.7131, -0.086],
        [-0.9995, 0.0253, -0.0168, 0.0088],
        [-0.0298, -0.7127, 0.7008, -0.0749],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
u_, s_, vt_ = np.linalg.svd(TOOL_CAM[:3, :3])
TOOL_CAM[:3, :3] = u_ @ vt_
K = np.array([[197.0, 0.0, 160.0], [0.0, 196.8, 120.0], [0.0, 0.0, 1.0]])
HW = (240, 320)
TOP_DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}

GREEN, DARK, TABLE = (40, 200, 60), (30, 30, 30), (150, 140, 120)
PEN_XY = np.array([0.40, 0.10])
# An EXPO-like pen: 11.5 mm knob 4 mm tall on an 18.5 mm cap 24 mm tall on a 16 mm barrel.
PEN_TOP = 0.12
PEN = [
    (0.00575, PEN_TOP - 0.004, PEN_TOP, GREEN),
    (0.00925, PEN_TOP - 0.028, PEN_TOP - 0.004, GREEN),
    (0.008, 0.0, PEN_TOP - 0.028, (240, 240, 240)),
]
TRAY = (np.array([0.40, -0.06, 0.0125]), np.array([0.06, 0.08, 0.025]), DARK)
OVERHEAD = np.diag([1.0, -1.0, -1.0, 1.0])
OVERHEAD[:3, 3] = [0.40, 0.0, 1.0]


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["resolve_doubts"].module


@pytest.fixture(scope="module")
def geo(script):
    return script.geo


# ----------------------------------------------------------------------------- clouds


def cylinder_shell(r, z0, z1, centre=(0.0, 0.0), n=4000, top=True, arc=(0.0, 2 * math.pi)):
    a = RNG.uniform(*arc, n)
    z = RNG.uniform(z0, z1, n)
    pts = [np.c_[centre[0] + r * np.cos(a), centre[1] + r * np.sin(a), z]]
    if top:
        rr = r * np.sqrt(RNG.uniform(0, 1, n // 2))
        aa = RNG.uniform(0, 2 * math.pi, n // 2)
        pts.append(
            np.c_[centre[0] + rr * np.cos(aa), centre[1] + rr * np.sin(aa), np.full(n // 2, z1)]
        )
    return np.concatenate(pts)


def pen_cloud(centre=(0.0, 0.0), noise=0.0002):
    P = np.concatenate([cylinder_shell(r, z0, z1, centre) for r, z0, z1, _ in PEN[:2]])
    return P + RNG.normal(0, noise, P.shape)


def test_band_profile_reads_knob_cap_and_barrel(geo):
    P = np.concatenate(
        [pen_cloud(), cylinder_shell(0.008, PEN_TOP - 0.06, PEN_TOP - 0.028, top=False)]
    )
    top = geo.top_z(P)
    assert top == pytest.approx(PEN_TOP, abs=0.0007)
    prof = geo.band_profile(P, top, ((0, 3), (6, 10), (10, 16), (30, 40)))
    d = [b["diameter"] for b in prof]
    assert d[0] == pytest.approx(0.0115, abs=0.0008)
    assert d[1] == pytest.approx(0.0185, abs=0.0008)
    assert d[2] == pytest.approx(0.0185, abs=0.0008)
    assert d[3] == pytest.approx(0.016, abs=0.0008)


def test_a_band_seen_from_one_side_is_not_read(geo):
    P = cylinder_shell(0.00925, 0.0, 0.03, top=False, arc=(0.0, math.pi * 0.6))
    band = geo.band_profile(P, 0.03, ((5, 10),))[0]
    assert band["points"] > geo.MIN_BAND_POINTS
    assert band["coverage"] < geo.MIN_COVERAGE
    assert band["diameter"] is None


def test_edge_top_ignores_a_sparse_tail_the_percentile_follows(geo):
    tray = np.c_[
        RNG.uniform(-0.03, 0.03, 20000),
        RNG.uniform(-0.04, 0.04, 20000),
        RNG.uniform(0.0, 0.025, 20000),
    ]
    tail = np.c_[
        RNG.uniform(-0.02, 0.02, 1500),
        RNG.uniform(-0.02, 0.02, 1500),
        RNG.uniform(0.025, 0.06, 1500),
    ]
    P = np.concatenate([tray, tail])
    assert geo.top_z(P) > 0.05
    assert geo.edge_top_z(P) == pytest.approx(0.025, abs=0.0025)


@pytest.mark.parametrize("lean_deg, toward", [(0.0, 0.0), (5.0, 30.0), (8.0, -120.0)])
def test_fit_lean_recovers_a_leaning_cylinder(geo, lean_deg, toward):
    P = cylinder_shell(0.008, -0.06, 0.0, top=False, n=12000)
    t, a = math.radians(lean_deg), math.radians(toward)
    axis = np.array([math.sin(t) * math.cos(a), math.sin(t) * math.sin(a), math.cos(t)])
    R = geo.align(np.array([0.0, 0.0, 1.0]), axis)
    Q = P @ R.T
    fit = geo.fit_lean(Q, float(Q[:, 2].max()), 0.05)
    assert fit is not None
    assert fit["lean_deg"] == pytest.approx(lean_deg, abs=0.8)
    if lean_deg:
        assert abs((fit["azimuth_deg"] - toward + 180) % 360 - 180) < 8


def test_lean_from_base(geo):
    lb = geo.lean_from_base([0.01, 0.0, 0.1], [0.0, 0.0, 0.0])
    assert lb["lean_deg"] == pytest.approx(math.degrees(math.atan(0.1)), abs=1e-6)
    assert lb["azimuth_deg"] == pytest.approx(0.0)
    assert lb["offset"] == pytest.approx(0.01)


@pytest.mark.parametrize("yaw", [0.0, 25.0, -60.0, 89.0])
def test_footprint_rect_recovers_a_turned_rectangle(geo, yaw):
    xy = np.c_[RNG.uniform(-0.03, 0.03, 5000), RNG.uniform(-0.015, 0.015, 5000)]
    a = math.radians(yaw)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    rect = geo.footprint_rect(xy @ R.T + [0.4, 0.1])
    assert rect["length"] == pytest.approx(0.06, abs=0.0015)
    assert rect["width"] == pytest.approx(0.03, abs=0.0015)
    assert rect["centre"] == pytest.approx([0.4, 0.1], abs=0.001)
    assert abs((rect["yaw_deg"] - yaw + 90) % 180 - 90) < 1.0


@pytest.mark.parametrize(
    "tilt, az, dist", [(30.0, 0.0, 0.15), (75.0, 140.0, 0.2), (85.0, -90.0, 0.25)]
)
def test_look_pose_puts_the_optical_axis_on_the_target(geo, tilt, az, dist):
    target = np.array([0.4, 0.1, 0.08])
    hand = geo.align(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]))
    pos, rot, eye = geo.look_pose(target, tilt, az, dist, hand, TOOL_CAM)
    cam = np.eye(4)
    cam[:3, :3], cam[:3, 3] = rot, pos
    cam = cam @ TOOL_CAM
    assert cam[:3, 3] == pytest.approx(eye, abs=1e-9)
    assert np.linalg.norm(cam[:3, 3] - target) == pytest.approx(dist, abs=1e-9)
    axis = cam[:3, 2]
    assert axis @ (target - cam[:3, 3]) / dist == pytest.approx(1.0, abs=1e-9)
    assert math.degrees(math.acos(-axis[2])) == pytest.approx(tilt, abs=1e-6)
    u, v, z = geo.project(K, cam, target)
    assert (u, v, z) == pytest.approx((K[0, 2], K[1, 2], dist), abs=1e-6)


def test_hsv_and_colour_mask_follow_opencv_scale(geo):
    rgb = np.array(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255], [30, 30, 30], [250, 10, 40]]], dtype=np.uint8
    )
    hsv = geo.rgb_to_hsv(rgb)[0]
    assert hsv[:3, 0] == pytest.approx([0.0, 60.0, 120.0], abs=0.5)
    assert hsv[3, 1] == pytest.approx(0.0)
    m = geo.colour_mask(rgb, h=[170, 5], s=[100, 255])[0]
    assert m.tolist() == [True, False, False, False, True]
    assert geo.colour_mask(rgb, v=[0, 60])[0].tolist() == [False, False, False, True, False]


def test_back_project_inverts_project(geo):
    T = np.eye(4)
    T[:3, :3] = geo.align(np.array([0.0, 0.0, 1.0]), np.array([0.2, 0.1, -1.0]))
    T[:3, 3] = [0.3, 0.0, 0.3]
    depth = np.zeros(HW, dtype=np.float32)
    depth[100, 200] = 0.2
    P = geo.back_project({"depth": depth, "K": K, "T": T})
    assert P.shape == (1, 3)
    u, v, z = geo.project(K, T, P[0])
    assert (u, v, z) == pytest.approx((200.0, 100.0, 0.2), abs=1e-6)


def test_normalize_doubt_fills_defaults_and_rejects_unknowns(geo):
    d = geo.normalize_doubt({"point": {"x": 0.1, "y": 0.2, "z": 0.3}, "what": "position"}, 3)
    assert d["id"] == "doubt3" and d["point"] == [0.1, 0.2, 0.3]
    assert d["what"] == ["position"] and d["radius"] == geo.POSITION_RADIUS
    assert geo.normalize_doubt({"point": [0, 0, 0]})["radius"] == geo.DEFAULT_RADIUS
    with pytest.raises(ValueError):
        geo.normalize_doubt({"point": [0, 0, 0], "what": ["colour"]})
    with pytest.raises(ValueError):
        geo.normalize_doubt({"what": ["height"]})


def _halves(P, centre):
    """Two views, each of one side plus the whole top (as a 60-75 deg view sees a cap)."""
    a = np.arctan2(P[:, 1] - centre[1], P[:, 0] - centre[0])
    top = P[:, 2] > P[:, 2].max() - 0.003
    return P[(a >= -0.3) | top], P[(a < 0.3) | top]


def test_measure_agrees_across_views_and_names_them(geo):
    P = pen_cloud(centre=(0.4, 0.1))
    a, b = _halves(P, (0.4, 0.1))
    doubt = geo.normalize_doubt(
        {
            "id": "cap",
            "point": [0.402, 0.099, 0.11],
            "what": ["profile", "height", "position"],
            "radius": 0.015,
        }
    )
    r = geo.measure([("v1", a), ("v2", b), ("all", P)], doubt, surface_z=0.0)
    assert r["status"] == "measured", r["detail"]
    q = r["quantities"]
    assert q["height_over_surface"]["value"] == pytest.approx(PEN_TOP, abs=0.0007)
    assert set(q["top_z"]["views"]) == {"v1", "v2", "all"}
    bands = q["profile"]["value"]
    assert bands[0]["diameter"] == pytest.approx(0.0115, abs=0.0008)
    assert bands[2]["diameter"] == pytest.approx(0.0185, abs=0.0008)
    assert q["position"]["value"][:2] == pytest.approx([0.4, 0.1], abs=0.0007)
    assert q["position"]["offset_from_prior"] == pytest.approx(math.hypot(0.002, 0.001), abs=0.0007)


def test_measure_calls_views_that_saw_different_things_inconsistent(geo):
    here, there = pen_cloud(centre=(0.4, 0.1)), pen_cloud(centre=(0.412, 0.1))
    doubt = geo.normalize_doubt(
        {"id": "cap", "point": [0.406, 0.1, 0.11], "what": ["position"], "radius": 0.03}
    )
    r = geo.measure([("before", here), ("after", there)], doubt, surface_z=0.0)
    assert r["status"] == "inconsistent"
    assert "position" in r["detail"]
    assert r["quantities"]["position"]["checked"] is True


def test_measure_without_points_is_no_view(geo):
    doubt = geo.normalize_doubt({"point": [0.4, 0.1, 0.1], "what": ["height"]})
    r = geo.measure([("v1", np.zeros((0, 3)))], doubt, surface_z=0.0)
    assert r["status"] == "no_view"
    r = geo.measure([("v1", pen_cloud(centre=(0.4, 0.1)))], doubt, surface_z=None)
    assert r["status"] == "no_view" and "surface_z" in r["detail"]


# ----------------------------------------------------------------------------- the fake arm


def _rot(q):
    w, x, y, z = (q[k] for k in "wxyz")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _pose(m, script):
    return {
        "position": dict(zip("xyz", map(float, m[:3, 3]), strict=True)),
        "rotation": script._quat(m[:3, :3]),
    }


def render(cam_world, pen_xy, tray=TRAY):
    """rgb and z-depth of a table at z = 0, a stacked-cylinder pen and a box."""
    h, w = HW
    us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dc = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], -1)
    d = dc @ cam_world[:3, :3].T
    o = cam_world[:3, 3]
    t_best = np.full((h, w), np.inf)
    rgb = np.zeros((h, w, 3), np.uint8)

    def hit(t, colour):
        ok = np.isfinite(t) & (t > 1e-6) & (t < t_best)
        t_best[ok] = t[ok]
        rgb[ok] = colour

    with np.errstate(divide="ignore", invalid="ignore"):
        hit(-o[2] / d[..., 2], TABLE)
        for r, z0, z1, colour in PEN:
            ox, oy = o[0] - pen_xy[0], o[1] - pen_xy[1]
            a = d[..., 0] ** 2 + d[..., 1] ** 2
            b = 2 * (ox * d[..., 0] + oy * d[..., 1])
            c = ox * ox + oy * oy - r * r
            disc = b * b - 4 * a * c
            t = (-b - np.sqrt(np.where(disc >= 0, disc, np.nan))) / (2 * a)
            z = o[2] + t * d[..., 2]
            hit(np.where((z >= z0) & (z <= z1), t, np.inf), colour)
            tt = (z1 - o[2]) / d[..., 2]
            px, py = ox + tt * d[..., 0], oy + tt * d[..., 1]
            hit(np.where(px * px + py * py <= r * r, tt, np.inf), colour)
        centre, size, colour = tray
        lo, hi = centre - size / 2, centre + size / 2
        t1, t2 = (lo - o) / d, (hi - o) / d
        tmin, tmax = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
        hit(np.where(tmax >= tmin, tmin, np.inf), colour)
    depth = t_best  # z-depth: the ray's camera z is 1 per unit t
    depth = np.where(np.isfinite(depth), depth + RNG.normal(0, 0.0002, depth.shape), 0.0)
    return rgb, depth.astype(np.float32)


class Rig:
    def __init__(self, start=(0.25, 0.0, 0.30), ik=None, shift_after=None):
        self.tool = np.eye(4)
        self.tool[:3, :3] = _rot(TOP_DOWN)
        self.tool[:3, 3] = start
        self.start = self.tool.copy()
        self.pen_xy = PEN_XY.copy()
        self.ik = ik or (lambda pose: {"success": True, "within_servo_tolerance": True})
        self.moves = []
        self.frames = 0
        self.shift_after = shift_after

    def observation(self, script):
        cam = self.tool @ TOOL_CAM
        rgb, depth = render(cam, self.pen_xy)
        return {
            "cameras": [
                {
                    "name": "right_wrist",
                    "rgb": rgb,
                    "depth": depth,
                    "intrinsics": K,
                    "pose": _pose(cam, script),
                },
                {
                    "name": "overhead",
                    "rgb": rgb,
                    "depth": depth,
                    "intrinsics": K,
                    "pose": _pose(OVERHEAD, script),
                },
            ],
            "arms": [{"ee_pose": _pose(self.tool, script)}],
        }

    def ctx(self, script, extra=None):
        def go(pose, **kw):
            target = np.eye(4)
            target[:3, :3] = _rot(pose["rotation"])
            target[:3, 3] = [pose["position"][k] for k in "xyz"]
            self.moves.append(target.copy())
            self.tool = target
            return {"reached": True, "outcome": "reached"}

        def observe(**kw):
            self.frames += 1
            if self.shift_after is not None and self.frames > self.shift_after:
                self.pen_xy = PEN_XY + [0.012, 0.0]
            return self.observation(script)

        tools = {
            "robot.get_observation": observe,
            "robot.get_ee_pose": lambda **kw: {"pose": _pose(self.tool, script)},
            "robot.go_to_pose": go,
            "robot.solve_ik": lambda pose, **kw: self.ik(pose),
            "robot.describe_workspace": {"surface_z": 0.0},
            "robot.describe_gripper": {"finger": {"reach_m": 0.0102}},
        }
        tools.update(extra or {})
        return FakeContext(tools)


CAP = {
    "id": "green_cap",
    "point": [PEN_XY[0] + 0.002, PEN_XY[1] - 0.002, PEN_TOP - 0.01],
    "what": ["profile", "height", "position"],
    "radius": 0.015,
    "colour": {"h": [45, 85], "s": [120, 255], "v": [90, 255]},
    "prior": {"height_over_surface": 0.10},
    "write": {"height_over_surface": "pens.top_over_table_m"},
}
HOLDER = {
    "id": "tray",
    "point": [0.40, -0.06, 0.02],
    "what": ["height"],
    "radius": 0.035,
    "colour": {"v": [0, 60]},
    "write": {"height_over_surface": "tray.height_m"},
}
AROUND = [0.0, 120.0, -120.0]


def test_skill_measures_a_cap_and_a_tray_without_coming_near_them(script, tmp_path):
    rig = Rig()
    twin = tmp_path / "twin.json"
    twin.write_text(json.dumps({"pens": {"top_over_table_m": 0.10}}))
    out = script.run(
        rig.ctx(script),
        doubts=[CAP, HOLDER],
        azimuths_deg=AROUND,
        out_dir=str(tmp_path / "frames"),
        twin_path=str(twin),
    )
    assert out["route"] == "measured", out["detail"]
    assert out["n_measured"] == 2 and out["n_views"] >= 6
    cap, tray = out["doubts"]
    q = cap["quantities"]
    assert q["height_over_surface"]["value"] == pytest.approx(PEN_TOP, abs=0.0015)
    bands = q["profile"]["value"]
    assert bands[0]["diameter"] == pytest.approx(0.0115, abs=0.0015)
    assert bands[2]["diameter"] == pytest.approx(0.0185, abs=0.0015)
    assert q["position"]["value"][:2] == pytest.approx(PEN_XY, abs=0.0015)
    assert tray["quantities"]["height_over_surface"]["value"] == pytest.approx(0.025, abs=0.0015)
    # Nothing touched: every commanded tool centre kept the floor and the standoff.
    points = [np.array(CAP["point"]), np.array(HOLDER["point"])]
    for m in rig.moves:
        assert m[2, 3] >= 0.12 - 1e-9
        assert min(np.linalg.norm(m[:3, 3] - p) for p in points) >= 0.0102 + 0.03 - 1e-9
    # The hand is back where it started.
    assert rig.tool[:3, 3] == pytest.approx(rig.start[:3, 3], abs=1e-9)
    # Frames carry camera poses and files; the twin got the values with their source.
    assert all(f["camera"] == "right_wrist" and f["file"] for f in out["frames"])
    assert (tmp_path / "frames" / "measurements.json").exists()
    written = json.loads(twin.read_text())
    assert written["pens"]["top_over_table_m"] == pytest.approx(PEN_TOP, abs=0.0015)
    assert written["tray"]["height_m"] == pytest.approx(0.025, abs=0.0015)
    src = written["measured"]["green_cap"]["height_over_surface"]["source"]
    assert src["skill"] == BUNDLE and src["views"] and src["frames"]
    upd = next(
        u
        for u in out["twin_updates"]
        if u["doubt"] == "green_cap" and u["quantity"] == "height_over_surface"
    )
    assert upd["prior"] == 0.10 and upd["write_to"] == "pens.top_over_table_m"


def test_views_tilt_through_the_probes_angles(script):
    rig = Rig()
    out = script.run(rig.ctx(script), doubts=[CAP], azimuths_deg=[0.0])
    tilts = sorted({f["tilt_deg"] for f in out["frames"]})
    assert tilts and set(tilts) <= {30.0, 60.0, 75.0, 85.0}
    for f in out["frames"]:
        cam = script._matrix(f["camera_pose"])
        axis = cam[:3, 2]
        assert math.degrees(math.acos(-axis[2])) == pytest.approx(f["tilt_deg"], abs=0.5)
        assert np.linalg.norm(cam[:3, 3] - np.array(CAP["point"])) == pytest.approx(
            f["distance_m"], abs=0.002
        )
        assert "rgb" in f  # kept: no out_dir


def test_no_reachable_pose_is_not_reachable_and_nothing_moves(script):
    rig = Rig(ik=lambda pose: {"success": False})
    out = script.run(rig.ctx(script), doubts=[CAP])
    assert out["route"] == "not_reachable"
    assert rig.moves == []
    assert out["doubts"][0]["failures"]
    assert out["twin_updates"] == []


def test_an_empty_place_is_no_view_and_the_rest_is_still_measured(script):
    rig = Rig()
    ghost = dict(CAP, id="ghost", point=[0.30, 0.20, 0.10], write={})
    out = script.run(rig.ctx(script), doubts=[CAP, ghost], azimuths_deg=AROUND)
    assert out["route"] == "no_view"
    assert [d["status"] for d in out["doubts"]] == ["measured", "no_view"]
    assert out["n_measured"] == 1


def test_an_object_that_moves_between_views_is_inconsistent_and_not_written(script, tmp_path):
    rig = Rig(shift_after=6)
    twin = tmp_path / "twin.json"
    doubt = dict(CAP, what=["position"], radius=0.03)
    out = script.run(rig.ctx(script), doubts=[doubt], azimuths_deg=AROUND, twin_path=str(twin))
    assert out["route"] == "inconsistent", out["detail"]
    per_view = out["doubts"][0]["quantities"]["position"]["per_view"]
    assert max(per_view.values()) > 0.005
    assert json.loads(twin.read_text()) == {}


def test_a_segmenter_mask_selects_the_object(script):
    rig = Rig()
    calls = []

    def segment(image, query, max_results=5, **kw):
        calls.append(query)
        green = np.all(np.asarray(image) == GREEN, axis=-1)
        return {"masks": [green.astype(np.uint8) * 255], "scores": [0.9]}

    doubt = {k: v for k, v in CAP.items() if k != "colour"}
    doubt["object_name"] = "green pen cap"
    out = script.run(
        rig.ctx(script, {"sam3.segment_text": segment}), doubts=[doubt], azimuths_deg=AROUND
    )
    assert out["route"] == "measured", out["detail"]
    assert calls and set(calls) == {"green pen cap"}
    assert out["doubts"][0]["quantities"]["height_over_surface"]["value"] == pytest.approx(
        PEN_TOP, abs=0.0015
    )
