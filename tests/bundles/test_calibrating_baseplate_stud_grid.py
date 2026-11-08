"""calibrating-baseplate-stud-grid: FakeContext tests on ray-cast RGB-D renders of a baseplate
(tests/bundles/_plate_render.py): the lattice maths, the coarse sources (markers, a given pose, a
plate mask, sam3), every exit, the look-down move with a fake arm, and read_studs on bricks."""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

from . import _plate_render as R

BUNDLE = "calibrating-baseplate-stud-grid"
TW_P = R.plate_pose_world(0.45, -0.05, 0.02, 17.0, (1.0, -0.5))
WRIST_EYE = ([0.03, -0.06, 0.21], [0.01, 0.0, 0.0])
TOP_EYE = ([0.05, -0.25, 0.85], [0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def cal(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["calibrate_plate"].module


@pytest.fixture(scope="module")
def studs_reader(skills_registry):
    return skills_registry.get(BUNDLE).canonical_scripts["read_studs"].module


@pytest.fixture(scope="module")
def lat(cal):
    return cal.lat


def _se3(m):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(m[:3, :3]).as_quat()
    return {
        "position": dict(zip("xyz", map(float, m[:3, 3]), strict=True)),
        "rotation": {"w": w, "x": x, "y": y, "z": z},
    }


def _m(pose):
    from scipy.spatial.transform import Rotation

    q = pose["rotation"]
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    m[:3, 3] = [pose["position"][k] for k in "xyz"]
    return m


def _err(pose, truth=TW_P):
    e = np.linalg.inv(truth) @ _m(pose)
    return (
        np.abs(e[:3, 3]) * 1e3,
        abs(math.degrees(math.atan2(e[1, 0], e[0, 0]))),
        math.degrees(math.acos(min(1.0, e[2, 2]))),
    )


def _scene(system="lego", studs=(32, 32), bricks=None, markers=True, **kw):
    p, pt = R.SYSTEMS[system]["pitch"], R.SYSTEMS[system]["plate_t"]
    mk = []
    if markers:
        mk = [
            R.Marker(3, 0.04, studs[0] * p / 2 + 0.035, 0.0, -pt),
            R.Marker(7, 0.04, -studs[0] * p / 2 - 0.035, 0.05, -pt, 30.0),
        ]
    if bricks is None:
        bricks = [R.Brick(3, 4, 2, 4, 1, 0), R.Brick(8, 9, 2, 2, 1, 1)]
    return R.Scene(system=system, studs=studs, bricks=bricks, markers=mk, **kw)


def _layout(system="lego", studs=(32, 32)):
    p, pt = R.SYSTEMS[system]["pitch"], R.SYSTEMS[system]["plate_t"]
    return [
        {"id": 3, "size": 0.04, "position": [studs[0] * p / 2 + 0.035, 0.0, -pt]},
        {
            "id": 7,
            "size": 0.04,
            "position": [-studs[0] * p / 2 - 0.035, 0.05, -pt],
            "yaw_deg": 30.0,
        },
    ]


def _frame(name, scene, k, cam_p, seed, baseline=0.018, truth=TW_P):
    rgb, d = R.render(
        scene, k, cam_p, rng=np.random.default_rng(seed), baseline=baseline, supersample=1
    )
    return {"name": name, "rgb": rgb, "depth": d, "intrinsics": k, "pose": _se3(truth @ cam_p)}


@cache
def _obs(key="lego32"):
    system, studs, extra = {
        "lego32": ("lego", (32, 32), {}),
        "duplo16": ("duplo", (16, 16), {}),
        "lego15x9": ("lego", (15, 9), {}),
        "bare": ("lego", (32, 32), {"bare": True}),
        "covered": (
            "lego",
            (32, 32),
            {
                "bricks": [
                    R.Brick(x, y, 4, 2, 1, (x + y) % 3)
                    for x in range(4, 28, 4)
                    for y in range(6, 28, 2)
                    if (x + y) % 6
                ]
            },
        ),
    }[key]
    scene = _scene(system, studs, **extra)
    wrist = _frame("left_wrist", scene, R.D405_K, R.look_at(*WRIST_EYE), 1)
    top = _frame("top", scene, R.ORBBEC_K, R.look_at(*TOP_EYE), 2, baseline=0.05)
    return {"cameras": [top, wrist], "arms": [{"ee_pose": _se3(TW_P @ R.look_at(*WRIST_EYE))}]}


def _ctx(obs, **tools):
    return FakeContext({"robot.get_observation": obs, **tools})


def _coarse(dx=0.002, dy=-0.0015, dz=0.003, yaw=1.0, tilt=(0.5, 0.3)):
    return _se3(TW_P @ R.plate_pose_world(dx, dy, dz, yaw, tilt))


def _plate_mask(obs, studs, pitch):
    import cv2

    top = obs["cameras"][0]
    cp = np.linalg.inv(TW_P) @ _m(top["pose"])
    hx, hy = studs[0] * pitch / 2, studs[1] * pitch / 2
    c = (
        np.array([[-hx, -hy, 0, 1], [hx, -hy, 0, 1], [hx, hy, 0, 1], [-hx, hy, 0, 1]])
        @ np.linalg.inv(cp).T
    )
    uv = (R.ORBBEC_K @ c[:, :3].T).T
    mask = np.zeros((480, 640), np.uint8)
    cv2.fillConvexPoly(mask, np.round(uv[:, :2] / uv[:, 2:]).astype(np.int32), 1)
    return mask


# -- the lattice maths ----------------------------------------------------------------


def test_lattice_fit_recovers_yaw_and_offset_through_noise_and_outliers(lat):
    rng = np.random.default_rng(4)
    p, yaw, t = 0.008, math.radians(7.0), np.array([0.0021, -0.0013])
    ij = np.stack(np.meshgrid(np.arange(-6, 7), np.arange(-4, 5)), -1).reshape(-1, 2)
    g = (ij + 0.5) * p
    q = g @ lat.rz(yaw).T + t + rng.normal(0, 0.0003, g.shape)
    q = q[rng.random(len(q)) > 0.2]
    q = np.vstack([q, rng.uniform(-0.05, 0.05, (40, 2))])
    fit = lat.fit_lattice(q, p, rng, phase=(0.5, 0.5), yaw_prior=0.0)
    assert fit["n_inliers"] >= 0.75 * len(g) - 5
    assert abs(math.degrees(fit["yaw"]) - 7.0) < 0.1
    t2, _, margin = lat.minimal_shift(fit["yaw"], fit["t"], p, np.zeros(2))
    assert np.allclose(t2, t, atol=1e-4) and margin < 0.35
    assert fit["rms"] < 0.0006


def test_lattice_yaw_is_folded_into_plus_minus_45_deg(lat):
    rng = np.random.default_rng(0)
    p = 0.016
    ij = np.stack(np.meshgrid(np.arange(5), np.arange(4)), -1).reshape(-1, 2)
    q = (ij * p) @ lat.rz(math.radians(80.0)).T
    fit = lat.fit_lattice(q, p, rng)
    assert abs(math.degrees(fit["yaw"]) - (-10.0)) < 1e-6
    assert fit["n_inliers"] == len(q) and fit["rms"] < 1e-9


def test_minimal_shift_picks_the_nearest_row_and_reports_the_margin(lat):
    p = 0.008
    t, corr, margin = lat.minimal_shift(0.0, np.array([0.0110, -0.0045]), p, np.zeros(2))
    assert np.allclose(t, [0.0030, 0.0035]) and np.allclose(corr, t)
    assert margin == pytest.approx(0.4375)


def test_stepped_plane_goes_down_past_brick_tops_and_up_to_a_brick_top(lat):
    rng = np.random.default_rng(1)
    plate = np.c_[rng.uniform(0, 0.1, (300, 2)), rng.normal(0, 1e-4, 300)]
    tops = np.c_[rng.uniform(0, 0.1, (600, 2)), 0.0096 + rng.normal(0, 1e-4, 600)]
    pts = np.vstack([plate, tops])
    n, d, _ = lat.stepped_plane(pts, 0.0006, rng, drop=0.005)
    assert abs(d) < 3e-4 and n[2] > 0.999
    n, d, _ = lat.stepped_plane(
        np.vstack([tops[:200], plate]), 0.0006, rng, drop=0.0034, direction=1
    )
    assert -d == pytest.approx(0.0096, abs=3e-4)


def test_apex_entry_has_apex_mr_keys_and_yaw(cal):
    m = R.plate_pose_world(0.4089, 0.0457, 0.1899, math.degrees(0.013861))
    a = cal._apex(m, [48, 48])
    assert set(a) == {"x", "y", "z", "roll", "pitch", "yaw", "deg", "width", "height"}
    assert a["yaw"] == pytest.approx(0.013861) and a["deg"] == pytest.approx(0.7942, abs=1e-3)
    assert a["roll"] == pytest.approx(0.0, abs=1e-12) and (a["width"], a["height"]) == (48, 48)


# -- calibrate_plate on renders, no motion ------------------------------------------------


def test_a_given_coarse_pose_is_refined_to_sub_mm(cal):
    out = cal.run(_ctx(_obs()), [32, 32], "lego", coarse_plate_pose=_coarse(), look=False)
    assert out["route"] == "calibrated" and out["lattice_status"] == "ok"
    xyz, yaw, tilt = _err(out["plate_pose"])
    assert xyz.max() < 0.3 and yaw < 0.1 and tilt < 0.2
    assert out["studs_used"] > 500 and out["residual_mm"] < 1.0
    assert out["coarse_source"] == "input" and out["snap_margin"] < 0.35
    assert out["correction"]["yaw_deg"] == pytest.approx(-1.0, abs=0.1)
    assert out["apex_plate"]["width"] == 32 and out["views"][0]["view_point"] == []


def test_duplo_plate_is_refined(cal):
    obs = _obs("duplo16")
    out = cal.run(_ctx(obs), [16, 16], "duplo", coarse_plate_pose=_coarse(0.004, 0.003), look=False)
    assert out["route"] == "calibrated"
    xyz, yaw, _ = _err(out["plate_pose"])
    assert xyz.max() < 0.3 and yaw < 0.1 and out["stud_pitch"] == 0.016


def test_an_odd_plate_uses_the_shifted_lattice(cal):
    out = cal.run(_ctx(_obs("lego15x9")), [15, 9], "lego", coarse_plate_pose=_coarse(), look=False)
    assert out["route"] == "calibrated"
    assert _err(out["plate_pose"])[0].max() < 0.3


def test_markers_give_the_coarse_pose_and_the_lattice_refines_it(cal):
    obs = _obs()
    coarse = cal.run(_ctx(obs), [32, 32], "lego", markers=_layout(), refine=False)
    assert coarse["route"] == "coarse_only" and coarse["lattice_status"] == "not_requested"
    assert coarse["markers_used"] == [3, 7] and coarse["marker_reprojection_px"] < 1.5
    xyz, yaw, tilt = _err(coarse["plate_pose"])
    assert xyz[:2].max() < 3.0 and xyz[2] < 6.0 and yaw < 1.0 and tilt < 4.0
    assert coarse["plate_pose"] == coarse["coarse_plate_pose"]
    out = cal.run(_ctx(obs), [32, 32], "lego", markers=_layout(), look=False)
    assert out["route"] == "calibrated" and out["coarse_source"] == "markers"
    xyz, yaw, tilt = _err(out["plate_pose"])
    assert xyz.max() < 0.3 and yaw < 0.1 and tilt < 0.2


def test_slide_onto_plane_meets_the_plane_on_the_ray(cal):
    eye = np.array([-0.25, -0.1, 0.9])
    n = np.array([0.0, 0.0, 1.0])
    p = cal._slide_onto_plane(n, -0.013, eye)
    assert p[2] == pytest.approx(0.013)
    assert np.allclose(np.cross(p - eye, -eye), 0.0, atol=1e-12)
    assert cal._slide_onto_plane(n, 0.0, np.array([1.0, 0.0, 0.1])) is None


def test_a_marker_pose_off_along_the_overhead_ray_is_slid_onto_the_close_up_plane(cal, monkeypatch):
    # The YAM twin's overhead camera put two 50 mm markers' plate 1.4 % too far
    # along its ray at 1 m (13 mm low, 4.5 mm sideways): one stud row's snap
    # tolerance is 2.8 mm on LEGO, so the fit was ambiguous.
    obs = _obs()
    eye = _m(obs["cameras"][0]["pose"])[:3, 3]
    far = TW_P.copy()
    far[:3, 3] = eye + 1.014 * (TW_P[:3, 3] - eye)
    lateral = np.linalg.norm((np.linalg.inv(TW_P) @ far)[:2, 3])
    assert lateral > 0.35 * 0.008

    def found(cams, layout, dictionary):
        return {
            "pose": far,
            "camera": "top",
            "camera_position": eye,
            "ids": [3, 7],
            "reprojection_px": 0.5,
        }

    monkeypatch.setattr(cal, "_coarse_from_markers", found)
    out = cal.run(_ctx(obs), [32, 32], "lego", markers=_layout(), look=False)
    assert out["route"] == "calibrated" and out["lattice_status"] == "ok"
    xyz, yaw, tilt = _err(out["plate_pose"])
    assert xyz.max() < 0.3 and yaw < 0.1 and tilt < 0.2
    assert out["correction"]["ray_slide"] == pytest.approx(
        0.014 * np.linalg.norm(TW_P[:3, 3] - eye), rel=0.1
    )
    assert "slid" in out["detail"]
    given = cal.run(_ctx(obs), [32, 32], "lego", coarse_plate_pose=_se3(far), look=False)
    assert given["route"] == "no_lattice" and given["lattice_status"] == "ambiguous"


def test_an_apriltag_dictionary_is_detected(cal):
    scene = _scene(markers=False)
    scene.markers = [R.Marker(11, 0.05, 0.165, 0.02, -0.0032)]
    scene.dictionary = "DICT_APRILTAG_36h11"
    top = _frame("top", scene, R.ORBBEC_K, R.look_at(*TOP_EYE), 2, baseline=0.05)
    out = cal.run(
        _ctx({"cameras": [top], "arms": []}),
        [32, 32],
        markers=[{"id": 11, "size": 0.05, "position": [0.165, 0.02, -0.0032]}],
        marker_dictionary="DICT_APRILTAG_36h11",
    )
    # One tag and no wrist camera: a coarse pose only.
    assert out["route"] == "coarse_only" and out["lattice_status"] == "no_wrist_camera"
    # One 50 mm tag at 0.85 m: ~1 cm and ~4 deg (the renders: 3 / 7 mm, 3.7 deg tilt).
    assert out["markers_used"] == [11] and _err(out["plate_pose"])[0][:2].max() < 10.0


def test_a_plate_mask_gives_the_coarse_pose(cal):
    obs = _obs()
    mask = _plate_mask(obs, (32, 32), 0.008)
    out = cal.run(_ctx(obs), [32, 32], plate_mask=mask, look=False, yaw_hint_deg=17.0)
    assert out["route"] == "calibrated" and out["coarse_source"] == "mask_plane"
    assert out["yaw_ambiguity_deg"] == 90.0
    assert _err(out["coarse_plate_pose"])[0][:2].max() < 3.0
    assert _err(out["plate_pose"])[0].max() < 0.3


def test_a_plate_prompt_calls_sam3_on_the_overhead_frame(cal):
    obs = _obs()
    mask = _plate_mask(obs, (32, 32), 0.008) * 255
    seen = {}

    def segment(image, query, max_results):
        seen.update(query=query, shape=np.asarray(image).shape, max_results=max_results)
        return {"masks": [mask], "scores": [0.9], "boxes": [{}]}

    out = cal.run(
        _ctx(obs, **{"sam3.segment_text": segment}),
        [32, 32],
        plate_prompt="the green baseplate",
        refine=False,
        yaw_hint_deg=17.0,
    )
    assert out["route"] == "coarse_only" and out["coarse_source"] == "mask_plane"
    assert seen == {"query": "the green baseplate", "shape": (480, 640, 3), "max_results": 1}


def test_nothing_to_start_from_is_no_markers(cal):
    out = cal.run(_ctx(_obs()), [32, 32])
    assert out["route"] == "no_markers" and out["plate_pose"] == {}


def test_listed_markers_not_seen_is_no_markers(cal):
    out = cal.run(
        _ctx(_obs()), [32, 32], markers=[{"id": 40, "size": 0.04, "position": [0.2, 0, 0]}]
    )
    assert out["route"] == "no_markers" and "none of markers [40] seen" in out["detail"]


def test_listed_markers_not_seen_fall_back_to_the_given_pose(cal):
    out = cal.run(
        _ctx(_obs()),
        [32, 32],
        markers=[{"id": 40, "size": 0.04, "position": [0.2, 0, 0]}],
        coarse_plate_pose=_coarse(),
        look=False,
    )
    assert out["route"] == "calibrated" and out["coarse_source"] == "input"


def test_no_wrist_camera_is_coarse_only(cal):
    obs = _obs()
    out = cal.run(
        _ctx({"cameras": obs["cameras"][:1], "arms": obs["arms"]}),
        [32, 32],
        coarse_plate_pose=_coarse(),
    )
    assert out["route"] == "coarse_only" and out["lattice_status"] == "no_wrist_camera"
    assert out["plate_pose"] == out["coarse_plate_pose"]


def test_a_plate_without_studs_is_no_lattice(cal):
    out = cal.run(_ctx(_obs("bare")), [32, 32], coarse_plate_pose=_coarse(), look=False)
    assert out["route"] == "no_lattice" and out["lattice_status"] == "too_few_studs"
    assert out["plate_pose"] == out["coarse_plate_pose"]


def test_a_coarse_pose_half_a_pitch_off_is_ambiguous(cal):
    out = cal.run(
        _ctx(_obs()), [32, 32], coarse_plate_pose=_coarse(0.0041, 0.0, 0.0, 0.0, (0, 0)), look=False
    )
    assert out["route"] == "no_lattice" and out["lattice_status"] == "ambiguous"
    assert out["snap_margin"] > 0.35 and out["studs_used"] > 500


def test_a_coarse_yaw_off_by_20_deg_is_yaw_off(cal):
    out = cal.run(
        _ctx(_obs()), [32, 32], coarse_plate_pose=_coarse(0, 0, 0, 20.0, (0, 0)), look=False
    )
    assert out["route"] == "no_lattice" and out["lattice_status"] == "yaw_off"


def test_a_tight_residual_gate_is_high_residual(cal):
    out = cal.run(
        _ctx(_obs()), [32, 32], coarse_plate_pose=_coarse(), look=False, max_residual=0.00005
    )
    assert out["route"] == "no_lattice" and out["lattice_status"] == "high_residual"
    assert out["residual_mm"] > 0.05


def test_a_coarse_pose_far_above_the_plate_finds_no_plane(cal):
    out = cal.run(_ctx(_obs()), [32, 32], coarse_plate_pose=_coarse(0, 0, 0.05), look=False)
    assert out["route"] == "no_lattice" and out["lattice_status"] == "no_plane"


def test_bricks_covering_most_of_the_view_are_not_taken_for_the_plate(cal):
    out = cal.run(_ctx(_obs("covered")), [32, 32], coarse_plate_pose=_coarse(dz=0.004), look=False)
    assert out["route"] == "calibrated"
    xyz, yaw, _ = _err(out["plate_pose"])
    assert xyz.max() < 0.4 and yaw < 0.1


def test_bad_arguments_raise(cal):
    with pytest.raises(ValueError, match="system"):
        cal.run(_ctx(_obs()), [32, 32], "megablok")
    with pytest.raises(ValueError, match="plate_studs"):
        cal.run(_ctx(_obs()), [32])
    with pytest.raises(ValueError, match="marker needs id and size"):
        cal.run(_ctx(_obs()), [32, 32], markers=[{"size": 0.04}])
    with pytest.raises(ValueError, match="dictionary"):
        cal.run(_ctx(_obs()), [32, 32], markers=_layout(), marker_dictionary="DICT_NOPE")


# -- the look-down move ---------------------------------------------------------------------

T_TOOL_CAM = np.eye(4)
_s = math.sqrt(0.5)
T_TOOL_CAM[:3, :3] = np.column_stack([[0, 1, 0], [-_s, 0, _s], [_s, 0, _s]])
T_TOOL_CAM[:3, 3] = [-0.086, 0.0, -0.075]
"""A YAM-like wrist D405: 86 mm behind and 75 mm above the tool centre, looking 45 deg forward."""
TOP_DOWN = np.diag([1.0, -1.0, -1.0])


class Arm:
    """Moves land where commanded unless ``refuse`` says otherwise; the wrist frame is
    rendered from wherever the tool is."""

    def __init__(self, refuse=None, land_off=None, n_arms=1):
        self.tool = np.eye(4)
        self.tool[:3, :3] = TOP_DOWN
        self.tool[:3, 3] = (0.35, 0.0, 0.05)
        self.n_arms = n_arms
        self.refuse = refuse or (lambda pose: None)
        self.land_off = land_off
        self.moves = []
        self.kwargs = []
        self.top = _obs()["cameras"][0]

    def go(self, pose, **kwargs):
        self.moves.append(_m(pose))
        self.kwargs.append(kwargs)
        why = self.refuse(pose)
        if why == "raise":
            raise ToolError("go_to_pose", "joint limit")
        if why:
            return {"reached": False, "outcome": "unreachable", "reason": why}
        self.tool = _m(pose)
        if self.land_off is not None:
            self.tool[:3, 3] += self.land_off
        return {"reached": True, "outcome": "reached"}

    def ee(self, **_):
        return {"pose": _se3(self.tool)}

    def obs(self, **_):
        cam = self.tool @ T_TOOL_CAM
        return {
            "cameras": [self.top, _wrist_render(cam.round(6).tobytes())],
            "arms": [{"ee_pose": _se3(self.tool)}] * self.n_arms,
        }


@cache
def _wrist_render(cam_bytes):
    cam = np.frombuffer(cam_bytes).reshape(4, 4)
    return _frame("left_wrist", _scene(), R.D405_K, np.linalg.inv(TW_P) @ cam, 5)


def _grasp_frame(**_):
    return {"rotation": _se3(np.eye(4) @ np.diag([1, -1, -1, 1]))["rotation"]}


def _arm_ctx(arm, ik=None):
    return FakeContext(
        {
            "robot.get_observation": arm.obs,
            "robot.get_ee_pose": arm.ee,
            "robot.go_to_pose": arm.go,
            "robot.solve_ik": ik or {"success": True, "within_servo_tolerance": True},
            "robot.grasp_frame": _grasp_frame,
        }
    )


def test_look_places_the_camera_over_the_plate_looking_down_and_calibrates(cal):
    arm = Arm()
    out = cal.run(_arm_ctx(arm), [32, 32], "lego", coarse_plate_pose=_coarse())
    assert out["route"] == "calibrated"
    # A rise in place first, then the view.
    assert len(arm.moves) == 2 and np.allclose(arm.moves[0][:2, 3], [0.35, 0.0])
    cam = arm.tool @ T_TOOL_CAM
    up = _m(_coarse())[:3, 2]
    assert float(cam[:3, 2] @ -up) > math.cos(math.radians(0.5))
    centre = _m(_coarse())[:3, 3]
    assert float((cam[:3, 3] - centre) @ up) == pytest.approx(0.20, abs=1e-6)
    assert np.linalg.norm(np.cross(cam[:3, 3] - centre, up)) < 1e-6
    assert out["views"][0]["camera_range"] == 0.20 and out["views"][0]["view_point"] == [0.0, 0.0]
    xyz, yaw, tilt = _err(out["plate_pose"])
    assert xyz.max() < 0.3 and yaw < 0.1 and tilt < 0.2
    assert arm.kwargs == [{}, {}]


def test_two_view_points_pool_their_studs(cal):
    arm = Arm()
    out = cal.run(
        _arm_ctx(arm),
        [32, 32],
        coarse_plate_pose=_coarse(),
        view_points=[[-0.06, 0.0], [0.06, 0.0]],
        check_reach=False,
    )
    assert out["route"] == "calibrated" and len(out["views"]) == 2
    assert out["studs_used"] > 1000
    assert _err(out["plate_pose"])[1] < 0.1


def test_an_ik_preview_refusing_every_view_is_coarse_only_without_moving(cal):
    arm = Arm()
    out = cal.run(_arm_ctx(arm, ik={"success": False}), [32, 32], coarse_plate_pose=_coarse())
    assert out["route"] == "coarse_only" and out["lattice_status"] == "unreachable"
    assert arm.moves == [] and "out of reach (IK preview)" in out["detail"]


def test_routed_and_raised_refusals_are_coarse_only(cal):
    arm = Arm(refuse=lambda pose: "collision")
    out = cal.run(_arm_ctx(arm), [32, 32], coarse_plate_pose=_coarse())
    assert out["route"] == "coarse_only" and "unreachable: collision" in out["detail"]
    arm = Arm(refuse=lambda pose: "raise")
    out = cal.run(_arm_ctx(arm), [32, 32], coarse_plate_pose=_coarse())
    assert out["route"] == "coarse_only" and "joint limit" in out["detail"]
    assert len(arm.moves) <= 2 * cal.MAX_MOVES


def test_a_hand_landing_3_cm_off_is_not_used(cal):
    arm = Arm(land_off=np.array([0.03, 0.0, 0.0]))
    out = cal.run(_arm_ctx(arm), [32, 32], coarse_plate_pose=_coarse())
    assert out["route"] == "coarse_only" and "from the view pose" in out["detail"]


def test_connector_knobs_are_passed_only_when_set(cal):
    arm = Arm(n_arms=2)
    out = cal.run(
        _arm_ctx(arm),
        [32, 32],
        coarse_plate_pose=_coarse(),
        arm_id=1,
        tolerance=0.004,
        max_steps=500,
    )
    assert out["route"] == "calibrated"
    assert all(k == {"arm_id": 1, "tolerance": 0.004, "max_steps": 500} for k in arm.kwargs)


# -- read_studs ---------------------------------------------------------------------------


def _brick_frame(system, brick, eye, others=(), yaw=25.0, image_shift=(0.0, 0.0)):
    scene = R.Scene(system=system, studs=(16, 16), bricks=[brick, *others])
    p, bh = R.SYSTEMS[system]["pitch"], R.SYSTEMS[system]["brick_h"]
    cx = -8 * p + (brick.x + brick.nx / 2) * p
    cy = -8 * p + (brick.y + brick.ny / 2) * p
    tw = R.plate_pose_world(0.4, 0.1, 0.0, yaw)
    target = np.array([cx + image_shift[0], cy + image_shift[1], bh * brick.layer])
    cam_p = R.look_at(np.array(eye) + target * [1, 1, 0], target)
    rgb, d = R.render(scene, R.D405_K, cam_p, rng=np.random.default_rng(3), supersample=1)
    mask = R.render.last_ids == 2
    frame = {
        "name": "left_wrist",
        "rgb": rgb,
        "depth": d,
        "intrinsics": R.D405_K,
        "pose": _se3(tw @ cam_p),
    }
    top = tw @ np.array([cx, cy, bh * brick.layer, 1.0])
    return frame, mask, top[:3], tw, cam_p


@pytest.mark.parametrize(
    ("system", "brick", "eye"),
    [
        ("lego", R.Brick(5, 6, 2, 4), [0.0, -0.05, 0.17]),
        ("lego", R.Brick(5, 6, 1, 6), [0.02, -0.02, 0.15]),
        ("lego", R.Brick(5, 6, 4, 2), [0.0, -0.05, 0.17]),
        ("duplo", R.Brick(3, 3, 2, 3), [0.0, -0.06, 0.20]),
        ("lego", R.Brick(5, 6, 1, 1), [0.0, -0.05, 0.17]),
    ],
)
def test_read_studs_reads_the_footprint_and_top_pose(studs_reader, system, brick, eye):
    frame, mask, top, tw, _ = _brick_frame(system, brick, eye)
    out = studs_reader.run(FakeContext({}), mask=mask, frame=frame, system=system)
    assert out["route"] == "lattice_found" and out["footprint_complete"]
    assert (
        out["brick_size"] == sorted([brick.nx, brick.ny])
        and out["stud_count"] == brick.nx * brick.ny
    )
    t = _m(out["top_pose"])
    assert np.linalg.norm(t[:3, 3] - top) < 0.0005
    long_x = brick.nx > brick.ny
    want = 25.0 + (90.0 if long_x else 0.0)
    got = math.degrees(math.atan2(t[1, 0], t[0, 0]))
    fold = 90.0 if brick.nx == brick.ny else 180.0
    assert abs((got - want + fold / 2) % fold - fold / 2) < 1.5
    assert out["yaw_ambiguity_deg"] == fold and len(out["stud_px"]) == out["stud_count"]


def test_read_studs_pixels_land_on_the_studs(studs_reader):
    frame, mask, _, tw, cam_p = _brick_frame("lego", R.Brick(5, 6, 2, 4), [0.0, -0.05, 0.17])
    out = studs_reader.run(FakeContext({}), mask=mask, frame=frame)
    p = 0.008
    truth = []
    for i in range(2):
        for j in range(4):
            c = np.array(
                [-8 * p + (5 + i + 0.5) * p, -8 * p + (6 + j + 0.5) * p, 0.0096 + 0.0017, 1.0]
            )
            pc = np.linalg.inv(cam_p) @ c
            truth.append(
                [R.D405_K[0, 0] * pc[0] / pc[2] + 320, R.D405_K[1, 1] * pc[1] / pc[2] + 240]
            )
    got = np.array(out["stud_px"])
    d = np.linalg.norm(got[:, None] - np.array(truth)[None], axis=2).min(axis=1)
    assert d.max() < 1.0


def test_a_brick_cut_by_the_image_border_is_occluded(studs_reader):
    frame, mask, _, _, _ = _brick_frame(
        "lego", R.Brick(5, 4, 2, 8), [0.0, 0.0, 0.10], image_shift=(0.0, 0.03)
    )
    out = studs_reader.run(FakeContext({}), mask=mask, frame=frame)
    assert out["route"] == "occluded" and out["open_sides"]
    assert out["brick_size"][1] < 8


def test_a_brick_half_covered_by_another_is_occluded(studs_reader):
    cover = R.Brick(5, 8, 2, 2, layer=2, colour=1)
    frame, mask, _, _, _ = _brick_frame(
        "lego", R.Brick(5, 6, 2, 4), [0.0, -0.05, 0.17], others=[cover]
    )
    out = studs_reader.run(FakeContext({}), mask=mask, frame=frame)
    assert out["route"] == "occluded"
    assert out["brick_size"] == [2, 2] and out["open_sides"]


def test_a_mask_on_the_bare_table_has_too_few_studs(studs_reader):
    frame, mask, _, _, _ = _brick_frame("lego", R.Brick(5, 6, 2, 4), [0.0, -0.05, 0.17])
    table = R.render.last_ids == 0
    out = studs_reader.run(FakeContext({}), mask=table, frame=frame)
    assert out["route"] == "too_few_studs"
    out = studs_reader.run(FakeContext({}), mask=np.zeros_like(mask), frame=frame)
    assert out["route"] == "too_few_studs" and "empty" in out["detail"]


def test_read_studs_takes_the_wrist_frame_from_the_observation_and_unposed_frames(studs_reader):
    frame, mask, top, _, cam_p = _brick_frame("lego", R.Brick(5, 6, 2, 4), [0.0, -0.05, 0.17])
    out = studs_reader.run(
        FakeContext({"robot.get_observation": {"cameras": [frame], "arms": []}}), mask=mask
    )
    assert out["route"] == "lattice_found" and out["camera_name"] == "left_wrist"
    bare = {k: v for k, v in frame.items() if k != "pose"}
    out2 = studs_reader.run(FakeContext({}), mask=mask, frame=bare)
    assert out2["route"] == "lattice_found" and out2["brick_size"] == [2, 4]
    # Unposed: the top pose is in the camera frame, equal to the posed run's lattice_pose_camera.
    assert np.allclose(
        _m(out2["top_pose"])[:3, 3], _m(out["lattice_pose_camera"])[:3, 3], atol=3e-4
    )


def test_read_studs_without_a_mask_reads_the_plate(studs_reader):
    frame = _obs()["cameras"][1]
    out = studs_reader.run(FakeContext({}), frame=frame)
    assert out["route"] == "lattice_found" and out["stud_count"] > 100 and out["open_sides"] == []


def test_read_studs_with_no_frame_has_too_few_studs(studs_reader):
    out = studs_reader.run(FakeContext({"robot.get_observation": {"cameras": [], "arms": []}}))
    assert out["route"] == "too_few_studs" and "no RGB-D frame" in out["detail"]
    with pytest.raises(ValueError, match="system"):
        studs_reader.run(FakeContext({}), system="megablok")
