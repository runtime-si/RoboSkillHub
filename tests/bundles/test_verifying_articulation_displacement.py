"""verifying-articulation-displacement: handle re-localisation and the verdict on ray-cast
RGB-D frames of a cabinet with drawers or a side-hinged door (tests/bundles/_articulation_scene.py)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

from . import _articulation_scene as S

BUNDLE = "verifying-articulation-displacement"
EYE = (0.0, -0.25, 0.75)
TARGET = (0.45, 0.0, 0.20)
PULL = {"x": -1.0, "y": 0.0, "z": 0.0}
UP = {"x": 0.0, "y": 0.0, "z": 1.0}
HINGE = dict(zip("xyz", map(float, S.HINGE), strict=True))


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["verify_articulation"].module


def drawer(opening, **kw):
    cam = S.frame(S.drawer_scene(opening, **kw), EYE, TARGET)
    cam["_handle"] = S.drawer_handle_front(opening) + np.asarray(kw.get("shift", (0, 0, 0)))
    return cam


def door(deg):
    cam = S.frame(S.door_scene(math.radians(deg)), EYE, TARGET)
    cam["_handle"] = S.door_handle_front(math.radians(deg))
    return cam


def obs(cam):
    return {"cameras": [cam]}


def tools(*cams, aim=None, sam=True, **extra):
    """A pointer that points at each frame's handle (or ``aim(cam)``) and a SAM that
    returns the handle's mask."""

    def which(image):
        return next(c for c in cams if c["rgb"] is image)

    def point(image, query, **kw):
        cam = which(image)
        u, v = aim(cam) if aim else S.project(cam, cam["_handle"])
        return {"found": True, "pixel_x": u, "pixel_y": v, "points": [{"pixel_x": u, "pixel_y": v}]}

    def segment(image, pixel_x, pixel_y):
        return {"masks": [S.mask_of(which(image), "handle")], "scores": [0.9]}

    responses = {"gemini-er.point": point}
    if sam:
        responses["sam3.segment_point"] = segment
    responses.update(extra)
    return FakeContext(responses)


def check_drawer(script, before, after, expected, pull_direction=PULL, **kw):
    b, a = drawer(before), drawer(after)
    ctx = tools(a, b)
    out = script.run(
        ctx,
        "the handle of the upper drawer",
        "prismatic",
        expected,
        observation=obs(a),
        before_observation=obs(b),
        pull_direction=pull_direction,
        **kw,
    )
    return out, ctx


def check_door(script, before, after, expected_deg, **kw):
    b, a = door(before), door(after)
    out = script.run(
        tools(a, b),
        "the door handle",
        "revolute",
        math.radians(expected_deg),
        observation=obs(a),
        before_observation=obs(b),
        hinge_point=HINGE,
        hinge_axis=UP,
        pull_direction=PULL,
        **kw,
    )
    return out


# --- drawers -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("before", "after", "expected", "route"),
    [
        (0.0, 0.15, 0.15, "moved"),
        (0.0, 0.11, 0.15, "moved"),
        (0.0, 0.06, 0.15, "moved_short"),
        (0.0, 0.01, 0.15, "not_moved"),
        (0.15, 0.0, -0.15, "moved"),
        (0.15, 0.14, -0.15, "not_moved"),
        (0.05, 0.12, -0.10, "moved_wrong_way"),
        (0.10, 0.04, 0.10, "moved_wrong_way"),
    ],
)
def test_drawer_routes(script, before, after, expected, route):
    out, _ = check_drawer(script, before, after, expected)
    assert out["route"] == route, out["detail"]
    assert out["seen"]
    assert out["measured"] == pytest.approx(after - before, abs=0.003)
    assert out["fraction"] == pytest.approx(out["measured"] / expected)
    assert out["normal_dot"] > 0.99
    assert out["transverse"] < 0.02


def test_drawer_handle_point_and_normal(script):
    out, ctx = check_drawer(script, 0.0, 0.12, 0.12)
    after = np.array([out["handle_after"][k] for k in "xyz"])
    truth = S.drawer_handle_front(0.12)
    # The handle's front face, near the bar's middle (the window is 26 mm round the seed).
    assert abs(after[0] - truth[0]) < 0.004
    assert np.linalg.norm(after[1:] - truth[1:]) < 0.03
    n = np.array([out["face_normal_after"][k] for k in "xyz"])
    assert n == pytest.approx([-1.0, 0.0, 0.0], abs=0.05)  # PCA of the inliers, within 3 deg
    assert ctx.call_count("gemini-er.point") == 2
    assert ctx.call_count("sam3.segment_point") == 2
    assert ctx.calls_to("gemini-er.point")[0].kwargs["max_points"] == 3


def test_drawer_from_a_given_handle_point_and_pull_as_normal(script):
    a = drawer(0.15)
    out = script.run(
        tools(a),
        "the drawer handle",
        "prismatic",
        0.15,
        observation=obs(a),
        handle_before=dict(zip("xyz", map(float, S.drawer_handle_front(0.0)), strict=True)),
        pull_direction=PULL,
    )
    assert out["route"] == "moved", out["detail"]
    assert out["measured"] == pytest.approx(0.15, abs=0.003)
    assert "handle before given" in out["detail"]


def test_drawer_direction_from_the_face_normal_alone(script):
    """Without pull_direction a drawer moves along the face normal measured before."""
    out, _ = check_drawer(script, 0.0, 0.10, 0.10, pull_direction=None)
    assert out["route"] == "moved", out["detail"]
    assert out["measured"] == pytest.approx(0.10, abs=0.003)


def test_another_handle_found_after_is_inconsistent(script):
    b, a = drawer(0.0), drawer(0.0)
    lower = np.array([S.FRONT_X - 0.02 - S.HANDLE_PROUD[1], 0.0, 0.12])
    out = script.run(
        tools(a, b, aim=lambda c: S.project(c, lower if c is a else c["_handle"]), sam=False),
        "the handle of the upper drawer",
        "prismatic",
        0.15,
        observation=obs(a),
        before_observation=obs(b),
        pull_direction=PULL,
        refine_with_sam=False,
    )
    assert out["route"] == "inconsistent", out["detail"]
    assert out["transverse"] > 0.15


def test_turned_face_is_inconsistent(script):
    a = drawer(0.15)
    turned = {"x": -math.cos(math.radians(40)), "y": math.sin(math.radians(40)), "z": 0.0}
    out = script.run(
        tools(a),
        "the drawer handle",
        "prismatic",
        0.15,
        observation=obs(a),
        handle_before=dict(zip("xyz", map(float, S.drawer_handle_front(0.0)), strict=True)),
        face_normal_before=turned,
        pull_direction=PULL,
    )
    assert out["route"] == "inconsistent", out["detail"]
    assert out["normal_dot"] == pytest.approx(math.cos(math.radians(40)), abs=0.02)


def test_too_long_a_travel_is_inconsistent(script):
    a = drawer(0.0)
    far = S.drawer_handle_front(0.0) + np.array([0.30, 0.0, 0.0])
    out = script.run(
        tools(a),
        "the drawer handle",
        "prismatic",
        0.15,
        observation=obs(a),
        handle_before=dict(zip("xyz", map(float, far), strict=True)),
        pull_direction=PULL,
    )
    assert out["route"] == "inconsistent", out["detail"]
    assert out["measured"] == pytest.approx(0.30, abs=0.005)


def test_a_drawer_that_moved_sideways_is_inconsistent(script):
    b, a = drawer(0.0), drawer(0.10, shift=(0.0, 0.09, 0.0))
    out = script.run(
        tools(a, b),
        "the drawer handle",
        "prismatic",
        0.10,
        observation=obs(a),
        before_observation=obs(b),
        pull_direction=PULL,
    )
    assert out["route"] == "inconsistent", out["detail"]
    assert out["transverse"] == pytest.approx(0.09, abs=0.01)


# --- doors ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("before", "after", "expected", "route"),
    [
        (0, 45, 45, "moved"),
        (0, 60, 45, "moved"),
        (0, 20, 45, "moved_short"),
        (0, 3, 45, "not_moved"),
        (40, 0, -40, "moved"),
        (30, 10, 45, "moved_wrong_way"),
    ],
)
def test_door_routes(script, before, after, expected, route):
    out = check_door(script, before, after, expected)
    assert out["route"] == route, out["detail"]
    assert math.degrees(out["measured"]) == pytest.approx(after - before, abs=0.5)
    # The fitted normal of a door near shut takes in the cabinet's front beside its edge:
    # up to 8 deg off in this scene, inside the 26 deg the check allows.
    assert math.degrees(out["normal_angle"]) == pytest.approx(after - before, abs=10.0)
    assert out["normal_dot"] > 0.98
    assert out["radius"] == pytest.approx(0.36, abs=0.03)


def test_door_axis_sign_follows_the_pull_hint(script):
    """The same door with the axis given upside down still opens positive."""
    b, a = door(0), door(30)
    out = script.run(
        tools(a, b),
        "the door handle",
        "revolute",
        math.radians(30),
        observation=obs(a),
        before_observation=obs(b),
        hinge_point=HINGE,
        hinge_axis={"x": 0.0, "y": 0.0, "z": -1.0},
        pull_direction=PULL,
    )
    assert out["route"] == "moved", out["detail"]
    assert math.degrees(out["measured"]) == pytest.approx(30, abs=0.5)


def test_door_without_a_normal_before_skips_the_normal_check(script):
    a = door(45)
    out = script.run(
        tools(a),
        "the door handle",
        "revolute",
        math.radians(45),
        observation=obs(a),
        handle_before=dict(zip("xyz", map(float, S.door_handle_front(0.0)), strict=True)),
        hinge_point=HINGE,
        hinge_axis=UP,
    )
    assert out["route"] == "moved", out["detail"]
    assert math.isnan(out["normal_dot"])
    assert "no face normal before" in out["detail"]


#: The YAM station overhead camera's calibrated pose and intrinsics, in this scene's frame
#: (the cabinet 0.55 m in front of the left arm base): it sees the faces steeply from above.
STATION_POSE = np.array(
    [
        [-0.002095, -0.872016, 0.489473, 0.134908],
        [-0.999978, 0.004935, 0.004513, -0.074157],
        [-0.006351, -0.489453, -0.872007, 0.961199],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
STATION_K = np.array([[461.2, 0.0, 320.0], [0.0, 461.5, 240.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("kind", ["prismatic", "revolute"])
def test_seen_steeply_by_the_station_camera(script, kind):
    def view(parts, handle):
        cam = S.frame(parts, None, None, 640, 480, pose=STATION_POSE, k=STATION_K)
        cam["_handle"] = handle
        return cam

    if kind == "prismatic":
        b = view(S.drawer_scene(0.0), S.drawer_handle_front(0.0))
        a = view(S.drawer_scene(0.15), S.drawer_handle_front(0.15))
        expected, extra = 0.15, {}
    else:
        b = view(S.door_scene(0.0), S.door_handle_front(0.0))
        a = view(S.door_scene(math.radians(45)), S.door_handle_front(math.radians(45)))
        expected, extra = math.radians(45), {"hinge_point": HINGE, "hinge_axis": UP}
    out = script.run(
        tools(a, b),
        "the handle",
        kind,
        expected,
        observation=obs(a),
        before_observation=obs(b),
        pull_direction=PULL,
        **extra,
    )
    assert out["route"] == "moved", out["detail"]
    assert out["measured"] == pytest.approx(expected, abs=0.003 if kind == "prismatic" else 0.01)


# --- not seen ------------------------------------------------------------------


def test_pointer_finds_nothing(script):
    a = drawer(0.1)
    ctx = FakeContext(
        {"gemini-er.point": {"found": False, "pixel_x": 0.0, "pixel_y": 0.0, "points": []}}
    )
    out = script.run(
        ctx,
        "the drawer handle",
        "prismatic",
        0.1,
        observation=obs(a),
        handle_before=[0.5, 0, 0.3],
        pull_direction=PULL,
    )
    assert out["route"] == "not_seen"
    assert not out["seen"]
    assert "found no" in out["detail"]


def test_pointer_failure_is_not_seen(script):
    def boom(**kw):
        raise RuntimeError("quota")

    a = drawer(0.1)
    out = script.run(
        FakeContext({"gemini-er.point": boom}),
        "the drawer handle",
        "prismatic",
        0.1,
        observation=obs(a),
        handle_before=[0.5, 0, 0.3],
        pull_direction=PULL,
    )
    assert out["route"] == "not_seen"
    assert "quota" in out["detail"]


def test_pixel_on_the_table_has_no_vertical_face(script):
    a = drawer(0.1)
    out = script.run(
        tools(a, sam=False),
        "the drawer handle",
        "prismatic",
        0.1,
        observation=obs(a),
        handle_before=[0.5, 0, 0.3],
        pull_direction=PULL,
        handle_pixel=[300, 230],
        refine_with_sam=False,
    )
    assert out["route"] == "not_seen"
    assert "no vertical face plane" in out["detail"]


def test_pixel_on_a_bare_face_finds_no_handle(script):
    a = drawer(0.0)
    bare = np.array([S.FRONT_X - 0.02, -0.10, S.DRAWER_Z - 0.05])
    out = script.run(
        tools(a, sam=False),
        "the drawer handle",
        "prismatic",
        0.1,
        observation=obs(a),
        handle_before=[0.5, 0, 0.3],
        pull_direction=PULL,
        handle_pixel=S.project(a, bare),
        refine_with_sam=False,
    )
    assert out["route"] == "not_seen"
    assert "nothing proud" in out["detail"]


def test_handle_not_seen_before(script):
    b, a = drawer(0.0), drawer(0.1)
    out = script.run(
        tools(a, b, aim=lambda c: (300, 230) if c is b else S.project(c, c["_handle"])),
        "the drawer handle",
        "prismatic",
        0.1,
        observation=obs(a),
        before_observation=obs(b),
        pull_direction=PULL,
        refine_with_sam=False,
    )
    assert out["route"] == "not_seen"
    assert out["detail"].startswith("before actuation")


# --- tools and plumbing --------------------------------------------------------


def test_fresh_observation_when_none_given(script):
    a = drawer(0.12)
    ctx = tools(a, **{"robot.get_observation": obs(a)})
    out = script.run(
        ctx,
        "the drawer handle",
        "prismatic",
        0.12,
        handle_before=dict(zip("xyz", map(float, S.drawer_handle_front(0.0)), strict=True)),
        pull_direction=PULL,
    )
    assert out["route"] == "moved", out["detail"]
    assert ctx.call_count("robot.get_observation") == 1


def test_handle_pixels_skip_the_pointer(script):
    b, a = drawer(0.0), drawer(0.12)
    ctx = FakeContext({})
    out = script.run(
        ctx,
        "the drawer handle",
        "prismatic",
        0.12,
        observation=obs(a),
        before_observation=obs(b),
        pull_direction=PULL,
        refine_with_sam=False,
        handle_pixel=S.project(a, a["_handle"]),
        handle_pixel_before=S.project(b, b["_handle"]),
    )
    assert out["route"] == "moved", out["detail"]
    assert ctx.calls == []


def test_molmo_pointer_gets_no_max_points(script):
    a = drawer(0.12)

    def molmo(image, query):
        u, v = S.project(a, a["_handle"])
        return {"found": True, "pixel_x": u, "pixel_y": v}

    ctx = FakeContext({"molmo.point_prompt": molmo})
    out = script.run(
        ctx,
        "the drawer handle",
        "prismatic",
        0.12,
        observation=obs(a),
        handle_before=dict(zip("xyz", map(float, S.drawer_handle_front(0.0)), strict=True)),
        pull_direction=PULL,
        pointer="molmo.point_prompt",
        refine_with_sam=False,
    )
    assert out["route"] == "moved", out["detail"]
    assert set(ctx.calls_to("molmo.point_prompt")[0].kwargs) == {"image", "query"}


def test_sam_pulls_an_off_handle_point_onto_it_and_failure_keeps_the_point(script):
    a = drawer(0.12)
    handle = a["_handle"]
    near = lambda c: S.project(c, handle + np.array([0.0, 0.0, 0.035]))  # noqa: E731
    before = dict(zip("xyz", map(float, S.drawer_handle_front(0.0)), strict=True))
    run = lambda ctx: script.run(  # noqa: E731
        ctx,
        "the drawer handle",
        "prismatic",
        0.12,
        observation=obs(a),
        handle_before=before,
        pull_direction=PULL,
    )
    # 35 mm above the bar, the window still reaches it; SAM brings the seed onto it.
    out = run(tools(a, aim=near))
    assert out["route"] == "moved", out["detail"]

    def broken(**kw):
        raise RuntimeError("no GPU")

    out = run(tools(a, aim=lambda c: S.project(c, handle), **{"sam3.segment_point": broken}))
    assert out["route"] == "moved", out["detail"]


def test_far_sam_centroid_is_ignored(script):
    a = drawer(0.12)
    elsewhere = np.zeros_like(S.mask_of(a, "handle"))
    elsewhere[5:20, 5:20] = 255
    ctx = tools(a, **{"sam3.segment_point": {"masks": [elsewhere], "scores": [0.9]}})
    out = script.run(
        ctx,
        "the drawer handle",
        "prismatic",
        0.12,
        observation=obs(a),
        handle_before=dict(zip("xyz", map(float, S.drawer_handle_front(0.0)), strict=True)),
        pull_direction=PULL,
    )
    assert out["route"] == "moved", out["detail"]


@pytest.mark.parametrize(("answer", "route"), [(True, "moved"), (False, "not_seen")])
def test_vlm_confirmation(script, answer, route):
    a = drawer(0.12)
    ctx = tools(a, **{"vlm.query_yes_no": {"answer": answer, "text": "yes" if answer else "no"}})
    out = script.run(
        ctx,
        "the drawer handle",
        "prismatic",
        0.12,
        observation=obs(a),
        handle_before=dict(zip("xyz", map(float, S.drawer_handle_front(0.0)), strict=True)),
        pull_direction=PULL,
        confirm_with_vlm=True,
    )
    assert out["route"] == route, out["detail"]
    call = ctx.calls_to("vlm.query_yes_no")[0].kwargs
    full, crop = call["images"]
    assert full.shape == a["rgb"].shape and crop.shape == (256, 256, 3)
    assert "the drawer handle" in call["prompt"]


def test_depth_at_half_the_rgb_resolution(script):
    b, a = drawer(0.0), drawer(0.12)
    for cam in (a, b):
        cam["depth"] = cam["depth"][::2, ::2]
        k = cam["intrinsics"].copy()
        k[:2] /= 2.0
        cam["intrinsics"] = k
    out = script.run(
        tools(a, b),
        "the drawer handle",
        "prismatic",
        0.12,
        observation=obs(a),
        before_observation=obs(b),
        pull_direction=PULL,
    )
    assert out["route"] == "moved", out["detail"]
    assert out["measured"] == pytest.approx(0.12, abs=0.005)


def test_roborsi_single_pixel_handle_point(script):
    out, _ = check_drawer(script, 0.0, 0.15, 0.15, front_band=0.0)
    assert out["route"] == "moved", out["detail"]
    assert out["measured"] == pytest.approx(0.15, abs=0.004)


def test_noisy_depth_still_measures(script):
    rng = np.random.default_rng(3)

    def noisy(opening):
        cam = S.frame(
            S.drawer_scene(opening),
            EYE,
            TARGET,
            noise=lambda d, r: np.where(
                d > 0, np.round(d + r.normal(0, 0.003, d.shape) * d**2, 3), 0.0
            ),
            rng=rng,
        )
        cam["_handle"] = S.drawer_handle_front(opening)
        return cam

    b, a = noisy(0.0), noisy(0.10)
    out = script.run(
        tools(a, b),
        "the drawer handle",
        "prismatic",
        0.10,
        observation=obs(a),
        before_observation=obs(b),
        pull_direction=PULL,
    )
    assert out["route"] == "moved", out["detail"]
    assert out["measured"] == pytest.approx(0.10, abs=0.004)


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"joint_type": "ball"}, "joint_type"),
        ({"expected_opening": 0.0}, "non-zero"),
        ({"joint_type": "revolute"}, "hinge_point"),
        ({"handle_before": None}, "handle_before or before_observation"),
    ],
)
def test_bad_inputs_raise(script, kw, message):
    args = {
        "handle_description": "the drawer handle",
        "joint_type": "prismatic",
        "expected_opening": 0.1,
        "handle_before": [0.5, 0.0, 0.3],
        "observation": obs(drawer(0.0)),
    }
    args.update(kw)
    with pytest.raises(ValueError, match=message):
        script.run(FakeContext({}), **args)


# --- geometry ------------------------------------------------------------------


def test_face_plane_fit(script):
    rng = np.random.default_rng(0)
    y, z = rng.uniform(-0.1, 0.1, (2, 40))
    wall = np.c_[0.5 + rng.normal(0, 0.002, 40), y, 0.3 + z]
    camera = np.array([0.0, 0.0, 0.8])
    normal, centre, _ = script.fit_face_plane(wall, wall.mean(0), camera)
    assert normal == pytest.approx([-1.0, 0.0, 0.0], abs=0.02)
    assert centre[0] == pytest.approx(0.5, abs=0.002)
    floor = np.c_[0.5 + y, z, rng.normal(0, 0.002, 40)]
    assert script.fit_face_plane(floor, floor.mean(0), camera) is None
    lid = script.fit_face_plane(floor, floor.mean(0), camera, vertical=False)
    assert lid[0] == pytest.approx([0.0, 0.0, 1.0], abs=0.02)
    line = np.c_[np.full(40, 0.5), y, np.full(40, 0.3)]
    assert script.fit_face_plane(line, line.mean(0), camera) is None


# --- fits planning-articulation-opening ----------------------------------------


def test_takes_the_outputs_of_planning_articulation_opening(script, skills_registry):
    planner = (
        skills_registry.get("planning-articulation-opening")
        .canonical_scripts["plan_articulation_opening"]
        .module
    )
    handle = S.drawer_handle_front(0.0)
    handle_pose = {
        "position": dict(zip("xyz", map(float, handle), strict=True)),
        "rotation": {"w": 0.7071068, "x": 0.0, "y": 0.7071068, "z": 0.0},
    }
    plan = planner.run(
        FakeContext({"robot.solve_ik": lambda pose, **kw: {"success": True, "joints": [0.0] * 6}}),
        handle_pose=handle_pose,
        joint_type="prismatic",
        pull_direction=PULL,
        distance=0.14,
    )
    assert plan["route"] == "planned"
    a = drawer(plan["opening"])
    out = script.run(
        tools(a),
        "the drawer handle",
        plan["joint_type"],
        plan["opening"],
        observation=obs(a),
        handle_before=handle_pose,
        pull_direction=PULL,
    )
    assert out["route"] == "moved", out["detail"]
    assert out["measured"] == pytest.approx(0.14, abs=0.003)

    hinge_plan = planner.run(
        FakeContext({"robot.solve_ik": lambda pose, **kw: {"success": True}}),
        handle_pose={
            "position": dict(zip("xyz", map(float, S.door_handle_front(0.0)), strict=True)),
            "rotation": handle_pose["rotation"],
        },
        joint_type="revolute",
        hinge_point=HINGE,
        hinge_axis=UP,
        pull_direction=PULL,
        angle_deg=50,
    )
    a = door(math.degrees(hinge_plan["opening"]))
    out = script.run(
        tools(a),
        "the door handle",
        hinge_plan["joint_type"],
        hinge_plan["opening"],
        observation=obs(a),
        handle_before={
            "position": dict(zip("xyz", map(float, S.door_handle_front(0.0)), strict=True))
        },
        hinge_point=HINGE,
        hinge_axis=UP,
        pull_direction=PULL,
    )
    assert out["route"] == "moved", out["detail"]
    assert math.degrees(out["measured"]) == pytest.approx(50, abs=0.5)


# --- the bundle ----------------------------------------------------------------


def test_example_declares_every_route(script):
    root = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
    example = json.loads((root / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    source = (root / "scripts" / "verify_articulation.py").read_text()
    import re

    returned = set(re.findall(r'finish\(\s*"([a-z_]+)"', source))
    assert returned == declared
    skill = (root / "SKILL.md").read_text()
    for route in declared:
        assert f"    {route}:" in skill
