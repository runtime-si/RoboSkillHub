"""routing-parts-to-sort-bins-by-id-and-colour: FakeContext tests of the landing / settling / capture gates on
rendered RGB-D frames, the colour + footprint, the bin rules with capacities, and the pick-and-drop."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

from .test_grasping_flat_objects import Arm

BUNDLE = "routing-parts-to-sort-bins-by-id-and-colour"
TABLE_Z = 0.0
CAM_H = 1.0
F = 500.0
K = np.array([[F, 0.0, 320.0], [0.0, F, 240.0], [0.0, 0.0, 1.0]])
# A camera 1 m over the table looking straight down (OpenCV axes: x = world x, y = world -y).
CAM_POSE = {
    "position": {"x": 0.0, "y": 0.0, "z": CAM_H},
    "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0},
}
ZONE = {"center": {"x": 0.0, "y": 0.0, "z": TABLE_Z}, "half": [0.05, 0.05]}
TABLE_RGB = (196, 198, 202)
COLOURS = {
    "red": (200, 30, 20),
    "blue": (10, 80, 190),
    "yellow": (240, 200, 50),
    "grey": (196, 198, 202),
}
BINS = [
    {
        "name": "red",
        "drop_position": {"x": 0.30, "y": 0.10, "z": 0.04},
        "opening_half": [0.045, 0.045],
        "capacity": 2,
    },
    {
        "name": "blue",
        "drop_position": {"x": 0.30, "y": -0.10, "z": 0.04},
        "opening_half": [0.045, 0.045],
    },
    {
        "name": "other",
        "accepts": ["unknown"],
        "overflow": True,
        "drop_position": {"x": 0.40, "y": 0.0, "z": 0.04},
    },
]
FAST = {"poll_s": 0.0, "settle_s": 0.0, "max_wait_s": 5.0}


@pytest.fixture(scope="module")
def plan(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["plan_sort_route"].module


@pytest.fixture(scope="module")
def sort(skills_registry):
    return skills_registry.get(BUNDLE).canonical_scripts["sort_into_bin"].module


def render(parts=(), gain=1.0, seed=0):
    """An RGB-D frame of the table with boxes ``{at: (x, y), size: (l, w, h), yaw_deg, colour}`` on it."""
    rng = np.random.default_rng(seed)
    v, u = np.mgrid[0:480, 0:640].astype(float)
    x = (u - 320.0) / F * CAM_H
    y = -(v - 240.0) / F * CAM_H
    rgb = np.empty((480, 640, 3))
    rgb[:] = TABLE_RGB
    depth = np.full((480, 640), CAM_H - TABLE_Z)
    for p in parts:
        length, width, h = p["size"]
        a = math.radians(p.get("yaw_deg", 0.0))
        dx, dy = x - p["at"][0], y - p["at"][1]
        along = dx * math.cos(a) + dy * math.sin(a)
        across = -dx * math.sin(a) + dy * math.cos(a)
        inside = (np.abs(along) <= length / 2) & (np.abs(across) <= width / 2)
        top = CAM_H - TABLE_Z - h
        depth[inside] = top  # the top face, seen straight down
        rgb[inside] = COLOURS[p.get("colour", "red")]
    rgb = np.clip(rgb * gain + rng.normal(0.0, 1.5, rgb.shape), 0, 255).astype(np.uint8)
    return {
        "name": "top",
        "rgb": rgb,
        "depth": depth.astype(np.float32),
        "intrinsics": K,
        "pose": CAM_POSE,
    }


def obs(frame):
    return {"cameras": [frame], "arms": []}


def part(at=(0.0, 0.0), size=(0.025, 0.025, 0.025), yaw_deg=0.0, colour="red"):
    return {"at": at, "size": size, "yaw_deg": yaw_deg, "colour": colour}


def classify_by_mean(rgb, mask=None, palette=None, **kw):
    """A stand-in for colour.classify: nearest named colour to the mask's mean, 'unknown_colour' far off."""
    m = np.asarray(mask) > 0
    mean = np.asarray(rgb)[m].reshape(-1, 3).mean(axis=0)
    names = [n for n in COLOURS if n != "grey"]
    d = [float(np.linalg.norm(mean - np.array(COLOURS[n]))) for n in names]
    i = int(np.argmin(d))
    status = "classified" if d[i] < 60 else "unknown_colour"
    return {"status": status, "name": names[i], "delta_e": d[i], "second_name": None}


class Camera:
    """robot.get_observation serving the frames in order, then the last one forever."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.served = 0

    def __call__(self, **kw):
        i = min(self.served, len(self.frames) - 1)
        self.served += 1
        return obs(self.frames[i])


def plan_ctx(frames, classify=classify_by_mean):
    cam = Camera(frames)
    ctx = FakeContext({"robot.get_observation": cam, "colour.classify": classify})
    return ctx, cam


def run_plan(plan, frames, **kw):
    ctx, cam = plan_ctx(frames, kw.pop("classify", classify_by_mean))
    args = {"bins": BINS, "staging_zone": ZONE, **FAST, **kw}
    return plan.run(ctx, reference=obs(render()), **args), ctx, cam


# ---------------------------------------------------------------- plan: perception


def test_a_red_part_is_routed_to_the_red_bin_with_its_box(plan):
    p = part(at=(0.012, -0.008), size=(0.030, 0.020, 0.020), yaw_deg=30.0)
    out, ctx, _ = run_plan(plan, [render([p])] * 4)
    assert out["route"] == "routed", out["detail"]
    assert (
        out["bin_name"] == "red" and out["matched_by"] == "colour" and out["part_colour"] == "red"
    )
    assert out["drop_position"] == {"x": 0.30, "y": 0.10, "z": 0.04}
    assert out["sort_bin"]["name"] == "red"
    c = out["part_obb"]["center"]
    assert (c["x"], c["y"]) == pytest.approx((0.012, -0.008), abs=0.002)
    assert c["z"] == pytest.approx(0.010, abs=0.002)
    assert out["long_mm"] == pytest.approx(30.0, abs=2.5)
    assert out["short_mm"] == pytest.approx(20.0, abs=2.5)
    assert out["height_mm"] == pytest.approx(20.0, abs=1.0)
    q = out["part_obb"]["orientation"]
    yaw = math.degrees(2 * math.atan2(q["z"], q["w"]))
    assert (yaw - 30.0 + 90.0) % 180.0 - 90.0 == pytest.approx(0.0, abs=4.0)
    assert out["support_z"] == pytest.approx(TABLE_Z, abs=1e-3)
    # The colour mask is the part: its mean is the part's colour, not the table's.
    call = next(c for c in ctx.calls if c.tool == "colour.classify")
    mask = np.asarray(call.kwargs["mask"]) > 0
    assert 0.5 * 15 * 10 < mask.sum() <= 15 * 10 * 1.3
    assert out["bin_counts"] == {}


def test_lego_sorters_priority_colour_then_category_then_unknown(plan):
    bins = [*BINS, {"name": "bricks", "accepts": ["brick"]}]
    yellow = [render([part(colour="yellow")])] * 3
    out, _, _ = run_plan(plan, yellow, bins=bins)
    assert out["route"] == "routed" and out["bin_name"] == "other" and out["label"] == "unknown"
    out, _, _ = run_plan(plan, yellow, bins=bins, part_id="3001", categories={"3001": "Brick"})
    assert (
        out["bin_name"] == "bricks" and out["matched_by"] == "category" and out["part_id"] == "3001"
    )
    out, _, _ = run_plan(
        plan,
        [render([part(colour="red")])] * 3,
        bins=bins,
        part_id="3001",
        categories={"3001": "Brick"},
    )
    assert out["bin_name"] == "red" and out["matched_by"] == "colour"


def test_no_bin_for_the_part_is_unknown_part(plan):
    bins = [b for b in BINS if b["name"] != "other"]
    out, _, _ = run_plan(plan, [render([part(colour="yellow")])] * 3, bins=bins)
    assert out["route"] == "unknown_part"
    assert out["part_colour"] == "yellow"
    assert out["part_obb"], "the part was still measured"


def test_an_ambiguous_colour_is_unknown_unless_accepted(plan):
    def ambiguous(rgb, mask=None, **kw):
        return {"status": "ambiguous", "name": "red", "second_name": "orange", "delta_e": 9.0}

    frames = [render([part()])] * 3
    out, _, _ = run_plan(plan, frames, classify=ambiguous)
    assert out["bin_name"] == "other" and out["part_colour"] == "unknown"
    assert "ambiguous" in out["detail"]
    out, _, _ = run_plan(plan, frames, classify=ambiguous, accept_ambiguous=True)
    assert out["bin_name"] == "red"


def test_a_full_bin_overflows_then_routes_bin_full(plan):
    frames = [render([part()])] * 3
    out, _, _ = run_plan(plan, frames, bin_counts={"red": 2})
    assert out["route"] == "routed" and out["bin_name"] == "other"
    assert "overflow" in out["matched_by"] and out["bin_counts"] == {"red": 2}
    bins = [{**b, "overflow": False} for b in BINS]
    out, _, _ = run_plan(plan, frames, bins=bins, bin_counts={"red": 2})
    assert out["route"] == "bin_full" and out["bin_name"] == "red"
    full_other = [*BINS[:2], {**BINS[2], "capacity": 1}]
    out, _, _ = run_plan(plan, frames, bins=full_other, bin_counts={"red": 2, "other": 1})
    assert out["route"] == "bin_full"


def test_rules_route_by_size_before_colour(plan):
    rules = [{"bin": "blue", "min_long_mm": 28}, {"bin": "red", "colour": ["red", "orange"]}]
    big = [render([part(size=(0.032, 0.020, 0.020))])] * 3
    out, _, _ = run_plan(plan, big, rules=rules)
    assert out["bin_name"] == "blue" and out["matched_by"] == "rule 0"
    small = [render([part(size=(0.020, 0.020, 0.020))])] * 3
    out, _, _ = run_plan(plan, small, rules=rules)
    assert out["bin_name"] == "red" and out["matched_by"] == "rule 1"
    out, _, _ = run_plan(plan, small, rules=[{"bin": "nowhere", "colour": "red"}])
    assert out["route"] == "unknown_part"


def test_size_classes_name_the_category(plan):
    classes = [{"name": "small", "max_long_mm": 24}, {"name": "large"}]
    bins = [{"name": "s", "accepts": ["small"]}, {"name": "l", "accepts": ["large"]}]
    out, _, _ = run_plan(
        plan,
        [render([part(size=(0.030, 0.030, 0.02), colour="yellow")])] * 3,
        bins=bins,
        size_classes=classes,
    )
    assert out["category"] == "large" and out["bin_name"] == "l"
    out, _, _ = run_plan(
        plan,
        [render([part(size=(0.020, 0.020, 0.02), colour="yellow")])] * 3,
        bins=bins,
        size_classes=classes,
    )
    assert out["category"] == "small" and out["bin_name"] == "s"


def test_a_part_the_colour_of_the_table_is_found_by_depth(plan):
    out, _, _ = run_plan(plan, [render([part(colour="grey")])] * 3)
    assert out["route"] == "routed", out["detail"]
    assert out["long_mm"] == pytest.approx(25.0, abs=2.5)
    assert out["part_colour"] == "unknown" and out["bin_name"] == "other"


def test_rgb_only_frames_route_without_a_box(plan):
    def flat(frame):
        return {"name": "top", "rgb": frame["rgb"]}

    ctx, _ = plan_ctx([flat(render([part()]))] * 3)
    out = plan.run(
        ctx, reference=obs(flat(render())), bins=BINS,
        staging_roi={"top": 0.4, "bottom": 0.6, "left": 0.4, "right": 0.6}, **FAST,
    )  # fmt: skip
    assert out["route"] == "routed" and out["bin_name"] == "red"
    assert out["part_obb"] == {} and "no part box" in out["detail"]


# ---------------------------------------------------------------- plan: lego-sorter's cycle


def test_nothing_on_the_pad_is_staging_empty(plan):
    out, _, cam = run_plan(plan, [render()], max_wait_s=0.0)
    assert out["route"] == "staging_empty"
    out, _, cam = run_plan(plan, [render()], wait=False, max_wait_s=10.0)
    assert out["route"] == "staging_empty" and cam.served == 1


def test_an_exposure_change_is_not_a_part(plan):
    out, _, _ = run_plan(plan, [render(gain=1.3)], max_wait_s=0.0)
    assert out["route"] == "staging_empty", out["detail"]
    out, _, _ = run_plan(plan, [render([part()], gain=1.25)] * 3)
    assert out["route"] == "routed" and out["part_colour"] == "red"


def test_the_part_must_be_seen_on_two_frames_and_still(plan):
    frames = [
        render(),
        render([part(at=(0.0, 0.0))]),  # one frame occupied, then gone: a false trigger
        render(),
        render([part(at=(-0.025, 0.0))]),
        render([part(at=(0.020, 0.010))]),  # still moving (a zone-mean motion over 8)
        render([part(at=(0.008, 0.004))]),
        render([part(at=(0.008, 0.004))]),
    ]
    out, _, cam = run_plan(plan, frames)
    assert out["route"] == "routed", out["detail"]
    assert cam.served == len(frames)
    c = out["part_obb"]["center"]
    assert (c["x"], c["y"]) == pytest.approx((0.008, 0.004), abs=0.002)


def test_a_part_that_never_settles_is_staging_not_clear(plan):
    frames = [render([part(at=(0.03 * ((i % 3) - 1), 0.0))]) for i in range(200)]
    out, _, _ = run_plan(plan, frames, max_wait_s=0.2, poll_s=0.01)
    assert out["route"] == "staging_not_clear" and "did not settle" in out["detail"]


def test_two_parts_are_rejected_four_times(plan):
    two = render([part(at=(-0.025, 0.0)), part(at=(0.025, 0.0), colour="blue")])
    out, _, cam = run_plan(plan, [two] * 20)
    assert out["route"] == "staging_not_clear"
    assert "multiple parts" in out["detail"] and "4 times" in out["detail"]


def test_a_part_on_the_zone_edge_is_taken_whole_after_four_rejects(plan):
    edge = render([part(at=(0.048, 0.0), size=(0.030, 0.025, 0.02))])
    out, _, cam = run_plan(plan, [edge] * 30)
    assert out["route"] == "routed", out["detail"]
    assert out["edge_override"] is True
    # Straight down only the top face is seen: its outer pixel centres span one 2 mm pixel less.
    assert out["long_mm"] == pytest.approx(30.0 - 2.0, abs=1.5)
    assert out["part_obb"]["center"]["x"] == pytest.approx(0.048, abs=0.002)


def test_no_reference_or_bins(plan):
    ctx, _ = plan_ctx([render([part()])])
    assert (
        plan.run(ctx, reference=None, bins=BINS, staging_zone=ZONE, **FAST)["route"]
        == "staging_empty"
    )
    assert (
        plan.run(ctx, reference=obs(render()), bins=[], staging_zone=ZONE, **FAST)["route"]
        == "unknown_part"
    )
    far = {"center": {"x": 5.0, "y": 0.0, "z": 0.0}, "half": [0.05, 0.05]}
    out = plan.run(ctx, reference=obs(render()), bins=BINS, staging_zone=far, **FAST)
    assert out["route"] == "staging_empty" and "outside the frame" in out["detail"]


def test_a_missing_colour_tool_routes_on_the_other_labels(plan):
    def broken(**kw):
        raise RuntimeError("colour bundle not served")

    out, _, _ = run_plan(plan, [render([part()])] * 3, classify=broken)
    assert out["route"] == "routed" and out["bin_name"] == "other"
    assert out["colour_status"] == "error"


def test_brickognize_is_off_by_default_and_used_when_asked(plan, monkeypatch):
    calls = []

    class Reply:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def urlopen(req, timeout=0):
        calls.append(req.full_url)
        return Reply({"items": [{"id": "3003", "category": "Brick", "score": 0.9}]})

    monkeypatch.setattr(plan.urllib.request, "urlopen", urlopen)
    bins = [*BINS, {"name": "bricks", "accepts": ["brick"]}]
    frames = [render([part(colour="yellow")])] * 3
    out, _, _ = run_plan(plan, frames, bins=bins)
    assert calls == [] and out["bin_name"] == "other"
    out, _, _ = run_plan(plan, frames, bins=bins, use_brickognize=True)
    assert calls == [plan.BRICKOGNIZE_URL]
    assert out["part_id"] == "3003" and out["category"] == "brick" and out["bin_name"] == "bricks"

    def down(req, timeout=0):
        raise OSError("no network")

    monkeypatch.setattr(plan.urllib.request, "urlopen", down)
    out, _, _ = run_plan(plan, frames, bins=bins, use_brickognize=True)
    assert out["bin_name"] == "other" and "Brickognize unavailable" in out["detail"]


# ---------------------------------------------------------------- plan: lego-sorter's helpers


def test_lego_sorter_blob_helpers(plan):
    a = np.zeros((40, 40), bool)
    a[5:15, 5:15] = True  # 100 px part
    a[5:15, 16:18] = True  # 20 px shadow 1 px away: folded in
    assert plan.competing_blob_sizes(a) == [120]
    assert not plan.significant_secondary_blob(a)
    a[25:32, 25:32] = True  # 49 px, far: another part (>= 0.35 x 120)
    assert plan.significant_secondary_blob(a)
    assert plan.touches_edge((0, 5, 10, 10), (40, 40)) and not plan.touches_edge(
        (5, 5, 10, 10), (40, 40)
    )
    assert plan.slug("Brick, Modified") == "brick_modified"
    flat = np.full((30, 30, 3), 180, np.uint8)
    assert plan.crop_looks_empty(flat)
    flat[10:20, 10:20] = (200, 30, 20)
    assert not plan.crop_looks_empty(flat)
    assert plan.exposure_gain(np.full(10, 100.0), np.full(10, 50.0)) == pytest.approx(1.85)
    assert plan.exposure_gain(np.full(10, 100.0), np.full(10, 99.0)) == 1.0


def test_choose_bin_directly(plan):
    p = {
        "colour": "unknown",
        "category": "",
        "part_id": "",
        "long_mm": -1.0,
        "short_mm": -1.0,
        "height_mm": -1.0,
    }
    assert plan.choose_bin(p, BINS[:2], None, {})["route"] == "unknown_part"
    assert (
        plan.choose_bin(p, BINS, [{"bin": "blue", "max_long_mm": 30}], {})["bin"]["name"] == "other"
    )


# ---------------------------------------------------------------- sort_into_bin


PART_OBB = {
    "center": {"x": 0.40, "y": 0.10, "z": 0.10 + 0.0125},
    "extent": {"x": 0.0125, "y": 0.0125, "z": 0.0125},
    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}
BIN = {
    "name": "red",
    "drop_position": {"x": 0.30, "y": 0.30, "z": 0.14},
    "opening_half": [0.045, 0.045],
}


def sort_arm(**kw):
    return Arm(between=lambda arm: 0.025, **kw)


def test_pick_and_drop_into_a_taught_bin(sort):
    arm = sort_arm()
    ctx = arm.ctx()
    out = sort.run(ctx, part_obb=PART_OBB, sort_bin=BIN, bin_counts={"red": 1}, support_z=0.10)
    assert out["route"] == "sorted", out["detail"]
    assert out["grasp_route"] == "grasped" and out["place_route"] == "released"
    assert out["bin_counts"] == {"red": 2}
    # The part's bottom 15 mm over the rim at the opening's centre when the hand opened.
    r = out["release_position"]
    assert (r["x"], r["y"]) == pytest.approx((0.30, 0.30), abs=1e-6)
    assert r["z"] == pytest.approx(0.14 + 0.015, abs=1e-6)
    names = [c.tool for c in ctx.calls]
    assert names.count("robot.close_gripper") == 1 and names[-3:][0] in (
        "robot.open_gripper",
        "robot.get_ee_pose",
    )
    assert arm.gap == pytest.approx(0.094)
    # Carried with its bottom 40 mm over the rim, then down to the release.
    carry = [m for m in arm.moves if m[0] == "robot.go_to_pose"][-1][1]
    hang = 0.10 + 0.0125 + 0.0  # grasp tool z (0.45 x 25 mm tips + reach) minus the part's bottom
    assert carry[2] > 0.14 + 0.04
    assert hang > 0


def test_a_failed_grasp_is_grasp_failed_and_counts_stay(sort):
    arm = Arm(between=lambda arm: 0.0)
    out = sort.run(
        arm.ctx(), part_obb=PART_OBB, sort_bin=BIN, bin_counts={"red": 1}, support_z=0.10
    )
    assert out["route"] == "grasp_failed" and out["grasp_route"] == "missed"
    assert out["bin_counts"] == {"red": 1}
    assert not any(m[1][1] > 0.2 for m in arm.moves), "never carried toward the bin"


def test_no_part_box_or_no_bin(sort):
    arm = sort_arm()
    assert sort.run(arm.ctx(), part_obb={}, sort_bin=BIN)["route"] == "grasp_failed"
    assert sort.run(arm.ctx(), part_obb=PART_OBB, sort_bin={})["route"] == "drop_failed"
    assert arm.moves == []


def test_a_part_wider_than_the_bin_is_drop_failed_still_holding(sort):
    arm = sort_arm()
    small = {**BIN, "opening_half": [0.015, 0.015]}
    out = sort.run(arm.ctx(), part_obb=PART_OBB, sort_bin=small, support_z=0.10)
    assert out["route"] == "drop_failed" and "still holding" in out["detail"]
    assert arm.gap < 0.094


def test_an_unreachable_bin_is_drop_failed(sort):
    arm = sort_arm(ik_ok=lambda p, rot: p[1] < 0.2)
    out = sort.run(arm.ctx(), part_obb=PART_OBB, sort_bin=BIN, support_z=0.10)
    assert out["route"] == "drop_failed"
    assert "IK preview" in out["detail"] and out["place_route"] == "unreachable"


def test_a_refused_carry_tries_the_next_drop_point(sort):
    refused = {"n": 0}

    def refuse(tool, target):
        if tool == "robot.go_to_pose" and target[1] > 0.2 and refused["n"] == 0:
            refused["n"] += 1
            return "joint limit"
        return None

    arm = sort_arm(refuse=refuse)
    out = sort.run(arm.ctx(), part_obb=PART_OBB, sort_bin=BIN, support_z=0.10)
    assert out["route"] == "sorted", out["detail"]
    assert "centre: over the bin" in out["detail"]


def test_jaws_that_stay_closed_are_drop_failed(sort):
    arm = sort_arm()
    ctx = arm.ctx(robot__open_gripper=lambda **kw: {"position": 0.2})
    out = sort.run(ctx, part_obb=PART_OBB, sort_bin=BIN, support_z=0.10)
    assert out["route"] == "drop_failed" and out["place_route"] == "not_released"


def test_a_described_bin_goes_through_placing_into_container_obb(sort, monkeypatch):
    seen = {}

    class Place:
        footprint_reach = staticmethod(lambda *a: 0.0)

        @staticmethod
        def run(ctx, **kw):
            seen.update(kw)
            return {
                "route": "placed",
                "detail": "ok",
                "release_position": {"x": 1.0, "y": 2.0, "z": 3.0},
            }

    monkeypatch.setitem(sort._MODULES, "placing-into-container-obb", Place)
    arm = sort_arm()
    described = {"name": "tray", "description": "blue tray", "pixel": [100, 200]}
    out = sort.run(arm.ctx(), part_obb=PART_OBB, sort_bin=described, support_z=0.10)
    assert out["route"] == "sorted" and out["bin_counts"] == {"tray": 1}
    assert seen["container_description"] == "blue tray" and seen["container_pixel"] == [100, 200]
    held = seen["held_obb"]["center"]
    assert held["z"] > PART_OBB["center"]["z"] + 0.04, "the held box rose with the lift"

    Place.run = staticmethod(
        lambda ctx, **kw: {"route": "container_not_found", "detail": "no tray"}
    )
    out = sort.run(arm.ctx(), part_obb=PART_OBB, sort_bin=described, support_z=0.10)
    assert out["route"] == "drop_failed" and out["bin_counts"] == {}
    Place.run = staticmethod(lambda ctx, **kw: {"route": "not_holding", "detail": ""})
    out = sort.run(arm.ctx(), part_obb=PART_OBB, sort_bin=described, support_z=0.10)
    assert out["route"] == "grasp_failed"


def test_the_example_declares_every_route(skills_registry, plan, sort):
    ex = json.loads(
        (
            Path(skills_registry.get(BUNDLE).bundle_dir) / "examples" / "canonical_subgraph.json"
        ).read_text()
    )
    declared = set(ex["exit"]["success_values"]) | {ex["on_error"]}
    plan_routes = set(ex["conditional_edges"]["plan"]["mapping"])
    assert plan_routes == {
        "routed",
        "unknown_part",
        "bin_full",
        "staging_empty",
        "staging_not_clear",
    }
    assert (plan_routes - {"routed"}) | {"sorted", "grasp_failed", "drop_failed"} <= declared
