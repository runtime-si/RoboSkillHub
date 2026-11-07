"""executing-keypoint-motion: FakeContext tests of the MOKA planner (marks, parsing, 3D) and the executor's phases."""

from __future__ import annotations

import json

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "executing-keypoint-motion"
W, H, F = 320, 240, 500.0
CAM_Z = 1.0
CUBE = (0.40, 0.00, 0.015)  # centre of a 30 mm cube on a table at z = 0
PAD = (0.40, 0.12)  # a 40 mm flat pad (5 mm thick) to place on


@pytest.fixture(scope="module")
def planner(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["plan_keypoint_motion"].module


@pytest.fixture(scope="module")
def executor(skills_registry):
    return skills_registry.get(BUNDLE).canonical_scripts["execute_keypoint_motion"].module


# ---------------------------------------------------------------------------
# A top-down RGB-D camera 1 m over the table at (0.4, 0): image right = +x, image up = +y.
# ---------------------------------------------------------------------------


def _uv(x, y, depth):
    return 160.0 + F * (x - 0.40) / depth, 120.0 - F * y / depth


def _scene(blocks):
    """Depth + RGB + masks for axis-aligned blocks [(cx, cy, half_xy, height)]."""
    depth = np.full((H, W), CAM_Z, dtype=np.float32)
    rgb = np.full((H, W, 3), 180, dtype=np.uint8)
    masks = []
    for cx, cy, half, height in blocks:
        d = CAM_Z - height
        u0, v1 = _uv(cx - half, cy - half, d)
        u1, v0 = _uv(cx + half, cy + half, d)
        m = np.zeros((H, W), dtype=bool)
        m[int(round(v0)) : int(round(v1)) + 1, int(round(u0)) : int(round(u1)) + 1] = True
        depth[m] = d
        rgb[m] = (60, 60, 200)
        masks.append(m.astype(np.uint8))
    camera = {
        "name": "top",
        "rgb": rgb,
        "depth": depth,
        "intrinsics": np.array([[F, 0, 160.0], [0, F, 120.0], [0, 0, 1]]),
        "pose": np.array(
            [[1, 0, 0, 0.40], [0, -1, 0, 0.0], [0, 0, -1, CAM_Z], [0, 0, 0, 1]], dtype=float
        ),
    }
    return {"cameras": [camera], "arms": []}, masks


def _motion(**over):
    reply = {
        "grasp_keypoint": "",
        "function_keypoint": "",
        "target_keypoint": "Q1",
        "target_tile": "c3",
        "pre_contact_tile": "b3",
        "post_contact_tile": "d3",
        "pre_contact_height": "same",
        "post_contact_height": "same",
        "target_angle": "right",
    }
    reply.update(over)
    return {"text": "Reasoning first.\n```json\n" + json.dumps(reply) + "\n```"}


class Vlm:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []
        self.images = []

    def __call__(self, prompt, image=None, **kw):
        self.prompts.append(prompt)
        self.images.append(image)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def test_push_plan_geometry(planner):
    obs, (cube,) = _scene([(CUBE[0], CUBE[1], 0.015, 0.03)])
    vlm = Vlm(_motion())
    ctx = FakeContext({"vlm.query": vlm})
    out = planner.run(
        ctx,
        observation=obs,
        instruction="push the cube to the right",
        object_grasped="",
        object_unattached="cube",
        unattached_mask=cube,
    )
    assert out["route"] == "planned", out["detail"]
    plan = out["plan"]
    assert plan["mode"] == "push"
    assert plan["support_z"] == pytest.approx(0.0, abs=1e-6)
    assert plan["contact_z"] == pytest.approx(0.015, abs=1e-3)  # the cube's mid height
    pre, target, post = (
        np.array([plan[k][c] for c in "xyz"]) for k in ("pre_contact", "target", "post_contact")
    )
    assert pre[0] < target[0] < post[0]  # left to right in the image is +x
    assert target[:2] == pytest.approx(CUBE[:2], abs=0.003)  # Q1 is the mask's centre
    assert post[0] - pre[0] == pytest.approx(2 * plan["tile_m"], rel=0.05)
    assert plan["tile_m"] == pytest.approx(0.05, rel=0.1)
    assert plan["push_heading_deg"] == pytest.approx(0.0, abs=10.0)
    assert plan["grasp"] is None and plan["function"] is None
    assert out["marked_image"].shape == (768, 768, 3)
    assert "'a', 'b', 'c', 'd', 'e'" in vlm.prompts[0]
    assert '"object_unattached": "cube"' in vlm.prompts[0]


def test_task_parse_and_sam3_candidates(planner):
    obs, (cube,) = _scene([(CUBE[0], CUBE[1], 0.015, 0.03)])
    task = {
        "text": '```json\n{"object_grasped": "", "object_unattached": "cube", "motion_direction": "right"}\n```'
    }
    vlm = Vlm(task, _motion())
    seen = []

    def sam3(image, query, **kw):
        seen.append(query)
        return {"masks": [cube], "scores": [0.9]}

    out = planner.run(
        FakeContext({"vlm.query": vlm, "sam3.segment_text": sam3}),
        observation=obs,
        instruction="push it",
    )
    assert out["route"] == "planned", out["detail"]
    assert seen == ["cube"]
    assert out["plan"]["candidate_sources"] == {"unattached": "sam3"}
    assert len(vlm.prompts) == 2 and "one-step" in vlm.prompts[0]


def test_gemini_er_fallback_when_sam3_is_not_served(planner):
    obs, _ = _scene([(CUBE[0], CUBE[1], 0.015, 0.03)])
    u, v = _uv(CUBE[0], CUBE[1], CAM_Z - 0.03)

    def missing(**kw):
        raise ToolError("sam3.segment_text", "not served")

    points = {
        "found": True,
        "points": [{"pixel_x": u + dx, "pixel_y": v + dy} for dx, dy in ((0, 0), (5, 5), (-5, 5))],
    }
    ctx = FakeContext(
        {"vlm.query": Vlm(_motion()), "sam3.segment_text": missing, "gemini-er.point": points}
    )
    out = planner.run(
        ctx, observation=obs, instruction="push", object_grasped="", object_unattached="cube"
    )
    assert out["route"] == "planned", out["detail"]
    assert out["plan"]["candidate_sources"] == {"unattached": "gemini-er"}


def test_object_not_found_is_no_plan(planner):
    obs, _ = _scene([(CUBE[0], CUBE[1], 0.015, 0.03)])
    ctx = FakeContext(
        {"vlm.query": Vlm(_motion()), "sam3.segment_text": {"masks": [], "scores": []}}
    )
    out = planner.run(
        ctx, observation=obs, instruction="push the box", object_grasped="", object_unattached="box"
    )
    assert out["route"] == "no_plan"
    assert "box" in out["detail"]


def test_unoffered_marks_are_asked_again_then_no_plan(planner):
    obs, (cube,) = _scene([(CUBE[0], CUBE[1], 0.015, 0.03)])
    vlm = Vlm(_motion(target_keypoint="Q9"))
    out = planner.run(
        FakeContext({"vlm.query": vlm}),
        observation=obs,
        instruction="push",
        object_grasped="",
        object_unattached="cube",
        unattached_mask=cube,
    )
    assert out["route"] == "no_plan"
    assert "Q9" in out["detail"]
    assert len(vlm.prompts) == 3

    vlm = Vlm({"text": "no json here"}, _motion(post_contact_tile="z9"), _motion())
    out = planner.run(
        FakeContext({"vlm.query": vlm}),
        observation=obs,
        instruction="push",
        object_grasped="",
        object_unattached="cube",
        unattached_mask=cube,
    )
    assert out["route"] == "planned"
    assert len(vlm.prompts) == 3


def test_missing_grasp_for_a_grasped_object_is_rejected(planner):
    obs, (cube, pad) = _scene([(CUBE[0], CUBE[1], 0.015, 0.03), (PAD[0], PAD[1], 0.02, 0.005)])
    out = planner.run(
        FakeContext({"vlm.query": Vlm(_motion(grasp_keypoint=""))}),
        observation=obs,
        instruction="put the cube on the pad",
        object_grasped="cube",
        object_unattached="pad",
        grasped_mask=cube,
        unattached_mask=pad,
        attempts=1,
    )
    assert out["route"] == "no_plan"
    assert "grasp" in out["detail"]


def test_pick_and_place_heights_and_grasp_heading(planner):
    obs, (bar, pad) = _scene([(CUBE[0], CUBE[1], 0.015, 0.03), (PAD[0], PAD[1], 0.02, 0.005)])
    # make the grasped object a bar long in y so the hand must close across x
    bar = np.zeros_like(bar)
    u0, v1 = _uv(0.39, -0.04, CAM_Z - 0.03)
    u1, v0 = _uv(0.41, 0.04, CAM_Z - 0.03)
    bar[int(v0) : int(v1) + 1, int(u0) : int(u1) + 1] = 1
    obs["cameras"][0]["depth"][bar > 0] = CAM_Z - 0.03
    reply = _motion(
        grasp_keypoint="P1",
        target_keypoint="Q1",
        pre_contact_height="above",
        post_contact_height="above",
    )
    out = planner.run(
        FakeContext({"vlm.query": Vlm(reply)}),
        observation=obs,
        instruction="put the bar on the pad",
        object_grasped="bar",
        object_unattached="pad",
        grasped_mask=bar,
        unattached_mask=pad,
    )
    assert out["route"] == "planned", out["detail"]
    plan = out["plan"]
    assert plan["mode"] == "pick_and_place"
    # pad top 5 mm + grasp point 30 mm over the support + 10 mm offset
    assert plan["contact_z"] == pytest.approx(0.005 + 0.03 + 0.01, abs=2e-3)
    assert (
        abs(abs(plan["grasp_heading_deg"]) - 0.0) < 15
        or abs(abs(plan["grasp_heading_deg"]) - 180) < 15
    )
    assert plan["pre_contact"]["z"] == pytest.approx(plan["safe_z"])


def test_parse_and_tile_helpers(planner):
    text = (
        'thinking\n```json\n{\n "grasp_keypoint": "P2", // the handle\n "target_tile": "a1"\n}\n```'
    )
    assert planner._parse_json(text) == {"grasp_keypoint": "P2", "target_tile": "a1"}
    assert planner._parse_json('[{"a": 1}]') == {"a": 1}
    assert planner._parse_json("nothing") is None
    assert planner._tile("a1", 5) == (0, 4)
    assert planner._tile("e5", 5) == (4, 0)
    assert planner._tile("", 5) is None
    with pytest.raises(ValueError):
        planner._tile("f1", 5)
    assert planner._label_index("P3", "P", 6) == 2
    assert planner._label_index("", "P", 6) is None
    with pytest.raises(ValueError):
        planner._label_index("Q1", "P", 6)


def test_support_ignores_the_floor_past_the_table(planner):
    heights = np.concatenate([np.full(500, -0.70), np.full(300, 0.0), np.full(200, 0.012)])
    assert planner._support_height(heights, 0.012) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
TOP_A = {"w": 0.0, "x": 0.7071, "y": 0.7071, "z": 0.0}
TOP_B = {"w": 0.0, "x": -0.7071, "y": 0.7071, "z": 0.0}


def _v(x, y, z):
    return {"x": x, "y": y, "z": z}


PUSH = {
    "mode": "push",
    "grasp": None,
    "function": None,
    "target": _v(0.385, 0.0, 0.015),
    "pre_contact": _v(0.33, 0.0, 0.015),
    "post_contact": _v(0.44, 0.0, 0.015),
    "support_z": 0.0,
    "safe_z": 0.12,
    "contact_z": 0.015,
    "grasp_heading_deg": None,
    "push_heading_deg": 0.0,
}


class Arm:
    def __init__(self, start=(0.30, 0.0, 0.05), refuse=None, short=None, jaw=0.4):
        self.p = np.array(start, dtype=float)
        self.rot = DOWN
        self.refuse = refuse or (lambda tool, target: None)
        self.short = short
        self.jaw = jaw
        self.log = []

    def _move(self, tool):
        def go(pose, **kw):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.log.append((tool, target.copy()))
            why = self.refuse(tool, target)
            if why == "raise":
                raise ToolError(tool, "IK failed")
            if why:
                return {"reached": False, "outcome": "unreachable", "reason": why}
            if self.short is not None and tool.endswith("cartesian") and self.short(target):
                target = self.p + 0.5 * (target - self.p)
            self.p, self.rot = target, pose["rotation"]
            return {"reached": True, "outcome": "reached"}

        return go

    def _grip(self, name):
        def f(**kw):
            self.log.append((name, None))
            return {}

        return f

    def ctx(self):
        return FakeContext(
            {
                "robot.get_ee_pose": lambda **kw: {
                    "pose": {
                        "position": dict(zip("xyz", map(float, self.p), strict=True)),
                        "rotation": self.rot,
                    }
                },
                "robot.go_to_pose": self._move("robot.go_to_pose"),
                "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
                "robot.close_gripper": self._grip("close"),
                "robot.open_gripper": self._grip("open"),
                "robot.get_gripper": lambda **kw: {"position": self.jaw},
                "robot.grasp_frame": {"rotation": TOP_A, "rotation_flipped": TOP_B},
            }
        )

    def moves(self):
        return [(t.split(".")[-1], p) for t, p in self.log if p is not None]

    def grips(self):
        return [t for t, p in self.log if p is None]


def test_push_runs_pre_target_post_with_the_closed_hand(executor):
    arm = Arm()
    out = executor.run(arm.ctx(), plan=PUSH)
    assert out["route"] == "done", out["detail"]
    names = [p["phase"] for p in out["phases"]]
    assert names == [
        "lift_start",
        "reach_pre_motion",
        "reach_pre_contact",
        "reach_target",
        "reach_post_contact",
        "release_lift",
        "retreat",
    ]
    assert arm.grips() == ["close", "open"]
    assert arm.log[0][0] == "robot.go_to_pose" and arm.log[0][1][2] == pytest.approx(0.12)
    assert arm.rot == TOP_A  # fingers across the push from grasp_frame
    pre = [p for p in out["phases"] if p["phase"] == "reach_pre_contact"][0]["commanded"]
    assert pre["z"] == pytest.approx(0.015)  # at the floor (support + 15 mm), not under it
    assert out["final_function_point"]["x"] == pytest.approx(0.44)
    assert "55 mm past the target" in out["detail"]


def test_contact_height_is_floored(executor):
    arm = Arm()
    plan = dict(
        PUSH,
        target=_v(0.385, 0.0, 0.004),
        pre_contact=_v(0.33, 0.0, 0.004),
        post_contact=_v(0.44, 0.0, 0.004),
    )
    out = executor.run(arm.ctx(), plan=plan)
    assert out["route"] == "done"
    assert min(p[2] for _, p in arm.moves()) == pytest.approx(0.015)


def test_plan_as_json_string(executor):
    out = executor.run(Arm().ctx(), plan=json.dumps(PUSH))
    assert out["route"] == "done"


TOOL = {
    "mode": "tool_push",
    "grasp": _v(0.30, 0.10, 0.02),
    "function": _v(0.30, 0.20, 0.01),
    "target": _v(0.40, 0.0, 0.015),
    "pre_contact": _v(0.35, 0.0, 0.015),
    "post_contact": _v(0.45, 0.0, 0.015),
    "support_z": 0.0,
    "safe_z": 0.12,
    "contact_z": 0.015,
    "grasp_heading_deg": 90.0,
    "push_heading_deg": 0.0,
}


def test_tool_motion_carries_the_function_point(executor):
    arm = Arm(start=(0.3, 0.0, 0.2))
    out = executor.run(arm.ctx(), plan=TOOL)
    assert out["route"] == "done", out["detail"]
    assert out["held"] is True
    assert arm.grips() == ["open", "close", "open"]
    phases = {p["phase"]: p["commanded"] for p in out["phases"]}
    # the hand grasped at (0.30, 0.10, 0.015 floor); offset to the function point = (0, -0.10, 0.005)
    assert phases["reach_grasp"]["z"] == pytest.approx(0.015)
    assert out["hand_offset"]["y"] == pytest.approx(-0.10)
    assert phases["reach_post_contact"]["x"] == pytest.approx(0.45)
    assert phases["reach_post_contact"]["y"] == pytest.approx(-0.10)
    assert out["final_function_point"]["y"] == pytest.approx(0.0)
    # holding: open first, then lift
    assert [t for t, p in arm.log][-3:-1] == ["open", "robot.go_to_pose_cartesian"]


def test_empty_jaws_are_grasp_failed(executor):
    arm = Arm(jaw=0.01)
    out = executor.run(arm.ctx(), plan=TOOL)
    assert out["route"] == "grasp_failed"
    assert arm.grips() == ["open", "close", "open"]
    assert not any(p["phase"].startswith("reach_pre_contact") for p in out["phases"])


def test_refused_hover_with_every_rotation_is_unreachable(executor):
    arm = Arm(
        start=(0.3, 0.0, 0.2),
        refuse=lambda tool, t: "no IK" if t[2] > 0.1 and t[0] < 0.34 else None,
    )
    out = executor.run(arm.ctx(), plan=PUSH)
    assert out["route"] == "unreachable"
    assert out["detail"].startswith("reach_pre_motion")
    assert sum(p["phase"] == "reach_pre_motion" for p in out["phases"]) == 3  # TOP_A, TOP_B, own
    assert not any(t.endswith("cartesian") for t, _ in arm.moves())


def test_raised_refusal_is_routed(executor):
    arm = Arm(
        start=(0.3, 0.0, 0.2),
        refuse=lambda tool, t: "raise" if tool.endswith("cartesian") else None,
    )
    out = executor.run(arm.ctx(), plan=TOOL)
    assert out["route"] == "unreachable"
    assert "reach_grasp" in out["detail"] and "IK failed" in out["detail"]


def test_contact_shortfall_releases_then_reports(executor):
    arm = Arm(short=lambda t: t[0] > 0.43 and t[2] < 0.05)
    out = executor.run(arm.ctx(), plan=PUSH)
    assert out["route"] == "unreachable"
    assert out["detail"].startswith("reach_post_contact")
    assert "released and lifted" in out["detail"]
    assert arm.grips() == ["close", "open"]


def test_pick_and_place_goes_over_and_down_to_the_target(executor):
    plan = dict(
        TOOL, mode="pick_and_place", function=None, target=_v(0.40, 0.12, 0.045), pre_contact=None
    )
    arm = Arm(start=(0.3, 0.0, 0.2))
    out = executor.run(arm.ctx(), plan=plan)
    assert out["route"] == "done", out["detail"]
    names = [p["phase"] for p in out["phases"]]
    assert names[:5] == [
        "reach_pre_grasp",
        "reach_grasp",
        "lift",
        "reach_pre_target",
        "reach_target",
    ]
    target = [p for p in out["phases"] if p["phase"] == "reach_target"][0]["commanded"]
    # hand at the target + (hand - grasp point) measured at the grasp: 0.015 - 0.02 = -0.005
    assert target["z"] == pytest.approx(0.040)
