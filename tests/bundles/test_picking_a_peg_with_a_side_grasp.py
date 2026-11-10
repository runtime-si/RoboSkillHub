"""picking-a-peg-with-a-side-grasp: FakeContext tests of the grasp, the turn upright, the hold checks and the exits."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "picking-a-peg-with-a-side-grasp"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
Z = np.array([0.0, 0.0, 1.0])
SPAN = 0.094
WIDE = [[-9.0] * 5 + [-2.0944], [9.0] * 5 + [2.0944]]
PEG = {"length_m": 0.10, "diameter_m": 0.010, "axis_in_object": [0, 0, 1], "tip_end": "-z"}
MEET = {"x": 0.45, "y": 0.0, "z": 0.20}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["pick_peg"].module


def _rot(axis, deg):
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    a = math.radians(deg)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * kx + (1 - math.cos(a)) * kx @ kx


def _quat(m):
    w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    if w > 1e-6:
        return {
            "w": w,
            "x": (m[2, 1] - m[1, 2]) / (4 * w),
            "y": (m[0, 2] - m[2, 0]) / (4 * w),
            "z": (m[1, 0] - m[0, 1]) / (4 * w),
        }
    x = math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2
    if x > 1e-6:
        return {
            "w": 0.0,
            "x": x,
            "y": (m[0, 1] + m[1, 0]) / (4 * x),
            "z": (m[0, 2] + m[2, 0]) / (4 * x),
        }
    y = math.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2
    if y > 1e-6:
        return {"w": 0.0, "x": 0.0, "y": y, "z": (m[1, 2] + m[2, 1]) / (4 * y)}
    return {"w": 0.0, "x": 0.0, "y": 0.0, "z": 1.0}


def _mat(q):
    w, x, y, z = (q[k] for k in "wxyz")
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _p(v):
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def peg_pose(center, axis):
    """A pose whose z is ``axis``."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(ref, a)
    x /= np.linalg.norm(x)
    return {"position": _p(center), "rotation": _quat(np.stack([x, np.cross(a, x), a], axis=1))}


class Station:
    """Two arms. Joints are [x, y, z, k/100, 0, roll]: k indexes a stored base rotation, the last joint rolls the
    hand about its approach (tool +z), so the skill's roll setting and joint-6 limit are exercised. The peg rides
    rigidly in the left hand from the close."""

    def __init__(
        self, peg=PEG, center=(0.40, 0.25, 0.005), axis=(1.0, 0.0, 0.0), roll_free_deg=0.0
    ):
        self.bases = [np.eye(3)]
        self.state = {
            0: [np.array([0.30, 0.25, 0.20]), _rot([1, 0, 0], 180)],
            1: [np.array([0.30, -0.25, 0.20]), _rot([1, 0, 0], 180)],
        }
        self.q = {a: self._q(*self.state[a]) for a in (0, 1)}
        self.peg = peg
        self.peg_c, self.peg_r = np.array(center, float), _mat(peg_pose(center, axis)["rotation"])
        self.frac = 1.0
        self.close_width = peg["diameter_m"] + 0.0015
        self.held = None
        self.roll_free = roll_free_deg
        self.refuse_ik = lambda arm, p, r: False
        self.raise_move = False
        self.after_close = []  # widths read after the close, one per get_gripper
        self.extra_tilt = 0.0
        self.moves = []

    def _q(self, p, r, roll=0.0):
        self.bases.append(np.asarray(r) @ _rot(Z, -math.degrees(roll)))
        return np.array([p[0], p[1], p[2], (len(self.bases) - 1) / 100.0, 0.0, roll])

    def _fk(self, q):
        base = self.bases[int(round(q[3] * 100))]
        return np.array(q[:3]), base @ _rot(Z, math.degrees(q[5]))

    def _set(self, arm, p, r):
        self.state[arm] = [np.asarray(p, float), np.asarray(r, float)]

    def peg_now(self):
        if self.held is None:
            return self.peg_c, self.peg_r
        p, r = self.state[0]
        t, rr = self.held
        tilt = _rot(r @ np.array([1.0, 0, 0]), self.extra_tilt)
        return p + r @ t, tilt @ r @ rr

    def ctx(self, **inputs):
        def pose(arm_id=0):
            p, r = self.state[arm_id]
            return {"pose": {"position": _p(p), "rotation": _quat(r)}}

        def obs():
            return {
                "arms": [
                    {"joint_state": {"positions": list(map(float, self.q[a]))}} for a in (0, 1)
                ]
            }

        def ik(pose, arm_id=0):
            p, r = np.array([pose["position"][k] for k in "xyz"]), _mat(pose["rotation"])
            if self.refuse_ik(arm_id, p, r):
                return {"success": False}
            q = self._q(p, r @ _rot(Z, self.roll_free))
            return {"success": True, "joint_config": {"positions": list(map(float, q))}}

        def fk(joint_configs, arm_id=0):
            return {
                "poses": [
                    dict(zip(("position", "rotation"), (_p(p), _quat(r)), strict=True))
                    for p, r in (self._fk(q) for q in joint_configs)
                ]
            }

        def to_q(joint_config, arm_id=0, **kw):
            if self.raise_move:
                raise ToolError("robot.move_to_joints", "unreachable")
            q = np.array(joint_config["positions"], float)
            self.q[arm_id] = q
            self._set(arm_id, *self._fk(q))
            self.moves.append(("joints", arm_id))
            return {"reached": True}

        def go(pose, arm_id=0, **kw):
            p, r = np.array([pose["position"][k] for k in "xyz"]), _mat(pose["rotation"])
            q = self._q(p, r @ _rot(Z, self.roll_free))
            self.q[arm_id] = q
            self._set(arm_id, *self._fk(q))
            self.moves.append(("go", arm_id))
            return {"reached": True, "outcome": "reached"}

        def frame(close_heading_deg=0.0, approach=None, arm_id=0):
            a = np.array([approach[k] for k in "xyz"]) if approach else -Z
            a = a / np.linalg.norm(a)
            h = math.radians(close_heading_deg)
            c = np.array([math.cos(h), math.sin(h), 0.0])
            c = c - (c @ a) * a
            c /= np.linalg.norm(c)

            def r(cc):
                return _quat(np.stack([cc, np.cross(a, cc), a], axis=1))

            return {
                "rotation": r(c),
                "rotation_flipped": r(-c),
                "approach": _p(a),
                "close_axis": _p(c),
            }

        def close(arm_id=0):
            p, r = self.state[0]
            pc, pr = self.peg_c, self.peg_r
            axis_dist = np.linalg.norm(np.cross(p - pc, pr @ Z))
            if abs((p - pc) @ (pr @ Z)) < self.peg["length_m"] / 2 and axis_dist < 0.012:
                self.held = (r.T @ (pc - p), r.T @ pr)
                self.frac = self.close_width / SPAN
            else:
                self.frac = 0.0
            return {"position": self.frac}

        def opn(arm_id=0):
            self.held, self.frac = None, 1.0
            return {"position": 1.0}

        def grip(arm_id=0):
            if self.held is not None and self.after_close:
                self.frac = self.after_close.pop(0) / SPAN
            return {"position": self.frac}

        def obb(object_name):
            c, r = self.peg_now()
            return {
                "obb": {
                    "center": _p(c),
                    "orientation": _quat(r),
                    "extent": _p([0.005, 0.005, 0.05]),
                }
            }

        tools = {
            "robot.get_ee_pose": pose,
            "robot.get_observation": obs,
            "robot.solve_ik": ik,
            "robot.forward_kinematics": fk,
            "robot.move_to_joints": to_q,
            "robot.go_to_pose": go,
            "robot.grasp_frame": frame,
            "robot.close_gripper": close,
            "robot.open_gripper": opn,
            "robot.get_gripper": grip,
            "robot.describe_gripper": {
                "width_fit": {"at_open_m": SPAN, "at_closed_m": 0.0},
                "finger": {"stated": True, "reach_m": 0.0102, "clearance_m": 0.004},
            },
            "robot.describe_arm": {"joint_limits": {"lower": WIDE[0], "upper": WIDE[1]}},
            "sim.get_object_obb": obb,
        }
        return FakeContext(tools)


def _run(script, st, **kw):
    args = {
        "peg": st.peg,
        "peg_pose": {"position": _p(st.peg_c), "rotation": _quat(st.peg_r)},
        "table_z": 0.0,
        "handover_point": MEET,
    }
    args.update(kw)
    return script.run(st.ctx(), **args)


def _free_dir(out, st):
    tcp = out["left_grip"]["T_base_tcp"]
    obj = out["left_grip"]["T_TCP_object"]
    r = _mat(tcp["rotation"]) @ _mat(obj["rotation"])
    return r @ np.array([0.0, 0.0, 1.0 if st.peg["tip_end"] == "-z" else -1.0])


def test_lying_peg_ends_vertical_free_end_up(script):
    st = Station()
    out = _run(script, st, object_name="peg")
    assert out["route"] == "holding_vertical", out["detail"]
    assert out["posture"] == "lying" and out["picked"]
    assert out["axis_tilt_deg"] < 1.0 and out["tilt_source"] == "perceived"
    assert _free_dir(out, st) @ Z > 0.99
    assert out["left_grip"]["exposed_len_m"] == pytest.approx(0.05 - 0.0155 / 2, abs=1e-3)
    p, r = st.state[0]
    assert np.linalg.norm(p - np.array([MEET[k] for k in "xyz"])) < 1e-6
    assert abs(r @ Z @ Z) < 0.02  # approach level after the pitch
    assert out["free_end"]["z"] > MEET["z"] + 0.04
    assert set(out["left_grip"]) == {"T_base_tcp", "T_TCP_object", "exposed_len_m"}
    json.dumps(out)


def test_lying_peg_grasp_is_top_down_at_the_floor_across_the_shaft(script):
    st = Station()
    seen = []
    orig = st.ctx

    def ctx(**inputs):
        c = orig(**inputs)
        close = c._responses["robot.close_gripper"]

        def spy(arm_id=0):
            seen.append([x.copy() for x in st.state[0]])
            return close(arm_id)

        c._responses["robot.close_gripper"] = spy
        return c

    st.ctx = ctx
    out = _run(script, st)
    assert out["route"] == "holding_vertical", out["detail"]
    p, r = seen[0]
    assert p[2] == pytest.approx(0.014, abs=1e-6)
    assert r @ Z @ -Z > 0.999
    assert (
        abs((r @ np.array([1.0, 0, 0])) @ np.array([1.0, 0, 0])) < 1e-6
    )  # jaws close across the x-lying shaft
    assert p[0] == pytest.approx(0.40, abs=1e-6)  # gripped at the middle


def test_standing_peg_side_grasp(script):
    st = Station(center=(0.40, 0.25, 0.05), axis=(0, 0, 1))  # tip (-z) down, free end up
    out = _run(script, st, object_name="peg")
    assert out["route"] == "holding_vertical", out["detail"]
    assert out["posture"] == "standing"
    assert out["axis_tilt_deg"] < 1.0
    assert _free_dir(out, st) @ Z > 0.99


def test_tip_end_decides_which_end_goes_up(script):
    peg = {**PEG, "tip_end": "+z"}
    st = Station(peg=peg)
    out = _run(script, st, object_name="peg")
    assert out["route"] == "holding_vertical", out["detail"]
    assert _free_dir(out, st) @ Z > 0.99
    tip_world = (
        _mat(out["left_grip"]["T_base_tcp"]["rotation"])
        @ _mat(out["left_grip"]["T_TCP_object"]["rotation"])
        @ Z
    )
    assert tip_world @ Z < -0.99


def test_upside_down_standing_peg_is_refused(script):
    st = Station(center=(0.40, 0.25, 0.05), axis=(0, 0, -1))  # tip (-z) up
    out = _run(script, st)
    assert out["route"] == "upside_down"
    assert not st.moves


def test_leaning_peg_is_refused(script):
    out = _run(script, Station(axis=(1.0, 0.0, 1.0)))
    assert out["route"] == "ungraspable" and "neither" in out["detail"]


def test_short_peg_grip_moves_toward_the_far_end(script):
    st = Station(peg={**PEG, "length_m": 0.06})
    out = _run(script, st, object_name="peg")
    assert out["route"] == "holding_vertical", out["detail"]
    assert out["grip_from_free_end_m"] == pytest.approx(0.025 + 0.003 + 0.0155 / 2)
    assert out["left_grip"]["exposed_len_m"] >= 0.025


def test_too_short_peg_is_ungraspable(script):
    st = Station(peg={**PEG, "length_m": 0.04})
    out = _run(script, st)
    assert out["route"] == "ungraspable" and not st.moves


def test_thin_lying_peg_needs_a_lower_floor(script):
    peg = {**PEG, "diameter_m": 0.008}
    out = _run(script, Station(peg=peg, center=(0.40, 0.25, 0.004)))
    assert out["route"] == "ungraspable" and "equator" in out["detail"]
    out = _run(script, Station(peg=peg, center=(0.40, 0.25, 0.004)), min_tool_height=0.012)
    assert out["route"] == "holding_vertical", out["detail"]


def test_too_wide_peg_is_ungraspable(script):
    out = _run(script, Station(peg={**PEG, "diameter_m": 0.09}, center=(0.4, 0.25, 0.045)))
    assert out["route"] == "ungraspable"


def test_jaws_closing_on_nothing_is_grasp_failed(script):
    st = Station()
    st.peg_c = st.peg_c + np.array([0.0, 0.03, 0.0])  # the peg is not where it was seen
    out = _run(
        script, st, peg_pose={"position": _p([0.40, 0.25, 0.005]), "rotation": _quat(st.peg_r)}
    )
    assert out["route"] == "grasp_failed" and "nothing" in out["detail"]
    assert st.frac == 1.0  # opened again


def test_jaws_emptying_after_the_lift_is_dropped(script):
    st = Station()
    st.after_close = [0.0]
    out = _run(script, st)
    assert out["route"] == "dropped" and "after the lift" in out["detail"]


def test_perceived_tilt_over_threshold_is_slipped(script):
    st = Station()
    st.extra_tilt = 15.0
    out = _run(script, st, object_name="peg")
    assert out["route"] == "slipped"
    assert out["axis_tilt_deg"] == pytest.approx(15.0, abs=0.5)
    assert out["left_grip"]["T_TCP_object"]


def test_hand_only_tilt_misses_a_pivot_in_the_pads(script):
    st = Station()
    st.extra_tilt = 15.0
    out = _run(script, st)
    assert out["route"] == "holding_vertical" and out["tilt_source"] == "hand"


def test_roll_free_solver_is_corrected_on_the_last_joint(script):
    st = Station(roll_free_deg=35.0)
    out = _run(script, st, object_name="peg")
    assert out["route"] == "holding_vertical", out["detail"]
    assert out["axis_tilt_deg"] < 1.0


def test_no_ik_anywhere_is_unreachable_before_moving(script):
    st = Station()
    st.refuse_ik = lambda arm, p, r: True
    out = _run(script, st)
    assert out["route"] == "unreachable" and "nothing moved" in out["detail"]
    assert not st.moves


def test_receiver_out_of_reach_is_unreachable(script):
    st = Station()
    st.refuse_ik = lambda arm, p, r: arm == 1
    out = _run(script, st)
    assert out["route"] == "unreachable" and not st.moves


def test_receiver_reading_the_same_pose_is_refused(script):
    st = Station()
    st.state[1] = [x.copy() for x in st.state[0]]
    out = _run(script, st)
    assert out["route"] == "unreachable" and "same pose" in out["detail"]


def test_raised_move_is_routed(script):
    st = Station()
    st.raise_move = True
    out = _run(script, st)
    assert out["route"] == "unreachable"


def test_joint_six_limit_refuses_the_turn(script):
    st = Station(roll_free_deg=170.0)
    out = _run(script, st, joint_limits=[[-9.0] * 5 + [-0.5], [9.0] * 5 + [0.5]])
    assert out["route"] == "unreachable"
    assert "joint 6" in out["detail"] or "nothing moved" in out["detail"]


def test_choose_grip_and_floor_helpers(script):
    assert script.choose_grip(0.10, 0.025, 0.0155) == pytest.approx(0.05)
    assert script.choose_grip(0.04, 0.025, 0.0155) == -1.0
    z, why = script.topdown_tool_height(0.006, 0.0, 0.0102, 0.014)
    assert z == pytest.approx(0.014) and not why


def test_example_declares_every_route(script):
    example = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    src = (SKILL_DIR / "scripts" / "pick_peg.py").read_text()
    routes = set(re.findall(r'finish\(\s*"([a-z_]+)"', src)) | {"unreachable"}
    assert routes <= declared, routes - declared
    skill_md = (SKILL_DIR / "SKILL.md").read_text()
    for route in routes:
        assert f"    {route}:" in skill_md
