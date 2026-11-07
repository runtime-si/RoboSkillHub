"""checking-dual-arm-clearance: FakeContext tests of the reading, the model check, the path sampling and the routing."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "checking-dual-arm-clearance"

# Configurations recorded every 20 ms during handing-over-between-arms runs in the
# YAM station twin (bases 0.61 m apart), with the harness's measured mesh clearance.
FAR = {  # 217.7 mm measured
    "q0": [-0.76279, 2.04594, 2.12375, -1.57081, -0.00753, 0.0004],
    "q1": [-0.05884, 0.6043, 0.43901, -0.10849, -0.17114, 0.00063],
    "f0": 0.01587 / 0.0475,
    "f1": 0.04705 / 0.0475,
    "measured": 0.21772,
}
NEAR = {  # 10.05 mm measured, fingertip to fingertip
    "q0": [-0.76279, 2.04594, 2.12375, -1.57081, -0.00753, 0.0004],
    "q1": [0.45644, 1.89266, 0.76254, 1.12768, -1.06317, 0.00065],
    "f0": 0.01588 / 0.0475,
    "f1": 0.04676 / 0.0475,
    "measured": 0.01005,
}
TOUCH = {  # 0.48 mm measured (the receiver closing beside the donor's fingers)
    "q0": [-0.76047, 2.04618, 2.12421, -1.5703, -0.00839, 0.00024],
    "q1": [0.4808, 1.90241, 0.77606, 1.12352, -1.07407, 0.00022],
    "f0": 0.01742 / 0.0475,
    "f1": 0.02405 / 0.0475,
    "measured": 0.00048,
}
TCP = {
    "tcp_offset": {"x": 0.0, "y": 0.0, "z": -0.1347},
    "tcp_rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0},
}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["check_clearance"].module


def _base(y: float, yaw_deg: float = 0.0) -> np.ndarray:
    t = np.eye(4)
    h = math.radians(yaw_deg)
    t[:3, :3] = [[math.cos(h), -math.sin(h), 0], [math.sin(h), math.cos(h), 0], [0, 0, 1]]
    t[:3, 3] = [0.2525, y, 0.773]
    return t


class Station:
    """Two YAM arms whose readings follow the skill's own model from known bases."""

    def __init__(
        self, script, config, bases=None, arms=2, fk=True, describe=True, ik=None, fk_scale=1.0
    ):
        self.s = script
        self.q = {0: np.array(config["q0"], float), 1: np.array(config["q1"], float)}
        self.grip = {0: config["f0"], 1: config["f1"]}
        self.bases = bases or {0: _base(0.305), 1: _base(-0.305)}
        self.tcp = script._frame((0.0, 0.0, -0.1347), (0.0, 1.0, 0.0, 0.0))
        self.n_arms, self.fk, self.describe, self.fk_scale = arms, fk, describe, fk_scale
        self.ik = ik
        self.calls: list[tuple] = []

    def tool_pose(self, arm_id, q=None):
        q = self.q[arm_id] if q is None else np.asarray(q, float)
        return self.bases[arm_id] @ self.s.forward_kinematics(q)["gripper"] @ self.tcp

    def ctx(self):
        def observation():
            return {
                "arms": [
                    {
                        "joint_state": {"positions": self.q[a].copy()},
                        "gripper_fraction": self.grip[a],
                        "ee_pose": self.s._pose_dict(self.tool_pose(a)),
                    }
                    for a in range(self.n_arms)
                ]
            }

        def forward_kinematics(joint_configs, arm_id=0):
            self.calls.append(("fk", arm_id))
            if not self.fk:
                raise ToolError("robot.forward_kinematics", "not served")
            poses = []
            for q in joint_configs:
                t = self.tool_pose(arm_id, q)
                t[:3, 3] *= self.fk_scale
                poses.append(self.s._pose_dict(t))
            return {"joint_config": self.q[arm_id].tolist(), "poses": poses}

        def describe_arm(arm_id=0):
            if not self.describe:
                raise ToolError("robot.describe_arm", "not served")
            return dict(TCP, arm_id=arm_id)

        def solve_ik(pose, arm_id=0, **kw):
            self.calls.append(("ik", arm_id))
            if self.ik is not None:
                return self.ik(self, pose, arm_id)
            return {
                "joint_config": {"positions": _dls(self, pose, arm_id).tolist()},
                "success": True,
            }

        return FakeContext(
            {
                "robot.get_observation": observation,
                "robot.forward_kinematics": forward_kinematics,
                "robot.describe_arm": describe_arm,
                "robot.solve_ik": solve_ik,
            }
        )


def _dls(st: Station, pose: dict, arm_id: int) -> np.ndarray:
    """Damped least-squares position + approach-axis IK on the model, seeded from the last solve."""
    target = st.s._pose_matrix(pose)
    q = getattr(st, "_seed", {}).get(arm_id, st.q[arm_id]).copy()

    def residual(q):
        t = st.tool_pose(arm_id, q)
        return np.r_[t[:3, 3] - target[:3, 3], 0.05 * (t[:3, :3] - target[:3, :3]).ravel()]

    for _ in range(60):
        r = residual(q)
        if np.linalg.norm(r[:3]) < 1e-5:
            break
        jac = np.array([(residual(q + dq) - r) / 1e-6 for dq in np.eye(6) * 1e-6]).T
        q = q - np.linalg.solve(jac.T @ jac + 1e-6 * np.eye(6), jac.T @ r)
    st.__dict__.setdefault("_seed", {})[arm_id] = q
    return q


# ---------------------------------------------------------------- geometry
def test_segment_distance_cases(script):
    sd = script._segment_distance
    p = lambda *v: np.array(v, float)  # noqa: E731
    d, _, _ = sd(p(0, 0, 0), p(1, 0, 0), p(0.5, 1, 0), p(0.5, 2, 0))
    assert d == pytest.approx(1.0)
    d, _, _ = sd(p(0, 0, 0), p(1, 0, 0), p(0, 0.3, 0), p(1, 0.3, 0))  # parallel
    assert d == pytest.approx(0.3)
    d, c1, c2 = sd(p(0, 0, 0), p(1, 0, 0), p(0.5, -1, 0.2), p(0.5, 1, 0.2))  # crossing
    assert d == pytest.approx(0.2)
    np.testing.assert_allclose(c1, [0.5, 0, 0], atol=1e-12)
    d, _, _ = sd(p(0, 0, 0), p(0, 0, 0), p(3, 4, 0), p(3, 4, 0))  # two points
    assert d == pytest.approx(5.0)
    d, _, _ = sd(p(2, 1, 0), p(2, 1, 0), p(0, 0, 0), p(1, 0, 0))  # point past a segment's end
    assert d == pytest.approx(math.hypot(1, 1))


def test_capsules_are_a_lower_bound_of_the_measured_clearance(script):
    """At the recorded handover configurations the prediction sits 0-12 mm under the twin's measure."""
    for config in (FAR, NEAR, TOUCH):
        out = script.run(Station(script, config).ctx(), skip_links="gripper,base")
        assert out["route"] in ("clear", "too_close"), out["detail"]
        assert config["measured"] - 0.012 <= out["min_clearance_m"] <= config["measured"], (
            config,
            out,
        )


# ---------------------------------------------------------------- routes
def test_far_apart_is_clear(script):
    out = script.run(Station(script, FAR).ctx())
    assert out["route"] == "clear", out["detail"]
    assert 0.19 < out["min_clearance_m"] < 0.2178
    assert out["samples"] == 1 and out["at_fraction"] == 0.0
    assert out["closest_link_a"].startswith("arm0:") and out["closest_link_b"].startswith("arm1:")
    assert out["model_error_m"] < 1e-6
    assert "now" in out["detail"]


def test_touching_fingertips_are_too_close(script):
    out = script.run(Station(script, TOUCH).ctx())
    assert out["route"] == "too_close"
    assert out["min_clearance_m"] < 0.0
    assert (out["closest_link_a"], out["closest_link_b"]) == ("arm0:tip_right", "arm1:tip_left")
    a = np.array([out["closest_point_a"][k] for k in "xyz"])
    b = np.array([out["closest_point_b"][k] for k in "xyz"])
    assert np.linalg.norm(a - b) < 0.05


def test_threshold_decides_between_clear_and_too_close(script):
    st = Station(script, NEAR)
    tight = script.run(st.ctx(), too_close_m=0.001)
    loose = script.run(st.ctx(), too_close_m=0.02)
    assert tight["route"] == "clear" and loose["route"] == "too_close"
    assert tight["min_clearance_m"] == pytest.approx(loose["min_clearance_m"])


def test_bases_come_from_the_readings_not_from_constants(script):
    """Rotate and move the whole station: the answer does not change."""
    rot = np.eye(4)
    rot[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    rot[:3, 3] = [1.0, -2.0, 0.3]
    bases = {0: rot @ _base(0.305), 1: rot @ _base(-0.305)}
    plain = script.run(Station(script, NEAR).ctx())
    moved = script.run(Station(script, NEAR, bases=bases).ctx())
    assert moved["min_clearance_m"] == pytest.approx(plain["min_clearance_m"], abs=1e-9)
    assert (moved["closest_link_a"], moved["closest_link_b"]) == (
        plain["closest_link_a"],
        plain["closest_link_b"],
    )


def test_jaw_opening_moves_the_fingertips(script):
    st = Station(script, NEAR)
    opened = script.run(st.ctx(), grip_b=1.0)["min_clearance_m"]
    closed = script.run(st.ctx(), grip_b=0.0)["min_clearance_m"]
    assert opened != pytest.approx(closed, abs=1e-4)


def test_attached_object_counts_against_the_other_arm(script):
    st = Station(script, NEAR)
    bare = script.run(st.ctx())
    held = {"frame": "tcp", "spheres": [{"center": [0.0, 0.0, 0.0], "radius": 0.04}]}
    out = script.run(st.ctx(), attached_a=held)
    assert out["min_clearance_m"] < bare["min_clearance_m"]
    assert out["closest_link_a"] == "arm0:attached"


# ---------------------------------------------------------------- paths
def test_joint_path_into_contact_is_too_close_part_way(script):
    st = Station(script, FAR)
    out = script.run(
        st.ctx(), trajectory_b={"waypoints": [{"positions": TOUCH["q1"]}]}, grip_b=TOUCH["f1"]
    )
    assert out["route"] == "too_close"
    assert out["start_clearance_m"] > 0.15 and out["end_clearance_m"] < 0.0
    assert 0.0 < out["at_fraction"] <= 1.0
    largest = float(np.max(np.abs(np.subtract(TOUCH["q1"], FAR["q1"]))))
    assert out["samples"] == math.ceil(largest / 0.01) + 1
    assert st.q[1].tolist() == FAR["q1"]  # nothing moved


def test_order_of_two_moves(script):
    st = Station(script, FAR)
    away = {"waypoints": [{"positions": FAR["q1"]}]}
    # b moves in first (contact) then a: contact is in the path; a then b: also; together resamples.
    trajectories = {
        "trajectory_a": {"waypoints": [{"positions": TOUCH["q0"]}]},
        "trajectory_b": {"waypoints": [{"positions": TOUCH["q1"]}]},
    }
    seq = script.run(st.ctx(), order="a_then_b", **trajectories)
    rev = script.run(st.ctx(), order="b_then_a", **trajectories)
    tog = script.run(st.ctx(), order="together", **trajectories)
    assert seq["samples"] == rev["samples"]
    assert tog["samples"] < seq["samples"]
    assert seq["end_clearance_m"] == pytest.approx(rev["end_clearance_m"])
    assert seq["end_clearance_m"] == pytest.approx(tog["end_clearance_m"])
    # A path that returns b to where it stands is just the start.
    same = script.run(st.ctx(), trajectory_b=away)
    assert same["min_clearance_m"] == pytest.approx(same["start_clearance_m"])


def test_pose_sequence_is_solved_along_the_line(script):
    st = Station(script, FAR)
    goal = st.tool_pose(1, NEAR["q1"])
    seq = {"waypoints": [{"pose": script._pose_dict(goal)}]}
    out = script.run(st.ctx(), poses_b=seq)
    assert out["route"] in ("clear", "too_close"), out["detail"]
    solves = [c for c in st.calls if c[0] == "ik"]
    dist = float(np.linalg.norm(goal[:3, 3] - st.tool_pose(1)[:3, 3]))
    assert all(c[1] == 1 for c in solves)
    assert len(solves) >= math.ceil(dist / 0.02)
    near = script.run(Station(script, NEAR).ctx())
    assert out["end_clearance_m"] == pytest.approx(near["min_clearance_m"], abs=0.003)


def test_pose_waypoint_planned_joint_is_solved_once(script):
    st = Station(script, FAR)
    goal = script._pose_dict(st.tool_pose(1, NEAR["q1"]))
    script.run(st.ctx(), poses_b={"waypoints": [{"pose": goal, "cartesian": False}]})
    assert len([c for c in st.calls if c[0] == "ik"]) == 1


def test_pose_sequence_attached_object_is_used(script):
    st = Station(script, NEAR)
    here = script._pose_dict(st.tool_pose(0))
    held = {"frame": "tcp", "spheres": [{"center": [0.0, 0.0, 0.0], "radius": 0.04}]}
    out = script.run(st.ctx(), poses_a={"waypoints": [{"pose": here}], "attached_object": held})
    assert out["closest_link_a"] == "arm0:attached"


@pytest.mark.parametrize(
    "ik, words",
    [
        (
            lambda st, pose, arm: (_ for _ in ()).throw(
                ToolError("robot.solve_ik", "Target pose is unreachable")
            ),
            "IK refused",
        ),
        (
            lambda st, pose, arm: {
                "joint_config": {"positions": [0.0] * 6},
                "success": True,
                "within_servo_tolerance": False,
            },
            "no solution within tolerance",
        ),
        (
            lambda st, pose, arm: {"joint_config": {"positions": [0.0] * 6}, "success": False},
            "no solution within tolerance",
        ),
        (
            lambda st, pose, arm: {
                "joint_config": {"positions": [3.0, 0, 0, 0, 0, 0]},
                "success": True,
            },
            "changed branch",
        ),
    ],
)
def test_pose_sequence_failures_are_unknown(script, ik, words):
    st = Station(script, FAR, ik=ik)
    goal = script._pose_dict(st.tool_pose(1, NEAR["q1"]))
    out = script.run(st.ctx(), poses_b={"waypoints": [{"pose": goal}]})
    assert out["route"] == "unknown"
    assert words in out["detail"]
    assert math.isnan(out["min_clearance_m"])


# ---------------------------------------------------------------- refusals
def test_one_arm_station_is_unknown(script):
    out = script.run(Station(script, FAR, arms=1).ctx())
    assert out["route"] == "unknown"
    assert "1 arm(s)" in out["detail"]


def test_second_arm_reading_the_first_is_unknown(script):
    config = dict(FAR, q1=FAR["q0"])
    st = Station(script, config, bases={0: _base(0.305), 1: _base(0.305)})
    out = script.run(st.ctx())
    assert out["route"] == "unknown"
    assert "same pose" in out["detail"]


def test_wrong_joint_count_is_unknown(script):
    st = Station(script, FAR)
    st.q[1] = np.r_[st.q[1], 0.0]
    st.tool_pose = lambda arm_id, q=None: Station.tool_pose(
        st, arm_id, (st.q[arm_id] if q is None else q)[:6]
    )
    out = script.run(st.ctx())
    assert out["route"] == "unknown" and "7 joint readings" in out["detail"]


def test_model_mismatch_is_unknown(script):
    out = script.run(Station(script, FAR, fk_scale=1.2).ctx())
    assert out["route"] == "unknown"
    assert "not the modelled arm" in out["detail"]
    assert out["model_error_m"] > 0.005


def test_without_forward_kinematics_it_answers_and_says_so(script):
    out = script.run(Station(script, FAR, fk=False).ctx())
    assert out["route"] == "clear"
    assert "not cross-checked" in out["detail"]
    assert math.isnan(out["model_error_m"])


def test_without_describe_arm_the_yam_tcp_is_used(script):
    a = script.run(Station(script, NEAR, describe=False).ctx())
    b = script.run(Station(script, NEAR).ctx())
    assert a["min_clearance_m"] == pytest.approx(b["min_clearance_m"], abs=1e-9)


@pytest.mark.parametrize(
    "kw, words",
    [
        ({"arm_a": 1, "arm_b": 1}, "both 1"),
        ({"order": "sideways"}, "order must be"),
        (
            {"skip_links": "base,link1,link2,link3,link4,link5,link6,gripper,tip_left,tip_right"},
            "no capsules",
        ),
        ({"trajectory_a": {"waypoints": [{"positions": [0.0] * 5}]}}, "6 joints"),
    ],
)
def test_bad_inputs_are_unknown(script, kw, words):
    out = script.run(Station(script, FAR).ctx(), **kw)
    assert out["route"] == "unknown"
    assert words in out["detail"]


def test_trajectory_and_poses_for_one_arm_conflict(script):
    st = Station(script, FAR)
    out = script.run(
        st.ctx(),
        trajectory_b={"waypoints": [{"positions": NEAR["q1"]}]},
        poses_b={"waypoints": [{"pose": script._pose_dict(st.tool_pose(1))}]},
    )
    assert out["route"] == "unknown" and "not both" in out["detail"]


def test_observation_failure_is_unknown(script):
    ctx = FakeContext(
        {
            "robot.get_observation": lambda: (_ for _ in ()).throw(
                ToolError("robot.get_observation", "down")
            )
        }
    )
    out = script.run(ctx)
    assert out["route"] == "unknown" and "no observation" in out["detail"]
