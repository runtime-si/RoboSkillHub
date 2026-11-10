"""roboskillhub.tasks.lego_hub: hub skills as task skills on a fake LEGO station context."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from roboskillhub.runtime.graph import load_graph
from roboskillhub.runtime.task import SkillContext, run_task_skill
from roboskillhub.tasks import load_registry
from roboskillhub.tasks.lego_hub import depth as D
from roboskillhub.tasks.lego_hub import lego_press_hub
from roboskillhub.tasks.lego_hub.bridge import GapTools

PLATE = {
    "position_m": [0.3, 0.0, 0.0032],
    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    "class": "baseplate",
    "geometry": {"type": "box", "size_m": [0.256, 0.256, 0.0032]},
}
SEAT = [0.3 + 0.008, 0.0 + 0.016, 0.0032]  # a 2x4 on whole cells of the 32 x 32 plate
TOP_DOWN = [1.0, 0.0, 0.0, 0.0]  # TCP z down
CAPS = {
    "pose_feedback", "cartesian_motion", "gripper", "gripper_feedback", "stop", "object_feedback",
    "object_pose_feedback", "contact_force_feedback", "compliant_motion",
}  # fmt: skip


def brick(pos, yaw_deg=0.0, tilt_deg=0.0):
    a, b = math.radians(yaw_deg) / 2, math.radians(tilt_deg) / 2
    qz = [0.0, 0.0, math.sin(a), math.cos(a)]
    qy = [0.0, math.sin(b), 0.0, math.cos(b)]
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qy
    q = [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]
    return {
        "position_m": list(pos),
        "quaternion_xyzw": q,
        "class": "lego_brick",
        "geometry": {"type": "box", "size_m": [0.016, 0.032, 0.0096]},
    }


class Station:
    """Objects at fixed poses; move_pose puts the TCP where asked (top-down only)."""

    def __init__(self, objects, tcp=(0.3, 0.1, 0.25), holding=False, width=0.03, stall=None):
        self.objects = dict(objects)
        self.tcp = list(tcp)
        self.holding, self.width = holding, width
        self.stall = stall  # z the hand cannot rise above (a held brick that does not come up)
        self.calls: list[tuple[str, dict]] = []

    def primitive(self, capability, request, rid):
        self.calls.append((capability, request))
        if capability == "observe_objects":
            obs = []
            for oid in request["object_ids"]:
                o = self.objects.get(oid)
                obs.append({"object_id": oid, "frame_id": "world", "timestamp_s": 1.0,
                            "visibility": "visible" if o else "not_found", "privileged": True,
                            "source": "simulation_truth",
                            **(o or {"position_m": None, "quaternion_xyzw": None})})  # fmt: skip
            return {"status": "succeeded", "evidence": {"observations": obs}}
        if capability == "move_pose":
            q = request["target_pose"]["quaternion_xyzw"]
            if D.quat_matrix(q)[2, 2] > -0.999:
                return {"status": "rejected", "failure_type": "unreachable"}
            p = list(request["target_pose"]["position_m"])
            if self.stall is not None:
                p[2] = min(p[2], self.stall)
            self.tcp = p
            return {"status": "succeeded"}
        if capability == "set_gripper":
            self.width = request["width_m"]
            self.holding = False
            return {"status": "succeeded"}
        raise AssertionError(capability)

    def state(self):
        return {
            "tcp_pose": {
                "frame_id": "world",
                "position_m": list(self.tcp),
                "quaternion_xyzw": TOP_DOWN,
            },
            "gripper": {"holding": self.holding, "width_m": self.width, "source": "sim"},
        }

    def ctx(self):
        return SkillContext(
            task_id="t", request_id="r", capabilities=frozenset(CAPS), privileged_allowed=True,
            primitive=self.primitive, read_state=self.state, sleep=lambda s: None,
        )  # fmt: skip


def mating(pos=SEAT):
    return {
        "target_id": "b1",
        "mating_pose": {
            "frame_id": "world",
            "position_m": list(pos),
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        "baseplate": {"object_id": "plate"},
        "brick": {"studs": [2, 4], "system": "lego"},
    }


@pytest.fixture(scope="module")
def K():
    reg, unavailable = load_registry()
    assert "roboskillhub.tasks.lego_hub" not in unavailable
    return reg


def test_render_sees_a_seated_brick_top_and_the_plate():
    objs = {"plate": PLATE, "b1": brick(SEAT)}
    cam = np.eye(4)
    cam[:3, :3] = np.diag([1.0, -1.0, -1.0])
    cam[:3, 3] = [SEAT[0], SEAT[1], 0.25]
    d = D.render(objs, cam)
    # Straight down through the brick's centre: its top (no stud at the centre of a 2x4).
    assert d[240, 320] == pytest.approx(0.25 - (0.0032 + 0.0096), abs=1e-4)
    # 7.6 cm off the brick, still on the plate: a stud top or the plate between studs.
    assert d[240, 200] == pytest.approx(0.25 - 0.0032 - 0.00085, abs=0.0009)


def test_gap_tools_map_onto_the_primitives():
    st = Station({"plate": PLATE})
    tools = GapTools(st.ctx(), ["plate"])
    pose = {
        "position": {"x": 0.3, "y": 0.0, "z": 0.1},
        "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0},
    }
    assert tools.tool("robot.go_to_pose", pose=pose, tolerance=0.001)["reached"] is True
    assert st.calls[-1][0] == "move_pose" and st.tcp == [0.3, 0.0, 0.1]
    tilted = {**pose, "rotation": {"w": 0.0, "x": 0.966, "y": 0.0, "z": 0.259}}
    assert tools.tool("robot.solve_ik", pose=tilted)["success"] is False
    assert tools.tool("robot.solve_ik", pose=pose)["success"] is True
    assert tools.tool("robot.go_to_pose", pose=tilted)["outcome"] == "unreachable"
    assert tools.tool("robot.get_gripper")["position"] == pytest.approx(0.6)
    obs = tools.tool("robot.get_observation")
    cam = obs["cameras"][0]
    assert cam["depth"].shape == (480, 640) and cam["intrinsics"][0][0] == 394.0
    with pytest.raises(RuntimeError):
        tools.tool("robot.move_to_joints", joint_config={})


def _run(K, skill, request, st):
    return run_task_skill(K[skill], request, st.ctx())


def test_verify_seat_depth_succeeds_on_a_seated_brick(K):
    st = Station({"plate": PLATE, "b1": brick(SEAT)})
    r = _run(K, "hub.verify_seat_depth", {"mating": mating(), "pieces": ["b1"]}, st)
    assert r["status"] == "succeeded", r["evidence"].get("reason")
    assert r["evidence"]["postcondition"]["passed"] is True
    assert r["observations"] and r["outputs"]["seated"] is True


def test_verify_seat_depth_fails_a_brick_resting_on_the_studs(K):
    up = [SEAT[0], SEAT[1], SEAT[2] + 0.0017]
    st = Station({"plate": PLATE, "b1": brick(up)})
    r = _run(K, "hub.verify_seat_depth", {"mating": mating(), "pieces": ["b1"]}, st)
    assert r["status"] == "failed" and r["failure_type"] == "lego.not_seated"
    assert r["outputs"]["route"] == "high_by_stud"


def test_classify_resting_face_on_the_feeder(K):
    st = Station({"b1": brick([0.1, -0.15, 0.0])})
    r = _run(
        K, "hub.classify_resting_face", {"object_id": "b1", "pieces": ["b1"], "table_z": 0.0}, st
    )
    assert r["status"] == "succeeded", r["evidence"].get("reason")
    assert r["outputs"]["face"] == "studs_up"


def test_pull_up_reads_a_stalled_hand_as_connected_and_a_free_one_as_not(K):
    seat_tcp = (SEAT[0], SEAT[1], SEAT[2] + 0.013)
    st = Station(
        {"b1": brick(SEAT)}, tcp=seat_tcp, holding=True, width=0.016, stall=seat_tcp[2] + 0.0004
    )
    r = _run(K, "hub.pull_up_check", {"object_id": "b1"}, st)
    assert r["status"] == "succeeded" and r["outputs"]["route"] == "connected"
    assert not any(c == "set_gripper" for c, _ in st.calls)  # the graph's release node opens
    st = Station({"b1": brick(SEAT)}, tcp=seat_tcp, holding=True, width=0.016)
    r = _run(K, "hub.pull_up_check", {"object_id": "b1"}, st)
    assert r["status"] == "failed" and r["failure_type"] == "lego.not_seated"
    assert r["evidence"]["pull_log"]


def test_track_build_seeds_places_and_flags_floating_and_moved_bricks(K):
    pre = {"target_id": "p1"}
    st = Station({"plate": PLATE, "p1": brick(SEAT)})
    r = _run(K, "hub.track_build", {"action": "init", "prebuilt": [pre], "plate_id": "plate"}, st)
    assert r["status"] == "succeeded" and r["outputs"]["n_bricks"] == 1
    state = r["outputs"]["state_json"]
    # b1 seated on p1: connected.
    st.objects["b1"] = brick([SEAT[0], SEAT[1], SEAT[2] + 0.0096])
    ok = _run(K, "hub.track_build", {"action": "place", "object_id": "b1", "state_json": state,
                                      "mating": mating()}, st)  # fmt: skip
    assert ok["status"] == "succeeded" and ok["outputs"]["n_bricks"] == 2
    # b2 resting a stud height up on the plate elsewhere: not on any studs.
    st.objects["b2"] = brick([SEAT[0] + 0.05, SEAT[1], SEAT[2] + 0.0017])
    r = _run(K, "hub.track_build", {"action": "place", "object_id": "b2",
                                     "state_json": ok["outputs"]["state_json"], "mating": mating()}, st)  # fmt: skip
    assert r["status"] == "failed" and r["failure_type"] == "lego.not_seated"
    # p1 knocked 6 mm sideways since it was tracked.
    st.objects["p1"] = brick([SEAT[0] + 0.006, SEAT[1], SEAT[2]])
    st.objects["b3"] = brick([SEAT[0] - 0.06, SEAT[1], SEAT[2]])
    r = _run(K, "hub.track_build", {"action": "place", "object_id": "b3",
                                     "state_json": ok["outputs"]["state_json"], "mating": mating()}, st)  # fmt: skip
    assert r["status"] == "failed" and r["failure_type"] == "lego.structure_damaged"
    assert {d["id"] for d in r["evidence"]["discrepancies"] if "id" in d} >= {"p1"}


def test_track_build_init_without_prebuilt_gives_an_empty_model(K):
    st = Station({"plate": PLATE})
    r = _run(K, "hub.track_build", {"action": "init", "prebuilt": [], "plate_id": "plate"}, st)
    assert r["status"] == "succeeded"
    assert json.loads(r["outputs"]["state_json"])["bricks"] == []


def test_gate_re_emits_a_status(K):
    st = Station({})
    assert (
        _run(K, "hub.gate", {"status": "succeeded", "failure_type": None}, st)["status"]
        == "succeeded"
    )
    r = _run(K, "hub.gate", {"status": "failed", "failure_type": "lego.misaligned"}, st)
    assert (r["status"], r["failure_type"]) == ("failed", "lego.misaligned")


def test_graph_wires_the_hub_nodes_and_keeps_verify_in_charge(K):
    g = lego_press_hub(recovery=False, attempts=1, bricks=3, grasps="configured")
    load_graph(g, K)
    n = g["nodes"]
    assert g["start"] == "track_init" and n["track_init"]["on"]["succeeded"] == "select"
    assert n["bind"]["on"]["succeeded"] == "face" and n["face"]["on"]["failed"] == "observe_piece"
    assert n["seat"]["on"]["succeeded"] == "pullup" and n["pullup"]["on"]["failed"] == "release"
    assert n["verify"]["on"]["failed"] == "depth" and n["depth"]["on"]["failed"] == "track"
    assert n["gate"]["on"] == {
        "succeeded": "select",
        "failed": "abort",
        "unverified": "unsure",
        "rejected": "abort",
    }
    assert g["budgets"]["timeout_s"] <= 600
    strict = lego_press_hub(
        mode="strict", recovery=False, attempts=1, bricks=1, grasps="configured"
    )
    load_graph(strict, K)
    assert strict["nodes"]["depth"]["on"]["failed"] == "abort"
    through = lego_press_hub(
        through_seat=True, recovery=False, attempts=1, bricks=1, grasps="configured"
    )
    assert through["nodes"]["seat"]["on"]["failed"] == "pullup"
    with pytest.raises(ValueError):
        lego_press_hub(mode="loose")


class _RenderingPhysics:
    """Stands in for LegoIsaacPhysics(render=True): records the camera poses asked for."""

    render = True

    def __init__(self):
        self.asked = []

    def camera_frame(self, name, pose_cv=None):
        self.asked.append((name, None if pose_cv is None else np.array(pose_cv)))
        pose = np.eye(4) if pose_cv is None else np.array(pose_cv)
        return {
            "name": name,
            "rgb": np.full((480, 640, 3), 7, np.uint8),
            "depth": np.full((480, 640), 0.2 if name == "wrist" else 0.45, np.float32),
            "intrinsics": np.array(D.D405_K),
            "pose": pose,
        }


def test_observation_uses_rendered_frames_when_the_physics_renders(monkeypatch):
    import sys
    import types

    from roboskillhub.tasks.lego_hub import bridge

    assert bridge._rendering_physics() is None  # default: ray-cast depth
    fake = _RenderingPhysics()
    mod = types.SimpleNamespace(ACTIVE=fake)
    monkeypatch.setitem(sys.modules, "roboskillhub.tasks.assembly_eval.lego_station_isaac", mod)
    monkeypatch.setenv("LEGO_RENDER_NOISE", "0")
    st = Station({"plate": PLATE})
    tools = GapTools(st.ctx(), ["plate"])
    obs = tools.tool("robot.get_observation")
    names = [c["name"] for c in obs["cameras"]]
    assert names == ["hand_eye_in_hand", "overhead"]
    wrist = obs["cameras"][0]
    assert wrist["depth"][0, 0] == pytest.approx(0.2) and wrist["rgb"].shape == (480, 640, 3)
    assert "Isaac RTX render" in wrist["source"]
    # The wrist camera is posed where the ray-cast one would be: TCP x CAMERA_IN_TCP.
    name, pose = fake.asked[0]
    assert name == "wrist" and pose[2, 3] == pytest.approx(st.tcp[2] + 0.08)
    assert "plate" in tools.observations  # the verdict still carries independent observations
    fake.render = False
    assert bridge._rendering_physics() is None
