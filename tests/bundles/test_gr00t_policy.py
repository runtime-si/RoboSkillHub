"""policies/gr00t with the model faked: the gr00t-wam/1 server (fake backend) over HTTP and ZMQ, the client's
observation encoding, joint frame and marks, and run_skill's loop, guards, jaw switching and exits."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

from .conftest import SKILLS_ROOT

BUNDLE = SKILLS_ROOT / "policies" / "gr00t"
HOME = [0.0] * 6


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve their annotations through the module
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def srv():
    return _load("gr00t_server", BUNDLE / "server.py")


@pytest.fixture(scope="module")
def g(skills_registry):
    info = skills_registry.get("gr00t")
    assert info.kind == "policy"
    return info.tools_module


def _serve(srv, backend):
    service = srv.Service(backend)
    httpd = srv.make_http_server(service, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return service, httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


@pytest.fixture
def server(srv, g, monkeypatch):
    backend = srv.FakeBackend()
    service, httpd, url = _serve(srv, backend)
    monkeypatch.setenv("GAP_GR00T_URL", url)
    monkeypatch.setenv("GAP_GR00T_IMAGE_CODEC", "npy")
    monkeypatch.delenv("GAP_GR00T_JOINT_OFFSET", raising=False)
    monkeypatch.delenv("GAP_GR00T_CAMERAS", raising=False)
    g._clients.clear()
    yield backend
    httpd.shutdown()
    httpd.server_close()
    g._clients.clear()


K = np.array([[200.0, 0, 64.0], [0, 200.0, 48.0], [0, 0, 1]])
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
"""A camera turned 180 deg about x: its +z (OpenCV forward) points down the world's -z."""


def _cam(name, value=90, pose=None):
    return {
        "name": name,
        "rgb": np.full((96, 128, 3), value, np.uint8),
        "intrinsics": K,
        "pose": pose or {"position": {"x": 0.0, "y": 0.0, "z": 1.0}, "rotation": DOWN},
    }


def _arm(q, opening=1.0):
    return {
        "joint_state": {"positions": np.asarray(q, float)},
        "gripper_fraction": opening,
        "ee_pose": {"position": {"x": 0.3, "y": 0.1, "z": 0.2}, "rotation": DOWN},
    }


def _obs(left=HOME, right=HOME, lo=1.0, ro=1.0, cams=("top", "left_wrist", "right_wrist")):
    return {"cameras": [_cam(c) for c in cams], "arms": [_arm(left, lo), _arm(right, ro)]}


class Robot:
    """A YAM pair whose moves land where commanded; jaws open / close on command."""

    def __init__(self):
        self.q = {0: np.zeros(6), 1: np.zeros(6)}
        self.opening = {0: 1.0, 1: 1.0}
        self.waypoints = {0: [], 1: []}
        self.jaw_calls = []

    def observation(self):
        return _obs(self.q[0], self.q[1], self.opening[0], self.opening[1])

    def execute(self, trajectory, arm_id=0, **_):
        wps = [np.asarray(w["positions"], float) for w in trajectory["waypoints"]]
        self.waypoints[arm_id].extend(wps)
        self.q[arm_id] = wps[-1]
        return {"status": "ok"}

    def dual(self, tracks, **_):
        for a, wps in tracks.items():
            self.waypoints[int(a)].extend(np.asarray(w, float) for w in wps)
            self.q[int(a)] = np.asarray(wps[-1], float)
        return {"arms": sorted(tracks)}

    def jaw(self, value):
        def call(arm_id=0, **_):
            self.opening[arm_id] = value
            self.jaw_calls.append((arm_id, value))
            return {"position": value}

        return call

    def ctx(self, **extra):
        return FakeContext(
            {
                "robot.get_observation": lambda **_: self.observation(),
                "robot.execute_trajectory": self.execute,
                "robot.stream_dual": self.dual,
                "robot.open_gripper": self.jaw(1.0),
                "robot.close_gripper": self.jaw(0.0),
                **extra,
            }
        )


# ---------------------------------------------------------------------------
# Server and protocol
# ---------------------------------------------------------------------------


def test_info_names_the_protocol_backend_and_layout(server, g):
    info = g.info()
    assert info["protocol"] == "gr00t-wam/1"
    assert info["backend"] == "fake"
    assert info["roles"] == ["top", "left_wrist", "right_wrist"]
    assert "opening 0 closed" in info["layout"]


def test_act_returns_a_chunk_in_the_station_joint_frame(server, g):
    q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    out = g.act(_obs(left=q, right=[-v for v in q]), "Pick up the block.")
    assert out["actions"].shape == (16, 14)
    assert out["horizon"] == 16 and out["dt"] == pytest.approx(1 / 30)
    np.testing.assert_allclose(out["actions"][0, :6], q, atol=1e-6)
    np.testing.assert_allclose(out["actions"][0, 7:13], [-v for v in q], atol=1e-6)
    assert out["steering"] == "emulated-overlay" and out["backend"] == "fake"
    assert out["latency_ms"] > 0 and "total" in out["server_ms"]


def test_joint_offset_is_removed_on_the_way_in_and_added_on_the_way_out(server, g):
    off = {"left": [np.pi / 2, 0, 0, 0, 0, 0], "right": [np.pi / 2, 0, 0, 0, 0, 0]}
    station = [np.pi / 2 + 0.1, 0.2, 0, 0, 0, 0]
    body = g.request(_obs(left=station), "x", offset=g.joint_offset(off))
    assert body["joints"][0] == pytest.approx(0.1)
    out = g.act(_obs(left=station), "x", joint_offset_rad=off)
    assert out["actions"][0, 0] == pytest.approx(station[0], abs=1e-6)


def test_joint_offset_from_the_environment(server, g, monkeypatch):
    monkeypatch.setenv("GAP_GR00T_JOINT_OFFSET", json.dumps({"left": [1, 0, 0, 0, 0, 0]}))
    assert g.joint_offset()["left"][0] == 1.0 and g.joint_offset()["right"][0] == 0.0
    with pytest.raises(ValueError):
        g.joint_offset({"left": [1, 2]})


@pytest.mark.parametrize("codec", ["jpeg", "png", "npy"])
def test_every_image_codec_reaches_the_server(srv, g, monkeypatch, codec):
    seen = {}

    class Spy(srv.FakeBackend):
        def act(self, req):
            seen.update({k: v.shape for k, v in req.images.items()})
            return super().act(req)

    _, httpd, url = _serve(srv, Spy())
    monkeypatch.setenv("GAP_GR00T_URL", url)
    monkeypatch.setenv("GAP_GR00T_IMAGE_CODEC", codec)
    g._clients.clear()
    try:
        g.act(_obs(), "x")
    finally:
        httpd.shutdown()
        httpd.server_close()
        g._clients.clear()
    assert seen == {r: (96, 128, 3) for r in ("top", "left_wrist", "right_wrist")}


def test_a_bad_request_comes_back_as_a_tool_error(server, g, monkeypatch):
    with pytest.raises(ToolError, match="instruction"):
        g.act(_obs(), "  ")
    with pytest.raises(ToolError, match="no arms"):
        g.act({"cameras": [_cam("top")], "arms": []}, "x")
    monkeypatch.setenv("GAP_GR00T_URL", "http://127.0.0.1:9")
    g._clients.clear()
    with pytest.raises(ToolError, match="server"):
        g.act(_obs(), "x")


def test_the_server_refuses_malformed_bodies(srv):
    service = srv.Service(srv.FakeBackend())
    with pytest.raises(ValueError, match="joints"):
        service.handle("act", {"instruction": "x", "joints": [0.0] * 7})
    with pytest.raises(ValueError, match="roles"):
        service.handle(
            "act",
            {
                "instruction": "x",
                "joints": [0.0] * 14,
                "images": {"front": np.zeros((4, 4, 3), np.uint8)},
            },
        )
    with pytest.raises(ValueError, match="mark kind"):
        service.handle(
            "act",
            {"instruction": "x", "joints": [0.0] * 14, "marks": [{"kind": "poke", "uv": [1, 1]}]},
        )
    with pytest.raises(ValueError, match="endpoint"):
        service.handle("train", {})


def test_zmq_transport_speaks_the_same_protocol(srv, g, monkeypatch):
    pytest.importorskip("zmq")
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    service = srv.Service(srv.FakeBackend())
    threading.Thread(target=srv.serve_zmq, args=(service, "127.0.0.1", port), daemon=True).start()
    monkeypatch.setenv("GAP_GR00T_URL", f"zmq://127.0.0.1:{port}")
    g._clients.clear()
    try:
        assert g.info()["protocol"] == "gr00t-wam/1"
        assert g.act(_obs(), "x")["actions"].shape == (16, 14)
    finally:
        g._clients.clear()


def test_draw_marks_paints_a_disc_in_the_kind_colour(srv):
    img = np.zeros((100, 100, 3), np.uint8)
    out = srv.draw_marks(
        img, [{"kind": "grasp", "uv": [30, 40]}, {"kind": "release", "uv": [70, 60]}]
    )
    assert tuple(out[40, 30]) == srv.MARK_COLOURS["grasp"]
    assert tuple(out[60, 70]) == srv.MARK_COLOURS["release"]
    assert tuple(out[5, 5]) == (0, 0, 0) and img.max() == 0
    assert "green dot" in srv.mark_sentence([{"kind": "grasp"}]) and srv.mark_sentence([]) == ""


def test_profiles_assemble_14_d_chunks_and_rot6d_state(srv):
    joints = np.arange(14, dtype=float)
    prof = srv.PROFILES["yam-molmoact2"]
    action = {
        "left_arm": np.ones((1, 16, 6)),
        "left_gripper": np.zeros((1, 16, 1)),
        "right_arm": 2 * np.ones((1, 16, 6)),
        "right_gripper": np.ones((1, 16, 1)),
    }
    out = srv.assemble_actions(prof, action, joints)
    assert out.shape == (16, 14)
    assert (
        (out[:, :6] == 1).all()
        and (out[:, 6] == 0).all()
        and (out[:, 7:13] == 2).all()
        and (out[:, 13] == 1).all()
    )
    xdof = srv.PROFILES["xdof"]
    held = srv.assemble_actions(xdof, {"left_joint_pos": np.zeros((1, 40, 6))}, joints)
    assert held.shape == (40, 14) and (held[:, 6:] == joints[6:]).all()
    ee = srv.state_value("left_ee", joints, {"left": [1, 2, 3, 1, 0, 0, 0]})
    np.testing.assert_allclose(ee, [1, 2, 3, 1, 0, 0, 0, 1, 0])


def test_gr00t2_backend_is_a_documented_placeholder(srv):
    with pytest.raises(NotImplementedError, match="GR00T 2"):
        srv.Gr00t2Backend()
    assert "predicted_frames" in srv.Gr00t2Backend.__doc__
    assert srv.Gr00t2Backend.steering == "native"


# ---------------------------------------------------------------------------
# Observation encoding and marks
# ---------------------------------------------------------------------------


def test_camera_roles_by_name_and_by_override(g):
    obs = {"cameras": [_cam("overhead"), _cam("wrist"), _cam("cam_right_wrist")], "arms": []}
    roles = g.camera_roles(obs)
    assert roles["top"]["name"] == "overhead"
    assert (
        roles["left_wrist"]["name"] == "wrist" and roles["right_wrist"]["name"] == "cam_right_wrist"
    )
    roles = g.camera_roles(
        {"cameras": [_cam("scene_a"), _cam("scene_b")], "arms": []}, {"top": "scene_b"}
    )
    assert roles["top"]["name"] == "scene_b"
    with pytest.raises(ValueError):
        g.camera_roles(obs, {"top": "nope"})


def test_one_armed_observation_holds_the_missing_arm_open(g):
    obs = {"cameras": [_cam("top")], "arms": [_arm([0.1] * 6, 0.2)]}
    j = g.measured_joints(obs)
    assert j[6] == pytest.approx(0.2) and (j[7:13] == 0).all() and j[13] == 1.0


def test_marks_from_points_projects_onto_the_view_and_drops_what_it_cannot_see(g):
    obs = _obs()
    out = g.marks_from_points(
        obs,
        [
            {"kind": "grasp", "point": [0.0, 0.0, 0.0]},  # straight below: the principal point
            {
                "kind": "release",
                "point": {"x": 0.1, "y": 0.0, "z": 0.5},
            },  # 0.5 m away, 0.1 m along +x
            {"kind": "release", "point": [0.0, 0.0, 2.0]},  # behind the camera
            {"kind": "grasp", "point": [5.0, 0.0, 0.0]},  # far off the view
        ],
    )
    assert [m["uv"] for m in out["marks"]] == [[64.0, 48.0], [104.0, 48.0]]
    assert len(out["dropped"]) == 2 and all("off the" in d["why"] for d in out["dropped"])
    with pytest.raises(ToolError):
        g.marks_from_points(obs, [{"kind": "poke", "point": [0, 0, 0]}])


def test_pixel_marks_pass_through_and_reach_the_backend(server, g):
    out = g.act(_obs(), "x", marks=[{"kind": "grasp", "uv": [10, 20], "camera": "left_wrist"}])
    assert out["marks_sent"] == [{"kind": "grasp", "camera": "left_wrist", "uv": [10.0, 20.0]}]
    assert out["actions"][-1, 6] == pytest.approx(
        0.0
    )  # the fake closes the left jaw on a grasp mark


# ---------------------------------------------------------------------------
# run_skill
# ---------------------------------------------------------------------------


def test_run_skill_closes_the_jaw_when_predicted_and_succeeds_on_the_check(server, g):
    robot = Robot()
    ctx = robot.ctx()
    r = g.run_skill(
        ctx,
        "Pick up the block.",
        marks=[{"kind": "grasp", "uv": [64, 48]}],
        max_steps=64,
        success_check=lambda c: robot.opening[0] == 0.0,
    )
    assert r["route"] == "succeeded" and r["success"]
    assert robot.jaw_calls == [(0, 0.0)]
    assert r["gripper_events"][0]["jaw"] == "close" and r["gripper_events"][0]["arm"] == "left"
    # Cut where the opening crosses 0.35: row 4 of the fake's six-row ramp, so five rows ran.
    assert r["steps"] == 5 and r["chunks"] == 1
    assert r["latency_ms_mean"] > 0


def test_run_skill_uses_a_tool_success_check_and_runs_the_budget_out(server, g):
    robot = Robot()
    ctx = robot.ctx(**{"check.done": {"state": {"done": False}}})
    r = g.run_skill(
        ctx, "x", max_steps=20, success_check={"tool": "check.done", "field": "state.done"}
    )
    assert r["route"] == "max_steps" and not r["success"]
    assert r["steps"] == 20 and r["chunks"] == 3
    assert r["check"] == {"state": {"done": False}}
    assert len(ctx.calls_to("check.done")) == 4  # every chunk and once at the end


def test_run_skill_one_arm_moves_only_that_arm(server, g):
    robot = Robot()
    g.run_skill(robot.ctx(), "x", max_steps=8, arms="right")
    assert robot.waypoints[0] == [] and len(robot.waypoints[1]) == 8


def test_run_skill_streams_both_arms_together_and_interpolates_to_the_control_rate(server, g):
    robot = Robot()
    ctx = robot.ctx()
    g.run_skill(
        ctx,
        "x",
        max_steps=8,
        executor="stream_dual",
        control_hz=60,
        executor_args={"settle_steps": 2},
    )
    assert len(robot.waypoints[0]) == 16 and len(robot.waypoints[1]) == 16
    call = ctx.calls_to("robot.stream_dual")[0]
    assert call.kwargs["settle_steps"] == 2
    assert not ctx.calls_to("robot.execute_trajectory")


def test_run_skill_falls_back_to_per_arm_trajectories_without_stream_dual(server, g):
    robot = Robot()

    def missing(**_):
        raise ToolError("robot.stream_dual", "this facade cannot stream several arms together")

    ctx = robot.ctx(**{"robot.stream_dual": missing})
    r = g.run_skill(ctx, "x", max_steps=8, executor="auto")
    assert len(ctx.calls_to("robot.execute_trajectory")) == 2
    assert "unavailable" in r["gripper_events"][0]["note"]


class Jumpy:
    """A backend whose second row moves joint 2 by 0.5 rad."""

    steering, dt = "none", 1 / 30

    def __init__(self, srv, first_gap=0.0):
        self.srv, self.first_gap = srv, first_gap

    def info(self):
        return {"backend": "jumpy", "model": "", "revision": "", "profile": "", "horizon": 16}

    def reset(self):
        pass

    def act(self, req):
        a = np.tile(req.joints.astype(np.float32), (16, 1))
        if self.first_gap:
            a[:, 1] += self.first_gap
        else:
            a[1:, 1] += 0.5
        return {"actions": a}


def test_run_skill_refuses_a_chunk_that_jumps_between_rows(srv, g, monkeypatch):
    _, httpd, url = _serve(srv, Jumpy(srv))
    monkeypatch.setenv("GAP_GR00T_URL", url)
    g._clients.clear()
    robot = Robot()
    try:
        r = g.run_skill(robot.ctx(), "x", max_steps=32)
    finally:
        httpd.shutdown()
        httpd.server_close()
        g._clients.clear()
    assert r["route"] == "unsafe_action" and "joint 1" in r["detail"]
    assert robot.waypoints[0] == [] and robot.waypoints[1] == []


def test_run_skill_ramps_to_a_far_first_target_or_refuses_it(srv, g, monkeypatch):
    _, httpd, url = _serve(srv, Jumpy(srv, first_gap=0.6))
    monkeypatch.setenv("GAP_GR00T_URL", url)
    g._clients.clear()
    try:
        robot = Robot()
        r = g.run_skill(robot.ctx(), "x", max_steps=8)
        steps = np.diff([w[1] for w in robot.waypoints[0]], prepend=0.0)
        assert r["route"] == "max_steps" and "ramped 0.600 rad" in r["gripper_events"][0]["note"]
        assert np.abs(steps).max() <= 1.0 / 30 + 1e-9  # 1 rad/s at 30 Hz
        assert robot.waypoints[0][-1][1] == pytest.approx(0.6, abs=1e-6)
        refused = g.run_skill(Robot().ctx(), "x", max_steps=8, ramp_rad_s=0)
        assert refused["route"] == "unsafe_action" and "first target" in refused["detail"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        g._clients.clear()


def test_run_skill_rejects_unknown_arms_and_executors(server, g):
    with pytest.raises(ToolError):
        g.run_skill(Robot().ctx(), "x", arms="third")
    with pytest.raises(ToolError):
        g.run_skill(Robot().ctx(), "x", executor="teleport")


def test_bundle_declares_its_licences_and_exits():
    text = (BUNDLE / "SKILL.md").read_text()
    assert "NVIDIA Open Model License" in text and "Apache-2.0" in text
    for exit_name in ("succeeded", "max_steps", "unsafe_action", "failed"):
        assert f"    {exit_name}:" in text
    notice = (SKILLS_ROOT / "NOTICE").read_text()
    assert "Isaac-GR00T" in notice


def test_server_and_tools_import_no_model_stack():
    for name in ("server.py", "tools.py"):
        head = [
            ln
            for ln in (BUNDLE / name).read_text().splitlines()
            if ln.startswith(("import ", "from "))
        ]
        assert not any(m in ln for ln in head for m in ("torch", "gr00t", "transformers")), name


EXAMPLE = BUNDLE / "examples" / "lego_place_with_gr00t"


def test_example_graph_builds_and_validates(tmp_path, skills_registry):
    from gap.testing import assert_graph_valid

    make = _load("gr00t_make_graph", EXAMPLE / "make_graph.py")
    out = make.build(tmp_path / "lego")
    assert_graph_valid(out, skill_registry=skills_registry)
    wf = json.loads((out / "workflow.json").read_text())
    assert wf["subgraphs"]["gr00t_sg"]["skill"] == "gr00t"
    for sg in wf["subgraphs"].values():
        for node in sg["nodes"].values():
            if node["type"] == "script":
                assert (out / node["script"]).is_file(), node["script"]
    # Every route a subgraph can return is mapped at the top level (lesson 9).
    for name, sg in wf["subgraphs"].items():
        exits = set(sg["exit"]["success_values"]) | {sg["on_error"]}
        assert exits == set(wf["conditional_edges"][name]["mapping"]), name


def test_example_plan_feeds_the_next_brick_glue(tmp_path):
    make = _load("gr00t_make_graph", EXAMPLE / "make_graph.py")
    plan = _load(
        "plan_order", SKILLS_ROOT / "skills/planning-brick-assembly-order/scripts/plan_order.py"
    )
    nxt = _load("next_brick", EXAMPLE / "scripts/next_brick.py")
    planned = plan.run(FakeContext({}), steps_json=json.dumps(make.DEFAULT_BUILD))
    assert planned["route"] == "sequence_ready", planned["detail"]
    out = nxt.run(FakeContext({}), sequence_json=planned["sequence_json"])
    assert out["part_name"] == "red 2x2 DUPLO brick" and out["remaining"] == 0
    assert out["cell"]["x"] == 4 and out["brick_studs"] == [2, 2]
    with pytest.raises(ValueError, match="nothing left"):
        nxt.run(FakeContext({}), sequence_json=planned["sequence_json"], done=1)


def test_example_gr00t_step_marks_the_brick_and_seat_and_runs_a_grasp_release_cycle(server, g, srv):
    class Cycle(srv.FakeBackend):
        """Closes the left jaw on its first call, opens it on the second."""

        def act(self, req):
            out = super().act(req)
            target = 0.0 if self.calls == 1 else 1.0
            out["actions"][:, 6] = np.concatenate(
                [np.linspace(req.joints[6], target, 6), np.full(10, target)]
            )
            return out

    step = _load("place_with_gr00t", EXAMPLE / "scripts/gr00t_sg/place_with_gr00t.py")
    robot = Robot()
    robot_ctx = robot.ctx()
    _, httpd, url = _serve(srv, Cycle())
    os.environ["GAP_GR00T_URL"] = url
    g._clients.clear()
    ctx = FakeContext(
        {
            "robot.get_observation": lambda **_: robot.observation(),
            "gr00t.marks_from_points": lambda **kw: g.marks_from_points(**kw),
            "gr00t.run_skill": lambda **kw: g.run_skill(robot_ctx, **kw),
        }
    )
    try:
        out = step.run(
            ctx,
            target_obb={"center": {"x": 0.0, "y": 0.0, "z": 0.0}},
            seat_position={"x": 0.1, "y": 0.0, "z": 0.47},
            part_name="red 2x2 DUPLO brick",
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        g._clients.clear()
    assert out["route"] == "succeeded"
    assert [m["kind"] for m in out["marks"]] == ["grasp", "release"]
    assert out["marks"][1]["uv"] == [104.0, 48.0]  # 0.5 m below the camera, 0.1 m along +x
    assert robot.jaw_calls == [(0, 0.0), (0, 1.0)]
    call = ctx.calls_to("gr00t.run_skill")[0]
    assert call.kwargs["success_check"] == "gripper_cycle"
    assert "red 2x2 DUPLO brick" in call.kwargs["instruction"]
