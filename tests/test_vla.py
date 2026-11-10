"""roboskillhub.vla with the robot and the model faked: profiles, observations, the chunk executor's
frame / gripper / delta handling, refusals, segment cuts, resampling and the closed loop."""

from __future__ import annotations

import math

import numpy as np
import pytest

from roboskillhub import vla
from roboskillhub.vla import executor as ex

PROFILE = {
    "profile": "test-yam",
    "checkpoint": {"repo": "x/y", "revision": "abc"},
    "cameras": ["top", "left_wrist", "right_wrist"],
    "state": ["left.j1..j6", "left.gripper", "right.j1..j6", "right.gripper"],
    "joint_offset_rad": {
        "left": [math.pi / 2, 0, 0, 0, 0, 0.1],
        "right": [math.pi / 2, 0, 0, 0, 0, 0],
    },
    "gripper": {"station_open": 1.0, "station_closed": 0.0, "model_open": 0.0, "model_closed": 0.8},
    "action": {"type": "abs_joint", "dim": 14, "horizon": 30, "execute_steps": 25, "hz": 30},
}


def _profile(**over):
    raw = {k: (dict(v) if isinstance(v, dict) else v) for k, v in PROFILE.items()}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(raw.get(k), dict):
            raw[k].update(v)
        else:
            raw[k] = v
    return vla.load_profile(raw)


Q0 = np.array([math.pi / 2, 0.2, 0.3, 0.0, 0.1, 0.0])


class Ctx:
    """A two-arm station: moves land where commanded, jaws follow commands, tools optional."""

    def __init__(self, tools=("robot.stream_dual",), clock=None, fk_z=None):
        self.q = {"left": Q0.copy(), "right": Q0.copy()}
        self.grip = {"left": 1.0, "right": 1.0}
        self.tools = set(tools) | {
            "robot.get_observation",
            "robot.open_gripper",
            "robot.close_gripper",
        }
        if fk_z is not None:
            self.tools.add("robot.forward_kinematics")
        self.fk_z = fk_z
        self.calls = []
        self.t = 100.0
        self.clock = clock or (lambda: self.t)

    def tool(self, name, **kw):
        self.calls.append((name, kw))
        if name not in self.tools:
            raise RuntimeError(f"unknown tool {name}")
        if name == "robot.get_observation":
            return {
                "cameras": [
                    {"name": n, "rgb": np.full((8, 8, 3), i, np.uint8), "intrinsics": np.eye(3)}
                    for i, n in enumerate(["wrist_right", "top_cam", "wrist_left"])
                ],
                "arms": [
                    {
                        "joint_state": {"positions": self.q[a].copy()},
                        "gripper_fraction": self.grip[a],
                    }
                    for a in ("left", "right")
                ],
            }
        if name == "robot.stream_dual":
            for arm_id, wps in kw["tracks"].items():
                self.q[("left", "right")[int(arm_id)]] = np.asarray(wps[-1])
            return {}
        if name == "robot.stream_joints":
            for arm, wps in kw["tracks"].items():
                self.q[arm] = np.asarray(wps[-1][:6])
                self.grip[arm] = float(wps[-1][6])
            return {}
        if name in ("robot.open_gripper", "robot.close_gripper"):
            arm = ("left", "right")[kw.get("arm_id", 0)]
            self.grip[arm] = 1.0 if name == "robot.open_gripper" else 0.0
            return {}
        if name == "robot.forward_kinematics":
            return {
                "poses": [
                    {"position": {"x": 0, "y": 0, "z": self.fk_z(q)}} for q in kw["joint_configs"]
                ]
            }
        raise AssertionError(name)

    def obs(self):
        return vla.get_observation(self, clock=self.clock)

    def names(self):
        return [c[0] for c in self.calls]


def _chunk(p, obs, rows=10, dq=0.0, grip=None):
    """Model-frame chunk that holds the measured pose (plus a ramp of dq per row on j2)."""
    state = vla.model_input(obs, p, now=obs["t"])["state"]
    a = np.repeat(state[None], rows, axis=0)
    a[:, 1] += dq * np.arange(1, rows + 1)
    a[:, 8] += dq * np.arange(1, rows + 1)
    if grip is not None:
        a[:, 6] = grip
    return a


# -- profile -------------------------------------------------------------------------------------


def test_profile_expands_state_and_validates():
    p = _profile()
    assert [s.name for s in p.state][:7] == [*(f"left.j{i}" for i in range(1, 7)), "left.gripper"]
    assert p.action.layout == p.state and p.arms == ("left", "right")
    assert p.arm_columns("right") == ([7, 8, 9, 10, 11, 12], 13)


def test_profile_lists_every_problem():
    bad = {
        **PROFILE,
        "cameras": ["top", "front"],
        "gripper": {
            "station_open": 1,
            "station_closed": 0,
            "model_open": "<measure>",
            "model_closed": 0,
        },
        "action": {"type": "eef", "dim": 12, "horizon": 5, "execute_steps": 8, "hz": 30},
    }
    with pytest.raises(vla.ProfileError) as e:
        vla.load_profile(bad)
    text = " | ".join(e.value.problems)
    for needle in ("front", "model_open", "action.type", "execute_steps 8", "action.dim 12"):
        assert needle in text


def test_profile_from_yaml_text():
    p = vla.load_profile(
        "profile: y\ncameras: [top]\nstate: [left.j1..j6, left.gripper]\n"
        "gripper: {station_open: 1, station_closed: 0, model_open: 1, model_closed: 0}\n"
        "action: {type: delta_joint, dim: 7, horizon: 4, execute_steps: 4, hz: 10}\n"
    )
    assert p.arms == ("left",) and p.action.type == "delta_joint"


def test_offsets_and_gripper_round_trip():
    p = _profile()
    rng = np.random.default_rng(0)
    station = rng.uniform(-1, 1, (5, 14))
    station[:, [6, 13]] = rng.uniform(0, 1, (5, 2))
    model = p.to_model(station)
    assert np.allclose(model[:, 0], station[:, 0] - math.pi / 2)
    assert np.allclose(model[:, 5], station[:, 5] - 0.1)
    assert np.allclose(p.to_station(model), station)


def test_gripper_mapping_is_inverted_linear():
    g = _profile().gripper
    assert g.to_model(1.0) == pytest.approx(0.0) and g.to_model(0.0) == pytest.approx(0.8)
    assert g.to_station(0.4) == pytest.approx(0.5)
    assert g.opening(0.25) == pytest.approx(0.25)


# -- observation ---------------------------------------------------------------------------------


def test_observation_roles_order_and_state():
    p = _profile()
    ctx = Ctx()
    obs = ctx.obs()
    assert set(obs["cameras"]) == {"top", "left_wrist", "right_wrist"}
    assert obs["cameras"]["right_wrist"]["rgb"][0, 0, 0] == 0
    inp = vla.model_input(obs, p, now=obs["t"])
    assert inp["image_order"] == ["top", "left_wrist", "right_wrist"]
    assert [int(v[0, 0, 0]) for v in inp["images"].values()] == [1, 2, 0]
    assert inp["state"][0] == pytest.approx(0.0)  # pi/2 station base is the model's 0
    assert inp["state"][6] == pytest.approx(0.0)  # station open -> model_open


def test_observation_staleness_and_missing_role():
    p = _profile()
    raw = Ctx().tool("robot.get_observation")
    raw["t"] = 10.0
    raw["cameras"][1]["t"] = 9.8
    obs = vla.from_connector(raw)
    with pytest.raises(vla.ObservationError, match="camera top is 0.200 s older"):
        vla.model_input(obs, p)
    raw["cameras"][1]["t"] = 10.0
    with pytest.raises(vla.ObservationError, match="observation is 1.000 s old"):
        vla.model_input(vla.from_connector(raw), p, now=11.0)
    del raw["cameras"][0]
    with pytest.raises(vla.ObservationError, match="no right_wrist camera"):
        vla.model_input(vla.from_connector(raw), p, now=10.0)


# -- executor: interpretation --------------------------------------------------------------------


def test_chunk_comes_back_in_station_frame_and_units():
    p = _profile()
    ctx = Ctx()
    obs = ctx.obs()
    a = _chunk(p, obs, rows=5, dq=0.01, grip=0.0)  # model 0.0 = open
    rows = ex.interpret(a, p, obs)
    assert np.allclose(rows[:, 0], math.pi / 2)
    assert np.allclose(rows[:, 6], 1.0)
    assert np.allclose(rows[:, 1], 0.2 + 0.01 * np.arange(1, 6))
    same = ex.interpret(rows, p, obs, frame="station")
    assert np.allclose(same, rows)


def test_delta_actions_obs_and_step_refs():
    for ref, expect in (("obs", [0.25, 0.25]), ("step", [0.25, 0.30])):
        p = _profile(action={"type": "delta_joint", "delta_ref": ref, "execute_steps": 2})
        ctx = Ctx()
        obs = ctx.obs()
        d = np.zeros((2, 14))
        d[:, 1] = 0.05
        d[:, [6, 13]] = 0.0  # gripper stays absolute (model 0 = open)
        rows = ex.interpret(d, p, obs)
        assert np.allclose(rows[:, 1], expect)
        assert np.allclose(rows[:, 0], math.pi / 2) and np.allclose(rows[:, 6], 1.0)


def test_execute_steps_truncates():
    p = _profile(action={"execute_steps": 4})
    obs = Ctx().obs()
    assert len(ex.interpret(_chunk(p, obs, rows=30), p, obs)) == 4


# -- executor: refusals --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate, code",
    [
        (lambda a: a[:, :13], "bad_shape"),
        (lambda a: np.where(np.arange(14) == 3, np.nan, a), "non_finite"),
        (lambda a: a + np.eye(14)[5] * 2.5, "joint_limit"),
        (lambda a: a + (np.arange(len(a)) >= 3)[:, None] * np.eye(14)[2] * 0.5, "jump"),
        (lambda a: a + np.eye(14)[1] * 1.0, "first_gap"),
    ],
)
def test_refusals_carry_a_reason(mutate, code):
    p = _profile()
    ctx = Ctx()
    obs = ctx.obs()
    out = ex.execute_chunk(ctx, mutate(_chunk(p, obs)), p, obs, clock=ctx.clock)
    assert out["ok"] is False and out["code"] == code and out["reason"]
    assert "robot.stream_dual" not in ctx.names()


def test_stale_chunk_and_estop_refused():
    p = _profile()
    ctx = Ctx()
    obs = ctx.obs()
    a = _chunk(p, obs)
    ctx.t += 0.6
    out = ex.execute_chunk(ctx, a, p, obs, clock=ctx.clock)
    assert out["code"] == "stale_chunk" and "0.600 s" in out["reason"]
    ctx.t = obs["t"]
    out = ex.execute_chunk(ctx, a, p, obs, clock=ctx.clock, estop=lambda: True)
    assert out["code"] == "estop"
    assert ex.execute_chunk(ctx, a, p, {**obs, "estop": True}, clock=ctx.clock)["code"] == "estop"


def test_tcp_floor_by_fk():
    p = _profile(safety={"table_z": 0.0})
    low = Ctx(fk_z=lambda q: 0.01 if q[1] > 0.25 else 0.1)
    obs = low.obs()
    out = ex.execute_chunk(low, _chunk(p, obs, rows=10, dq=0.01), p, obs, clock=low.clock)
    assert out["code"] == "tcp_floor" and "14 mm" in out["reason"]
    ok = Ctx(fk_z=lambda q: 0.1)
    obs = ok.obs()
    out = ex.execute_chunk(ok, _chunk(p, obs, rows=10, dq=0.01), p, obs, clock=ok.clock)
    assert out["ok"] and "robot.forward_kinematics" in ok.names()
    assert not any("TCP floor" in n for n in out["notes"])


def test_unchecked_floor_is_noted():
    p = _profile()
    ctx = Ctx()
    obs = ctx.obs()
    out = ex.execute_chunk(ctx, _chunk(p, obs), p, obs, clock=ctx.clock)
    assert out["ok"] and any("TCP floor not checked (no table_z)" in n for n in out["notes"])


# -- executor: segments, resampling, transports --------------------------------------------------


def test_segments_cut_at_gripper_switches():
    p = _profile()
    obs = Ctx().obs()
    a = _chunk(p, obs, rows=10)
    a[3:, 6] = 0.8  # left closes at row 3 (model_closed)
    a[7:, 6] = 0.0  # and opens at row 7
    a[3:, 13] = 0.8  # right closes at row 3 too
    plan = ex.plan_chunk(a, p, obs, now=obs["t"])
    assert [s.rows for s in plan.segments] == [4, 4, 2]
    assert plan.segments[0].switches == {"left": True, "right": True}
    assert plan.segments[1].switches == {"left": False}
    assert plan.segments[2].switches == {}


def test_hysteresis_holds_jaw_inside_the_band():
    p = _profile()
    obs = Ctx().obs()
    a = _chunk(p, obs, rows=6)
    a[:, 6] = 0.4  # opening 0.5: neither side of the band
    plan = ex.plan_chunk(a, p, obs, now=obs["t"])
    assert len(plan.segments) == 1 and plan.segments[0].switches == {}


def test_resample_to_station_rate_and_velocity_clip():
    p = _profile()
    obs = Ctx().obs()
    slow = ex.plan_chunk(_chunk(p, obs, rows=6, dq=0.01), p, obs, now=obs["t"], station_hz=90)
    tr = slow.segments[0].tracks["left"]
    assert len(tr) == 18  # 3 waypoints per 30 Hz row
    assert np.allclose(tr[2::3, 1], 0.2 + 0.01 * np.arange(1, 7))
    assert slow.duration_s == pytest.approx(6 / 30)

    fast = ex.plan_chunk(_chunk(p, obs, rows=6, dq=0.2), p, obs, now=obs["t"], station_hz=30)
    tr = fast.segments[0].tracks["left"]
    v = np.abs(np.diff(np.vstack([obs["arms"]["left"]["joints"], tr[:, :6]]), axis=0)).max() * 30
    assert v <= 1.5 + 1e-9
    assert np.allclose(tr[-1, 1], 0.2 + 1.2)
    assert any("slowed" in n for n in fast.notes)


def test_first_target_ramped():
    p = _profile()
    obs = Ctx().obs()
    a = _chunk(p, obs, rows=3)
    a[:, 1] += 0.5
    plan = ex.plan_chunk(a, p, obs, now=obs["t"])
    tr = plan.segments[0].tracks["left"]
    assert len(tr) > 3 and any("ramped 0.500" in n for n in plan.notes)
    assert np.abs(np.diff(tr[:, 1])).max() * 30 <= 1.5 + 1e-9


def test_stream_dual_with_gripper_commands_and_requery():
    p = _profile()
    ctx = Ctx()
    obs = ctx.obs()
    a = _chunk(p, obs, rows=10)
    a[3:, 13] = 0.8
    out = ex.execute_chunk(ctx, a, p, obs, clock=ctx.clock)
    assert out["ok"] and out["transport"] == "stream_dual" and out["rows"] == 4
    assert out["jaw_events"] == [{"segment": 0, "arm": "right", "jaw": "close"}]
    names = ctx.names()
    assert "robot.stream_joints" in names  # probed first
    assert names[-2:] == ["robot.stream_dual", "robot.close_gripper"]
    assert ctx.calls[-1][1] == {"arm_id": 1} and ctx.grip["right"] == 0.0


def test_stream_joints_preferred_with_gripper_in_track():
    p = _profile(safety={"gripper_force_cap": 20.0})
    ctx = Ctx(tools=("robot.stream_joints",))
    obs = ctx.obs()
    a = _chunk(p, obs, rows=5, dq=0.01)
    out = ex.execute_chunk(ctx, a, p, obs, clock=ctx.clock, requery_after_switch=False)
    assert out["ok"] and out["transport"] == "stream_joints"
    name, kw = ctx.calls[-1]
    assert name == "robot.stream_joints" and kw["mode"] == "abs" and kw["hz"] == 30
    assert kw["gripper_force_cap"] == 20.0 and len(kw["tracks"]["left"][0]) == 7


def test_stream_joints_error_is_not_masked_by_fallback():
    p = _profile()

    class Refusing(Ctx):
        def tool(self, name, **kw):
            if name == "robot.stream_joints":
                raise RuntimeError("station refused: torque limit")
            return super().tool(name, **kw)

    ctx = Refusing()
    obs = ctx.obs()
    out = ex.execute_chunk(ctx, _chunk(p, obs), p, obs, clock=ctx.clock)
    assert out["code"] == "transport" and "torque limit" in out["reason"]
    assert "robot.stream_dual" not in ctx.names()


# -- loop ----------------------------------------------------------------------------------------


class Model:
    """Moves left j2 up by 0.01 rad per row and closes the left jaw after 12 rows."""

    def __init__(self, p):
        self.p, self.rows, self.inputs = p, 0, []

    def act(self, inp, instruction):
        self.inputs.append(inp)
        a = np.repeat(inp["state"][None], 10, axis=0)
        a[:, 1] += 0.01 * np.arange(1, 11)
        if self.rows >= 12:
            a[:, 6] = self.p.gripper.model_closed
        self.rows += 8
        return {"actions": a, "dt_ms": 5.0}


def test_run_policy_until_stop_check():
    p = _profile(action={"execute_steps": 8})
    ctx = Ctx()
    model = Model(p)
    res = vla.run_policy(
        model,
        p,
        "pick",
        max_steps=200,
        stop_check=lambda obs, info: {
            "stop": obs["arms"]["left"]["gripper"] < 0.5,
            "exit": "picked",
        },
        robot=ctx,
        clock=ctx.clock,
    )
    assert res["route"] == "stopped" and res["exit"] == "picked"
    assert res["jaw_events"][0]["arm"] == "left" and res["jaw_events"][0]["jaw"] == "close"
    assert res["chunks"] == 3 and res["steps"] == 17  # 8 + 8 + 1 (cut at the jaw switch)
    assert res["chunk_log"][0]["dt_ms"] == 5.0
    assert model.inputs[0]["image_order"] == list(p.cameras)


def test_run_policy_max_steps_and_refusal():
    p = _profile(action={"execute_steps": 8})
    ctx = Ctx()
    res = vla.run_policy(Model(p), p, "x", max_steps=10, robot=ctx, clock=ctx.clock)
    assert res["route"] == "max_steps" and res["steps"] == 10

    class Wild:
        def act(self, inp, instruction):
            return {"actions": np.full((10, 14), 3.0)}

    res = vla.run_policy(Wild(), p, "x", max_steps=10, robot=Ctx(), clock=ctx.clock)
    assert res["route"] == "refused" and res["code"] and res["detail"]
