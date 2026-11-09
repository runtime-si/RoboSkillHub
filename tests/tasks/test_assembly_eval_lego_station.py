"""LEGO StationBackend logic on a fake 1-D physics (no simulator): guarded_move stops, holding, observations."""

from __future__ import annotations

import math

import pytest

from roboskillhub.tasks.assembly_eval.lego_geometry import q_rotate
from roboskillhub.tasks.assembly_eval.lego_station import LegoPhysicsStation
from roboskillhub.tasks.assembly_eval.lego_station_isaac import HandSpec


class FakePhysics:
    """Hand z follows the set-point with the steady lag of a spring-damper drive (z - z_cmd = -D v / K) until a
    rigid floor at ``floor_z``; below it the hand sinks 10 % of the remaining set-point travel (a stiff
    obstacle)."""

    name = "fake"

    def __init__(self, floor_z: float = -1.0, pad_contact: tuple[set, set] = (set(), set())):
        self.hand_spec = HandSpec()
        self.floor_z = floor_z
        self.contacts = pad_contact
        self.control_dt = 1.0 / 120.0
        self._t = 0.0
        self._cmd = (0.3, 0.0, 0.1, 0.0)
        self._open = 0.04
        self._z = 0.1
        self._zc = 0.1
        self._vz = 0.0
        self.objects = {
            "baseplate": {"position": [0.3, 0.0, 0.0], "quat_xyzw": [0, 0, 0, 1]},
            "brick_0": {"position": [0.2, -0.2, 0.0], "quat_xyzw": [0, 0, 0, 1]},
        }

    def reset(self, scene, *, breakage=False):
        return {"baseplate": "/p", "brick_0": "/b"}

    def describe(self):
        return {"name": "fake"}

    def step(self, hand_pose, opening):
        self._cmd, self._open = tuple(hand_pose), opening
        z_cmd = hand_pose[2]
        v_cmd = (z_cmd - self._zc) / self.control_dt
        self._zc = z_cmd
        h = self.hand_spec
        free = z_cmd - h.xyz_damping * v_cmd / h.xyz_stiffness
        if free >= self.floor_z:
            z, self._vz = free, v_cmd
        else:
            z, self._vz = self.floor_z - 0.1 * (self.floor_z - z_cmd), 0.1 * v_cmd
        self._z = z
        self._t += self.control_dt
        return self._t

    def hand(self):
        return (self._cmd[0], self._cmd[1], self._z, self._cmd[3]), self._open

    def hand_velocity(self):
        return (0.0, 0.0, self._vz)

    def drain_events(self):
        return []

    def pad_contacts(self):
        return {"gripper_pad_l": self.contacts[0], "gripper_pad_r": self.contacts[1]}

    def truth(self):
        return {"t": self._t, "objects": self.objects}

    def connections(self):
        return []

    def interface_forces(self):
        return {}


def station(**kw) -> LegoPhysicsStation:
    phys = FakePhysics(**kw)
    scene = {
        "plate_id": "baseplate",
        "feeder": [{"id": "brick_0"}],
        "hand_start": (0.3, 0.0, 0.1, 0.0),
    }
    return LegoPhysicsStation(phys, scene)


DOWN = {"frame_id": "world", "vector": [0.0, 0.0, -1.0]}


def test_describe_declares_signal_controller_and_limits():
    d = station().describe()
    sig = d["contact_signal"]
    assert sig["unit"] == "N" and sig["rate_hz"] == 120 and 0 < sig["max_latency_s"] <= 0.02
    # BrickSim snaps the brick to its seat: the force signal cannot show a press-fit
    assert sig["insertion_force_observable"] is False
    assert d["controller"]["kind"] in ("force", "admittance", "impedance", "hybrid")
    assert {"contact_force_feedback", "compliant_motion"} <= set(d["capabilities"])
    assert "NOT a YAM arm" in d["robot"] and "DexMate" in d["robot"]
    cd = station().contact_description()
    for k in (
        "source",
        "quantity",
        "unit",
        "frame_id",
        "rate_hz",
        "max_latency_s",
        "calibration_id",
    ):
        assert cd["contact_signal"][k]
    assert cd["limits"]["max_force_N"] == 10.0


def test_press_stops_on_force_at_the_obstacle():
    s = station(floor_z=0.095)
    r = s.guarded_move(
        axis=DOWN,
        max_travel_m=0.0105,
        speed_m_s=0.004,
        force_limit_N=4.0,
        hold_s=0.0,
        timeout_s=10.0,
    )
    assert r["stop_reason"] == "force_threshold" and r["stop_confirmed"]
    onset = next(x["travel_m"] for x in r["samples"] if x["force_N"] > 0.8)
    assert onset == pytest.approx(0.005, abs=3e-4)
    peak = max(x["force_N"] for x in r["samples"])
    assert 4.0 <= peak <= 1.25 * 4.0
    ts = [x["t_s"] for x in r["samples"]]
    assert all(b > a for a, b in zip(ts, ts[1:], strict=False))


def test_press_in_free_space_stops_on_travel_and_reads_no_force():
    s = station()
    r = s.guarded_move(
        axis=DOWN,
        max_travel_m=0.005,
        speed_m_s=0.004,
        force_limit_N=4.0,
        hold_s=0.0,
        timeout_s=10.0,
    )
    assert r["stop_reason"] == "travel_limit"
    assert r["samples"][-1]["travel_m"] == pytest.approx(0.005, abs=1e-6)
    assert max(abs(x["force_N"]) for x in r["samples"]) < 0.1


@pytest.mark.parametrize("vector", [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.1, 0.0, -1.0]])
def test_press_refuses_other_axes_and_moves_nothing(vector):
    s = station()
    r = s.guarded_move(
        axis={"frame_id": "world", "vector": vector},
        max_travel_m=0.005,
        speed_m_s=0.004,
        force_limit_N=4.0,
        hold_s=0.0,
        timeout_s=10.0,
    )
    assert r["stop_reason"] == "fault" and not r["command_issued"] and r["samples"] == []
    assert s.physics._z == pytest.approx(0.1)


def test_press_refuses_force_above_station_limit():
    r = station().guarded_move(
        axis=DOWN,
        max_travel_m=0.005,
        speed_m_s=0.004,
        force_limit_N=11.0,
        hold_s=0.0,
        timeout_s=10.0,
    )
    assert r["stop_reason"] == "fault" and not r["command_issued"]


def test_holding_needs_both_pads_on_the_same_brick_and_closed_jaws():
    s = station(pad_contact=({"brick_0"}, {"brick_0"}))
    s.physics._open = 0.015
    assert s.get_state(timeout_s=1.0)["gripper"]["holding"]
    s.physics.contacts = ({"brick_0"}, set())
    assert not s.get_state(timeout_s=1.0)["gripper"]["holding"]
    s.physics.contacts = ({"brick_0"}, {"brick_0"})
    s.physics._open = 0.04
    assert not s.get_state(timeout_s=1.0)["gripper"]["holding"]


def test_observations_are_privileged_truth_in_contract_frames():
    st = station().get_state(timeout_s=1.0)
    b = st["objects"]["brick_0"]
    assert b["privileged"] and b["source"] == "simulation_truth" and b["class"] == "lego_brick"
    x = q_rotate(b["quaternion_xyzw"], [1.0, 0.0, 0.0])
    assert x == pytest.approx(
        [0.0, 1.0, 0.0], abs=1e-9
    )  # contract x = BrickSim y (the 2-stud side)
    p = st["objects"]["baseplate"]
    assert p["position_m"][2] == pytest.approx(0.0032)
    assert st["objects"]["table"]["class"] == "support"
    assert math.isclose(st["timestamp_s"], b["timestamp_s"])
