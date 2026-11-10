"""picking-an-object-with-a-learned-policy with the station and the policy faked: the policy's claim goes
through roboskillhub.vla.run_policy and the exit comes from verify_pick (jaw + lift), refusals before
anything moves, and a timeout."""

from __future__ import annotations

import time

import numpy as np
import pytest

from roboskillhub import vla

BUNDLE = "picking-an-object-with-a-learned-policy"
HOME = np.array([0.0, 0.5, 0.6, -0.3, 0.0, 0.0])


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["pick_with_policy"].module


class Station:
    """Two arms; the left TCP rises with j2. A cube in the left jaws when ``object_present``: the jaw
    stops on it at opening 0.3 and the cube rides along; without it the jaw closes to 0."""

    def __init__(self, object_present=True, jaw=1.0):
        self.q = {"left": HOME.copy(), "right": HOME.copy()}
        self.grip = {"left": jaw, "right": 1.0}
        self.object_present = object_present
        self.held = False
        self.calls = []

    def tcp(self, arm):
        return [0.3, 0.1 if arm == "left" else -0.1, 0.05 + 0.1 * (self.q[arm][1] - HOME[1])]

    def tool(self, name, **kw):
        self.calls.append(name)
        if name == "robot.get_observation":
            t = time.time()
            return {
                "timestamp": t,
                "cameras": [
                    {"name": n, "rgb": np.zeros((24, 32, 3), np.uint8), "timestamp": t}
                    for n in ("top_cam", "wrist_left", "wrist_right")
                ],
                "arms": [
                    {
                        "joint_state": {"positions": self.q[a].copy()},
                        "gripper_fraction": self.grip[a],
                        "ee_pose": [*self.tcp(a), 1, 0, 0, 0],
                        "timestamp": t,
                    }
                    for a in ("left", "right")
                ],
            }
        if name == "robot.stream_dual":
            for arm_id, wps in kw["tracks"].items():
                self.q[("left", "right")[int(arm_id)]] = np.asarray(wps[-1])
            return {}
        if name == "robot.close_gripper":
            arm = ("left", "right")[kw.get("arm_id", 0)]
            self.held = arm == "left" and self.object_present
            self.grip[arm] = 0.3 if self.held else 0.0
            return {}
        if name == "robot.open_gripper":
            self.grip[("left", "right")[kw.get("arm_id", 0)]] = 1.0
            self.held = False
            return {}
        if name == "sim.object_position":
            tcp = self.tcp("left")
            return {"position": [tcp[0], tcp[1], tcp[2] - 0.02] if self.held else [0.3, 0.1, 0.03]}
        raise RuntimeError(f"unknown tool {name}")


class Policy:
    """Closes the left jaw where it is, then lifts (j2 +10 mrad per row). Model frame = station here."""

    url = "fake://policy"

    def __init__(self, profile, fail=False):
        self.profile, self.fail = profile, fail
        self.marks = []
        self.instructions = []

    def info(self):
        if self.fail:
            raise vla.PolicyServerError("no server")
        return {"protocol": "vla-yam/1", "profile": self.profile.name, "backend": "fake"}

    def act(self, inp, instruction):
        self.instructions.append(instruction)
        s = np.asarray(inp["state"])
        a = np.repeat(s[None], 30, axis=0)
        if s[6] > 0.5:
            a[:, 6] = 0.0
        else:
            a[:, 1] += 0.01 * np.arange(1, 31)
        return {"actions": a, "frame": "model"}


def _run(script, station, **kw):
    pol = kw.pop("policy", None) or Policy(script.resolve_profile(None, None))
    args = {
        "object_ref": {"id": "cube_1", "text": "red cube"},
        "arm": "left",
        "object_position_tool": "sim.object_position",
        "lift_source": "sim_truth",
        "client": pol,
    }
    args.update(kw)
    return script.run(station, **args), pol


def test_picked_needs_the_verifier_not_the_claim(script):
    out, pol = _run(script, Station())
    assert out["route"] == "picked" and out["held_object"] == "red cube", out["reason"]
    v = out["evidence"]["verdict"]
    assert v["votes"]["gripper"] == "held" and v["votes"]["lift"] == "held" and v["privileged"]
    assert v["claim"] == "picked" and v["claim_agrees"] is True
    assert pol.instructions[0] == "pick up the red cube with the left arm"
    assert out["evidence"]["run"]["route"] == "stopped" and out["evidence"]["policy"]["profile"]


def test_policy_claims_a_pick_on_an_empty_jaw_and_the_verifier_says_missed(script):
    out, _ = _run(script, Station(object_present=False))
    v = out["evidence"]["verdict"]
    assert out["route"] == "missed" and out["held_object"] == ""
    assert v["claim"] == "picked" and v["claim_agrees"] is False
    assert v["votes"]["gripper"] == "empty" and v["votes"]["lift"] == "empty"


def test_unavailable_server_is_refused_without_motion(script):
    st = Station()
    out, _ = _run(script, st, policy=Policy(script.resolve_profile(None, None), fail=True))
    assert out["route"] == "refused" and "no server" in out["reason"]
    assert "robot.stream_dual" not in st.calls


def test_closed_jaw_fails_the_precondition(script):
    st = Station(jaw=0.0)
    out, _ = _run(script, st)
    assert out["route"] == "refused" and "gripper_open" in out["reason"]
    assert "robot.stream_dual" not in st.calls


def test_timeout_and_string_object_ref(script):
    out, pol = _run(script, Station(), object_ref="blue mug", arm="auto", timeout_s=0.0)
    assert out["route"] == "timeout"
    assert pol.instructions == [] or pol.instructions[0] == "pick up the blue mug"


def test_profiles_resolve_by_name_with_station(script):
    p = script.resolve_profile("yam-pi05-molmoact2", "real")
    assert p.backend == "pi05" and p.station == "real"
    with pytest.raises(vla.ProfileError):
        script.resolve_profile("no-such-profile", None)
