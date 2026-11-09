"""tracking-gripper-opening-stall: a fake jaw that moves toward its command until something stops it."""

from __future__ import annotations

import pytest
from gap.testing import FakeContext

BUNDLE = "tracking-gripper-opening-stall"
SPAN = 0.094
GRIPPER = {"span_m": SPAN, "width_fit": {"source": "span", "at_open_m": SPAN, "at_closed_m": 0.0}}


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["track_opening"].module


class Jaw:
    """Gap in metres; moves ``speed`` per wait toward the command, stops at ``block_close`` /
    ``block_open``; reads back as a fraction (1.0 = open) like robot.get_gripper."""

    def __init__(self, gap=SPAN, speed=0.004, block_close=None, block_open=None, stuck=False, metric=False):
        self.gap = gap
        self.cmd = gap
        self.speed = speed
        self.block_close = block_close
        self.block_open = block_open
        self.stuck = stuck
        self.metric = metric
        self.calls = []
        self.waits = 0

    def _step(self):
        if self.stuck:
            return
        lo = self.block_close if self.block_close is not None else -1.0
        hi = self.block_open if self.block_open is not None else 9.0
        d = self.cmd - self.gap
        self.gap += max(-self.speed, min(self.speed, d))
        self.gap = min(max(self.gap, lo), hi)

    def set_grip(self, width_m, **kw):
        self.calls.append(("set_grip", width_m, kw))
        self.cmd = width_m
        return {"position": self.gap / SPAN}

    def close(self, **kw):
        self.calls.append(("close", kw))
        self.cmd = 0.0
        return {"position": self.gap / SPAN}

    def open(self, **kw):
        self.calls.append(("open", kw))
        self.cmd = SPAN
        return {"position": self.gap / SPAN}

    def wait(self, steps, **kw):
        self.waits += 1
        self._step()
        return {}

    def get(self, **kw):
        if self.metric:
            return {"width_m": self.gap}
        return {"position": self.gap / SPAN}

    def ctx(self, gripper=GRIPPER):
        return FakeContext(
            {
                "robot.describe_gripper": gripper,
                "robot.get_gripper": self.get,
                "robot.set_grip": self.set_grip,
                "robot.close_gripper": self.close,
                "robot.open_gripper": self.open,
                "robot.wait_steps": self.wait,
            }
        )


def test_closing_on_nothing_reaches_the_stop(script):
    jaw = Jaw()
    out = script.run(jaw.ctx(), close=True)
    assert out["route"] == "reached"
    assert out["actual_m"] <= 0.002 and out["object_width_m"] == -1.0
    assert jaw.calls[0][0] == "close"
    assert out["trace_m"][-1] == out["actual_m"]


def test_closing_on_a_part_stalls_with_its_width(script):
    jaw = Jaw(block_close=0.0265)
    out = script.run(jaw.ctx(), close=True, expected_width_m=0.025)
    assert out["route"] == "stalled_on_object"
    assert out["object_width_m"] == pytest.approx(0.0265, abs=1e-6)
    assert out["width_error_m"] == pytest.approx(0.0015, abs=1e-6)
    assert "closing stopped at 26.5 mm" in out["reason"]


def test_stall_needs_a_full_flat_window(script):
    jaw = Jaw(block_close=0.03, speed=0.01)
    out = script.run(jaw.ctx(), close=True, window=8)
    assert out["route"] == "stalled_on_object"
    flat = [v for v in out["trace_m"] if abs(v - 0.03) < 1e-9]
    assert len(flat) >= 8


def test_wrong_width_routes_unexpected_width(script):
    jaw = Jaw(block_close=0.0318)
    out = script.run(jaw.ctx(), close=True, expected_width_m=0.0158)
    assert out["route"] == "unexpected_width"
    assert out["width_error_m"] == pytest.approx(0.016, abs=1e-6)
    assert "expected 15.8" in out["reason"]


def test_partial_target_uses_set_grip_and_reaches(script):
    jaw = Jaw()
    out = script.run(jaw.ctx(), target_width_m=0.05)
    assert out["route"] == "reached" and abs(out["actual_m"] - 0.05) <= 0.002
    assert jaw.calls[0] == ("set_grip", 0.05, {})


def test_opening_blocked_is_jammed(script):
    jaw = Jaw(gap=0.02, block_open=0.035)
    out = script.run(jaw.ctx(), target_width_m=0.07)
    assert out["route"] == "stalled_jammed"
    assert "opening settled" in out["reason"] and "short of the target" in out["reason"]


def test_no_travel_is_jammed_not_an_object(script):
    jaw = Jaw(stuck=True)
    out = script.run(jaw.ctx(), close=True)
    assert out["route"] == "stalled_jammed"
    assert "did not move" in out["reason"]


def test_slow_jaw_runs_out_of_samples(script):
    jaw = Jaw(speed=0.0008)
    out = script.run(jaw.ctx(), close=True, max_samples=10)
    assert out["route"] == "still_moving" and out["samples"] == 10


def test_target_is_clipped_to_the_span_and_full_open_uses_open_gripper(script):
    jaw = Jaw(gap=0.01)
    out = script.run(jaw.ctx(), target_width_m=0.2)
    assert out["target_m"] == pytest.approx(SPAN)
    assert jaw.calls[0][0] == "open" and out["route"] == "reached"


def test_watch_only_does_not_command(script):
    jaw = Jaw(block_close=0.02)
    jaw.cmd = 0.0
    out = script.run(jaw.ctx(), close=True, command=False)
    assert not jaw.calls
    assert out["route"] == "stalled_on_object"


def test_metric_width_reply_is_used_directly(script):
    jaw = Jaw(block_close=0.012, metric=True)
    out = script.run(jaw.ctx(gripper={}), close=True)
    assert out["route"] == "stalled_on_object"
    assert out["object_width_m"] == pytest.approx(0.012)


def test_arm_id_is_passed_only_when_nonzero(script):
    jaw = Jaw()
    script.run(jaw.ctx(), target_width_m=0.05, arm_id=1)
    assert jaw.calls[0] == ("set_grip", 0.05, {"arm_id": 1})


def test_refused_command_routes_refused(script):
    def refuse(**kw):
        raise RuntimeError("gripper fault")

    jaw = Jaw()
    ctx = jaw.ctx()
    ctx = FakeContext(
        {
            "robot.describe_gripper": GRIPPER,
            "robot.get_gripper": jaw.get,
            "robot.close_gripper": refuse,
            "robot.wait_steps": jaw.wait,
        }
    )
    out = script.run(ctx, close=True)
    assert out["route"] == "refused" and "gripper fault" in out["reason"]


def test_soft_refusal_routes_refused(script):
    jaw = Jaw()
    ctx = FakeContext(
        {
            "robot.describe_gripper": GRIPPER,
            "robot.get_gripper": jaw.get,
            "robot.set_grip": {"success": False, "message": "busy"},
            "robot.wait_steps": jaw.wait,
        }
    )
    out = script.run(ctx, target_width_m=0.05)
    assert out["route"] == "refused" and "busy" in out["reason"]


def test_unreadable_gripper_is_unknown(script):
    ctx = FakeContext({"robot.describe_gripper": GRIPPER, "robot.get_gripper": {}})
    out = script.run(ctx, close=True)
    assert out["route"] == "unknown" and out["actual_m"] == -1.0


def test_without_wait_steps_it_polls_on_the_clock(script, monkeypatch):
    jaw = Jaw(block_close=0.03, speed=0.05)
    sleeps = []
    monkeypatch.setattr(script.time, "sleep", lambda s: (sleeps.append(s), jaw._step()))
    ctx = FakeContext(
        {
            "robot.describe_gripper": GRIPPER,
            "robot.get_gripper": jaw.get,
            "robot.close_gripper": jaw.close,
        }
    )
    out = script.run(ctx, close=True, poll_s=0.01)
    assert out["route"] == "stalled_on_object" and sleeps and set(sleeps) == {0.01}


def test_needs_a_target(script):
    with pytest.raises(ValueError):
        script.run(Jaw().ctx())


def test_tracker_matches_upstream_semantics(script):
    t = script.OpeningTracker(0.0, tolerance_m=0.002, progress_m=0.0005, window=4)
    assert t.status == "unknown"
    for v in (0.03, 0.0299, 0.0298, 0.0297):
        t.update(v)
    assert t.status == "stalled"  # 0.3 mm over the window <= 0.5 mm
    t.update(None)
    assert t.status == "unknown" and t.errors == []
    for v in (0.03, 0.028, 0.026, 0.024):
        t.update(v)
    assert t.status == "pending"
    t.update(0.0015)
    assert t.status == "reached"
