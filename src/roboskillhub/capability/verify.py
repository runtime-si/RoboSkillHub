"""The shared ``pick_object`` verifier: the exit comes from measurements, never from the
implementation's own claim.

Signals, each voting ``held``, ``empty``, ``lost`` (held, then not) or abstaining:

- gripper: tracking-gripper-opening-stall's route (or a raw gap trace run through its
  tracker), plus the gap re-read after the lift;
- scene: checking-pick-by-scene-diff's route, or its region numbers on a before/after pair
  (the VLM part of that skill is optional);
- lift: the object's position follows the TCP over a short lift, from observations or from
  simulator truth (flagged ``privileged`` in the evidence);
- wrist: an optional re-measure of the held part from the wrist camera.

A VLM verdict only breaks a tie between the measurements; it never overrides one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

from . import hub_skills

VERIFIER_VERSION = "verify_pick/1"

EXITS = ("picked", "missed", "slipped", "unreachable", "refused", "timeout")

Vote = Literal["held", "empty", "lost"]

MIN_LIFT_M = 0.01
"""TCP travel under which the lift check abstains: too short to tell following from resting."""

FOLLOW_RATIO = 0.7
"""Object travel along the TCP travel, as a share of it, at or over which the object follows."""

STAY_RATIO = 0.3
"""Share under which the object stayed behind."""

FOLLOW_RESIDUAL_M = 0.015
"""Largest object-vs-TCP displacement mismatch still counted as following (a part can swing
or settle a little in the jaws)."""

SLIP_GAP_DROP_M = 0.003
"""Gap closing by more than this after the lift, from where it stalled on the part, is a slip."""

MIN_AGREE = 2
"""Independent ``held`` votes required for ``picked`` when nothing votes against: one sensor
alone does not certify a pick (the false-``picked`` budget is 1 %)."""

GRIPPER_HELD = ("stalled_on_object",)
GRIPPER_EMPTY = ("reached",)
SCENE_HELD = ("changed_as_expected", "vacated")
SCENE_EMPTY = ("unchanged", "changed_otherwise")
WRIST_HELD = ("measured", "held", "seen")
WRIST_EMPTY = ("not_seen", "empty")


@dataclass
class PickObservation:
    """What the harness saw around one pick attempt. Every field is optional."""

    object_id: str | None = None
    preconditions: dict[str, bool] | None = None
    """``object_visible``, ``arm_free``, ``gripper_open`` as measured before the attempt."""
    elapsed_s: float | None = None
    """Wall (or sim) time the attempt took, measured by the harness."""
    timeout_s: float | None = None

    gripper: dict[str, Any] | None = None
    """tracking-gripper-opening-stall's output (``route``, ``actual_m``, ...)."""
    gripper_trace_m: Sequence[float] | None = None
    """Raw jaw gaps sampled while closing, when the skill did not run."""
    gripper_target_m: float | None = None
    expected_width_m: float | None = None
    gripper_after_lift_m: float | None = None

    scene: dict[str, Any] | None = None
    """checking-pick-by-scene-diff's output, or ``{"route": "vacated"|"unchanged"}``."""
    before_rgb: Any = None
    after_rgb: Any = None
    source_box_px: Sequence[float] | None = None

    lift: Sequence[dict[str, Sequence[float]]] | None = None
    """Samples ``{"tcp": [x, y, z], "object": [x, y, z]}`` over the lift, first one before it."""
    lift_source: Literal["observation", "sim_truth"] = "observation"

    wrist: dict[str, Any] | None = None
    """``{"held": bool}`` or a wrist skill's output with ``route``."""
    vlm: dict[str, Any] | None = None
    """``{"held": bool, "reason": str}``; tie-break only."""


@dataclass
class PickVerdict:
    exit: str
    reason: str
    uncertain: bool = False
    claim: str | None = None
    claim_agrees: bool | None = None
    privileged: bool = False
    votes: dict[str, str | None] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    verifier: str = VERIFIER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _vec(v: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = (float(a) for a in v)
    return x, y, z


def _sub(a, b):
    return tuple(p - q for p, q in zip(a, b))


def _dot(a, b) -> float:
    return sum(p * q for p, q in zip(a, b))


def _norm(a) -> float:
    return math.sqrt(_dot(a, a))


def classify_trace(
    trace_m: Sequence[float], target_m: float, expected_width_m: float | None = None
) -> dict[str, Any]:
    """Run a recorded closing trace through tracking-gripper-opening-stall's tracker and
    its stall split; returns the same ``route`` names the skill routes."""
    mod = hub_skills.load(*hub_skills.GRIPPER_STALL)
    tracker = mod.OpeningTracker(target_m)
    trace = [float(v) for v in trace_m]
    if not trace:
        return {"route": "unknown", "reason": "empty trace"}
    start = trace[0]
    for i, value in enumerate(trace):
        tracker.update(value)
        status = tracker.status
        if status == "reached":
            return {"route": "reached", "actual_m": value, "samples": i + 1}
        if status != "stalled":
            continue
        travel = abs(value - start)
        if travel < mod.MIN_TRAVEL_M or value < target_m or target_m >= start:
            return {"route": "stalled_jammed", "actual_m": value, "samples": i + 1}
        if (
            expected_width_m is not None
            and abs(value - expected_width_m) > mod.WIDTH_TOLERANCE_M
        ):
            return {"route": "unexpected_width", "actual_m": value, "samples": i + 1}
        return {"route": "stalled_on_object", "actual_m": value, "samples": i + 1}
    return {"route": "still_moving", "actual_m": trace[-1], "samples": len(trace)}


def gripper_vote(obs: PickObservation) -> tuple[Vote | None, dict[str, Any]]:
    out = dict(obs.gripper or {})
    if not out and obs.gripper_trace_m is not None and obs.gripper_target_m is not None:
        out = classify_trace(obs.gripper_trace_m, obs.gripper_target_m, obs.expected_width_m)
        out["from"] = "trace"
    route = out.get("route")
    ev: dict[str, Any] = {"route": route, **{k: out[k] for k in ("actual_m", "reason") if k in out}}
    if route is None:
        return None, {"reason": "no gripper reading"}
    vote: Vote | None = None
    if route in GRIPPER_HELD:
        vote = "held"
        stall = out.get("actual_m")
        after = obs.gripper_after_lift_m
        if after is not None and stall is not None and stall >= 0:
            ev["after_lift_m"] = after
            if float(stall) - float(after) > SLIP_GAP_DROP_M:
                vote = "lost"
                ev["reason"] = (
                    f"jaws closed {1e3 * (float(stall) - float(after)):.1f} mm further after the lift"
                )
    elif route in GRIPPER_EMPTY:
        vote = "empty"
    return vote, ev


def scene_numbers(before_rgb, after_rgb, box_px) -> dict[str, Any]:
    """checking-pick-by-scene-diff's region numbers (no VLM) on a before/after pair."""
    import numpy as np

    mod = hub_skills.load(*hub_skills.SCENE_DIFF)
    b, a = np.asarray(before_rgb)[..., :3], np.asarray(after_rgb)[..., :3]
    diff = mod.diff_map(b, a)
    region = mod.source_mask(diff.shape, box_px)
    share = float((diff > mod.DIFF_THRESHOLD)[region].mean())
    ratio = mod.region_contrast(a, region) / max(mod.region_contrast(b, region), 1e-6)
    ok, why = mod.region_verdict("vacated", share, ratio, mod.MIN_SOURCE_CHANGE)
    if ok:
        route = "vacated"
    elif share < mod.MIN_SOURCE_CHANGE:
        route = "unchanged"
    else:
        route = "uncertain"
    return {
        "route": route,
        "source_changed_fraction": round(share, 4),
        "source_contrast_ratio": round(ratio, 3),
        "reason": why,
    }


def scene_vote(obs: PickObservation) -> tuple[Vote | None, dict[str, Any]]:
    out = obs.scene
    if out is None and obs.before_rgb is not None and obs.after_rgb is not None:
        if obs.source_box_px is None:
            return None, {"reason": "before/after frames but no source_box_px"}
        out = scene_numbers(obs.before_rgb, obs.after_rgb, obs.source_box_px)
    if not out:
        return None, {"reason": "no scene check"}
    route = out.get("route")
    ev = {
        k: out[k]
        for k in ("route", "source_changed_fraction", "source_contrast_ratio", "reason")
        if k in out
    }
    if route in SCENE_HELD:
        return "held", ev
    if route in SCENE_EMPTY:
        return "empty", ev
    return None, ev


def lift_vote(obs: PickObservation) -> tuple[Vote | None, dict[str, Any]]:
    samples = list(obs.lift or [])
    if len(samples) < 2:
        return None, {"reason": "no lift samples"}
    tcp0, obj0 = _vec(samples[0]["tcp"]), _vec(samples[0]["object"])
    ratios: list[float] = []
    lifted = 0.0
    for s in samples[1:]:
        dt = _sub(_vec(s["tcp"]), tcp0)
        do = _sub(_vec(s["object"]), obj0)
        n = _norm(dt)
        lifted = max(lifted, n)
        if n >= MIN_LIFT_M:
            ratios.append(_dot(do, dt) / (n * n))
    dt = _sub(_vec(samples[-1]["tcp"]), tcp0)
    do = _sub(_vec(samples[-1]["object"]), obj0)
    ev: dict[str, Any] = {
        "source": obs.lift_source,
        "tcp_travel_m": round(_norm(dt), 4),
        "object_travel_m": round(_norm(do), 4),
        "residual_m": round(_norm(_sub(do, dt)), 4),
    }
    if _norm(dt) < MIN_LIFT_M:
        ev["reason"] = f"TCP moved {_norm(dt) * 1e3:.1f} mm, under {MIN_LIFT_M * 1e3:.0f} mm"
        return None, ev
    final = ratios[-1]
    ev["follow_ratio"] = round(final, 3)
    ev["max_follow_ratio"] = round(max(ratios), 3)
    if final >= FOLLOW_RATIO and ev["residual_m"] <= FOLLOW_RESIDUAL_M:
        return "held", ev
    if max(ratios) >= FOLLOW_RATIO:
        ev["reason"] = "the object followed the TCP, then fell behind"
        return "lost", ev
    if final < STAY_RATIO:
        ev["reason"] = "the object stayed where it was"
        return "empty", ev
    ev["reason"] = "the object moved partway with the TCP"
    return "lost", ev


def wrist_vote(obs: PickObservation) -> tuple[Vote | None, dict[str, Any]]:
    out = obs.wrist
    if not out:
        return None, {"reason": "no wrist re-measure"}
    ev = dict(out)
    if isinstance(out.get("held"), bool):
        return ("held" if out["held"] else "empty"), ev
    route = out.get("route")
    if route in WRIST_HELD:
        return "held", ev
    if route in WRIST_EMPTY:
        return "empty", ev
    return None, ev


def _precondition_failures(obs: PickObservation) -> list[str]:
    return [k for k, ok in (obs.preconditions or {}).items() if ok is False]


def _attempted(gripper_ev: dict[str, Any], votes: dict[str, str | None]) -> bool:
    """Whether anything physical happened: the jaws moved or a sensor saw the object held."""
    if any(v in ("held", "lost") for v in votes.values()):
        return True
    return gripper_ev.get("route") not in (None, "unknown", "refused", "stalled_jammed")


def verify_pick(obs: PickObservation, claim: str | None = None) -> PickVerdict:
    """Decide the ``pick_object`` exit from *obs*.

    *claim* is the exit the implementation reported. It is recorded and compared, and it is
    only taken for ``unreachable`` / ``refused`` when no physical attempt is visible; a claim
    of ``picked``, ``missed`` or ``slipped`` never decides anything.
    """
    signals = {
        "gripper": gripper_vote(obs),
        "lift": lift_vote(obs),
        "scene": scene_vote(obs),
        "wrist": wrist_vote(obs),
    }
    votes = {k: v for k, (v, _) in signals.items()}
    evidence: dict[str, Any] = {k: e for k, (_, e) in signals.items()}
    privileged = obs.lift_source == "sim_truth" and votes["lift"] is not None

    def done(exit_: str, reason: str, uncertain: bool = False) -> PickVerdict:
        return PickVerdict(
            exit=exit_,
            reason=reason,
            uncertain=uncertain,
            claim=claim,
            claim_agrees=None if claim is None else claim == exit_,
            privileged=privileged,
            votes=votes,
            evidence=evidence,
        )

    failed = _precondition_failures(obs)
    if failed:
        return done("refused", f"preconditions not met: {', '.join(failed)}")
    if (
        obs.elapsed_s is not None
        and obs.timeout_s is not None
        and obs.elapsed_s > obs.timeout_s
    ):
        return done("timeout", f"took {obs.elapsed_s:.1f} s, over the {obs.timeout_s:.0f} s limit")
    attempted = _attempted(evidence["gripper"], votes)
    if claim in ("unreachable", "refused") and not attempted:
        return done(claim, f"implementation reported {claim} and no attempt is visible")

    held = [k for k, v in votes.items() if v == "held"]
    empty = [k for k, v in votes.items() if v == "empty"]
    lost = [k for k, v in votes.items() if v == "lost"]
    if lost:
        return done("slipped", f"held, then lost ({', '.join(lost)})")
    if votes["gripper"] == "held" and (votes["lift"] == "empty" or votes["wrist"] == "empty"):
        why = "lift" if votes["lift"] == "empty" else "wrist"
        return done("slipped", f"the jaws stopped on the part but the {why} check lost it")
    if held and not empty and len(held) >= MIN_AGREE:
        return done("picked", f"{len(held)} independent checks agree ({', '.join(held)})")
    if empty and not held:
        return done("missed", f"nothing held ({', '.join(empty)})")
    if not held and not empty:
        if claim in ("unreachable", "refused"):
            return done(claim, f"implementation reported {claim}; no measurement contradicts it")
        return done("missed", "no measurement says the object is held", uncertain=True)

    # One held vote alone, or held against empty: the VLM may break the tie, never override.
    vlm = obs.vlm or {}
    evidence["vlm"] = dict(vlm) if vlm else {"reason": "no VLM tie-break"}
    if isinstance(vlm.get("held"), bool):
        votes["vlm"] = "held" if vlm["held"] else "empty"
        if vlm["held"] and not empty:
            return done(
                "picked",
                f"one check ({', '.join(held)}) plus the VLM tie-break",
                uncertain=True,
            )
    return done(
        "missed",
        f"checks disagree or too few (held: {held or '-'}, empty: {empty or '-'})",
        uncertain=True,
    )
