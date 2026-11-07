# Adapted from VoLoAgent (https://github.com/NVlabs/VoLoAgent),
# vlm_orchestrator/failure_handlers/signal_detector.py (FailureSignalDetector,
# SignalConfig), Apache-2.0, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# Changes: one classification per call over a window of proprioception samples
# (passed in from earlier nodes and/or sampled here) instead of a per-step
# update loop; heights are offsets over the work surface plus the fingertip
# reach instead of absolute robolab / LIBERO z values; the VLA's gripper command
# is replaced by the measured opening (a jaw "closed" once it is a few percent
# of its stroke off the open stop; a sample's own commanded_closed wins), and
# the opening is read against robot.describe_gripper's stroke so a jaw closed to
# its stop (closed on nothing) and a held object closing to the stop with the
# hand still shut (a slip) are their own signals; "frozen" counts as a stall
# only while the arm is short of a commanded target (tracking lag) or the caller
# says motion was expected, because a blocking connector's arm is always still
# when a node runs; a frozen, elevated, holding hand is stuck_mid_air instead of
# stall; the three-way done / progress / failure status becomes one exit per
# class (ok carries upstream's done / progress label in `state`).
"""Classify the last few moments of a manipulation from proprioception alone:
stall / never_gripped / fumble / drop / stuck_mid_air / ok."""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

WINDOW = 16
"""Samples in the rolling window (upstream ``window_size``)."""

STALL_Z_RANGE = 0.03
"""Tool-centre z range [m] under which the window "barely moves" (upstream
``stall_z_range``, robolab and LIBERO alike)."""

STALL_SPEED = 0.002
"""Mean tool-centre travel per sample [m] under which the window is "frozen"
(upstream ``stall_speed``, per VLA action chunk). A sample here is one control
segment or one node boundary; segments of about a centimetre keep the two
comparable."""

CARRY_DZ = 0.03
"""Tool-centre height over table level above which a closed hand is carrying
[m]: LIBERO's carry_z - table_z (0.85 - 0.82). robolab's is 0, which calls a
hand closed at grasp height "carrying"."""

DROP_DZ = 0.06
"""Height over table level above which an opening is a drop [m]: robolab's
drop_z - table_z (0.28 - 0.22)."""

PLACE_DZ = 0.03
"""Height over table level below which an open hand has placed [m]: robolab's
place_z - table_z (0.25 - 0.22). Also the "low" bound of the fumble-at-table
rule (upstream compares with table_z itself, which a tool centre held up by
its fingertips never goes under)."""

NEVER_CLOSED_FRACTION = 0.05 / (np.pi / 2)
"""Fraction of the stroke the jaw must leave the open stop to count as closed:
upstream's grip_never_closed_width (-0.05) over its fully closed finger sum
(-pi/2), about 3 %. On the linear_4310's 94 mm stroke, 3.0 mm."""

MIN_CLOSED_TRAVEL = 0.002
"""Floor under that travel [m], for a short-stroke hand."""

EMPTY_MARGIN = 0.003
"""A closed jaw within this of its closed stop has nothing between the pads [m]
(verifying-grasps' verify_grip: an empty linear_4310 close measured 1.8 mm)."""

LAG_TOLERANCE = 0.015
"""Distance from the commanded tool-centre target [m] beyond which a still arm
is stuck, not idle: no tighter than a real YAM station's 12 mm per-move bound
(a servo arrives within 4 mm)."""

MIN_CHUNKS_STALL = 10
MIN_CHUNKS_FUMBLE = 15
MIN_CHUNKS_NEVER_GRIPPED = 30
"""Upstream's gates on how long the step has run before each failure may fire,
in the caller's own units (``chunks_elapsed``; 999 = judge everything)."""

FUMBLE_TRANSITIONS_HIGH = 6
FUMBLE_TRANSITIONS_LOW = 4
"""Grip transitions for a fumble at table level / anywhere (upstream)."""

PRE_OPEN_SAMPLES = 15
DESCENT_BEFORE_OPEN = 0.005
DESCENDING_FRACTION = 0.4
OPEN_WAS_CLOSED = 20
"""Upstream's placement rules: the tool centre came down more than 5 mm over
the 15 samples before the last opening, descending in more than 40 % of them;
or the hand opened after more than 20 consecutive closed samples."""

GRIPPER_SPAN = 0.094
"""Stroke [m] when robot.describe_gripper gives none: the linear_4310's 94 mm
(a real YAM station's measurement)."""

ROUTES = ("ok", "stall", "never_gripped", "fumble", "drop", "stuck_mid_air")


class Output(TypedDict):
    route: str
    state: str
    reason: str
    confidence: float
    features: dict[str, Any]
    history: list[dict[str, Any]]
    samples: int


def _vec(v: Any) -> np.ndarray | None:
    if v is None:
        return None
    if isinstance(v, dict):
        if "position" in v:
            v = v["position"]
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    a = np.asarray(v, dtype=float).reshape(-1)
    return a[:3] if a.size >= 3 else None


def _xyz(v: np.ndarray | None) -> dict[str, float] | None:
    return None if v is None else {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def sample_now(
    ctx: NodeContext,
    arm_id: int = 0,
    target_position: Any = None,
    commanded_closed: bool | None = None,
) -> dict[str, Any]:
    """One proprioception sample from robot.get_observation: the tool centre,
    the gripper's open fraction (1 = open) and the arm joints, with the
    commanded target and gripper state when the caller knows them."""
    obs = ctx.tool("robot.get_observation")
    arms = obs.get("arms") or []
    if len(arms) <= arm_id:
        raise ValueError(f"the observation has no arm {arm_id}")
    arm = arms[arm_id]
    joints = (arm.get("joint_state") or {}).get("positions")
    out: dict[str, Any] = {
        "ee_position": _xyz(_vec(arm["ee_pose"])),
        "gripper_fraction": float(arm["gripper_fraction"]),
        "joints": [float(q) for q in np.asarray(joints, dtype=float).reshape(-1)]
        if joints is not None
        else None,
    }
    target = _vec(target_position)
    if target is not None:
        out["target_position"] = _xyz(target)
    if commanded_closed is not None:
        out["commanded_closed"] = bool(commanded_closed)
    return out


def _gripper(ctx: NodeContext, on_arm: dict[str, Any]) -> tuple[float, float, float, str]:
    """(open width, closed stop, fingertip reach) [m] and where they came from."""
    try:
        g = ctx.tool("robot.describe_gripper", **on_arm) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("[proprio] describe_gripper failed (%s); assuming a linear_4310", exc)
        return GRIPPER_SPAN, 0.0, 0.0102, "default linear_4310"
    span = float(g.get("span_m") or GRIPPER_SPAN)
    fit = g.get("width_fit") or {}
    stop = float(fit.get("at_closed_m") or g.get("min_grasp_width_m") or 0.0)
    reach = float((g.get("finger") or {}).get("reach_m") or 0.0)
    return span, stop, reach, str(g.get("gripper") or "described")


def _surface(ctx: NodeContext, surface_z: float | None, on_arm: dict[str, Any]) -> float:
    if surface_z is not None:
        return float(surface_z)
    ws = ctx.tool("robot.describe_workspace", **on_arm) or {}
    if ws.get("surface_z") is None:
        raise ValueError("no surface_z given and robot.describe_workspace has none")
    return float(ws["surface_z"])


def features(
    window: list[dict[str, Any]],
    span: float,
    stop: float,
) -> dict[str, Any]:
    """Upstream's SignalFeatures over the window, plus the YAM jaw and lag signals."""
    ee = np.stack([_vec(s["ee_position"]) for s in window])
    z = ee[:, 2]
    opening = np.array([float(s["gripper_fraction"]) * span for s in window])
    travel = max(NEVER_CLOSED_FRACTION * span, MIN_CLOSED_TRAVEL)
    closed = np.array(
        [
            bool(s["commanded_closed"])
            if s.get("commanded_closed") is not None
            else bool(w < span - travel)
            for s, w in zip(window, opening, strict=True)
        ],
        dtype=int,
    )
    empty = (closed == 1) & (opening <= stop + EMPTY_MARGIN)
    held = (closed == 1) & ~empty

    run = max_consec = 0
    for c in closed:
        run = run + 1 if c else 0
        max_consec = max(max_consec, run)

    speed = np.linalg.norm(np.diff(ee, axis=0), axis=1)
    joints = [s.get("joints") for s in window]
    joint_speed = 0.0
    if all(j for j in joints) and len({len(j) for j in joints}) == 1:
        q = np.asarray(joints, dtype=float)
        joint_speed = float(np.linalg.norm(np.diff(q, axis=0), axis=1).mean())

    descent = z_at_open = descending = 0.0
    opens = np.where(np.diff(closed) == -1)[0]
    if len(opens):
        i = int(opens[-1])
        z_at_open = float(z[i])
        pre = z[max(0, i - PRE_OPEN_SAMPLES) : i + 1]
        if len(pre) > 1:
            descent = float(pre[0] - pre[-1])
            descending = float(np.mean(np.diff(pre) < 0))

    # A held object lost with the hand still shut: held, then at the stop, closed throughout.
    slip_z = None
    for j in range(1, len(window)):
        if empty[j] and held[:j].any():
            i = int(np.where(held[:j])[0][-1])
            if closed[i : j + 1].all():
                slip_z = float(z[j])
                break

    lag = None
    target = _vec(window[-1].get("target_position"))
    if target is not None:
        lag = float(np.linalg.norm(target - ee[-1]))

    return {
        "n_samples": len(window),
        "n_grip_transitions": int(np.abs(np.diff(closed)).sum()),
        "grip_final_closed": bool(closed[-1]),
        "max_consecutive_closed": int(max_consec),
        "opening_final_m": float(opening[-1]),
        "opening_min_m": float(opening.min()),
        "held_final": bool(held[-1]),
        "empty_final": bool(empty[-1]),
        "ever_closed": bool(closed.any()),
        "ever_held": bool(held.any()),
        "slip_z": slip_z,
        "ee_z_final": float(z[-1]),
        "ee_z_max": float(z.max()),
        "ee_z_range": float(z.max() - z.min()),
        "mean_ee_speed": float(speed.mean()),
        "mean_joint_speed": joint_speed,
        "descent_before_open": descent,
        "ee_z_at_last_open": z_at_open,
        "descending_before_open": descending,
        "lag_final_m": lag,
    }


def classify(
    f: dict[str, Any],
    table_level: float,
    chunks_elapsed: int = 999,
    expect_motion: bool = False,
    carry_dz: float = CARRY_DZ,
    drop_dz: float = DROP_DZ,
    place_dz: float = PLACE_DZ,
    lag_tolerance: float = LAG_TOLERANCE,
) -> tuple[str, str, float, str]:
    """(route, state, confidence, reason): upstream's rule order, YAM signals added."""
    carry_z, drop_z, place_z = table_level + carry_dz, table_level + drop_dz, table_level + place_dz
    z = f["ee_z_final"]
    elevated = z > carry_z
    lag = f["lag_final_m"]
    motion_expected = (lag > lag_tolerance) if lag is not None else bool(expect_motion)
    lag_txt = f", {lag * 1e3:.0f} mm short of its target" if lag is not None else ""
    frozen = f["ee_z_range"] < STALL_Z_RANGE and f["mean_ee_speed"] < STALL_SPEED

    # Stall: frozen while the arm should be moving.
    if chunks_elapsed >= MIN_CHUNKS_STALL and frozen and motion_expected:
        if f["held_final"] and elevated:
            return (
                "stuck_mid_air",
                "stuck",
                0.7,
                f"holding at {z - table_level:.3f} m over table level but frozen{lag_txt}",
            )
        return (
            "stall",
            "stalled",
            0.9,
            f"tool centre barely moving (z range {f['ee_z_range'] * 1e3:.1f} mm, "
            f"{f['mean_ee_speed'] * 1e3:.2f} mm/sample){lag_txt}",
        )

    # A held object lost while the hand stayed shut.
    if f["slip_z"] is not None and f["slip_z"] > carry_z and f["empty_final"]:
        return (
            "drop",
            "dropped",
            0.75,
            f"the jaw closed to its stop while shut at {f['slip_z'] - table_level:.3f} m "
            "over table level: the held object slipped out",
        )

    # Never gripped: closed on nothing, or never closed at all.
    if f["empty_final"] and not f["ever_held"]:
        return (
            "never_gripped",
            "empty",
            0.8,
            f"the jaw closed to its stop ({f['opening_final_m'] * 1e3:.1f} mm) without holding anything",
        )
    if (
        chunks_elapsed >= MIN_CHUNKS_NEVER_GRIPPED
        and f["n_grip_transitions"] == 0
        and not f["ever_closed"]
    ):
        return ("never_gripped", "open", 0.8, "no grip transitions, the hand never closed")

    if (
        chunks_elapsed >= MIN_CHUNKS_FUMBLE
        and f["n_grip_transitions"] >= FUMBLE_TRANSITIONS_HIGH
        and z < place_z
    ):
        return (
            "fumble",
            "fumbling",
            0.85,
            f"{f['n_grip_transitions']} grip transitions at low height "
            f"({z - table_level:.3f} m over table level)",
        )

    # Holding, elevated.
    if f["grip_final_closed"] and elevated:
        if f["ee_z_range"] > STALL_Z_RANGE:
            return ("ok", "held", 0.85, f"holding, elevated {z - table_level:.3f} m, moving")
        if f["mean_ee_speed"] < STALL_SPEED and motion_expected:
            return ("stuck_mid_air", "stuck", 0.7, f"holding, elevated but frozen{lag_txt}")
        return (
            "ok",
            "held",
            0.6,
            f"holding, elevated {z - table_level:.3f} m, still and on target",
        )

    if chunks_elapsed >= MIN_CHUNKS_FUMBLE and f["n_grip_transitions"] >= FUMBLE_TRANSITIONS_LOW:
        return (
            "fumble",
            "fumbling",
            0.75,
            f"{f['n_grip_transitions']} grip transitions (repeated open/close)",
        )

    if not f["grip_final_closed"]:
        if (
            f["descent_before_open"] > DESCENT_BEFORE_OPEN
            and f["descending_before_open"] > DESCENDING_FRACTION
        ):
            return (
                "ok",
                "placed",
                0.7,
                f"descended {f['descent_before_open'] * 1e3:.0f} mm before opening",
            )
        if f["n_grip_transitions"] and f["ee_z_at_last_open"] > drop_z:
            return (
                "drop",
                "dropped",
                0.65,
                f"opened at {f['ee_z_at_last_open'] - table_level:.3f} m over table level (likely drop)",
            )
        if f["max_consecutive_closed"] > OPEN_WAS_CLOSED:
            return (
                "ok",
                "placed",
                0.5,
                f"open after {f['max_consecutive_closed']} consecutive closed samples",
            )
        if z < place_z:
            return ("ok", "placed", 0.4, f"open, low ({z - table_level:.3f} m over table level)")
        return ("ok", "open", 0.3, "open, no strong signal")

    return ("ok", "approaching", 0.5, f"closed, {z - table_level:.3f} m over table level")


def run(
    ctx: NodeContext,
    history: list[dict[str, Any]] | None = None,
    n_samples: int = 4,
    sample_interval_s: float = 0.05,
    target_position: Any = None,
    commanded_closed: bool | None = None,
    surface_z: float | None = None,
    window: int = WINDOW,
    chunks_elapsed: int = 999,
    expect_motion: bool = False,
    carry_dz: float = CARRY_DZ,
    drop_dz: float = DROP_DZ,
    place_dz: float = PLACE_DZ,
    lag_tolerance: float = LAG_TOLERANCE,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    samples = [dict(s) for s in (history or [])]
    if target_position is None and samples:
        target_position = samples[-1].get("target_position")
    for i in range(max(0, int(n_samples))):
        if i:
            time.sleep(float(sample_interval_s))
        samples.append(sample_now(ctx, arm_id, target_position, commanded_closed))
    if len(samples) < 2:
        raise ValueError("need at least two proprioception samples (history and/or n_samples)")

    span, stop, reach, source = _gripper(ctx, on_arm)
    table_level = _surface(ctx, surface_z, on_arm) + reach
    tail = samples[-max(2, int(window)) :]
    f = features(tail, span, stop)
    route, state, confidence, reason = classify(
        f,
        table_level,
        chunks_elapsed=chunks_elapsed,
        expect_motion=expect_motion,
        carry_dz=carry_dz,
        drop_dz=drop_dz,
        place_dz=place_dz,
        lag_tolerance=lag_tolerance,
    )
    f["table_level_z"] = table_level
    f["gripper_span_m"], f["gripper_stop_m"], f["gripper_source"] = span, stop, source
    logger.info(
        "[proprio] route=%s state=%s conf=%.2f samples=%d reason=%s",
        route,
        state,
        confidence,
        len(tail),
        reason,
    )
    return {
        "route": route,
        "state": state,
        "reason": reason,
        "confidence": confidence,
        "features": f,
        "history": tail,
        "samples": len(tail),
    }
