"""pick_object through a learned action model (vla-yam/1): run the policy, let the shared verifier decide.

The policy is any vla-yam/1 or gr00t-wam/1 server (policies/molmoact2-yam, policies/gr00t with the
``pi05`` or ``n1.7`` backend) reached through ``roboskillhub.vla.VlaYamClient``; the loop, the frame
mapping and the safety checks are ``roboskillhub.vla.run_policy``'s. The policy's own end (its jaw
closed and the hand moved on) is only a claim: the exit is ``verify_pick``'s, from the jaw opening,
the scene diff at the object's spot and, when the harness can say where the object is, the lift.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

from roboskillhub import vla
from roboskillhub.capability import PickObservation, verify_pick

HUB = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE = "yam-molmoact2-native"

OPEN_MIN = 0.65
"""Jaw opening (0 closed .. 1 open, through the profile's gripper map) at or over which the jaw is open:
the ``gripper_open`` precondition."""

CLOSED_BELOW = 0.5
"""Opening under which the policy has closed the jaw (the measured jaw, not the predicted one)."""

EMPTY_MAX = 0.03
"""Opening at or under which a closed jaw holds nothing (``reached`` for the gripper vote); over it,
the jaw stopped on something (``stalled_on_object``)."""

BOX_HALF_PX = 24
"""Half side of the scene-diff region around the object's projected point."""

EXITS = ("picked", "missed", "slipped", "unreachable", "refused", "timeout")


class Output(TypedDict):
    route: str
    held_object: str
    T_TCP_object: list[list[float]] | None
    evidence: dict[str, Any]
    reason: str


def resolve_profile(profile: str | dict[str, Any] | None, station: str | None) -> vla.Profile:
    """A profile mapping, a YAML path, or a name found under the hub's ``policies/*/profiles/``."""
    src: Any = profile or DEFAULT_PROFILE
    if isinstance(src, str) and not Path(src).is_file() and "\n" not in src:
        hits = sorted(HUB.glob(f"policies/*/profiles/{src}.yaml"))
        if not hits:
            raise vla.ProfileError([f"no policies/*/profiles/{src}.yaml in {HUB}"], src)
        src = hits[0]
    return vla.load_profile(src, station=station)


def object_label(object_ref: dict[str, Any]) -> str:
    return str(object_ref.get("text") or object_ref.get("id") or "object")


def project(point: Any, cam: dict[str, Any]) -> tuple[float, float] | None:
    """World point -> pixel (u, v) on a camera with ``K`` and ``T_world_cam`` (OpenCV axes)."""
    if cam.get("K") is None or cam.get("T_world_cam") is None:
        return None
    p = (
        np.linalg.inv(np.asarray(cam["T_world_cam"], float))
        @ np.r_[np.asarray(point, float)[:3], 1.0]
    )
    if p[2] <= 1e-6:
        return None
    uv = np.asarray(cam["K"], float) @ (p[:3] / p[2])
    return float(uv[0]), float(uv[1])


def opening(obs: dict[str, Any], arm: str, profile: vla.Profile) -> float:
    """The arm's jaw, 0 closed .. 1 open (the station's units through the profile's map)."""
    return float(profile.gripper.opening(obs["arms"][arm]["gripper"]))


def tcp(obs: dict[str, Any], arm: str) -> list[float] | None:
    T = obs["arms"][arm].get("T_world_tcp")
    return None if T is None else [float(v) for v in np.asarray(T)[:3, 3]]


def _out(route: str, reason: str, evidence: dict[str, Any], held: str = "") -> Output:
    return {
        "route": route,
        "held_object": held,
        "T_TCP_object": None,
        "evidence": evidence,
        "reason": reason,
    }


def run(
    ctx: NodeContext,
    object_ref: str | dict[str, Any],
    arm: str = "auto",
    grasp_hint: Any = None,
    server_url: str | None = None,
    profile: Any = None,
    station: str | None = None,
    instruction: str | None = None,
    max_steps: int = 600,
    timeout_s: float = 30.0,
    hold_checks: int = 2,
    object_position_tool: str | None = None,
    lift_source: str = "observation",
    cameras: dict[str, str] | None = None,
    check_profile: bool = True,
    client: Any = None,
) -> Output:
    """Pick ``object_ref`` (``{id|text|point}``, or a string: the object's text) with a learned policy and
    return the verifier's exit.

    ``server_url``: the policy server (default ``$ROBOSKILLHUB_VLA_URL``, then ``$GAP_GR00T_URL``).
    ``profile``: the checkpoint profile (name, path or mapping; default yam-molmoact2-native), ``station``
    one of its ``stations`` (``real``) or the twin. ``arm``: left / right restricts execution to that arm
    (the other holds still); auto executes both. ``hold_checks``: observations after the jaw closed
    before the loop stops (the policy lifts meanwhile). ``object_position_tool``: a tool returning the
    object's ``position`` (``{object_id}`` in), for the lift check; ``lift_source: sim_truth`` when it
    reads the simulator. ``client`` replaces the HTTP / ZMQ client (tests)."""
    t_start = time.monotonic()
    if isinstance(object_ref, str):
        object_ref = {"text": object_ref}
    if arm not in ("left", "right", "auto"):
        raise ValueError(f"arm is left | right | auto, not {arm!r}")
    if not isinstance(object_ref, dict) or not any(
        object_ref.get(k) is not None for k in ("id", "text", "point")
    ):
        raise ValueError("object_ref needs one of id, text, point")
    label = object_label(object_ref)
    evidence: dict[str, Any] = {"object_ref": object_ref, "arm": arm}

    def refused(reason: str, **pre: bool) -> Output:
        v = verify_pick(
            PickObservation(object_id=label, preconditions=pre or None), claim="refused"
        )
        return _out(v.exit, reason, {**evidence, "verdict": v.to_dict()})

    try:
        prof = resolve_profile(profile, station)
        pol = client or vla.VlaYamClient(server_url, prof, check_profile=check_profile)
        info = pol.info() if hasattr(pol, "info") else {}
    except (vla.ProfileError, vla.PolicyServerError, ValueError, OSError) as e:
        return refused(f"policy not available: {e}")
    evidence["policy"] = {
        "profile": prof.name,
        "station": prof.station,
        "backend": info.get("backend") or prof.backend,
        "model": info.get("model"),
        "revision": info.get("revision"),
        "server": getattr(pol, "url", None),
    }

    try:
        before = vla.get_observation(ctx, cameras=cameras)
    except vla.ObservationError as e:
        return refused(f"no usable observation: {e}")
    arms = ("left", "right") if arm == "auto" else (arm,)
    missing = [a for a in arms if a not in before["arms"]]
    if missing:
        return refused(f"the station reports no {missing} arm")

    top = before["cameras"].get("top") or {}
    uv = project(object_ref["point"], top) if object_ref.get("point") is not None else None
    box = None
    pre: dict[str, bool] = {"gripper_open": all(opening(before, a, prof) >= OPEN_MIN for a in arms)}
    if uv is not None and top.get("rgb") is not None:
        h, w = np.asarray(top["rgb"]).shape[:2]
        pre["object_visible"] = 0 <= uv[0] < w and 0 <= uv[1] < h
        box = [uv[0] - BOX_HALF_PX, uv[1] - BOX_HALF_PX, uv[0] + BOX_HALF_PX, uv[1] + BOX_HALF_PX]
    if not all(pre.values()):
        return refused(f"preconditions: {pre}", **pre)

    text = instruction or f"pick up the {label}" + ("" if arm == "auto" else f" with the {arm} arm")
    if uv is not None and hasattr(pol, "marks"):
        pol.marks = [
            {
                "kind": "grasp",
                "camera": "top",
                "uv": list(uv),
                **({} if arm == "auto" else {"arm": arm}),
            }
        ]
    evidence["instruction"] = text

    state: dict[str, Any] = {"closed_arm": None, "after_close": 0, "lift": [], "trace": []}

    def object_position() -> list[float] | None:
        if not object_position_tool:
            return None
        reply = ctx.tool(object_position_tool, object_id=str(object_ref.get("id") or label))
        pos = reply.get("position") if isinstance(reply, dict) else reply
        return None if pos is None else [float(v) for v in np.asarray(pos, float).reshape(-1)[:3]]

    def stop_check(obs: dict[str, Any], info: dict[str, Any]) -> Any:
        if time.monotonic() - t_start > timeout_s:
            return {"stop": True, "exit": "timeout"}
        state["trace"].append({a: round(opening(obs, a, prof), 4) for a in arms})
        if state["closed_arm"] is None:
            closed = [a for a in arms if opening(obs, a, prof) < CLOSED_BELOW]
            if not closed:
                return False
            state["closed_arm"] = closed[0]
        a = state["closed_arm"]
        t, o = tcp(obs, a), object_position()
        if t is not None and o is not None:
            state["lift"].append({"tcp": t, "object": o})
        state["after_close"] += 1
        return {"stop": state["after_close"] > hold_checks, "exit": "picked"}

    res = vla.run_policy(
        pol,
        prof,
        text,
        max_steps=max_steps,
        stop_check=stop_check,
        robot=ctx,
        arms=None if arm == "auto" else arms,
        cameras=cameras,
    )
    elapsed = time.monotonic() - t_start
    evidence["run"] = {
        k: res.get(k) for k in ("route", "steps", "chunks", "detail", "code", "jaw_events")
    }
    evidence["run"]["latency_ms"] = [c.get("latency_ms") for c in res.get("chunk_log") or []]
    evidence["opening_trace"] = state["trace"]

    route = res["route"]
    if route == "stopped":
        claim = res.get("exit") or "picked"
    elif route in ("client_error", "bad_observation", "refused") and res["steps"] == 0:
        claim = "refused"
    else:
        claim = "missed"

    try:
        after = vla.get_observation(ctx, cameras=cameras)
    except vla.ObservationError:
        after = None
    held_arm = state["closed_arm"] or (arms[0] if len(arms) == 1 else None)
    gripper: dict[str, Any] | None = None
    if after is not None and held_arm is not None and state["closed_arm"] is not None:
        o = opening(after, held_arm, prof)
        gripper = {
            "route": "reached" if o <= EMPTY_MAX else "stalled_on_object",
            "opening": round(o, 4),
            "reason": f"{held_arm} jaw closed to opening {o:.3f} (empty at <= {EMPTY_MAX})",
        }
    after_top = (after or {}).get("cameras", {}).get("top") or {}
    verdict = verify_pick(
        PickObservation(
            object_id=label,
            preconditions=pre,
            elapsed_s=elapsed,
            timeout_s=timeout_s,
            gripper=gripper,
            before_rgb=top.get("rgb") if box is not None else None,
            after_rgb=after_top.get("rgb") if box is not None else None,
            source_box_px=box,
            lift=state["lift"] or None,
            lift_source="sim_truth" if lift_source == "sim_truth" else "observation",
        ),
        claim=claim,
    )
    evidence["verdict"] = verdict.to_dict()
    evidence["arm_used"] = held_arm
    if verdict.exit not in EXITS:
        raise RuntimeError(f"verifier returned {verdict.exit!r}")
    return _out(
        verdict.exit, verdict.reason, evidence, held=label if verdict.exit == "picked" else ""
    )
