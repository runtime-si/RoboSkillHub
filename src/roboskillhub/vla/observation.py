"""Observations (VLA_SKILL_SPEC section 1) from a gap connector, and model inputs from them per profile.

``from_connector`` turns ``robot.get_observation`` (a gap ``Observation``: ``cameras`` as a list of named
frames, ``arms`` as a list, left first) into the spec's shape::

    {"t": s, "cameras": {role: {"rgb", "t", "K"?, "T_world_cam"?}},
     "arms": {"left": {"joints": [6], "gripper": station units, "t"}, "right": {...}}}

Camera roles are matched by name as policies/gr00t (#173) does: top / base / scene / overhead for the
top view, a wrist (or hand) name with "left" or alone for the left wrist, one with "right" for the right
wrist; ``cameras={role: name}`` overrides. Connectors that stamp their frames (``t`` / ``stamp`` on the
observation, a camera or an arm) are checked against ``max_obs_age_s``; unstamped parts take the time
the observation was fetched and are listed in ``obs["unstamped"]``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from .profile import ARMS, JOINTS, ROLES, Profile


class ObservationError(ValueError):
    """An observation the model must not be given (stale, missing a role or an arm, malformed)."""


def _stamp(d: dict[str, Any]) -> float | None:
    for k in ("t", "stamp", "timestamp"):
        if d.get(k) is not None:
            return float(d[k])
    return None


def _camera_name(cam: dict[str, Any]) -> str:
    return str(cam.get("name") or "").lower()


def camera_roles(
    frames: list[dict[str, Any]], names: dict[str, str] | None = None
) -> dict[str, dict]:
    """Connector camera frames by role; roles not found are left out."""
    frames = [c for c in frames if isinstance(c, dict)]
    by_name = {_camera_name(c): c for c in frames}
    out: dict[str, dict] = {}
    for role, name in (names or {}).items():
        if role not in ROLES:
            raise ObservationError(f"camera role {role!r} is not one of {list(ROLES)}")
        if str(name).lower() not in by_name:
            raise ObservationError(f"no camera {name!r} in the observation ({sorted(by_name)})")
        out[role] = by_name[str(name).lower()]
    for c in frames:
        n = _camera_name(c)
        if "wrist" in n or "hand" in n:
            role = "right_wrist" if "right" in n else "left_wrist"
        elif any(k in n for k in ("top", "base", "scene", "overhead")):
            role = "top"
        else:
            continue
        out.setdefault(role, c)
    if "top" not in out:
        rest = [c for c in frames if all(c is not v for v in out.values())]
        if rest:
            out["top"] = rest[0]
    return out


def rgb(frame: dict[str, Any]) -> np.ndarray:
    """uint8 [H, W, 3] from a frame's ``rgb`` (or ``images.rgb``); RGBA dropped, floats in 0..1 scaled."""
    a = frame.get("rgb")
    if a is None:
        a = (frame.get("images") or {}).get("rgb")
    if a is None:
        raise ObservationError(f"camera {frame.get('name')!r} has no rgb")
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    if a.ndim != 3 or a.shape[2] != 3:
        raise ObservationError(f"camera {frame.get('name')!r}: rgb is {a.dtype} {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def _pose_matrix(pose: Any) -> np.ndarray | None:
    if pose is None:
        return None
    if isinstance(pose, dict):
        p, q = pose["position"], pose["rotation"]
        pos = np.array([p["x"], p["y"], p["z"]] if isinstance(p, dict) else p, dtype=np.float64)
        w, x, y, z = (q["w"], q["x"], q["y"], q["z"]) if isinstance(q, dict) else q
    else:
        a = np.asarray(pose, dtype=np.float64)
        if a.shape == (4, 4):
            return a
        pos, (w, x, y, z) = a[:3], a[3:7]
    n = float(np.linalg.norm([w, x, y, z])) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    T[:3, 3] = pos
    return T


def _connector_arms(raw: Any) -> dict[str, dict[str, Any]]:
    if isinstance(raw, dict):
        return {a: raw[a] for a in ARMS if a in raw}
    return dict(zip(ARMS, list(raw or []), strict=False))


def from_connector(
    raw: dict[str, Any], *, cameras: dict[str, str] | None = None, now: float | None = None
) -> dict[str, Any]:
    """The spec observation from a connector observation. ``now`` stamps unstamped parts."""
    t_fetch = float(now if now is not None else time.time())
    t = _stamp(raw)
    t = t_fetch if t is None else t
    unstamped: list[str] = [] if _stamp(raw) is not None else ["observation"]
    cams: dict[str, Any] = {}
    for role, frame in camera_roles(raw.get("cameras") or [], cameras).items():
        ct = _stamp(frame)
        if ct is None:
            unstamped.append(f"camera {role}")
        cam: dict[str, Any] = {"rgb": rgb(frame), "t": t if ct is None else ct}
        if frame.get("intrinsics") is not None:
            cam["K"] = np.asarray(frame["intrinsics"], dtype=np.float64).reshape(3, 3)
        T = _pose_matrix(frame.get("pose"))
        if T is not None:
            cam["T_world_cam"] = T
        cam["name"] = frame.get("name")
        cams[role] = cam
    arms: dict[str, Any] = {}
    for side, arm in _connector_arms(raw.get("arms")).items():
        if arm is None:
            continue
        js = arm.get("joint_state") or {}
        q = arm.get("joints", js.get("positions") if isinstance(js, dict) else None)
        q = np.asarray(q if q is not None else [], dtype=np.float64).reshape(-1)
        if q.size < JOINTS:
            raise ObservationError(f"{side} arm reports {q.size} joints; the YAM has {JOINTS}")
        g = arm.get("gripper", arm.get("gripper_fraction"))
        if g is None:
            raise ObservationError(f"{side} arm reports no gripper opening")
        at = _stamp(arm)
        if at is None:
            unstamped.append(f"arm {side}")
        entry = {"joints": q[:JOINTS].copy(), "gripper": float(g), "t": t if at is None else at}
        if arm.get("ee_pose") is not None:
            entry["T_world_tcp"] = _pose_matrix(arm["ee_pose"])
        arms[side] = entry
    out = {"t": t, "cameras": cams, "arms": arms}
    if unstamped:
        out["unstamped"] = unstamped
    if raw.get("estop") is not None:
        out["estop"] = bool(raw["estop"])
    return out


def get_observation(
    ctx: Any,
    *,
    cameras: dict[str, str] | None = None,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """``robot.get_observation`` through ``ctx.tool``, as a spec observation."""
    return from_connector(ctx.tool("robot.get_observation"), cameras=cameras, now=clock())


def staleness(obs: dict[str, Any], max_age_s: float, now: float | None = None) -> list[str]:
    """Every frame older than ``max_age_s`` against ``obs["t"]`` (and the observation against ``now``)."""
    t = float(obs["t"])
    late = []
    if now is not None and now - t > max_age_s:
        late.append(f"observation is {now - t:.3f} s old (max {max_age_s} s)")
    for role, cam in (obs.get("cameras") or {}).items():
        if t - float(cam["t"]) > max_age_s:
            late.append(f"camera {role} is {t - float(cam['t']):.3f} s older than the observation")
    for side, arm in (obs.get("arms") or {}).items():
        if t - float(arm["t"]) > max_age_s:
            late.append(f"arm {side} is {t - float(arm['t']):.3f} s older than the observation")
    return late


def check(obs: dict[str, Any], profile: Profile, now: float | None = None) -> None:
    """Raise :class:`ObservationError` unless every role and arm the profile needs is present and fresh."""
    problems = [
        f"no {r} camera (have {sorted(obs.get('cameras') or {})})"
        for r in profile.cameras
        if r not in (obs.get("cameras") or {})
    ]
    problems += [f"no {a} arm" for a in profile.arms if a not in (obs.get("arms") or {})]
    problems += staleness(obs, profile.max_obs_age_s, now)
    if problems:
        raise ObservationError("; ".join(problems))


def station_state(obs: dict[str, Any], profile: Profile, layout: tuple | None = None) -> np.ndarray:
    """The state vector in station frame and units, in the profile's state layout (or ``layout``)."""
    slots = layout or profile.state
    out = np.zeros(len(slots))
    for i, s in enumerate(slots):
        arm = obs["arms"][s.arm]
        out[i] = arm["gripper"] if s.joint is None else arm["joints"][s.joint]
    return out


def model_input(obs: dict[str, Any], profile: Profile, now: float | None = None) -> dict[str, Any]:
    """What a backend gets: images in the profile's camera order, the state in model frame and units.

    ``images`` is ordered (dicts keep insertion order) and ``image_order`` names it explicitly."""
    check(obs, profile, now)
    return {
        "images": {r: obs["cameras"][r]["rgb"] for r in profile.cameras},
        "image_order": list(profile.cameras),
        "state": profile.to_model(station_state(obs, profile)),
        "state_names": [s.name for s in profile.state],
        "t": float(obs["t"]),
        "profile": profile.name,
    }
