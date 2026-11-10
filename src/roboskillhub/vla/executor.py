"""Action chunks to station motion (VLA_SKILL_SPEC section 4), with the safety rules in the executor.

``plan_chunk`` interprets one chunk per profile - model frame and units back to the station's (or taken
as they are when the server already answers in station frame), ``delta_joint`` rows made absolute, the
first ``execute_steps`` rows kept - checks it and cuts it into segments at the jaw switches, retimed to
the station rate. ``send_plan`` streams the segments through ``robot.stream_joints`` when the connector
has it, else ``robot.stream_dual`` with ``robot.open_gripper`` / ``robot.close_gripper`` at the cuts
(as #173's gr00t.run_skill does). ``execute_chunk`` does both and never raises for a refused chunk: the
result carries ``ok``, a ``code`` and a ``reason``.

Safety (refuse = nothing of the chunk moves):

- joint position limits (j6 |q| <= 2.09 rad by default) on every commanded row - refuse;
- a joint moving more than ``max_step_rad`` (0.35) between two rows of the chunk - refuse;
- the first row further than ``max_first_gap_rad`` (0.8) from the measured arm - refuse (usually a wrong
  ``joint_offset_rad``); closer than that but beyond ``max_step_rad`` - ramped;
- joint speed above ``max_vel_rad_s`` (1.5) - clipped by stretching that interval (the ramp is the
  first interval stretched the same way);
- TCP below ``table_z + tcp_clearance_m`` (14 mm) on any streamed waypoint, by FK when the connector
  has ``robot.forward_kinematics`` and ``table_z`` is set - refuse; otherwise a note says it was not checked;
- a chunk older than ``stale_chunk_s`` (0.5 s, from the observation it was computed on) - refuse,
  checked before every segment against the time the earlier segments were planned to take;
- an e-stop flag (a callable, or ``estop`` on the observation) - refuse;
- ``gripper_force_cap`` is passed to ``robot.stream_joints`` (the station enforces it).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from .observation import station_state
from .profile import JOINTS, Profile

ARM_ID = {"left": 0, "right": 1}
GRIP_CLOSE_BELOW = 0.35
GRIP_OPEN_ABOVE = 0.65
"""Hysteresis on the opening (0 closed .. 1 open), as #173: the jaws are open or closed."""

_TWO_PI = 2 * math.pi


class Refusal(Exception):
    """A chunk the executor will not run; ``code`` is machine-readable, ``reason`` says why."""

    def __init__(self, code: str, reason: str) -> None:
        self.code, self.reason = code, reason
        super().__init__(f"{code}: {reason}")


@dataclass(frozen=True)
class SafetyLimits:
    joint_lo: tuple[float, ...] = (-_TWO_PI,) * 5 + (-2.09,)
    joint_hi: tuple[float, ...] = (_TWO_PI,) * 5 + (2.09,)
    max_step_rad: float = 0.35
    max_first_gap_rad: float = 0.8
    max_vel_rad_s: float = 1.5
    table_z: float | None = None
    tcp_clearance_m: float = 0.014
    stale_chunk_s: float = 0.5
    gripper_force_cap: float | None = None

    @classmethod
    def from_profile(cls, profile: Profile, **overrides: Any) -> SafetyLimits:
        """Defaults, then the profile's ``safety:`` block, then ``overrides``."""
        names = {f.name for f in fields(cls)}
        merged = {**profile.safety, **overrides}
        unknown = sorted(set(merged) - names)
        if unknown:
            raise ValueError(f"unknown safety keys {unknown} (known: {sorted(names)})")
        for k in ("joint_lo", "joint_hi"):
            if k in merged:
                v = tuple(float(x) for x in merged[k])
                if len(v) != JOINTS:
                    raise ValueError(f"safety.{k} needs {JOINTS} values")
                merged[k] = v
        return cls(**merged)


@dataclass
class Segment:
    tracks: dict[
        str, np.ndarray
    ]  # arm -> [N, 7] station-rate waypoints, 6 joints + gripper (station units)
    switches: dict[str, bool]  # arm -> True = close, False = open, after the segment
    rows: int  # chunk rows this segment covers
    duration_s: float


@dataclass
class Plan:
    segments: list[Segment]
    hz: float
    t_obs: float
    rows: int
    notes: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return sum(s.duration_s for s in self.segments)


FK = Callable[[str, np.ndarray], np.ndarray]
"""``fk(arm, joints[N, 6]) -> tcp z [N]`` in the frame ``table_z`` is given in."""


def fk_from_ctx(ctx: Any) -> FK:
    """FK through ``robot.forward_kinematics`` (``joint_configs`` -> ``poses[i].position.z``)."""

    def fk(arm: str, joints: np.ndarray) -> np.ndarray:
        reply = ctx.tool(
            "robot.forward_kinematics",
            joint_configs=np.asarray(joints, dtype=np.float64).tolist(),
            arm_id=ARM_ID[arm],
        )
        zs = []
        for p in reply["poses"]:
            pos = p["position"]
            zs.append(float(pos["z"] if isinstance(pos, dict) else pos[2]))
        return np.asarray(zs)

    return fk


def interpret(
    actions: Any,
    profile: Profile,
    obs: dict[str, Any],
    *,
    frame: str = "model",
    execute_steps: int | None = None,
) -> np.ndarray:
    """Chunk rows -> absolute station-frame rows ``[n, dim]`` in the action layout, first ``execute_steps``."""
    spec = profile.action
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != spec.dim or a.shape[0] < 1:
        raise Refusal("bad_shape", f"chunk is {a.shape}, the profile wants [T>=1, {spec.dim}]")
    if not np.all(np.isfinite(a)):
        raise Refusal("non_finite", "chunk has NaN or inf values")
    if frame not in ("model", "station"):
        raise Refusal("bad_frame", f"chunk frame {frame!r} is not model or station")
    a = a[: max(1, int(execute_steps or spec.execute_steps))]
    layout = spec.layout
    if spec.type == "delta_joint":
        base = station_state(obs, profile, layout)
        if frame == "model":
            base = profile.to_model(base, layout)
        jcols = [i for i, s in enumerate(layout) if s.joint is not None]
        d = a[:, jcols]
        if spec.delta_ref == "step":
            d = np.cumsum(d, axis=0)
        a[:, jcols] = base[jcols][None] + d
    return profile.to_station(a, layout) if frame == "model" else a


def _cuts(rows: dict[str, np.ndarray], profile: Profile, closed: dict[str, bool]):
    """[(end_row_exclusive, {arm: close?})] at each jaw switch, then the chunk's end."""
    n = len(next(iter(rows.values())))
    state = dict(closed)
    out = []
    start = 0
    for i in range(n):
        sw = {}
        for arm, r in rows.items():
            o = profile.gripper.opening(r[i, JOINTS])
            want = True if o < GRIP_CLOSE_BELOW else False if o > GRIP_OPEN_ABOVE else state[arm]
            if want != state[arm]:
                sw[arm] = want
                state[arm] = want
        if sw:
            out.append((i + 1, sw))
            start = i + 1
    if start < n:
        out.append((n, {}))
    return out


def _retime(
    start: dict[str, np.ndarray], rows: dict[str, np.ndarray], dt: float, hz: float, vmax: float
) -> tuple[dict[str, np.ndarray], float, int]:
    """Station-rate waypoints from ``start`` through ``rows`` (lockstep across arms); each interval
    lasts at least ``dt`` and long enough that no joint exceeds ``vmax``. Returns (tracks, s, clipped)."""
    arms = list(rows)
    n = len(rows[arms[0]])
    step = 1.0 / hz
    out: dict[str, list[np.ndarray]] = {a: [] for a in arms}
    prev = {a: start[a] for a in arms}
    total, clipped = 0.0, 0
    for i in range(n):
        dq = max(float(np.abs(rows[a][i, :JOINTS] - prev[a][:JOINTS]).max()) for a in arms)
        need = dq / vmax if vmax > 0 else 0.0
        if need > dt + 1e-12:
            clipped += 1
        dur = max(dt, need)
        k = max(1, math.ceil(dur / step - 1e-9))
        for a in arms:
            f = (np.arange(1, k + 1) / k)[:, None]
            out[a].extend(prev[a][None] + f * (rows[a][i] - prev[a])[None])
            prev[a] = rows[a][i]
        total += k * step
    return {a: np.asarray(v) for a, v in out.items()}, total, clipped


def plan_chunk(
    actions: Any,
    profile: Profile,
    obs: dict[str, Any],
    *,
    t_obs: float | None = None,
    now: float | None = None,
    frame: str = "model",
    arms: tuple[str, ...] | None = None,
    jaw_closed: dict[str, bool | None] | None = None,
    execute_steps: int | None = None,
    station_hz: float | None = None,
    limits: SafetyLimits | None = None,
    fk: FK | None = None,
) -> Plan:
    """Check and cut one chunk. Raises :class:`Refusal` with the reason when it must not run."""
    lim = limits or SafetyLimits.from_profile(profile)
    t_obs = float(obs["t"] if t_obs is None else t_obs)
    now = time.time() if now is None else float(now)
    if now - t_obs > lim.stale_chunk_s:
        raise Refusal(
            "stale_chunk", f"chunk is {now - t_obs:.3f} s old (limit {lim.stale_chunk_s} s)"
        )
    if obs.get("estop"):
        raise Refusal("estop", "the observation carries the e-stop flag")
    moving = tuple(arms or profile.arms)
    layout = profile.action.layout
    bad = [a for a in moving if a not in {s.arm for s in layout}]
    if bad:
        raise Refusal("bad_arms", f"the action layout does not drive {bad}")

    station = interpret(actions, profile, obs, frame=frame, execute_steps=execute_steps)
    notes: list[str] = []
    rows: dict[str, np.ndarray] = {}
    current: dict[str, np.ndarray] = {}
    lo, hi = np.asarray(lim.joint_lo), np.asarray(lim.joint_hi)
    for arm in moving:
        jc, gc = profile.arm_columns(arm, layout)
        r = station[:, [*jc, gc]]
        g = profile.gripper.clip_station(r[:, JOINTS])
        if not np.allclose(g, r[:, JOINTS]):
            notes.append(f"{arm} gripper targets clipped to the station range")
            r[:, JOINTS] = g
        rows[arm] = r
        a_obs = obs["arms"][arm]
        current[arm] = np.array([*a_obs["joints"][:JOINTS], a_obs["gripper"]], dtype=np.float64)

        q = r[:, :JOINTS]
        out = (q < lo[None] - 1e-9) | (q > hi[None] + 1e-9)
        if out.any():
            i, j = (int(x[0]) for x in np.nonzero(out))
            raise Refusal(
                "joint_limit",
                f"{arm} row {i}: j{j + 1} = {q[i, j]:.3f} rad is outside [{lo[j]:.3f}, {hi[j]:.3f}]",
            )
        if len(q) > 1:
            jump = np.abs(np.diff(q, axis=0))
            if float(jump.max()) > lim.max_step_rad:
                i, j = np.unravel_index(int(np.argmax(jump)), jump.shape)
                raise Refusal(
                    "jump",
                    f"{arm} rows {i}-{i + 1}: j{j + 1} would move {jump[i, j]:.3f} rad (limit {lim.max_step_rad})",
                )
        gap = np.abs(q[0] - current[arm][:JOINTS])
        if float(gap.max()) > lim.max_first_gap_rad:
            j = int(np.argmax(gap))
            raise Refusal(
                "first_gap",
                f"{arm} first target j{j + 1} is {gap[j]:.3f} rad from the measured arm (limit "
                f"{lim.max_first_gap_rad}; check joint_offset_rad)",
            )
        if float(gap.max()) > lim.max_step_rad:
            notes.append(f"{arm}: ramped {float(gap.max()):.3f} rad to the first target")

    closed = {}
    for arm in moving:
        jc = (jaw_closed or {}).get(arm)
        closed[arm] = (
            bool(jc) if jc is not None else profile.gripper.opening(current[arm][JOINTS]) < 0.5
        )
    hz = float(station_hz or profile.action.hz)
    segments: list[Segment] = []
    start, begin, clipped = dict(current), 0, 0
    for end, sw in _cuts(rows, profile, closed):
        part = {a: r[begin:end] for a, r in rows.items()}
        tracks, dur, c = _retime(start, part, profile.action.dt, hz, lim.max_vel_rad_s)
        clipped += c
        segments.append(Segment(tracks, sw, end - begin, dur))
        start = {a: r[end - 1] for a, r in rows.items()}
        begin = end
    if clipped:
        notes.append(f"{clipped} interval(s) slowed to {lim.max_vel_rad_s} rad/s")

    if lim.table_z is None or fk is None:
        notes.append(
            "TCP floor not checked (" + ("no table_z" if lim.table_z is None else "no FK") + ")"
        )
    else:
        floor = lim.table_z + lim.tcp_clearance_m
        for k, seg in enumerate(segments):
            for arm, tr in seg.tracks.items():
                z = np.asarray(fk(arm, tr[:, :JOINTS]), dtype=np.float64).reshape(-1)
                if z.size and float(z.min()) < floor:
                    i = int(np.argmin(z))
                    raise Refusal(
                        "tcp_floor",
                        f"{arm} segment {k} waypoint {i}: TCP z {z[i]:.4f} m is below table + "
                        f"{lim.tcp_clearance_m * 1e3:.0f} mm ({floor:.4f} m)",
                    )
    return Plan(segments, hz, t_obs, int(sum(s.rows for s in segments)), notes)


_MISSING = ("no scripted response", "unknown tool", "not registered", "no such tool", "not found")


def _missing_tool(e: Exception) -> bool:
    text = str(e).lower()
    return any(m in text for m in _MISSING)


def send_plan(
    ctx: Any,
    plan: Plan,
    *,
    transport: str = "auto",
    limits: SafetyLimits | None = None,
    estop: Callable[[], bool] | None = None,
    clock: Callable[[], float] = time.time,
    requery_after_switch: bool = True,
    transport_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stream the plan's segments. ``transport``: ``auto`` (stream_joints, falling back to stream_dual
    when the connector does not have it), ``stream_joints`` or ``stream_dual``. With
    ``requery_after_switch`` the plan stops after the first segment that ends in a jaw switch, so the
    next chunk sees the grasp (or release)."""
    lim = limits or SafetyLimits()
    if transport not in ("auto", "stream_joints", "stream_dual"):
        raise ValueError(f"unknown transport {transport!r}")
    extra = dict(transport_args or {})
    result: dict[str, Any] = {
        "ok": True,
        "code": "ok",
        "reason": "",
        "segments_sent": 0,
        "rows": 0,
        "waypoints": 0,
        "jaw_events": [],
        "notes": list(plan.notes),
        "transport": transport,
    }

    def refuse(code: str, reason: str) -> dict[str, Any]:
        result.update(ok=False, code=code, reason=reason)
        return result

    planned = 0.0
    for k, seg in enumerate(plan.segments):
        if estop is not None and estop():
            return refuse("estop", f"e-stop set before segment {k}")
        late = clock() - plan.t_obs - planned
        if late > lim.stale_chunk_s:
            return refuse(
                "stale_chunk",
                f"segment {k} would start {late:.3f} s behind its chunk (limit {lim.stale_chunk_s} s)",
            )
        if transport in ("auto", "stream_joints"):
            kw = dict(extra)
            if lim.gripper_force_cap is not None:
                kw["gripper_force_cap"] = lim.gripper_force_cap
            try:
                ctx.tool(
                    "robot.stream_joints",
                    tracks={a: t.tolist() for a, t in seg.tracks.items()},
                    hz=plan.hz,
                    mode="abs",
                    **kw,
                )
                transport = "stream_joints"
            except Exception as e:
                if transport == "stream_joints" or not _missing_tool(e):
                    return refuse("transport", f"robot.stream_joints: {e}")
                transport = "stream_dual"
                result["notes"].append("robot.stream_joints unavailable; using robot.stream_dual")
        if transport == "stream_dual":
            try:
                ctx.tool(
                    "robot.stream_dual",
                    tracks={ARM_ID[a]: t[:, :JOINTS].tolist() for a, t in seg.tracks.items()},
                    **extra,
                )
                for arm, close in seg.switches.items():
                    kw = {"arm_id": ARM_ID[arm]} if ARM_ID[arm] else {}
                    ctx.tool("robot.close_gripper" if close else "robot.open_gripper", **kw)
            except Exception as e:
                return refuse("transport", f"robot.stream_dual / gripper: {e}")
        result["transport"] = transport
        planned += seg.duration_s
        result["segments_sent"] += 1
        result["rows"] += seg.rows
        result["waypoints"] += max(len(t) for t in seg.tracks.values())
        for arm, close in seg.switches.items():
            result["jaw_events"].append(
                {"segment": k, "arm": arm, "jaw": "close" if close else "open"}
            )
        if seg.switches and requery_after_switch:
            break
    return result


def execute_chunk(
    ctx: Any,
    actions: Any,
    profile: Profile,
    obs: dict[str, Any],
    *,
    estop: Callable[[], bool] | None = None,
    clock: Callable[[], float] = time.time,
    transport: str = "auto",
    transport_args: dict[str, Any] | None = None,
    requery_after_switch: bool = True,
    use_fk: bool = True,
    limits: SafetyLimits | None = None,
    **plan_kw: Any,
) -> dict[str, Any]:
    """``plan_chunk`` then ``send_plan``; a refused chunk comes back as ``{"ok": False, "code", "reason"}``."""
    lim = limits or SafetyLimits.from_profile(profile)
    fk = plan_kw.pop("fk", None)
    if fk is None and use_fk and lim.table_z is not None:
        fk = fk_from_ctx(ctx)
    try:
        if estop is not None and estop():
            raise Refusal("estop", "e-stop set")
        plan = plan_chunk(actions, profile, obs, now=clock(), limits=lim, fk=fk, **plan_kw)
    except Refusal as r:
        return {"ok": False, "code": r.code, "reason": r.reason, "segments_sent": 0, "rows": 0}
    except Exception as e:  # FK failure: refuse rather than move unchecked
        return {"ok": False, "code": "check_failed", "reason": f"{type(e).__name__}: {e}"}
    return send_plan(
        ctx,
        plan,
        transport=transport,
        limits=lim,
        estop=estop,
        clock=clock,
        requery_after_switch=requery_after_switch,
        transport_args=transport_args,
    )
