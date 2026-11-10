"""Checkpoint profiles (VLA_SKILL_SPEC section 2): everything checkpoint-specific between the station and a model.

A profile is one YAML per checkpoint, shipped in the policy bundle::

    profile: yam-molmoact2-native
    checkpoint: {repo: allenai/MolmoAct2-BimanualYAM, revision: <commit>}
    cameras: [top, left_wrist, right_wrist]          # model input order
    state: [left.j1..j6, left.gripper, right.j1..j6, right.gripper]
    joint_offset_rad: {left: [0, 0, 0, 0, 0, 0], right: [0, 0, 0, 0, 0, 0]}
    gripper: {station_open: 1.0, station_closed: 0.0, model_open: 1.0, model_closed: 0.0}
    action: {type: abs_joint, dim: 14, horizon: 30, execute_steps: 25, hz: 30}
    extra: {norm_tag: yam_dual_molmoact2}
    safety: {table_z: 0.0}                           # optional, see executor.SafetyLimits

Joint offsets follow #173 (``GAP_GR00T_JOINT_OFFSET``): ``station joint = model joint + offset``, so the
real YAM station's ``{left: [pi/2, 0, ...]}`` maps a hardware base at 90 deg to the MolmoAct2 data's 0.
The gripper maps linearly between the two (open, closed) pairs. Actions use the state layout unless
``action.layout`` lists another; ``delta_joint`` actions are joint deltas against the state the chunk was
computed from (``action.delta_ref: obs``) or against the previous row (``step``); grippers stay absolute.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROLES = ("top", "left_wrist", "right_wrist")
ARMS = ("left", "right")
JOINTS = 6
ACTION_TYPES = ("abs_joint", "delta_joint")
DELTA_REFS = ("obs", "step")
DEFAULT_MAX_OBS_AGE_S = 0.1

_RANGE = re.compile(r"^(left|right)\.j(\d)\.\.j?(\d)$")
_SLOT = re.compile(r"^(left|right)\.(?:j(\d)|(gripper))$")


class ProfileError(ValueError):
    """A profile that cannot drive the station; ``problems`` lists every reason."""

    def __init__(self, problems: list[str], name: str = "") -> None:
        self.problems = list(problems)
        super().__init__(f"profile {name or '?'}: " + "; ".join(self.problems))


@dataclass(frozen=True)
class Slot:
    """One state / action column: an arm's joint (0-based) or its gripper."""

    arm: str
    joint: int | None  # None = gripper

    @property
    def name(self) -> str:
        return f"{self.arm}.gripper" if self.joint is None else f"{self.arm}.j{self.joint + 1}"


@dataclass(frozen=True)
class GripperMap:
    station_open: float
    station_closed: float
    model_open: float
    model_closed: float

    def to_model(self, station: Any) -> Any:
        return self.model_closed + self.opening(station) * (self.model_open - self.model_closed)

    def to_station(self, model: Any) -> Any:
        frac = (np.asarray(model, dtype=np.float64) - self.model_closed) / (
            self.model_open - self.model_closed
        )
        out = self.station_closed + frac * (self.station_open - self.station_closed)
        return float(out) if np.ndim(out) == 0 else out

    def opening(self, station: Any) -> Any:
        """Station units -> 0 closed .. 1 open (not clipped)."""
        out = (np.asarray(station, dtype=np.float64) - self.station_closed) / (
            self.station_open - self.station_closed
        )
        return float(out) if np.ndim(out) == 0 else out

    def clip_station(self, station: np.ndarray) -> np.ndarray:
        lo, hi = sorted((self.station_open, self.station_closed))
        return np.clip(station, lo, hi)


@dataclass(frozen=True)
class ActionSpec:
    type: str
    dim: int
    horizon: int
    execute_steps: int
    hz: float
    delta_ref: str = "obs"
    layout: tuple[Slot, ...] = ()

    @property
    def dt(self) -> float:
        return 1.0 / self.hz


@dataclass(frozen=True)
class Profile:
    name: str
    cameras: tuple[str, ...]
    state: tuple[Slot, ...]
    joint_offset_rad: dict[str, np.ndarray]
    gripper: GripperMap
    action: ActionSpec
    checkpoint: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    max_obs_age_s: float = DEFAULT_MAX_OBS_AGE_S

    # -- station <-> model, one vector or [T, n] rows in a given layout --------------------------

    def to_model(self, station: np.ndarray, layout: tuple[Slot, ...] | None = None) -> np.ndarray:
        """Station frame and units -> model frame and units."""
        out = np.array(station, dtype=np.float64)
        for i, s in enumerate(layout or self.state):
            if s.joint is None:
                out[..., i] = self.gripper.to_model(out[..., i])
            else:
                out[..., i] -= self.joint_offset_rad[s.arm][s.joint]
        return out

    def to_station(self, model: np.ndarray, layout: tuple[Slot, ...] | None = None) -> np.ndarray:
        """Model frame and units -> station frame and units."""
        out = np.array(model, dtype=np.float64)
        for i, s in enumerate(layout or self.state):
            if s.joint is None:
                out[..., i] = self.gripper.to_station(out[..., i])
            else:
                out[..., i] += self.joint_offset_rad[s.arm][s.joint]
        return out

    def arm_columns(
        self, arm: str, layout: tuple[Slot, ...] | None = None
    ) -> tuple[list[int], int]:
        """(six joint columns in joint order, gripper column) of ``arm`` in ``layout``."""
        cols = {s: i for i, s in enumerate(layout or self.state)}
        return [cols[Slot(arm, j)] for j in range(JOINTS)], cols[Slot(arm, None)]

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(a for a in ARMS if any(s.arm == a for s in self.state))


def _slots(raw: Any, what: str, problems: list[str]) -> tuple[Slot, ...]:
    if not isinstance(raw, list) or not raw:
        problems.append(f"{what} must be a non-empty list")
        return ()
    out: list[Slot] = []
    for item in raw:
        text = str(item).strip()
        m = _RANGE.match(text)
        if m:
            a, b = int(m.group(2)), int(m.group(3))
            if not 1 <= a <= b <= JOINTS:
                problems.append(f"{what}: bad joint range {text!r}")
                continue
            out.extend(Slot(m.group(1), j - 1) for j in range(a, b + 1))
            continue
        m = _SLOT.match(text)
        if not m or (m.group(2) and not 1 <= int(m.group(2)) <= JOINTS):
            problems.append(f"{what}: {text!r} is not <left|right>.j1..j6 or <left|right>.gripper")
            continue
        out.append(Slot(m.group(1), None if m.group(3) else int(m.group(2)) - 1))
    if len(set(out)) != len(out):
        problems.append(f"{what} names a column twice")
    for arm in {s.arm for s in out}:
        missing = [s.name for s in [*(Slot(arm, j) for j in range(JOINTS)), Slot(arm, None)]]
        missing = [n for n in missing if n not in {s.name for s in out}]
        if missing:
            problems.append(f"{what}: arm {arm} lacks {missing}")
    return tuple(out)


def _number(raw: Any, key: str, problems: list[str]) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        problems.append(f"{key} is {raw!r}, not a number (measure it)")
        return math.nan
    if not math.isfinite(v):
        problems.append(f"{key} is not finite")
    return v


def profile_from_dict(raw: dict[str, Any]) -> Profile:
    """Validate a parsed profile; raises :class:`ProfileError` listing every problem at once."""
    problems: list[str] = []
    if not isinstance(raw, dict):
        raise ProfileError(["a profile is a mapping"])
    name = str(raw.get("profile") or "")
    if not name:
        problems.append("profile (the name) is missing")

    cams = raw.get("cameras")
    cameras: tuple[str, ...] = ()
    if not isinstance(cams, list) or not cams:
        problems.append("cameras must list the model's camera roles in input order")
    else:
        cameras = tuple(str(c) for c in cams)
        bad = [c for c in cameras if c not in ROLES]
        if bad:
            problems.append(f"cameras {bad} are not roles {list(ROLES)}")
        if len(set(cameras)) != len(cameras):
            problems.append("cameras repeats a role")

    state = _slots(raw.get("state"), "state", problems)

    offsets: dict[str, np.ndarray] = {}
    off_raw = raw.get("joint_offset_rad") or {}
    if not isinstance(off_raw, dict):
        problems.append("joint_offset_rad must map left/right to 6 values")
        off_raw = {}
    for k in off_raw:
        if k not in ARMS:
            problems.append(f"joint_offset_rad has unknown arm {k!r}")
    for arm in ARMS:
        v = np.asarray(off_raw.get(arm, [0.0] * JOINTS), dtype=np.float64).reshape(-1)
        if v.shape != (JOINTS,) or not np.all(np.isfinite(v)):
            problems.append(f"joint_offset_rad.{arm} needs 6 finite values, got {v.tolist()}")
            v = np.zeros(JOINTS)
        offsets[arm] = v

    g = raw.get("gripper") or {}
    if not isinstance(g, dict):
        problems.append("gripper must be a mapping")
        g = {}
    vals = {
        k: _number(g.get(k), f"gripper.{k}", problems)
        for k in ("station_open", "station_closed", "model_open", "model_closed")
    }
    for a, b in (("station_open", "station_closed"), ("model_open", "model_closed")):
        if vals[a] == vals[b]:
            problems.append(f"gripper.{a} equals gripper.{b}; the mapping is undefined")
    gripper = GripperMap(**vals)

    a = raw.get("action") or {}
    if not isinstance(a, dict):
        problems.append("action must be a mapping")
        a = {}
    kind = str(a.get("type", ""))
    if kind not in ACTION_TYPES:
        problems.append(f"action.type {kind!r} is not one of {list(ACTION_TYPES)}")
    ints = {}
    for k in ("dim", "horizon", "execute_steps"):
        try:
            ints[k] = int(a.get(k))
            if ints[k] < 1:
                raise ValueError
        except (TypeError, ValueError):
            problems.append(f"action.{k} must be a positive integer, got {a.get(k)!r}")
            ints[k] = 1
    hz = _number(a.get("hz"), "action.hz", problems)
    if not hz > 0:
        problems.append("action.hz must be > 0")
        hz = 1.0
    if ints["execute_steps"] > ints["horizon"]:
        problems.append(
            f"action.execute_steps {ints['execute_steps']} > action.horizon {ints['horizon']}"
        )
    delta_ref = str(a.get("delta_ref", "obs"))
    if delta_ref not in DELTA_REFS:
        problems.append(f"action.delta_ref {delta_ref!r} is not one of {list(DELTA_REFS)}")
    layout = _slots(a["layout"], "action.layout", problems) if "layout" in a else state
    if layout and ints["dim"] != len(layout):
        problems.append(f"action.dim {ints['dim']} != {len(layout)} action columns")
    if state and layout and {s.arm for s in layout} - {s.arm for s in state}:
        problems.append("action.layout drives an arm the state does not carry")

    age = _number(raw.get("max_obs_age_s", DEFAULT_MAX_OBS_AGE_S), "max_obs_age_s", problems)
    safety = raw.get("safety") or {}
    if not isinstance(safety, dict):
        problems.append("safety must be a mapping")
        safety = {}

    if problems:
        raise ProfileError(problems, name)
    return Profile(
        name=name,
        cameras=cameras,
        state=state,
        joint_offset_rad=offsets,
        gripper=gripper,
        action=ActionSpec(
            type=kind,
            dim=ints["dim"],
            horizon=ints["horizon"],
            execute_steps=ints["execute_steps"],
            hz=hz,
            delta_ref=delta_ref,
            layout=layout,
        ),
        checkpoint=dict(raw.get("checkpoint") or {}),
        extra=dict(raw.get("extra") or {}),
        safety=dict(safety),
        max_obs_age_s=age,
    )


def load_profile(source: str | Path | dict[str, Any]) -> Profile:
    """A profile from a YAML path, YAML text or an already parsed mapping."""
    if isinstance(source, dict):
        return profile_from_dict(source)
    path = Path(source) if not isinstance(source, str) or "\n" not in source else None
    text = path.read_text() if path is not None else str(source)
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ProfileError([f"not YAML: {e}"], str(path or "")) from e
    return profile_from_dict(raw)
