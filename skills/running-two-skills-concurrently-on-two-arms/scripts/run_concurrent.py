"""Run two hub skills at once, one thread per arm, every move gated by a shared clearance arbiter."""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

try:
    from gap import NodeContext
except ImportError:  # annotation only
    NodeContext = Any

logger = logging.getLogger(__name__)

#: Predicted capsule clearance a move must keep [m]. checking-dual-arm-clearance's
#: capsules are a lower bound on the YAM's collision meshes (too_close 5 mm); on top
#: of that the real YAM station's 12 mm arrival error and the D405 bracket, which
#: is not in the capsule model (about 15 mm off the hand housing).
MIN_CLEARANCE = 0.03
#: How long an arm waits for clearance before the run is a deadlock [s]: two of the
#: twin's longest single moves (a 0.4 m carry at the servo's speed takes ~6 s).
WAIT_TIMEOUT = 15.0
#: Re-check period while waiting [s]; a finished move wakes the waiter at once.
POLL = 0.05
#: Tool-centre separation every interleaved / sequential segment keeps from the other
#: arm's current tool centre [m]. The real dual_motion_server's live limit is 0.05 m
#: (min_tool_separation_m in nomad-dual-workspace.json, 2026-11-08), and a violation
#: powers both arms off, so twice that.
MIN_TCP_SEPARATION = 0.10
#: Interleaved segment length: the real motion server's BASE_STEP (0.05 m, 0.30 rad,
#: motion_server.py), one server step per segment, each waited to arrival (12 mm).
SEGMENT_M = 0.05
SEGMENT_RAD = 0.30
#: Back-off waypoints are recorded at least this far apart [m] (the server's 12 mm arrival).
BACKOFF_MIN_STEP = 0.012
MODES = ("auto", "threaded", "interleaved", "sequential")

#: Robot tools that move an arm; every one goes through the arbiter.
MOVES = {
    "robot.go_to_pose",
    "robot.go_to_pose_cartesian",
    "robot.move_to_joints",
    "robot.execute_trajectory",
    "robot.open_gripper",
    "robot.close_gripper",
    "robot.go_home",
    "robot.move_cartesian_until_contact",
    "robot.move_through_waypoints",
}
#: Robot tools that read both arms at once; arm_id is not injected.
BOTH_ARMS = {"robot.get_observation", "robot.describe_robot"}


class Output(TypedDict):
    route: str
    detail: str
    arm_a: int
    arm_b: int
    route_a: str
    route_b: str
    mode: str
    outputs_a: str
    outputs_b: str
    paused_arms: str
    sequential_fallback: bool
    waits: int
    min_predicted_clearance_m: float


class _Halted(BaseException):
    """Raised into a skill's thread when the run stops; BaseException so a skill's
    ``except Exception`` around a tool call does not swallow it."""


# ---------------------------------------------------------------- loading skills
def _skill_roots() -> list[Path]:
    roots = [Path(__file__).resolve().parents[2]]
    for entry in os.environ.get("GAP_SKILLS_PATH", "").split(os.pathsep):
        if entry:
            p = Path(entry)
            roots += [p / "skills", p]
    return roots


def _frontmatter(path: Path) -> dict:
    import yaml

    text = path.read_text()
    if not text.startswith("---"):
        return {}
    return yaml.safe_load(text.split("---", 2)[1]) or {}


#: name -> (run, success exits or None); tests register fake skills here.
RUNNERS: dict[str, tuple[Callable[..., Any], set[str] | None]] = {}


def load_skill(name: str) -> tuple[Callable[..., Any], set[str] | None]:
    """A hub skill's canonical ``run`` and its subgraph's success exits (None = any)."""
    if name in RUNNERS:
        return RUNNERS[name]
    for root in _skill_roots():
        md = root / name / "SKILL.md"
        if not md.is_file():
            continue
        meta = _frontmatter(md)
        scripts = (meta.get("gap") or {}).get("canonical_scripts") or []
        if not scripts:
            raise ValueError(f"skill {name} has no canonical script")
        rel = next(iter(scripts[0].values())) if isinstance(scripts[0], dict) else scripts[0]
        mod_name = f"rsh_concurrent_{name.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(mod_name, md.parent / rel)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        success = None
        example = md.parent / "examples" / "canonical_subgraph.json"
        if example.is_file():
            import json

            values = (json.loads(example.read_text()).get("exit") or {}).get("success_values")
            success = set(values) if values else None
        return module.run, success
    raise ValueError(f"no skill named {name} under {[str(r) for r in _skill_roots()]}")


def _clearance_module():
    return load_skill_module("checking-dual-arm-clearance", "scripts/check_clearance.py")


def load_skill_module(name: str, rel: str):
    mod_name = f"rsh_concurrent_{name.replace('-', '_')}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    for root in _skill_roots():
        path = root / name / rel
        if path.is_file():
            spec = importlib.util.spec_from_file_location(mod_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
            return module
    raise ValueError(f"{name}/{rel} not found")


# ---------------------------------------------------------------- the arbiter
def _plan(tool: str, kw: dict) -> tuple[str, Any] | None:
    """A move as checking-dual-arm-clearance inputs: ("trajectory", T), ("poses", P),
    ("grip", fraction) or ("static", None); None when it cannot be modelled."""
    if tool == "robot.move_to_joints":
        jc = kw.get("joint_config", kw.get("joints", kw.get("positions")))
        return ("trajectory", {"waypoints": [jc]}) if jc is not None else None
    if tool == "robot.execute_trajectory":
        t = kw.get("trajectory")
        return ("trajectory", t) if isinstance(t, dict) and t.get("waypoints") else None
    if tool in ("robot.go_to_pose", "robot.go_to_pose_cartesian") and kw.get("pose"):
        wp = {"pose": kw["pose"], "cartesian": tool == "robot.go_to_pose_cartesian"}
        return ("poses", {"waypoints": [wp]})
    if tool == "robot.open_gripper":
        return ("grip", 1.0)
    if tool == "robot.close_gripper":
        return ("static", None)  # the fingers only close in
    return None


def predict(ctx, arm: int, plan, other: int, other_plan) -> tuple[str, float, str]:
    """checking-dual-arm-clearance for ``arm``'s move against ``other``, standing or
    on its own move in flight (both orders, since the two are not synchronised)."""
    if plan is None:
        return "unknown", math.nan, "a move the clearance model cannot follow"
    check = _clearance_module().run
    kw: dict[str, Any] = {"arm_a": arm, "arm_b": other}
    kind, value = plan
    if kind in ("trajectory", "poses"):
        kw[f"{kind}_a"] = value
    elif kind == "grip":
        kw["grip_a"] = value
    orders = ["a_then_b"]
    if other_plan is not None:
        okind, ovalue = other_plan
        if okind in ("trajectory", "poses"):
            kw[f"{okind}_b"] = ovalue
            orders += ["b_then_a", "together"]
        elif okind == "grip":
            kw["grip_b"] = ovalue
    worst = None
    for order in orders:
        out = check(ctx, order=order, **kw)
        if out["route"] == "unknown":
            return "unknown", math.nan, out["detail"]
        if worst is None or out["min_clearance_m"] < worst["min_clearance_m"]:
            worst = out
    return "ok", float(worst["min_clearance_m"]), worst["detail"]


def _tcp_world(ctx, arm: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This arm's tool pose (observation frame), joints, and the map from its command
    frame (robot.get_ee_pose) to the observation frame."""
    cl = _clearance_module()
    reading = (ctx.tool("robot.get_observation") or {}).get("arms")[arm]
    t_obs = cl._pose_matrix(reading["ee_pose"])
    q = cl._positions(reading["joint_state"])
    try:
        t_cmd = cl._pose_matrix(ctx.tool("robot.get_ee_pose", arm_id=arm))
        to_obs = t_obs @ np.linalg.inv(t_cmd)
    except Exception:  # no separate command frame: take them as the same
        to_obs = np.eye(4)
    return t_obs, q, to_obs


def _point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    u = 0.0 if float(d @ d) == 0 else float(np.clip((p - a) @ d / (d @ d), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + u * d)))


def tcp_gap(ctx, arm: int, plan, other: int) -> float:
    """Least distance from the other arm's current tool centre to this arm's tool centre
    along the planned segment (straight line for poses, connector FK for joints)."""
    cl = _clearance_module()
    t_now, q_now, to_obs = _tcp_world(ctx, arm)
    p_other = _tcp_world(ctx, other)[0][:3, 3]
    points = [t_now[:3, 3]]
    kind, value = plan if plan else ("static", None)
    if kind == "poses":
        for wp in value.get("waypoints", []):
            points.append((to_obs @ cl._pose_matrix(wp["pose"] if "pose" in wp else wp))[:3, 3])
    elif kind == "trajectory":
        qs = cl._densify([q_now, *(cl._positions(w)[:6] for w in value.get("waypoints", []))], 0.05)
        fk = ctx.tool(
            "robot.forward_kinematics", joint_configs=[q.tolist() for q in qs], arm_id=arm
        )
        poses = [cl._pose_matrix(p) for p in fk["poses"]]
        rel = t_now @ np.linalg.inv(poses[0])
        points += [(rel @ t)[:3, 3] for t in poses[1:]]
    if len(points) == 1:
        return float(np.linalg.norm(points[0] - p_other))
    return min(_point_segment(p_other, a, b) for a, b in zip(points, points[1:], strict=False))


def with_tcp_separation(min_tcp: float, base=None):
    """predict, plus the tool-centre separation from the other arm standing where it is."""
    base = base or predict

    def predictor(ctx, arm, plan, other, other_plan):
        verdict, gap, detail = base(ctx, arm, plan, other, other_plan)
        if verdict != "ok" or min_tcp <= 0:
            return verdict, gap, detail
        try:
            sep = tcp_gap(ctx, arm, plan, other)
        except Exception as exc:
            return "unknown", math.nan, f"tool-centre separation unknown ({exc})"
        if sep < min_tcp:
            return (
                "ok",
                -1.0,
                (
                    f"tool centres would come {sep * 1e3:.0f} mm apart, under {min_tcp * 1e3:.0f} mm; {detail}"
                ),
            )
        return verdict, gap, detail

    return predictor


class _Fallback:
    """On a deadlock, either stop (``pause``) or go sequential for the rest (``sequential``):
    the arm that is not priority_arm backs off along its own path until the other arm's
    pending move clears, waits there until the other skill has finished, then goes on."""

    on_deadlock = "sequential"

    def _init_fallback(self) -> None:
        self.yield_arm: int | None = None
        self.yield_retreated = False
        self.fallback_at = 0.0
        self.finished_arms: set[int] = set()
        self.path: dict[int, list[dict]] = {}
        self.backoff_steps = 0

    def _try_fallback(self, arm: int, other: int, gap: float, detail: str) -> bool:
        """Called under the lock when ``arm`` timed out; True when the run goes sequential."""
        if self.on_deadlock != "sequential" or self.yield_arm is not None:
            return False
        if other in self.finished_arms:
            return False  # nothing left to wait for: the other skill is done and stays in the way
        self.yield_arm = other if arm == self.priority_arm else arm
        self.fallback_at = time.monotonic()
        self.log.append((self.yield_arm, "yield", self.fallback_at))
        self.fallback_why = (
            f"deadlock after {self.timeout:.1f} s (arm {arm}'s next move {gap * 1e3:.1f} mm from arm {other}; {detail}); "
            f"went sequential: arm {self.yield_arm} backed off along its path, then on after arm {self._other(self.yield_arm)} finished"
        )
        self.cond.notify_all()
        return True

    def _yielding(self, arm: int) -> bool:
        return self.yield_arm == arm and self._other(arm) not in self.finished_arms

    def _record(self, arm: int) -> None:
        """The arm's tool position after a move (command frame), for backing off along it."""
        try:
            pose = self.ctx.tool("robot.get_ee_pose", arm_id=arm)
        except Exception as exc:
            logger.info("arm %d: no tool pose to record (%s)", arm, exc)
            return
        pose = pose.get("pose", pose)
        p = {k: float(pose["position"][k]) for k in "xyz"}
        path = self.path.setdefault(arm, [])
        if (
            path
            and math.dist([path[-1][k] for k in "xyz"], [p[k] for k in "xyz"]) < BACKOFF_MIN_STEP
        ):
            return
        path.append(p)

    def _clear_for(self, other: int, pending, arm: int) -> bool:
        verdict, gap, _ = self.predictor(self.ctx, other, pending, arm, None)
        return verdict == "ok" and gap >= self.min_clearance

    def _retreat_yield(self, arm: int) -> None:
        """Under the lock: back the yielding arm off along its own recorded path (reverse
        order, straight Cartesian legs, its tool orientation held, so a grasp is kept)
        only until the other arm's pending move is predicted clear."""
        self.yield_retreated = True
        other = self._other(arm)
        pending = self._pending_of(other)
        if pending is None or self._clear_for(other, pending, arm):
            return
        try:
            now = self.ctx.tool("robot.get_ee_pose", arm_id=arm)
            rotation = now.get("pose", now)["rotation"]
        except Exception as exc:
            self.halt, self.why = (
                "deadlock",
                f"deadlock; arm {arm} has no tool pose to back off from ({exc})",
            )
            self.cond.notify_all()
            return
        path = self.path.setdefault(arm, [])
        while len(path) > 1:
            path.pop()  # where it stands
            pose = {"position": dict(path[-1]), "rotation": rotation}
            plan = ("poses", {"waypoints": [{"pose": pose, "cartesian": True}]})
            verdict, gap, detail = self.predictor(self.ctx, arm, plan, other, None)
            if verdict != "ok" or gap < self.min_clearance:
                self.halt = "deadlock"
                self.why = (
                    f"deadlock; arm {arm} cannot back off further ({gap * 1e3:.1f} mm, {detail})"
                )
                self.cond.notify_all()
                return
            with self.motion:
                try:
                    reply = self.ctx.tool("robot.go_to_pose_cartesian", pose=pose, arm_id=arm)
                except Exception as exc:
                    reply = {"reached": False, "reason": str(exc)}
            if not _arrived(reply):
                self.halt = "deadlock"
                self.why = f"deadlock; arm {arm}'s back-off move was refused ({reply})"
                self.cond.notify_all()
                return
            self.backoff_steps += 1
            self.log.append((arm, "backed_off", time.monotonic()))
            if self._clear_for(other, pending, arm):
                self.cond.notify_all()
                return
        self.halt = "deadlock"
        self.why = f"deadlock; arm {arm} backed off its whole path and arm {other}'s move is still not clear"
        self.cond.notify_all()


class Arbiter(_Fallback):
    """Grants one move at a time per arm; a move whose predicted clearance against the
    other arm (standing, or on its move in flight) is under ``min_clearance`` waits."""

    def __init__(self, ctx, arms, priority, min_clearance, timeout, predictor=predict):
        self.ctx = ctx
        self.arms = list(arms)
        self.priority = priority
        self.min_clearance = float(min_clearance)
        self.timeout = float(timeout)
        self.predictor = predictor
        self.cond = threading.Condition()
        self.pending: dict[int, Any] = {}
        self.blocked = {a: False for a in self.arms}
        self.inflight: dict[int, Any] = {a: None for a in self.arms}
        self.halt: str = ""  # "", "failed", "deadlock", "unsafe"
        self.why = ""
        self.waits = 0
        self.min_seen = math.inf
        self.log: list[tuple[int, str, float]] = []
        self.seen: set[int] = set()  # arms that have asked for a move or finished
        self.motion = threading.Lock()  # held by a retreat
        self.priority_arm = priority
        self._init_fallback()

    def stop(self, kind: str, why: str) -> None:
        with self.cond:
            if not self.halt:
                self.halt, self.why = kind, why
            self.cond.notify_all()

    def _other(self, arm: int) -> int:
        return self.arms[1] if arm == self.arms[0] else self.arms[0]

    def acquire(self, arm: int, plan) -> None:
        other = self._other(arm)
        started = time.monotonic()
        counted = False
        with self.cond:
            self.pending[arm] = plan
            self.seen.add(arm)
            self.cond.notify_all()
            try:
                while True:
                    if self.halt:
                        raise _Halted(self.halt)
                    if self._yielding(arm):
                        if not self.yield_retreated or self.blocked[other]:
                            self._retreat_yield(arm)
                        self.blocked[arm] = True
                        self.cond.notify_all()
                        self.cond.wait(POLL)
                        continue
                    if (
                        other == self.priority
                        and not self._yielding(other)
                        and (
                            other not in self.seen
                            or (other in self.pending and not self.blocked[other])
                        )
                    ):
                        self.cond.wait(POLL)  # the priority arm decides first
                        continue
                    verdict, gap, detail = self.predictor(
                        self.ctx, arm, plan, other, self.inflight[other]
                    )
                    if verdict == "unknown":
                        self.halt, self.why = "unsafe", f"arm {arm}: clearance unknown ({detail})"
                        self.cond.notify_all()
                        raise _Halted("unsafe")
                    if gap >= self.min_clearance:
                        self.min_seen = min(self.min_seen, gap)
                        self.inflight[arm] = plan
                        self.blocked[arm] = False
                        self.log.append((arm, "go", time.monotonic()))
                        return
                    if not counted:
                        self.waits += 1
                        counted = True
                        self.log.append((arm, "wait", time.monotonic()))
                    self.blocked[arm] = True
                    self.cond.notify_all()
                    if time.monotonic() >= max(started, self.fallback_at) + self.timeout:
                        if self._try_fallback(arm, other, gap, detail):
                            continue
                        self.halt = "deadlock"
                        self.why = (
                            f"arm {arm} waited {self.timeout:.1f} s for clearance; "
                            f"its next move would come {gap * 1e3:.1f} mm from arm {other} ({detail})"
                        )
                        self.cond.notify_all()
                        raise _Halted("deadlock")
                    self.cond.wait(POLL)
            finally:
                self.pending.pop(arm, None)

    def finished(self, arm: int) -> None:
        with self.cond:
            self.seen.add(arm)
            self.finished_arms.add(arm)
            self.cond.notify_all()

    def execute(self, arm: int, name: str, kwargs: dict):
        self.acquire(arm, _plan(name, kwargs))
        try:
            return self.ctx.tool(name, **kwargs)
        finally:
            self.release(arm)

    def _pending_of(self, other: int):
        return self.pending.get(other)

    def release(self, arm: int) -> None:
        self._record(arm)
        with self.cond:
            self.inflight[arm] = None
            self.log.append((arm, "done", time.monotonic()))
            for a in self.arms:
                self.blocked[a] = False
            self.cond.notify_all()


def _segments(ctx, arm: int, name: str, kw: dict, seg_m: float, seg_rad: float):
    """A move cut into server-step segments: [(tool, kwargs, plan)]."""
    cl = _clearance_module()
    if name in ("robot.move_to_joints", "robot.execute_trajectory"):
        plan = _plan(name, kw)
        if plan is None:
            return [(name, kw, None)]
        reading = (ctx.tool("robot.get_observation") or {}).get("arms")[arm]
        prev = cl._positions(reading["joint_state"])
        extra = {
            k: v
            for k, v in kw.items()
            if k not in ("joint_config", "joints", "positions", "trajectory")
        }
        out = []
        for w in plan[1]["waypoints"]:
            target = cl._positions(w)
            n = max(1, math.ceil(float(np.max(np.abs(target - prev))) / seg_rad))
            for i in range(1, n + 1):
                q = prev + (target - prev) * i / n
                jc = {"positions": [float(v) for v in q]}
                out.append(
                    (
                        "robot.move_to_joints",
                        {**extra, "joint_config": jc},
                        ("trajectory", {"waypoints": [jc]}),
                    )
                )
            prev = target
        return out
    if name in ("robot.go_to_pose", "robot.go_to_pose_cartesian") and kw.get("pose"):
        start = cl._pose_matrix(ctx.tool("robot.get_ee_pose", arm_id=arm))
        target = cl._pose_matrix(kw["pose"])
        n = max(
            1,
            math.ceil(float(np.linalg.norm(target[:3, 3] - start[:3, 3])) / seg_m),
            math.ceil(cl._angle(start, target) / seg_rad),
        )
        out = []
        for i in range(1, n + 1):
            pose = kw["pose"] if i == n else cl._pose_dict(cl._slerp_pose(start, target, i / n))
            seg = {**kw, "pose": pose}
            out.append((name, seg, _plan(name, seg)))
        return out
    return [(name, kw, _plan(name, kw))]


def _whole(segs) -> Any:
    """The remaining segments of a move as one clearance plan (None if any is unmodelled)."""
    plans = [p for _, _, p in segs]
    if not plans or any(p is None for p in plans):
        return None
    kinds = {p[0] for p in plans}
    if len(plans) == 1 or len(kinds) > 1 or kinds & {"grip", "static"}:
        return plans[0]
    kind = plans[0][0]
    return (kind, {"waypoints": [w for p in plans for w in p[1]["waypoints"]]})


def _arrived(reply) -> bool:
    if not isinstance(reply, dict):
        return True
    return reply.get("reached", True) is not False and reply.get("success", True) is not False


class Dispatcher(_Fallback):
    """For a server that moves one arm at a time: the two skills' moves are cut into
    server steps and sent from one thread, A1, B1, A2, B2, ..., each waited to arrival
    and checked against the other arm standing where it is now."""

    def __init__(self, ctx, arms, priority, min_clearance, timeout, predictor, seg_m, seg_rad):
        self.ctx = ctx
        self.arms = list(arms)
        self.turn = priority if priority in self.arms else self.arms[0]
        self.min_clearance = float(min_clearance)
        self.timeout = float(timeout)
        self.predictor = predictor
        self.seg = (float(seg_m), float(seg_rad))
        self.cond = threading.Condition()
        self.queue: dict[int, dict | None] = {a: None for a in self.arms}
        self.inflight: dict[int, Any] = {a: None for a in self.arms}
        self.done: set[int] = set()
        self.halt = ""
        self.why = ""
        self.waits = 0
        self.min_seen = math.inf
        self.log: list[tuple[int, str, float]] = []
        self.motion = threading.Lock()
        self.priority_arm = self.turn
        self._init_fallback()

    def _other(self, arm: int) -> int:
        return self.arms[1] if arm == self.arms[0] else self.arms[0]

    def stop(self, kind: str, why: str) -> None:
        with self.cond:
            if not self.halt:
                self.halt, self.why = kind, why
            self.cond.notify_all()

    def finished(self, arm: int) -> None:
        with self.cond:
            self.done.add(arm)
            self.finished_arms.add(arm)
            self.cond.notify_all()

    def _pending_of(self, other: int):
        job = self.queue[other]
        return None if job is None or job["end"] else _whole(job["segs"][job["i"] :])

    def execute(self, arm: int, name: str, kwargs: dict):
        """Called from a skill's thread: queue the move, block until all its segments ran."""
        segs = _segments(self.ctx, arm, name, kwargs, *self.seg)
        job = {
            "segs": segs,
            "i": 0,
            "reply": None,
            "error": None,
            "end": False,
            "since": None,
            "call": (name, kwargs),
        }
        with self.cond:
            if self.halt:
                raise _Halted(self.halt)
            self.queue[arm] = job
            self.cond.notify_all()
            while not job["end"] and not self.halt:
                self.cond.wait(POLL)
            self.queue[arm] = None
        if job["error"] is not None:
            raise job["error"]
        if not job["end"]:
            raise _Halted(self.halt)
        return job["reply"]

    def loop(self) -> None:
        while True:
            with self.cond:
                if self.halt or all(a in self.done for a in self.arms):
                    self.cond.notify_all()
                    return
                arm = self.turn
                other = self._other(arm)
                if self._yielding(arm):
                    theirs = self.queue[other]
                    if not self.yield_retreated or (
                        theirs is not None and theirs["since"] is not None
                    ):
                        self._retreat_yield(arm)
                        for j in self.queue.values():
                            if j is not None:
                                j["since"] = None
                        mine = self.queue[arm]
                        if mine is not None and not mine["end"]:
                            # its pending move was cut from where it stood before going back
                            mine["segs"] = _segments(self.ctx, arm, *mine["call"], *self.seg)
                            mine["i"] = 0
                    self.turn = other
                    self.cond.wait(POLL)
                    continue
                job = self.queue[arm]
                if job is None or job["end"]:
                    nxt = self.queue[other]
                    if arm in self.done and nxt is not None and not nxt["end"]:
                        self.turn = other
                    else:
                        self.cond.wait(POLL)  # strict turns: wait for this arm's next move
                    continue
                tool, kw, plan = job["segs"][job["i"]]
                committed = None
                if job["i"] == 0:
                    # a move starts only if all of it clears the other arm's committed rest
                    plan = _whole(job["segs"])
                    nxt = self.queue[other]
                    if (
                        nxt is not None
                        and not nxt["end"]
                        and nxt["i"] > 0
                        and not self._yielding(other)
                    ):
                        committed = _whole(nxt["segs"][nxt["i"] :])
                verdict, gap, detail = self.predictor(self.ctx, arm, plan, other, committed)
                if verdict == "unknown":
                    self.halt, self.why = "unsafe", f"arm {arm}: clearance unknown ({detail})"
                    self.cond.notify_all()
                    return
                if gap < self.min_clearance:
                    now = time.monotonic()
                    if job["since"] is None:
                        job["since"] = now
                        self.waits += 1
                        self.log.append((arm, "wait", now))
                    if now - max(job["since"], self.fallback_at) >= self.timeout:
                        if self._try_fallback(arm, other, gap, detail):
                            continue
                        self.halt = "deadlock"
                        self.why = (
                            f"arm {arm} waited {self.timeout:.1f} s for clearance; "
                            f"its next segment would come {gap * 1e3:.1f} mm from arm {other} ({detail})"
                        )
                        self.cond.notify_all()
                        return
                    nxt = self.queue[other]
                    if nxt is not None and not nxt["end"]:
                        self.turn = other  # let the other arm take its step
                    else:
                        self.cond.wait(POLL)
                    continue
                job["since"] = None
                self.min_seen = min(self.min_seen, gap)
                self.log.append((arm, "go", time.monotonic()))
            try:
                with self.motion:
                    reply, error = self.ctx.tool(tool, **kw), None
            except Exception as exc:
                reply, error = None, exc
            self._record(arm)
            with self.cond:
                self.log.append((arm, "done", time.monotonic()))
                job["i"] += 1
                job["reply"] = reply
                if error is not None:
                    job["error"] = error
                    job["end"] = True
                elif not _arrived(reply) or job["i"] >= len(job["segs"]):
                    job["end"] = True  # a segment that did not arrive ends the move with its reply
                self.turn = other
                self.cond.notify_all()


#: Guard poll period [s] and the stop tools tried on a breach (first one the connector has).
GUARD_PERIOD = 0.05
STOP_TOOLS = ("robot.stop", "robot.halt")


def guard(ctx, arbiter, arms, min_clearance: float, stop: threading.Event) -> None:
    """Measured, not predicted: the arms' capsule clearance as they stand, every
    GUARD_PERIOD while the skills run. Under min_clearance both arms are stopped
    (no further move is granted, and robot.stop is sent to both if the connector has it)."""
    check = _clearance_module().run
    while not stop.wait(GUARD_PERIOD):
        try:
            now = check(ctx, arm_a=arms[0], arm_b=arms[1])
        except Exception as exc:
            logger.warning("guard: no reading (%s)", exc)
            continue
        gap = now.get("min_clearance_m", math.nan)
        if now.get("route") == "unknown" or not (gap >= min_clearance):
            arbiter.stop(
                "unsafe",
                f"guard: measured clearance {gap * 1e3:.1f} mm under {min_clearance * 1e3:.0f} mm "
                f"({now.get('detail', '')}); both arms stopped",
            )
            for arm in arms:
                for tool in STOP_TOOLS:
                    try:
                        ctx.tool(tool, arm_id=arm)
                        break
                    except Exception:
                        continue
            arbiter.guard_tripped = True
            return


class ArmContext:
    """The run's ctx bound to one arm: robot.* calls get its arm_id, moves go
    through the arbiter, and a halted run pauses the arm at its safe pose."""

    def __init__(self, ctx, arm: int, arbiter: Arbiter, safe_joints):
        self._ctx = ctx
        self.arm = arm
        self._arbiter = arbiter
        self._safe = safe_joints
        self.paused = False

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def _retreat(self) -> None:
        """Back to the joints the arm started from, if that is clear; then stop."""
        arb = self._arbiter
        if self.paused or self._safe is None:
            self.paused = True
            return
        self.paused = True
        plan = ("trajectory", {"waypoints": [{"positions": list(self._safe)}]})
        with arb.motion:
            with arb.cond:
                verdict, gap, _ = arb.predictor(
                    self._ctx,
                    self.arm,
                    plan,
                    arb._other(self.arm),
                    arb.inflight[arb._other(self.arm)],
                )
            if verdict != "ok" or gap < arb.min_clearance:
                return
            try:
                self._ctx.tool(
                    "robot.move_to_joints",
                    joint_config={"positions": list(self._safe)},
                    arm_id=self.arm,
                )
            except Exception as exc:
                logger.warning("arm %d: retreat refused (%s)", self.arm, exc)

    def tool(self, name: str, **kwargs):
        if not name.startswith("robot.") or name in BOTH_ARMS:
            return self._ctx.tool(name, **kwargs)
        kwargs["arm_id"] = self.arm
        if name not in MOVES:
            return self._ctx.tool(name, **kwargs)
        if self._arbiter.halt:
            if self._arbiter.halt != "unsafe":
                self._retreat()
            raise _Halted(self._arbiter.halt)
        try:
            return self._arbiter.execute(self.arm, name, kwargs)
        except _Halted:
            if self._arbiter.halt != "unsafe":
                self._retreat()
            raise


# ---------------------------------------------------------------- assignment
def _xyz(v) -> tuple[float, float, float]:
    if isinstance(v, dict):
        v = v.get("position", v)
        return float(v["x"]), float(v["y"]), float(v["z"])
    return tuple(float(x) for x in v)[:3]


def _assign(ctx, arms, target_a, target_b, obs) -> tuple[int, int, str]:
    """By IK reach (solve_ik at each target with the arm's current tool rotation),
    then the smaller sum of horizontal distances from the tool centres."""
    reach: dict[tuple[int, int], bool] = {}
    tool_xy: dict[int, tuple[float, float]] = {}
    for arm in arms:
        pose = obs[arm]["ee_pose"]
        pose = pose.get("pose", pose)
        tool_xy[arm] = _xyz(pose["position"])[:2]
        for k, target in enumerate((target_a, target_b)):
            x, y, z = _xyz(target)
            try:
                reply = ctx.tool(
                    "robot.solve_ik",
                    pose={"position": {"x": x, "y": y, "z": z}, "rotation": pose["rotation"]},
                    arm_id=arm,
                )
                ok = isinstance(reply, dict) and reply.get("success") is not False
                ok = ok and reply.get("within_servo_tolerance") is not False
            except Exception:
                ok = False
            reach[(arm, k)] = ok

    def dist(arm, target):
        x, y, _ = _xyz(target)
        return math.hypot(x - tool_xy[arm][0], y - tool_xy[arm][1])

    options = []
    for a, b in ((arms[0], arms[1]), (arms[1], arms[0])):
        missing = (not reach[(a, 0)]) + (not reach[(b, 1)])
        options.append((missing, dist(a, target_a) + dist(b, target_b), a, b))
    missing, _, a, b = min(options)
    if missing:
        return -1, -1, "no assignment of the two skills to the two arms reaches both targets"
    return a, b, f"auto: skill a on arm {a}, skill b on arm {b} (IK reach, then nearer side)"


# ---------------------------------------------------------------- the skill
def detect_mode(ctx) -> tuple[str, str]:
    """threaded when robot.describe_robot says both arms move at once (concurrent_arms:
    true, or a move_dual / stream_joints op), otherwise interleaved: the real
    dual_motion_server runs one move at a time behind one lock."""
    try:
        info = ctx.tool("robot.describe_robot") or {}
    except Exception as exc:
        return "interleaved", f"mode interleaved (no robot.describe_robot: {exc})"
    if not isinstance(info, dict):
        return "interleaved", "mode interleaved (robot.describe_robot gave no mapping)"
    ops = set(info.get("ops") or [])
    if info.get("serializes_moves") is True:
        return "interleaved", "mode interleaved (the connector moves one arm at a time)"
    if info.get("concurrent_arms") is True or ops & {"move_dual", "stream_joints"}:
        return "threaded", "mode threaded (the connector moves both arms at once)"
    return "interleaved", "mode interleaved (the connector does not say it moves both arms at once)"


def run(
    ctx: NodeContext,
    skill_a: str,
    skill_b: str,
    inputs_a: dict | None = None,
    inputs_b: dict | None = None,
    arm_a: int = 0,
    arm_b: int = 1,
    target_a: dict | None = None,
    target_b: dict | None = None,
    priority_arm: int = 0,
    min_clearance_m: float = MIN_CLEARANCE,
    wait_timeout_s: float = WAIT_TIMEOUT,
    mode: str = "auto",
    min_tcp_separation_m: float = MIN_TCP_SEPARATION,
    segment_m: float = SEGMENT_M,
    segment_rad: float = SEGMENT_RAD,
    on_deadlock: str = "sequential",
) -> Output:
    out: Output = {
        "route": "refused",
        "mode": str(mode),
        "detail": "",
        "arm_a": int(arm_a),
        "arm_b": int(arm_b),
        "route_a": "",
        "route_b": "",
        "outputs_a": "{}",
        "outputs_b": "{}",
        "paused_arms": "",
        "sequential_fallback": False,
        "waits": 0,
        "min_predicted_clearance_m": math.nan,
    }

    def refuse(why: str) -> Output:
        out["detail"] = why
        return out

    try:
        run_a, ok_a = load_skill(skill_a)
        run_b, ok_b = load_skill(skill_b)
    except Exception as exc:
        return refuse(f"cannot load the skills: {exc}")
    try:
        obs = (ctx.tool("robot.get_observation") or {}).get("arms") or []
    except Exception as exc:
        return refuse(f"no observation ({exc})")
    if len(obs) < 2:
        return refuse(f"the connector reports {len(obs)} arm(s); this needs two")

    notes = []
    if int(arm_a) < 0 or int(arm_b) < 0:
        if target_a is None or target_b is None:
            return refuse("auto assignment (arm_a / arm_b = -1) needs target_a and target_b")
        a, b, why = _assign(ctx, [0, 1], target_a, target_b, obs)
        if a < 0:
            return refuse(why + "; nothing moved")
        arm_a, arm_b = a, b
        notes.append(why)
    arm_a, arm_b = int(arm_a), int(arm_b)
    out["arm_a"], out["arm_b"] = arm_a, arm_b
    if arm_a == arm_b or max(arm_a, arm_b) >= len(obs):
        return refuse(f"arms {arm_a} and {arm_b} are not two arms of this connector")

    if on_deadlock not in ("sequential", "pause"):
        return refuse(f"on_deadlock must be sequential or pause, not {on_deadlock!r}")
    if mode not in MODES:
        return refuse(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
    if mode == "auto":
        mode, why = detect_mode(ctx)
        notes.append(why)
    out["mode"] = mode
    arms = [arm_a, arm_b]
    if mode == "threaded":
        arbiter = Arbiter(ctx, arms, int(priority_arm), min_clearance_m, wait_timeout_s)
    elif mode == "interleaved":
        arbiter = Dispatcher(
            ctx,
            arms,
            int(priority_arm),
            min_clearance_m,
            wait_timeout_s,
            with_tcp_separation(float(min_tcp_separation_m)),
            segment_m,
            segment_rad,
        )
    else:
        arbiter = Arbiter(
            ctx,
            arms,
            int(priority_arm),
            min_clearance_m,
            wait_timeout_s,
            predictor=with_tcp_separation(float(min_tcp_separation_m)),
        )
        arbiter.seen = set(arms)  # one skill after the other: nobody to yield to
    verdict, gap, detail = arbiter.predictor(ctx, arm_a, ("static", None), arm_b, None)
    if verdict == "unknown":
        return refuse(f"clearance unknown before starting ({detail}); nothing moved")
    if gap < float(min_clearance_m):
        return refuse(
            f"the arms are already {gap * 1e3:.1f} mm apart, under {float(min_clearance_m) * 1e3:.0f} mm; nothing moved"
        )
    arbiter.min_seen = gap

    def safe(arm):
        js = obs[arm].get("joint_state")
        js = js.get("positions", js) if isinstance(js, dict) else js
        return [float(v) for v in js] if js is not None else None

    arbiter.on_deadlock = on_deadlock if mode != "sequential" else "pause"
    for arm in arms:
        arbiter._record(arm)

    jobs = {
        "a": (run_a, ok_a, ArmContext(ctx, arm_a, arbiter, safe(arm_a)), inputs_a or {}),
        "b": (run_b, ok_b, ArmContext(ctx, arm_b, arbiter, safe(arm_b)), inputs_b or {}),
    }
    result: dict[str, tuple[str, Any]] = {}
    failed: list[str] = []

    def work(key: str) -> None:
        fn, success, actx, kw = jobs[key]
        try:
            _work(key, fn, success, actx, kw)
        finally:
            arbiter.finished(actx.arm)

    def _work(key, fn, success, actx, kw) -> None:
        try:
            reply = fn(actx, **kw)
        except _Halted:
            result[key] = ("paused" if actx.paused else "stopped", {})
            return
        except Exception as exc:
            result[key] = ("raised", {"error": f"{type(exc).__name__}: {exc}"})
            failed.append(key)
            arbiter.stop("failed", f"skill {key} on arm {actx.arm} raised {exc}")
            return
        route = str((reply or {}).get("route", "")) if isinstance(reply, dict) else ""
        result[key] = (route or "done", reply if isinstance(reply, dict) else {"value": reply})
        if success is not None and route not in success:
            failed.append(key)
            arbiter.stop("failed", f"skill {key} on arm {actx.arm} ended {route!r}")

    if mode == "sequential":
        for key in jobs:
            if arbiter.halt:
                result[key] = ("not_run", {})
                arbiter.finished(jobs[key][2].arm)
                continue
            work(key)
    else:
        threads = [threading.Thread(target=work, args=(k,), name=f"arm-{k}") for k in jobs]
        if mode == "interleaved":
            threads.append(threading.Thread(target=arbiter.loop, name="dispatcher"))
        halt_guard = threading.Event()
        watcher = threading.Thread(
            target=guard, args=(ctx, arbiter, arms, float(min_clearance_m), halt_guard), daemon=True
        )
        watcher.start()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        halt_guard.set()
        watcher.join(timeout=5)

    import json

    for key in ("a", "b"):
        route, reply = result.get(key, ("stopped", {}))
        out[f"route_{key}"] = route
        out[f"outputs_{key}"] = json.dumps(reply, default=str)
    out["paused_arms"] = ",".join(str(j[2].arm) for j in jobs.values() if j[2].paused)
    out["waits"] = arbiter.waits
    out["sequential_fallback"] = arbiter.yield_arm is not None
    out["min_predicted_clearance_m"] = arbiter.min_seen
    out["route"] = {
        "unsafe": "refused",
        "deadlock": "deadlock_timeout",
        "failed": "both_failed" if len(failed) == 2 else "one_failed",
    }.get(arbiter.halt, "both_succeeded")
    parts = [
        f"skill a ({skill_a}) on arm {arm_a}: {out['route_a']}",
        f"skill b ({skill_b}) on arm {arm_b}: {out['route_b']}",
    ]
    if arbiter.yield_arm is not None:
        parts.append(arbiter.fallback_why)
    if arbiter.why:
        parts.append(arbiter.why)
    if out["paused_arms"]:
        parts.append(f"paused at the start pose (or where it stood): arm {out['paused_arms']}")
    if arbiter.waits:
        parts.append(f"{arbiter.waits} wait(s) for clearance")
    out["detail"] = "; ".join(notes + parts)
    return out
