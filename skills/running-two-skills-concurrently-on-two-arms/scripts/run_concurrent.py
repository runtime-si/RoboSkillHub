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
    outputs_a: str
    outputs_b: str
    paused_arms: str
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
            orders.append("b_then_a")
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


class Arbiter:
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

    def stop(self, kind: str, why: str) -> None:
        with self.cond:
            if not self.halt:
                self.halt, self.why = kind, why
            self.cond.notify_all()

    def _other(self, arm: int) -> int:
        return self.arms[1] if arm == self.arms[0] else self.arms[0]

    def acquire(self, arm: int, plan) -> None:
        other = self._other(arm)
        deadline = time.monotonic() + self.timeout
        counted = False
        with self.cond:
            self.pending[arm] = plan
            self.seen.add(arm)
            self.cond.notify_all()
            try:
                while True:
                    if self.halt:
                        raise _Halted(self.halt)
                    if other == self.priority and (
                        other not in self.seen
                        or (other in self.pending and not self.blocked[other])
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
                    if time.monotonic() >= deadline:
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
            self.cond.notify_all()

    def release(self, arm: int) -> None:
        with self.cond:
            self.inflight[arm] = None
            self.log.append((arm, "done", time.monotonic()))
            for a in self.arms:
                self.blocked[a] = False
            self.cond.notify_all()


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
        with arb.cond:
            verdict, gap, _ = arb.predictor(
                self._ctx, self.arm, plan, arb._other(self.arm), arb.inflight[arb._other(self.arm)]
            )
        if verdict == "ok" and gap >= arb.min_clearance:
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
            self._arbiter.acquire(self.arm, _plan(name, kwargs))
        except _Halted:
            if self._arbiter.halt != "unsafe":
                self._retreat()
            raise
        try:
            return self._ctx.tool(name, **kwargs)
        finally:
            self._arbiter.release(self.arm)


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
) -> Output:
    out: Output = {
        "route": "unsafe_refused",
        "detail": "",
        "arm_a": int(arm_a),
        "arm_b": int(arm_b),
        "route_a": "",
        "route_b": "",
        "outputs_a": "{}",
        "outputs_b": "{}",
        "paused_arms": "",
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

    arbiter = Arbiter(ctx, [arm_a, arm_b], int(priority_arm), min_clearance_m, wait_timeout_s)
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

    jobs = {
        "a": (run_a, ok_a, ArmContext(ctx, arm_a, arbiter, safe(arm_a)), inputs_a or {}),
        "b": (run_b, ok_b, ArmContext(ctx, arm_b, arbiter, safe(arm_b)), inputs_b or {}),
    }
    result: dict[str, tuple[str, Any]] = {}

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
            arbiter.stop("failed", f"skill {key} on arm {actx.arm} raised {exc}")
            return
        route = str((reply or {}).get("route", "")) if isinstance(reply, dict) else ""
        result[key] = (route or "done", reply if isinstance(reply, dict) else {"value": reply})
        if success is not None and route not in success:
            arbiter.stop("failed", f"skill {key} on arm {actx.arm} ended {route!r}")

    threads = [threading.Thread(target=work, args=(k,), name=f"arm-{k}") for k in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    import json

    for key in ("a", "b"):
        route, reply = result.get(key, ("stopped", {}))
        out[f"route_{key}"] = route
        out[f"outputs_{key}"] = json.dumps(reply, default=str)
    out["paused_arms"] = ",".join(str(j[2].arm) for j in jobs.values() if j[2].paused)
    out["waits"] = arbiter.waits
    out["min_predicted_clearance_m"] = arbiter.min_seen
    out["route"] = {
        "unsafe": "unsafe_refused",
        "deadlock": "deadlock_timeout",
        "failed": "one_failed",
    }.get(arbiter.halt, "both_done")
    parts = [
        f"skill a ({skill_a}) on arm {arm_a}: {out['route_a']}",
        f"skill b ({skill_b}) on arm {arm_b}: {out['route_b']}",
    ]
    if arbiter.why:
        parts.append(arbiter.why)
    if out["paused_arms"]:
        parts.append(f"paused at the start pose (or where it stood): arm {out['paused_arms']}")
    if arbiter.waits:
        parts.append(f"{arbiter.waits} wait(s) for clearance")
    out["detail"] = "; ".join(notes + parts)
    return out
