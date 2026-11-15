"""Run a queue of pick-and-place tasks with the next task's perception and planning done while the arm moves."""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, TypedDict

try:
    from gap import NodeContext
except ImportError:  # annotation only
    NodeContext = Any

logger = logging.getLogger(__name__)

CONCURRENT = "running-two-skills-concurrently-on-two-arms"
#: Tool-pose difference between the expected end of task k and where the arm really stands
#: at the start of task k+1 that makes the prepared plan stale [m]: the real YAM station's
#: 12 mm arrival tolerance (motion_server REACHED_M).
POSE_TOLERANCE = 0.012
#: Object displacement between the perception the plan used and the latest one that makes
#: it stale [m]: a 30 mm cube's half width, under which the same grasp still closes on it.
OBJECT_TOLERANCE = 0.015
#: Planning tools whose replies are kept for the real run, keyed by their exact arguments.
PLANNING = {"robot.solve_ik", "robot.grasp_frame", "robot.forward_kinematics", "robot.describe_arm"}
#: Robot reads answered from the expected end state during the look-ahead.
STATE = {"robot.get_ee_pose", "robot.get_gripper"}
#: Tools that read both arms; arm_id is not injected.
BOTH_ARMS = {"robot.get_observation", "robot.describe_robot"}


class Output(TypedDict):
    route: str
    detail: str
    done: int
    failed: int
    stale: int
    makespan_s: float
    idle_s: float
    cache_hits: int
    cache_misses: int
    tasks: str


class _LookaheadDone(BaseException):
    """Raised in the look-ahead at the child's first move: everything before it is planning."""


def _concurrent():
    """#246's module (Arbiter, ArmContext, MOVES, load_skill), from this hub's skills/."""
    name = "rsh_pipeline_" + CONCURRENT.replace("-", "_")
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[2] / CONCURRENT / "scripts" / "run_concurrent.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _key(name: str, kw: dict) -> str:
    return name + json.dumps(kw, sort_keys=True, default=str)


def _xyz(pose) -> tuple[float, float, float]:
    p = pose.get("pose", pose) if isinstance(pose, dict) else pose
    p = p.get("position", p) if isinstance(p, dict) else p
    if isinstance(p, dict) and "center" in p:
        p = p["center"]
    return float(p["x"]), float(p["y"]), float(p["z"])


class Plan:
    """Task k+1 prepared while task k runs: its perception, and the planning replies its
    child skill asked for, from task k's expected end state."""

    def __init__(self, task: dict):
        self.task = task
        self.obb: dict | None = None
        self.cache: dict[str, Any] = {}
        self.start_pose: dict | None = None
        self.error = ""
        self.latest_obb: dict | None = None
        self.ready = threading.Event()


class _LookaheadContext:
    """The child skill run up to its first move: planning tools go to the connector and are
    kept, the arm's own state reads answer from the expected end state, moves stop it."""

    def __init__(self, ctx, arm: int, moves: set[str], plan: Plan):
        self._ctx, self.arm, self._moves, self._plan = ctx, arm, moves, plan

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def tool(self, name: str, **kw):
        if name.startswith("robot.") and name not in BOTH_ARMS:
            kw["arm_id"] = self.arm
        if name in self._moves:
            raise _LookaheadDone()
        if name == "robot.get_ee_pose" and self._plan.start_pose is not None:
            return {"pose": self._plan.start_pose}
        if name == "robot.get_gripper":
            return {"position": 1.0}
        reply = self._ctx.tool(name, **kw)
        if name in PLANNING:
            self._plan.cache[_key(name, kw)] = reply
        return reply


class _LiveContext:
    """The arm's ctx (arm_id bound, moves arbitrated when two arms run) with the prepared
    planning replies served for identical calls; notes the first move's time."""

    def __init__(self, arm_ctx, cache: dict[str, Any], moves: set[str], arm: int):
        self._ctx, self._cache, self._moves, self.arm = arm_ctx, cache, moves, arm
        self.hits = self.misses = 0
        self.first_move: float | None = None

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def tool(self, name: str, **kw):
        if name.startswith("robot.") and name not in BOTH_ARMS:
            kw["arm_id"] = self.arm
        if name in PLANNING:
            k = _key(name, kw)
            if k in self._cache:
                self.hits += 1
                return self._cache[k]
            self.misses += 1
        elif name in self._moves and self.first_move is None:
            self.first_move = time.monotonic()
        return self._ctx.tool(name, **kw)


class _ArmBinder:
    """One arm, nothing to arbitrate against: robot.* calls get its arm_id."""

    def __init__(self, ctx, arm: int):
        self._ctx, self.arm = ctx, arm

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def tool(self, name: str, **kw):
        if name.startswith("robot.") and name not in BOTH_ARMS:
            kw["arm_id"] = self.arm
        return self._ctx.tool(name, **kw)


# ---------------------------------------------------------------- the skill
def run(
    ctx: NodeContext,
    tasks: list | str,
    pick_place_skill: str,
    obb_input: str = "object_obb",
    place_input: str = "place",
    perceive_tool: str = "",
    perceive_kwargs: dict | None = None,
    perceive_output: str = "obb",
    camera: str = "",
    wrist_cameras: dict | None = None,
    pipelined: bool = True,
    pose_tolerance_m: float = POSE_TOLERANCE,
    object_tolerance_m: float = OBJECT_TOLERANCE,
    min_clearance_m: float = 0.03,
    wait_timeout_s: float = 15.0,
) -> Output:
    if isinstance(tasks, str):
        tasks = json.loads(tasks)
    conc = _concurrent()
    out: Output = {
        "route": "one_failed",
        "detail": "",
        "done": 0,
        "failed": 0,
        "stale": 0,
        "makespan_s": 0.0,
        "idle_s": 0.0,
        "cache_hits": 0,
        "cache_misses": 0,
        "tasks": "[]",
    }
    try:
        child, success = conc.load_skill(pick_place_skill)
    except Exception as exc:
        out["detail"] = f"cannot load {pick_place_skill}: {exc}"
        return out
    success = success or set()
    moves = set(conc.MOVES)
    wrists = {
        int(k): str(v) for k, v in (wrist_cameras or {0: "left_wrist", 1: "right_wrist"}).items()
    }
    queues: dict[int, list[dict]] = {}
    for i, t in enumerate(tasks):
        queues.setdefault(int(t.get("arm", 0)), []).append({**t, "index": i})
    arms = sorted(queues)
    busy: set[int] = set()  # arms executing a task now
    lock = threading.Lock()
    records: list[dict] = [{} for _ in tasks]

    def perceive(task: dict, executing: int) -> dict | None:
        if task.get("object_obb"):
            return task["object_obb"]
        if not perceive_tool:
            raise ValueError("a task without object_obb needs perceive_tool")
        cam = camera or ""
        with lock:
            moving = {wrists.get(a) for a in busy} | {wrists.get(executing)}
        if cam and cam in moving:
            raise ValueError(
                f"camera {cam} is on a moving arm; use the fixed camera or an idle arm's wrist"
            )
        kw = {
            k: (v.replace("{object}", str(task.get("object", ""))) if isinstance(v, str) else v)
            for k, v in (perceive_kwargs or {"object_name": "{object}"}).items()
        }
        if cam:
            kw["camera"] = cam
        reply = ctx.tool(perceive_tool, **kw)
        return reply.get(perceive_output, reply) if isinstance(reply, dict) else reply

    def lookahead(plan: Plan, arm: int, executing_done: threading.Event) -> None:
        try:
            plan.obb = perceive(plan.task, arm)
            inputs = {
                obb_input: plan.obb,
                place_input: plan.task["place"],
                **plan.task.get("inputs", {}),
            }
            try:
                child(_LookaheadContext(ctx, arm, moves, plan), **inputs)
            except _LookaheadDone:
                pass
        except Exception as exc:
            plan.error = f"{type(exc).__name__}: {exc}"
        finally:
            plan.ready.set()
        # keep the freshest view of the object until the arm is free (fixed camera only)
        while not executing_done.wait(0.2):
            try:
                if not plan.task.get("object_obb") and perceive_tool:
                    plan.latest_obb = perceive(plan.task, arm)
            except Exception:
                break

    def worker(arm: int, arm_ctx) -> None:
        queue = queues[arm]
        pending: Plan | None = None
        end_prev: float | None = None
        for n, task in enumerate(queue):
            rec = records[task["index"]]
            rec.update(index=task["index"], arm=arm, object=task.get("object", ""))
            t_ready = time.monotonic()
            stale_why = ""
            if pending is not None:
                pending.ready.wait()
                plan = pending
                now_pose = _ArmBinder(ctx, arm).tool("robot.get_ee_pose")
                if plan.error:
                    stale_why = f"look-ahead failed ({plan.error})"
                elif (
                    plan.start_pose is not None
                    and math.dist(_xyz(now_pose), _xyz(plan.start_pose)) > pose_tolerance_m
                ):
                    stale_why = f"arm ended {math.dist(_xyz(now_pose), _xyz(plan.start_pose)) * 1e3:.0f} mm from the expected end"
                if plan.latest_obb is not None and plan.obb is not None:
                    moved = math.dist(_xyz(plan.latest_obb), _xyz(plan.obb))
                    if moved > object_tolerance_m:
                        stale_why = (
                            stale_why + "; " if stale_why else ""
                        ) + f"object moved {moved * 1e3:.0f} mm"
                        plan.obb = plan.latest_obb
                if stale_why:
                    plan.cache = {}
                obb, cache = plan.obb, plan.cache
            else:
                obb, cache = None, {}
            if obb is None:
                try:
                    obb = perceive(task, arm)
                except Exception as exc:
                    rec.update(
                        route=f"perception failed: {exc}", ok=False, stale=stale_why, idle_s=0.0
                    )
                    pending = None
                    continue
            # look ahead at the next task while this one runs
            nxt = queue[n + 1] if pipelined and n + 1 < len(queue) else None
            running = threading.Event()
            if nxt is not None:
                pending = Plan(nxt)
                end = _ArmBinder(ctx, arm).tool("robot.get_ee_pose")
                pending.start_pose = end.get("pose", end)  # the child returns to where it started
                threading.Thread(
                    target=lookahead, args=(pending, arm, running), daemon=True
                ).start()
            else:
                pending = None
            live = _LiveContext(arm_ctx, cache, moves, arm)
            with lock:
                busy.add(arm)
            t0 = time.monotonic()
            try:
                reply = child(
                    live, **{obb_input: obb, place_input: task["place"], **task.get("inputs", {})}
                )
                route = str((reply or {}).get("route", "")) if isinstance(reply, dict) else ""
            except conc._Halted as exc:
                route = f"stopped ({exc})"
            except Exception as exc:
                route = f"raised {type(exc).__name__}: {exc}"
            finally:
                with lock:
                    busy.discard(arm)
                running.set()
            t1 = time.monotonic()
            ok = route in success if success else not route.startswith(("raised", "stopped"))
            rec.update(
                route=route,
                ok=ok,
                stale=stale_why,
                start_s=t0,
                end_s=t1,
                idle_s=((live.first_move or t1) - (end_prev if end_prev is not None else t_ready)),
                hits=live.hits,
                misses=live.misses,
            )
            end_prev = t1
        if hasattr(arm_ctx, "_arbiter"):
            arm_ctx._arbiter.finished(arm)

    t_start = time.monotonic()
    if len(arms) == 1:
        worker(arms[0], _ArmBinder(ctx, arms[0]))
    else:
        arbiter = conc.Arbiter(ctx, arms[:2], arms[0], min_clearance_m, wait_timeout_s)
        try:
            obs = (ctx.tool("robot.get_observation") or {}).get("arms") or []
        except Exception:  # no joint reading: a failed task stops its arm where it is
            obs = []

        def safe(a):
            js = obs[a].get("joint_state") if a < len(obs) else None
            js = js.get("positions", js) if isinstance(js, dict) else js
            return [float(v) for v in js] if js is not None else None

        threads = [
            threading.Thread(
                target=worker, args=(a, conc.ArmContext(ctx, a, arbiter, safe(a))), name=f"arm-{a}"
            )
            for a in arms[:2]
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    makespan = time.monotonic() - t_start

    done = [r for r in records if r.get("ok")]
    failed = [r for r in records if r and not r.get("ok")]
    stale = [r for r in records if r.get("stale")]
    out.update(
        done=len(done),
        failed=len(failed),
        stale=len(stale),
        makespan_s=round(makespan, 2),
        idle_s=round(sum(r.get("idle_s", 0.0) for r in records), 2),
        cache_hits=sum(r.get("hits", 0) for r in records),
        cache_misses=sum(r.get("misses", 0) for r in records),
        tasks=json.dumps(
            [
                {
                    k: (
                        round(v - t_start, 2)
                        if k in ("start_s", "end_s")
                        else round(v, 2)
                        if isinstance(v, float)
                        else v
                    )
                    for k, v in r.items()
                }
                for r in records
            ]
        ),
    )
    if any(r.get("stale") and not r.get("ok") for r in records):
        out["route"] = "stale_replan_failed"
    elif failed:
        out["route"] = "one_failed"
    else:
        out["route"] = "all_done"
    out["detail"] = (
        f"{len(done)}/{len(tasks)} tasks done on arm(s) {arms}, {'pipelined' if pipelined else 'one after the other'}; "
        f"makespan {makespan:.1f} s, idle between tasks {out['idle_s']:.1f} s; {len(stale)} stale plan(s) re-planned; "
        f"plan cache {out['cache_hits']} hit(s) / {out['cache_misses']} miss(es)"
        + "".join(f"; task {r['index']} on arm {r['arm']}: {r['route']}" for r in failed)
    )
    return out
