"""Run one hub skill with its ctx wrapped, time every tool call, and split the run into phases.

Phases: motion, settle (after a motion's arrival until the next command, and explicit waits), gripper,
perception, planning, other (state reads, and client time between calls that does not follow a motion).

    python profile_skill.py --events motion_*/events.jsonl [--moves runs/real-x/moves.json] [--json out.json]

profiles a recorded real run offline (motion server events, and the client's moves.json for settle times).
"""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

PHASES = ("motion", "settle", "gripper", "perception", "planning", "other")

#: (phase, fnmatch patterns on the call's name without its namespace), first match wins.
RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("settle", ("wait*", "sleep*", "settle*")),
    ("gripper", ("open_gripper", "close_gripper", "set_gripper*", "grip", "grip_*", "*_gripper_*")),
    (
        "motion",
        (
            "go_to_pose*",
            "execute_trajectory*",
            "move_*",
            "move",
            "stream_*",
            "reset",
            "home*",
            "retreat*",
        ),
    ),
    (
        "perception",
        (
            "get_observation",
            "get_image*",
            "capture*",
            "*detect*",
            "*vlm*",
            "*sam*",
            "*grounding*",
            "perceiv*",
            "segment*",
            "get_object_obb",
            "get_object_pose*",
            "locate*",
            "survey*",
            "look*",
        ),
    ),
    (
        "planning",
        (
            "solve_ik*",
            "*curobo*",
            "plan*",
            "*planner*",
            "forward_kinematics",
            "grasp_frame",
            "*_ik",
        ),
    ),
)


def classify(name: str) -> str:
    """The phase of a tool call (``robot.go_to_pose``) or a recorded op (``move``)."""
    short = name.rsplit(".", 1)[-1].lower()
    bundle = name.rsplit(".", 1)[0].lower() if "." in name else ""
    for phase, patterns in RULES:
        if any(fnmatch.fnmatchcase(short, p) for p in patterns):
            return phase
    if bundle and not bundle.startswith(("robot", "sim", "motion")):
        if any(k in bundle for k in ("curobo", "plan", "ik")):
            return "planning"
        if any(
            k in bundle for k in ("vlm", "sam", "detect", "dino", "owl", "yolo", "percei", "depth")
        ):
            return "perception"
    return "other"


def arrival_s(reply: Any, control_dt_s: float | None) -> float | None:
    """Seconds from a motion call's start to its server-side arrival, when the reply says so."""
    if not isinstance(reply, dict):
        return None
    for key in ("arrived_s", "motion_s", "duration_s"):
        if isinstance(reply.get(key), (int, float)):
            return float(reply[key])
    return None


class Profiler:
    """A ctx stand-in: forwards everything to ``ctx`` and records each ``tool`` call.

    ``sim_clock`` (seconds of simulated time, optional) is read at every call's start and end.
    ``mark(label)`` starts a segment (an object, a pen) that later calls belong to.
    """

    def __init__(
        self,
        ctx,
        sim_clock: Callable[[], float] | None = None,
        control_dt_s: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._ctx = ctx
        self._clock = clock
        self._sim = sim_clock
        self._dt = control_dt_s
        self._lock = threading.Lock()
        self.t0 = clock()
        self.sim0 = sim_clock() if sim_clock else None
        self.calls: list[dict[str, Any]] = []
        self.segment = ""

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def _now(self) -> tuple[float, float | None]:
        return self._clock() - self.t0, (self._sim() - self.sim0) if self._sim else None

    def mark(self, label: str) -> None:
        self.segment = str(label)

    def tool(self, name: str, **kwargs):
        start, sim_start = self._now()
        error = None
        reply = None
        try:
            reply = self._ctx.tool(name, **kwargs)
            return reply
        except BaseException as exc:
            error = repr(exc)[:160]
            raise
        finally:
            end, sim_end = self._now()
            rec = {
                "name": name,
                "phase": classify(name),
                "start_s": start,
                "end_s": end,
                "segment": self.segment,
                "thread": threading.current_thread().name,
            }
            if sim_start is not None:
                rec["sim_start_s"], rec["sim_end_s"] = sim_start, sim_end
            if "arm_id" in kwargs:
                rec["arm_id"] = kwargs["arm_id"]
            if rec["phase"] == "motion":
                arr = arrival_s(reply, self._dt)
                if arr is not None:
                    rec["arrived_s"] = arr
            if error:
                rec["error"] = error
            with self._lock:
                self.calls.append(rec)

    def timeline(self) -> list[dict[str, Any]]:
        return build_timeline(self.calls)


def build_timeline(calls: list[dict[str, Any]], end_s: float | None = None) -> list[dict[str, Any]]:
    """Calls plus the gaps between them, per thread, each gap given a phase.

    A motion's time past its server-side arrival and the gap after it until the next command are ``settle``;
    any other gap is ``other`` (client code between calls). Sim times are split the same way when present.
    """
    out: list[dict[str, Any]] = []
    by_thread: dict[str, list[dict[str, Any]]] = {}
    for c in sorted(calls, key=lambda c: c["start_s"]):
        by_thread.setdefault(c.get("thread", ""), []).append(c)
    for thread, seq in by_thread.items():
        for i, c in enumerate(seq):
            c = dict(c)
            arr = c.pop("arrived_s", None)
            if c["phase"] == "motion" and arr is not None and arr < c["end_s"] - c["start_s"]:
                split = c["start_s"] + arr
                out.append({**c, "end_s": split, **_sim_cut(c, arr, before=True)})
                out.append(
                    {
                        **c,
                        "phase": "settle",
                        "kind": "after_arrival",
                        "start_s": split,
                        **_sim_cut(c, arr, before=False),
                    }
                )
            else:
                out.append(c)
            nxt = seq[i + 1]["start_s"] if i + 1 < len(seq) else end_s
            if nxt is None or nxt <= c["end_s"]:
                continue
            gap = {
                "name": "(between calls)",
                "phase": "settle" if c["phase"] in ("motion", "settle") else "other",
                "kind": "gap",
                "start_s": c["end_s"],
                "end_s": nxt,
                "segment": c.get("segment", ""),
                "thread": thread,
            }
            if "sim_end_s" in c:
                gap["sim_start_s"] = c["sim_end_s"]
                gap["sim_end_s"] = (
                    seq[i + 1].get("sim_start_s", c["sim_end_s"])
                    if i + 1 < len(seq)
                    else c["sim_end_s"]
                )
            out.append(gap)
    out.sort(key=lambda r: r["start_s"])
    for r in out:
        for k in ("start_s", "end_s", "sim_start_s", "sim_end_s"):
            if isinstance(r.get(k), float):
                r[k] = round(r[k], 4)
    return out


def _sim_cut(c: dict, arr_wall: float, before: bool) -> dict:
    if "sim_start_s" not in c:
        return {}
    wall = c["end_s"] - c["start_s"]
    s0, s1 = c["sim_start_s"], c["sim_end_s"]
    mid = s0 + (s1 - s0) * (arr_wall / wall if wall > 0 else 1.0)
    return {"sim_end_s": mid} if before else {"sim_start_s": mid}


def phase_table(
    timeline: list[dict[str, Any]], by: str | None = None
) -> dict[str, dict[str, dict[str, float]]]:
    """{group: {phase: {calls, wall_s, sim_s, share}}}; group is ``by``'s value (e.g. segment) or "all"."""
    groups: dict[str, dict[str, dict[str, float]]] = {}
    for r in timeline:
        g = str(r.get(by, "")) if by else "all"
        row = groups.setdefault(g, {p: {"calls": 0, "wall_s": 0.0, "sim_s": 0.0} for p in PHASES})[
            r["phase"]
        ]
        row["calls"] += int(r.get("kind") is None)
        row["wall_s"] += r["end_s"] - r["start_s"]
        if "sim_start_s" in r:
            row["sim_s"] += r["sim_end_s"] - r["sim_start_s"]
    for g in groups.values():
        tw = sum(v["wall_s"] for v in g.values()) or 1.0
        ts = sum(v["sim_s"] for v in g.values()) or 1.0
        for v in g.values():
            v["share"] = round(v["wall_s"] / tw, 4)
            v["sim_share"] = round(v["sim_s"] / ts, 4)
            v["wall_s"], v["sim_s"] = round(v["wall_s"], 3), round(v["sim_s"], 3)
    return groups


def format_table(table: dict[str, dict[str, dict[str, float]]]) -> str:
    lines = []
    sim = any(v["sim_s"] for g in table.values() for v in g.values())
    for group, rows in table.items():
        head = f"{'phase':<11}{'calls':>6}{'wall_s':>9}{'wall%':>7}" + (
            f"{'sim_s':>9}{'sim%':>7}" if sim else ""
        )
        lines += [f"[{group}]", head]
        for p in PHASES:
            v = rows[p]
            line = f"{p:<11}{v['calls']:>6}{v['wall_s']:>9.2f}{100 * v['share']:>6.1f}%"
            lines.append(line + (f"{v['sim_s']:>9.2f}{100 * v['sim_share']:>6.1f}%" if sim else ""))
        tw = sum(v["wall_s"] for v in rows.values())
        lines.append(
            f"{'total':<11}{'':>6}{tw:>9.2f}"
            + (f"{'':>7}{sum(v['sim_s'] for v in rows.values()):>9.2f}" if sim else "")
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- recorded real runs
_PEN = re.compile(r"pen[ _](\d+)")
#: a client that moves to a "look ..." pose images and detects there before its next command.
LOOK_PREFIXES = ("look", "survey", "relook")


def profile_recording(
    events: list[dict[str, Any]], moves: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """A timeline from a motion server's events.jsonl (``op`` move / grip / reset, start ``t``, ``duration_s``).

    The gap after a move is settle for the ``settle_s`` the client's moves.json records for that move (matched
    by label and time), then ``other`` (the client: imaging, detection, planning; the server cannot tell them
    apart), except after a ``look ...`` move, where it is perception. Segments are pens, from ``pen NN`` / ``pen_NN`` in the labels; before the first pen, ``setup``.
    """
    ops = sorted(
        (e for e in events if e.get("op") in ("move", "grip", "reset") and "duration_s" in e),
        key=lambda e: e["t"],
    )
    if not ops:
        return []
    t0 = ops[0]["t"]
    settles = sorted(
        (m["t"], m.get("label", ""), float(m["settle_s"]))
        for m in moves or []
        if m.get("settle_s") is not None
    )
    used: set[int] = set()
    segment = "setup"
    out = []
    for i, e in enumerate(ops):
        label = e.get("label", "")
        m = _PEN.search(label)
        if m:
            segment = f"pen {m.group(1)}"
        start, end = e["t"] - t0, e["t"] - t0 + float(e["duration_s"])
        phase = {"move": "motion", "reset": "motion", "grip": "gripper"}[e["op"]]
        out.append(
            {
                "name": f"{e['op']}: {label}".strip(": "),
                "phase": phase,
                "start_s": start,
                "end_s": end,
                "segment": segment,
                "side": e.get("side", ""),
            }
        )
        nxt = ops[i + 1]["t"] - t0 if i + 1 < len(ops) else None
        if nxt is None or nxt <= end:
            continue
        gap = nxt - end
        settle = 0.0
        if phase == "motion":
            j = _match_settle(settles, used, e["t"], label)
            if j is not None:
                used.add(j)
                settle = min(gap, settles[j][2])
        if settle > 0:
            out.append(
                {
                    "name": "(settle)",
                    "phase": "settle",
                    "kind": "gap",
                    "start_s": end,
                    "end_s": end + settle,
                    "segment": segment,
                }
            )
        if gap - settle > 1e-6:
            client = (
                "perception" if phase == "motion" and label.startswith(LOOK_PREFIXES) else "other"
            )
            out.append(
                {
                    "name": "(client)",
                    "phase": client,
                    "kind": "gap",
                    "start_s": end + settle,
                    "end_s": nxt,
                    "segment": segment,
                }
            )
    for r in out:
        r["start_s"], r["end_s"] = round(r["start_s"], 3), round(r["end_s"], 3)
    return out


def _match_settle(settles, used, t, label, window_s: float = 3.0) -> int | None:
    """The moves.json entry whose label starts this move's label (server labels add ': leg 1 of 2' etc.) and
    whose time is within ``window_s`` of the move's start (moves.json logs when the client sent it)."""
    best, best_dt = None, window_s
    for j, (tm, lab, _) in enumerate(settles):
        if j in used or not label.startswith(lab) or not lab:
            continue
        if abs(tm - t) <= best_dt:
            best, best_dt = j, abs(tm - t)
    return best


# ---------------------------------------------------------------- running a skill
def _skill_roots() -> list[Path]:
    roots = [Path(__file__).resolve().parents[2]]
    for entry in os.environ.get("GAP_SKILLS_PATH", "").split(os.pathsep):
        if entry:
            roots += [Path(entry) / "skills", Path(entry)]
    return roots


#: name -> run; tests register fake skills here.
RUNNERS: dict[str, Callable[..., Any]] = {}


def load_skill(name: str) -> Callable[..., Any]:
    if name in RUNNERS:
        return RUNNERS[name]
    import yaml

    for root in _skill_roots():
        md = root / name / "SKILL.md"
        if not md.is_file():
            continue
        meta = yaml.safe_load(md.read_text().split("---", 2)[1]) or {}
        scripts = (meta.get("gap") or {}).get("canonical_scripts") or []
        if not scripts:
            raise ValueError(f"skill {name} has no canonical script")
        rel = next(iter(scripts[0].values())) if isinstance(scripts[0], dict) else scripts[0]
        mod_name = f"rsh_profiled_{name.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(mod_name, md.parent / rel)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module.run
    raise ValueError(f"no skill named {name} under {[str(r) for r in _skill_roots()]}")


class Output(TypedDict):
    route: str
    child_route: str
    child_outputs: str
    phase_table: str
    timeline_json: str


def run(ctx, skill: str, inputs_json: str = "{}", out_path: str = "") -> Output:
    """Run ``skill`` with ``inputs_json`` through a profiling ctx; the table and timeline go to the outputs and,
    when ``out_path`` is given, to ``<out_path>.json`` / ``<out_path>.txt``."""
    try:
        child = load_skill(skill)
        inputs = json.loads(inputs_json or "{}")
    except Exception as exc:  # noqa: BLE001
        return {
            "route": "refused",
            "child_route": "",
            "child_outputs": json.dumps({"error": repr(exc)[:300]}),
            "phase_table": "",
            "timeline_json": "[]",
        }
    prof = Profiler(ctx)
    route, outputs = "failed", {}
    try:
        outputs = child(prof, **inputs) or {}
        route = "profiled"
    except Exception as exc:  # noqa: BLE001
        outputs = {"error": repr(exc)[:300]}
    timeline = build_timeline(prof.calls, end_s=time.perf_counter() - prof.t0)
    table = phase_table(timeline)
    text = format_table(table)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path + ".json").write_text(
            json.dumps({"table": table, "timeline": timeline}, indent=1)
        )
        Path(out_path + ".txt").write_text(text + "\n")
    child_route = str(outputs.get("route", "")) if isinstance(outputs, dict) else ""
    return {
        "route": route,
        "child_route": child_route,
        "child_outputs": json.dumps(outputs, default=str)[:4000],
        "phase_table": text,
        "timeline_json": json.dumps(timeline),
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--events", required=True, help="motion server events.jsonl")
    ap.add_argument("--moves", help="the client's moves.json (settle_s per move)")
    ap.add_argument("--json", help="write {table, by_segment, timeline} here")
    a = ap.parse_args(argv)
    events = [json.loads(line) for line in Path(a.events).read_text().splitlines() if line.strip()]
    moves = json.loads(Path(a.moves).read_text()) if a.moves else None
    tl = profile_recording(events, moves)
    total, per = phase_table(tl), phase_table(tl, by="segment")
    print(format_table(total))
    print(format_table(per))
    if a.json:
        Path(a.json).write_text(
            json.dumps({"table": total, "by_segment": per, "timeline": tl}, indent=1)
        )


if __name__ == "__main__":
    main()
