# SPDX-License-Identifier: Apache-2.0
"""Backend-agnostic domino episode loop and per-episode records.

A backend (Isaac/PhysX or MuJoCo) implements :class:`DominoBackend`: it builds the scene
from a :class:`DominoSceneSpec`, steps physics while driving the hand's joints toward a
commanded TCP pose and jaw opening, and reports simulator truth (object poses and
contacts). The runner plans with the scripted privileged baseline, logs every
observation, and judges each placement with :mod:`domino_postcondition` only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Protocol

from .domino_baseline import (
    BASELINE_VERSION,
    COMMAND_FAULTS,
    OBSERVER_FAULTS,
    interpolate,
    plan_tile,
    plan_to_dict,
)
from .domino_postcondition import (
    DEFAULT_THRESHOLDS,
    EVAL_VERSION,
    Thresholds,
    evaluate_layout,
    evaluate_placement,
)
from .domino_scene import DominoSceneSpec, check_scene, config_hash

RECORD_SCHEMA = "roboskillhub.assembly_eval.domino_episode/v1"
G_VERSION = "fixed_sequence_0_to_n/v1"


class DominoBackend(Protocol):
    name: str

    def reset(self, spec: DominoSceneSpec, feeder_poses: list[dict]) -> None: ...

    def step(self, hand_pose: tuple, opening: float) -> float: ...  # returns sim time

    def truth(self) -> dict: ...  # {"t", "objects": {...}, "contacts": {...}}

    def hand(self) -> tuple[tuple, float]: ...

    def describe(self) -> dict: ...

    @property
    def control_dt(self) -> float: ...

    @property
    def physics_steps(self) -> int: ...


class Observer:
    """Turns simulator truth into observation samples at ``observe_hz``. Observer faults
    are applied here, never in physics: ``stale_observation`` freezes the last frame from
    ``fault_from`` on (same stamp and sequence number); ``lost_target`` drops the target
    tile's pose from ``fault_from`` on."""

    def __init__(self, observe_hz: float, fault: str | None = None):
        self.period = 1.0 / observe_hz
        self.fault = fault if fault in OBSERVER_FAULTS else None
        self.fault_from: float | None = None
        self.fault_target: str | None = None
        self.samples: list[dict] = []
        self._next = 0.0
        self._seq = 0
        self._frozen: dict | None = None

    def arm(self, t: float, target: str) -> None:
        if self.fault:
            self.fault_from, self.fault_target = t, target

    def maybe_sample(self, t: float, backend: DominoBackend) -> None:
        if t + 1e-9 < self._next:
            return
        self._next = t + self.period
        if (
            self.fault == "stale_observation"
            and self.fault_from is not None
            and t >= self.fault_from
        ):
            if self._frozen is None:
                self._frozen = copy.deepcopy(self.samples[-1]) if self.samples else None
            if self._frozen is not None:
                s = copy.deepcopy(self._frozen)
                s["t"] = t  # wall/sim time of delivery advances; the frame does not
                self.samples.append(s)
            return
        truth = backend.truth()
        self._seq += 1
        s = {
            "t": t,
            "stamp": truth["t"],
            "seq": self._seq,
            "source": "simulation_truth",
            "frame": "table",
            "objects": truth["objects"],
            "contacts": truth.get("contacts"),
        }
        if self.fault == "lost_target" and self.fault_from is not None and t >= self.fault_from:
            s["objects"] = {k: v for k, v in s["objects"].items() if k != self.fault_target}
        self.samples.append(s)


def feeder_poses_for_seed(
    spec: DominoSceneSpec, seed: int, jitter_xy: float = 0.0008, jitter_yaw_deg: float = 2.0
) -> list[dict]:
    """Initial rack poses: nominal plus a seeded jitter inside the rack's slack (setup only)."""
    import math

    from .domino_geometry import quat_from_yaw

    rng = random.Random(seed)
    out = []
    for k in range(spec.n_tiles):
        p = spec.feeder_pose(k)
        yaw = p["yaw"] + math.radians(rng.uniform(-jitter_yaw_deg, jitter_yaw_deg))
        pos = [
            p["position"][0] + rng.uniform(-jitter_xy, jitter_xy),
            p["position"][1] + rng.uniform(-jitter_xy, jitter_xy),
            p["position"][2],
        ]
        out.append(
            {"frame": "table", "position": pos, "quat_xyzw": list(quat_from_yaw(yaw)), "yaw": yaw}
        )
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_episode(
    backend: DominoBackend,
    spec: DominoSceneSpec,
    seed: int,
    out_dir: Path,
    fault: str | None = None,
    fault_tile: int | None = None,
    fault_offset: float | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    label: str = "",
) -> dict:
    """Run one episode (all tiles in order 0..n-1) and write its record and logs."""
    wall0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    if fault is not None and fault not in COMMAND_FAULTS + OBSERVER_FAULTS:
        raise ValueError(f"unknown fault {fault!r}")
    fault_tile = spec.n_tiles - 1 if fault_tile is None else fault_tile
    issues = check_scene(spec)
    record: dict = {
        "schema": RECORD_SCHEMA,
        "label": label or (fault or "nominal"),
        "evaluation_role": "negative" if fault else "positive",
        "seed": seed,
        "backend": backend.describe(),
        "scene_version": spec.version,
        "scene_config_sha256": spec.config_hash(),
        "physics_config_sha256": config_hash(
            vars(spec.physics) if hasattr(spec.physics, "__dict__") else spec.to_dict()["physics"]
        ),
        "K_version": {"pick_place": BASELINE_VERSION, "evaluator": EVAL_VERSION},
        "G_version": G_VERSION,
        "policy": {
            "kind": "scripted_baseline",
            "privileged": True,
            "privileged_inputs": [
                "true pose of the tile to pick",
                "true table height (unless faulted)",
            ],
        },
        "observation_mode": "simulation_truth (evaluator); privileged truth (baseline)",
        "goal": {
            "slots": [spec.slot_pose(k) for k in range(spec.n_tiles)],
            "order": list(range(spec.n_tiles)),
        },
        "fault": {"name": fault, "tile": fault_tile if fault else None, "offset": fault_offset},
        "thresholds": vars(thresholds) if hasattr(thresholds, "__dict__") else None,
        "scene_issues": issues,
        "recovery_count": 0,
        "human_interventions": 0,
        "placements": [],
    }
    if issues:
        record.update(task_success=False, status="rejected", failure_type="scene_check_failed")
        (out_dir / "record.json").write_text(json.dumps(record, indent=1))
        return record
    feeders = feeder_poses_for_seed(spec, seed)
    record["initial_feeder_poses"] = feeders
    backend.reset(spec, feeders)
    observer = Observer(spec.physics.observe_hz, fault if fault in OBSERVER_FAULTS else None)
    cmd_log = []
    planning_s = 0.0
    t = 0.0
    # settle the rack before anything moves
    hand, opening = backend.hand()
    for _ in range(int(0.3 / backend.control_dt)):
        t = backend.step(hand, opening)
        observer.maybe_sample(t, backend)
    placement_results = []
    executor_trace = []
    for k in range(spec.n_tiles):
        tid = f"tile_{k}"
        truth = backend.truth()
        neighbours = {f"tile_{j}": truth["objects"][f"tile_{j}"] for j in range(k)}
        this_fault = fault if (fault and k == fault_tile) else None
        p0 = time.perf_counter()
        plan = plan_tile(
            spec,
            k,
            k,
            truth["objects"][tid],
            hand,
            opening,
            thresholds.window_s,
            fault=this_fault,
            fault_offset=fault_offset,
        )
        planning_s += time.perf_counter() - p0
        phases, events = [], {}
        for seg in plan.segments:
            start_pose, start_open = hand, opening
            t0 = t
            if seg.phase == "release" and "release_t" not in events:
                events["release_t"] = t
                if this_fault in OBSERVER_FAULTS:
                    observer.arm(t, tid)
            n = max(1, round(seg.duration / backend.control_dt))
            for i in range(1, n + 1):
                cmd_pose, cmd_open = interpolate(seg, start_pose, start_open, i / n)
                t = backend.step(cmd_pose, cmd_open)
                observer.maybe_sample(t, backend)
                if i % 6 == 0 or i == n:
                    cmd_log.append(
                        {
                            "t": t,
                            "tile": k,
                            "phase": seg.phase,
                            "pose": list(cmd_pose),
                            "opening": cmd_open,
                        }
                    )
            hand, opening = seg.pose, seg.opening
            if phases and phases[-1]["name"] == seg.phase:
                phases[-1]["t1"] = t
            else:
                phases.append({"name": seg.phase, "t0": t0, "t1": t})
            if seg.phase == "retreat":
                events["retreat_done_t"] = t
            if seg.phase == "observe":
                events["eval_t"] = t
        events["phases"] = phases
        # The scripted executor always reports success: this is what the evaluator must
        # not believe.
        executor_report = {"status": "finished", "tile": k}
        executor_trace.append(executor_report)
        res = evaluate_placement(
            spec,
            k,
            tid,
            neighbours,
            observer.samples,
            events,
            thresholds,
            executor_report=executor_report,
            feeder_pose=feeders[k],
        )
        res["events"] = events
        res["plan"] = plan_to_dict(plan)
        placement_results.append(res)
        record["placements"].append(res)
        if this_fault:
            break  # a faulted placement ends the episode
    last_events = placement_results[-1]["events"]
    layout = evaluate_layout(
        spec,
        observer.samples,
        last_events["retreat_done_t"],
        last_events["eval_t"],
        placement_results,
        thresholds,
    )
    obs_path = out_dir / "observations.jsonl"
    with obs_path.open("w") as f:
        for s in observer.samples:
            f.write(json.dumps(s) + "\n")
    cmd_path = out_dir / "commands.jsonl"
    with cmd_path.open("w") as f:
        for c in cmd_log:
            f.write(json.dumps(c) + "\n")
    failure_types = []
    for r in placement_results:
        for ft in r.get("failure_types", []):
            if ft not in failure_types:
                failure_types.append(ft)
    record.update(
        status=layout["status"],
        task_success=layout["status"] == "succeeded",
        layout=layout,
        placements_attempted=len(placement_results),
        placements_succeeded=sum(1 for r in placement_results if r["status"] == "succeeded"),
        failure_types=failure_types,
        executor_reports=executor_trace,
        executor_claimed_all_finished=all(e["status"] == "finished" for e in executor_trace),
        sim_physics_steps=backend.physics_steps,
        sim_time_s=t,
        planning_s=planning_s,
        wall_s=time.time() - wall0,
        trajectory_ref={
            "observations": obs_path.name,
            "observations_sha256": _sha(obs_path),
            "commands": cmd_path.name,
            "commands_sha256": _sha(cmd_path),
        },
        video_ref=None,
    )
    (out_dir / "record.json").write_text(json.dumps(record, indent=1))
    return record
