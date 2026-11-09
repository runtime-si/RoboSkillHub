# SPDX-License-Identifier: Apache-2.0
"""Domino part of ``assembly_eval`` as rsh-task/1 TaskSkills (domino-only ``SKILLS``).

``eval.domino_layout_check`` (tool): the independent layout verdict of
``domino_postcondition.evaluate_slots`` over timestamped object frames, usable as a
separate evaluator node in a graph or offline on logged frames. It never reads a skill's
own result. Agent 5 registers this list; ``assembly_eval/__init__.py`` belongs to 4a and may
re-export it next to the LEGO list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .domino_postcondition import DEFAULT_THRESHOLDS, EVAL_VERSION, evaluate_slots
from .domino_scene import DominoSceneSpec, TableSpec, TileSpec

_POSE = {
    "type": "object",
    "required": ["frame_id", "position_m", "quaternion_xyzw"],
    "properties": {
        "frame_id": {"type": "string"},
        "position_m": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
        "quaternion_xyzw": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "number"},
        },
    },
}


def _frames_to_samples(frames: list[dict], frame_id: str) -> list[dict]:
    samples = []
    for i, f in enumerate(frames):
        objs = {}
        for oid, o in (f.get("objects") or {}).items():
            if o.get("position_m") is None or o.get("frame_id", frame_id) != frame_id:
                continue
            objs[oid] = {"position": list(o["position_m"]), "quat_xyzw": o.get("quaternion_xyzw")}
        samples.append(
            {
                "t": f["timestamp_s"],
                "stamp": f["timestamp_s"],
                "seq": f.get("seq", i),
                "source": f.get("source", "unknown"),
                "frame": "table",
                "objects": objs,
                "contacts": f.get("contacts"),
            }
        )
    return samples


@dataclass(frozen=True)
class DominoLayoutCheck:
    skill_id: str = "eval.domino_layout_check"
    version: str = "0.1.0"
    kind: str = "tool"
    requires: frozenset = frozenset()
    readiness: str = "sim-validated"
    input_schema: dict = field(
        default_factory=lambda: {
            "type": "object",
            "additionalProperties": False,
            "required": ["slots", "frames", "table_z"],
            "properties": {
                "frame_id": {"type": "string"},
                "slots": {"type": "array", "minItems": 1, "items": _POSE},
                "frames": {"type": "array", "minItems": 1, "items": {"type": "object"}},
                "table_z": {"type": "number"},
                "size_m": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "number"},
                },
                "window_s": {"type": "number"},
            },
        }
    )

    def run(self, request: dict, ctx=None) -> dict:
        from roboskillhub.runtime.task import make_result

        frame_id = request.get("frame_id", "world")
        h, w, t = request.get("size_m", [0.048, 0.024, 0.0075])
        spec = DominoSceneSpec(
            tile=TileSpec(thickness=t, width=w, height=h), table=TableSpec(top_z=request["table_z"])
        )
        frames = sorted(request["frames"], key=lambda f: f["timestamp_s"])
        samples = _frames_to_samples(frames, frame_id)
        t1 = frames[-1]["timestamp_s"]
        window = float(request.get("window_s", DEFAULT_THRESHOLDS.window_s))
        slots = []
        for s in request["slots"]:
            q = s["quaternion_xyzw"]
            yaw = math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
            slots.append({"position": list(s["position_m"]), "yaw": yaw})
        verdict = evaluate_slots(spec, samples, t1 - window, t1, slots, DEFAULT_THRESHOLDS)
        status = verdict["status"]
        last = frames[-1].get("objects") or {}
        observations = [
            {
                "object_id": oid,
                "frame_id": frame_id,
                "position_m": o.get("position_m"),
                "quaternion_xyzw": o.get("quaternion_xyzw"),
                "timestamp_s": o.get("timestamp_s", t1),
                "source": o.get("source", "unknown"),
                "privileged": bool(o.get("privileged")),
            }
            for oid, o in last.items()
            if o.get("position_m") is not None
        ]
        failure = {"failed": "eval.layout_not_standing", "unverified": "evidence_missing"}.get(
            status
        )
        return make_result(
            self,
            status,
            phase="verify",
            failure_type=failure,
            evidence={
                "postcondition": {
                    "name": "every_slot_standing",
                    "passed": status == "succeeded",
                    "criterion": "per slot: position, yaw, tilt over the window, bottom on "
                    "the table, drift, no gripper contact",
                    "measured": verdict,
                    "thresholds_ref": EVAL_VERSION,
                }
            },
            observations=observations,
            outputs={"standing_at_slot": verdict.get("standing_at_slot", 0)},
            task_id=getattr(ctx, "task_id", None),
            request_id=getattr(ctx, "request_id", None),
            ctx=ctx,
        )


SKILLS = [DominoLayoutCheck()]
