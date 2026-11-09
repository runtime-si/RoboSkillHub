"""Inputs for verify_stable_placement from other modules' formats.

``from_eval_samples`` reads the sample stream of Agent 04's assembly_eval domino runner
(``{"t", "stamp", "seq", "source", "frame", "objects": {id: {"position", "quat_xyzw"}},
"contacts": {id: [bodies]} | None}``) so its Isaac / MuJoCo episodes can be judged by
this observer as well as by its own post-condition. Simulator truth stays labelled
``privileged``; contacts become gripper states only when the backend reported them.
"""

from __future__ import annotations

from typing import Any

GRIPPER_BODIES = ("gripper_pad_l", "gripper_pad_r", "gripper_palm")


def from_eval_samples(
    samples: list[dict[str, Any]], *, target_id: str, frame_id: str | None = None
) -> tuple[list[dict], list[dict]]:
    """(Observation records, gripper states) from assembly_eval observation samples."""
    records, gripper = [], []
    for s in samples:
        stamp = s.get("stamp", s.get("t"))
        source = str(s.get("source") or "")
        privileged = source in ("simulation_truth", "sim_truth")
        frame = frame_id or s.get("frame")
        for oid, o in (s.get("objects") or {}).items():
            records.append(
                {
                    "object_id": oid,
                    "class": "domino",
                    "frame_id": frame,
                    "position_m": list(o["position"]) if o.get("position") is not None else None,
                    "quaternion_xyzw": list(o["quat_xyzw"])
                    if o.get("quat_xyzw") is not None
                    else None,
                    "timestamp_s": stamp,
                    "source": source,
                    "privileged": privileged,
                    "confidence": None,
                    "uncertainty": None,
                    "visibility": "visible",
                    "state_label": None,
                }
            )
        contacts = s.get("contacts")
        if isinstance(contacts, dict):
            touching = set(contacts.get(target_id) or ())
            gripper.append(
                {
                    "holding": bool(touching & set(GRIPPER_BODIES)),
                    "timestamp_s": stamp,
                    "source": f"{source}:contacts",
                }
            )
    return records, gripper
