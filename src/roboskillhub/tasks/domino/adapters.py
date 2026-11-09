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
    """(Observation records, gripper states) from assembly_eval observation samples.

    ``stamp`` is the acquisition time (sim seconds in that runner), ``seq`` the frame
    number: a sample whose ``seq`` was already seen, or whose stamp does not increase, is a
    frozen / repeated frame and is dropped here (the observer then sees a short window).
    """
    records, gripper = [], []
    last_seq, last_stamp = None, None
    for s in samples:
        stamp = s.get("stamp", s.get("t"))
        seq = s.get("seq")
        if (seq is not None and last_seq is not None and seq <= last_seq) or (
            last_stamp is not None and stamp is not None and stamp <= last_stamp
        ):
            continue
        last_seq, last_stamp = (seq if seq is not None else last_seq), stamp
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


def verify_request_from_eval_record(
    record: dict[str, Any],
    samples: list[dict[str, Any]],
    placement_index: int,
    *,
    size_m: tuple[float, float, float] = (0.048, 0.024, 0.0075),
    thresholds_override: dict | None = None,
) -> dict[str, Any]:
    """A verify_stable_placement request for one placement of an assembly_eval episode
    (``roboskillhub.assembly_eval.domino_episode/v1`` record + its observations.jsonl).

    Neighbours are the tiles placed before this one; their baseline is the last sample
    before this placement's first phase. The support plane is the slot's bottom face
    height (slot poses are tile centres). Event times come from the runner's log."""
    p = record["placements"][placement_index]
    ev = p["events"]
    target = p["target_id"]
    tp = p["target_pose"]
    start = min(ph["t0"] for ph in ev["phases"])
    earlier = [q["target_id"] for q in record["placements"][:placement_index]]
    frame = tp["frame"]
    records, gripper = from_eval_samples(samples, target_id=target, frame_id=frame)
    before = [s for s in samples if s.get("stamp", s.get("t")) <= start]
    baseline = []
    if before:
        base_recs, _ = from_eval_samples([before[-1]], target_id=target, frame_id=frame)
        baseline = [r for r in base_recs if r["object_id"] in earlier]
    h = float(size_m[0])
    return {
        "request_id": f"{record.get('label')}/seed{record.get('seed')}/p{placement_index}",
        "frame_id": frame,
        "target_id": target,
        "geometry": {"size_m": list(size_m), "yaw_symmetry_deg": 180.0},
        "target_object_pose": {
            "frame_id": frame,
            "position_m": list(tp["position"]),
            "quaternion_xyzw": list(tp["quat_xyzw"]),
        },
        "support": {
            "frame_id": frame,
            "point_m": [0.0, 0.0, float(tp["position"][2]) - h / 2.0],
            "normal": [0.0, 0.0, 1.0],
            "height_std_m": 0.0,
            "calibration_version": f"scene:{record.get('scene_config_sha256', 'unknown')[:12]}",
        },
        "neighbors_baseline": baseline,
        "events": {
            "action_start_s": start,
            "release_s": ev["release_t"],
            "retreat_done_s": ev["retreat_done_t"],
        },
        # up to the evaluation instant: later samples belong to the next placement
        "observations": [
            r
            for r in records
            if r["object_id"] in [target, *earlier] and start < r["timestamp_s"] <= ev["eval_t"]
        ],
        "gripper_states": [g for g in gripper if g["timestamp_s"] <= ev["eval_t"]],
        "now_s": ev["eval_t"],
        "thresholds_override": thresholds_override or {},
    }
