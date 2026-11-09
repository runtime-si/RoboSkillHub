"""Candidate graph G (``rsh-graph/1``) for one LEGO connection over K = this module's SKILLS.

held confirmed (pre-fact) -> align -> guarded seat -> release & retreat -> verify -> done

A candidate, not a validated G: it has been loaded and run only with scripted fakes
(unit_or_dryrun). Run it with ``roboskillhub.runtime.graph.run_graph``; this module has
no interpreter of its own. Branch rules:

- errors found before approach (rejected) stop with nothing moved (``stop_replan``);
- alignment re-observes once on unverified; convergence retries live inside the skill's
  own correction budget, never by re-running it on a failure;
- a press that is rejected for missing contact capability has no position-only fallback;
- any press or release anomaly stops for a human/upper G: no blind retreat or release;
- verification ``unverified`` re-observes once then stops unknown (never ``done``);
  ``failed`` stops with the evidence: automatic disassembly/repair is not implemented.
"""

from __future__ import annotations

from typing import Any

VERSION = "0.1.0"

CANDIDATE_GRAPH: dict[str, Any] = {
    "graph_id": "lego.single_connection",
    "version": VERSION,
    "contract_version": "rsh-task/1",
    "skills": {
        "lego.align_connection": "0.1.0",
        "lego.guarded_seat": "0.1.0",
        "lego.release_and_retreat": "0.1.0",
        "lego.verify_connection": "0.1.0",
    },
    "budgets": {"max_steps": 10, "max_total_attempts": 8, "timeout_s": 120.0},
    "start": "align",
    "nodes": {
        "align": {
            "skill": "lego.align_connection",
            "params": {
                "connection_id": "$input.connection_id",
                "target": "$input.target",
                "held": "$input.held",
                "build": "$input.build",
                "limits": "$input.align_limits",
            },
            "pre": [{"path": "$input.held.valid", "equals": True}],
            "attempts": 2,
            "retry_on": ["unverified"],
            "max_visits": 1,
            "timeout_s": 30.0,
            "on": {
                "succeeded": "seat",
                "rejected": "stop_replan",
                "failed": "stop_human",
                "failed:held_object_slipped": "stop_regrasp",
                "unverified": "stop_unknown",
            },
            "note": "guard: held.valid and fresh; brick within tolerance by a fresh observation",
        },
        "seat": {
            "skill": "lego.guarded_seat",
            "params": {
                "connection_id": "$input.connection_id",
                "alignment": "$last.align.outputs.alignment",
                "press_limits": "$input.press_limits",
            },
            "attempts": 1,
            "max_visits": 1,
            "timeout_s": 15.0,
            "on": {
                "succeeded": "release",
                "unverified:lego.seat_depth_without_threshold": "release",
                "rejected": "stop_replan",
                "failed": "stop_human",
                "unverified": "stop_human",
            },
            "note": "requires contact_force_feedback + compliant_motion; no position fallback",
        },
        "release": {
            "skill": "lego.release_and_retreat",
            "params": {
                "connection_id": "$input.connection_id",
                "object_id": "$input.held.object_id",
                "press": "$last.seat.outputs.press",
                "open_width_m": "$input.release.open_width_m",
                "retreat_height_m": "$input.release.retreat_height_m",
            },
            "attempts": 1,
            "max_visits": 1,
            "on": {
                "succeeded": "verify",
                "failed": "stop_human",
                "unverified": "stop_human",
                "rejected": "stop_human",
            },
        },
        "verify": {
            "skill": "lego.verify_connection",
            "params": {
                "connection_id": "$input.connection_id",
                "target": "$input.target",
                "build": "$input.build",
                "executor_id": "$last.seat.outputs.executor_id",
                "release": "$last.release.outputs.release",
                "pre_release_observation": "$last.release.outputs.pre_release_observation",
                "window": "$input.verify_window",
            },
            "attempts": 2,
            "retry_on": ["unverified"],
            "max_visits": 1,
            "on": {
                "succeeded": "done",
                "failed": "connection_failed",
                "unverified": "stop_unknown",
                "rejected": "stop_unknown",
            },
        },
    },
    "terminals": {
        "done": "succeeded",
        "stop_replan": "rejected",
        "stop_human": "failed",
        "stop_regrasp": "failed",
        "stop_unknown": "unverified",
        "connection_failed": "failed",
    },
}

CANDIDATE_STATUS = {
    "status": "candidate",
    "validated_under": None,
    "evidence_level": "unit_or_dryrun",
    "external_dependencies": {
        "held": "HeldObject from Agent 01 (perception_regrasp) or a gripper provider",
        "seat": "runtime primitive guarded_press + ctx.contact_description (integration "
        "request to Agent 05); a contact backend from Agent 04",
    },
}
