"""Local manifest of the LEGO task skills, their provenance, and the candidate graph.

Nothing here is registered globally; Agent 05 owns registration (``suggested_bundle``).
``python -m roboskillhub.tasks.lego.manifest OUT_DIR`` writes the JSON forms.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from . import contract_v1, seat
from .graph import CANDIDATE_GRAPH, CANDIDATE_STATUS
from .skills import SKILLS as TASK_SKILLS

HUB_BASE = "d015358cb7881bc215ada1aec419c787a79b1f12"

CONVENTIONS = {
    "contract": "rsh-task/1 (Agent 05 contract-v1); results are its SkillResult envelope, "
    "module detail under evidence.lego",
    "units": "metres, seconds, quaternion_xyzw; thresholds in m and deg",
    "frames": "T_A_B maps B to A; T_TCP_object = object pose in the TCP frame",
    "lego_brick_frame": "bottom-face centre, z up through the studs, x along studs[0]",
    "T_W_baseplate": "top-surface centre (stud bases), z out of the studs",
    "placement.layer": "0 = standing on the plate",
    "added_target_fields": [
        "baseplate.studs",
        "baseplate.calibration_id",
        "baseplate.system?",
        "baseplate.bias_m?",
        "brick.system?",
    ],
    "failure_types": contract_v1.FAILURE_TYPES,
}

NOTES: dict[str, dict[str, Any]] = {
    "lego.align_connection": {
        "summary": "Non-contact pre-alignment of a held brick over its target studs with "
        "bounded correction from fresh independent observations of the brick.",
        "pure_entrypoint": "roboskillhub.tasks.lego.align:align_connection",
        "inputs": "connection_id, target (LegoMatingTarget), held (HeldObject), build[], "
        "limits{xy_tolerance_m, yaw_tolerance_deg, workspace_min_m, workspace_max_m, "
        "limits_source, preassembly_height_m?, max_corrections?, max_total_correction_m?, "
        "speed_m_s?, deadline_s?, max_obs_age_s?, support_required_fraction?}, close_axis?",
        "preconditions": [
            "held.valid and fresh",
            "interface match",
            "cell on the plate",
            "target cells free",
            "support under layer > 0",
            "jaw rows clear of the build",
            "unambiguous symmetric orientation",
            "hover inside workspace",
            "observer resolves the tolerance",
            "tolerance within the press's known capture",
            "mating_pose agrees with placement",
        ],
        "executable": False,
        "executable_note": "runs on any backend with object_pose_feedback; none declares it "
        "unprivileged today (Isaac only with privileged truth)",
    },
    "lego.guarded_seat": {
        "summary": "Bounded contact press along the engagement axis with a measured force "
        "signal; returns the contact trace and stop reason; operation_complete only.",
        "pure_entrypoint": "roboskillhub.tasks.lego.seat:guarded_seat",
        "inputs": "connection_id, alignment (align outputs), press_limits{max_travel_m, "
        "speed_m_s, timeout_s, force_limit_N, limits_source, hold_s?, torque_limit_Nm?}, axis?",
        "control_statement": seat.NOT_FORCE_CONTROL,
        "needs_from_runtime": ["primitive guarded_press", "SkillContext.contact_description"],
        "executable": False,
        "executable_note": "no backend declares contact_force_feedback + compliant_motion; "
        "the guarded_press primitive does not exist yet",
    },
    "lego.release_and_retreat": {
        "summary": "set_gripper(release) with measured holding=false, then a bounded retreat "
        "along the engagement axis; observes the brick before and after.",
        "inputs": "connection_id, object_id, press (seat outputs), open_width_m, "
        "retreat_height_m, speed_m_s?",
        "executable": False,
        "executable_note": "retreat path is backend-defined (no linear_cartesian_motion)",
    },
    "lego.verify_connection": {
        "summary": "Executor-independent verdict satisfied/violated/unknown on stud cell, yaw, "
        "engagement height, tilt, post-release stability and the surrounding build.",
        "pure_entrypoint": "roboskillhub.tasks.lego.verify:verify_connection",
        "inputs": "connection_id, target, build[], executor_id, release (release outputs), "
        "pre_release_observation?, must_observe_neighbours?, window?, observations? (offline)",
        "verdict_to_status": {
            "satisfied": "succeeded",
            "violated": "failed",
            "unknown": "unverified",
        },
        "claim_on_success": "geometric_seating_verified (never retention force or interlock)",
        "executable": False,
        "executable_note": "meaningful only with an observer whose 2 sigma is under 0.35 stud "
        "heights (0.6 mm LEGO, 1.6 mm DUPLO); the pure function runs offline",
    },
}


def _entry(s: Any) -> dict[str, Any]:
    return {
        "skill_id": s.skill_id,
        "version": s.version,
        "kind": s.kind,
        "readiness": s.readiness,
        "evidence_level": "unit_or_dryrun",
        "entrypoint": f"roboskillhub.tasks.lego.skills:{type(s).__name__}",
        "requires": sorted(s.requires),
        "input_schema": s.input_schema,
        "postcondition": {"name": s.postcondition_name, "criterion": s.criterion},
        **NOTES[s.skill_id],
    }


SKILLS: list[dict[str, Any]] = [_entry(s) for s in TASK_SKILLS]

PROVENANCE: list[dict[str, Any]] = [
    {
        "id": "rsh-pressing-a-brick-onto-studs",
        "repository": "https://github.com/runtime-si/RoboSkillHub",
        "revision": HUB_BASE,
        "path": "skills/pressing-a-brick-onto-studs/scripts/press_brick.py",
        "sha256": "256076099325a1b5f4f1f9e43bc15ccbdb94feb69643c6be58f9d492f4dba6c8",
        "symbols": [
            "PRESETS",
            "FOOTPRINT_PLAY",
            "SEAT_FRACTION",
            "RESTING_FRACTION",
            "MISALIGNED_FRACTION",
            "PRESS_DEPTH",
        ],
        "license": "MIT (bundle; upstream BrickSim MIT, ICL Robotic_Lego_Manipulation MIT)",
        "use": "constants re-stated (LEGO/DUPLO presets, seat/resting bands, overtravel bound "
        "from PRESS_DEPTH idea); capture evidence (DUPLO 1 mm seated, 2 mm refused) quoted. "
        "Its position/lag press is NOT reused as contact control.",
        "reuse": "adapted_constants",
    },
    {
        "id": "rsh-computing-brick-pose-from-stud-grid",
        "repository": "https://github.com/runtime-si/RoboSkillHub",
        "revision": HUB_BASE,
        "path": "skills/computing-brick-pose-from-stud-grid/scripts/brick_pose.py",
        "sha256": "54aa253ee9b4c3bb451192a37085da83b22564b2fff993ff19dc449349b6769a",
        "symbols": [
            "_brick_top_in_plate",
            "_footprint",
            "_cells",
            "finger_cells (in run)",
            "the inverse (nearest cell + residual)",
            "collides/unsupported/ungraspable",
        ],
        "license": "MIT (bundle; upstream ICL Robotic_Lego_Manipulation Lego::calc_brick_loc, "
        "MIT, commit 9bc91654e708e4d8428ecfa76840103cacb2a19a)",
        "use": "grid chain ported to pure Python (no numpy) and cross-checked to 1e-9 m "
        "against the original in tests when numpy+gap are importable",
        "reuse": "ported",
    },
    {
        "id": "rsh-verifying-build-against-model",
        "repository": "https://github.com/runtime-si/RoboSkillHub",
        "revision": HUB_BASE,
        "path": "skills/verifying-build-against-model/scripts/verify_build.py",
        "sha256": "5bf32d1fe70081a84e71e7a6ea67762556bc5ac4f8840f53adc656d017d23fe3",
        "symbols": ["unseated / 'one end' rule", "SEAT_SIGMAS idea"],
        "license": "MIT (bundle; upstream LTRON MIT)",
        "use": "concept only: unseated = top high by about a stud, one-end = partial; "
        "thresholds widened by observation sigmas. Its LTRON matching is not used here (that "
        "skill stays the whole-build checker).",
        "reuse": "reference",
    },
    {
        "id": "rsh-checking-design-connectivity-and-collisions",
        "repository": "https://github.com/runtime-si/RoboSkillHub",
        "revision": HUB_BASE,
        "path": "skills/checking-design-connectivity-and-collisions/scripts/check_design.py",
        "sha256": "1611f77f08b109c91bec8352ebba95844a3d60ace6b53952fbb4a35d4ddbc8ef",
        "symbols": ["_brick_floats", "_connected_to_ground", "_overlap"],
        "license": "MIT (bundle; upstream BrickGPT MIT)",
        "use": "concept only: voxel collision and 'supported by the layer below' rules "
        "re-expressed per connection",
        "reuse": "reference",
    },
    {
        "id": "rsh-calibrating-baseplate-stud-grid",
        "repository": "https://github.com/runtime-si/RoboSkillHub",
        "revision": HUB_BASE,
        "path": "skills/calibrating-baseplate-stud-grid/scripts/stud_lattice.py",
        "sha256": "24d7732c5ea7989d4eca786fd16bfc27575326641faa6e462fc5143a2d7785a6",
        "symbols": ["plate pose / lattice"],
        "license": "MIT (bundle; upstream APEX-MR MIT)",
        "use": "upstream producer of plate.pose + calibration_id; no code used",
        "reuse": "consumer_of_output",
    },
    {
        "id": "lego-apexmr",
        "repository": "https://github.com/intelligent-control-lab/APEX-MR",
        "revision": "46a9448d9eaac5bd5973e0cd064623c5e7f5254e",
        "path": "src/lego_policy.cpp",
        "symbols": ["LegoPolicy::pressdown", "LegoPolicy::checkForce"],
        "license": "MIT",
        "license_evidence": "https://github.com/intelligent-control-lab/APEX-MR/blob/"
        "46a9448d9eaac5bd5973e0cd064623c5e7f5254e/LICENSE",
        "use": "structure only (approach / force-checked press / exit); no code, thresholds, "
        "tool transforms or distances copied (GP4 + F/T settings are not YAM defaults)",
        "reuse": "reference",
    },
    {
        "id": "lego-aidf",
        "repository": "https://github.com/intelligent-control-lab/AIDF",
        "revision": "2a2221f79e0e1cadbcd3809cfefa76f6ca4f6291",
        "path": "skillgraph/api/skills.hpp; skillgraph/src/lego/lego_skillgraph.cpp",
        "symbols": ["Skill/AtomicSkill/MetaSkill", "LegoSkillGraph::is_feasible"],
        "license": "license_unknown (package.xml <license>TODO</license>)",
        "use": "concept only: fixed K with feasibility preconditions, G over it; no code",
        "reuse": "reference_only_license_unknown",
    },
    {
        "id": "lego-brickcraft",
        "repository": "https://intelligent-control-lab.github.io/BrickCraft/",
        "revision": None,
        "license": "no public code",
        "use": "concept only: reference brick + relative xy/theta connection; not a dependency",
        "reuse": "reference_only",
    },
    {
        "id": "lego-roco",
        "repository": "https://github.com/rocochallenge/RoCo_BrickAssembly",
        "revision": "8e7ad4dae651fd96cad5b8aa145f3fe81d3a48f3",
        "license": "MIT (devkit only; BrickSim engine license/contact model unchecked)",
        "use": "not used by this module; Agent 04's scene/contact backend",
        "reuse": "none",
    },
]


def manifest() -> dict[str, Any]:
    return {
        "module": "roboskillhub.tasks.lego",
        "owner": "agent-03",
        "hub_base": HUB_BASE,
        "contract_version": "rsh-task/1",
        "conventions": CONVENTIONS,
        "skills": SKILLS,
        "suggested_bundle": {
            "name": "assembling-lego-connections",
            "capabilities": [s["skill_id"] for s in SKILLS],
            "registered": False,
            "note": "Agent 05 decides registration; no SKILL.md created",
        },
        "systems": {
            "lego": "8 mm pitch, 9.6 mm bricks, 1.7 mm studs",
            "duplo": "16 mm pitch, 19.2 mm bricks, 4.5 mm studs (not standard LEGO evidence)",
        },
        "provenance": PROVENANCE,
    }


def write(out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "skill-manifest.json": manifest(),
        "candidate-graph.json": {**CANDIDATE_GRAPH, "candidate": CANDIDATE_STATUS},
        "source-provenance.json": PROVENANCE,
    }
    paths = []
    for name, data in files.items():
        p = out / name
        p.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in write(sys.argv[1] if len(sys.argv) > 1 else "."):
        print(p)
