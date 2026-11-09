"""The LEGO placement verifier as a contract-v1 tool skill (``rsh-task/1``, section 8).

It judges one recorded placement from simulator snapshots; it does not move anything and needs no backend
capability. Inputs are privileged simulator truth, so results carry ``privileged_inputs: true``.
"""

from __future__ import annotations

import time
import uuid

from .lego_episode import load_config
from .lego_postconditions import evaluate_placement

CONTRACT_VERSION = "rsh-task/1"

RECOVERY = {
    "grasp_missed": ["regrasp", "reobserve"],
    "lego.misaligned": ["back_off", "retry_same"],
    "lego.not_seated": ["retry_same"],
    "lego.partially_seated": ["back_off", "retry_same"],
    "lego.lifted_with_gripper": ["reset_piece", "replan"],
    "lego.structure_damaged": ["replan", "human"],
    "lego.insufficient_support": ["replan"],
    "evidence_missing": ["reobserve"],
    "timeout": ["abort"],
}


class VerifyPlacement:
    skill_id = "assembly_eval.lego_verify_placement"
    version = "0.1.0"
    kind = "tool"
    requires: frozenset[str] = frozenset()
    readiness = "unit-tested"
    input_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["record", "index"],
        "properties": {
            "record": {
                "type": "object",
                "description": "episode record written by lego_isaac_runner",
            },
            "index": {"type": "integer", "minimum": 0},
            "accepted": {"type": "object", "description": "earlier accepted part -> {p, q}"},
        },
    }

    def run(self, request: dict, ctx=None) -> dict:
        t0 = time.monotonic()
        cfg = load_config()
        accepted = dict(request.get("accepted") or {})
        v = evaluate_placement(request["record"], request["index"], cfg, accepted)
        checks = v.get("postconditions", [])
        passed = v["status"] == "succeeded"
        return {
            "schema_version": 1,
            "contract_version": CONTRACT_VERSION,
            "invocation_id": str(uuid.uuid4()),
            "task_id": getattr(ctx, "task_id", None),
            "request_id": getattr(ctx, "request_id", None),
            "capability": self.skill_id,
            "skill": self.skill_id,
            "version": self.version,
            "kind": self.kind,
            "status": v["status"],
            "phase": v.get("phase", "verify"),
            "failure_type": v.get("failure_type"),
            "recovery_hints": [] if passed else RECOVERY.get(v.get("failure_type"), ["abort"]),
            "evidence": {
                "postcondition": {
                    "name": "lego_connection_engaged",
                    "passed": passed,
                    "criterion": "all of: " + ", ".join(c["name"] for c in checks),
                    "measured": {c["name"]: c["measured"] for c in checks},
                    "thresholds_ref": "src/roboskillhub/tasks/assembly_eval/config/lego_eval_v1.json",
                },
                "checks": checks,
                "evidence_level": "dynamic_sim_privileged",
            },
            "observations": [
                {
                    "object_id": f"part-{v['part']}",
                    "class": "lego_brick",
                    "frame_id": "world",
                    "position_m": s["parts"][v["part"]]["p"],
                    "quaternion_xyzw": s["parts"][v["part"]]["q"],
                    "timestamp_s": None,
                    "source": "simulation_truth",
                    "privileged": True,
                    "state_label": "seated" if passed else "uncertain",
                }
                for s in [
                    request["record"]["steps"][request["index"]].get("snapshots", {}).get("settled")
                ]
                if s is not None
            ],
            "privileged_inputs": True,
            "accepted_after": accepted,
            "elapsed_s": time.monotonic() - t0,
        }


SKILLS = [VerifyPlacement()]
