# SPDX-License-Identifier: Apache-2.0
"""``lego_press`` (``roboskillhub.tasks.graphs``) with the hub's verifiers and world model in the loop.

    select -> bind -> [face] -> observe_piece -> ... -> pick -> transport -> confirm -> align -> seat
      -> [pullup] -> release -> verify -> [depth] -> [track] -> gate -> select

- ``face``: ``hub.classify_resting_face`` on the piece before it is picked;
- ``pullup``: ``hub.pull_up_check`` while the hand still grips the seated brick;
- ``depth``: ``hub.verify_seat_depth`` over the seat after the release;
- ``track``: ``hub.track_build`` (seeded by ``track_init`` from the pre-built bricks at the start);
- ``gate``: ``hub.gate`` re-emits ``lego.verify_connection``'s status, so the graph's control flow is the
  original's.

``mode="report"`` (default): every hub node goes on to the next whatever it finds (its verdict is in the
episode, for comparison with the independent evaluator); only ``lego.verify_connection`` decides, through the
gate. ``mode="strict"``: a hub node's ``failed`` aborts (``unverified`` goes to ``unsure``).
"""

from __future__ import annotations

from typing import Any

from roboskillhub.runtime.task import SkillContext
from roboskillhub.tasks.graphs import _edges, lego_press

HUB_VERSION = "0.1.0"


class Gate:
    """Re-emit a status (and failure type) taken from an earlier node."""

    skill_id = "hub.gate"
    version = HUB_VERSION
    kind = "tool"
    requires = frozenset()
    readiness = "unit-tested"
    input_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["status"],
        "properties": {"status": {"type": "string"}, "failure_type": {}},
    }

    def run(self, request: dict, ctx: SkillContext) -> dict:
        status = request["status"]
        if status not in ("succeeded", "failed", "unverified", "rejected"):
            return ctx.result(self, "rejected", phase="validate", failure_type="invalid_request")
        failure = (
            None
            if status == "succeeded"
            else (request.get("failure_type") or "postcondition_failed")
        )
        return ctx.result(
            self, status, phase="verify", failure_type=failure, evidence={"gated_on": status}
        )


def _pass(next_node: str, mode: str) -> dict:
    if mode == "report":
        return {
            "succeeded": next_node,
            "failed": next_node,
            "unverified": next_node,
            "rejected": next_node,
        }
    return _edges(next_node)


def lego_press_hub(
    *,
    mode: str = "report",
    face: bool = True,
    pull_up: bool = True,
    depth: bool = True,
    track: bool = True,
    through_seat: bool = False,
    **press_kwargs: Any,
) -> dict:
    if mode not in ("report", "strict"):
        raise ValueError(f"mode must be report or strict, got {mode!r}")
    g = lego_press(**press_kwargs)
    nodes = g["nodes"]
    pieces = "$input.feeder.pieces"

    if face:
        after_bind = nodes["bind"]["on"]["succeeded"]
        nodes["bind"]["on"]["succeeded"] = "face"
        nodes["face"] = {
            "skill": "hub.classify_resting_face",
            "params": {
                "object_id": "$last.select.outputs.piece",
                "pieces": pieces,
                "table_z": "$input.table_z",
            },
            "timeout_s": 60,
            "max_visits": nodes["bind"].get("max_visits", 10),
            "on": _pass(after_bind, mode),
        }
        g["skills"]["hub.classify_resting_face"] = HUB_VERSION

    if through_seat:
        # Diagnostic: release the brick wherever a failed seat left it, so the verifiers and the evaluator
        # judge the same physical state (lego.guarded_seat's own verdict is in the episode).
        nodes["seat"]["on"] = {**nodes["seat"]["on"], "failed": "release", "unverified": "release"}
    if pull_up:
        nodes["seat"]["on"] = {
            k: ("pullup" if v == "release" else v) for k, v in nodes["seat"]["on"].items()
        }
        nodes["pullup"] = {
            "skill": "hub.pull_up_check",
            "params": {"object_id": "$last.select.outputs.piece"},
            "timeout_s": 60,
            "max_visits": nodes["seat"].get("max_visits", 10),
            "on": _pass("release", mode),
        }
        g["skills"]["hub.pull_up_check"] = HUB_VERSION

    # verify -> depth -> track -> gate -> (verify's own edges)
    verify_on = dict(nodes["verify"]["on"])
    chain = [n for n, on in (("depth", depth), ("track", track)) if on]
    if chain:
        first = chain[0]
        nodes["verify"]["on"] = {
            "succeeded": first,
            "failed": first,
            "unverified": first,
            "rejected": first,
        }
        if depth:
            nodes["depth"] = {
                "skill": "hub.verify_seat_depth",
                "params": {
                    "mating": "$last.bind.outputs.mating",
                    "pieces": pieces,
                    "prebuilt": "$input.prebuilt",
                },
                "timeout_s": 90,
                "max_visits": nodes["verify"].get("max_visits", 10),
                "on": _pass(chain[chain.index("depth") + 1] if "track" in chain else "gate", mode),
            }
            g["skills"]["hub.verify_seat_depth"] = HUB_VERSION
        if track:
            nodes["track"] = {
                "skill": "hub.track_build",
                "params": {
                    "action": "place",
                    "object_id": "$last.select.outputs.piece",
                    "state_json": "$state.build_state",
                    "mating": "$last.bind.outputs.mating",
                },
                "timeout_s": 30,
                "max_visits": nodes["verify"].get("max_visits", 10),
                "assign": {"build_state": "$last.track.outputs.state_json"},
                "on": _pass("gate", mode),
            }
            nodes["track_init"] = {
                "skill": "hub.track_build",
                "params": {
                    "action": "init",
                    "prebuilt": "$input.prebuilt",
                    "plate_id": "$input.targets.0.mating.baseplate.object_id",
                },
                "timeout_s": 30,
                "assign": {"build_state": "$last.track_init.outputs.state_json"},
                "on": _pass(g["start"], mode),
            }
            g["start"] = "track_init"
            g["state"]["build_state"] = ""
            g["skills"]["hub.track_build"] = HUB_VERSION
        nodes["gate"] = {
            "skill": "hub.gate",
            "params": {
                "status": "$last.verify.status",
                "failure_type": "$last.verify.failure_type",
            },
            "timeout_s": 5,
            "max_visits": nodes["verify"].get("max_visits", 10),
            "on": verify_on,
        }
        g["skills"]["hub.gate"] = HUB_VERSION
        # verify's assignment of done / placed only runs when verify succeeded, as before.

    g["graph_id"] += f"+hub-{mode}"
    g["params"] = {**g.get("params", {}), "hub": {"mode": mode, "face": face, "pull_up": pull_up,
                                                   "depth": depth, "track": track, "through_seat": through_seat}}  # fmt: skip
    b = g["budgets"]
    g["budgets"] = {
        **b,
        "max_steps": 2 * b["max_steps"],
        "max_total_attempts": 2 * b["max_total_attempts"],
        "timeout_s": min(600.0, 2 * b["timeout_s"]),  # the runtime's lease limit
    }
    return g
