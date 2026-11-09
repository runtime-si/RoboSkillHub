"""rsh-task/1 TaskSkill entries for the domino module (Agent 05 registers them).

``run(request, ctx)``: tools accept ``ctx=None`` and never touch a robot; place_upright
reaches the station only through ``ctx.invoke`` / ``ctx.observe``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contract import RECOVERY_HINTS, SKILL_VERSION, envelope, place_requires
from .layout import plan_straight_layout
from .place import place_upright
from .recovery import plan_recovery
from .verify import verify_stable_placement

_POSE = {"$ref": "rsh-task/1#/schemas/Pose"}
_OBS_LIST = {"type": "array", "items": {"$ref": "rsh-task/1#/schemas/Observation"}}
_OBJ = {"type": "object"}
_NUM = {"type": "number"}
_STR = {"type": "string"}


def _schema(required: list[str], properties: dict[str, Any]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


_COMMON = {
    "task_id": _STR,
    "request_id": _STR,
    "frame_id": _STR,
    "target_id": _STR,
    "geometry": _OBJ,
    "tolerances_ref": _STR,
    "thresholds_override": _OBJ,
    "now_s": _NUM,
}

PLACE_SCHEMA = _schema(
    [
        "frame_id",
        "target_id",
        "geometry",
        "target_object_pose",
        "held",
        "support",
        "workspace",
        "timeout_s",
    ],
    {
        **_COMMON,
        "scene_version": _STR,
        "target_object_pose": _POSE,
        "held": {"$ref": "rsh-task/1#/schemas/HeldObject"},
        "support": _OBJ,
        "neighbors": _OBS_LIST,
        "approach": _OBJ,
        "retreat": _OBJ,
        "release": _OBJ,
        "hand": _OBJ,
        "workspace": _OBJ,
        "mode": {"enum": ["geometric", "contact_feedback"]},
        "descent_path": {"enum": ["dense_waypoints", "linear"]},
        "contact": _OBJ,
        "speed_m_s": _NUM,
        "descent_speed_m_s": _NUM,
        "timeout_s": _NUM,
        "primitive_timeout_s": _NUM,
        "recovery_attempts_used": {"type": "integer"},
        "max_recovery_attempts": {"type": "integer"},
    },
)

VERIFY_SCHEMA = _schema(
    [
        "frame_id",
        "target_id",
        "geometry",
        "target_object_pose",
        "support",
        "events",
        "observations",
    ],
    {
        **_COMMON,
        "target_object_pose": _POSE,
        "support": _OBJ,
        "neighbors_baseline": _OBS_LIST,
        "events": _OBJ,
        "observations": _OBS_LIST,
        "gripper_states": {"type": "array", "items": _OBJ},
        "tcp_states": {"type": "array", "items": _OBJ},
        "gripper_tip_reach_m": _NUM,
        "require_unprivileged": {"type": "boolean"},
        "snapshot_diff": _OBJ,
    },
)

LAYOUT_SCHEMA = _schema(
    ["frame_id", "geometry", "start_m", "heading_deg", "n_tiles", "table_z_m", "spacing"],
    {
        "task_id": _STR,
        "request_id": _STR,
        "layout_id": _STR,
        "frame_id": _STR,
        "geometry": _OBJ,
        "start_m": {"type": "array", "minItems": 2, "maxItems": 2, "items": _NUM},
        "heading_deg": _NUM,
        "n_tiles": {"type": "integer"},
        "table_z_m": _NUM,
        "spacing": _OBJ,
        "arm_base_xy_m": {"type": "array", "minItems": 2, "maxItems": 2, "items": _NUM},
        "workspace": _OBJ,
        "open_width_m": _NUM,
        "order_preference": {
            "type": "array",
            "items": {"enum": ["far_first", "forward", "reverse"]},
        },
        "existing": {"type": "array", "items": _OBJ},
        "cascade": _OBJ,
        "tolerances_ref": _STR,
    },
)

RECOVERY_SCHEMA = _schema(
    ["result"],
    {"result": _OBJ, "attempts_used": {"type": "integer"}, "max_attempts": {"type": "integer"}},
)


@dataclass(frozen=True)
class DominoSkill:
    skill_id: str
    kind: str
    requires: frozenset[str]
    input_schema: dict
    readiness: str
    _run: Any = field(repr=False)
    version: str = SKILL_VERSION

    def run(self, request: dict, ctx: Any = None) -> dict:
        return self._run(request, ctx)


def _recovery_tool(request: dict, ctx: Any) -> dict:
    plan = plan_recovery(
        request["result"],
        attempts_used=int(request.get("attempts_used", 0)),
        max_attempts=int(request.get("max_attempts", 1)),
    )
    out = envelope("domino.plan_recovery", request, kind="tool", ctx=ctx)
    out.update(
        status="succeeded",
        phase="verify",
        evidence={"plan": plan},
        outputs={"recovery": plan},
        recovery_hints=RECOVERY_HINTS[plan["action"]],
    )
    return out


SKILLS = [
    DominoSkill(
        "domino.place_upright",
        "skill",
        place_requires("geometric", "dense_waypoints"),
        PLACE_SCHEMA,
        "unit-tested",
        lambda request, ctx: place_upright(request, ctx),
    ),
    DominoSkill(
        "domino.verify_stable_placement",
        "tool",
        frozenset(),
        VERIFY_SCHEMA,
        "unit-tested",
        lambda request, ctx: verify_stable_placement(request, ctx),
    ),
    DominoSkill(
        "domino.plan_straight_layout",
        "tool",
        frozenset(),
        LAYOUT_SCHEMA,
        "unit-tested",
        lambda request, ctx: plan_straight_layout(request),
    ),
    DominoSkill(
        "domino.plan_recovery",
        "tool",
        frozenset(),
        RECOVERY_SCHEMA,
        "unit-tested",
        lambda request, ctx: _recovery_tool(request, ctx),
    ),
]
