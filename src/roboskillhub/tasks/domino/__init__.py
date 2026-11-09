"""Domino task module (Agent 02): upright placement, independent stability check,
failure classification, bounded recovery plans and a straight-row layout tool.

Aligned to the shared contract rsh-task/1. ``SKILLS`` lists the TaskSkill entries
Agent 05 registers; nothing here registers itself in the runtime/API.

- ``domino.place_upright`` (skill; ctx.invoke move_waypoints / set_gripper,
  ctx.observe; capability-gated)
- ``domino.verify_stable_placement`` (tool; observations in, verdict out)
- ``domino.plan_straight_layout`` (tool; target poses and build order, no motion)
- ``domino.plan_recovery`` (tool; a bounded plan, never executed here)
"""

from .contract import CAPABILITIES, CONTRACT_VERSION, threshold_profile
from .layout import plan_straight_layout
from .place import TaskContext, place_upright, plan_place
from .recovery import plan_recovery
from .skills import SKILLS
from .verify import verify_stable_placement

__all__ = [
    "CAPABILITIES",
    "CONTRACT_VERSION",
    "SKILLS",
    "TaskContext",
    "place_upright",
    "plan_place",
    "plan_recovery",
    "plan_straight_layout",
    "threshold_profile",
    "verify_stable_placement",
]
