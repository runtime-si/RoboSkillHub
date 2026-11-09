"""Domino task module (Agent 02): upright placement, independent stability check,
failure classification, bounded recovery plans and a straight-row layout tool.

Entry points (all return a result dict with ``status`` in succeeded / failed /
unverified / rejected):

- :func:`place_upright` (needs a :class:`PlacementBackend`; capability-gated)
- :func:`verify_stable_placement` (pure; observations in, verdict out)
- :func:`plan_straight_layout` (pure; target poses and build order, no motion)
- :func:`plan_recovery` (pure; a bounded plan, never executed here)

Not registered in the runtime/API yet: Agent 05 owns that wiring.
"""

from .contract import CAPABILITIES, CONTRACT_VERSION, threshold_profile
from .layout import plan_straight_layout
from .place import Clock, PlacementBackend, place_upright, plan_place
from .recovery import plan_recovery
from .verify import verify_stable_placement

__all__ = [
    "CAPABILITIES",
    "CONTRACT_VERSION",
    "Clock",
    "PlacementBackend",
    "place_upright",
    "plan_place",
    "plan_recovery",
    "plan_straight_layout",
    "threshold_profile",
    "verify_stable_placement",
]
