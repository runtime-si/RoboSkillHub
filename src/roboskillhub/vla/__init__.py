"""VLA-as-skill core (VLA_SKILL_SPEC): checkpoint profiles, observations, a safe chunk executor and a
closed loop that any action-model backend (gr00t, molmoact2, pi05) runs in.

Pure numpy + PyYAML (no gap, no torch), so a twin harness can import it beside a connector.
"""

from .executor import (
    Plan,
    Refusal,
    SafetyLimits,
    Segment,
    execute_chunk,
    fk_from_ctx,
    interpret,
    plan_chunk,
    send_plan,
)
from .loop import PolicyClient, run_policy
from .observation import (
    ObservationError,
    camera_roles,
    check,
    from_connector,
    get_observation,
    model_input,
    staleness,
    station_state,
)
from .profile import (
    ActionSpec,
    GripperMap,
    Profile,
    ProfileError,
    Slot,
    load_profile,
    profile_from_dict,
)

__all__ = [
    "ActionSpec",
    "GripperMap",
    "ObservationError",
    "Plan",
    "PolicyClient",
    "Profile",
    "ProfileError",
    "Refusal",
    "SafetyLimits",
    "Segment",
    "Slot",
    "camera_roles",
    "check",
    "execute_chunk",
    "fk_from_ctx",
    "from_connector",
    "get_observation",
    "interpret",
    "load_profile",
    "model_input",
    "plan_chunk",
    "profile_from_dict",
    "run_policy",
    "send_plan",
    "staleness",
    "station_state",
]
