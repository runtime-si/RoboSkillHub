"""VLA-as-skill core (VLA_SKILL_SPEC): checkpoint profiles, observations, a safe chunk executor and a
closed loop that any action-model backend (gr00t, molmoact2, pi05) runs in.

Pure numpy + PyYAML (no gap, no torch), so a twin harness can import it beside a connector.
"""

from .client import PolicyServerError, VlaYamClient, default_url
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
    with_station,
)

__all__ = [
    "ActionSpec",
    "GripperMap",
    "ObservationError",
    "Plan",
    "PolicyClient",
    "PolicyServerError",
    "Profile",
    "ProfileError",
    "Refusal",
    "SafetyLimits",
    "Segment",
    "Slot",
    "VlaYamClient",
    "camera_roles",
    "check",
    "default_url",
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
    "with_station",
]
