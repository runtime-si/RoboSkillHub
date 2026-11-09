"""Injectable ports the LEGO skills drive. None of these exists in the shared runtime yet;
they are the interface requested from Agent 05 (motion) and Agent 04 (contact, observers).

Every method must honour ``timeout_s``. ``describe()`` must not move anything.
"""

from __future__ import annotations

from typing import Any, Protocol


class MotionPort(Protocol):
    """Non-contact Cartesian motion of the TCP, e.g. an adapter over the runtime's
    ``move_pose`` that holds the station lease. Returns ``{ok, measured_tcp_pose?, ...}``."""

    def describe(self) -> dict[str, Any]: ...
    def move_pose(self, pose: dict, *, speed_m_s: float, timeout_s: float) -> dict[str, Any]: ...
    def stop(self) -> dict[str, Any]: ...


class BrickObserver(Protocol):
    """Independent observation of a brick, not derived from commands.

    ``observe(object_id, not_before_s, timeout_s)`` returns an Observation dict whose
    ``value`` is the brick's bottom-centre pose (``{frame_id, position_m,
    quaternion_xyzw}``) or ``None`` when the brick is not visible. ``describe()`` gives
    ``{observer_id, evidence_level, uncertainty_m}``."""

    def describe(self) -> dict[str, Any]: ...
    def observe(self, object_id: str, *, not_before_s: float, timeout_s: float) -> dict: ...


class ContactBackend(Protocol):
    """Bounded contact motion along one axis with a measured contact signal.

    ``describe()``::

        {backend_id, evidence_level, capabilities: [...],
         contact_signal: {source, quantity, unit, frame_id, rate_hz, max_latency_s,
                          calibration_id},
         controller: {kind, loop_hz, process},     # where the closed loop runs
         limits: {max_force_N, max_speed_m_s, max_travel_m}}

    ``guarded_move(...)`` runs the whole bounded motion in the controller and returns::

        {command_issued: bool, stop_reason: str, stop_confirmed: bool,
         samples: [{t_s, travel_m, force_N}], fault: str}

    ``travel_m`` is the measured TCP travel along ``axis`` from the start; ``force_N`` the
    measured contact force opposing the motion (positive), in the signal's units."""

    def describe(self) -> dict[str, Any]: ...
    def guarded_move(
        self,
        *,
        axis: dict,
        max_travel_m: float,
        speed_m_s: float,
        force_limit_N: float,
        hold_s: float,
        timeout_s: float,
    ) -> dict[str, Any]: ...
    def stop(self) -> dict[str, Any]: ...


CONTACT_CAPABILITIES = frozenset(
    {"pose_feedback", "contact_force_feedback", "compliant_motion", "stop"}
)
"""What lego.guarded_seat requires (contract rsh-task/1 capability names). Every
``guarded_move`` must also report ``stop_confirmed``. ``bounded_position_approach``
(position control with a travel limit, tracking lag or joint-effort heuristics) is NOT in
this set: position-only pressing is not compliant force control."""

STOP_REASONS = (
    "force_threshold",  # measured force reached force_limit_N; motion stopped
    "travel_limit",  # max_travel_m reached
    "timeout",
    "signal_lost",
    "fault",
)
