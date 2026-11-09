"""LEGO connection skills: align a held brick, press it with guarded contact, release and
retreat, verify the connection.

``SKILLS`` are contract ``rsh-task/1`` TaskSkills (registration is Agent 05's). The pure
functions below them are usable offline.
"""

from .align import align_connection, plan_alignment
from .seat import NOT_FORCE_CONTROL, check_contact_capabilities, guarded_seat
from .skills import REGISTRY, SKILLS
from .verify import verify_connection

__all__ = [
    "NOT_FORCE_CONTROL",
    "REGISTRY",
    "SKILLS",
    "align_connection",
    "check_contact_capabilities",
    "guarded_seat",
    "plan_alignment",
    "verify_connection",
]
