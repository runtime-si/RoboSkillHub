"""LEGO connection skills: align a held brick, press it with guarded contact, verify it.

Module-local (Agent 03). Not registered in the shared runtime; see manifest.py.
"""

from .align import align_connection, plan_alignment
from .seat import NOT_FORCE_CONTROL, check_contact_capabilities, guarded_seat
from .verify import verify_connection

__all__ = [
    "NOT_FORCE_CONTROL",
    "align_connection",
    "check_contact_capabilities",
    "guarded_seat",
    "plan_alignment",
    "verify_connection",
]
