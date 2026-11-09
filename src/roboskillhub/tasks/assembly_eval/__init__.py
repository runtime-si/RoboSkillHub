"""Simulation evaluation of the LEGO and domino tasks: scenes, contact models, injected faults and independent
post-conditions. Files prefixed ``lego_`` are the LEGO part (4a), ``domino_`` the domino part (4b). Isaac-only
runners and stations import the simulator inside their entry points, so the rest imports on a CPU-only machine.

``SKILLS`` (contract rsh-task/1) is the domino tools followed by the LEGO tools; both are evaluators (kind
``tool``) that read recorded simulator truth and move nothing.
"""

from .domino_skills import SKILLS as DOMINO_SKILLS
from .lego_tools import SKILLS as LEGO_SKILLS

SKILLS = [*DOMINO_SKILLS, *LEGO_SKILLS]

__all__ = ["DOMINO_SKILLS", "LEGO_SKILLS", "SKILLS"]
