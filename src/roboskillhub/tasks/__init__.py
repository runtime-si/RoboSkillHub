"""Task-skill library K: explicit registration of every task module's ``SKILLS``.

Modules are listed here by Agent 5 only after review; nothing is discovered
implicitly. A module that is absent from this checkout, or does not yet export
``SKILLS``, is reported in ``unavailable`` instead of silently dropped.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping

from roboskillhub.runtime.task import TaskSkill

MODULES = (
    "roboskillhub.tasks.core",
    "roboskillhub.tasks.perception_regrasp",
    "roboskillhub.tasks.domino",
    "roboskillhub.tasks.lego",
    "roboskillhub.tasks.assembly_eval",
)


def load_registry(modules: tuple[str, ...] = MODULES) -> tuple[dict[str, TaskSkill], dict]:
    """Return ``(K, unavailable)``: skills by id, and why each missing module is missing."""
    registry: dict[str, TaskSkill] = {}
    unavailable: dict[str, str] = {}
    for name in modules:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name and name.startswith(exc.name):
                unavailable[name] = "module not in this checkout"
                continue
            raise
        skills = getattr(module, "SKILLS", None)
        if skills is None:
            unavailable[name] = "module does not export SKILLS (contract rsh-task/1)"
            continue
        for skill in skills:
            if skill.skill_id in registry:
                raise ValueError(f"duplicate task skill id {skill.skill_id}")
            registry[skill.skill_id] = skill
    return registry, unavailable


def describe_registry(registry: Mapping[str, TaskSkill]) -> list[dict]:
    return [
        {
            "skill_id": s.skill_id,
            "version": s.version,
            "kind": s.kind,
            "requires": sorted(s.requires),
            "readiness": s.readiness,
            "input_schema": s.input_schema,
        }
        for s in sorted(registry.values(), key=lambda s: s.skill_id)
    ]
