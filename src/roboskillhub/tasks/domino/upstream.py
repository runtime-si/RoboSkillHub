"""The hub's own domino skills, loaded in place so their algorithms are reused, not copied.

- ``place``: skills/placing-an-upright-domino-beside-neighbours/scripts/place_domino.py
  (the hand's finger-clearance rule ``finger_conflicts``, the depth upright check
  ``judge_tile`` / ``order_cameras``, and the place constants it measured in a YAM
  station twin: 44/44 tiles stood).
- ``layout``: skills/planning-a-domino-layout/scripts/plan_layout.py (footprints,
  ``poly_distance``, the arm-corridor rule ``in_corridor``, the toppling physics
  ``critical_gap``, ``contact``; the far-first build-order rule).

Both scripts import ``gap.NodeContext`` (the ``gap`` extra) and plan_layout needs scipy.
When they cannot be loaded the caller gets :class:`UpstreamUnavailable`, which the
task functions turn into a ``rejected`` result naming the missing dependency.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS = {
    "place": ("placing-an-upright-domino-beside-neighbours", "place_domino.py"),
    "layout": ("planning-a-domino-layout", "plan_layout.py"),
}


class UpstreamUnavailable(RuntimeError):
    """A hub skill script this module reuses could not be found or imported."""


def skills_root() -> Path:
    """The hub's ``skills/``: $ROBOSKILLHUB_ROOT/skills, else the checkout this file is in."""
    env = os.environ.get("ROBOSKILLHUB_ROOT")
    if env:
        return Path(env) / "skills"
    for parent in Path(__file__).resolve().parents:
        if (parent / "skills" / SCRIPTS["place"][0]).is_dir():
            return parent / "skills"
    raise UpstreamUnavailable(
        "no RoboSkillHub skills/ directory found (set ROBOSKILLHUB_ROOT to a hub checkout)"
    )


@functools.cache
def load(name: str) -> ModuleType:
    bundle, script = SCRIPTS[name]
    path = skills_root() / bundle / "scripts" / script
    if not path.is_file():
        raise UpstreamUnavailable(f"{bundle}/scripts/{script} is missing")
    module_name = f"roboskillhub_domino_upstream_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise UpstreamUnavailable(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        sys.modules.pop(module_name, None)
        raise UpstreamUnavailable(
            f"{bundle} needs {exc.name or exc} (install the hub's 'gap' extra and scipy)"
        ) from exc
    return module


def place() -> ModuleType:
    return load("place")


def layout() -> ModuleType:
    return load("layout")


def provenance() -> dict[str, str]:
    """Which hub script each reused symbol comes from (for result evidence)."""
    return {name: f"skills/{bundle}/scripts/{script}" for name, (bundle, script) in SCRIPTS.items()}
