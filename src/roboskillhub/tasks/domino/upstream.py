"""The hub's own domino skills, loaded in place so their algorithms are reused, not copied.

- ``place``: skills/placing-an-upright-domino-beside-neighbours/scripts/place_domino.py
  (the hand's finger-clearance rule ``finger_conflicts``, the depth upright check
  ``judge_tile`` / ``order_cameras``, and the place constants it measured in a YAM
  station twin: 44/44 tiles stood).
- ``layout``: skills/planning-a-domino-layout/scripts/plan_layout.py (footprints,
  ``poly_distance``, the arm-corridor rule ``in_corridor``, the toppling physics
  ``critical_gap``, ``contact``; the far-first build-order rule).

Where the scripts are looked for, in order:

1. ``$ROBOSKILLHUB_ROOT/skills`` (a hub checkout named explicitly);
2. ``roboskillhub/_skills/`` inside the installed package, if the wheel ships the two
   scripts there (a build-config change requested from Agent 05; not in place yet);
3. the source checkout this file sits in (``<repo>/skills``).

Both scripts import ``gap.NodeContext`` (the ``gap`` extra) and plan_layout needs scipy.
When a script is absent or cannot be imported the caller gets
:class:`UpstreamUnavailable`, which every entry that needs it turns into
``rejected / precondition_unmet`` naming what is missing and where it looked.
``verify_stable_placement`` and ``plan_recovery`` never need these scripts.
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
PACKAGED = Path(__file__).resolve().parents[2] / "_skills"


class UpstreamUnavailable(RuntimeError):
    """A hub skill script this module reuses could not be found or imported."""


def candidate_roots() -> list[Path]:
    roots = []
    env = os.environ.get("ROBOSKILLHUB_ROOT")
    if env:
        roots.append(Path(env) / "skills")
    roots.append(PACKAGED)
    for parent in Path(__file__).resolve().parents:
        if (parent / "skills").is_dir() and (parent / "pyproject.toml").is_file():
            roots.append(parent / "skills")
            break
    return roots


def script_path(name: str) -> Path:
    bundle, script = SCRIPTS[name]
    tried = []
    for root in candidate_roots():
        path = root / bundle / "scripts" / script
        if path.is_file():
            return path
        tried.append(str(path))
    raise UpstreamUnavailable(
        f"skills/{bundle}/scripts/{script} is not installed (looked in: {', '.join(tried)}). "
        "This entry reuses that hub skill's code in place: run from a RoboSkillHub checkout or "
        "set ROBOSKILLHUB_ROOT to one; a wheel does not ship skills/."
    )


@functools.cache
def load(name: str) -> ModuleType:
    bundle, script = SCRIPTS[name]
    path = script_path(name)
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
            f"{path} needs {exc.name or exc} (install the hub's 'gap' extra and scipy)"
        ) from exc
    return module


def place() -> ModuleType:
    return load("place")


def layout() -> ModuleType:
    return load("layout")


def available() -> dict[str, str]:
    """{name: "" if loadable else why} without raising (for describe / diagnostics)."""
    out = {}
    for name in SCRIPTS:
        try:
            load(name)
            out[name] = ""
        except UpstreamUnavailable as exc:
            out[name] = str(exc)
    return out


def provenance() -> dict[str, str]:
    """Which hub script each reused symbol comes from (for result evidence)."""
    return {name: f"skills/{bundle}/scripts/{script}" for name, (bundle, script) in SCRIPTS.items()}
