"""Load the hub's merged tool and skill code that this task reuses.

Skill scripts and tool bundles live outside the installed package (``tools/``,
``skills/`` at the hub root), so they are loaded by path from the checkout.
Nothing is copied: a fix upstream in those files reaches this task directly.
A wheel without the checkout raises :class:`HubSourceMissing`, which callers
turn into a ``rejected`` result rather than a fallback.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

#: Hub-relative files this task loads, by the name used here.
SOURCES = {
    "pose6d": "tools/pose6d/_impl.py",
    "pose6d_synth": "tools/pose6d/_synth.py",
    "domino_reader": "skills/reading-domino-pips-and-orientation/scripts/read_dominoes.py",
    "brick_pose": "skills/computing-brick-pose-from-stud-grid/scripts/brick_pose.py",
    "yam_chain": "skills/checking-dual-arm-clearance/scripts/check_clearance.py",
}


class HubSourceMissing(RuntimeError):
    """A reused hub file (or its import dependency) is not available."""


def hub_root() -> Path:
    env = os.environ.get("ROBOSKILLHUB_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "skills").is_dir() and (parent / "tools").is_dir():
            return parent
    raise HubSourceMissing("no RoboSkillHub checkout above this package; set ROBOSKILLHUB_ROOT")


@cache
def load(name: str) -> ModuleType:
    rel = SOURCES[name]
    path = hub_root() / rel
    if not path.is_file():
        raise HubSourceMissing(f"{rel} not found under {hub_root()}")
    mod_name = f"_rsh_reuse_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise HubSourceMissing(f"cannot load {rel}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        del sys.modules[mod_name]
        raise HubSourceMissing(f"{rel} needs {exc.name or exc}") from exc
    return module


def source_path(name: str) -> str:
    return SOURCES[name]
