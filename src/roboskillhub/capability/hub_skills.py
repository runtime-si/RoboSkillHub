"""Load the pure helpers of in-hub skill scripts without a graph runtime.

The verifier composes skills' measurement code (the opening tracker, the scene-diff region
numbers) in-process. Those scripts import ``gap.NodeContext`` only as a type, so when gap
is not installed a placeholder module stands in while the script loads and is removed after.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from functools import cache
from pathlib import Path
from types import ModuleType

SKILLS = Path(__file__).resolve().parents[3] / "skills"

GRIPPER_STALL = ("tracking-gripper-opening-stall", "track_opening.py")
SCENE_DIFF = ("checking-pick-by-scene-diff", "check_scene_diff.py")


@cache
def load(skill: str, script: str) -> ModuleType:
    path = SKILLS / skill / "scripts" / script
    if not path.is_file():
        raise FileNotFoundError(f"hub skill script not found: {path}")
    placeholder = None
    if "gap" not in sys.modules and importlib.util.find_spec("gap") is None:
        placeholder = types.ModuleType("gap")
        placeholder.NodeContext = object  # type: ignore[attr-defined]
        sys.modules["gap"] = placeholder
    try:
        spec = importlib.util.spec_from_file_location(f"_rsh_skill_{skill.replace('-', '_')}", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if placeholder is not None and sys.modules.get("gap") is placeholder:
            del sys.modules["gap"]
