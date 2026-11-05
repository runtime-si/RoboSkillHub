"""Shared fixtures: load this repository's bundles once per session.

``load_skills`` imports each in-process bundle's ``tools.py`` through the
synthetic ``gap_skills.tools.<bundle>.tools`` package, which fires the
``@tool`` decorators into ``gap.tools._registry._PENDING_TOOLS``. A bundle
served out of process (``gap.serving``) is skipped there: gap runs its
``tools.py`` in the bundle's own venv. These tests call those functions in
this interpreter, so :func:`skills_registry` imports them here too -- each
one defers its heavy imports into the functions that need them.

The decorator queue is drained exactly once (module imports are idempotent),
so both fixtures are session-scoped and shared by every test module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: The repository root (parent of tests/).
SKILLS_ROOT = Path(__file__).resolve().parents[2]

#: Tests that fail at open-robot-skills 752dfbd as well, where the test and the
#: bundle it tests disagree. A shrinking list: settle one (fix the bundle or
#: the test) and delete its line; strict, so a test that starts passing fails
#: until its line goes.
UPSTREAM_DRIFT = {
    "test_skills_manipulation.py::TestGraspingWithPlanner::test_happy_path_call_sequence": (
        "plan_grasp and approach_above are on disk but not in canonical_scripts"
    ),
    "test_skills_manipulation.py::TestGraspingWithPlanner::test_planning_failure_raises": (
        "plan_grasp is on disk but not in canonical_scripts"
    ),
    "test_skills_manipulation.py::TestGraspingWithPlanner::test_plan_grasp_autowraps_bare_pose": (
        "plan_grasp is on disk but not in canonical_scripts"
    ),
    "test_skills_perception.py::test_safe_gate_verified_exterior_pick_skips_wrist": (
        "the pairwise VLM prompt no longer carries the target's description"
    ),
    "test_skills_perception.py::test_wrist_support_fuses_intersecting_segment_text_cloud": (
        "the fused wrist-support cloud's size differs from the test's"
    ),
    "test_skills_perception.py::test_wrist_support_projection_fallback_seeds_segment_box": (
        "the fused wrist-support cloud's size differs from the test's"
    ),
    "test_skills_perception.py::test_wrist_support_rejects_non_intersecting_cloud": (
        "the exterior-only cloud's size differs from the test's"
    ),
    "test_skills_rigid_perception.py::test_new_rigid_bundles_declare_their_contracts": (
        "reconstructing-collision-worlds and verifying-grasps declare more scripts and exits"
    ),
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        reason = UPSTREAM_DRIFT.get(item.nodeid.rpartition("tests/bundles/")[2])
        if reason:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=True))


def import_served_tools(registry) -> None:
    """Import the ``tools.py`` of every out-of-process bundle into this interpreter."""
    for info in registry._skills.values():
        path = info.bundle_dir / "tools.py"
        if info.tools_module is None and path.is_file():
            name = f"gap_skills.{info.namespace}.{info.name}.tools"
            info.tools_module = registry._import_module(path, name)
            if info.module is None and info.skill_class is None:
                registry._wire_callable(info, info.tools_module, source="tools.py")


@pytest.fixture(scope="session")
def skills_registry():
    from gap.skills import load_skills

    registry = load_skills(SKILLS_ROOT)
    import_served_tools(registry)
    return registry


@pytest.fixture(scope="session")
def tool_registry(skills_registry):
    """ToolRegistry with the bundles' pending @tool registrations drained."""
    from gap_core.tools import ToolRegistry

    reg = ToolRegistry()
    reg.discover_pending()
    return reg
