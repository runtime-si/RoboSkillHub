"""Loader-level checks for the four tool bundles + repo-wide format gate.

- the checkout exposes exactly the expected tool bundles;
- every name declared under ``gap.tools:`` in a bundle's SKILL.md has a
  matching ``@tool`` registration from that bundle's tools.py (and nothing
  undeclared leaks out);
- schema extraction works on every tool (no empty-schema fallback);
- importing the bundles is lazy: no torch / transformers / curobo / sam3
  in ``sys.modules`` after the loader has imported every tools.py;
- every bundle in the repo passes the engine-side format validation that
  ``gap skills check`` runs (:mod:`gap.skills.validate` — the rules live in
  the gap engine so third-party checkouts get the same checker).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SKILLS_ROOT = Path(__file__).resolve().parents[2]

TOOL_BUNDLES = ("curobo", "curve", "geometry", "grounding-dino", "sam3")

#: Heavy ML packages that must only load inside tool function bodies.
_FORBIDDEN_TOP_LEVEL = {"torch", "transformers", "curobo", "sam3", "skimage"}


def test_four_tool_bundles_discovered(skills_registry):
    names = {info.name for info in skills_registry.list_skills(kind="tool")}
    assert set(TOOL_BUNDLES) <= names
    for bundle in TOOL_BUNDLES:
        info = skills_registry.get(bundle)
        assert info.kind == "tool"
        assert info.namespace == "tools"
        assert info.tools_module is not None, f"{bundle}: tools.py did not import"
        assert info.meta.tools, f"{bundle}: SKILL.md declares no gap.tools"


def test_lazy_import_no_heavy_deps_in_sys_modules():
    """Importing every bundle's tools.py must not pull GPU/ML deps."""
    # Real tracker tests legitimately import torch. Check loader side effects in
    # a clean interpreter so the result does not depend on test collection order.
    script = """
import sys
from gap.skills import load_skills
registry = load_skills(sys.argv[1])
for info in registry._skills.values():
    path = info.bundle_dir / "tools.py"
    if info.tools_module is None and path.is_file():
        registry._import_module(path, f"gap_skills.{info.namespace}.{info.name}.tools")
forbidden = set(sys.argv[2:])
leaked = forbidden & {name.split(".")[0] for name in sys.modules}
assert not leaked, f"bundle import pulled heavy deps eagerly: {sorted(leaked)}"
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(SKILLS_ROOT), *sorted(_FORBIDDEN_TOP_LEVEL)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_declared_tools_match_registered_tools(skills_registry, tool_registry):
    for bundle in TOOL_BUNDLES:
        info = skills_registry.get(bundle)
        declared = set(info.meta.tools)

        # Names follow the <bundle>.<func> convention.
        for name in declared:
            assert name.startswith(f"{bundle}."), (
                f"{bundle}: declared tool {name!r} not namespaced under the bundle"
            )

        registered = {
            name
            for name, desc in tool_registry.runtime_tools().items()
            if desc.metadata.get("module") == f"gap_skills.tools.{bundle}.tools"
        }
        assert declared == registered, (
            f"{bundle}: SKILL.md gap.tools and @tool registrations disagree; "
            f"declared-only={sorted(declared - registered)}, "
            f"registered-only={sorted(registered - declared)}"
        )


def test_schema_extraction_works_on_every_tool(skills_registry, tool_registry):
    for bundle in TOOL_BUNDLES:
        info = skills_registry.get(bundle)
        for name in info.meta.tools:
            desc = tool_registry.get(name)
            assert desc.summary, f"{name}: empty summary"
            # Every bundle tool takes at least one input; an empty inputs dict
            # means extract_schema fell back to the bare UnitSchema.
            assert desc.schema.inputs, f"{name}: schema extraction fell back/empty"
            for field in desc.schema.inputs.values():
                assert field.type_str, f"{name}: untyped input {field.name}"


def test_repo_passes_engine_format_validation():
    """Every bundle passes the `gap skills check` format rules: frontmatter
    shape per kind, referenced resources on disk, allowed_tools resolvable
    against connector + declared bundle tools, and declared type names.

    The engine's one-pip-extra-per-bundle convention is only a warning there
    and is not kept here: the root pyproject is roboskillhub's own, and a
    `kind: skill` bundle runs in gap-runtime's venv with no dependencies of
    its own, while tool and policy bundles carry their own pyproject.toml.
    """
    from gap.skills.validate import validate_checkout

    reports = validate_checkout(SKILLS_ROOT)
    assert len(reports) >= 17, "bundle discovery regressed"

    failures = {r.name: [str(i) for i in r.errors] for r in reports if r.errors}
    assert not failures, f"format validation errors: {failures}"


def test_tool_tags_classify_bundles(skills_registry, tool_registry):
    perception_bundles = {"sam3", "grounding-dino"}
    for bundle in TOOL_BUNDLES:
        info = skills_registry.get(bundle)
        for name in info.meta.tools:
            tags = set(tool_registry.get(name).tags)
            if bundle in perception_bundles:
                assert "perception" in tags, f"{name}: missing perception tag"
            elif bundle == "curobo":
                assert "planning" in tags, f"{name}: missing planning tag"
            else:  # geometry is judged per tool
                assert tags & {"perception", "planning"}, f"{name}: untagged"
