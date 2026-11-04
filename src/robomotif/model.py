"""What the hub knows about one bundle, recipe or paper.

A bundle is a gap bundle unchanged. Everything the hub adds -- which stage of a
task it serves, which robots it is declared for, its version, where its weights
live -- sits under ``metadata.hub`` in its ``SKILL.md`` (gap reads only
``metadata.category`` and ``metadata.tags``), or, for a bundle we federate but do
not own, in that source's overlay file. What can be derived from the bundle
itself (its exits, the tools it calls, whether it reads simulator ground truth)
is derived, never declared.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KIND_DIRS = {"tools": "tool", "skills": "skill", "policies": "policy"}
"""Bundle roots inside a registry, and the kind each one holds."""

STAGES = ("perceive", "plan", "propose", "select", "execute", "verify", "recover", "judge")
"""Where in a task a skill or policy sits. A tool bundle may leave it unset."""

TIERS = ("camera_only", "full")
"""``full`` reads simulator ground truth (``sim.*``) and cannot leave simulation."""

MATURITY = ("community", "sim-verified", "real-verified")

CONNECTOR_PREFIXES = ("robot", "sim", "motion", "cable")
"""Tool namespaces a connector owns rather than a bundle."""

HUB_KEYS = frozenset(
    {
        "stage",
        "embodiments",
        "tuned_on",
        "arms",
        "tasks",
        "version",
        "provenance",
        "weights",
        "io",
        "tier",
        "notes",
    }
)
PROVENANCE_KEYS = frozenset({"authors", "paper", "code", "upstream_license", "data"})
WEIGHTS_KEYS = frozenset({"uri", "sha256", "license"})
IO_KEYS = frozenset({"action", "state", "cameras"})
"""A policy's ``io``: the observation and action layout its checkpoint was trained with."""
IO_ACTION_KEYS = frozenset({"space", "dim", "hz"})
IO_STATE_KEYS = frozenset({"dim", "order"})

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_SKIP = frozenset({".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".DS_Store", ".git"})


def tree_sha256(root: Path) -> str:
    """One digest over every file under *root*: relative path and content.

    Byte-identical checkouts hash identically wherever they sit, so a lock can
    say which bundle a graph ran with, not only which name.
    """
    digest = hashlib.sha256()
    for path in sorted(_files(root), key=lambda p: p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(file_sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP]
        for name in filenames:
            if name in _SKIP or name.endswith(".pyc"):
                continue
            yield Path(dirpath) / name


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple):
        return [str(v) for v in value]
    return []


@dataclass
class Bundle:
    """One gap bundle, wherever it came from."""

    name: str
    kind: str
    path: Path
    source: str
    """``hub`` for a bundle in this repository, else the source's name."""
    frontmatter: dict[str, Any]
    body: str
    hub: dict[str, Any] = field(default_factory=dict)
    evals: list[dict[str, Any]] = field(default_factory=list)

    @property
    def native(self) -> bool:
        return self.source == "hub"

    @property
    def gap(self) -> dict[str, Any]:
        value = self.frontmatter.get("gap")
        return value if isinstance(value, dict) else {}

    @property
    def metadata(self) -> dict[str, Any]:
        value = self.frontmatter.get("metadata")
        return value if isinstance(value, dict) else {}

    @property
    def description(self) -> str:
        return " ".join(str(self.frontmatter.get("description") or "").split())

    @property
    def category(self) -> str:
        return str(self.metadata.get("category") or "")

    @property
    def tags(self) -> list[str]:
        return _as_list(self.metadata.get("tags"))

    @property
    def exits(self) -> list[str]:
        value = self.gap.get("exit_conditions")
        return [str(k) for k in value] if isinstance(value, dict) else []

    @property
    def allowed_tools(self) -> list[str]:
        return _as_list(self.gap.get("allowed_tools"))

    @property
    def declared_tools(self) -> list[str]:
        """The ``<bundle>.<func>`` names a tool or policy bundle exports."""
        names = []
        for entry in self.gap.get("tools") or []:
            if isinstance(entry, dict):
                names.extend(str(k) for k in entry)
            elif isinstance(entry, str):
                names.append(entry.split(":", 1)[0].strip())
        return names

    @property
    def requires(self) -> dict[str, Any]:
        value = self.gap.get("requires")
        return value if isinstance(value, dict) else {}

    @property
    def serving(self) -> dict[str, Any]:
        value = self.gap.get("serving")
        return value if isinstance(value, dict) else {}

    @property
    def required_inputs(self) -> dict[str, str]:
        value = self.gap.get("required_inputs")
        return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}

    @property
    def produces_outputs(self) -> dict[str, str]:
        value = self.gap.get("produces_outputs")
        return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}

    @property
    def privileged_tools(self) -> list[str]:
        """Every ``sim.*`` tool the bundle may call or says it needs."""
        named = self.allowed_tools + _as_list(self.requires.get("connector"))
        return sorted({t for t in named if t.startswith("sim.")})

    @property
    def tier(self) -> str:
        return "full" if self.privileged_tools else "camera_only"

    @property
    def stage(self) -> str:
        return str(self.hub.get("stage") or "")

    @property
    def embodiments(self) -> list[str]:
        return _as_list(self.hub.get("embodiments"))

    @property
    def version(self) -> str:
        return str(self.hub.get("version") or "")

    @property
    def weights(self) -> dict[str, Any]:
        declared = self.hub.get("weights")
        if isinstance(declared, dict) and declared.get("uri"):
            return dict(declared)
        uri = self.serving.get("weights_uri")
        return {"uri": str(uri)} if uri else {}

    @property
    def maturity(self) -> str:
        best = "community"
        for record in self.evals:
            if not record.get("passed"):
                continue
            if record.get("setting") == "real":
                return "real-verified"
            if record.get("setting") == "sim":
                best = "sim-verified"
        return best

    def supports(self, embodiment: str) -> bool:
        return embodiment in self.embodiments or "any" in self.embodiments

    def content_sha256(self) -> str:
        return tree_sha256(self.path)


@dataclass
class Recipe:
    """A task-level template: which skills, and how their exits route."""

    name: str
    path: Path
    frontmatter: dict[str, Any]
    body: str

    @property
    def description(self) -> str:
        return " ".join(str(self.frontmatter.get("description") or "").split())

    @property
    def hub(self) -> dict[str, Any]:
        meta = self.frontmatter.get("metadata")
        hub = meta.get("hub") if isinstance(meta, dict) else None
        return hub if isinstance(hub, dict) else {}

    @property
    def topology(self) -> dict[str, Any]:
        value = self.frontmatter.get("recipe")
        return value if isinstance(value, dict) else {}

    @property
    def skill_nodes(self) -> dict[str, dict[str, Any]]:
        nodes = self.topology.get("nodes")
        if not isinstance(nodes, dict):
            return {}
        return {str(k): v for k, v in nodes.items() if isinstance(v, dict) and "skill" in v}

    @property
    def end_nodes(self) -> dict[str, dict[str, Any]]:
        nodes = self.topology.get("nodes")
        if not isinstance(nodes, dict):
            return {}
        return {str(k): v for k, v in nodes.items() if isinstance(v, dict) and "end" in v}


@dataclass
class Paper:
    """A paper card: what a method is, and what in the hub came from it."""

    id: str
    path: Path
    frontmatter: dict[str, Any]
    body: str

    @property
    def derived(self) -> list[str]:
        return _as_list(self.frontmatter.get("derived"))
