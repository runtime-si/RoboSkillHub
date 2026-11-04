"""The merged view of the hub: native bundles, federated sources, recipes, papers, evals."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import frontmatter
from .model import KIND_DIRS, Bundle, Paper, Recipe
from .sources import Manifest, cache_dir, find_root, load_manifest, source_dir


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():7} {self.where}: {self.message}"


@dataclass
class Catalog:
    root: Path
    manifest: Manifest
    cache: Path
    bundles: dict[str, Bundle] = field(default_factory=dict)
    shadowed: list[tuple[str, str, str]] = field(default_factory=list)
    """``(name, winning source, losing source)``."""
    recipes: dict[str, Recipe] = field(default_factory=dict)
    papers: dict[str, Paper] = field(default_factory=dict)
    overlays: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per source: the overlay mapping as written, for checking."""
    missing_sources: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    """Problems found while loading (unreadable frontmatter, duplicate names)."""

    def get(self, name: str) -> Bundle:
        return self.bundles[name]

    def of_kind(self, *kinds: str) -> list[Bundle]:
        return [b for b in self.bundles.values() if b.kind in kinds]

    def source_root(self, source: str) -> Path:
        return self.root if source == "hub" else source_dir(self.cache, source)


def _scan(registry: Path, source: str, overlay: dict[str, Any], issues: list[Issue]) -> list[Bundle]:
    found: list[Bundle] = []
    for folder, kind in KIND_DIRS.items():
        base = registry / folder
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if not entry.is_dir() or entry.name.startswith(("_", ".")):
                continue
            skill_md = entry / "SKILL.md"
            if not skill_md.is_file():
                continue
            where = f"{source}:{folder}/{entry.name}"
            try:
                data, body = frontmatter.read(skill_md)
            except frontmatter.FrontmatterError as exc:
                issues.append(Issue("error", where, str(exc)))
                continue
            meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
            hub = dict(meta.get("hub") or {}) if isinstance(meta.get("hub"), dict) else {}
            if entry.name in overlay:
                hub.update(overlay[entry.name] or {})
            found.append(Bundle(entry.name, kind, entry, source, data, body, hub))
    return found


def _load_overlay(root: Path, relpath: str, issues: list[Issue]) -> dict[str, Any]:
    if not relpath:
        return {}
    path = root / relpath
    if not path.is_file():
        issues.append(Issue("error", relpath, "overlay file named in sources.yaml does not exist"))
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or not all(isinstance(v, dict) for v in data.values()):
        issues.append(Issue("error", relpath, "an overlay maps bundle names to hub metadata mappings"))
        return {}
    return data


def _load_evals(root: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    base = root / "evals"
    if not base.is_dir():
        return out
    for record in sorted(base.glob("*/*.json")):
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            out.setdefault(record.parent.name, []).append(
                {**data, "_file": record.relative_to(root).as_posix()}
            )
    return out


def load_catalog(root: Path | None = None, *, cache: Path | None = None) -> Catalog:
    root = (root or find_root()).resolve()
    manifest = load_manifest(root)
    catalog = Catalog(root=root, manifest=manifest, cache=cache or cache_dir(root))

    registries: list[tuple[str, Path, dict[str, Any]]] = [("hub", root, {})]
    for pin in manifest.sources:
        overlay = _load_overlay(root, pin.overlay, catalog.issues)
        catalog.overlays[pin.name] = overlay
        path = source_dir(catalog.cache, pin.name)
        if not path.is_dir():
            catalog.missing_sources.append(pin.name)
            continue
        registries.append((pin.name, path, overlay))

    for source, path, overlay in registries:
        seen: set[str] = set()
        for bundle in _scan(path, source, overlay, catalog.issues):
            if bundle.name in seen:
                catalog.issues.append(
                    Issue("error", f"{source}:{bundle.name}", "two bundles share this name in one registry")
                )
                continue
            seen.add(bundle.name)
            if bundle.name in catalog.bundles:
                catalog.shadowed.append((bundle.name, catalog.bundles[bundle.name].source, source))
                continue
            catalog.bundles[bundle.name] = bundle

    for name, records in _load_evals(root).items():
        if name in catalog.bundles:
            catalog.bundles[name].evals = records

    for path in sorted((root / "recipes").glob("*/RECIPE.md")):
        try:
            data, body = frontmatter.read(path)
        except frontmatter.FrontmatterError as exc:
            catalog.issues.append(Issue("error", f"recipes/{path.parent.name}", str(exc)))
            continue
        catalog.recipes[path.parent.name] = Recipe(path.parent.name, path.parent, data, body)

    for path in sorted((root / "papers").glob("*/PAPER.md")):
        try:
            data, body = frontmatter.read(path)
        except frontmatter.FrontmatterError as exc:
            catalog.issues.append(Issue("error", f"papers/{path.parent.name}", str(exc)))
            continue
        catalog.papers[path.parent.name] = Paper(path.parent.name, path.parent, data, body)

    return catalog
