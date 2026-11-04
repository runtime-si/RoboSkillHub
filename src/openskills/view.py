"""A registry directory holding only the selected bundles, as symlinks.

Hand it to gap as the coordinator's registry (``GAP_SKILLS_PATH=<view>`` or
``skills=<view>``) when generating a graph. Run the graph against the synced
sources themselves, not the view: gap starts an RPC tool bundle with
``uv run --project <bundle dir>`` without resolving symlinks, and the bundle's
relative path to the engine only holds from where the bundle really lives.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

from .model import KIND_DIRS, Bundle, Recipe
from .search import Pick

VIEW_FILE = "VIEW.json"
_FOLDER = {kind: folder for folder, kind in KIND_DIRS.items()}


def recipe_prompt(recipe: Recipe) -> str:
    """The recipe as text a coordinator can follow: subgraphs, skills, routes, ends."""
    topology = recipe.topology
    routes = topology.get("routes") or {}
    lines = [f"## Recipe: {recipe.name}", "", recipe.description, ""]
    inputs = topology.get("inputs") or {}
    if inputs:
        lines.append("Graph inputs: " + ", ".join(f"{k} ({v})" for k, v in inputs.items()))
    lines.append(f"Start at: {topology.get('start')}")
    lines.append("Subgraphs (node: skill; exit -> next):")
    for node, spec in recipe.skill_nodes.items():
        bind = spec.get("bind") or {}
        bound = f" [{', '.join(f'{k}={v}' for k, v in bind.items())}]" if bind else ""
        route = ", ".join(f"{exit_} -> {target}" for exit_, target in (routes.get(node) or {}).items())
        lines.append(f"- {node}: {spec['skill']}{bound}; {route}")
    for node, spec in recipe.end_nodes.items():
        recovery = spec.get("recovery") or []
        tail = f" (recovery: {', '.join(recovery)})" if recovery else ""
        lines.append(f"- {node}: end {spec['end']}{tail}")
    return "\n".join(lines) + "\n"


def make_view(
    picks: Iterable[Pick | Bundle], out: Path, *, query: str = "", recipes: Iterable[Recipe] = ()
) -> Path:
    out = Path(out).resolve()
    if out.exists():
        if not (out / VIEW_FILE).is_file():
            raise FileExistsError(f"{out} exists and is not a openskills view; refusing to replace it")
        shutil.rmtree(out)
    for folder in KIND_DIRS:
        (out / folder).mkdir(parents=True)
    listed = []
    for item in picks:
        bundle, reason = (item.bundle, item.reason) if isinstance(item, Pick) else (item, "named")
        target = bundle.path.resolve()
        (out / _FOLDER[bundle.kind] / bundle.name).symlink_to(target, target_is_directory=True)
        listed.append(
            {
                "name": bundle.name,
                "kind": bundle.kind,
                "source": bundle.source,
                "path": str(target),
                "reason": reason,
            }
        )
    recipes = list(recipes)
    record = {"query": query, "bundles": listed, "recipes": [r.name for r in recipes]}
    (out / VIEW_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if recipes:
        (out / "RECIPES.md").write_text("\n".join(recipe_prompt(r) for r in recipes), encoding="utf-8")
    return out
