"""``skills.lock``: exactly which bundles, at which commits, a generated graph runs with.

A ``workflow.json`` names skills and tools, not versions. The lock adds the rest:
the engine commit, the hub and source commits, a content digest per bundle, the
pinned weights, and the connector tools the robot must provide. ``openskills fetch``
turns a lock back into the same registries on another machine -- the robot --
and refuses if any bundle's content is not what the lock recorded.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .catalog import Catalog
from .model import CONNECTOR_PREFIXES, KIND_DIRS, file_sha256, tree_sha256
from .sources import checkout, head_commit, is_dirty

LOCK_VERSION = 1
_FOLDER = {kind: folder for folder, kind in KIND_DIRS.items()}


class LockError(RuntimeError):
    pass


def graph_references(workflow: dict[str, Any]) -> tuple[set[str], set[str]]:
    """``(skills, tools)`` a workflow names: subgraph skills, tool nodes, recoveries."""
    skills: set[str] = set()
    tools: set[str] = set(workflow.get("requires", {}).get("tools", []) or [])

    def visit(nodes: Any) -> None:
        for node in (nodes or {}).values() if isinstance(nodes, dict) else []:
            if not isinstance(node, dict):
                continue
            if node.get("type") == "tool" and node.get("tool"):
                tools.add(str(node["tool"]))
            for op in node.get("recovery") or []:
                if isinstance(op, dict) and op.get("tool"):
                    tools.add(str(op["tool"]))

    visit(workflow.get("nodes"))
    for sub in (workflow.get("subgraphs") or {}).values():
        if not isinstance(sub, dict):
            continue
        skill = sub.get("skill") or sub.get("agent")
        if skill:
            skills.add(str(skill))
        visit(sub.get("nodes"))
    return skills, tools


def graph_sha256(graph_dir: Path) -> str:
    """Digest of what defines the graph: ``workflow.json``, ``scripts/`` and ``checkpoints/``."""
    files = [graph_dir / "workflow.json"]
    for folder in ("scripts", "checkpoints"):
        if (graph_dir / folder).is_dir():
            files += sorted(
                p for p in (graph_dir / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts
            )
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.relative_to(graph_dir).as_posix()}\0{file_sha256(path)}\n".encode())
    return digest.hexdigest()


def _remote_url(root: Path) -> str:
    result = subprocess.run(["git", "remote", "get-url", "origin"], cwd=root, capture_output=True, text=True)
    url = result.stdout.strip()
    if "@" in url and url.startswith("https://"):
        url = "https://" + url.split("@", 1)[1]
    return url.removesuffix(".git")


def build_lock(graph_dir: Path, catalog: Catalog) -> dict[str, Any]:
    graph_dir = Path(graph_dir)
    workflow = json.loads((graph_dir / "workflow.json").read_text(encoding="utf-8"))
    skills, tools = graph_references(workflow)

    used: set[str] = set()
    generated: list[str] = []
    for name in sorted(skills):
        if name in catalog.bundles:
            used.add(name)
        else:
            generated.append(name)
    connector: set[str] = set()
    unresolved: list[str] = []
    names = set(tools)
    for name in list(used):
        names.update(catalog.bundles[name].allowed_tools)
    for tool_name in sorted(names):
        prefix = tool_name.split(".", 1)[0]
        if prefix in CONNECTOR_PREFIXES:
            connector.add(tool_name)
        elif prefix in catalog.bundles:
            used.add(prefix)
        else:
            unresolved.append(tool_name)

    bundles: dict[str, Any] = {}
    for name in sorted(used):
        bundle = catalog.bundles[name]
        entry: dict[str, Any] = {
            "kind": bundle.kind,
            "source": bundle.source,
            "path": f"{_FOLDER[bundle.kind]}/{bundle.name}",
            "content_sha256": bundle.content_sha256(),
            "tier": bundle.tier,
        }
        if bundle.version:
            entry["version"] = bundle.version
        if bundle.weights:
            entry["weights"] = bundle.weights
        if bundle.serving.get("protocol"):
            entry["serving"] = bundle.serving["protocol"]
        bundles[name] = entry

    privileged = sorted({t for t in connector if t.startswith("sim.")})
    hub_commit = head_commit(catalog.root)
    lock: dict[str, Any] = {
        "lock_version": LOCK_VERSION,
        "graph": {
            "sha256": graph_sha256(graph_dir),
            "skills": sorted(skills & used),
            "generated_skills": generated,
        },
        "engine": {
            "name": catalog.manifest.engine.name,
            "repo": catalog.manifest.engine.repo,
            "commit": catalog.manifest.engine.commit,
        },
        "hub": {"repo": _remote_url(catalog.root), "commit": hub_commit, "dirty": is_dirty(catalog.root)},
        "sources": {
            pin.name: {"repo": pin.repo, "commit": pin.commit}
            for pin in catalog.manifest.sources
            if any(b["source"] == pin.name for b in bundles.values())
        },
        "bundles": bundles,
        "connector_tools": sorted(connector),
        "privileged_tools": privileged,
        "deployable": not privileged and all(b["tier"] == "camera_only" for b in bundles.values()),
    }
    if unresolved:
        lock["unresolved_tools"] = unresolved
    return lock


def write_lock(lock: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fetch(lock: dict[str, Any], dest: Path, *, weights: bool = True) -> dict[str, Any]:
    """Recreate a lock's registries under *dest*; verify every bundle; pull weights.

    Returns ``{"registries": [paths in precedence order], "weights": {bundle: path}}``.
    """
    if lock.get("lock_version") != LOCK_VERSION:
        raise LockError(f"lock_version {lock.get('lock_version')!r} is not {LOCK_VERSION}")
    dest = Path(dest).resolve()
    roots: dict[str, Path] = {}
    if any(b["source"] == "hub" for b in lock["bundles"].values()):
        hub = lock["hub"]
        if hub.get("dirty"):
            raise LockError("the lock was written from a hub checkout with uncommitted changes")
        roots["hub"] = checkout(hub["repo"], hub["commit"], dest / "sources" / "hub")
    for name, pin in lock["sources"].items():
        roots[name] = checkout(pin["repo"], pin["commit"], dest / "sources" / name)
    engine = lock["engine"]
    checkout(engine["repo"], engine["commit"], dest / "sources" / engine["name"])

    for name, entry in lock["bundles"].items():
        path = roots[entry["source"]] / entry["path"]
        got = tree_sha256(path) if path.is_dir() else "missing"
        if got != entry["content_sha256"]:
            raise LockError(f"{name}: content {got} at {path} is not the locked {entry['content_sha256']}")

    pulled: dict[str, str] = {}
    if weights:
        from .weights import pull

        for name, entry in lock["bundles"].items():
            uri = (entry.get("weights") or {}).get("uri", "")
            if uri.startswith("hf://"):
                pulled[name] = str(pull(uri, sha256=entry["weights"].get("sha256")))
    order = ["hub", *lock["sources"]]
    return {"registries": [str(roots[n]) for n in order if n in roots], "weights": pulled}
