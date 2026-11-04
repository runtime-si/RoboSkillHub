"""``index.json``: the whole catalog as one machine-readable file.

What a website, a search API, or a pipeline that does not want to parse YAML
reads. Built from the catalog, never edited by hand, and deterministic: the same
commits give byte-identical output.
"""

from __future__ import annotations

from typing import Any

from .catalog import Catalog
from .model import CONNECTOR_PREFIXES, KIND_DIRS, Bundle
from .sources import head_commit

INDEX_VERSION = 1
_FOLDER = {kind: folder for folder, kind in KIND_DIRS.items()}


def bundle_record(bundle: Bundle, catalog: Catalog) -> dict[str, Any]:
    hub = bundle.hub
    tool_bundles = sorted(
        {t.split(".", 1)[0] for t in bundle.allowed_tools if t.split(".", 1)[0] in catalog.bundles}
    )
    connector_tools = sorted(t for t in bundle.allowed_tools if t.split(".", 1)[0] in CONNECTOR_PREFIXES)
    commit = head_commit(catalog.root) if bundle.native else catalog.manifest.source(bundle.source).commit
    record: dict[str, Any] = {
        "name": bundle.name,
        "kind": bundle.kind,
        "source": bundle.source,
        "commit": commit,
        "path": f"{_FOLDER[bundle.kind]}/{bundle.name}",
        "description": bundle.description,
        "license": str(bundle.frontmatter.get("license") or ""),
        "category": bundle.category,
        "tags": bundle.tags,
        "stage": bundle.stage,
        "embodiments": bundle.embodiments,
        "tuned_on": list(hub.get("tuned_on") or []),
        "arms": int(hub.get("arms") or 1),
        "tasks": list(hub.get("tasks") or []),
        "tier": bundle.tier,
        "privileged_tools": bundle.privileged_tools,
        "maturity": bundle.maturity,
        "exits": bundle.exits,
        "inputs": bundle.required_inputs,
        "outputs": bundle.produces_outputs,
        "tools": bundle.declared_tools,
        "tool_bundles": tool_bundles,
        "connector_tools": connector_tools,
        "content_sha256": bundle.content_sha256(),
    }
    for key in ("version", "provenance", "notes"):
        if hub.get(key):
            record[key] = hub[key]
    if bundle.weights:
        record["weights"] = bundle.weights
    if bundle.serving:
        record["serving"] = {
            k: bundle.serving[k] for k in ("protocol", "requires_gpu") if k in bundle.serving
        }
    if bundle.evals:
        record["evals"] = [{k: v for k, v in e.items() if not k.startswith("_")} for e in bundle.evals]
    return record


def build_index(catalog: Catalog) -> dict[str, Any]:
    bundles = sorted(catalog.bundles.values(), key=lambda b: (b.kind, b.name))
    return {
        "index_version": INDEX_VERSION,
        "hub": {"commit": head_commit(catalog.root)},
        "engine": {"repo": catalog.manifest.engine.repo, "commit": catalog.manifest.engine.commit},
        "sources": [
            {"name": p.name, "repo": p.repo, "commit": p.commit, "license": p.license}
            for p in catalog.manifest.sources
        ],
        "counts": {kind: sum(1 for b in bundles if b.kind == kind) for kind in ("tool", "skill", "policy")}
        | {"recipe": len(catalog.recipes), "paper": len(catalog.papers)},
        "bundles": [bundle_record(b, catalog) for b in bundles],
        "shadowed": [{"name": n, "winner": w, "loser": lo} for n, w, lo in catalog.shadowed],
        "recipes": [
            {
                "name": r.name,
                "description": r.description,
                "skills": sorted({str(s["skill"]) for s in r.skill_nodes.values()}),
                **{k: r.hub[k] for k in ("version", "embodiments", "tuned_on", "tasks") if k in r.hub},
            }
            for r in sorted(catalog.recipes.values(), key=lambda r: r.name)
        ],
        "papers": [
            {
                "id": p.id,
                **{
                    k: p.frontmatter[k]
                    for k in ("title", "year", "arxiv", "url", "code")
                    if k in p.frontmatter
                },
                "derived": p.derived,
            }
            for p in sorted(catalog.papers.values(), key=lambda p: p.id)
        ],
    }
