"""``openskills check``: is everything in the catalog something the hub can serve?

Two layers. The engine's own SKILL.md parser and format validator run first when
gap is importable, so a bundle that passes here is a bundle gap loads. The hub's
rules run on top: its metadata is complete, a declared tier matches what the
bundle actually calls, recipes route every exit their skills can take, paper
cards point at things that exist, eval records are well-formed.

A rule broken by a bundle this repository owns is an error. The same rule broken
by an upstream bundle is a warning -- we cannot fix it here -- except where the
engine itself rejects the bundle, or where the broken part is ours (its overlay).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .catalog import Catalog, Issue
from .model import (
    CONNECTOR_PREFIXES,
    HUB_KEYS,
    IO_ACTION_KEYS,
    IO_KEYS,
    IO_STATE_KEYS,
    KIND_DIRS,
    NAME_RE,
    PROVENANCE_KEYS,
    SEMVER_RE,
    SHA256_RE,
    STAGES,
    TIERS,
    WEIGHTS_KEYS,
    Bundle,
    Paper,
    Recipe,
)
from .weights import parse_uri

GAP_KEYS = frozenset(
    {
        "allowed_tools",
        "exit_conditions",
        "produces_outputs",
        "required_inputs",
        "canonical_scripts",
        "prompts",
        "references",
        "examples",
        "errors",
        "tips",
        "hard_rules",
        "streaming",
        "tools",
        "requires",
        "serving",
    }
)
HEAVY_IMPORTS = frozenset(
    {
        "torch",
        "torchvision",
        "transformers",
        "jax",
        "jaxlib",
        "tensorflow",
        "open3d",
        "curobo",
        "vllm",
        "diffusers",
    }
)
"""Modules a tool bundle must import lazily: importing ``tools.py`` has to stay cheap."""

_USE_WHEN = re.compile(r"\buse\s+(?:it\s+|this\s+)?(?:when|for|after|before|to|on|with|if)\b", re.I)
_LIST_TYPE = re.compile(r"^list\[(.+)\]$")
_FOLDER = {kind: folder for folder, kind in KIND_DIRS.items()}


@dataclass
class GapApi:
    parse_skill_md: Callable[..., Any] | None = None
    validate_bundle_meta: Callable[..., Any] | None = None
    type_registry: dict[str, Any] | None = None
    error: str = ""

    @property
    def available(self) -> bool:
        return self.parse_skill_md is not None


def load_gap_api() -> GapApi:
    api = GapApi()
    try:
        from gap.skills import parse_skill_md
        from gap.skills.validate import validate_bundle_meta

        api.parse_skill_md = parse_skill_md
        api.validate_bundle_meta = validate_bundle_meta
    except Exception as exc:  # noqa: BLE001 - any import failure means "engine absent"
        api.error = f"{type(exc).__name__}: {exc}"
    try:
        from gap_core.schema import TYPE_REGISTRY

        api.type_registry = dict(TYPE_REGISTRY)
    except Exception as exc:  # noqa: BLE001
        api.error = api.error or f"{type(exc).__name__}: {exc}"
    return api


def _type_ok(name: str, registry: dict[str, Any]) -> bool:
    name = name.strip()
    if name in registry or name in {"Any", "dict", "list"}:
        return True
    match = _LIST_TYPE.match(name)
    return bool(match) and _type_ok(match.group(1), registry)


def _str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) and v for v in value)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _io_problems(io: Any) -> list[str]:
    if not isinstance(io, dict):
        return ["io must be a mapping of action, cameras and optionally state"]
    problems = []
    if set(io) - IO_KEYS:
        problems.append(f"unknown io keys {sorted(set(io) - IO_KEYS)} (known: {sorted(IO_KEYS)})")
    action = io.get("action") if isinstance(io.get("action"), dict) else {}
    if set(action) - IO_ACTION_KEYS:
        problems.append(f"unknown io.action keys {sorted(set(action) - IO_ACTION_KEYS)}")
    hz = action.get("hz")
    if not (
        isinstance(action.get("space"), str)
        and action["space"]
        and _positive_int(action.get("dim"))
        and isinstance(hz, int | float)
        and not isinstance(hz, bool)
        and hz > 0
    ):
        problems.append("io.action needs space (a name), dim (a positive integer) and hz (a positive number)")
    state = io.get("state")
    if state is not None:
        if not isinstance(state, dict) or not _positive_int(state.get("dim")):
            problems.append("io.state needs dim, a positive integer")
        elif set(state) - IO_STATE_KEYS:
            problems.append(f"unknown io.state keys {sorted(set(state) - IO_STATE_KEYS)}")
        elif "order" in state and not _str_list(state["order"]):
            problems.append("io.state.order must be a list of strings")
    cameras = io.get("cameras")
    if not (_str_list(cameras) and cameras):
        problems.append("io.cameras must list the cameras the checkpoint reads, by name")
    return problems


def _run_signature_ok(script: str) -> bool:
    """A canonical script defines ``run(ctx, ...)``."""
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            args = node.args.posonlyargs + node.args.args
            return bool(args) and args[0].arg == "ctx"
    return False


def _heavy_top_level_imports(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return sorted(found & HEAVY_IMPORTS)


def check_hub_metadata(
    hub: dict[str, Any], *, where: str, kind: str, native: bool, derived_tier: str | None
) -> list[Issue]:
    out: list[Issue] = []

    def error(msg: str) -> None:
        out.append(Issue("error", where, msg))

    unknown = sorted(set(hub) - HUB_KEYS)
    if unknown:
        error(f"unknown hub metadata keys {unknown} (known: {sorted(HUB_KEYS)})")
    stage = hub.get("stage")
    if stage is None:
        if kind in ("skill", "policy"):
            error(f"hub metadata needs a stage, one of {list(STAGES)}")
    elif stage not in STAGES:
        error(f"stage {stage!r} is not one of {list(STAGES)}")
    embodiments = hub.get("embodiments")
    if not _str_list(embodiments):
        error("hub metadata needs embodiments: a list of robot ids (rss.robot() names) or [any]")
    elif kind == "policy" and "any" in embodiments:
        error("a checkpoint is trained on particular robots: embodiments names them, never [any]")
    io = hub.get("io")
    if kind == "policy" and io is None:
        error("a policy declares io: the action space and rate, state and cameras it was trained with")
    elif io is not None and kind != "policy":
        error("io describes a checkpoint's observations and actions; only a policy declares it")
    elif io is not None:
        for problem in _io_problems(io):
            error(problem)
    for key in ("tuned_on", "tasks"):
        if key in hub and not _str_list(hub[key]):
            error(f"{key} must be a list of strings")
    if "arms" in hub and not (isinstance(hub["arms"], int) and hub["arms"] >= 1):
        error("arms must be a positive integer")
    version = hub.get("version")
    if version is None:
        if native:
            error("a bundle in this repository needs a semver version")
    elif not SEMVER_RE.match(str(version)):
        error(f"version {version!r} is not semver (MAJOR.MINOR.PATCH)")
    provenance = hub.get("provenance")
    if provenance is not None:
        if not isinstance(provenance, dict):
            error("provenance must be a mapping")
        elif set(provenance) - PROVENANCE_KEYS:
            error(f"unknown provenance keys {sorted(set(provenance) - PROVENANCE_KEYS)}")
    weights = hub.get("weights")
    if weights is not None:
        if not isinstance(weights, dict) or set(weights) - WEIGHTS_KEYS or not weights.get("uri"):
            error(f"weights must be a mapping with a uri and only {sorted(WEIGHTS_KEYS)}")
        else:
            uri = str(weights["uri"])
            if uri.startswith("hf://"):
                try:
                    parse_uri(uri, require_revision=True)
                except ValueError as exc:
                    error(str(exc))
                if not SHA256_RE.match(str(weights.get("sha256") or "")):
                    error(
                        "hf weights need the sha256 of their file manifest "
                        "(openskills weights push prints it)"
                    )
    tier = hub.get("tier")
    if tier is not None:
        if tier not in TIERS:
            error(f"tier {tier!r} is not one of {list(TIERS)}")
        elif derived_tier is not None and tier != derived_tier:
            error(f"declares tier {tier!r} but what it calls makes it {derived_tier!r}")
    return out


def check_bundle(bundle: Bundle, catalog: Catalog, api: GapApi) -> list[Issue]:
    where = f"{bundle.source}:{_FOLDER[bundle.kind]}/{bundle.name}"
    soft = "error" if bundle.native else "warning"
    out: list[Issue] = []

    def add(level: str, msg: str) -> None:
        out.append(Issue(level, where, msg))

    if api.available:
        meta = None
        try:
            meta = api.parse_skill_md(bundle.path / "SKILL.md")
        except (ValueError, OSError) as exc:
            add("error", f"gap rejects this SKILL.md: {exc}")
        if meta is not None and api.validate_bundle_meta is not None:
            for issue in api.validate_bundle_meta(
                meta, kind=bundle.kind, bundle_dir=bundle.path, known_tools=None, extras=None
            ):
                level = getattr(issue, "severity", "error")
                add("error" if level == "error" else soft, f"gap: {issue.message}")
    else:
        if not NAME_RE.match(bundle.name) or len(bundle.name) > 64:
            add(
                "error",
                "name must be lowercase letters and digits joined by single hyphens, at most 64 chars",
            )
        if str(bundle.frontmatter.get("name")) != bundle.name:
            add("error", f"frontmatter name {bundle.frontmatter.get('name')!r} differs from its directory")
        if not bundle.description or len(bundle.description) > 1024:
            add("error", "description must be present and at most 1024 characters")

    if bundle.native and not _USE_WHEN.search(bundle.description):
        add("error", 'description needs a "Use when ..." sentence: it is all the coordinator sees')
    if bundle.native and not bundle.frontmatter.get("license"):
        add("error", "a bundle in this repository declares its license")
    gap_block = bundle.frontmatter.get("gap")
    if not isinstance(gap_block, dict):
        add(soft, "has no gap: block")
    elif set(gap_block) - GAP_KEYS:
        add(soft, f"unknown gap keys {sorted(set(gap_block) - GAP_KEYS)}")

    out.extend(
        check_hub_metadata(
            bundle.hub, where=where, kind=bundle.kind, native=bundle.native, derived_tier=bundle.tier
        )
    )

    # A streaming skill runs beside the graph (a tracker under a policy), never routed on.
    if bundle.kind in ("skill", "policy") and not bundle.exits and not bundle.gap.get("streaming"):
        add("error", "declares no exit_conditions, so no graph can route on it")
    if bundle.kind == "skill":
        for entry in bundle.gap.get("canonical_scripts") or []:
            if not isinstance(entry, dict):
                continue
            for logical, rel in entry.items():
                path = bundle.path / str(rel)
                if not path.is_file():
                    add("error", f"canonical script {logical!r} points at missing {rel}")
                elif not _run_signature_ok(path.read_text(encoding="utf-8")):
                    add(soft, f"canonical script {rel} does not define run(ctx, ...)")
    if bundle.kind == "tool":
        tools_py = bundle.path / "tools.py"
        if not tools_py.is_file():
            add(soft, "a tool bundle ships its @tool functions in tools.py")
        else:
            heavy = _heavy_top_level_imports(tools_py.read_text(encoding="utf-8"))
            if heavy:
                add(soft, f"tools.py imports {heavy} at module level; load models on first call")
        if "requires" not in bundle.gap:
            add(soft, "a tool bundle declares gap.requires ({} when it needs nothing)")
        for name in bundle.declared_tools:
            if not name.startswith(f"{bundle.name}."):
                add("error", f"tool {name!r} is not namespaced as {bundle.name}.<func>")
    if bundle.kind == "policy":
        serving = bundle.serving
        if not isinstance(serving.get("command"), list) or not serving.get("protocol"):
            add("error", "a policy bundle declares gap.serving with a command list and a protocol")

    for tool_name in bundle.allowed_tools:
        prefix = tool_name.split(".", 1)[0]
        if prefix in CONNECTOR_PREFIXES:
            continue
        owner = catalog.bundles.get(prefix)
        if owner is None:
            add(soft, f"allowed_tools names {tool_name!r} but no bundle or connector provides it")
        elif owner.declared_tools and tool_name not in owner.declared_tools:
            add("warning", f"{tool_name!r} is not among the tools {prefix!r} declares")

    if api.type_registry is not None:
        for key, type_name in {**bundle.required_inputs, **bundle.produces_outputs}.items():
            if not _type_ok(type_name, api.type_registry):
                add(soft, f"{key}: type {type_name!r} is not in gap's type registry")
    return out


def check_overlays(catalog: Catalog) -> list[Issue]:
    out: list[Issue] = []
    for pin in catalog.manifest.sources:
        if pin.name in catalog.missing_sources:
            continue
        overlay = catalog.overlays.get(pin.name, {})
        upstream = {b.name for b in catalog.bundles.values() if b.source == pin.name}
        upstream |= {name for name, _, loser in catalog.shadowed if loser == pin.name}
        for name in sorted(set(overlay) - upstream):
            out.append(Issue("error", pin.overlay, f"overlay names {name!r}, which {pin.name} does not ship"))
        for name in sorted(upstream - set(overlay)):
            if catalog.bundles.get(name) and catalog.bundles[name].source == pin.name:
                out.append(
                    Issue("error", pin.overlay, f"{pin.name} ships {name!r}; add its hub metadata here")
                )
    return out


def _reachable(start: str, routes: dict[str, dict[str, str]]) -> set[str]:
    seen, stack = set(), [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend((routes.get(node) or {}).values())
    return seen


def check_recipe(recipe: Recipe, catalog: Catalog) -> list[Issue]:
    where = f"recipes/{recipe.name}"
    out: list[Issue] = []

    def error(msg: str) -> None:
        out.append(Issue("error", where, msg))

    fm = recipe.frontmatter
    if str(fm.get("name")) != recipe.name or not NAME_RE.match(recipe.name):
        error("name must equal the directory name, lowercase and hyphenated")
    if not _USE_WHEN.search(recipe.description):
        error('description needs a "Use when ..." sentence')
    if not fm.get("license"):
        error("declares its license")
    topology = recipe.topology
    if not topology:
        error("has no recipe: block")
        return out
    unknown = set(topology) - {"inputs", "start", "nodes", "routes"}
    if unknown:
        error(f"unknown recipe keys {sorted(unknown)}")
    nodes = topology.get("nodes") if isinstance(topology.get("nodes"), dict) else {}
    skills, ends = recipe.skill_nodes, recipe.end_nodes
    for name in sorted(set(nodes) - set(skills) - set(ends)):
        error(f"node {name!r} is neither {{skill: ...}} nor {{end: success|failure}}")
    inputs = topology.get("inputs") or {}
    routes = topology.get("routes") if isinstance(topology.get("routes"), dict) else {}
    start = topology.get("start")
    if start not in skills:
        error(f"start {start!r} is not a skill node")

    privileged: set[str] = set()
    available = set(inputs)
    resolved: dict[str, Bundle] = {}
    for node, spec in skills.items():
        if set(spec) - {"skill", "bind", "note"}:
            error(f"node {node!r}: unknown keys {sorted(set(spec) - {'skill', 'bind', 'note'})}")
        bundle = catalog.bundles.get(str(spec["skill"]))
        if bundle is None or bundle.kind not in ("skill", "policy"):
            error(f"node {node!r}: no skill or policy named {spec['skill']!r} in the catalog")
            continue
        resolved[node] = bundle
        privileged.update(bundle.privileged_tools)
        bind = spec.get("bind") or {}
        for key in bundle.produces_outputs:
            available.add(key.replace("<name>", str(bind.get("name", "<name>"))))
        route = routes.get(node)
        if not isinstance(route, dict):
            error(f"node {node!r} has no routes; every exit of {bundle.name} must lead somewhere")
            continue
        missing = sorted(set(bundle.exits) - set(route))
        extra = sorted(set(route) - set(bundle.exits))
        if missing:
            error(f"node {node!r}: exits {missing} of {bundle.name} are not routed")
        if extra:
            error(f"node {node!r}: {extra} are not exits of {bundle.name} (it has {bundle.exits})")
        for target in route.values():
            if target not in nodes:
                error(f"node {node!r} routes to {target!r}, which is not a node")
    for node in sorted(set(routes) - set(skills)):
        error(f"routes names {node!r}, which is not a skill node")
    for node, spec in ends.items():
        if spec.get("end") not in ("success", "failure"):
            error(f"end node {node!r}: end must be success or failure")
        if "recovery" in spec and not _str_list(spec["recovery"]):
            error(f"end node {node!r}: recovery is a list of tool names")
    if start in skills:
        reach = _reachable(str(start), routes)
        for node in sorted(set(nodes) - reach):
            error(f"node {node!r} is unreachable from {start!r}")
        if not any(ends.get(n, {}).get("end") == "success" for n in reach):
            error("no success end is reachable")

    for node, bundle in resolved.items():
        for key in bundle.required_inputs:
            if key not in available:
                out.append(
                    Issue("warning", where, f"node {node!r}: input {key!r} of {bundle.name} has no producer")
                )

    hub = recipe.hub
    out.extend(
        check_hub_metadata(
            {k: v for k, v in hub.items() if k != "stage"},
            where=where,
            kind="recipe",
            native=True,
            derived_tier="full" if privileged else "camera_only",
        )
    )
    return out


PAPER_KEYS = frozenset(
    {"id", "title", "authors", "year", "venue", "arxiv", "url", "code", "license", "tasks", "derived"}
)


def check_paper(paper: Paper, catalog: Catalog) -> list[Issue]:
    where = f"papers/{paper.id}"
    out: list[Issue] = []

    def error(msg: str) -> None:
        out.append(Issue("error", where, msg))

    fm = paper.frontmatter
    if str(fm.get("id")) != paper.id or not NAME_RE.match(paper.id):
        error("id must equal the directory name, lowercase and hyphenated")
    if set(fm) - PAPER_KEYS:
        error(f"unknown keys {sorted(set(fm) - PAPER_KEYS)}")
    if not fm.get("title"):
        error("needs a title")
    if not isinstance(fm.get("year"), int):
        error("needs a year")
    if not (fm.get("url") or fm.get("arxiv")):
        error("needs a url or an arxiv id")
    for ref in paper.derived:
        folder, _, name = ref.partition("/")
        if folder == "recipes":
            if name not in catalog.recipes:
                error(f"derived names {ref!r}, which is not a recipe here")
        elif folder in KIND_DIRS:
            bundle = catalog.bundles.get(name)
            if bundle is None or bundle.kind != KIND_DIRS[folder]:
                error(f"derived names {ref!r}, which is not a {KIND_DIRS[folder]} in the catalog")
        else:
            error(f"derived entries are tools/, skills/, policies/ or recipes/ paths, not {ref!r}")
    return out


EVAL_KEYS = {
    "bundle": str,
    "version": str,
    "embodiment": str,
    "setting": str,
    "tier": str,
    "suite": str,
    "episodes": int,
    "successes": int,
    "passed": bool,
    "date": str,
    "evidence": str,
}


def check_evals(catalog: Catalog) -> list[Issue]:
    out: list[Issue] = []
    for bundle in catalog.bundles.values():
        for record in bundle.evals:
            where = record.get("_file", f"evals/{bundle.name}")
            for key, kind in EVAL_KEYS.items():
                if not isinstance(record.get(key), kind):
                    out.append(Issue("error", where, f"{key} must be a {kind.__name__}"))
            if record.get("bundle") != bundle.name:
                out.append(Issue("error", where, "bundle must match the directory it is filed under"))
            if record.get("setting") not in ("sim", "real"):
                out.append(Issue("error", where, "setting is sim or real"))
            episodes, successes = record.get("episodes"), record.get("successes")
            if isinstance(episodes, int) and isinstance(successes, int) and not 0 <= successes <= episodes:
                out.append(Issue("error", where, "successes must lie between 0 and episodes"))
    return out


def check_all(catalog: Catalog, api: GapApi | None = None, *, require_gap: bool = False) -> list[Issue]:
    api = api or load_gap_api()
    out = list(catalog.issues)
    if not api.available:
        level = "error" if require_gap else "warning"
        out.append(Issue(level, "engine", f"gap is not importable, engine validation skipped ({api.error})"))
    for name in catalog.missing_sources:
        out.append(Issue("error", name, "source is not synced; run `openskills sync`"))
    for bundle in sorted(catalog.bundles.values(), key=lambda b: (b.source != "hub", b.name)):
        out.extend(check_bundle(bundle, catalog, api))
    out.extend(check_overlays(catalog))
    for recipe in catalog.recipes.values():
        out.extend(check_recipe(recipe, catalog))
    for paper in catalog.papers.values():
        out.extend(check_paper(paper, catalog))
    out.extend(check_evals(catalog))
    return out
