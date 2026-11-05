"""``CATALOG.md``: every bundle on one page, for people reviewing the library.

An overview table per kind, then one section per bundle: its full description,
inputs and outputs, what each exit means, the tools it calls and the scripts it
runs. Generated from the catalog by ``roboskillhub catalog`` and never edited by
hand; deterministic, so a test can tell when it is stale.
"""

from __future__ import annotations

import re
from typing import Any

from .catalog import Catalog
from .model import CONNECTOR_PREFIXES, KIND_DIRS, STAGES, Bundle

PAGE = "CATALOG.md"
_FOLDER = {kind: folder for folder, kind in KIND_DIRS.items()}
_SECTIONS = (
    ("tool", "Tools", "Tool"),
    ("skill", "Skills", "Skill"),
    ("policy", "Policies", "Policy"),
)
_FIRST_SENTENCE = re.compile(r"(.+?[.!?])(?=\s+[A-Z(`]|$)")


def render(catalog: Catalog) -> str:
    bundles = sorted(catalog.bundles.values(), key=_order)
    counts = {kind: sum(1 for b in bundles if b.kind == kind) for kind, _, _ in _SECTIONS}
    lines = [
        "# Catalog",
        "",
        "Every bundle in this library, generated from the `SKILL.md` files by",
        "`roboskillhub catalog`; a test fails when this page is out of date. The",
        "overview has one line per bundle, and each name leads to its section: the",
        "full description a planner reads, what goes in and comes out, what each exit",
        "means, and what it calls.",
        "",
        f"{counts['tool']} tools, {counts['skill']} skills, {counts['policy']} policies, "
        f"{_plural(len(catalog.recipes), 'recipe')} and {_plural(len(catalog.papers), 'paper')}.",
        "",
        "Under robots, `any` means the bundle reads the robot through `describe_*`",
        "tools; *tuned on* names the robot its constants were measured on; **sim only**",
        "means it reads simulator ground truth (`sim.*`) and never runs on a real robot.",
        "",
    ]
    for kind, plural, singular in _SECTIONS:
        of_kind = [b for b in bundles if b.kind == kind]
        if of_kind:
            lines += [f"## {plural}", "", *_overview(singular, of_kind), ""]
    for kind, _, singular in _SECTIONS:
        of_kind = [b for b in bundles if b.kind == kind]
        if of_kind:
            lines += [f"## {singular} details", ""]
            for bundle in of_kind:
                lines += [*_section(bundle, catalog), ""]
    lines += _recipes_and_papers(catalog)
    return "\n".join(lines).rstrip() + "\n"


def summary(description: str, width: int = 120) -> str:
    """The first sentence of *description*: cut at its first clause, then at *width*, when long."""
    text = description.split(" Use when")[0].strip()
    match = _FIRST_SENTENCE.match(text)
    first = (match.group(1) if match else text).rstrip(".")
    if len(first) > width:
        cuts = [at for sep in (" — ", " -- ", " - ", ": ") if (at := first.find(sep)) > 20]
        if cuts:
            first = first[: min(cuts)]
    if len(first) > width:
        return first[: first.rfind(" ", 0, width - 1)].rstrip(",;") + "…"
    return first + "."


def _order(bundle: Bundle) -> tuple[int, int, str]:
    kinds = [kind for kind, _, _ in _SECTIONS]
    stage = STAGES.index(bundle.stage) if bundle.stage in STAGES else len(STAGES)
    return kinds.index(bundle.kind), stage, bundle.name


def _overview(head: str, bundles: list[Bundle]) -> list[str]:
    rows = [f"| {head} | Stage | What it does | Robots |", "| --- | --- | --- | --- |"]
    for b in bundles:
        rows.append(
            f"| [`{b.name}`](#{b.name}) | {b.stage or '-'} | {_cell(summary(b.description))} "
            f"| {_robots(b)} |"
        )
    return rows


def _section(bundle: Bundle, catalog: Catalog) -> list[str]:
    hub = bundle.hub
    facts = [bundle.stage or bundle.kind, f"robots: {_robots(bundle)}"]
    if hub.get("tasks"):
        facts.append("tasks: " + ", ".join(map(str, hub["tasks"])))
    if bundle.frontmatter.get("license"):
        facts.append(str(bundle.frontmatter["license"]))
    if bundle.version:
        facts.append(f"v{bundle.version}")
    facts.append(f"[`SKILL.md`]({_link(bundle, catalog, 'SKILL.md')})")
    upstream = (hub.get("provenance") or {}).get("code")
    if upstream:
        facts.append(f"[upstream]({upstream})")

    lines = [f"### {bundle.name}", "", " · ".join(facts), "", _prose(bundle.description), ""]
    items = _items(bundle, catalog)
    if hub.get("notes"):
        items.append(f"- **Note**: {_prose(str(hub['notes']))}")
    lines += items
    exits = bundle.gap.get("exit_conditions")
    if isinstance(exits, dict) and exits:
        lines.append("- **Exits**:")
        lines += [
            f"  - `{name}`: {_prose(' '.join(str(why).split()))}" for name, why in exits.items()
        ]
    functions = _functions(bundle)
    if functions:
        lines.append("- **Functions**:")
        lines += [
            f"  - `{name}`: {_prose(text)}" if text else f"  - `{name}`" for name, text in functions
        ]
    return lines


def _items(bundle: Bundle, catalog: Catalog) -> list[str]:
    items = []
    if bundle.kind == "skill" or bundle.required_inputs or bundle.produces_outputs:
        items.append(f"- **In**: {_typed(bundle.required_inputs)}")
        items.append(f"- **Out**: {_typed(bundle.produces_outputs)}")
    calls = _calls(bundle, catalog)
    if calls:
        items.append(f"- **Calls**: {calls}")
    scripts = _scripts(bundle, catalog)
    if scripts:
        items.append(f"- **Scripts**: {scripts}")
    io = bundle.hub.get("io")
    if isinstance(io, dict) and io:
        items.append(f"- **Trained on**: {_io(io)}")
    if bundle.weights:
        items.append(f"- **Weights**: `{bundle.weights['uri']}`")
    serving = bundle.serving
    if serving.get("protocol") and serving["protocol"] != "in-process":
        gpu = ", needs a GPU" if serving.get("requires_gpu") else ""
        items.append(f"- **Runs**: in its own environment over `{serving['protocol']}`{gpu}")
    return items


def _calls(bundle: Bundle, catalog: Catalog) -> str:
    named = bundle.allowed_tools + [
        str(t) for t in bundle.requires.get("connector") or [] if t not in bundle.allowed_tools
    ]
    tools = sorted({t.split(".", 1)[0] for t in named if t.split(".", 1)[0] in catalog.bundles})
    tools = [t for t in tools if t != bundle.name]
    connector = sorted({t for t in named if t.split(".", 1)[0] in CONNECTOR_PREFIXES})
    parts = []
    if tools:
        parts.append(", ".join(f"[`{t}`](#{t})" for t in tools))
    if connector:
        parts.append("connector " + ", ".join(f"`{t}`" for t in connector))
    return "; ".join(parts)


def _scripts(bundle: Bundle, catalog: Catalog) -> str:
    links = []
    for entry in bundle.gap.get("canonical_scripts") or []:
        if isinstance(entry, dict):
            links += [f"[`{name}`]({_link(bundle, catalog, path)})" for name, path in entry.items()]
    return ", ".join(links)


def _functions(bundle: Bundle) -> list[tuple[str, str]]:
    found = []
    for entry in bundle.gap.get("tools") or []:
        if isinstance(entry, dict):
            found += [
                (str(name), " ".join(str(text or "").split())) for name, text in entry.items()
            ]
        elif isinstance(entry, str):
            name, _, text = entry.partition(":")
            found.append((name.strip(), text.strip()))
    return found


def _io(io: dict[str, Any]) -> str:
    parts = []
    action = io.get("action") or {}
    if action:
        parts.append(
            f"action `{action.get('space')}` ×{action.get('dim')} at {action.get('hz')} Hz"
        )
    state = io.get("state") or {}
    if state:
        order = ", ".join(map(str, state.get("order") or []))
        parts.append(f"state {state.get('dim')}" + (f" ({order})" if order else ""))
    if io.get("cameras"):
        parts.append("cameras " + ", ".join(f"`{c}`" for c in io["cameras"]))
    return "; ".join(parts)


def _robots(bundle: Bundle) -> str:
    text = ", ".join(bundle.embodiments) or "-"
    tuned = [str(r) for r in bundle.hub.get("tuned_on") or [] if r not in bundle.embodiments]
    if tuned:
        text += f", *tuned on* {', '.join(tuned)}"
    if int(bundle.hub.get("arms") or 1) > 1:
        text += f", {bundle.hub['arms']} arms"
    if bundle.tier == "full":
        text += ", **sim only**"
    return text


def _typed(fields: dict[str, str]) -> str:
    return ", ".join(f"`{name}` {kind}" for name, kind in fields.items()) or "-"


def _link(bundle: Bundle, catalog: Catalog, relative: str) -> str:
    path = f"{_FOLDER[bundle.kind]}/{bundle.name}/{relative}"
    if bundle.native:
        return path
    pin = catalog.manifest.source(bundle.source)
    return f"{pin.repo}/blob/{pin.commit}/{path}"


def _recipes_and_papers(catalog: Catalog) -> list[str]:
    lines = []
    if catalog.recipes:
        lines += ["## Recipes", ""]
        for r in sorted(catalog.recipes.values(), key=lambda r: r.name):
            skills = sorted({str(s["skill"]) for s in r.skill_nodes.values()})
            uses = ", ".join(f"[`{s}`](#{s})" for s in skills)
            lines.append(
                f"- [`{r.name}`](recipes/{r.name}/RECIPE.md): {_prose(summary(r.description))} "
                f"Uses {uses}."
            )
        lines.append("")
    if catalog.papers:
        lines += ["## Papers", ""]
        for p in sorted(catalog.papers.values(), key=lambda p: p.id):
            title = _prose(str(p.frontmatter.get("title") or p.id))
            year = p.frontmatter.get("year")
            lines.append(
                f"- [`{p.id}`](papers/{p.id}/PAPER.md): {title}" + (f" ({year})" if year else "")
            )
        lines.append("")
    return lines


def _prose(text: str) -> str:
    """Text a description wrote, safe in Markdown: a ``<name>`` outside backticks is not a tag."""
    parts = text.split("`")
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("<", "&lt;").replace(">", "&gt;")
    return "`".join(parts)


def _cell(text: str) -> str:
    return _prose(text).replace("|", "\\|")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")
