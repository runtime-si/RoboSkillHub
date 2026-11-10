"""``roboskillhub``: check, search, slice, lock and fetch the skill hub."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

from .catalog import load_catalog
from .sources import cache_dir, find_root, load_manifest, sync


def _catalog(args):
    return load_catalog(args.root)


def cmd_sync(args) -> int:
    root = args.root or find_root()
    manifest = load_manifest(root)
    for name, path in sync(manifest, cache_dir(root)).items():
        print(f"{name:24} {path}")
    return 0


def cmd_check(args) -> int:
    from .checks import check_all

    catalog = _catalog(args)
    issues = check_all(catalog, require_gap=args.require_gap)
    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]
    for issue in issues if args.verbose else errors:
        print(issue)
    counts = {k: len(catalog.of_kind(k)) for k in ("tool", "skill", "policy")}
    print(
        f"{counts['tool']} tools, {counts['skill']} skills, {counts['policy']} policies, "
        f"{len(catalog.recipes)} recipes, {len(catalog.papers)} papers: "
        f"{len(errors)} errors, {len(warnings)} warnings"
        + ("" if args.verbose else " (-v lists warnings)")
    )
    return 1 if errors else 0


def cmd_index(args) -> int:
    from .index import build_index

    text = json.dumps(build_index(_catalog(args)), indent=2, sort_keys=False) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_site(args) -> int:
    from .site import write

    n = write(_catalog(args), Path(args.out), args.repo)
    print(f"wrote {n} pages to {args.out}")
    return 0


def cmd_catalog(args) -> int:
    from .catalog_page import PAGE, render

    catalog = _catalog(args)
    path = catalog.root / PAGE
    text = render(catalog)
    current = path.read_text(encoding="utf-8") if path.is_file() else ""
    if args.check:
        if current != text:
            print(f"{PAGE} is out of date; run `roboskillhub catalog`")
            return 1
        print(f"{PAGE} is current")
        return 0
    if current != text:
        path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")
    return 0


def _select(args):
    from .search import select

    catalog = _catalog(args)
    return catalog, select(
        catalog,
        args.query,
        embodiment=args.embodiment,
        tier=args.tier,
        min_maturity=args.min_maturity,
        arms=args.arms,
        kinds=tuple(args.kind.split(",")),
        k=args.k,
        stages=tuple(s for s in (args.stage or "").split(",") if s),
        failure=args.failure or "",
    )


def cmd_search(args) -> int:
    _, selection = _select(args)
    if args.json:
        print(
            json.dumps(
                {
                    "query": selection.query,
                    "picks": [
                        {
                            "name": p.bundle.name,
                            "kind": p.bundle.kind,
                            "score": round(p.score, 3),
                            "reason": p.reason,
                            "evidence": selection.evidence.get(p.bundle.name, ""),
                        }
                        for p in selection.picks
                    ],
                    "recipes": [r.name for r, _ in selection.recipes],
                    "excluded": selection.excluded,
                },
                indent=2,
            )
        )
        return 0
    for pick in selection.picks:
        b = pick.bundle
        proof = selection.evidence.get(b.name)
        print(
            f"{pick.score:6.2f}  {b.kind:6} {b.name:36} {b.stage or '-':9} {b.maturity:13} {pick.reason}"
            + (f" ({proof})" if proof else "")
        )
    for recipe, score in selection.recipes:
        print(f"{score:6.2f}  recipe {recipe.name}")
    if args.why:
        for name, why in sorted(selection.excluded.items()):
            print(f"  excluded {name}: {why}")
    else:
        print(f"({len(selection.excluded)} excluded; --why lists them)")
    return 0


def cmd_view(args) -> int:
    from .view import make_view

    catalog, selection = _select(args)
    if args.bundles:
        picks = [catalog.get(n) for n in args.bundles.split(",")]
        recipes = []
    else:
        picks, recipes = selection.picks, [r for r, _ in selection.recipes]
    out = make_view(picks, Path(args.out), query=args.query, recipes=recipes)
    print(f"{len(list(picks))} bundles, {len(recipes)} recipes -> {out}")
    print(f"export GAP_SKILLS_PATH={out}")
    return 0


def cmd_lock(args) -> int:
    from .lock import build_lock, write_lock

    graph = Path(args.graph)
    lock = build_lock(graph, _catalog(args))
    out = Path(args.out) if args.out else graph / "skills.lock"
    write_lock(lock, out)
    print(
        f"{len(lock['bundles'])} bundles, {len(lock['connector_tools'])} connector tools -> {out}"
    )
    if lock.get("unresolved_tools"):
        print(f"unresolved tools: {lock['unresolved_tools']}")
    if lock["hub"]["dirty"]:
        print(
            "warning: the hub checkout has uncommitted changes; roboskillhub fetch will refuse hub bundles"
        )
    if args.deploy and not lock["deployable"]:
        tiers = [n for n, b in lock["bundles"].items() if b["tier"] != "camera_only"]
        print(
            f"not deployable: privileged tools {lock['privileged_tools']}, simulator-only bundles {tiers}"
        )
        return 1
    return 0


def cmd_fetch(args) -> int:
    from .lock import fetch

    lock = json.loads(Path(args.lock).read_text(encoding="utf-8"))
    result = fetch(lock, Path(args.dest), weights=not args.no_weights)
    for name, path in result["weights"].items():
        print(f"weights {name:20} {path}")
    print(f"export GAP_SKILLS_PATH={os.pathsep.join(result['registries'])}")
    return 0


def cmd_weights(args) -> int:
    from . import weights

    if args.action == "push":
        pushed = weights.push(
            Path(args.local), args.repo, private=not args.public, message=args.message
        )
        print(f"uri:    {pushed.ref.uri}")
        print(f"sha256: {pushed.sha256}")
        print(f"files:  {len(pushed.files)}")
        if args.bundle:
            skill_md = Path(args.bundle) / "SKILL.md"
            if weights.write_into_skill_md(skill_md, pushed):
                print(f"updated {skill_md}")
            else:
                print(f"{skill_md} has no weights_uri:/sha256: lines; paste the two values above")
        return 0
    path = weights.pull(args.uri, sha256=args.sha256)
    print(path)
    return 0


def cmd_new(args) -> int:
    from .scaffold import new

    dest = new(args.root or find_root(), args.kind, args.name)
    print(f"created {dest}")
    return 0


def cmd_discover(args) -> int:
    from .discovery import discover, load_candidates

    data = load_candidates(args.root or find_root())
    rows = discover(
        data,
        args.query,
        category=args.category,
        destination=args.destination,
        priority=args.priority,
        include_research_only=args.include_research_only,
    )
    if args.json:
        print(
            json.dumps(
                {"query": args.query, "invocable_in_hub": False, "candidates": rows},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("Source-reviewed candidates; adaptation and validation required before invocation.")
        for row in rows:
            license_id = row["code_license"]["id"]
            print(f"{row['priority']}  {row['id']}  [{row['hub_destination']}]  {row['name']}")
            print(f"    code license: {license_id}; weights: {row['weights']['license']}")
            print(f"    {row['source_url']}")
        print(
            f"{len(rows)} candidates; --json includes entrypoints, prerequisites and intake steps"
        )
    return 0


def cmd_check_candidates(args) -> int:
    from .discovery import load_candidates

    data = load_candidates(args.root or find_root())
    unpinned = sum(row["source_revision"] is None for row in data["candidates"])
    print(f"{len(data['candidates'])} source candidates: metadata valid, {unpinned} unpinned")
    print(
        "This checks intake metadata only; it does not execute upstream code or validate hardware."
    )
    return 0


def cmd_runtime(args) -> int:
    from .runtime.api import describe, invoke

    if args.runtime_action in ("list", "describe"):
        with contextlib.redirect_stdout(sys.stderr):
            result = describe(args.station)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.runtime_action == "invoke":
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        with contextlib.redirect_stdout(sys.stderr):
            result = invoke(args.capability, request, args.station)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result["status"] == "succeeded" else 1
    # Local JSON-lines protocol: one request/reply per line, no network listener.
    # Station selection is trusted process configuration, never request data.
    for line in sys.stdin:
        ident = None
        try:
            if len(line) > 8 * 1024 * 1024:
                raise ValueError("request exceeds 8 MiB; pass frames_path instead of image bytes")
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("message must be an object")
            ident = message.get("id")
            method = message.get("method")
            if method == "describe":
                with contextlib.redirect_stdout(sys.stderr):
                    result = describe(args.station)
            elif method == "invoke":
                params = message.get("params", {})
                with contextlib.redirect_stdout(sys.stderr):
                    result = invoke(params["capability"], params["request"], args.station)
            else:
                raise ValueError("method must be describe or invoke")
            response = {"id": ident, "result": result}
            encoded = json.dumps(response, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as exc:
            encoded = json.dumps({"id": ident, "error": {"message": str(exc)}}, ensure_ascii=False)
        print(encoded, flush=True)
    return 0


def cmd_capability(args) -> int:
    from .capability.select import select
    from .capability.spec import list_capabilities, load_capability

    root = args.root or find_root()
    if args.capability_action == "list":
        for name in list_capabilities(root):
            spec = load_capability(name, root)
            impls = ", ".join(i.id for i in spec.implementations)
            print(f"{name:20} {len(spec.exits)} exits; {impls}")
        return 0
    spec = load_capability(args.name, root)
    if args.capability_action == "show":
        print(json.dumps(spec.raw, ensure_ascii=False, indent=2))
        return 0
    choices = select(spec, root, args.embodiment, args.setting)
    if args.json:
        print(json.dumps([c.to_dict() for c in choices], ensure_ascii=False, indent=2))
    else:
        for c in choices:
            if not c.eligible and not args.why:
                continue
            mark = "ok " if c.eligible else "-- "
            print(f"{mark}{c.implementation.id}")
            if args.why:
                for line in c.why:
                    print(f"     {line}")
        if not any(c.eligible for c in choices):
            print(
                f"no implementation of {spec.name} passed the {args.setting} gate on "
                f"{args.embodiment}" + ("" if args.why else " (--why shows each)"),
                file=sys.stderr,
            )
    return 0 if any(c.eligible for c in choices) else 1


def cmd_sources(args) -> int:
    from .source_index import find_sources, load_source_index

    data = load_source_index(args.root or find_root())
    matches = find_sources(
        data,
        args.query,
        category=args.category,
        license_status=args.license_status,
        review_level=args.review_level,
    )
    limit = args.limit if args.limit is not None else (None if args.json else 30)
    rows = matches if limit is None else matches[:limit]
    if args.json:
        print(
            json.dumps(
                {
                    "collection": data["collection"],
                    "snapshot_date": data["snapshot_date"],
                    "query": args.query,
                    "invocable_in_hub": False,
                    "matched_count": len(matches),
                    "returned_count": len(rows),
                    "entries": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("Source index only; every entry is non-invocable and requires adaptation.")
        print(
            "All license statuses are included; code labels do not clear weights or dependencies."
        )
        for row in rows:
            license_record = row["license"]
            print(f"{row['priority']}  {row['id']}  [{row['category']}]  {row['name']}")
            print(
                f"    {row['review_level']}; code license: {license_record['status']} "
                f"({license_record['spdx'] or 'unresolved'}); invocable: false"
            )
            print(f"    {row['source_url']}")
        print(
            f"{len(matches)} matched; showing {len(rows)} (--limit N; --json includes full metadata)"
        )
    return 0


def cmd_check_sources(args) -> int:
    from .source_index import LICENSE_STATUSES, REVIEW_LEVELS, load_source_index

    data = load_source_index(args.root or find_root())
    rows = data["entries"]
    licenses = ", ".join(
        f"{status}: {sum(row['license']['status'] == status for row in rows)}"
        for status in LICENSE_STATUSES
    )
    reviews = ", ".join(
        f"{level}: {sum(row['review_level'] == level for row in rows)}" for level in REVIEW_LEVELS
    )
    print(f"{len(rows)} {data['collection']} source entries: metadata valid")
    print(f"    license: {licenses}")
    print(f"    review:  {reviews}")
    print(
        "Metadata validation only; no upstream execution, license clearance or hardware validation."
    )
    return 0


def _positive_limit(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("limit must be a positive integer")
    return number


def _selection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("query", help="the task instruction, in words")
    p.add_argument("--embodiment", help="robot id (rss.robot() name), e.g. yam_i2rt")
    p.add_argument("--tier", choices=("camera_only", "full"), default="camera_only")
    p.add_argument(
        "--min-maturity",
        default="community",
        choices=("community", "sim-verified", "real-verified"),
    )
    p.add_argument("--arms", type=int, help="arms the robot has; drops skills that need more")
    p.add_argument("--kind", default="skill,policy", help="comma-separated kinds to rank")
    p.add_argument(
        "-k", type=int, default=16, help="how many to keep before adding the tools they call"
    )
    p.add_argument(
        "--stage",
        help="comma-separated stages; rank only these (what could replace a failed skill)",
    )
    p.add_argument("--failure", help="how the run failed, in words; ranked with the query")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roboskillhub", description=__doc__)
    parser.add_argument(
        "--root", type=Path, help="hub checkout (default: nearest parent with sources.yaml)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("sync", help="check out every pinned source and the engine").set_defaults(
        fn=cmd_sync
    )

    p = sub.add_parser("check", help="validate bundles, overlays, recipes, papers, evals")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--require-gap", action="store_true", help="fail when gap is not importable")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("index", help="write the catalog as index.json")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("catalog", help="write CATALOG.md, every bundle on one page")
    p.add_argument("--check", action="store_true", help="fail if CATALOG.md is out of date")
    p.set_defaults(fn=cmd_catalog)

    p = sub.add_parser(
        "site", help="write the library as a static website (what GitHub Pages serves)"
    )
    p.add_argument("--out", default="_site")
    p.add_argument(
        "--repo", default="runtime-si/RoboSkillHub", help="owner/name, for the links back to GitHub"
    )
    p.set_defaults(fn=cmd_site)

    p = sub.add_parser("search", help="rank the catalog against a task")
    _selection_args(p)
    p.add_argument("--json", action="store_true")
    p.add_argument("--why", action="store_true", help="list what was filtered out and why")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("view", help="write a registry holding only what a task needs")
    _selection_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--bundles", help="comma-separated names instead of a search")
    p.set_defaults(fn=cmd_view)

    p = sub.add_parser("lock", help="pin every bundle a generated graph uses")
    p.add_argument("graph", help="directory holding workflow.json")
    p.add_argument("--out")
    p.add_argument(
        "--deploy", action="store_true", help="fail unless the graph can leave simulation"
    )
    p.set_defaults(fn=cmd_lock)

    p = sub.add_parser("fetch", help="recreate a lock's registries and weights on this machine")
    p.add_argument("lock")
    p.add_argument("--dest", required=True)
    p.add_argument("--no-weights", action="store_true")
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("weights", help="push or pull checkpoints on the Hugging Face Hub")
    wsub = p.add_subparsers(dest="action", required=True)
    w = wsub.add_parser("push")
    w.add_argument("local", help="directory holding the checkpoint files")
    w.add_argument("--repo", required=True, help="<owner>/<name> on the Hub")
    w.add_argument(
        "--bundle", help="bundle directory whose SKILL.md gets the pinned uri and sha256"
    )
    w.add_argument("--public", action="store_true", help="create the repo public (default private)")
    w.add_argument("--message", default="Upload checkpoint")
    w = wsub.add_parser("pull")
    w.add_argument("uri")
    w.add_argument("--sha256")
    p.set_defaults(fn=cmd_weights)

    p = sub.add_parser("new", help="start a bundle, recipe or paper from its template")
    p.add_argument("kind", choices=("tool", "skill", "policy", "recipe", "paper"))
    p.add_argument("name")
    p.set_defaults(fn=cmd_new)

    p = sub.add_parser("discover", help="find reviewed upstream materials (not executable bundles)")
    p.add_argument(
        "query", nargs="?", default="", help="English or Chinese keywords, separated by spaces"
    )
    p.add_argument("--category", help="exact category from candidates/catalog.json")
    p.add_argument("--destination", choices=("tool", "skill", "policy", "recipe"))
    p.add_argument("--priority", choices=("P1", "P2", "P3"))
    p.add_argument(
        "--include-research-only",
        action="store_true",
        help="also list noncommercial research sources; weights have separate terms",
    )
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_discover)

    sub.add_parser("check-candidates", help="validate source-intake metadata").set_defaults(
        fn=cmd_check_candidates
    )

    from .source_index import LICENSE_STATUSES, REVIEW_LEVELS

    p = sub.add_parser("sources", help="find indexed upstream sources (never executable bundles)")
    p.add_argument("query", nargs="?", default="", help="keywords; all terms must match")
    p.add_argument("--category", help="exact category from candidates/pick-place-index.json")
    p.add_argument("--license-status", choices=LICENSE_STATUSES)
    p.add_argument("--review-level", choices=REVIEW_LEVELS)
    p.add_argument(
        "--limit", type=_positive_limit, help="maximum results; default 30 text, all JSON"
    )
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_sources)
    sub.add_parser("check-sources", help="validate the separate source index").set_defaults(
        fn=cmd_check_sources
    )

    p = sub.add_parser("runtime", help="inspect and invoke station-bound capabilities")
    rsub = p.add_subparsers(dest="runtime_action", required=True)
    for action in ("list", "describe", "invoke", "serve"):
        r = rsub.add_parser(action)
        r.add_argument("--station", help="trusted station YAML; defaults to ROBOSKILLHUB_STATION")
        if action == "invoke":
            from .runtime.contracts import SCHEMAS

            r.add_argument("capability", choices=tuple(SCHEMAS))
            r.add_argument("--request", required=True, help="path to input JSON")
        r.set_defaults(fn=cmd_runtime)

    p = sub.add_parser(
        "capability", help="capability contracts and their eval-gated implementations"
    )
    csub = p.add_subparsers(dest="capability_action", required=True)
    csub.add_parser("list").set_defaults(fn=cmd_capability)
    c = csub.add_parser("show")
    c.add_argument("name")
    c.set_defaults(fn=cmd_capability)
    c = csub.add_parser("select", help="implementations that passed the gate on a robot")
    c.add_argument("name")
    c.add_argument("--embodiment", required=True)
    c.add_argument("--setting", choices=("sim", "real"), default="sim")
    c.add_argument("--why", action="store_true", help="also list rejected ones, with reasons")
    c.add_argument("--json", action="store_true")
    c.set_defaults(fn=cmd_capability)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, KeyError) as exc:
        print(f"roboskillhub: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
