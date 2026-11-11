"""Read-only catalog of fine-tuned robot action models (VLA checkpoints) and their datasets.

Like ``source_index``, nothing here is an executable Hub bundle: an entry records where a checkpoint is,
what it was trained on, how its inputs and outputs look and whether the YAM station can use it. Running one
still goes through a policy bundle's server and a checked-in profile (``policies/*/profiles``).
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

CHECKPOINTS_PATH = Path("candidates/vla-checkpoints.json")
DATASETS_PATH = Path("candidates/vla-datasets.json")
PAGE_PATH = Path("candidates/vla-catalog.md")
REJECTED_PATH = Path("candidates/vla-rejected.json")
YAM_COMPATIBLE = ("as_is", "adapter", "finetune_base", "no")
OUR_SERVERS = ("policies/molmoact2-yam", "policies/gr00t:n1.7", "policies/gr00t:pi05")
CHECKPOINT_FIELDS = (
    "id",
    "revision",
    "family",
    "base_model",
    "format",
    "embodiment",
    "arms",
    "action",
    "state",
    "cameras",
    "training",
    "license",
    "gated",
    "size_gb",
    "servable_by",
    "yam_compatible",
    "task_relevance",
    "notes",
    "source_urls",
    "checked_at",
)
DATASET_FIELDS = (
    "id",
    "revision",
    "embodiment",
    "episodes",
    "tasks",
    "format",
    "license",
    "size_gb",
    "relevance",
    "source_urls",
)
_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _http(value: object) -> bool:
    if not _text(value) or any(char.isspace() for char in value):
        return False
    try:
        parsed = urlsplit(value)
        return parsed.scheme in ("https", "http") and bool(parsed.hostname)
    except ValueError:
        return False


def _iso_date(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value[:10])
    except ValueError:
        return False
    return True


def _header(data: object, collection: str) -> tuple[list[str], list | None]:
    if not isinstance(data, dict):
        return [f"{collection} must be an object"], None
    errors: list[str] = []
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        errors.append(f"{collection} must have schema_version: 1")
    if data.get("collection") != collection:
        errors.append(f"{collection}: collection must be {collection}")
    if not _text(data.get("scope")):
        errors.append(f"{collection}: scope must be a nonempty string")
    snapshot = data.get("snapshot_date")
    if not (_iso_date(snapshot) and len(snapshot) == 10):
        errors.append(f"{collection}: snapshot_date must be an ISO YYYY-MM-DD date")
    rows = data.get("entries")
    if not isinstance(rows, list):
        return errors + [f"{collection}: entries must be a list"], None
    return errors, rows


def _common(row: dict, where: str, fields: tuple[str, ...], ids: set[str]) -> list[str]:
    errors = [f"{where}: missing {key}" for key in fields if key not in row]
    ident = row.get("id")
    if not isinstance(ident, str) or not _REPO.fullmatch(ident):
        errors.append(f"{where}: id must be a Hugging Face repo id owner/name")
    elif ident.casefold() in ids:
        errors.append(f"{where}: duplicate id {ident}")
    else:
        ids.add(ident.casefold())
    revision = row.get("revision")
    if "revision" in row and not (isinstance(revision, str) and _COMMIT.fullmatch(revision)):
        errors.append(f"{where}: revision must be a full 40-character commit (pinned)")
    urls = row.get("source_urls")
    if "source_urls" in row and (
        not isinstance(urls, list) or not urls or not all(_http(u) for u in urls)
    ):
        errors.append(f"{where}: source_urls must be a nonempty list of HTTP(S) URLs")
    if "size_gb" in row and row["size_gb"] is not None and not _number(row["size_gb"]):
        errors.append(f"{where}: size_gb must be a number or null")
    return errors


def _license(row: dict, where: str, *, required: tuple[str, ...]) -> list[str]:
    record = row.get("license")
    if isinstance(record, str) and not required:
        return [] if _text(record) else [f"{where}: license must be nonempty"]
    if not isinstance(record, dict):
        return [f"{where}: license must be an object"]
    errors = []
    for key in required:
        if key not in record:
            errors.append(f"{where}: license.{key} is required")
    if "verified" in record and not isinstance(record["verified"], bool):
        errors.append(f"{where}: license.verified must be true or false")
    if record.get("model") is not None and not _text(record["model"]):
        errors.append(f"{where}: license.model must be a nonempty string or null")
    return errors


def _evaluations(value: object, where: str) -> list[str]:
    if not isinstance(value, list):
        return [f"{where}: evaluations must be a list"]
    errors = []
    for number, item in enumerate(value):
        position = f"{where}.evaluations[{number}]"
        if not isinstance(item, dict):
            errors.append(f"{position}: expected an object")
            continue
        for key in ("setting", "task", "date", "evidence"):
            if not _text(item.get(key)):
                errors.append(f"{position}: {key} must be a nonempty string")
        episodes, successes = item.get("episodes"), item.get("successes")
        if type(episodes) is not int or episodes < 1:
            errors.append(f"{position}: episodes must be a positive integer")
        elif type(successes) is not int or not 0 <= successes <= episodes:
            errors.append(f"{position}: successes must be an integer in 0..episodes")
    return errors


def validate_checkpoints(data: object) -> list[str]:
    """Required fields, pinned revisions, explicit licence verification, known compatibility."""
    errors, rows = _header(data, "vla-checkpoints")
    if rows is None:
        return errors
    ids: set[str] = set()
    for index, row in enumerate(rows):
        where = f"entries[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{where}: expected an object")
            continue
        errors += _common(row, where, CHECKPOINT_FIELDS, ids)
        errors += _license(row, where, required=("model", "verified"))
        if "yam_compatible" in row and row["yam_compatible"] not in YAM_COMPATIBLE:
            errors.append(f"{where}: yam_compatible must be one of {', '.join(YAM_COMPATIBLE)}")
        for key in ("family", "embodiment"):
            if key in row and not _text(row[key]):
                errors.append(f"{where}: {key} must be a nonempty string")
        if "gated" in row and not isinstance(row["gated"], bool):
            errors.append(f"{where}: gated must be true or false")
        if row.get("arms") is not None and (type(row["arms"]) is not int or row["arms"] < 1):
            errors.append(f"{where}: arms must be a positive integer or null")
        for key in ("cameras", "servable_by"):
            if key in row and (
                not isinstance(row[key], list) or not all(_text(v) for v in row[key])
            ):
                errors.append(f"{where}: {key} must be a list of strings")
        for key in ("action", "state", "training"):
            if key in row and not isinstance(row[key], dict):
                errors.append(f"{where}: {key} must be an object")
        action = row.get("action")
        if (
            isinstance(action, dict)
            and action.get("space") is not None
            and not _text(action["space"])
        ):
            errors.append(f"{where}: action.space must be a nonempty string or null")
        if "checked_at" in row and not _iso_date(row["checked_at"]):
            errors.append(f"{where}: checked_at must be an ISO date")
        if "evaluations" in row:
            errors += _evaluations(row["evaluations"], where)
    return errors


def validate_datasets(data: object) -> list[str]:
    errors, rows = _header(data, "vla-datasets")
    if rows is None:
        return errors
    ids: set[str] = set()
    for index, row in enumerate(rows):
        where = f"entries[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{where}: expected an object")
            continue
        errors += _common(row, where, DATASET_FIELDS, ids)
        errors += _license(row, where, required=())
        if "embodiment" in row and not _text(row["embodiment"]):
            errors.append(f"{where}: embodiment must be a nonempty string")
    return errors


def load_vla_index(root: Path) -> tuple[dict, dict]:
    """Both collections, or ValueError listing every problem."""
    out = []
    for path, check in (
        (CHECKPOINTS_PATH, validate_checkpoints),
        (DATASETS_PATH, validate_datasets),
    ):
        data = json.loads((root / path).read_text(encoding="utf-8"))
        errors = check(data)
        if errors:
            raise ValueError(f"invalid {path}:\n" + "\n".join(errors))
        out.append(data)
    return out[0], out[1]


def our_servers(row: dict) -> list[str]:
    """Which of this hub's policy servers can run the checkpoint (``servable_by`` names them)."""
    found = set()
    for name in (str(v).casefold() for v in row.get("servable_by") or ()):
        if "molmoact2-yam" in name:
            found.add("policies/molmoact2-yam")
        if "policies/gr00t" in name:
            if re.search(r"pi0?\.?5", name):
                found.add("policies/gr00t:pi05")
            if re.search(r"n1\.?7", name):
                found.add("policies/gr00t:n1.7")
    return [s for s in OUR_SERVERS if s in found]


PROFILE_DIRS = {
    "policies/molmoact2-yam": Path("policies/molmoact2-yam/profiles"),
    "policies/gr00t:n1.7": Path("policies/gr00t/profiles"),
    "policies/gr00t:pi05": Path("policies/gr00t/profiles"),
}


def profiled_checkpoints(root: Path) -> dict[str, list[Path]]:
    """Checkpoint repo id (casefolded) -> the policy-bundle profiles that pin it."""
    import yaml

    out: dict[str, list[Path]] = {}
    for directory in sorted(set(PROFILE_DIRS.values())):
        for path in sorted((root / directory).glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            repo = (raw.get("checkpoint") or {}).get("repo")
            if isinstance(repo, str):
                out.setdefault(repo.casefold(), []).append(path.relative_to(root))
    return out


def profile_gaps(root: Path, rows: list[dict]) -> list[tuple[str, list[str]]]:
    """``as_is`` checkpoints one of our servers can run that have no profile yet, with where it belongs."""
    have = profiled_checkpoints(root)
    gaps = []
    for row in rows:
        servers = our_servers(row)
        if row.get("yam_compatible") == "as_is" and servers and row["id"].casefold() not in have:
            gaps.append((row["id"], sorted({str(PROFILE_DIRS[s]) for s in servers})))
    return gaps


def embodiment_matches(row: dict, embodiment: str) -> bool:
    """``yam_i2rt`` also matches entries whose embodiment names YAM in other words."""
    wanted = embodiment.casefold()
    have = str(row.get("embodiment", "")).casefold()
    if wanted == have:
        return True
    if wanted == "yam_i2rt":
        return bool(re.search(r"(^|[^a-z])yam([^a-z]|$)", have))
    return wanted in have


def _label(row: dict) -> str:
    training = row.get("training") if isinstance(row.get("training"), dict) else {}
    tasks = row.get("tasks")
    tasks = tasks if isinstance(tasks, list) else [tasks or ""]
    parts = [
        row.get("id", ""),
        row.get("family", ""),
        row.get("base_model", ""),
        row.get("format", ""),
        row.get("embodiment", ""),
        row.get("task_relevance", ""),
        row.get("relevance", ""),
        row.get("notes", ""),
        f"{row.get('arms', '')} arms" if row.get("arms") else "",
        "bimanual dual-arm" if row.get("arms") == 2 else "",
        *map(str, training.get("datasets") or ()),
        *map(str, training.get("tasks") or ()),
        *map(str, tasks),
    ]
    return " ".join(str(p) for p in parts).casefold()


def find_vla(
    rows: list[dict],
    query: str = "",
    *,
    embodiment: str | None = None,
    compatible: str | None = None,
    family: str | None = None,
) -> list[dict]:
    """Rank by matched query terms (any term), then YAM compatibility, then whether we can serve it."""
    terms = query.casefold().split()
    scored = []
    for order, row in enumerate(rows):
        if embodiment is not None and not embodiment_matches(row, embodiment):
            continue
        if compatible is not None and row.get("yam_compatible") != compatible:
            continue
        if family is not None and str(row.get("family", "")).casefold() != family.casefold():
            continue
        label = _label(row)
        hits = sum(term in label for term in terms)
        if terms and not hits:
            continue
        compat = row.get("yam_compatible")
        rank = YAM_COMPATIBLE.index(compat) if compat in YAM_COMPATIBLE else len(YAM_COMPATIBLE)
        scored.append((-hits, rank, not our_servers(row), order, row))
    return [item[-1] for item in sorted(scored, key=lambda item: item[:4])]


def _filled(value: object) -> int:
    if value is None or value == "" or value == [] or value == {}:
        return 0
    if isinstance(value, dict):
        return sum(_filled(v) for v in value.values())
    if isinstance(value, list):
        return 1 + sum(_filled(v) for v in value)
    return 1


def merge_entries(*sources: list[dict]) -> list[dict]:
    """Dedupe by id (case-insensitive) keeping the most complete record; first-seen order.

    The kept record gains ``evaluations`` and ``source_urls`` from the others, so a hand-written
    evaluation is not lost when a fuller upstream record wins."""
    groups: dict[str, list[dict]] = {}
    for rows in sources:
        for row in rows:
            if isinstance(row, dict) and _text(row.get("id")):
                groups.setdefault(row["id"].strip().casefold(), []).append(row)
    merged = []
    for group in groups.values():
        best = dict(
            max(
                group,
                key=lambda row: _filled(
                    {k: v for k, v in row.items() if k not in ("evaluations", "source_urls")}
                ),
            )
        )
        urls = list(dict.fromkeys(u for row in group for u in row.get("source_urls") or ()))
        if urls:
            best["source_urls"] = urls
        evaluations = [e for row in group for e in row.get("evaluations") or ()]
        if evaluations:
            seen, unique = set(), []
            for item in evaluations:
                key = json.dumps(item, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    unique.append(item)
            best["evaluations"] = unique
        merged.append(best)
    return merged


def _rows(data: object) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("entries", "models", "checkpoints", "datasets", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    return []


_FAMILIES = (
    (r"molmoact\s*2", "molmoact2"),
    (r"molmoact", "molmoact"),
    (r"(pi|π)\s*0?\.?5|pi05", "pi05"),
    (r"(pi|π)[\s_-]*0(?![\d.])", "pi0"),
    (r"gr00t|groot", "gr00t"),
    (r"smol\s*vla", "smolvla"),
    (r"x-?vla", "xvla"),
    (r"openvla", "openvla"),
    (r"octo\b", "octo"),
    (r"rdt", "rdt"),
    (r"diffusion", "diffusion-policy"),
    (r"\bact\b", "act"),
    (r"wam\b|world.action", "wam"),
    (r"lingbot", "lingbot-vla"),
    (r"wall-?oss", "wall-oss"),
    (r"galaxea", "galaxea-g0"),
    (r"cosmos", "cosmos-policy"),
)
_EMBODIMENTS = (
    (r"(^|[^a-z])yam([^a-z]|$)", "yam_i2rt"),
    (r"aloha|trossen", "aloha"),
    (r"so-?10[01]|koch", "so100"),
    (r"franka|panda|droid", "franka"),
    (r"piper", "piper"),
    (r"ur5|ur10", "ur"),
    (r"widowx|bridge", "widowx"),
    (r"galaxea|r1 ", "galaxea"),
    (r"unitree|g1\b|h1\b", "unitree"),
    (r"cross|multi|generalist|base", "cross_embodiment"),
)
_COMPAT = (
    (r"^as_is$|^direct", "as_is"),
    (r"fine-?tune|finetune", "finetune_base"),
    (r"adapter|retarget|map", "adapter"),
    (r"^no\b|reference|none", "no"),
)


def _canonical(value: object, table: tuple, default: str) -> str:
    text = str(value or "").casefold()
    for pattern, name in table:
        if re.search(pattern, text):
            return name
    return default


def _normalize(row: dict) -> dict:
    """Survey spellings -> the schema. Free-text family / embodiment / compatibility are mapped to
    canonical values and the original kept in ``*_detail``; ``gated: "auto"|"manual"`` is gated."""
    row = dict(row)
    for key, table, default in (
        ("family", _FAMILIES, None),
        ("embodiment", _EMBODIMENTS, "other"),
    ):
        value = row.get(key)
        if isinstance(value, str) and value not in {name for _p, name in table}:
            canonical = _canonical(value, table, default or "")
            if not canonical:
                canonical = re.sub(r"[^a-z0-9.]+", "-", value.casefold().split("(")[0]).strip("-")
            row.setdefault(f"{key}_detail", value)
            row[key] = canonical or "unknown"
    compat = row.get("yam_compatible")
    if isinstance(compat, str) and compat not in YAM_COMPATIBLE:
        row.setdefault("yam_compatible_detail", compat)
        text = compat.casefold()
        if text.startswith("direct") and re.search(r"fine-?tune", text):
            row["yam_compatible"] = "finetune_base"
        else:
            row["yam_compatible"] = _canonical(compat, _COMPAT, "no")
    servable = row.get("servable_by")
    if isinstance(servable, str):
        text = servable.strip()
        row["servable_by"] = [] if re.fullmatch(r"(?i)(no|none|-|)", text) else [text]
    arms = row.get("arms")
    if isinstance(arms, str):
        numbers = [int(n) for n in re.findall(r"\d+", arms)]
        row.setdefault("arms_detail", arms)
        row["arms"] = max(numbers) if numbers and max(numbers) > 0 else None
    cameras = row.get("cameras")
    if isinstance(cameras, list):
        row["cameras"] = [
            " ".join(str(v) for v in c.values() if v) if isinstance(c, dict) else str(c)
            for c in cameras
            if c
        ]
    elif cameras is None:
        row["cameras"] = []
    if isinstance(row.get("gated"), str):
        row["gated"] = row["gated"].strip().casefold() not in ("", "false", "no", "none")
    if isinstance(row.get("size_gb"), str):
        try:
            row["size_gb"] = float(row["size_gb"])
        except ValueError:
            row["size_gb"] = None
    return row


def _row_errors(row: dict, check, collection: str) -> list[str]:
    wrapped = {
        "schema_version": 1,
        "collection": collection,
        "scope": "row",
        "snapshot_date": "2000-01-01",
        "entries": [row],
    }
    return [e.removeprefix("entries[0]: ") for e in check(wrapped)]


def merge_files(root: Path, files: list[Path], *, write: bool = False) -> list[str]:
    """Merge survey files into both collections; rows that fail validation go to the rejected file.

    A file whose name or ``collection`` mentions datasets feeds ``vla-datasets``; the rest feed
    ``vla-checkpoints``. Existing entries take part in the dedupe like any other source."""
    collections = {
        "vla-checkpoints": (CHECKPOINTS_PATH, validate_checkpoints, [], []),
        "vla-datasets": (DATASETS_PATH, validate_datasets, [], []),
    }
    for path, _check, incoming, _rejected in collections.values():
        incoming.append(json.loads((root / path).read_text(encoding="utf-8"))["entries"])
    report = []
    for file in files:
        if not file.exists():
            report.append(f"{file}: missing, skipped")
            continue
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.append(f"{file}: not JSON ({exc}), skipped")
            continue
        kind = "dataset" in file.name or (
            isinstance(data, dict) and "dataset" in str(data.get("collection", ""))
        )
        rows = _rows(data)
        collections["vla-datasets" if kind else "vla-checkpoints"][2].append(rows)
        report.append(
            f"{file}: {len(rows)} rows -> {'vla-datasets' if kind else 'vla-checkpoints'}"
        )
    rejected_out = []
    for name, (path, check, incoming, rejected) in collections.items():
        kept = []
        for row in map(_normalize, merge_entries(*incoming)):
            errors = _row_errors(row, check, name)
            if errors:
                rejected.append({"id": row.get("id"), "collection": name, "errors": errors})
            else:
                kept.append(row)
        document = json.loads((root / path).read_text(encoding="utf-8"))
        document["entries"] = kept
        document["snapshot_date"] = date.today().isoformat()
        errors = (validate_checkpoints if name == "vla-checkpoints" else validate_datasets)(
            document
        )
        if errors:
            raise ValueError(f"merged {name} is invalid:\n" + "\n".join(errors))
        report.append(f"{name}: {len(kept)} kept, {len(rejected)} rejected")
        rejected_out += rejected
        if write:
            (root / path).write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    for item in rejected_out:
        report.append(f"  rejected {item['id']}: {'; '.join(item['errors'])}")
    if write:
        (root / REJECTED_PATH).write_text(
            json.dumps(rejected_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (root / PAGE_PATH).write_text(render_page(*load_vla_index(root)), encoding="utf-8")
        report.append(f"wrote {CHECKPOINTS_PATH}, {DATASETS_PATH}, {REJECTED_PATH}, {PAGE_PATH}")
    return report


def counts(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda item: (-item[1], item[0])))


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _eval_text(row: dict) -> str:
    return "; ".join(
        f"{e['setting']} {e['successes']}/{e['episodes']}" for e in row.get("evaluations") or ()
    )


def render_page(checkpoints: dict, datasets: dict) -> str:
    """The generated ``candidates/vla-catalog.md``."""
    ckpts = find_vla(checkpoints["entries"])
    lines = [
        "# VLA checkpoint and dataset catalog",
        "",
        "Generated from `vla-checkpoints.json` and `vla-datasets.json` by `roboskillhub vla page --write`;"
        " do not edit by hand.",
        "Index entries are not executable bundles: a checkpoint runs only through a policy bundle's server"
        " with a checked-in profile.",
        "",
        f"Snapshot {checkpoints['snapshot_date']}: {len(ckpts)} checkpoints,"
        f" {len(datasets['entries'])} datasets.",
        "",
    ]
    compat = counts(ckpts, "yam_compatible")
    lines += [
        "**YAM compatibility:** " + ", ".join(f"{c} {compat.get(c, 0)}" for c in YAM_COMPATIBLE),
        "",
    ]
    for title, key in (("Family", "family"), ("Embodiment", "embodiment")):
        lines.append(
            f"**{title}:** " + ", ".join(f"{k} {v}" for k, v in counts(ckpts, key).items())
        )
        lines.append("")
    lines += [
        "## Checkpoints",
        "",
        "| id | rev | family | embodiment | arms | action | YAM | our server | licence (verified) | evals |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in ckpts:
        action = row.get("action") or {}
        act = "/".join(
            str(action.get(k))
            for k in ("space", "dim", "horizon", "hz")
            if action.get(k) is not None
        )
        lic = row.get("license") or {}
        lines.append(
            "| "
            + " | ".join(
                _cell(v)
                for v in (
                    row["id"],
                    row["revision"][:8],
                    row.get("family", ""),
                    row.get("embodiment", ""),
                    row.get("arms", ""),
                    act,
                    row.get("yam_compatible", ""),
                    ", ".join(our_servers(row)) or "-",
                    f"{lic.get('model', '')} ({'yes' if lic.get('verified') else 'no'})",
                    _eval_text(row) or "-",
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Datasets",
        "",
        "| id | rev | embodiment | episodes | format | licence | size GB |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in datasets["entries"]:
        lic = row.get("license")
        lic = lic.get("model") or lic.get("spdx") or "" if isinstance(lic, dict) else lic
        lines.append(
            "| "
            + " | ".join(
                _cell(v if v is not None else "")
                for v in (
                    row["id"],
                    row["revision"][:8],
                    row.get("embodiment", ""),
                    row.get("episodes", ""),
                    row.get("format", ""),
                    lic,
                    row.get("size_gb", ""),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"
