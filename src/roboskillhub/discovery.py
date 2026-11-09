"""Search reviewed upstream material without treating it as a runtime bundle."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

CATALOG_PATH = Path("candidates/catalog.json")
DESTINATIONS = ("tool", "skill", "policy", "recipe")
PRIORITIES = ("P1", "P2", "P3")
_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SHA = re.compile(r"^[0-9a-f]{40}$")


def _url(value: object, schemes: tuple[str, ...]) -> bool:
    if not isinstance(value, str) or any(char.isspace() for char in value):
        return False
    try:
        parsed = urlparse(value)
        return parsed.scheme in schemes and bool(parsed.hostname)
    except ValueError:
        return False


def _http(value: object) -> bool:
    return _url(value, ("http", "https"))


def validate_candidates(data: object) -> list[str]:
    """Validate intake metadata, not upstream code, licenses or physical ability."""
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        return ["candidate catalog must have schema_version: 1"]
    rows = data.get("candidates")
    if not isinstance(rows, list):
        return ["candidates must be a list"]
    errors: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        where = f"candidates[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{where}: expected an object")
            continue
        name = row.get("id")
        if not isinstance(name, str) or not _ID.fullmatch(name):
            errors.append(f"{where}: invalid id")
        elif name in seen:
            errors.append(f"{where}: duplicate id {name}")
        else:
            seen.add(name)
        for key in (
            "name",
            "category",
            "repo",
            "artifact_type",
            "inputs",
            "outputs",
            "sim_privilege",
            "existing_overlap",
            "verified_date",
        ):
            if not isinstance(row.get(key), str) or not row[key].strip():
                errors.append(f"{where}: {key} must be a nonempty string")
        for key in (
            "capabilities",
            "entrypoints",
            "robots",
            "sensors",
            "adaptation_steps",
            "limitations",
            "evidence_urls",
        ):
            value = row.get(key)
            if (
                not isinstance(value, list)
                or not value
                or any(not isinstance(v, str) or not v.strip() for v in value)
            ):
                errors.append(f"{where}: {key} must be a nonempty list of strings")
        if not _http(row.get("source_url")):
            errors.append(f"{where}: source_url must be an HTTP(S) URL")
        evidence = row.get("evidence_urls")
        if isinstance(evidence, list) and any(not _http(v) for v in evidence):
            errors.append(f"{where}: evidence_urls must contain HTTP(S) URLs")
        revision = row.get("source_revision")
        if "source_revision" not in row:
            errors.append(f"{where}: source_revision is required (explicit null is allowed)")
        elif revision is not None and (
            not isinstance(revision, str) or not _SHA.fullmatch(revision)
        ):
            errors.append(f"{where}: source_revision must be a full commit or null")
        if row.get("hub_destination") not in DESTINATIONS:
            errors.append(f"{where}: unknown hub_destination")
        if row.get("priority") not in PRIORITIES:
            errors.append(f"{where}: unknown priority")
        group = row.get("intake_group")
        if not isinstance(group, str) or group not in {"open-source-code", "research-only"}:
            errors.append(f"{where}: unknown intake_group")
        if row.get("readiness") != "source-reviewed" or row.get("invocable_in_hub") is not False:
            errors.append(f"{where}: intake records must be source-reviewed and not invocable")
        for key, fields in (
            ("code_license", ("id", "url", "notes")),
            ("weights", ("availability", "url", "license", "notes")),
        ):
            value = row.get(key)
            if not isinstance(value, dict) or any(field not in value for field in fields):
                errors.append(f"{where}: incomplete {key} record")
                continue
            for field in fields:
                if field != "url" and (
                    not isinstance(value[field], str) or not value[field].strip()
                ):
                    errors.append(f"{where}: {key}.{field} must be a nonempty string")
            if key == "code_license" and not _http(value["url"]):
                errors.append(f"{where}: code_license.url must be an HTTP(S) URL")
            elif (
                key == "weights"
                and value["url"] is not None
                and not _url(value["url"], ("http", "https", "gs"))
            ):
                errors.append(f"{where}: weights.url must be an HTTP(S) or gs URL, or null")
    return errors


def load_candidates(root: Path) -> dict:
    data = json.loads((root / CATALOG_PATH).read_text(encoding="utf-8"))
    errors = validate_candidates(data)
    if errors:
        raise ValueError("invalid candidate catalog:\n" + "\n".join(errors))
    return data


def discover(
    data: dict,
    query: str = "",
    *,
    category: str | None = None,
    destination: str | None = None,
    priority: str | None = None,
    include_research_only: bool = False,
) -> list[dict]:
    """Case-insensitive keyword discovery (OR), with exact categorical filters.

    Chinese capability words are retained. This is source discovery, not semantic
    planning, task-coverage analysis, a dependency resolver, or permission to run.
    """
    terms = query.casefold().split()
    ranked: list[tuple[int, dict]] = []
    for row in data["candidates"]:
        if row["intake_group"] == "research-only" and not include_research_only:
            continue
        if category is not None and row["category"] != category:
            continue
        if destination is not None and row["hub_destination"] != destination:
            continue
        if priority is not None and row["priority"] != priority:
            continue
        label = " ".join(
            [row["id"], row["name"], row["category"], row["repo"], *row["capabilities"]]
        ).casefold()
        detail = " ".join(
            [row["inputs"], row["outputs"], *row["entrypoints"], *row["robots"]]
        ).casefold()
        score = sum(3 * (term in label) + (term in detail) for term in terms)
        if terms and not score:
            continue
        ranked.append((score, row))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]["priority"], pair[1]["id"]))
    return [row for _, row in ranked]
