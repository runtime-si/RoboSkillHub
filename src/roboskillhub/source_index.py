"""Read-only discovery of upstream sources, separate from executable Hub bundles."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

INDEX_PATH = Path("candidates/pick-place-index.json")
LICENSE_STATUSES = ("permissive", "copyleft", "research-only", "unresolved")
REVIEW_LEVELS = ("metadata-indexed", "implementation-inspected")
RECORD_TYPES = ("implementation", "task-expert", "tool", "policy", "recipe")
DESTINATIONS = ("tool", "skill", "policy", "recipe")
PRIORITIES = ("P1", "P2", "P3")
_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _http(value: object) -> bool:
    if not _text(value) or any(char.isspace() for char in value):
        return False
    try:
        parsed = urlsplit(value)
        return parsed.scheme in ("https", "http") and bool(parsed.hostname)
    except ValueError:
        return False


def validate_source_index(data: object) -> list[str]:
    """Check index integrity, never infer upstream execution or license clearance."""
    if not isinstance(data, dict):
        return ["source index must be an object"]
    errors: list[str] = []
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        errors.append("source index must have schema_version: 1")
    if data.get("collection") != "pick-place":
        errors.append("source index collection must be pick-place")
    for key in ("scope", "counting_policy"):
        if not _text(data.get(key)):
            errors.append(f"{key} must be a nonempty string")
    snapshot = data.get("snapshot_date")
    try:
        if not isinstance(snapshot, str) or date.fromisoformat(snapshot).isoformat() != snapshot:
            raise ValueError
    except ValueError:
        errors.append("snapshot_date must be an ISO YYYY-MM-DD date")
    rows = data.get("entries")
    if not isinstance(rows, list):
        return errors + ["entries must be a list"]
    ids: set[str] = set()
    dedup_keys: set[str] = set()
    for index, row in enumerate(rows):
        where = f"entries[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{where}: expected an object")
            continue
        ident = row.get("id")
        if not isinstance(ident, str) or not _ID.fullmatch(ident):
            errors.append(f"{where}: id must be a lowercase slug")
        elif ident in ids:
            errors.append(f"{where}: duplicate id {ident}")
        else:
            ids.add(ident)
        key = row.get("dedup_key")
        if not _text(key):
            errors.append(f"{where}: dedup_key must be a nonempty string")
        elif key.strip().casefold() in dedup_keys:
            errors.append(f"{where}: duplicate dedup_key {key}")
        else:
            dedup_keys.add(key.strip().casefold())
        for key in ("name", "category", "summary"):
            if not _text(row.get(key)):
                errors.append(f"{where}: {key} must be a nonempty string")
        for key in ("repository_url", "source_url"):
            if not _http(row.get(key)):
                errors.append(f"{where}: {key} must be an HTTP(S) URL")
        revision = row.get("source_revision")
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            errors.append(f"{where}: source_revision must be a full 40-character commit")
        for key, choices in (
            ("record_type", RECORD_TYPES),
            ("review_level", REVIEW_LEVELS),
            ("hub_destination", DESTINATIONS),
            ("priority", PRIORITIES),
        ):
            if not isinstance(row.get(key), str) or row[key] not in choices:
                errors.append(f"{where}: unknown {key}")
        for key in ("capabilities", "adaptation_required", "entrypoints"):
            values = row.get(key)
            if (
                not isinstance(values, list)
                or (key != "entrypoints" and not values)
                or any(not _text(value) for value in values)
            ):
                errors.append(f"{where}: {key} must be a list of nonempty strings")
        if row.get("invocable_in_hub") is not False:
            errors.append(f"{where}: indexed sources must have invocable_in_hub=false")
        for key in ("readiness", "maturity", "ready", "executable", "runnable"):
            if key in row:
                errors.append(f"{where}: {key} is not an index claim; use review_level")
        license_record = row.get("license")
        if not isinstance(license_record, dict):
            errors.append(f"{where}: license must be an object")
        else:
            status = license_record.get("status")
            if not isinstance(status, str) or status not in LICENSE_STATUSES:
                errors.append(f"{where}: unknown license.status")
            spdx = license_record.get("spdx")
            if "spdx" not in license_record or not (
                _text(spdx) or (spdx is None and status == "unresolved")
            ):
                errors.append(f"{where}: license.spdx must be text, or null if unresolved")
            if not _text(license_record.get("notes")):
                errors.append(f"{where}: license.notes must be a nonempty string")
            url = license_record.get("url")
            if "url" not in license_record or (url is not None and not _http(url)):
                errors.append(f"{where}: license.url must be an HTTP(S) URL or explicit null")
            if status in ("permissive", "copyleft") and not _http(url):
                errors.append(f"{where}: {status} code license requires a license URL")
            if status in ("permissive", "copyleft") and spdx in ("UNKNOWN", "NOASSERTION"):
                errors.append(f"{where}: unresolved SPDX cannot claim {status}")
        evidence = row.get("evidence")
        has_source = False
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{where}: evidence must be a nonempty list")
        else:
            for number, item in enumerate(evidence):
                position = f"{where}.evidence[{number}]"
                if not isinstance(item, dict):
                    errors.append(f"{position}: expected an object")
                    continue
                if not _http(item.get("url")):
                    errors.append(f"{position}: url must be an HTTP(S) URL")
                if item.get("kind") not in ("readme", "source", "license"):
                    errors.append(f"{position}: unknown kind")
                digest = item.get("sha256")
                if "sha256" not in item or not (
                    digest is None or (isinstance(digest, str) and _DIGEST.fullmatch(digest))
                ):
                    errors.append(f"{position}: sha256 must be a full digest or explicit null")
                has_source |= item.get("kind") == "source" and _http(item.get("url"))
        if row.get("review_level") == "implementation-inspected" and (
            not row.get("entrypoints") or not has_source
        ):
            errors.append(
                f"{where}: implementation-inspected requires entrypoints and source evidence"
            )
    return errors


def load_source_index(root: Path) -> dict:
    data = json.loads((root / INDEX_PATH).read_text(encoding="utf-8"))
    errors = validate_source_index(data)
    if errors:
        raise ValueError("invalid source index:\n" + "\n".join(errors))
    return data


def find_sources(
    data: dict,
    query: str = "",
    *,
    category: str | None = None,
    license_status: str | None = None,
    review_level: str | None = None,
) -> list[dict]:
    """Filter without ranking; all whitespace-separated query terms must match."""
    terms = query.casefold().split()
    rows = []
    for row in data["entries"]:
        if category is not None and row["category"] != category:
            continue
        if license_status is not None and row["license"]["status"] != license_status:
            continue
        if review_level is not None and row["review_level"] != review_level:
            continue
        label = " ".join(
            [
                row["id"],
                row["name"],
                row["repository_url"],
                row["category"],
                row["summary"],
                *row["capabilities"],
            ]
        ).casefold()
        if all(term in label for term in terms):
            rows.append(row)
    return rows
