"""Read-only catalog of fine-tuned models that are not robot action models.

Language models, text classifiers, vision and multimodal models, speech and audio models, embedders and
rerankers. An entry pins a Hugging Face revision and records what the model was tuned from and for, its weight
files and the licence its card states. Like ``vla_index``, nothing here is an executable Hub bundle: using a
model still means writing a tool bundle that pins and serves it. Robot action checkpoints belong in
``vla_index``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .vla_index import _common, _header, _iso_date, _text

MODELS_PATH = Path("candidates/finetuned-models.json")
COLLECTION = "finetuned-models"
FORMS = ("full", "adapter")
MODEL_FIELDS = (
    "id",
    "revision",
    "name",
    "domain",
    "category",
    "task",
    "base_model",
    "method",
    "form",
    "weight_type",
    "params",
    "weight_files",
    "license",
    "card_license",
    "gated",
    "notes",
    "evidence",
    "source_urls",
    "checked_at",
)
_NONEMPTY = (
    "name",
    "domain",
    "category",
    "task",
    "base_model",
    "method",
    "weight_type",
    "params",
    "license",
    "evidence",
)
_SEARCHED = (
    "id",
    "name",
    "domain",
    "category",
    "task",
    "pipeline_tag",
    "base_model",
    "method",
    "form",
    "weight_type",
    "license",
    "notes",
)


def validate_models(data: object) -> list[str]:
    """Required fields, pinned revisions, listed weight files, a known form."""
    errors, rows = _header(data, COLLECTION)
    if rows is None:
        return errors
    ids: set[str] = set()
    for index, row in enumerate(rows):
        where = f"entries[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{where}: expected an object")
            continue
        errors += _common(row, where, MODEL_FIELDS, ids)
        for key in _NONEMPTY:
            if key in row and not _text(row[key]):
                errors.append(f"{where}: {key} must be a nonempty string")
        if "form" in row and row["form"] not in FORMS:
            errors.append(f"{where}: form must be one of {', '.join(FORMS)}")
        files = row.get("weight_files")
        if "weight_files" in row and (
            not isinstance(files, list) or not files or not all(_text(name) for name in files)
        ):
            errors.append(f"{where}: weight_files must be a nonempty list of file names")
        for key in ("card_license", "pipeline_tag"):
            if key in row and row[key] is not None and not _text(row[key]):
                errors.append(f"{where}: {key} must be a nonempty string or null")
        if "gated" in row and not isinstance(row["gated"], bool):
            errors.append(f"{where}: gated must be true or false")
        if "notes" in row and not isinstance(row["notes"], str):
            errors.append(f"{where}: notes must be a string")
        if "checked_at" in row and not _iso_date(row["checked_at"]):
            errors.append(f"{where}: checked_at must be an ISO date")
    return errors


def load_model_index(root: Path) -> dict:
    """The collection, or ValueError listing every problem."""
    data = json.loads((root / MODELS_PATH).read_text(encoding="utf-8"))
    errors = validate_models(data)
    if errors:
        raise ValueError(f"invalid {MODELS_PATH}:\n" + "\n".join(errors))
    return data


def find_models(rows: list[dict], query: str = "") -> list[dict]:
    """Rank by matched query terms (any term counts, English or Chinese), then catalog order."""
    terms = query.casefold().split()
    scored = []
    for order, row in enumerate(rows):
        label = " ".join(str(row.get(key) or "") for key in _SEARCHED).casefold()
        hits = sum(term in label for term in terms)
        if terms and not hits:
            continue
        scored.append((-hits, order, row))
    return [item[-1] for item in sorted(scored, key=lambda item: item[:2])]
