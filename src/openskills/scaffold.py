"""``openskills new``: start a bundle, recipe or paper card from its template."""

from __future__ import annotations

import shutil
from pathlib import Path

from .model import NAME_RE

DESTINATIONS = {
    "tool": "tools",
    "skill": "skills",
    "policy": "policies",
    "recipe": "recipes",
    "paper": "papers",
}


def new(root: Path, kind: str, name: str) -> Path:
    if kind not in DESTINATIONS:
        raise ValueError(f"kind must be one of {sorted(DESTINATIONS)}")
    if not NAME_RE.match(name) or len(name) > 64:
        raise ValueError(f"{name!r}: lowercase letters and digits joined by single hyphens, at most 64 chars")
    template = root / "templates" / kind
    dest = root / DESTINATIONS[kind] / name
    if dest.exists():
        raise FileExistsError(f"{dest} already exists")
    shutil.copytree(template, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    for path in dest.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if "{{name}}" in text or "{{module}}" in text:
                path.write_text(
                    text.replace("{{name}}", name).replace("{{module}}", name.replace("-", "_")),
                    encoding="utf-8",
                )
    return dest
