"""Documents with a YAML block between two ``---`` lines, then Markdown.

``SKILL.md``, ``RECIPE.md`` and ``PAPER.md`` all share this shape, which is the
Agent Skills one gap reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class FrontmatterError(ValueError):
    """The document has no frontmatter, or it is not a YAML mapping."""


def split(text: str, *, source: str = "<text>") -> tuple[dict[str, Any], str]:
    """Return ``(frontmatter, body)`` for one document."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise FrontmatterError(f"{source}: does not begin with a '---' line")
    for end in range(1, len(lines)):
        if lines[end].strip() == "---":
            break
    else:
        raise FrontmatterError(f"{source}: frontmatter is not closed by a second '---' line")
    try:
        data = yaml.safe_load("".join(lines[1:end])) or {}
    except yaml.YAMLError as exc:
        raise FrontmatterError(f"{source}: frontmatter is not valid YAML ({exc})") from exc
    if not isinstance(data, dict):
        raise FrontmatterError(f"{source}: frontmatter must be a mapping, not {type(data).__name__}")
    return data, "".join(lines[end + 1 :])


def read(path: Path) -> tuple[dict[str, Any], str]:
    return split(path.read_text(encoding="utf-8"), source=str(path))
