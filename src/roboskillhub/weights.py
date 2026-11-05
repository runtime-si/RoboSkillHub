"""Checkpoints on the Hugging Face Hub, addressed by an immutable ``hf://`` URI.

A bundle never names weights by a branch: ``hf://<owner>/<repo>@<commit>`` pins
the exact upload, and ``sha256`` -- one digest over the snapshot's file list and
every file's content -- lets whoever downloads it confirm they got that upload.
Weights live on the Hub, not in git: the bundle carries only these two strings.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .model import file_sha256

_URI_RE = re.compile(
    r"^hf://(?:(?P<type>models|datasets)/)?(?P<repo>[\w.-]+/[\w.-]+)(?:@(?P<rev>[\w.-]+))?$"
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_IGNORED = frozenset({".gitattributes"})


@dataclass(frozen=True)
class HfRef:
    repo_id: str
    revision: str
    repo_type: str = "model"

    @property
    def uri(self) -> str:
        prefix = "datasets/" if self.repo_type == "dataset" else ""
        return f"hf://{prefix}{self.repo_id}@{self.revision}"


def parse_uri(uri: str, *, require_revision: bool = False) -> HfRef:
    match = _URI_RE.match(uri)
    if not match:
        raise ValueError(f"{uri!r} is not hf://<owner>/<repo>@<commit>")
    revision = match.group("rev") or ""
    if require_revision and not _COMMIT_RE.match(revision):
        raise ValueError(
            f"{uri!r} must pin a full 40-character commit after '@', not a branch or tag"
        )
    repo_type = "dataset" if match.group("type") == "datasets" else "model"
    return HfRef(match.group("repo"), revision or "main", repo_type)


def manifest(root: Path) -> dict[str, str]:
    """``{relative path: sha256}`` for every file a snapshot carries."""
    out = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if not path.is_file() or rel in _IGNORED or rel.split("/", 1)[0] in {".cache", ".git"}:
            continue
        out[rel] = file_sha256(path)
    return out


def manifest_sha256(files: dict[str, str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for rel in sorted(files):
        digest.update(f"{rel}\0{files[rel]}\n".encode())
    return digest.hexdigest()


def _api(token: str | None = None):
    from huggingface_hub import HfApi

    return HfApi(token=token or os.environ.get("HF_TOKEN"))


@dataclass(frozen=True)
class Pushed:
    ref: HfRef
    sha256: str
    files: dict[str, str]


def push(
    local_dir: Path,
    repo_id: str,
    *,
    private: bool = True,
    message: str = "Upload checkpoint",
    token: str | None = None,
) -> Pushed:
    """Upload *local_dir* as one commit; return the URI pinned to that commit."""
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise FileNotFoundError(f"{local_dir} is not a directory")
    files = manifest(local_dir)
    if not files:
        raise ValueError(f"{local_dir} has no files to upload")
    api = _api(token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    # delete_patterns makes the commit mirror the folder exactly, so the digest
    # computed here is the digest of the snapshot anyone pulls at that commit.
    commit = api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=message,
        ignore_patterns=[".git/*", ".cache/*"],
        delete_patterns=["*"],
    )
    revision = getattr(commit, "oid", None) or api.repo_info(repo_id).sha
    return Pushed(HfRef(repo_id, revision), manifest_sha256(files), files)


def pull(
    uri: str, *, sha256: str | None = None, token: str | None = None, cache_dir: Path | None = None
) -> Path:
    """Download the pinned snapshot and, when *sha256* is given, prove it is that upload."""
    from huggingface_hub import snapshot_download

    ref = parse_uri(uri, require_revision=sha256 is not None)
    path = Path(
        snapshot_download(
            repo_id=ref.repo_id,
            revision=ref.revision,
            repo_type=ref.repo_type,
            token=token or os.environ.get("HF_TOKEN"),
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )
    if sha256 is not None:
        got = manifest_sha256(manifest(path))
        if got != sha256:
            raise ValueError(f"{uri}: snapshot digest {got} does not match the declared {sha256}")
    return path


def write_into_skill_md(skill_md: Path, pushed: Pushed) -> bool:
    """Replace the template's ``weights_uri:`` and ``sha256:`` lines in place.

    Returns False when the file has neither line; the caller then prints the
    snippet instead of rewriting YAML it did not lay out.
    """
    text = skill_md.read_text(encoding="utf-8")
    new, n_uri = re.subn(
        r"(?m)^(\s*weights_uri:\s*).*$", lambda m: m.group(1) + pushed.ref.uri, text
    )
    new, n_uri_hub = re.subn(
        r"(?m)^(\s*uri:\s*)hf://.*$", lambda m: m.group(1) + pushed.ref.uri, new
    )
    new, n_sha = re.subn(r"(?m)^(\s*sha256:\s*).*$", lambda m: m.group(1) + pushed.sha256, new)
    if not (n_uri or n_uri_hub) or not n_sha:
        return False
    skill_md.write_text(new, encoding="utf-8")
    return True
