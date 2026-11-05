"""``sources.yaml``: the engine and the upstream registries, each pinned to a commit.

``roboskillhub sync`` clones them side by side under ``<cache>/sources/``, the layout
gap's own bundles assume (an open-robot-skills tool bundle resolves the engine
as ``../../../graph-as-policy``), so a synced upstream bundle installs and runs
exactly as it does beside a gap checkout.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

MANIFEST = "sources.yaml"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class Pin:
    name: str
    repo: str
    commit: str
    license: str = ""
    overlay: str = ""


@dataclass(frozen=True)
class Manifest:
    root: Path
    engine: Pin
    sources: tuple[Pin, ...]

    def source(self, name: str) -> Pin:
        for pin in self.sources:
            if pin.name == name:
                return pin
        raise KeyError(name)


def find_root(start: Path | None = None) -> Path:
    """The hub checkout: ``$ROBOSKILLHUB_ROOT``, else the nearest parent with ``sources.yaml``."""
    env = os.environ.get("ROBOSKILLHUB_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / MANIFEST).is_file():
            return candidate
    raise ManifestError(
        f"no {MANIFEST} in {here} or any parent; run inside the hub or set ROBOSKILLHUB_ROOT"
    )


def cache_dir(root: Path) -> Path:
    env = os.environ.get("ROBOSKILLHUB_CACHE")
    return Path(env).expanduser().resolve() if env else root / ".roboskillhub"


def source_dir(cache: Path, name: str) -> Path:
    return cache / "sources" / name


def _pin(entry: object, where: str) -> Pin:
    if not isinstance(entry, dict):
        raise ManifestError(f"{where}: expected a mapping")
    unknown = set(entry) - {"name", "repo", "commit", "license", "overlay"}
    if unknown:
        raise ManifestError(f"{where}: unknown keys {sorted(unknown)}")
    for key in ("name", "repo", "commit"):
        if entry.get(key) in (None, ""):
            raise ManifestError(f"{where}: missing {key!r}")
    if not isinstance(entry["commit"], str):
        raise ManifestError(f"{where}: quote the commit; YAML read {entry['commit']!r}, not a sha")
    commit = entry["commit"]
    if not _COMMIT_RE.match(commit):
        raise ManifestError(f"{where}: commit must be a full 40-character sha, got {commit!r}")
    return Pin(
        name=str(entry["name"]),
        repo=str(entry["repo"]),
        commit=commit,
        license=str(entry.get("license") or ""),
        overlay=str(entry.get("overlay") or ""),
    )


def load_manifest(root: Path) -> Manifest:
    path = root / MANIFEST
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ManifestError(f"{path}: expected a mapping")
    engine = _pin(data.get("engine"), f"{path}: engine")
    sources = tuple(
        _pin(e, f"{path}: sources[{i}]") for i, e in enumerate(data.get("sources") or [])
    )
    names = [p.name for p in sources]
    if len(set(names)) != len(names) or engine.name in names or "hub" in names:
        raise ManifestError(f"{path}: source names must be unique and not 'hub' or the engine's")
    return Manifest(root=root, engine=engine, sources=sources)


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def head_commit(path: Path) -> str:
    """The commit *path* is checked out at, or ``""`` when it is not a git checkout."""
    try:
        return _git("rev-parse", "HEAD", cwd=path)
    except (RuntimeError, FileNotFoundError, NotADirectoryError):
        return ""


def is_dirty(path: Path) -> bool:
    try:
        return bool(_git("status", "--porcelain", cwd=path))
    except (RuntimeError, FileNotFoundError, NotADirectoryError):
        return False


def checkout(repo: str, commit: str, dest: Path) -> Path:
    """*dest* holding *repo* at exactly *commit*; fetches only that commit."""
    if (dest / ".git").is_dir() and head_commit(dest) == commit:
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").is_dir():
        _git("init", "-q", cwd=dest)
        _git("remote", "add", "origin", repo, cwd=dest)
    else:
        _git("remote", "set-url", "origin", repo, cwd=dest)
    _git("fetch", "-q", "--depth", "1", "origin", commit, cwd=dest)
    _git("checkout", "-q", "--force", "--detach", "FETCH_HEAD", cwd=dest)
    got = head_commit(dest)
    if got != commit:
        raise RuntimeError(f"{dest}: checked out {got}, expected {commit}")
    return dest


def sync(manifest: Manifest, cache: Path, *, engine: bool = True) -> dict[str, Path]:
    """Check out every pinned source (and the engine) under ``<cache>/sources``."""
    out: dict[str, Path] = {}
    pins = (*manifest.sources, manifest.engine) if engine else manifest.sources
    for pin in pins:
        out[pin.name] = checkout(pin.repo, pin.commit, source_dir(cache, pin.name))
    return out
