"""molmoact2-yam: weight probes for ``gap check`` / ``gap fetch``. The policy runs in ``server.py``; graphs
reach it through policies/gr00t's client (``gr00t.act`` / ``gr00t.run_skill`` with ``GAP_GR00T_URL``)."""

from __future__ import annotations

MODEL_REPO = "allenai/MolmoAct2-BimanualYAM"
MODEL_REVISION = "8dcbed66f2380e4393189c303ea72488eb9e63c2"
_FILES = (
    "config.json",
    "norm_stats.json",
    "modeling_molmoact2.py",
    "model.safetensors.index.json",
) + tuple(f"model-{i:05d}-of-00005.safetensors" for i in range(1, 6))


def weights_cached() -> bool | None:
    """Filesystem-only probe: the pinned snapshot's config, norm stats, code and shards are on disk."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        hits = [try_to_load_from_cache(MODEL_REPO, f, revision=MODEL_REVISION) for f in _FILES]
    except Exception:
        return None
    return all(isinstance(h, str) for h in hits)


def prefetch() -> str:
    """Download the pinned snapshot (~22 GB); returns its directory."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION)
