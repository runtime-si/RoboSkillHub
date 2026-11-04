from __future__ import annotations

import os
import uuid

import pytest
from conftest import REPO

from robomotif import weights

SHA = "a" * 40


def test_uri_must_pin_a_commit_when_asked():
    ref = weights.parse_uri(f"hf://org/model@{SHA}", require_revision=True)
    assert (ref.repo_id, ref.revision, ref.repo_type) == ("org/model", SHA, "model")
    assert weights.parse_uri(f"hf://datasets/org/data@{SHA}").repo_type == "dataset"
    assert weights.parse_uri("hf://org/model").revision == "main"
    with pytest.raises(ValueError, match="must pin a full 40-character commit"):
        weights.parse_uri("hf://org/model@main", require_revision=True)
    with pytest.raises(ValueError):
        weights.parse_uri("s3://bucket/key")


def test_manifest_digest_ignores_hub_bookkeeping(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 10)
    (tmp_path / "config.json").write_text("{}")
    digest = weights.manifest_sha256(weights.manifest(tmp_path))
    (tmp_path / ".gitattributes").write_text("*.safetensors filter=lfs\n")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "x").write_text("y")
    assert weights.manifest_sha256(weights.manifest(tmp_path)) == digest
    (tmp_path / "config.json").write_text('{"a": 1}')
    assert weights.manifest_sha256(weights.manifest(tmp_path)) != digest


def test_push_result_is_written_into_the_policy_template(tmp_path):
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text((REPO / "templates/policy/SKILL.md").read_text())
    pushed = weights.Pushed(weights.HfRef("me/ckpt", SHA), "b" * 64, {})
    assert weights.write_into_skill_md(skill_md, pushed)
    text = skill_md.read_text()
    assert text.count(f"hf://me/ckpt@{SHA}") == 2
    assert f"sha256: {'b' * 64}" in text and "OWNER/REPO" not in text


def test_server_template_digest_matches_motif(tmp_path, monkeypatch):
    """The policy server re-derives the digest in its own venv; it must agree."""
    import importlib.util
    import sys
    import types

    # The server's own venv has these; this one need not.
    for name in ("websockets", "openpi_client"):
        if importlib.util.find_spec(name) is None:
            stub = types.ModuleType(name)
            stub.msgpack_numpy = types.ModuleType("msgpack_numpy")
            monkeypatch.setitem(sys.modules, name, stub)
    spec = importlib.util.spec_from_file_location("policy_server", REPO / "templates/policy/server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "w.bin").write_bytes(b"123")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / ".gitattributes").write_text("x")
    assert module.snapshot_sha256(tmp_path) == weights.manifest_sha256(weights.manifest(tmp_path))


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="needs HF_TOKEN")
def test_round_trip_through_the_hub(tmp_path):
    from huggingface_hub import HfApi

    api = HfApi()
    repo = f"{api.whoami()['name']}/motif-roundtrip-{uuid.uuid4().hex[:8]}"
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "weights.bin").write_bytes(os.urandom(4096))
    (tmp_path / "ckpt" / "config.json").write_text('{"hidden": 8}')
    try:
        pushed = weights.push(tmp_path / "ckpt", repo, private=True)
        assert len(pushed.ref.revision) == 40
        path = weights.pull(pushed.ref.uri, sha256=pushed.sha256, cache_dir=tmp_path / "cache")
        assert (path / "weights.bin").read_bytes() == (tmp_path / "ckpt" / "weights.bin").read_bytes()
        with pytest.raises(ValueError, match="does not match"):
            weights.pull(pushed.ref.uri, sha256="0" * 64, cache_dir=tmp_path / "cache")
    finally:
        api.delete_repo(repo, missing_ok=True)
