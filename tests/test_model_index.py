from __future__ import annotations

import json

import pytest
from conftest import REPO

from roboskillhub.catalog import load_catalog
from roboskillhub.cli import main
from roboskillhub.model_index import MODELS_PATH, find_models, load_model_index, validate_models


def model(ident="org/zh-asr", **changes):
    row = {
        "id": ident,
        "revision": "a" * 40,
        "name": "Wav2Vec2 中文 ASR",
        "domain": "语音与音频",
        "category": "中文语音识别",
        "task": "输入中文语音，输出中文转写文本。",
        "pipeline_tag": "automatic-speech-recognition",
        "base_model": "org/wav2vec2-base",
        "method": "CTC 监督微调",
        "form": "full",
        "weight_type": "完整权重",
        "params": "315M",
        "weight_files": ["model.safetensors"],
        "size_gb": 1.26,
        "license": "Apache-2.0",
        "card_license": "apache-2.0",
        "gated": False,
        "notes": "音频需重采样为 16 kHz",
        "evidence": "model card",
        "source_urls": [f"https://huggingface.co/{ident}"],
        "checked_at": "2026-11-11",
    }
    row.update(changes)
    return row


def document(entries):
    return {
        "schema_version": 1,
        "collection": "finetuned-models",
        "scope": "test",
        "snapshot_date": "2026-11-11",
        "entries": entries,
    }


@pytest.fixture
def rows():
    reranker = model(
        "org/reranker",
        name="Reranker",
        domain="检索与评分",
        category="多语言检索重排",
        task="输入查询与候选文档对，输出相关性分数。",
        pipeline_tag="text-ranking",
        notes="",
    )
    lora = model(
        "org/zh-lora",
        name="Chinese LoRA",
        domain="大语言模型",
        category="中文对话 / 指令",
        task="输入中文指令，输出助手回复。",
        pipeline_tag=None,
        form="adapter",
        card_license=None,
        size_gb=None,
        notes="不能独立运行",
    )
    return [model(), reranker, lora]


@pytest.fixture
def models_root(tmp_path, rows):
    (tmp_path / "candidates").mkdir()
    (tmp_path / MODELS_PATH).write_text(json.dumps(document(rows)), encoding="utf-8")
    return tmp_path


def test_checked_in_catalog_is_valid_pinned_and_not_in_the_registry(capsys):
    entries = load_model_index(REPO)["entries"]
    assert len(entries) >= 36
    assert main(["--root", str(REPO), "check-sources"]) == 0
    assert f"{len(entries)} finetuned-models: metadata valid" in capsys.readouterr().out
    ids = {row["id"] for row in entries}
    assert {"BAAI/bge-reranker-v2-m3", "llava-hf/llava-1.5-7b-hf"} <= ids
    assert not ids & set(load_catalog(REPO).bundles)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"revision": "main"}, "revision must be a full 40-character commit"),
        ({"id": "not a repo"}, "id must be a Hugging Face repo id"),
        ({"form": "lora"}, "form must be one of full, adapter"),
        ({"weight_files": []}, "weight_files must be a nonempty list of file names"),
        ({"weight_files": "model.safetensors"}, "weight_files must be a nonempty list"),
        ({"card_license": ""}, "card_license must be a nonempty string or null"),
        ({"pipeline_tag": 3}, "pipeline_tag must be a nonempty string or null"),
        ({"license": ""}, "license must be a nonempty string"),
        ({"task": " "}, "task must be a nonempty string"),
        ({"gated": "false"}, "gated must be true or false"),
        ({"notes": None}, "notes must be a string"),
        ({"size_gb": "1.2"}, "size_gb must be a number or null"),
        ({"source_urls": []}, "source_urls must be a nonempty list"),
        ({"checked_at": "yesterday"}, "checked_at must be an ISO date"),
    ],
)
def test_validator_rejects(change, message):
    errors = validate_models(document([model(**change)]))
    assert any(message in e for e in errors), errors


@pytest.mark.parametrize(
    "field", ["revision", "form", "weight_files", "license", "card_license", "checked_at"]
)
def test_every_schema_field_is_required(field):
    row = model()
    del row[field]
    assert f"entries[0]: missing {field}" in validate_models(document([row]))


def test_optional_fields_duplicates_and_header(rows):
    assert validate_models(document(rows)) == []
    bare = model()
    del bare["pipeline_tag"], bare["size_gb"]
    assert validate_models(document([bare])) == []
    assert any(
        "duplicate id" in e for e in validate_models(document([model("Org/M"), model("org/m")]))
    )
    assert validate_models([]) == ["finetuned-models must be an object"]
    wrong = document([model()]) | {"collection": "vla-checkpoints"}
    assert validate_models(wrong) == ["finetuned-models: collection must be finetuned-models"]
    assert validate_models(document(["org/zh-asr"])) == ["entries[0]: expected an object"]


def test_load_reports_every_problem(models_root, rows):
    assert [row["id"] for row in load_model_index(models_root)["entries"]] == [
        "org/zh-asr",
        "org/reranker",
        "org/zh-lora",
    ]
    rows[0]["revision"] = "main"
    rows[1]["form"] = "lora"
    (models_root / MODELS_PATH).write_text(json.dumps(document(rows)), encoding="utf-8")
    with pytest.raises(ValueError, match=r"entries\[0\]: revision(.|\n)*entries\[1\]: form"):
        load_model_index(models_root)


def test_search_ranks_by_matched_terms_in_either_language(rows):
    assert [row["id"] for row in find_models(rows, "语音识别")] == ["org/zh-asr"]
    assert [row["id"] for row in find_models(rows, "TEXT-RANKING")] == ["org/reranker"]
    assert [row["id"] for row in find_models(rows, "adapter")] == ["org/zh-lora"]
    # Any term counts; more matched terms rank first, ties keep catalog order.
    ranked = [row["id"] for row in find_models(rows, "中文 指令")]
    assert ranked == ["org/zh-lora", "org/zh-asr"]
    assert find_models(rows, "nothing-matches") == []
    assert find_models(rows) == rows


def test_cli_sources_kind_models(models_root, capsys):
    args = ["--root", str(models_root), "sources", "--kind", "models"]
    assert main(args + ["重排", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["collection"] == "finetuned-models" and out["invocable_in_hub"] is False
    assert (out["matched_count"], out["returned_count"]) == (1, 1)
    assert out["entries"][0]["id"] == "org/reranker"
    assert main(args) == 0
    text = capsys.readouterr().out
    assert "not bundles" in text and "3 matched; showing 3" in text
    assert "adapter  org/zh-lora@aaaaaaaa  [大语言模型: 中文对话 / 指令]" in text
    assert "licence: Apache-2.0; 1.26 GB" in text
    assert main(args + ["--limit", "1"]) == 0
    assert "3 matched; showing 1" in capsys.readouterr().out
    assert main(args + ["nothing-matches"]) == 1
