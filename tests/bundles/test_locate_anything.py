"""Tests for the locate-anything tool bundle -- the worker mocked, no torch, no download.

The model is faked through the bundle's ``_get_worker`` seam with an object
that records each call and returns a canned answer in the model's token format.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest
from gap.skills import load_skills
from gap_core.errors import ToolError

from .conftest import import_served_tools

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "tools" / "locate-anything"


@pytest.fixture(scope="module")
def la():
    """The locate-anything bundle's tools module, loaded through the engine."""
    reg = load_skills(ROOT, only=["locate-anything"])
    import_served_tools(reg)
    return reg.get("locate-anything").tools_module


@pytest.fixture()
def image() -> np.ndarray:
    # 480 high x 640 wide -- pixel-conversion assertions depend on this.
    return np.zeros((480, 640, 3), dtype=np.uint8)


class _FakeWorker:
    def __init__(self, answer: str):
        self.answer = answer
        self.calls: list[tuple] = []

    def _reply(self, method, image, arg, **kwargs):
        self.calls.append((method, image, arg, kwargs))
        return {"answer": self.answer}

    def detect(self, image, categories, **kwargs):
        return self._reply("detect", image, categories, **kwargs)

    def ground_multi(self, image, phrase, **kwargs):
        return self._reply("ground_multi", image, phrase, **kwargs)

    def ground_single(self, image, phrase, **kwargs):
        return self._reply("ground_single", image, phrase, **kwargs)

    def point(self, image, phrase, **kwargs):
        return self._reply("point", image, phrase, **kwargs)


@pytest.fixture()
def acked(monkeypatch):
    monkeypatch.setenv("GAP_LOCATE_ANYTHING_NONCOMMERCIAL", "1")


def _patch(la, monkeypatch, answer: str) -> _FakeWorker:
    worker = _FakeWorker(answer)
    monkeypatch.setattr(la, "_get_worker", lambda: worker)
    return worker


# ---------------------------------------------------------------------------
# Licence guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "0", "no", "commercial"])
def test_every_tool_refuses_without_the_non_commercial_acknowledgement(
    la, image, monkeypatch, value
):
    if value is None:
        monkeypatch.delenv("GAP_LOCATE_ANYTHING_NONCOMMERCIAL", raising=False)
    else:
        monkeypatch.setenv("GAP_LOCATE_ANYTHING_NONCOMMERCIAL", value)
    worker = _patch(la, monkeypatch, "<ref>x</ref><box><1><2><3><4></box>")
    for call in (
        lambda: la.detect(image=image, categories=["cube"]),
        lambda: la.ground(image=image, query="cube"),
        lambda: la.point(image=image, query="cube"),
    ):
        with pytest.raises(ToolError, match="non-commercial"):
            call()
    assert worker.calls == []  # never reaches the model


def test_the_bundle_declares_the_acknowledgement_and_the_licence():
    reg = load_skills(ROOT, only=["locate-anything"])
    meta = reg.get("locate-anything").meta
    assert meta.requires.env == ["GAP_LOCATE_ANYTHING_NONCOMMERCIAL"]
    assert meta.requires.gpu and meta.requires.weights
    text = (BUNDLE / "SKILL.md").read_text()
    assert "non-commercial" in text.split("---")[1]  # in the frontmatter's license/notes
    assert "hf://nvidia/LocateAnything-3B@c32291ca5e996f5a7a485845b4f57a233936bba0" in text


# ---------------------------------------------------------------------------
# Grounding: drop-in for gemini-er.detect
# ---------------------------------------------------------------------------


def test_ground_converts_x_y_boxes_to_pixels(la, image, monkeypatch, acked):
    worker = _patch(
        la,
        monkeypatch,
        "<ref>small dark grey cube</ref><box><431><633><514><748></box><|im_end|>",
    )

    out = la.ground(image=image, query="small dark grey cube")

    (det,) = out["detections"]
    # [x1, y1, x2, y2] = [431, 633, 514, 748] / 1000 on a 640 x 480 frame:
    assert det["box"] == {
        "x1": pytest.approx(431 / 1000 * 640),
        "y1": pytest.approx(633 / 1000 * 480),
        "x2": pytest.approx(514 / 1000 * 640),
        "y2": pytest.approx(748 / 1000 * 480),
    }
    assert det["label"] == "small dark grey cube"
    assert det["score"] == 1.0
    method, pil_image, arg, kwargs = worker.calls[0]
    assert method == "ground_multi" and arg == "small dark grey cube"
    assert pil_image.size == (640, 480) and pil_image.mode == "RGB"
    assert kwargs["temperature"] == 0.0  # greedy: repeat calls agree
    assert kwargs["verbose"] is False


def test_labels_drop_the_period_the_batch_runtime_echoes(la, image, monkeypatch, acked):
    _patch(la, monkeypatch, "<ref>open grey box.</ref><box><1><2><3><4></box>")
    assert (
        la.ground(image=image, query="open grey box")["detections"][0]["label"] == "open grey box"
    )


def test_ground_single_uses_the_single_instance_prompt(la, image, monkeypatch, acked):
    worker = _patch(la, monkeypatch, "<ref>cube</ref><box><1><2><3><4></box>")
    la.ground(image=image, query="cube", single=True)
    assert worker.calls[0][0] == "ground_single"


def test_ground_keeps_every_instance_and_orders_and_clamps_corners(la, image, monkeypatch, acked):
    _patch(
        la,
        monkeypatch,
        "<ref>robot gripper</ref><box><94><752><285><1000></box><box><935><1000><779><729></box>",
    )
    dets = la.ground(image=image, query="robot gripper")["detections"]
    assert len(dets) == 2
    second = dets[1]["box"]
    # Swapped corners come back ordered, and stay inside the frame.
    assert second["x1"] < second["x2"] and second["y1"] < second["y2"]
    assert second["y2"] == 480.0


def test_the_return_shape_matches_gemini_er(la, image, monkeypatch, acked):
    ger = load_skills(ROOT, only=["gemini-er"])
    import_served_tools(ger)
    ger_mod = ger.get("gemini-er").tools_module
    _patch(la, monkeypatch, "<ref>cup</ref><box><1><2><3><4></box>")
    det = la.ground(image=image, query="cup")["detections"][0]
    assert set(det) == set(ger_mod.Detection.__annotations__)
    assert set(det["box"]) == {"x1", "y1", "x2", "y2"}
    _patch(la, monkeypatch, "<ref>cup</ref><box><500><500></box>")
    pt = la.point(image=image, query="cup")
    assert set(pt) == set(ger_mod.PointResult.__annotations__)
    assert set(pt["points"][0]) == set(ger_mod.Point.__annotations__)


# ---------------------------------------------------------------------------
# Not found and malformed output never raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "<ref>open grey box</ref><box>None</box><|im_end|>",
        "",
        "I cannot find it.",
        "<ref>x</ref><box><1><2><3></box>",  # wrong arity
        "<ref>x</ref><box><a><b><c><d></box>",  # non-numeric
        "<ref>x</ref><box><1><2><3><4>",  # truncated
    ],
)
def test_ground_finds_nothing_without_raising(la, image, monkeypatch, acked, answer):
    _patch(la, monkeypatch, answer)
    assert la.ground(image=image, query="open grey box") == {"detections": []}


def test_ground_ignores_points_in_a_box_answer(la, image, monkeypatch, acked):
    _patch(la, monkeypatch, "<ref>x</ref><box><100><200></box><box><1><2><3><4></box>")
    assert len(la.ground(image=image, query="x")["detections"]) == 1


# ---------------------------------------------------------------------------
# Detection: several categories, each box labelled by its <ref>
# ---------------------------------------------------------------------------


def test_detect_labels_boxes_by_the_category_they_follow(la, image, monkeypatch, acked):
    worker = _patch(
        la,
        monkeypatch,
        "<ref>cube</ref><box><270><680><305><730></box>"
        "<ref>box</ref><box><431><633><514><748></box>"
        "<ref>gripper</ref><box><94><752><285><1000></box><box><779><729><935><1000></box>",
    )

    dets = la.detect(image=image, categories=["cube", "box", "gripper"])["detections"]

    assert [d["label"] for d in dets] == ["cube", "box", "gripper", "gripper"]
    assert worker.calls[0][0] == "detect"
    assert worker.calls[0][2] == ["cube", "box", "gripper"]


def test_detect_snaps_a_garbled_ref_to_the_category_it_names(la, image, monkeypatch, acked):
    _patch(
        la,
        monkeypatch,
        "<ref>gripperper</ref><box><1><2><3><4></box><ref>Cube.</ref><box><5><6><7><8></box>"
        "<ref>table</ref><box><9><9><20><20></box>",
    )
    dets = la.detect(image=image, categories=["cube", "gripper"])["detections"]
    assert [d["label"] for d in dets] == ["gripper", "cube", "table"]


def test_detect_takes_one_category_as_a_string(la, image, monkeypatch, acked):
    worker = _patch(la, monkeypatch, "<box><1><2><3><4></box>")
    dets = la.detect(image=image, categories="cube")["detections"]
    assert worker.calls[0][2] == ["cube"]
    assert dets[0]["label"] == "cube"  # no <ref>: the first category


def test_detect_needs_a_category(la, image, monkeypatch, acked):
    _patch(la, monkeypatch, "")
    with pytest.raises(ValueError, match="at least one category"):
        la.detect(image=image, categories=["", "  "])


# ---------------------------------------------------------------------------
# Pointing: drop-in for gemini-er.point
# ---------------------------------------------------------------------------


def test_point_converts_x_y_to_pixels_and_keeps_every_point(la, image, monkeypatch, acked):
    worker = _patch(
        la,
        monkeypatch,
        "<ref>the centre of each ring</ref><box><500><250></box><box><1200><980></box><|im_end|>",
    )

    out = la.point(image=image, query="the centre of each ring", max_points=3)

    assert out["found"] is True
    # [x, y] = [500, 250] on a 640x480 frame (x first, unlike Gemini's [y, x]):
    assert out["pixel_x"] == pytest.approx(320.0)
    assert out["pixel_y"] == pytest.approx(120.0)
    first, second = out["points"]
    assert first == {
        "pixel_x": pytest.approx(320.0),
        "pixel_y": pytest.approx(120.0),
        "label": "the centre of each ring",
    }
    assert second["pixel_x"] == 640.0  # clamped
    assert second["pixel_y"] == pytest.approx(980 / 1000 * 480)
    assert worker.calls[0][0] == "point"


def test_point_caps_the_count(la, image, monkeypatch, acked):
    _patch(la, monkeypatch, "<box><1><1></box><box><2><2></box><box><3><3></box>")
    assert len(la.point(image=image, query="dots", max_points=2)["points"]) == 2


@pytest.mark.parametrize(
    "answer",
    ["<ref>x</ref><box>None</box>", "", "nothing", "<box><1></box>", "<box><1><2><3><4></box>"],
)
def test_point_finds_nothing_without_raising(la, image, monkeypatch, acked, answer):
    _patch(la, monkeypatch, answer)
    assert la.point(image=image, query="unicorn") == {
        "pixel_x": 0.0,
        "pixel_y": 0.0,
        "found": False,
        "points": [],
    }


# ---------------------------------------------------------------------------
# Input validation and model errors
# ---------------------------------------------------------------------------


def test_rejects_a_non_rgb_image(la, monkeypatch, acked):
    _patch(la, monkeypatch, "")
    with pytest.raises(ValueError, match=r"uint8 \[H, W, 3\]"):
        la.ground(image=np.zeros((4, 6), dtype=np.float32), query="cup")


def test_model_errors_are_wrapped_as_tool_errors(la, image, monkeypatch, acked):
    def _boom():
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(la, "_get_worker", _boom)
    with pytest.raises(ToolError, match="CUDA out of memory"):
        la.point(image=image, query="cup")


# ---------------------------------------------------------------------------
# Runtime: pinned worker download, runtime selection, cache probe
# ---------------------------------------------------------------------------


def test_the_worker_download_is_checked_against_its_pin(la, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("GAP_LOCATE_ANYTHING_WORKER", raising=False)
    payload = b"class LocateAnythingWorker: pass\n"

    class _Resp:
        def __init__(self, data):
            self.data = data

        def read(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(la.urllib.request, "urlopen", lambda url, timeout: _Resp(payload))
    with pytest.raises(RuntimeError, match="is not the pinned"):
        la._worker_path()
    assert not list(tmp_path.rglob("locateanything_worker.py"))

    monkeypatch.setattr(la, "WORKER_SHA256", hashlib.sha256(payload).hexdigest())
    path = la._worker_path()
    assert path.read_bytes() == payload
    assert la.EAGLE_COMMIT[:12] in str(path)
    assert la._import_worker_class(path).__name__ == "LocateAnythingWorker"
    assert la._worker_path(download=False) == path


def test_build_worker_selects_the_runtime(la, monkeypatch, tmp_path):
    made: list[tuple] = []

    class _Worker:
        def __init__(self, model_path, **kwargs):
            made.append((model_path, kwargs))

    monkeypatch.setattr(la, "_model_dir", lambda: str(tmp_path))
    monkeypatch.setattr(la, "_worker_path", lambda download=True: tmp_path / "w.py")
    monkeypatch.setattr(la, "_import_worker_class", lambda path: _Worker)

    monkeypatch.setattr(la.sys, "path", list(la.sys.path))
    monkeypatch.delenv("GAP_LOCATE_ANYTHING_RUNTIME", raising=False)
    monkeypatch.delenv("GAP_LOCATE_ANYTHING_ATTN", raising=False)
    la._build_worker()
    assert made[-1] == (
        str(tmp_path),
        {"device": "cuda", "use_batch_runtime": True, "attn": "sdpa"},
    )
    assert la.sys.path[0] == str(tmp_path)  # batch_utils ships in the snapshot

    monkeypatch.setenv("GAP_LOCATE_ANYTHING_ATTN", "la_flash")
    la._build_worker()
    assert made[-1][1]["attn"] == "la_flash"

    monkeypatch.setenv("GAP_LOCATE_ANYTHING_RUNTIME", "standard")
    la._build_worker()
    assert made[-1] == (str(tmp_path), {"device": "cuda"})

    monkeypatch.setenv("GAP_LOCATE_ANYTHING_RUNTIME", "vllm")
    with pytest.raises(ValueError, match="standard or batch"):
        la._build_worker()


def test_weights_cached_needs_the_model_and_the_worker(la, monkeypatch, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setenv("GAP_LOCATE_ANYTHING_MODEL", str(model))
    worker = tmp_path / "w.py"
    worker.write_text("")
    monkeypatch.setenv("GAP_LOCATE_ANYTHING_WORKER", str(worker))
    assert la.weights_cached() is False  # no config.json yet
    (model / "config.json").write_text("{}")
    assert la.weights_cached() is True


def test_importing_the_bundle_does_not_import_a_model_library():
    tree = ast.parse((BUNDLE / "tools.py").read_text())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert not top & {"torch", "transformers", "huggingface_hub", "flash_attn"}
