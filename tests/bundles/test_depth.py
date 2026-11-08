"""depth bundle: input checks, alignment, fill / refine modes and the xFormers shim, with LingBot-Depth faked."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from gap_core.errors import ToolError

W, H = 80, 60  # 4800 px: enough measured depth to align (MIN_ALIGN_PIXELS)
K = np.array([[50.0, 0.0, 40.0], [0.0, 50.0, 30.0], [0.0, 0.0, 1.0]])
BUNDLE = Path(__file__).resolve().parents[2] / "tools" / "depth"


@pytest.fixture(scope="module")
def dp(skills_registry):
    return skills_registry.get("depth").tools_module


class FakeRuntime:
    """Returns ``depth_fn(raw)`` as the model's depth and ``mask`` as its mask head's call."""

    def __init__(self, depth_fn=None, mask=None, fail=None):
        self.depth_fn = depth_fn or (lambda raw: np.full(raw.shape, 0.5, np.float32))
        self.mask = mask
        self.fail = fail
        self.calls = []

    def infer(self, rgb, depth, resolution_level):
        if self.fail:
            raise self.fail
        self.calls.append({"rgb": rgb, "depth": depth.copy(), "level": resolution_level})
        return self.depth_fn(depth), self.mask


@pytest.fixture
def fake(dp, monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(dp, "_get_runtime", lambda: (runtime, 0.0))
    return runtime


def rgb(h=H, w=W):
    return np.zeros((h, w, 3), dtype=np.uint8)


def plane(h=H, w=W, z=0.5, tilt=0.002):
    """A tilted table: depth grows 2 mm per row from 0.5 m."""
    return (z + tilt * np.arange(h)[:, None] + np.zeros((1, w))).astype(np.float32)


def holes(depth, rows=slice(10, 20), cols=slice(30, 40)):
    out = depth.copy()
    out[rows, cols] = 0.0
    return out


# --- registration ---------------------------------------------------------


def test_tool_registered_with_its_schema(tool_registry):
    schema = tool_registry.get("depth.refine").schema
    assert set(schema.inputs) == {
        "rgb",
        "depth_m",
        "K",
        "keep_measured",
        "align",
        "resolution_level",
        "return_points",
    }
    assert schema.inputs["keep_measured"].default is True
    assert schema.inputs["align"].default is True
    assert schema.inputs["resolution_level"].default == 9
    assert {"depth_m", "valid", "filled", "aligned", "align_scale", "timing"} <= set(schema.outputs)


def test_import_does_not_pull_torch(dp):
    assert dp._runtime is None
    assert not {"torch", "mdm"} & set(vars(dp))


def test_pins_agree_with_skill_md_and_pyproject(dp):
    skill = (BUNDLE / "SKILL.md").read_text()
    assert f"uri: hf://{dp.MODEL_REPO}@{dp.MODEL_REVISION}" in skill
    assert f"blob/{dp.LINGBOT_COMMIT}/" in skill
    assert f'rev = "{dp.LINGBOT_COMMIT}"' in (BUNDLE / "pyproject.toml").read_text()


def test_weights_probe_honours_a_local_override(dp, monkeypatch, tmp_path):
    ckpt = tmp_path / "model.pt"
    monkeypatch.setenv("GAP_DEPTH_MODEL", str(ckpt))
    assert dp.weights_cached() is False
    ckpt.write_bytes(b"x")
    assert dp.weights_cached() is True
    assert dp._model_id() == str(ckpt)


# --- fill (default) ------------------------------------------------------------


def test_fill_keeps_measured_depth_and_fills_holes_with_the_aligned_model(dp, fake):
    truth = plane()
    raw = holes(truth)
    fake.depth_fn = lambda d: (truth - 0.012) / 1.05  # off in scale and shift, like the station
    out = dp.refine(rgb(), raw, K)
    hole = raw == 0
    assert np.array_equal(out["depth_m"][~hole], raw[~hole])
    assert out["depth_m"][hole] == pytest.approx(truth[hole], abs=1e-5)
    assert out["aligned"] and out["align_scale"] == pytest.approx(1.05, abs=1e-4)
    assert out["align_shift_mm"] == pytest.approx(12.0, abs=0.01)
    assert out["model_vs_measured_mm"] == pytest.approx(0.0, abs=0.01)
    assert out["filled"].dtype == np.uint8 and set(np.unique(out["filled"])) == {0, 255}
    assert np.array_equal(out["filled"] > 0, hole)
    assert (out["valid"] == 255).all()
    assert out["raw_valid_fraction"] == pytest.approx(1 - 100 / (H * W), abs=1e-4)
    assert out["filled_fraction"] == 1.0 and out["valid_fraction"] == 1.0
    assert out["backend"] == "lingbot-depth" and out["points_m"] is None
    assert out["model"].startswith("robbyant/lingbot-depth-pretrain-vitl-14-v0.5@")


def test_alignment_ignores_outliers(dp, fake):
    truth = plane()
    model = (truth - 0.02) / 0.95
    model[:6, :] += 0.3  # the model misses a band badly
    raw = holes(truth)
    raw[40:, :8] = 2.5  # flying sensor pixels
    fake.depth_fn = lambda d: model
    out = dp.refine(rgb(), raw)
    assert out["align_scale"] == pytest.approx(0.95, abs=1e-4)
    assert out["align_shift_mm"] == pytest.approx(20.0, abs=0.05)
    hole = raw == 0
    assert out["depth_m"][hole] == pytest.approx(truth[hole], abs=1e-5)


def test_without_align_the_model_fills_as_it_comes(dp, fake):
    raw = holes(plane())
    out = dp.refine(rgb(), raw, align=False)
    assert not out["aligned"] and (out["align_scale"], out["align_shift_mm"]) == (1.0, 0.0)
    assert out["depth_m"][raw == 0] == pytest.approx(0.5)


def test_too_little_measured_depth_is_not_aligned(dp, fake):
    raw = np.zeros((H, W), np.float32)
    raw[0, :10] = 0.6
    out = dp.refine(rgb(), raw)
    assert not out["aligned"] and out["align_scale"] == 1.0
    assert out["depth_m"][5, 5] == pytest.approx(0.5)
    assert out["depth_m"][0, 0] == pytest.approx(0.6)


def test_an_all_hole_frame_is_filled_by_the_model(dp, fake):
    out = dp.refine(rgb(), np.zeros((H, W), np.float32))
    assert out["raw_valid_fraction"] == 0.0 and out["filled_fraction"] == 1.0
    assert out["model_vs_measured_mm"] == 0.0 and not out["aligned"]


def test_a_hole_the_mask_head_rejects_stays_empty(dp, fake):
    raw = holes(plane())
    mask = np.ones((H, W), np.float32)
    mask[10:15, 30:40] = 0.0  # half the hole
    mask[0:5, 0:5] = 0.0  # measured pixels the model would not vouch for
    fake.mask = mask
    out = dp.refine(rgb(), raw)
    assert (out["depth_m"][10:15, 30:40] == 0).all() and (out["valid"][10:15, 30:40] == 0).all()
    assert (out["depth_m"][15:20, 30:40] > 0).all()
    assert (out["valid"][0:5, 0:5] == 255).all()  # measured stays valid in fill mode
    assert out["filled_fraction"] == pytest.approx(0.5)


# --- refine (keep_measured=False) ------------------------------------------------


def test_refine_returns_the_aligned_model_everywhere(dp, fake):
    truth = plane()
    noisy = truth + np.random.default_rng(0).normal(0, 0.002, truth.shape).astype(np.float32)
    raw = holes(noisy)
    fake.depth_fn = lambda d: truth / 1.02
    out = dp.refine(rgb(), raw, keep_measured=False)
    assert out["depth_m"] == pytest.approx(truth, abs=2e-3)
    assert np.abs(out["depth_m"] - truth).max() < np.abs(noisy - truth).max()
    assert np.array_equal(out["filled"] > 0, raw == 0)


def test_refine_drops_what_the_mask_head_rejects(dp, fake):
    fake.mask = np.zeros((H, W), np.float32)
    fake.mask[:, : W // 2] = 1.0
    out = dp.refine(rgb(), plane(), keep_measured=False)
    assert (out["depth_m"][:, W // 2 :] == 0).all() and out["valid_fraction"] == 0.5


def test_non_finite_model_output_is_not_valid(dp, fake):
    def model(d):
        out = plane()
        out[0, 0], out[0, 1], out[0, 2] = np.nan, np.inf, -1.0
        return out

    fake.depth_fn = model
    raw = plane()
    raw[0, :3] = 0
    out = dp.refine(rgb(), raw)
    assert (out["depth_m"][0, :3] == 0).all() and (out["filled"][0, :3] == 0).all()


# --- inputs, points, failures --------------------------------------------------


def test_bad_depth_values_reach_the_model_as_holes(dp, fake):
    raw = plane()
    raw[0, 0], raw[0, 1], raw[0, 2], raw[0, 3], raw[0, 4] = np.nan, np.inf, -0.3, 0.005, 0.0
    out = dp.refine(rgb(), raw.astype(np.float64), resolution_level=3)
    sent = fake.calls[0]["depth"]
    assert sent.dtype == np.float32 and (sent[0, :5] == 0).all() and sent[0, 5] > 0
    assert fake.calls[0]["level"] == 3
    assert (out["filled"][0, :5] == 255).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rgb": np.zeros((H, W, 3), np.float32)}, "rgb must be uint8"),
        ({"rgb": np.zeros((H, W), np.uint8)}, "rgb must be uint8"),
        ({"depth_m": np.zeros((H, W), np.uint16)}, "float metres"),
        ({"depth_m": np.zeros((H + 1, W), np.float32)}, "depth_m must be"),
        ({"K": np.eye(2)}, "3x3"),
        ({"K": np.zeros((3, 3))}, "fx, fy > 0"),
        ({"resolution_level": 10}, "0..9"),
        ({"return_points": True, "K": None}, "needs K"),
    ],
)
def test_bad_inputs_are_refused_before_the_model(dp, fake, kwargs, match):
    args = {"rgb": rgb(), "depth_m": plane(), "K": K} | kwargs
    with pytest.raises(ValueError, match=match):
        dp.refine(**args)
    assert fake.calls == []


def test_points_back_project_the_output(dp, fake):
    raw = holes(plane())
    fake.mask = np.ones((H, W), np.float32)
    fake.mask[12, 33] = 0.0
    out = dp.refine(rgb(), raw, K, return_points=True)
    pts = out["points_m"]
    assert pts.shape == (H, W, 3) and pts.dtype == np.float32
    v, u = 45, 70
    z = out["depth_m"][v, u]
    assert pts[v, u] == pytest.approx([(u - 40) / 50 * z, (v - 30) / 50 * z, z], rel=1e-5)
    assert np.isnan(pts[12, 33]).all()


def test_model_failures_become_tool_errors(dp, monkeypatch):
    def broken():
        raise RuntimeError("no CUDA")

    monkeypatch.setattr(dp, "_get_runtime", broken)
    with pytest.raises(ToolError, match="failed to load: no CUDA"):
        dp.refine(rgb(), plane())
    runtime = FakeRuntime(fail=RuntimeError("out of memory"))
    monkeypatch.setattr(dp, "_get_runtime", lambda: (runtime, 0.0))
    with pytest.raises(ToolError, match="LingBot-Depth failed: out of memory"):
        dp.refine(rgb(), plane())
    runtime = FakeRuntime(depth_fn=lambda d: np.zeros((H // 2, W), np.float32))
    monkeypatch.setattr(dp, "_get_runtime", lambda: (runtime, 0.0))
    with pytest.raises(ToolError, match="expected"):
        dp.refine(rgb(), plane())


# --- the xFormers shim -----------------------------------------------------------


@pytest.fixture
def fake_mdm(monkeypatch):
    """A stand-in for mdm.model.dinov2_rgbd.layers.block without xFormers."""

    class Block:
        def forward(self, x):
            return x * 2

    class NestedTensorBlock(Block):
        def forward(self, x_or_x_list):
            if isinstance(x_or_x_list, list):
                raise AssertionError("xFormers is required for using nested tensors")
            return Block.forward(self, x_or_x_list)

    block = types.ModuleType("mdm.model.dinov2_rgbd.layers.block")
    block.XFORMERS_AVAILABLE = False
    block.Block, block.NestedTensorBlock = Block, NestedTensorBlock
    names = ["mdm", "mdm.model", "mdm.model.dinov2_rgbd", "mdm.model.dinov2_rgbd.layers"]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, block.__name__, block)
    sys.modules["mdm.model.dinov2_rgbd.layers"].block = block
    return block


def test_one_frame_token_list_runs_without_xformers(dp, fake_mdm):
    dp._run_single_sequences_without_xformers()
    blk = fake_mdm.NestedTensorBlock()
    assert blk.forward([3]) == [6] and blk.forward(4) == 8
    with pytest.raises(AssertionError, match="xFormers"):
        blk.forward([1, 2])  # a real batch still needs xFormers
    patched = fake_mdm.NestedTensorBlock.forward
    dp._run_single_sequences_without_xformers()
    assert fake_mdm.NestedTensorBlock.forward is patched


def test_shim_leaves_an_xformers_install_alone(dp, fake_mdm):
    fake_mdm.XFORMERS_AVAILABLE = True
    original = fake_mdm.NestedTensorBlock.forward
    dp._run_single_sequences_without_xformers()
    assert fake_mdm.NestedTensorBlock.forward is original
