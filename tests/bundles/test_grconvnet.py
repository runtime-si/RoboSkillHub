"""grconvnet bundle: crop mapping, post-processing, mask gating, span check, world grasps and
checkpoint handling, with GR-ConvNet faked."""

from __future__ import annotations

import io
import math
import pickle
from pathlib import Path

import numpy as np
import pytest
from gap_core.errors import ToolError

H, W = 120, 160
F = 200.0
K = np.array([[F, 0.0, 80.0], [0.0, F, 60.0], [0.0, 0.0, 1.0]])
BUNDLE = Path(__file__).resolve().parents[2] / "tools" / "grconvnet"
#: A camera 1 m over the table looking straight down: camera x = world x, camera y = world -y.
#: A sigma-3 blob after the quality map's sigma-2 smoothing keeps 9/13 of its height.
SMOOTHED = 9 / 13
DOWN = np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 1.0], [0, 0, 0, 1.0]])


@pytest.fixture(scope="module")
def gc(skills_registry):
    return skills_registry.get("grconvnet").tools_module


class FakeRuntime:
    """Quality blobs at network pixels, one angle and one width (network px) everywhere."""

    def __init__(self, size, peaks=((150, 150, 0.9),), angle=0.0, width_px=15.0, fail=None):
        self.size, self.peaks, self.angle, self.width_px, self.fail = (
            size,
            list(peaks),
            angle,
            width_px,
            fail,
        )
        self.calls = []
        self.device = "fake"

    def infer(self, x):
        if self.fail:
            raise self.fail
        self.calls.append(x)
        s = self.size
        rr, cc = np.mgrid[0:s, 0:s]
        q = np.zeros((s, s))
        for r, c, h in self.peaks:
            q = np.maximum(q, h * np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * 3.0**2)))
        cos2 = np.full((s, s), math.cos(2 * self.angle))
        sin2 = np.full((s, s), math.sin(2 * self.angle))
        width = np.full((s, s), self.width_px / 150.0)
        return q, cos2, sin2, width


@pytest.fixture
def fake(gc, monkeypatch):
    runtime = FakeRuntime(224)
    monkeypatch.setattr(gc, "_get_runtime", lambda key: (runtime, 0.0))
    return runtime


def propose(gc, *args, **kwargs):
    """``grconvnet.propose`` on the Cornell checkpoint's 224 px input at native resolution."""
    kwargs.setdefault("checkpoint", "cornell-rgbd")
    kwargs.setdefault("zoom", 1.0)
    return gc.propose(*args, **kwargs)


def rgb():
    return np.full((H, W, 3), 200, dtype=np.uint8)


def table(z=1.0):
    return np.full((H, W), z, dtype=np.float32)


def scene(block=((-0.02, 0.02), (-0.01, 0.01)), height=0.03):
    """The DOWN camera's depth of a ``block`` (world x / y ranges, m) ``height`` tall on z = 0."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    depth = np.ones((H, W))
    top = 1.0 - height
    x = (u - K[0, 2]) / F * top
    y = -(v - K[1, 2]) / F * top
    (x0, x1), (y0, y1) = block
    on = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    depth[on] = top
    return depth.astype(np.float32), on


# --- registration, pins, imports ----------------------------------------------------


def test_tool_registered_with_its_schema(tool_registry):
    schema = tool_registry.get("grconvnet.propose").schema
    assert {"rgb", "depth_m", "K", "mask", "camera_pose", "n", "gripper_width_max"} <= set(
        schema.inputs
    )
    assert schema.inputs["n"].default == 10
    assert schema.inputs["gripper_width_max"].default == pytest.approx(0.094)
    assert {"grasps", "n_too_wide", "crop", "timing"} <= set(schema.outputs)


def test_import_does_not_pull_torch(gc):
    assert gc._runtimes == {} or all(not isinstance(r, FakeRuntime) for r in gc._runtimes.values())
    assert "torch" not in vars(gc)


def test_pins_agree_with_skill_md(gc):
    skill = (BUNDLE / "SKILL.md").read_text()
    default = gc.CHECKPOINTS[gc.DEFAULT_CHECKPOINT]
    assert f"{gc.UPSTREAM_COMMIT}/trained-models/{default['path']}" in skill
    assert f"sha256: {default['sha256']}" in skill
    assert f"tree/{gc.UPSTREAM_COMMIT}" in skill


# --- pixels: crop mapping and post-processing ---------------------------------------


def test_centre_crop_maps_a_peak_back_to_the_frame(gc, fake):
    out = propose(gc, rgb(), table(), K)
    assert out["crop"] == {"x0": -32.0, "y0": -52.0, "side": 224.0, "scale": 1.0}
    (g,) = out["grasps"]
    assert g["center_px"] == pytest.approx([150 - 32, 150 - 52], abs=0.01)
    assert g["quality"] == pytest.approx(0.9 * SMOOTHED, abs=0.01) and g["rank"] == 0
    assert g["angle_rad"] == pytest.approx(0.0, abs=1e-6)
    assert g["width_px"] == pytest.approx(15.0, rel=1e-3)
    assert g["width_m"] == pytest.approx(15.0 * 1.0 / F, rel=1e-3)
    assert g["jaw_px"] == [
        pytest.approx([110.5, 98], abs=0.01),
        pytest.approx([125.5, 98], abs=0.01),
    ]
    assert g["depth_m"] == pytest.approx(1.0) and g["depth_measured"]
    assert g["position"] is None and g["obb"] is None
    assert out["checkpoint"] == "cornell-rgbd" and out["n_peaks"] == 1


def test_network_input_is_upstreams_normalisation(gc, fake):
    depth = table()
    depth[:, :80] = 0.9
    img = rgb()
    img[..., 0] = 0
    propose(gc, img, depth, K)
    (x,) = fake.calls
    assert x.shape == (4, 224, 224) and x.dtype == np.float32
    assert x[0].mean() == pytest.approx(0.0, abs=1e-5)  # depth first, mean-subtracted
    assert x[0].max() - x[0].min() == pytest.approx(0.1, abs=1e-3)
    assert x[1:].mean() == pytest.approx(0.0, abs=1e-5)
    assert x[1, 100, 100] < x[2, 100, 100]  # red channel is darker than green


def test_depth_only_checkpoint_gets_one_channel(gc, monkeypatch):
    runtime = FakeRuntime(300, peaks=[(150, 150, 0.8)])
    monkeypatch.setattr(gc, "_get_runtime", lambda key: (runtime, 0.0))
    out = propose(gc, rgb(), table(), K, checkpoint="jacquard-d")
    assert runtime.calls[0].shape == (1, 300, 300) and out["checkpoint"] == "jacquard-d"


def test_angle_is_taken_after_smoothing_cos_and_sin(gc, fake):
    fake.angle = math.radians(89.0)
    out = propose(gc, rgb(), table(), K)
    assert out["grasps"][0]["angle_deg"] == pytest.approx(89.0, abs=0.01)
    fake.angle = math.radians(-60.0)
    g = propose(gc, rgb(), table(), K)["grasps"][0]
    assert g["angle_deg"] == pytest.approx(-60.0, abs=0.01)
    u, v = g["center_px"]
    (au, av), (bu, bv) = g["jaw_px"]
    assert bu > au and bv > av  # image y points down: -60 deg runs down to the right


def test_peaks_are_ranked_spaced_and_limited(gc, fake):
    fake.peaks = [(100, 100, 0.5), (100, 110, 0.6), (160, 160, 0.9), (60, 170, 0.7), (150, 60, 0.4)]
    out = propose(gc, rgb(), table(), K, n=2)
    assert [g["quality"] for g in out["grasps"]] == pytest.approx(
        [0.9 * SMOOTHED, 0.7 * SMOOTHED], abs=0.01
    )
    out = propose(gc, rgb(), table(), K, n=10)
    q = [g["quality"] for g in out["grasps"]]
    assert q == sorted(q, reverse=True) and len(q) == 4  # 100/100 is within 20 px of 100/110
    assert [g["rank"] for g in out["grasps"]] == [0, 1, 2, 3]


def test_border_peaks_and_weak_peaks_are_not_proposals(gc, fake):
    fake.peaks = [(5, 100, 0.9), (150, 150, 0.15)]
    out = propose(gc, rgb(), table(), K)
    assert out["grasps"] == [] and out["n_peaks"] == 0
    assert propose(gc, rgb(), table(), K, quality_threshold=0.1)["n_peaks"] == 1


def test_mask_centres_the_crop_and_gates_the_peaks(gc, fake):
    mask = np.zeros((H, W), np.uint8)
    mask[50:70, 70:90] = 255  # 20 px square -> 40 px crop at fill 0.5, floored at MIN_CROP_PX
    fake.peaks = [(112, 112, 0.6), (20 + 30, 30 + 30, 0.95)]
    out = propose(gc, rgb(), table(), K, mask=mask, object_fill=0.5)
    c = out["crop"]
    assert (
        c["side"] == gc.MIN_CROP_PX
        and c["x0"] == pytest.approx(80 - 32)
        and c["y0"] == pytest.approx(60 - 32)
    )
    (g,) = out["grasps"]
    assert g["in_mask"] and g["quality"] == pytest.approx(0.6 * SMOOTHED, abs=0.01)
    s = c["scale"]
    assert g["center_px"] == pytest.approx([48 + 112.5 * s - 0.5, 28 + 112.5 * s - 0.5], abs=0.01)
    assert g["width_px"] == pytest.approx(15.0 * s, abs=0.01)
    assert out["best_quality"] < 0.6


def test_crop_sizes(gc, fake):
    mask = np.zeros((H, W), bool)
    mask[30:90, 40:120] = True  # 80 px wide, centred on (80, 60)
    side = lambda **kw: propose(gc, rgb(), table(), K, mask=mask, **kw)["crop"]["side"]  # noqa: E731
    assert side() == 224.0  # zoom 1: the network's input at native resolution
    assert side(zoom=4 / 3) == 168.0
    assert side(zoom=4.0) == 89.0  # grown so the mask fills at most MAX_MASK_SHARE of it
    assert side(object_fill=0.5) == 160.0 and side(object_fill=1.0) == 80.0
    assert side(crop_px=100) == 100.0
    assert propose(gc, rgb(), table(), K, roi=[10, 20, 50, 40])["crop"] == pytest.approx(
        {"x0": 10.0, "y0": 10.0, "side": 40.0, "scale": 40 / 224}, abs=1e-5
    )


def test_defaults_are_the_depth_only_checkpoint_at_four_thirds_zoom(gc, monkeypatch):
    runtime = FakeRuntime(300, peaks=[(150, 150, 0.8)])
    monkeypatch.setattr(gc, "_get_runtime", lambda key: (runtime, 0.0))
    out = gc.propose(rgb(), table(), K)
    assert out["checkpoint"] == "jacquard-d" and runtime.calls[0].shape == (1, 300, 300)
    assert out["crop"]["side"] == 225.0 and out["crop"]["scale"] == pytest.approx(0.75)


def test_off_frame_padding_is_not_the_mask(gc, fake):
    mask = np.zeros((H, W), bool)
    mask[:10, :10] = True  # corner: the crop runs off the frame
    fake.peaks = [(20, 20, 0.9), (150, 150, 0.6)]
    out = propose(gc, rgb(), table(), K, mask=mask)
    assert all(0 <= g["center_px"][0] < W and 0 <= g["center_px"][1] < H for g in out["grasps"])


def test_no_grasp_centre_in_the_padding_without_a_mask(gc, fake):
    fake.peaks = [(115, 25, 0.9)]  # frame u = 25 - 32 < 0: the crop's padded margin
    assert propose(gc, rgb(), table(), K)["grasps"] == []


def test_too_wide_grasps_are_dropped_and_counted(gc, fake):
    fake.width_px = 30.0  # 30 px at 1 m = 0.15 m
    out = propose(gc, rgb(), table(), K)
    assert out["grasps"] == [] and out["n_too_wide"] == 1 and out["n_peaks"] == 1
    out = propose(gc, rgb(), table(), K, gripper_width_max=0.2)
    assert len(out["grasps"]) == 1 and out["n_too_wide"] == 0


def test_holes_take_the_nearest_measured_depth(gc, fake):
    depth = table()
    depth[90:110, 110:130] = 0.0  # under the peak (frame 118, 98)
    depth[90:110, 110:115] = np.nan
    g = propose(gc, rgb(), depth, K)["grasps"][0]
    assert not g["depth_measured"] and g["depth_m"] == pytest.approx(1.0)
    assert np.isfinite(fake.calls[0]).all()


# --- world grasps --------------------------------------------------------------------


def _peak_at_world_origin(fake, angle=0.0, width_px=15.0):
    """A peak on the DOWN camera's principal point (frame 80, 60) for a centre crop."""
    fake.peaks = [(60 + 52, 80 + 32, 0.9)]
    fake.angle, fake.width_px = angle, width_px


def test_world_grasp_on_a_block_seen_from_above(gc, fake):
    depth, _ = scene()
    _peak_at_world_origin(fake)
    g = propose(gc, rgb(), depth, K, camera_pose=DOWN)["grasps"][0]
    assert g["top_z"] == pytest.approx(0.03, abs=1e-6)
    assert g["support_z"] == pytest.approx(0.0, abs=1e-6)
    assert g["object_height_m"] == pytest.approx(0.03, abs=1e-6)
    assert g["position"] == pytest.approx({"x": 0.0, "y": 0.0, "z": 0.015}, abs=1e-3)
    assert g["close_heading_deg"] == pytest.approx(0.0, abs=1e-6)
    assert g["width_m"] == pytest.approx(15 * 0.97 / F, rel=1e-3)
    # Measured from depth along the jaw line: 40 mm across, 20 mm along the faces.
    assert g["object_width_m"] == pytest.approx(0.04, abs=0.006)
    assert g["object_length_m"] == pytest.approx(0.02, abs=0.006)
    assert g["closing_is_short_axis"] is False
    obb = g["obb"]
    assert obb["center"] == pytest.approx({"x": 0.0, "y": 0.0, "z": 0.015}, abs=1e-3)
    assert obb["extent"]["x"] == pytest.approx(g["object_width_m"] / 2)
    assert obb["extent"]["z"] == pytest.approx(0.015, abs=1e-6)
    assert obb["orientation"] == pytest.approx({"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})


def test_a_vertical_image_axis_closes_along_world_y(gc, fake):
    depth, _ = scene()
    _peak_at_world_origin(fake, angle=math.radians(-90.0))
    g = propose(gc, rgb(), depth, K, camera_pose=DOWN)["grasps"][0]
    assert abs(g["close_heading_deg"]) == pytest.approx(90.0, abs=1e-6)
    assert g["object_width_m"] == pytest.approx(0.02, abs=0.006)
    assert g["object_length_m"] == pytest.approx(0.04, abs=0.006)
    assert g["closing_is_short_axis"] is True
    q = g["obb"]["orientation"]
    assert abs(q["z"]) == pytest.approx(math.sin(math.pi / 4), abs=1e-6)


def test_image_angle_turns_to_the_world_heading(gc, fake):
    depth, _ = scene(block=((-0.03, 0.03), (-0.03, 0.03)))
    _peak_at_world_origin(fake, angle=math.radians(30.0))
    g = propose(gc, rgb(), depth, K, camera_pose=DOWN)["grasps"][0]
    # Image +30 deg anticlockwise as seen; the DOWN camera's image y is world -y, so world +30 deg.
    assert g["close_heading_deg"] == pytest.approx(30.0, abs=1e-4)


def test_se3pose_dict_and_matrix_agree(gc, fake):
    depth, _ = scene()
    _peak_at_world_origin(fake)
    pose = {
        "position": {"x": 0.0, "y": 0.0, "z": 1.0},
        "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0},
    }
    a = propose(gc, rgb(), depth, K, camera_pose=pose)["grasps"][0]
    b = propose(gc, rgb(), depth, K, camera_pose=DOWN)["grasps"][0]
    assert (
        a["position"] == pytest.approx(b["position"])
        and a["close_heading_deg"] == b["close_heading_deg"]
    )


def test_given_support_height_is_used(gc, fake):
    depth, _ = scene()
    _peak_at_world_origin(fake)
    g = propose(gc, rgb(), depth, K, camera_pose=DOWN, support_z=0.01)["grasps"][0]
    assert g["support_z"] == 0.01 and g["object_height_m"] == pytest.approx(0.02, abs=1e-6)
    assert g["position"]["z"] == pytest.approx(0.02, abs=1e-6)


def test_flat_object_falls_back_to_the_network_width(gc, fake):
    _peak_at_world_origin(fake)
    g = propose(gc, rgb(), table(), K, camera_pose=DOWN)["grasps"][0]
    assert g["object_height_m"] == pytest.approx(0.0, abs=1e-6)
    assert g["object_width_m"] == pytest.approx(g["width_m"]) == pytest.approx(15 / F, rel=1e-3)


def test_the_floor_past_the_table_edge_is_not_the_support(gc, fake):
    depth, _ = scene()
    depth[:, :60] = 2.0  # a metre below the table
    _peak_at_world_origin(fake)
    g = propose(gc, rgb(), depth, K, camera_pose=DOWN)["grasps"][0]
    assert g["support_z"] == pytest.approx(0.0, abs=1e-6)


def test_world_width_is_the_span_check(gc, fake):
    depth, _ = scene()
    _peak_at_world_origin(fake, width_px=15.0)  # 15 px at 0.97 m = 72.75 mm
    assert (
        propose(gc, rgb(), depth, K, camera_pose=DOWN, gripper_width_max=0.072)["n_too_wide"] == 1
    )
    assert (
        len(propose(gc, rgb(), depth, K, camera_pose=DOWN, gripper_width_max=0.074)["grasps"]) == 1
    )


# --- inputs and failures --------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rgb": np.zeros((H, W, 3), np.float32)}, "rgb must be uint8"),
        ({"depth_m": np.zeros((H, W), np.uint16)}, "float metres"),
        ({"depth_m": np.zeros((H + 1, W), np.float32)}, "depth_m must be"),
        ({"depth_m": np.zeros((H, W), np.float32)}, "no valid pixel"),
        ({"K": np.eye(2)}, "3x3"),
        ({"mask": np.zeros((H, W), np.uint8)}, "mask is empty"),
        ({"mask": np.ones((H, W + 1), np.uint8)}, "mask must be"),
        ({"camera_pose": np.eye(3)}, "4x4"),
        ({"n": 0}, "n must be"),
        ({"object_fill": 2.0}, "object_fill"),
        ({"zoom": 0.0}, "zoom"),
        ({"gripper_width_max": 0.0}, "gripper_width_max"),
        ({"roi": [10, 10, 5, 20]}, "roi must be"),
        ({"checkpoint": "imagenet"}, "checkpoint must be one of"),
    ],
)
def test_bad_inputs_are_refused_before_the_model(gc, fake, kwargs, match):
    args = {"rgb": rgb(), "depth_m": table(), "K": K} | kwargs
    with pytest.raises(ValueError, match=match):
        propose(gc, **args)
    assert fake.calls == []


def test_model_failures_become_tool_errors(gc, monkeypatch):
    def broken(key):
        raise RuntimeError("no network")

    monkeypatch.setattr(gc, "_get_runtime", broken)
    with pytest.raises(ToolError, match="failed to load: no network"):
        propose(gc, rgb(), table(), K)
    runtime = FakeRuntime(224, fail=RuntimeError("bad input"))
    monkeypatch.setattr(gc, "_get_runtime", lambda key: (runtime, 0.0))
    with pytest.raises(ToolError, match="GR-ConvNet failed: bad input"):
        propose(gc, rgb(), table(), K)
    runtime = FakeRuntime(100)
    monkeypatch.setattr(gc, "_get_runtime", lambda key: (runtime, 0.0))
    with pytest.raises(ToolError, match="expected"):
        propose(gc, rgb(), table(), K)


# --- checkpoints ---------------------------------------------------------------------


def test_upstream_pickled_classes_map_onto_the_bundles(gc, monkeypatch):
    class Net:
        pass

    class Block:
        pass

    monkeypatch.setattr(gc, "_net_classes", (Net, Block))

    def pickled(module, name):  # protocol 2: GLOBAL module name, empty args, NEWOBJ
        return b"\x80\x02c" + f"{module}\n{name}\n".encode() + b")\x81."

    load = lambda data: gc._Unpickler(io.BytesIO(data)).load()  # noqa: E731
    assert isinstance(load(pickled("inference.models.grconvnet3", "GenerativeResnet")), Net)
    assert isinstance(load(pickled("inference.models.grasp_model", "ResidualBlock")), Block)
    with pytest.raises(pickle.UnpicklingError, match="unexpected class"):
        load(pickled("inference.models.grconvnet3", "Evil"))


def test_download_is_checked_against_its_digest(gc, monkeypatch, tmp_path):
    monkeypatch.setenv("GAP_GRCONVNET_CACHE", str(tmp_path))
    monkeypatch.delenv("GAP_GRCONVNET_CHECKPOINT_PATH", raising=False)
    urls = []

    def fetch(url, dest):
        urls.append(url)
        Path(dest).write_bytes(b"not the checkpoint")

    monkeypatch.setattr(gc.urllib.request, "urlretrieve", fetch)
    with pytest.raises(RuntimeError, match="expected"):
        gc._checkpoint_file("cornell-rgbd")
    assert urls == [gc._RAW + gc.CHECKPOINTS["cornell-rgbd"]["path"]]
    assert not list(tmp_path.iterdir())
    assert gc.weights_cached() is False


def test_local_checkpoint_override(gc, monkeypatch, tmp_path):
    ckpt = tmp_path / "net.pt"
    monkeypatch.setenv("GAP_GRCONVNET_CHECKPOINT_PATH", str(ckpt))
    assert gc.weights_cached() is False
    ckpt.write_bytes(b"x")
    assert gc.weights_cached() is True
    assert gc._checkpoint_file(gc.DEFAULT_CHECKPOINT) == str(ckpt)
    assert gc._model_id(gc.DEFAULT_CHECKPOINT) == str(ckpt)
