from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from roboskillhub.runtime.tracking import LKPointTracker, TapirPointTracker


def moving_texture():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
    frames = [image]
    for dx in (3, 6):
        frames.append(
            cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, 2 * dx / 3]]), (128, 96))
        )
    return np.stack(frames)


def test_lk_tracks_measured_translation_and_has_json_contract():
    frames = moving_texture()
    points = np.array([[70, 60], [90, 50]], dtype=np.float32)
    result = LKPointTracker().track(frames, points)
    assert result["backend"] == "opencv-lk"
    assert result["coordinate_frame"] == "image_pixels"
    assert result["status"] == "tracked"
    tracks = np.asarray(result["tracks"])
    assert tracks.shape == (3, 2, 2)
    np.testing.assert_allclose(tracks[-1] - points, [[6, 4], [6, 4]], atol=0.3)
    json.dumps(result, allow_nan=False)


def test_lk_loss_is_not_silently_reinitialized_when_texture_returns():
    frames = moving_texture()
    frames[1] = 0
    frames[2] = frames[0]
    result = LKPointTracker().track(frames, [[40, 40]])
    assert result["status"] == "lost"
    assert result["visible"] == [[True], [False], [False]]
    assert result["tracks"][1:] == [[[None, None]], [[None, None]]]
    json.dumps(result, allow_nan=False)


def test_lk_handles_nested_rgb_integer_lists():
    result = LKPointTracker().track(moving_texture().tolist(), [[70, 60]])
    assert result["status"] == "tracked"


@pytest.mark.parametrize("points", [[], [[-1, 3]], [[128, 3]], [[1, float("nan")]], [[1, 2, 3]]])
def test_rejects_invalid_query_points(points):
    with pytest.raises(ValueError, match="query_points"):
        LKPointTracker().track(np.zeros((2, 96, 128, 3), dtype=np.uint8), points)


@pytest.mark.parametrize(
    "frames",
    [
        np.zeros((1, 8, 8, 3), dtype=np.uint8),
        np.zeros((2, 8, 8)),
        np.zeros((2, 8, 8, 3), dtype=np.float32),
        np.full((2, 8, 8, 3), 256),
    ],
)
def test_rejects_invalid_frames(frames):
    with pytest.raises(ValueError, match="frames"):
        LKPointTracker().track(frames, [[2, 2]])


def test_lk_forward_backward_disagreement_is_reported_as_lost(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    calls = []

    def flow(_a, _b, seeds, _initial, **_options):
        calls.append(seeds)
        # Forward goes 1 px; erroneous inverse goes another 4 px.
        return seeds + (1 if len(calls) == 1 else 4), np.ones((1, 1), np.uint8), np.zeros((1, 1))

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", flow)
    result = LKPointTracker(max_fb_error_px=1).track(
        np.zeros((2, 96, 128, 3), dtype=np.uint8), [[40, 40]]
    )
    assert result["status"] == "lost"
    assert result["forward_backward_error_px"][1][0] > 1


def test_tapir_coordinate_conventions_resizing_visibility_and_no_grad():
    torch = pytest.importorskip("torch")

    class ContractModel(torch.nn.Module):
        def forward(self, video, queries, is_training):
            assert not torch.is_grad_enabled()
            assert is_training is False
            assert tuple(video.shape) == (1, 2, 64, 64, 3)
            assert video.min() == -1 and video.max() == 1
            # Original [x,y]=[30,20] in W=120,H=80 → t,y,x=[0,16,16].
            torch.testing.assert_close(queries, torch.tensor([[[0, 16, 16]]], dtype=torch.float32))
            tracks = torch.tensor([[[[16.0, 16.0], [20.0, 16.0]]]])
            return {
                "tracks": tracks,
                "occlusion": torch.tensor([[[-20.0, 20.0]]]),
                "expected_dist": torch.full((1, 1, 2), -20.0),
            }

    frames = np.zeros((2, 80, 120, 3), dtype=np.uint8)
    frames[1] = 255
    result = TapirPointTracker(ContractModel(), resize_hw=(64, 64)).track(frames, [[30, 20]])
    assert result["tracks"] == [[[30.0, 20.0]], [[None, None]]]
    assert result["visible"] == [[True], [False]]
    assert result["backend"] == "tapir"
    assert result["status"] == "lost"
    assert result["checkpoint_sha256"] is None  # injected model isn't a pinned checkpoint
    json.dumps(result, allow_nan=False)


def test_tapir_nonfinite_or_out_of_image_predictions_cannot_be_visible():
    torch = pytest.importorskip("torch")

    def model(_video, _queries, **_kwargs):
        return {
            "tracks": torch.tensor([[[[float("nan"), 4.0], [99.0, 3.0]]]]),
            "occlusion": torch.full((1, 1, 2), -20.0),
            "expected_dist": torch.full((1, 1, 2), -20.0),
        }

    result = TapirPointTracker(model, resize_hw=(32, 32)).track(
        np.zeros((2, 32, 32, 3), dtype=np.uint8), [[8, 8]]
    )
    assert result["visible"] == [[False], [False]]
    json.dumps(result, allow_nan=False)


def test_tapir_rejects_wrong_model_output_contract():
    pytest.importorskip("torch")
    tracker = TapirPointTracker(lambda *_args, **_kwargs: {})
    with pytest.raises(ValueError, match="TAPIR.forward"):
        tracker.track(np.zeros((2, 32, 32, 3), dtype=np.uint8), [[8, 8]])


def test_local_checkpoint_must_exist_and_match_digest_before_model_loading(tmp_path):
    with pytest.raises(FileNotFoundError, match="Local TAPIR checkpoint"):
        TapirPointTracker.from_checkpoint(tmp_path / "missing.pt")
    path = tmp_path / "model.pt"
    path.write_bytes(b"not a model")
    assert hashlib.sha256(path.read_bytes()).hexdigest() != "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        TapirPointTracker.from_checkpoint(path, expected_sha256="0" * 64)
