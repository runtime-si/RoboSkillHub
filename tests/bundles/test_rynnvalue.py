"""rynnvalue bundle: frame selection, meta defaults and Analysis parsing with the model replaced by a fake."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def rv(skills_registry):
    return skills_registry.get("rynnvalue").tools_module


class FakeRuntime:
    name = "fake-rynnvalue"

    def __init__(self, text="- Video Description: a brick is placed.\n- Match: Yes\n- Success: No"):
        self.text = text
        self.calls = []

    def run(self, images, instruction, robot, camera, analysis, max_new_tokens):
        self.calls.append(
            {
                "images": images,
                "instruction": instruction,
                "robot": robot,
                "camera": camera,
                "analysis": analysis,
            }
        )
        n = len(images)
        return {
            "remaining": [float(n - i) for i in range(n)],
            "entropy": [0.5] * n,
            "delta": [-1.0] * max(n - 1, 0),
            "text": self.text if analysis else "",
            "timing": {"value_s": 0.1, "analysis_s": 0.2},
        }


def frame(v=0, h=48, w=64):
    return np.full((h, w, 3), v, dtype=np.uint8)


def test_tool_registered_with_its_schema(tool_registry):
    t = tool_registry.get("rynnvalue.remaining_time")
    assert {"images", "instruction"} <= set(t.schema.inputs)
    assert {"remaining_s", "final_remaining_s", "match", "success"} <= set(t.schema.outputs)


def test_import_does_not_pull_torch(rv):
    import sys

    assert "torch" not in sys.modules or not hasattr(rv, "torch")


@pytest.mark.parametrize(
    ("n", "want"),
    [
        (0, []),
        (1, [0]),
        (8, list(range(8))),
        (9, [0, 1, 2, 3, 4, 5, 6, 8]),
        (30, [0, 4, 8, 12, 16, 20, 24, 29]),
    ],
)
def test_select_frames_is_uniform_and_keeps_the_last(rv, n, want):
    assert rv.select_frames(n) == want


def test_remaining_time_returns_per_frame_values_and_parsed_analysis(rv):
    fake = FakeRuntime()
    out = rv.remaining_time_impl(
        [frame(i) for i in range(3)], "put the brick on the plate", runtime=fake
    )
    assert out["remaining_s"] == [3.0, 2.0, 1.0] and out["final_remaining_s"] == 1.0
    assert out["match"] is True and out["success"] is False
    assert out["description"] == "a brick is placed."
    assert out["frames_used"] == 3 and out["model"] == "fake-rynnvalue"
    call = fake.calls[0]
    assert call["robot"] == rv.DEFAULT_ROBOT and call["camera"] == rv.DEFAULT_CAMERA


def test_given_meta_is_passed_through(rv):
    fake = FakeRuntime()
    rv.remaining_time_impl([frame()], "x", robot_description="a YAM arm", runtime=fake)
    assert fake.calls[0]["robot"] == "a YAM arm" and fake.calls[0]["camera"] is None


def test_long_clips_are_subsampled_to_eight_frames(rv):
    fake = FakeRuntime()
    out = rv.remaining_time_impl([frame(i) for i in range(20)], "x", runtime=fake)
    assert len(fake.calls[0]["images"]) == 8 and out["frames_used"] == 8
    assert np.asarray(fake.calls[0]["images"][-1])[0, 0, 0] == 19


def test_large_frames_are_resized_and_float_frames_accepted(rv):
    fake = FakeRuntime()
    big = np.zeros((960, 1280, 3), dtype=np.float32)
    rv.remaining_time_impl([big], "x", runtime=fake)
    assert max(fake.calls[0]["images"][0].size) == rv.MAX_IMAGE_SIDE


def test_missing_analysis_fields_are_none(rv):
    out = rv.remaining_time_impl([frame()], "x", analysis=False, runtime=FakeRuntime())
    assert out["match"] is None and out["success"] is None and out["analysis"] == ""


def test_parse_analysis_matches_upstream_layout(rv):
    p = rv.parse_analysis("- Video Description: arm stacks.\n- Match: no\n- Success: YES")
    assert p == {"description": "arm stacks.", "match": False, "success": True}


@pytest.mark.parametrize("bad", [[], [np.zeros((4, 4))]])
def test_bad_frames_raise(rv, bad):
    with pytest.raises(ValueError):
        rv.remaining_time_impl(bad, "x", runtime=FakeRuntime())


def test_instruction_is_required(rv):
    with pytest.raises(ValueError):
        rv.remaining_time_impl([frame()], "  ", runtime=FakeRuntime())
