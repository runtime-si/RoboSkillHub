"""tracking-identical-parts-by-id: the ByteTrack port and the skill's routing, with FakeContext."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

BUNDLE = "tracking-identical-parts-by-id"


@pytest.fixture(scope="module")
def bundle(skills_registry):
    b = skills_registry.get(BUNDLE)
    assert b.kind == "skill"
    return b


@pytest.fixture(scope="module")
def script(bundle):
    return bundle.canonical_scripts["track_parts"].module


@pytest.fixture(scope="module")
def bt(bundle):
    path = Path(bundle.bundle_dir) / "scripts" / "bytetrack.py"
    spec = importlib.util.spec_from_file_location("bytetrack_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def det(cx, cy, score=0.9, size=20.0, label=""):
    h = size / 2
    return {"box": [cx - h, cy - h, cx + h, cy + h], "score": score, "label": label}


ROW = [(100.0 + 40 * i, 200.0) for i in range(5)]


def frame(points=ROW, score=0.9, skip=()):
    return [det(x, y, score) for i, (x, y) in enumerate(points) if i not in skip]


def ids_by_x(tracker):
    return {
        round(float(tracker.box(t)[0] + 10)): t["id"]
        for t in tracker.tracks
        if t["state"] == "tracked"
    }


# --- the port ----------------------------------------------------------------


def test_assignment_matches_only_pairs_under_the_limit(bt):
    cost = np.array([[0.1, 0.9], [0.95, 0.85], [0.2, 0.3]])
    matches, ur, uc = bt.linear_assignment(cost, 0.8)
    assert sorted(matches) == [(0, 0), (2, 1)]
    assert ur == [1] and uc == []
    assert bt.linear_assignment(np.zeros((0, 3)), 0.8) == ([], [], [0, 1, 2])


def test_assignment_minimises_the_total_with_unmatched_pairs_costing_the_limit(bt):
    # lapjv with cost_limit: leaving a row and a column unmatched costs the limit
    matches, _, _ = bt.linear_assignment(np.array([[0.1, 0.3], [0.3, 0.79]]), 0.8)
    assert sorted(matches) == [(0, 1), (1, 0)]  # 0.6 < 0.1 + 0.79
    matches, ur, uc = bt.linear_assignment(np.array([[0.1, 0.5], [0.5, 0.9]]), 0.8)
    assert matches == [(0, 0)] and ur == [1] and uc == [1]  # 0.1 + 0.8 < 0.5 + 0.5


def test_kalman_update_pulls_to_the_measurement(bt):
    mean, cov = bt.kf_initiate(np.array([10.0, 20.0, 1.0, 30.0]))
    mean, cov = bt.kf_predict(mean, cov)
    mean2, cov2 = bt.kf_update(mean, cov, np.array([14.0, 20.0, 1.0, 30.0]))
    assert 10.0 < mean2[0] < 14.0
    assert np.trace(cov2) < np.trace(cov)
    assert np.allclose(cov2, cov2.T)


def test_static_parts_keep_ids_and_new_tracks_confirm_on_the_second_frame(bt):
    tr = bt.ByteTracker()
    ev = tr.update(frame())
    assert ev["new"] == [1, 2, 3, 4, 5]
    assert all(t["activated"] for t in tr.tracks)  # frame 1 activates at once (upstream)
    first = ids_by_x(tr)
    for _ in range(10):
        tr.update(frame())
    assert ids_by_x(tr) == first
    tr.update(frame() + [det(400, 300)])
    late = [t for t in tr.tracks if t["id"] == 6][0]
    assert not late["activated"] and all(o["id"] != 6 for o in tr.outputs())
    tr.update(frame() + [det(400, 300)])
    assert late["activated"] and any(o["id"] == 6 for o in tr.outputs())


def test_a_covered_part_is_lost_then_refound_with_its_id(bt):
    tr = bt.ByteTracker(track_buffer=30)
    tr.update(frame())
    ev = tr.update(frame(skip=(2,)))
    assert ev["lost"] == [3]
    for _ in range(20):
        tr.update(frame(skip=(2,)))
    lost = [o for o in tr.outputs() if o["id"] == 3][0]
    assert lost["state"] == "lost" and lost["frames_lost"] == 21
    ev = tr.update(frame())
    assert ev["refound"] == [3] and ev["new"] == []


def test_a_part_covered_past_the_buffer_comes_back_with_a_new_id(bt):
    tr = bt.ByteTracker(track_buffer=5)
    tr.update(frame())
    for _ in range(7):
        ev = tr.update(frame(skip=(2,)))
    assert 3 in ev["removed"] or all(t["id"] != 3 for t in tr.tracks)
    ev = tr.update(frame())
    assert ev["new"] == [6]


def test_low_score_boxes_keep_a_track_but_do_not_start_one(bt):
    tr = bt.ByteTracker(track_thresh=0.5)
    tr.update([det(100, 100, 0.9)])
    ev = tr.update([det(100, 100, 0.3), det(300, 300, 0.3)])
    assert ev["lost"] == [] and ev["new"] == []
    assert [t["id"] for t in tr.tracks] == [1]


def test_fuse_score_makes_low_scores_unmatchable(bt):
    for fuse, expect_lost in ((True, [1]), (False, [])):
        tr = bt.ByteTracker(track_thresh=0.1, low_thresh=0.05, fuse_score=fuse)
        tr.update([det(100, 100, 0.25)])
        ev = tr.update([det(101, 100, 0.15)])
        assert ev["lost"] == expect_lost


def test_labels_never_share_a_track(bt):
    tr = bt.ByteTracker()
    tr.update([det(100, 100, label="red")])
    ev = tr.update([det(100, 100, label="blue")])
    assert ev["lost"] == [1] and ev["new"] == [2]


def test_freeze_lost_keeps_a_covered_part_where_it_was(bt):
    moving = [det(100 + 3 * k, 100) for k in range(6)]
    boxes = {}
    for freeze in (True, False):
        tr = bt.ByteTracker(freeze_lost=freeze)
        for d in moving:
            tr.update([d])
        for _ in range(10):
            tr.update([])
        boxes[freeze] = tr.box(tr.tracks[0])
    last_seen = moving[-1]["box"][0]
    assert abs(boxes[True][0] - last_seen) < abs(boxes[False][0] - last_seen)


def test_state_round_trips_through_json(bt):
    a = bt.ByteTracker()
    b_state = None
    seq = [frame(), frame(skip=(1,)), frame(), frame(skip=(0, 4)), frame()]
    for f in seq:
        a.update(f)
        b = bt.ByteTracker(json.loads(json.dumps(b_state)) if b_state else None)
        b.update(f)
        b_state = b.to_state()
    assert [(o["id"], o["state"], o["box"]) for o in a.outputs()] == [
        (o["id"], o["state"], o["box"]) for o in bt.ByteTracker(b_state).outputs()
    ]


# --- the skill ---------------------------------------------------------------


def test_routes_updated_then_lost(script):
    out = script.run(FakeContext({}), detections=frame())
    assert (
        out["route"] == "tracks_updated"
        and out["n_tracked"] == 5
        and out["new_ids"] == [1, 2, 3, 4, 5]
    )
    out = script.run(FakeContext({}), state=out["state"], detections=frame(skip=(3,)))
    assert out["route"] == "track_lost" and out["lost_ids"] == [4]
    assert out["n_lost"] == 1 and out["n_tracked"] == 4
    assert [t["id"] for t in out["tracks"] if t["state"] == "lost"] == [4]
    out = script.run(FakeContext({}), state=out["state"], detections=frame())
    assert out["route"] == "tracks_updated" and out["refound_ids"] == [4]


def test_a_part_moved_while_covered_routes_id_switch_suspected(script):
    state = None
    for f in (frame(), frame(), frame(skip=(2,)), frame(skip=(2,))):
        state = script.run(FakeContext({}), state=state, detections=f)["state"]
    moved = list(ROW)
    moved[2] = (ROW[2][0] + 25, ROW[2][1] + 3)
    out = script.run(FakeContext({}), state=state, detections=frame(moved))
    assert out["route"] == "id_switch_suspected"
    (s,) = out["suspected_switches"]
    assert (
        s["lost_id"] == 3 and s["new_id"] == 6 and s["distance_px"] == pytest.approx(25.2, abs=0.2)
    )


def test_a_new_box_far_away_or_of_another_size_is_not_suspected(script):
    state = None
    for f in (frame(), frame(skip=(2,))):
        state = script.run(FakeContext({}), state=state, detections=f)["state"]
    far = frame(skip=(2,)) + [det(600, 400)]
    big = frame(skip=(2,)) + [det(ROW[2][0], ROW[2][1], size=60)]
    for dets in (far, big):
        out = script.run(FakeContext({}), state=state, detections=dets)
        assert out["route"] == "tracks_updated" and out["new_ids"] == [6]


def test_detector_path_calls_owlv2_with_the_prompt_and_reads_box_dicts(script):
    calls = []

    def owl(**kw):
        calls.append(kw)
        return {
            "detections": [
                {"box": {"x1": 10, "y1": 10, "x2": 30, "y2": 26}, "score": 0.2, "label": "domino"}
            ]
        }

    obs = {
        "cameras": [
            {"name": "left_wrist", "rgb": np.zeros((4, 4, 3), np.uint8)},
            {"name": "top", "rgb": np.ones((4, 4, 3), np.uint8)},
        ]
    }
    ctx = FakeContext({"owlv2.detect": owl, "robot.get_observation": obs})
    out = script.run(ctx, prompts=["domino", "brick"])
    (call,) = calls
    assert call["query"] == "domino. brick" and call["threshold"] == pytest.approx(0.05)
    assert int(call["image"][0, 0, 0]) == 1  # the external camera
    assert out["tracks"][0]["box"] == [10.0, 10.0, 30.0, 26.0]
    assert out["tracks"][0]["label"] == "domino"


def test_detector_args_and_other_detectors(script):
    calls = []

    def gd(**kw):
        calls.append(kw)
        return {"detections": []}

    ctx = FakeContext({"grounding-dino.detect": gd})
    obs = {"cameras": [{"name": "top", "rgb": np.zeros((2, 2, 3), np.uint8)}]}
    out = script.run(
        ctx,
        prompts="cube",
        detector="grounding-dino.detect",
        observation=obs,
        detector_args={"text_threshold": 0.2},
    )
    assert calls[0]["box_threshold"] == pytest.approx(0.05) and calls[0]["text_threshold"] == 0.2
    assert out["route"] == "tracks_updated" and out["n_detections"] == 0


def _raise(exc):
    def f(**_kw):
        raise exc

    return f


@pytest.mark.parametrize(
    ("tools", "kwargs", "why"),
    [
        ({"robot.get_observation": {"cameras": []}}, {}, "no RGB"),
        (
            {
                "owlv2.detect": _raise(ToolError("owlv2.detect", "CUDA OOM")),
                "robot.get_observation": {
                    "cameras": [{"name": "top", "rgb": np.zeros((2, 2, 3), np.uint8)}]
                },
            },
            {},
            "CUDA OOM",
        ),
    ],
)
def test_no_frame_keeps_the_state(script, tools, kwargs, why):
    state = script.run(FakeContext({}), detections=frame())["state"]
    out = script.run(FakeContext(tools), state=state, prompts="cube", **kwargs)
    assert out["route"] == "no_frame" and why in out["reason"]
    assert out["state"] == state


def test_needs_detections_or_prompts(script):
    with pytest.raises(ValueError, match="detections or prompts"):
        script.run(FakeContext({}))


def test_bad_boxes_are_dropped_and_labels_can_be_ignored(script):
    dets = [{"box": [5, 5, 1, 1], "score": 0.9}, {"box": [1, 2, 3]}, det(50, 50, label="a")]
    out = script.run(FakeContext({}), detections=dets, use_labels=False)
    assert out["n_detections"] == 1 and out["tracks"][0]["label"] == ""


def test_example_declares_every_route(bundle):
    ex = json.loads((Path(bundle.bundle_dir) / "examples" / "canonical_subgraph.json").read_text())
    declared = set(ex["exit"]["success_values"]) | {ex["on_error"]}
    assert declared == {"tracks_updated", "track_lost", "id_switch_suspected", "no_frame"}
