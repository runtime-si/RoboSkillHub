"""verifying-a-domino-cascade-from-video: the ported onset rule on hand-built traces (the
design note's checks), then rendered RGB-D clips of toppling lines (CPU ray casting,
noise, a hand) through every exit, both cameras, the rgb-only signal and the optional
trackers faked from the render's ground truth."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from gap_core.errors import ToolError

from . import _cascade_render as R

BUNDLE = "verifying-a-domino-cascade-from-video"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
ROUTES = {
    "all_fell",
    "stopped_at",
    "not_triggered",
    "fell_before_trigger",
    "irregular",
    "unverifiable",
}


@pytest.fixture(scope="module")
def mod(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["verify_cascade"].module


_CLIPS: dict = {}


def clip(**kw):
    key = json.dumps(kw, sort_keys=True, default=str)
    if key not in _CLIPS:
        hands = kw.pop("hands", [])
        sc = R.Scene(**kw)
        sc.hands = hands
        _CLIPS[key] = R.render(sc)
    return _CLIPS[key]


def run(mod, c, ctx=None, **kw):
    return mod.run(ctx or FakeContext({}), slots=c["slots"], frames=c["frames"], **kw)


def truth_up_frame(c, i, up):
    k = np.flatnonzero(c["angles"][i] >= up)
    return int(k[0])


# ----------------------------------------------------------------------------- bundle


def test_example_declares_every_route():
    ex = json.loads((SKILL_DIR / "examples/canonical_subgraph.json").read_text())
    declared = set(ex["exit"]["success_values"]) | {ex["on_error"]}
    assert declared == ROUTES
    text = (SKILL_DIR / "scripts/verify_cascade.py").read_text()
    for r in ROUTES:
        assert f'"{r}"' in text
    meta = (SKILL_DIR / "SKILL.md").read_text()
    for r in ROUTES:
        assert f"    {r}:" in meta


# ----------------------------------------------------------------------------- the onset rule


def onset(mod, a, usable=None, up=15.0, conf=45.0):
    a = np.asarray(a, dtype=float)
    u = np.ones(len(a), bool) if usable is None else np.asarray(usable, bool)
    return mod.topple_onset(a, u, up, conf, 3)


def test_onset_of_a_clean_topple_is_the_frame_after_the_last_upright_one(mod):
    a = [0, 1, 0, 2, 10, 30, 60, 80, 80, 80]
    ev = onset(mod, a)
    assert ev["onset_frame"] == 5 and ev["fall_frame"] == 6 and ev["confirmed"]


def test_drift_from_4_to_13_degrees_gives_no_onset(mod):
    assert onset(mod, np.linspace(4, 13, 60)) is None


def test_a_29_degree_occlusion_spike_15_frames_early_is_not_the_onset(mod):
    # The design note's Mechanism A: one frame at 29 deg (and with a 25 deg confirm
    # threshold it still needs 3 frames), 15 frames before the real fall.
    a = [0.0] * 40
    a[10] = 29.0
    for k, v in enumerate([5, 20, 50, 75, 80, 80, 80]):
        a[25 + k] = v
    ev = onset(mod, a, conf=25.0)
    assert ev["onset_frame"] == 26
    assert onset(mod, a[:20], conf=25.0) is None


def test_a_two_frame_excursion_that_recovers_is_not_a_fall(mod):
    assert onset(mod, [0, 0, 50, 50, 5, 0, 0, 0, 0, 0]) is None


def test_gate_dropped_frames_are_missing_data_not_zeros(mod):
    a = [0, 0, 0, 0, 0, 20, 0, 0, 70, 80, 80, 80]
    usable = [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]  # frames 6-7 hidden mid-fall
    ev = onset(mod, a, usable)
    assert ev["onset_frame"] == 5 and ev["fall_frame"] == 8
    # Hidden frames neither break nor extend a run.
    a2 = [0, 0, 60, 0, 0, 60, 60]
    assert onset(mod, a2, [1, 1, 1, 0, 0, 1, 1])["fall_frame"] == 2


def test_a_run_reaching_the_end_counts_unconfirmed(mod):
    ev = onset(mod, [0, 0, 0, 0, 60, 70])
    assert ev["onset_frame"] == 4 and not ev["confirmed"]
    # One bad last frame is not a fall.
    assert onset(mod, [0, 0, 0, 0, 0, 88]) is None


def test_never_seen_upright_dates_to_the_first_usable_frame(mod):
    ev = onset(mod, [60, 60, 60, 60], [0, 1, 1, 1])
    assert ev["onset_frame"] == 1 and ev["last_upright_frame"] is None


def test_a_wobble_that_recovers_before_the_fall_is_skipped(mod):
    a = [0, 12, 18, 12, 3, 0, 0, 25, 50, 70, 80, 80]
    assert onset(mod, a)["onset_frame"] == 7


# ----------------------------------------------------------------------------- rendered clips


def test_all_fell_overhead_depth(mod):
    c = clip()
    out = run(mod, c)
    assert out["route"] == "all_fell", out["detail"]
    assert out["signal"] == "depth" and out["n_fell"] == 10 and out["stop_index"] == -1
    assert out["wave_speed"] == pytest.approx(0.8, rel=0.05)
    assert out["wave_rate"] == pytest.approx(0.8 / 0.037, rel=0.05)
    assert len(out["intervals"]) == 9
    for p in out["tiles"]:
        assert p["confirmed"]
        assert abs(p["onset_frame"] - truth_up_frame(c, p["index"], p["upright_deg"])) <= 1
        assert 0.0 <= p["onset_time"] - c["onsets"][p["index"]] <= 0.05
    assert out["first_onset_time"] < out["last_onset_time"]


def test_stopped_at_names_the_first_standing_tile(mod):
    c = clip(stop_at=6)
    out = run(mod, c)
    assert out["route"] == "stopped_at", out["detail"]
    assert out["stop_index"] == 6 and out["last_fallen_index"] == 5 and out["n_fell"] == 6
    slot = c["slots"][6]["position"]
    assert (
        math.hypot(out["stop_point"]["x"] - slot["x"], out["stop_point"]["y"] - slot["y"]) < 0.006
    )
    # The last fallen tile rests on the standing one at its contact angle.
    last = out["tiles"][5]
    assert last["fallen_deg"] < last["final_angle"] < last["contact_deg"] + 3
    assert [p["state"] for p in out["tiles"][6:]] == ["standing"] * 4


def test_not_triggered(mod):
    out = run(mod, clip(no_trigger=True, duration=0.5))
    assert out["route"] == "not_triggered" and out["n_fell"] == 0


def test_pusher_hovering_over_the_first_tile_makes_no_false_onset(mod):
    sc = R.Scene()
    p0 = R.slots_for(sc)[0]["position"]
    over = (p0["x"] + 0.01, p0["y"], 0.075)
    back = (p0["x"] - 0.12, p0["y"], 0.075)
    hand = R.Hand([(0.0, over), (sc.t_trigger - 0.04, over), (sc.t_trigger + 0.02, back)])
    c = clip(hands=[hand])
    trig = int(round(sc.t_trigger * sc.fps))
    out = run(mod, c, trigger_frame=trig)
    assert out["route"] == "all_fell", out["detail"]
    assert min(out["visibility"][0][: trig - 4]) < 0.5  # the hand was seen and gated
    assert out["tiles"][0]["onset_frame"] >= trig


def test_fell_before_trigger(mod):
    c = clip(early=(5, 0.05), t_trigger=0.3)
    out = run(mod, c, trigger_time=0.3)
    assert out["route"] == "fell_before_trigger", out["detail"]
    assert "5 (frame" in out["detail"]


def test_a_tile_beyond_the_stop_falling_is_irregular(mod):
    out = run(mod, clip(stop_at=4, lone=(7, 0.6), duration=0.9))
    assert out["route"] == "irregular" and "tile 4 stood" in out["detail"]


def test_a_tile_falling_while_the_first_stands_is_irregular(mod):
    out = run(mod, clip(no_trigger=True, lone=(5, 0.2), duration=0.6))
    assert out["route"] == "irregular"


def test_a_hand_parked_over_the_stop_tile_is_unverifiable(mod):
    sc = R.Scene(stop_at=5)
    p = R.slots_for(sc)[5]["position"]
    t_hide = sc.t_trigger + 2 * sc.pitch / sc.speed
    hand = R.Hand(
        [(t_hide, (p["x"] + 0.01, p["y"], 0.08)), (10.0, (p["x"] + 0.01, p["y"], 0.08))],
        size=(0.03, 0.08, 0.04),
    )
    out = run(mod, clip(stop_at=5, hands=[hand]))
    assert out["route"] == "unverifiable" and "tile 5" in out["detail"], out["detail"]
    assert out["stop_index"] == 5


def test_a_tile_down_before_the_clip_is_unverifiable(mod):
    out = run(mod, clip(lone=(3, -1.0), no_trigger=True, duration=0.4))
    assert out["route"] == "unverifiable" and "already down" in out["detail"]
    assert out["tiles"][3]["state"] == "down_at_start"


def test_mid_cascade_occlusion_dates_onsets_across_the_gap(mod):
    sc = R.Scene()
    pk = R.slots_for(sc)[4]["position"]
    t0 = sc.t_trigger + 2 * sc.pitch / sc.speed
    hand = R.Hand(
        [(t0, (pk["x"], pk["y"] - 0.15, 0.09)), (t0 + 0.12, (pk["x"], pk["y"] + 0.15, 0.09))],
        size=(0.06, 0.08, 0.04),
    )
    out = run(mod, clip(hands=[hand]))
    assert out["route"] == "all_fell"
    hidden = [p for p in out["tiles"] if p["onset_uncertainty_frames"] > 0]
    assert hidden, "the sweep hid some onset"
    assert out["wave_speed"] == pytest.approx(0.8, rel=0.08)


def test_wrist_camera_depth(mod):
    out = run(mod, clip(camera="wrist", stop_at=7))
    assert out["route"] == "stopped_at" and out["stop_index"] == 7, out["detail"]


def test_wrist_camera_with_a_hand_over_the_first_tile_and_no_push(mod):
    # The hand hides tile 0 in the first frames, where the depth noise is measured.
    sc = R.Scene(camera="wrist", no_trigger=True, duration=0.5)
    p0 = R.slots_for(sc)[0]["position"]
    over = (p0["x"] + 0.01, p0["y"], 0.075)
    back = (p0["x"] - 0.12, p0["y"], 0.075)
    hand = R.Hand([(0.0, over), (sc.t_trigger - 0.04, over), (sc.t_trigger + 0.02, back)])
    out = run(mod, clip(camera="wrist", no_trigger=True, duration=0.5, hands=[hand]))
    assert out["route"] == "not_triggered", out["detail"]
    assert not any(p["inferred"] for p in out["tiles"])


def test_rgb_only_clip(mod):
    c = clip(stop_at=6)
    frames = [{k: v for k, v in f.items() if k != "depth"} for f in c["frames"]]
    out = mod.run(FakeContext({}), slots=c["slots"], frames=frames)
    assert out["signal"] == "rgb"
    assert out["route"] == "stopped_at" and out["stop_index"] == 6, out["detail"]
    assert out["wave_speed"] == pytest.approx(0.8, rel=0.1)


def test_rgb_with_a_hand_the_colour_of_neither(mod):
    sc = R.Scene()
    p0 = R.slots_for(sc)[0]["position"]
    over = (p0["x"] + 0.01, p0["y"], 0.075)
    hand = R.Hand(
        [
            (0.0, over),
            (sc.t_trigger - 0.04, over),
            (sc.t_trigger + 0.02, (p0["x"] - 0.12, p0["y"], 0.075)),
        ]
    )
    out = run(mod, clip(hands=[hand]), signal="rgb")
    assert out["route"] == "all_fell"
    assert min(out["visibility"][0]) < 0.5


def test_an_arc_from_the_slot_list(mod):
    out = run(mod, clip(layout="arc", stop_at=8, n=10))
    assert out["route"] == "stopped_at" and out["stop_index"] == 8, out["detail"]


def test_30_fps_flags_unresolvable_order_but_still_decides(mod):
    out = run(mod, clip(fps=30.0, stop_at=7))
    assert out["route"] == "stopped_at" and out["stop_index"] == 7
    assert any("not resolvable" in p for p in out["problems"])


def test_straight_line_from_first_slot_pitch_and_n_tiles(mod):
    c = clip(stop_at=6)
    out = mod.run(
        FakeContext({}), first_slot=c["slots"][0], pitch=0.037, n_tiles=10, frames=c["frames"]
    )
    assert out["route"] == "stopped_at" and out["stop_index"] == 6


def test_reader_tiles_correct_slots_that_are_off(mod):
    c = clip(stop_at=6)
    off = [
        {**s, "position": {**s["position"], "x": s["position"]["x"] + 0.012}} for s in c["slots"]
    ]
    tiles = [
        {"center_world": {"x": p[0] - 0.00375 * f[0], "y": p[1], "z": 0.024}}
        for p, f in c["placed"]
    ]
    out = mod.run(FakeContext({}), slots=off, tiles=tiles, frames=c["frames"])
    assert out["route"] == "stopped_at" and out["stop_index"] == 6, out["detail"]


def test_stacked_arrays_and_nanosecond_timestamps(mod):
    c = clip(stop_at=6)
    stacked = {
        "rgb": np.stack([f["rgb"] for f in c["frames"]]),
        "depth": np.stack([f["depth"] for f in c["frames"]]),
        "intrinsics": c["K"],
        "pose": c["T"],
        "timestamps": [int(1_700_000_000e9 + t * 1e9) for t in c["times"]],
    }
    out = mod.run(FakeContext({}), slots=c["slots"], frames=stacked)
    assert out["route"] == "stopped_at" and out["times"][1] == pytest.approx(1 / 60, abs=1e-4)


def test_fps_fallback_without_timestamps(mod):
    c = clip(stop_at=6)
    frames = [{k: v for k, v in f.items() if k != "timestamp"} for f in c["frames"]]
    out = mod.run(FakeContext({}), slots=c["slots"], frames=frames)
    assert any("30 fps assumed" in p for p in out["problems"])
    assert out["wave_speed"] == pytest.approx(0.8 * 30 / 60, rel=0.06)
    out = mod.run(FakeContext({}), slots=c["slots"], frames=frames, fps=60.0)
    assert out["wave_speed"] == pytest.approx(0.8, rel=0.06)


def test_records_frames_from_the_observation(mod):
    c = clip(stop_at=6)
    obs = [{"cameras": [{**f, "name": "overhead"}], "arms": []} for f in c["frames"]]
    ctx = FakeContext({"robot.get_observation": obs})
    out = mod.run(ctx, slots=c["slots"], camera_name="overhead", record_frames=len(obs))
    assert out["route"] == "stopped_at" and out["stop_index"] == 6
    assert len(ctx.calls_to("robot.get_observation")) == len(obs)


@pytest.mark.parametrize(
    "kw, phrase",
    [
        ({"slots": []}, "no slots"),
        ({"frames": None}, "no frames"),
        ({"frames": "one"}, "at least two"),
        ({"signal": "lidar"}, "unknown signal"),
    ],
)
def test_bad_inputs_are_unverifiable(mod, kw, phrase):
    c = clip(stop_at=6)
    args = {"slots": c["slots"], "frames": c["frames"], **kw}
    if args["frames"] == "one":
        args["frames"] = c["frames"][:1]
    out = mod.run(FakeContext({}), **args)
    assert out["route"] == "unverifiable" and phrase in out["detail"]


# ----------------------------------------------------------------------------- optional trackers (faked)


def _top_pixels(c, i):
    h, _, d = R.TILE
    piv, f = c["placed"][i]
    K, T = c["K"], c["T"]
    out = []
    for a in np.radians(c["angles"][i]):
        x = -d / 2 * math.cos(a) + h * math.sin(a)
        z = d / 2 * math.sin(a) + h * math.cos(a)
        p = piv + x * f + np.array([0, 0, z])
        pc = (p - T[:3, 3]) @ T[:3, :3]
        out.append([K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]])
    return out


def fake_track(c):
    def points(frames, query_points, query_frame=0, **_):
        tracks = []
        for i in range(len(query_points)):
            px = _top_pixels(c, i)
            vis = []
            for k, (u, v) in enumerate(px):
                ui, vi = int(round(u)), int(round(v))
                ids = c["ids"][k][max(vi - 1, 0) : vi + 2, max(ui - 1, 0) : ui + 2]
                vis.append(bool((ids != 100).all()))
            tracks.append({"pixels": px, "visible": vis, "confidence": [0.9] * len(px)})
        return {"found": True, "tracks": tracks}

    return points


def test_track_tracker(mod):
    c = clip(stop_at=6)
    ctx = FakeContext({"track.points": fake_track(c)})
    out = run(mod, c, ctx=ctx, tracker="track")
    assert out["signal"] == "track" and out["tracker_fallback"] == ""
    assert out["route"] == "stopped_at" and out["stop_index"] == 6, out["detail"]
    q = ctx.calls_to("track.points")[0].kwargs
    assert len(q["query_points"]) == 10 and q["query_frame"] == 0


def fake_sam3(c, drop_tile=None):
    state = {}

    def init(image, box=None, object_name="", **_):
        i = int(object_name.split()[-1])
        state[f"t{i}"] = 0
        return {
            "tracker_id": f"t{i}",
            "initial_mask": (c["ids"][0] == i + 1).astype(np.uint8) * 255,
            "score": 0.9,
            "object_present": True,
        }

    def update(tracker_id, image):
        state[tracker_id] += 1
        i, k = int(tracker_id[1:]), state[tracker_id]
        m = c["ids"][k] == i + 1
        if i == drop_tile or not m.any():
            return {"mask": None, "box": None, "confidence": 0.0, "object_present": False}
        return {
            "mask": m.astype(np.uint8) * 255,
            "box": None,
            "confidence": 0.9,
            "object_present": True,
        }

    return {
        "sam3.tracker_init": init,
        "sam3.tracker_update": update,
        "sam3.tracker_close": {"closed": True},
    }


def test_sam3_tracker(mod):
    c = clip(stop_at=6)
    ctx = FakeContext(fake_sam3(c))
    out = run(mod, c, ctx=ctx, tracker="sam3")
    assert out["signal"] == "sam3"
    assert out["route"] == "stopped_at" and out["stop_index"] == 6, out["detail"]
    assert len(ctx.calls_to("sam3.tracker_close")) == 10


def test_sam3_losing_the_stop_tile_is_unverifiable(mod):
    c = clip(stop_at=6)
    out = run(mod, c, ctx=FakeContext(fake_sam3(c, drop_tile=6)), tracker="sam3")
    assert out["route"] == "unverifiable" and "tile 6" in out["detail"]


def test_tracker_failure_falls_back_to_the_cpu_signal(mod):
    c = clip(stop_at=6)

    def broken(**_):
        raise ToolError("track.points", "no GPU host")

    out = run(mod, c, ctx=FakeContext({"track.points": broken}), tracker="track")
    assert out["signal"] == "depth" and "no GPU host" in out["tracker_fallback"]
    assert out["route"] == "stopped_at" and out["stop_index"] == 6
