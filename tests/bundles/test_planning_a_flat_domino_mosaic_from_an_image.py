"""planning-a-flat-domino-mosaic-from-an-image: the integer program, world poses, order, exits.

HiGHS (SciPy) runs every test; the CBC tests need PuLP (2.x/3.x with its bundled CBC, or 4.x with
cbcbox) and are skipped without it.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "planning-a-flat-domino-mosaic-from-an-image"
ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "skills" / BUNDLE

HAS_PULP = importlib.util.find_spec("pulp") is not None


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["plan_mosaic"].module


@pytest.fixture(scope="module")
def mm(script):
    return script.mm


def gradient(h=240, w=210):
    yy, xx = np.mgrid[0:h, 0:w]
    return (255 * (0.7 * xx / w + 0.3 * yy / h)).astype(np.uint8)


def two_tone(h=240, w=210):
    img = np.full((h, w), 235, np.uint8)
    img[:, : w // 2] = 20
    return img


def plan(script, image=None, **kw):
    kw.setdefault("solver", "highs")
    if image is None and "image_path" not in kw:
        image = gradient()
    return script.run(FakeContext(), image=image, **kw)


def footprint(p, length=0.048, width=0.024):
    """Axis-aligned (world, yaw 0 grids) box of a placed tile."""
    c = np.array(p["position"][:2])
    ax = np.abs(np.array(p["axis_world"][:2]))
    half = ax * length / 2 + (1 - ax) * width / 2
    return c - half, c + half


def crosses(p0, p1, lo, hi, shrink=0.001):
    lo, hi = lo + shrink, hi - shrink
    ts = np.linspace(0, 1, 2001)[:, None]
    pts = p0 + ts * (p1 - p0)
    return bool(((pts >= lo) & (pts <= hi)).all(axis=1).any())


# -- the integer program, against upstream -------------------------------------------------


def test_grid_choice_and_brightness_scale_are_upstreams(mm):
    # find_valid_row_col_combination's answers (110 fields per double-nine set).
    assert mm.grid_for_sets(110 * 18, 0.8) == (55, 36)
    assert mm.grid_for_sets(110 * 24, 1.0) == (55, 48)
    assert mm.grid_for_sets(110 * 8, 400 / 480) == (40, 22)
    # get_field_brightness' scale for double-nine, light pips.
    b = np.array([[10.0, 50.0], [200.0, 30.0]])
    assert np.allclose(mm.target_values(b, 9, False), [[-0.5, 1.6052631], [9.5, 0.5526315]])
    # Dark pips: the same scale turned round.
    assert np.allclose(mm.target_values(b, 6, True), 6 - mm.target_values(b, 6, False))
    assert len(mm.domino_tiles(6)) == 28 and len(mm.domino_tiles(9)) == 55
    assert mm.domino_tiles(9)[:3] == [(0, 0), (0, 1), (0, 2)]
    assert len(mm.placements(8, 7)) == 8 * 6 + 7 * 7


def test_upstream_cost_truncates_the_better_end_order(mm):
    target = np.array([[0.4, 0.6]])
    places = [((0, 0), (0, 1))]
    tiles = [(0, 1), (1, 1)]
    up = mm.cost_matrix(target, tiles, places, truncate=True)
    sq = mm.cost_matrix(target, tiles, places, truncate=False)
    assert sq[0, 0] == pytest.approx(0.16 + 0.16)  # 0 then 1
    assert up[0, 0] == 0 and up[1, 0] == 0  # int(0.32), int(0.36 + 0.16)
    # The end order uses the untruncated error: low value on the 0.4 cell.
    assert mm.lower_value_first(target, (0, 1), places[0])
    assert not mm.lower_value_first(np.array([[0.6, 0.4]]), (0, 1), places[0])


def brute_force(mm, target, max_pips, truncate):
    """Every domino tiling of the grid and every assignment of one set's tiles to it."""
    rows, cols = target.shape
    tiles = mm.domino_tiles(max_pips)
    places = mm.placements(rows, cols)
    cost = mm.cost_matrix(target, tiles, places, truncate)

    def tilings(covered, chosen):
        free = next((i for i in range(rows * cols) if not covered[i]), None)
        if free is None:
            yield list(chosen)
            return
        r, c = divmod(free, cols)
        for p, (a, b) in enumerate(places):
            if a == (r, c) and not covered[b[0] * cols + b[1]]:
                covered[free] = covered[b[0] * cols + b[1]] = True
                chosen.append(p)
                yield from tilings(covered, chosen)
                chosen.pop()
                covered[free] = covered[b[0] * cols + b[1]] = False

    best = math.inf
    for tiling in tilings([False] * (rows * cols), []):
        for perm in itertools.permutations(range(len(tiles))):
            best = min(best, sum(cost[t, p] for t, p in zip(perm, tiling, strict=True)))
    return best


@pytest.mark.parametrize("truncate", [True, False])
def test_solver_finds_the_brute_force_optimum_on_a_double_two_set(script, mm, truncate):
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (3, 4)).astype(np.uint8)
    img = cv2.resize(img, (40, 30), interpolation=cv2.INTER_NEAREST)
    out = plan(script, img, max_pips=2, rows=3, cols=4, cost="upstream" if truncate else "squared")
    assert out["route"] == "planned" and out["solver_status"] == "optimal"
    target = np.array(out["target"])
    assert out["objective"] == pytest.approx(brute_force(mm, target, 2, truncate), abs=1e-3)


@pytest.mark.skipif(not HAS_PULP, reason="PuLP not installed")
def test_cbc_and_highs_reach_the_same_optimum(script):
    a = plan(script, solver="cbc")
    b = plan(script, solver="highs")
    assert a["solver"] == "cbc" and b["solver"] == "highs"
    assert a["route"] == b["route"] == "planned"
    assert a["objective"] == pytest.approx(b["objective"])


# -- the plan --------------------------------------------------------------------------------


def test_one_set_uses_every_tile_once_and_covers_every_cell(script):
    out = plan(script, grid_origin=[0.4, 0.1, 0.02])
    assert out["route"] == "planned", out["detail"]
    assert (out["rows"], out["cols"]) == (8, 7)  # 210 x 240 px -> closest divisor pair of 56
    assert out["n_tiles"] == 28 and len(out["placements"]) == 28
    assert sorted(tuple(p["value"]) for p in out["placements"]) == [
        (a, b) for a in range(7) for b in range(a, 7)
    ]
    cells = [tuple(c) for p in out["placements"] for c in p["cells"]]
    assert len(set(cells)) == 56
    values = np.array(out["values"])
    for p in out["placements"]:
        (ra, ca), (rb, cb) = p["cells"]
        assert abs(ra - rb) + abs(ca - cb) == 1
        assert [values[ra, ca], values[rb, cb]] == p["value_ab"]
        assert p["value_ab"][0] <= p["value_ab"][1]  # end a carries the lower value
        assert p["set_index"] == 0
    assert out["per_set"] == [{"set_index": 0, "used": 28, "unused": []}]
    assert out["supply"] == {f"{a}-{b}": 1 for a in range(7) for b in range(a, 7)}
    assert out["grid"]["exact_usage"] is True
    assert json.loads(out["placements_json"])[0]["step"] == 0
    assert len(out["pose_sequence"]["waypoints"]) == 28
    assert out["first_pose"] == out["placements"][0]["pose"]
    assert out["cell_rmse"] < 2.0 and out["correlation"] > 0.7


def test_dark_pips_go_where_the_image_is_dark(script):
    img = two_tone()
    dark = np.array(plan(script, img)["values"])
    light = np.array(plan(script, img, pip_contrast="light_on_dark")["values"])
    half = dark.shape[1] // 2
    assert dark[:, :half].mean() > dark[:, half + 1 :].mean() + 2
    assert light[:, :half].mean() < light[:, half + 1 :].mean() - 2


def test_more_sets_keep_every_tile_count_and_split_copies_by_set(script):
    out = plan(script, sets=2)
    assert out["route"] == "planned"
    assert out["n_tiles"] == 56 and (out["rows"], out["cols"]) == (14, 8)
    copies = {}
    for p in out["placements"]:
        copies.setdefault(tuple(p["value"]), []).append(p["set_index"])
    assert all(sorted(v) == [0, 1] for v in copies.values()) and len(copies) == 28
    assert [s["used"] for s in out["per_set"]] == [28, 28]
    assert any("stretches the image" in m for m in out["problems"])  # 8/14 vs 210/240


def test_a_smaller_grid_chooses_from_spare_tiles(script):
    img = two_tone()
    full = plan(script, img)
    spare = plan(script, img, sets=2, rows=8, cols=7)
    assert spare["route"] == "planned" and spare["grid"]["exact_usage"] is False
    assert spare["n_tiles"] == 28
    counts = {}
    for p in spare["placements"]:
        counts[tuple(p["value"])] = counts.get(tuple(p["value"]), 0) + 1
    assert max(counts.values()) <= 2 and len(counts) < 28  # some tiles twice, some not at all
    assert sum(s["used"] for s in spare["per_set"]) == 28
    assert sum(len(s["unused"]) for s in spare["per_set"]) == 28
    assert spare["cell_rmse"] < full["cell_rmse"] - 0.5


# -- world poses -----------------------------------------------------------------------------


def test_world_poses_follow_origin_pitch_and_yaw(script):
    out = plan(
        script,
        grid_origin={
            "position": {"x": 0.5, "y": 0.2, "z": 0.01},
            "rotation": {"w": math.cos(math.pi / 4), "x": 0, "y": 0, "z": math.sin(math.pi / 4)},
        },
        pitch_m=0.03,
        arms=[{"arm_id": "left", "base": [0.0, 0.0, 0.0]}],
    )
    assert out["route"] == "planned"
    assert out["grid"]["yaw_rad"] == pytest.approx(math.pi / 2)
    for p in out["placements"]:
        (ra, ca), (rb, cb) = p["cells"]
        # Grid +x (columns) is world +y at 90 deg; grid -y (rows) is world +x.
        mid_r, mid_c = (ra + rb) / 2, (ca + cb) / 2
        assert p["position"][0] == pytest.approx(0.5 + mid_r * 0.03)
        assert p["position"][1] == pytest.approx(0.2 + mid_c * 0.03)
        assert p["position"][2] == pytest.approx(0.01) and p["top_z"] == pytest.approx(0.0175)
        axis = np.array(p["axis_world"][:2])
        expect = np.array([(rb - ra), (cb - ca)], float)
        assert np.allclose(axis, expect)
        ends = np.array(p["ends_world"])[:, :2]
        assert np.linalg.norm(ends[1] - ends[0]) == pytest.approx(0.048)
        assert np.allclose((ends[1] - ends[0]) / 0.048, axis)
        q = p["pose"]["rotation"]
        assert math.atan2(2 * q["w"] * q["z"], 1 - 2 * q["z"] ** 2) == pytest.approx(p["yaw_rad"])
        assert p["arm_id"] == "left"


def test_center_anchor_puts_the_grid_centre_on_the_origin(script):
    out = plan(script, grid_origin=[0.4, -0.1, 0.0], anchor="center")
    xy = np.array([p["position"][:2] for p in out["placements"]])
    assert np.allclose(xy.mean(axis=0), [0.4, -0.1])


def test_without_origin_the_plan_is_in_the_grid_frame(script):
    out = plan(script)
    assert out["route"] == "planned" and out["grid"]["frame"] == "grid"
    assert any("grid_origin" in m for m in out["problems"])
    # The default arm sits beyond the image's bottom edge: the top row goes first.
    assert min(c[0] for c in out["placements"][0]["cells"]) == 0


# -- placing order ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arms",
    [
        None,
        [{"arm_id": 0, "base": [0.25, 0.0, 0.0]}],
        [{"arm_id": "left", "base": [0.0, 0.3, 0.0]}, {"arm_id": "right", "base": [0.0, -0.3, 0]}],
    ],
)
def test_no_step_reaches_across_a_placed_tile(script, arms):
    out = plan(script, sets=2, grid_origin=[0.35, 0.1, 0.0], arms=arms)
    assert out["route"] == "planned"
    assert out["reach_over"] == 0
    bases = {
        a["arm_id"]: np.array(a["base"][:2]) for a in (arms or [{"arm_id": 0, "base": [0, 0]}])
    }
    placed = []
    for p in out["placements"]:
        base = bases[p["arm_id"]]
        c = np.array(p["position"][:2])
        assert not any(crosses(base, c, lo, hi) for lo, hi in placed), p["step"]
        placed.append(footprint(p))
    first, last = out["placements"][0], out["placements"][-1]
    assert first["distance_to_arm"] > last["distance_to_arm"]
    if arms and len(arms) == 2:
        assert set(out["by_arm"]) == {"left", "right"}
        for p in out["placements"]:
            c = np.array(p["position"][:2])
            d = {k: np.linalg.norm(c - b) for k, b in bases.items()}
            assert d[p["arm_id"]] == min(d.values())
        assert sorted(sum(out["by_arm"].values(), [])) == list(range(out["n_tiles"]))


def test_finger_room_needs_a_gap_of_a_fingertip(script):
    tight = plan(script, grid_origin=[0.35, 0.1, 0.0])
    assert tight["steps_without_finger_room"] > 0
    assert any("no grasp axis" in m for m in tight["problems"])
    assert tight["placements"][0]["finger_room"] == ["across", "along"]
    loose = plan(script, grid_origin=[0.35, 0.1, 0.0], pitch_m=0.037)
    assert loose["steps_without_finger_room"] == 0
    assert all("across" in p["finger_room"] for p in loose["placements"])


# -- inputs and preview ----------------------------------------------------------------------


def test_image_from_file_rgb_and_camera_frame_agree(script, tmp_path):
    grey = gradient()
    path = tmp_path / "in.png"
    cv2.imwrite(str(path), grey)
    rgb = np.stack([grey] * 3, axis=-1)
    a = plan(script, grey)
    b = plan(script, image_path=str(path))
    c = plan(script, rgb)
    d = plan(script, {"rgb": rgb, "depth": None})
    assert a["target"] == b["target"] == c["target"] == d["target"]
    assert a["objective"] == b["objective"] == c["objective"] == d["objective"]


def test_preview_shows_the_pips_and_is_written(script, tmp_path):
    path = tmp_path / "preview.png"
    out = plan(script, preview_path=str(path), preview_cell_px=20)
    prev = out["preview"]
    assert prev.shape == (8 * 20, 7 * 20, 3) and out["target_preview"].shape == prev.shape
    assert out["preview_path"] == str(path) and cv2.imread(str(path)) is not None
    values = np.array(out["values"])
    dark = np.array(
        [
            [
                (prev[r * 20 + 3 : r * 20 + 17, c * 20 + 3 : c * 20 + 17].mean(axis=2) < 80).sum()
                for c in range(7)
            ]
            for r in range(8)
        ]
    )
    assert np.corrcoef(dark.ravel(), values.ravel())[0, 1] > 0.9


# -- exits -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw,why",
    [
        ({"image": None}, "no `image`"),
        ({"image": None, "image_path": "/nonexistent/x.png"}, "cannot read"),
        ({"image": np.full((100, 100), 128, np.uint8)}, "no contrast"),
        ({"image": gradient(5, 5)}, "smaller than"),
        ({"image": np.zeros((10, 10, 2), np.uint8)}, "channels"),
    ],
)
def test_invalid_image(script, kw, why):
    kw.setdefault("solver", "highs")
    out = script.run(FakeContext(), **kw)
    assert out["route"] == "invalid_image" and why in out["detail"]
    assert out["placements"] == []


def test_not_enough_sets(script):
    assert plan(script, sets=0)["route"] == "not_enough_sets"
    out = plan(script, sets=1, rows=10, cols=8)
    assert out["route"] == "not_enough_sets" and "needs 2" in out["detail"]


@pytest.mark.parametrize(
    "kw",
    [
        {"rows": 7, "cols": 7},
        {"rows": 8},
        {"pitch_m": 0.02},
        {"pip_contrast": "grey"},
        {"solver": "gurobi"},
        {"anchor": "corner"},
        {"grid_origin": "here"},
        {"arms": [{"arm_id": 0}]},
        {"max_pips": 12},
        {"colour": "red"},
    ],
)
def test_invalid_input(script, kw):
    assert plan(script, **kw)["route"] == "invalid_input"


def test_out_of_reach_before_solving(script):
    out = plan(
        script,
        grid_origin=[0.4, 0.0, 0.0],
        arms=[{"arm_id": 0, "base": [0, 0, 0], "reach_m": 0.5}],
    )
    assert out["route"] == "out_of_reach"
    assert out["out_of_reach"] and out["solver"] == ""
    assert all(np.hypot(0.4 + c * 0.026, -r * 0.026) > 0.5 for r, c in out["out_of_reach"])


def fake_solution(mm, status, chosen=None):
    return mm.Solution(chosen or [], status, "highs", None, None, 0.1, "fake")


def test_solver_timeout_and_suboptimal(script, mm, monkeypatch):
    monkeypatch.setattr(
        mm.DominoModel, "solve", lambda self, s, t: fake_solution(mm, "no_solution")
    )
    out = plan(script, time_limit_s=1)
    assert out["route"] == "solver_timeout" and "no plan in 1 s" in out["detail"]

    real = mm.DominoModel._solve_highs

    def feasible(self, s, t):
        sol = real(self, t)
        sol.status = "feasible"
        return sol

    monkeypatch.setattr(mm.DominoModel, "solve", feasible)
    assert plan(script, accept_suboptimal=False)["route"] == "solver_timeout"
    out = plan(script)
    assert out["route"] == "planned" and any("not proven optimal" in m for m in out["problems"])


def test_solver_unavailable_and_failed(script, mm, monkeypatch):
    def missing(self, s, t):
        raise mm.SolverUnavailable("PuLP not installed; scipy too old")

    monkeypatch.setattr(mm.DominoModel, "solve", missing)
    assert plan(script)["route"] == "solver_unavailable"
    monkeypatch.setattr(
        mm.DominoModel, "solve", lambda self, s, t: fake_solution(mm, "optimal", [(0, 0)])
    )
    out = plan(script)
    assert out["route"] == "solver_failed" and "not a tiling" in out["detail"]


def test_real_time_limit_on_highs(script):
    # 16 sets of double-six: ~48k binaries; 0.01 s is not enough to find a tiling.
    out = plan(script, gradient(480, 420), sets=16, time_limit_s=0.01, accept_suboptimal=False)
    assert out["route"] == "solver_timeout"


@pytest.mark.skipif(not HAS_PULP, reason="PuLP not installed")
def test_cbc_time_limit_routes_timeout(script):
    out = plan(
        script,
        gradient(480, 420),
        sets=16,
        solver="cbc",
        time_limit_s=0.01,
        accept_suboptimal=False,
    )
    assert out["route"] == "solver_timeout" and out["solver"] == "cbc"


def test_every_route_is_declared_in_the_example():
    ex = json.loads((SKILL / "examples" / "canonical_subgraph.json").read_text())
    declared = set(ex["exit"]["success_values"]) | {ex["on_error"]}
    src = (SKILL / "scripts" / "plan_mosaic.py").read_text()
    routes = set(re.findall(r'_Route\(\s*"(\w+)"', src))
    routes |= set(re.findall(r'route = "(\w+)"', src))
    routes |= set(re.findall(r'_empty\(\s*"(\w+)"', src))
    routes |= {"planned", "solver_timeout", "solver_failed"}
    assert routes <= declared, routes - declared
    skill_md = (SKILL / "SKILL.md").read_text()
    for r in routes:
        assert f"    {r}:" in skill_md
