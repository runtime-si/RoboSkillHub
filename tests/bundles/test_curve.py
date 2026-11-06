"""curve bundle: lazy-import discipline + CPU numerics on a synthetic arc.

The bundle is out-of-process (``gap.serving``), so ``load_skills`` registers
its synthetic package ``gap_skills.tools.curve`` but does not import
``tools.py``; the numeric tests import ``_impl`` and ``tools`` through that
package the way the bundle itself does.
"""

from __future__ import annotations

import ast
import importlib
import math
from pathlib import Path

import numpy as np
import pytest

SKILLS_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = SKILLS_ROOT / "tools" / "curve"

#: Packages that must only load inside function bodies (stdout is the RPC
#: channel, and the in-process loader must stay cheap).
_LAZY_ONLY = {"cv2", "skimage", "PIL", "scipy.interpolate"}


def _module_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in tree.body:  # top level only -- imports inside defs are the point
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


@pytest.mark.parametrize("filename", ["tools.py", "_impl.py"])
def test_heavy_imports_are_lazy(filename):
    imported = _module_level_imports(BUNDLE / filename)
    leaked = {
        name
        for name in imported
        if any(name == lazy or name.startswith(lazy + ".") for lazy in _LAZY_ONLY)
    }
    assert not leaked, f"{filename}: module-level import of {sorted(leaked)}"


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

_H, _W = 240, 320
_RADIUS_PX = 120.0
_HALF_WIDTH_PX = 3.0  # 6 px wide
_DEPTH_M = 0.8
_FX = 400.0
_K = np.array([[_FX, 0.0, 160.0], [0.0, _FX, 120.0], [0.0, 0.0, 1.0]])
_PIXEL_M = _DEPTH_M / _FX


@pytest.fixture(scope="module")
def impl(skills_registry):
    pytest.importorskip("skimage")
    pytest.importorskip("cv2")
    assert skills_registry.get("curve").kind == "tool"
    return importlib.import_module("gap_skills.tools.curve._impl")


@pytest.fixture(scope="module")
def tools(impl):
    return importlib.import_module("gap_skills.tools.curve.tools")


def _arc_mask(shift_u: int = 0) -> np.ndarray:
    """A quarter-circle arc of radius 120 px about the image corner, 6 px wide."""
    v, u = np.mgrid[0:_H, 0:_W]
    r = np.hypot(u - shift_u, v)
    return (np.abs(r - _RADIUS_PX) <= _HALF_WIDTH_PX).astype(np.uint8) * 255


def _flat_depth() -> np.ndarray:
    return np.full((_H, _W), _DEPTH_M, dtype=np.float64)


def _polyline_length(points) -> float:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _arc_centre(points, shift_u: int = 0) -> tuple[float, float]:
    """The arc's centre pixel back-projected at the points' own depth."""
    z = float(np.median(np.asarray(points, dtype=np.float64)[:, 2]))
    return (shift_u - _K[0, 2]) * z / _FX, (0.0 - _K[1, 2]) * z / _FX


def _arc_angles(points) -> np.ndarray:
    """Angle of each point about the arc's centre (pixel (0, 0) back-projected)."""
    pts = np.asarray(points, dtype=np.float64)
    cx, cy = _arc_centre(pts)
    return np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)


def _arc_residuals(points, radius_m: float, shift_u: int = 0) -> np.ndarray:
    """Per-point distance from the arc of ``radius_m`` about the (shifted) centre."""
    pts = np.asarray(points, dtype=np.float64)
    cx, cy = _arc_centre(pts, shift_u)
    return np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - radius_m)


@pytest.fixture(scope="module")
def arc_fit(impl):
    return impl.fit_centerline(_arc_mask(), _flat_depth(), _K, np.eye(4))


def test_fit_orders_the_arc(arc_fit):
    assert arc_fit["ordered"] is True
    assert arc_fit["nodes"] == 40
    assert len(arc_fit["points"]) == 40
    assert all(len(p) == 3 for p in arc_fit["points"])


def test_fit_arclength_matches_the_arc(arc_fit):
    expected = (math.pi / 2.0) * _RADIUS_PX * _PIXEL_M
    assert arc_fit["arclength_m"] == pytest.approx(expected, rel=0.05)


def test_fit_points_are_monotone_along_the_arc(arc_fit):
    steps = np.diff(_arc_angles(arc_fit["points"]))
    assert np.all(steps > 0) or np.all(steps < 0)


def test_fit_radius_is_half_the_width(arc_fit):
    assert arc_fit["radius_m"] == pytest.approx(_HALF_WIDTH_PX * _PIXEL_M, rel=0.30)


def test_fit_pushes_the_curve_onto_the_axis(arc_fit):
    # Identity camera at the origin looks down +z: the surface is at 0.8 m and
    # the axis one radius further along every view ray.
    z = np.asarray(arc_fit["points"])[:, 2]
    assert np.all(z > _DEPTH_M)
    assert np.all(z < _DEPTH_M + 2.0 * arc_fit["radius_m"])


def test_track_preserves_the_prior_length(impl, arc_fit):
    prior = arc_fit["points"]
    out = impl.track_centerline(prior, _arc_mask(shift_u=10), _flat_depth(), _K, np.eye(4))
    assert out["tracked"] is True
    assert out["ordered"] is True
    assert out["nodes"] == 40
    assert len(out["visibility"]) == 40
    assert all(0.0 <= v <= 1.0 for v in out["visibility"])
    # Inextensible: the re-spacing restores the prior's length, less only the
    # corner-cutting of resampling a 40-vertex polyline (~6 um on 380 mm).
    assert out["arclength_m"] == pytest.approx(_polyline_length(prior), rel=1e-4)
    assert out["arclength_m"] == pytest.approx(_polyline_length(out["points"]), abs=1e-9)
    # The rod moved 10 px in +u and the nodes landed ON the shifted arc. (Only
    # the normal component of a featureless rod's motion is observable, so the
    # check is the residual from the arc, not the per-node displacement.)
    radius_m = float(np.median(np.hypot(*(np.asarray(prior)[:, :2] - _arc_centre(prior)).T)))
    assert np.median(_arc_residuals(prior, radius_m, shift_u=10)) > 0.010
    assert np.max(_arc_residuals(out["points"], radius_m, shift_u=10)) < 0.003
    assert np.all(np.asarray(out["points"])[:, 0] > np.asarray(prior)[:, 0])


def test_empty_mask_is_unordered(impl):
    out = impl.fit_centerline(np.zeros((_H, _W), dtype=np.uint8), _flat_depth(), _K, np.eye(4))
    assert out["points"] == []
    assert out["ordered"] is False
    assert out["nodes"] == 0
    assert out["arclength_m"] == 0.0


def test_two_point_prior_falls_back_to_a_cold_fit(impl, arc_fit):
    prior = arc_fit["points"][:2]
    out = impl.track_centerline(prior, _arc_mask(), _flat_depth(), _K, np.eye(4))
    assert out["ordered"] is True
    assert out["nodes"] == 40
    assert "tracked" not in out and "visibility" not in out
    assert out["arclength_m"] == pytest.approx(arc_fit["arclength_m"], abs=1e-9)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

_STRIP_U0, _STRIP_U1 = 60, 260  # inclusive pixel columns of a free-ended strip


def _strip_mask(half_width: int = 3) -> np.ndarray:
    """A straight horizontal strip with both ends free, (2 * half_width + 1) px wide."""
    m = np.zeros((_H, _W), dtype=np.uint8)
    m[120 - half_width : 120 + half_width + 1, _STRIP_U0 : _STRIP_U1 + 1] = 255
    return m


@pytest.fixture(scope="module")
def strip_fit(impl):
    return impl.fit_centerline(_strip_mask(), _flat_depth(), _K, np.eye(4))


def _u(position) -> float:
    x, _, z = position
    return x / z * _FX + _K[0, 2]


def test_endpoints_reach_the_strip_ends_once_extended(impl, strip_fit):
    out = impl.endpoints(strip_fit, K=_K, T=np.eye(4))
    assert out["found"] is True and out["tip"] == -1
    first, last = out["ends"]
    assert first["node"] == 0 and last["node"] == strip_fit["nodes"] - 1
    us = sorted([_u(first["position"]), _u(last["position"])])
    # The silhouette spans u in [60, 261) (pixel edges); extended ends land
    # within a pixel of it.
    assert us[0] == pytest.approx(_STRIP_U0, abs=1.5)
    assert us[1] == pytest.approx(_STRIP_U1 + 1, abs=1.5)
    for end in (first, last):
        assert end["pixel"][0] == pytest.approx(_u(end["position"]), abs=1e-6)
        assert end["pixel"][1] == pytest.approx(120.0, abs=1.5)
        assert end["visibility"] is None


def test_endpoints_unextended_sit_about_a_radius_inside(impl, strip_fit):
    short = impl.endpoints(strip_fit, extend=False)
    long_ = impl.endpoints(strip_fit, extend=True)
    for a, b in zip(short["ends"], long_["ends"], strict=True):
        gap = np.linalg.norm(np.subtract(b["position"], a["position"]))
        assert gap == pytest.approx(strip_fit["radius_m"], rel=1e-9)
        assert a["pixel"] is None


def test_endpoint_directions_point_out_of_the_rod(impl, strip_fit):
    first, last = impl.endpoints(strip_fit)["ends"]
    left, right = (first, last) if _u(first["position"]) < _u(last["position"]) else (last, first)
    np.testing.assert_allclose(left["direction"], [-1.0, 0.0, 0.0], atol=0.02)
    np.testing.assert_allclose(right["direction"], [1.0, 0.0, 0.0], atol=0.02)


def test_a_hint_picks_the_nearest_end(impl, strip_fit):
    ends = impl.endpoints(strip_fit, K=_K, T=np.eye(4))["ends"]
    right = int(np.argmax([_u(e["position"]) for e in ends]))
    by_pixel = impl.endpoints(strip_fit, near_pixel=[255.0, 118.0], K=_K, T=np.eye(4))
    assert by_pixel["tip"] == right
    assert by_pixel["hint_distance"] < 8.0
    by_point = impl.endpoints(strip_fit, near_point=ends[1 - right]["position"])
    assert by_point["tip"] == 1 - right
    assert by_point["hint_distance"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="near_pixel needs"):
        impl.endpoints(strip_fit, near_pixel=[255.0, 118.0])


def test_endpoints_carry_tracker_visibility_and_refuse_nothing(impl, arc_fit):
    tracked = impl.track_centerline(arc_fit["points"], _arc_mask(), _flat_depth(), _K, np.eye(4))
    first, last = impl.endpoints(tracked)["ends"]
    assert first["visibility"] == tracked["visibility"][0]
    assert last["visibility"] == tracked["visibility"][-1]
    empty = {"found": False, "ends": [], "tip": -1, "hint_distance": 0.0}
    assert impl.endpoints({"points": []}) == empty
    assert impl.endpoints([[0.0, 0.0, 1.0]]) == empty


# ---------------------------------------------------------------------------
# The @tool boundary
# ---------------------------------------------------------------------------


def test_tool_defaults_mirror_impl(impl, tools):
    assert tools._DEFAULT_NODES == impl.DEFAULT_NODES
    assert tools._SMOOTH == impl.SMOOTH
    assert tools._TANGENT_M == impl.TANGENT_M


def test_endpoints_tool_takes_a_pose_dict_and_a_pixel_hint(tools, strip_fit):
    from gap_core.types import identity_pose

    out = tools.endpoints(
        strip_fit, near_pixel=[62, 121], intrinsics=_K, camera_pose=identity_pose()
    )
    assert out["found"] is True
    assert _u(out["ends"][out["tip"]]["position"]) == pytest.approx(_STRIP_U0, abs=1.5)
    with pytest.raises(ValueError, match="curve.endpoints needs"):
        tools.endpoints(None)
    schema = _schema(tools, tools.endpoints)
    assert schema.inputs["centerline"].required
    assert not schema.inputs["near_pixel"].required
    assert {"found", "ends", "tip", "hint_distance"} <= set(schema.outputs)


def test_tools_accept_a_pose_dict_or_a_matrix(tools, arc_fit):
    from gap_core.types import identity_pose

    via_dict = tools.fit_centerline(_arc_mask(), _flat_depth(), _K, identity_pose())
    via_matrix = tools.fit_centerline(_arc_mask(), _flat_depth(), _K, np.eye(4))
    np.testing.assert_allclose(via_dict["points"], via_matrix["points"])
    np.testing.assert_allclose(via_dict["points"], arc_fit["points"])

    tracked = tools.track_centerline(
        arc_fit, _arc_mask(shift_u=10), _flat_depth(), _K, identity_pose()
    )
    assert tracked["tracked"] is True
    assert tracked["arclength_m"] == pytest.approx(_polyline_length(arc_fit["points"]), rel=1e-4)


def test_tools_refuse_missing_inputs(tools):
    with pytest.raises(ValueError, match="curve.fit_centerline needs"):
        tools.fit_centerline(None, _flat_depth(), _K, np.eye(4))
    with pytest.raises(ValueError, match="curve.track_centerline needs"):
        tools.track_centerline(None, _arc_mask(), _flat_depth(), _K, np.eye(4))


def _schema(tools, fn):
    """Extract a @tool's schema the way ``ToolRegistry.discover_pending`` does:
    a fake module whose globals are the tools module's, so the deferred
    annotations (``Mask``, ``Se3Pose``) resolve."""
    from types import SimpleNamespace

    from gap_core.tools.schema import extract_schema

    wrapper = SimpleNamespace(run=fn, _meta=None, __doc__=fn.__doc__, __name__=fn.__name__)
    wrapper.__dict__.update(vars(tools))
    return extract_schema(wrapper)


def test_tool_schemas_extract(tools):
    for fn in (tools.fit_centerline, tools.track_centerline):
        schema = _schema(tools, fn)
        assert {"mask", "depth", "intrinsics", "camera_pose"} <= set(schema.inputs)
        for field in schema.inputs.values():
            assert field.type_str, f"{fn.__name__}: untyped input {field.name}"
        assert {"points", "arclength_m", "radius_m", "nodes", "ordered"} <= set(schema.outputs)
    fit = _schema(tools, tools.fit_centerline)
    assert not fit.inputs["nodes"].required and fit.inputs["nodes"].default == 40
    assert _schema(tools, tools.track_centerline).inputs["prior"].required
