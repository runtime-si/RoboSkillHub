"""tools/dam4sam on CPU: the numpy stand-ins for the VOT toolkit, DAM4SAM's distractor-memory rule,
and the tools' argument checks (the model is never loaded)."""

from __future__ import annotations

import numpy as np
import pytest
from gap_core.errors import ToolError


@pytest.fixture(scope="module")
def dam(skills_registry):
    return skills_registry.get("dam4sam").tools_module


def _rect(x1, y1, x2, y2, shape=(60, 80)):
    m = np.zeros(shape, np.uint8)
    m[y1:y2, x1:x2] = 1
    return m


def test_mask_box_and_iou(dam):
    assert dam.mask_box(np.zeros((5, 5))) is None
    b = dam.mask_box(_rect(10, 5, 20, 15))
    assert b.tolist() == [10, 5, 20, 15]
    assert dam.box_iou(b, b) == 1.0 and dam.box_iou(b, None) == 0.0
    assert abs(dam.box_iou(b, np.array([15, 5, 25, 15])) - 1 / 3) < 1e-9


def test_keep_largest_component(dam):
    m = _rect(0, 0, 3, 3) | _rect(10, 10, 20, 20)
    k = dam.keep_largest_component(m)
    assert k.sum() == 100 and k[0, 0] == 0


def test_drm_adds_an_anchor_only_when_an_alternative_points_at_a_twin(dam):
    target = _rect(10, 10, 20, 20)
    twin = _rect(40, 10, 50, 20)
    grown = _rect(10, 10, 22, 20)  # the same part, a little larger: no distractor
    assert dam.drm_should_add(target, [twin], 0.9, 1.0, True)
    assert not dam.drm_should_add(target, [grown], 0.9, 1.0, True)
    assert not dam.drm_should_add(target, [twin], 0.7, 1.0, True)  # not confident
    assert not dam.drm_should_add(target, [twin], 0.9, 1.5, True)  # size jumped
    assert not dam.drm_should_add(target, [twin], 0.9, 1.0, False)  # anchored < 5 frames ago
    assert not dam.drm_should_add(target, [target], 0.9, 1.0, True)  # alternative = the target


def test_tool_argument_checks(dam):
    with pytest.raises(ToolError):
        dam.tracker_init(np.zeros((10, 10), np.uint8), box=[0, 0, 5, 5])
    with pytest.raises(ToolError):
        dam.tracker_init(np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ToolError):
        dam.tracker_init(np.zeros((10, 10, 3), np.uint8), box=[5, 5, 2, 8])
    with pytest.raises(ToolError):
        dam.tracker_update("nope", np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ToolError):
        dam.track([])
    assert dam.tracker_close("nope") == {"closed": False}
