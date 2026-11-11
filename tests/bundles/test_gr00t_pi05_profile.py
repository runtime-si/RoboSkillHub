"""policies/gr00t pi05 backend without a model: profile loading and the station <-> model frame map."""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from .conftest import SKILLS_ROOT

BUNDLE = SKILLS_ROOT / "policies" / "gr00t"


@pytest.fixture(scope="module")
def pi05():
    pytest.importorskip("yaml")
    spec = importlib.util.spec_from_file_location("gr00t_pi05_backend", BUNDLE / "pi05_backend.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_profile_loads_and_is_pinned(pi05):
    prof = pi05.load_profile(pi05.DEFAULT_PROFILE)
    assert prof["checkpoint"]["revision"] and prof["checkpoint"]["sha256"]
    assert prof["tokenizer"]["revision"] and prof["tokenizer"]["sha256"]
    assert set(prof["image_keys"]) == set(pi05.ROLES)
    assert prof["action"] == {
        "type": "abs_joint",
        "dim": 14,
        "horizon": 30,
        "execute_steps": 25,
        "hz": 30,
    }


def test_policy_wire_is_identity_and_station_wire_round_trips(pi05):
    twin = pi05.load_profile(pi05.DEFAULT_PROFILE)
    q = np.linspace(-0.5, 0.5, 14)
    assert np.allclose(pi05.FrameMap(twin, "station").to_model(q)[[0, 7]], q[[0, 7]])
    prof = pi05.load_profile(pi05.DEFAULT_PROFILE, "real")
    assert prof["station"] == "real"
    assert np.allclose(pi05.FrameMap(prof, "policy").to_model(q), q)
    fm = pi05.FrameMap(prof, "station")
    m = fm.to_model(q)
    assert math.isclose(m[0], q[0] - math.pi / 2, abs_tol=1e-6) and math.isclose(
        m[7], q[7] - math.pi / 2, abs_tol=1e-6
    )
    assert np.allclose(fm.to_station(m[None].astype(np.float32))[0], q, atol=1e-6)


def test_unknown_profile_is_refused(pi05):
    with pytest.raises(ValueError, match="no pi05 profile"):
        pi05.load_profile("nope")
    with pytest.raises(ValueError, match="no station"):
        pi05.load_profile(pi05.DEFAULT_PROFILE, "mars")
