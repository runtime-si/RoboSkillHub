"""roboskillhub.vla.VlaYamClient against the hub's own servers with the models faked: policies/molmoact2-yam
(vla-yam/1) and policies/gr00t (gr00t-wam/1, incl. a pi05-style station wire) over HTTP and ZMQ; the
profile's ``stations`` override and ``backend``; one offset sign (station = model + offset) on every side."""

from __future__ import annotations

import importlib.util
import math
import sys
import threading

import numpy as np
import pytest

from roboskillhub import vla

from .conftest import SKILLS_ROOT

MOLMO = SKILLS_ROOT / "policies" / "molmoact2-yam"
GR00T = SKILLS_ROOT / "policies" / "gr00t"
MOLMO_PROFILE = MOLMO / "profiles" / "yam-molmoact2-native.yaml"
PI05_PROFILE = GR00T / "profiles" / "yam-pi05-molmoact2.yaml"
ROLES = ("top", "left_wrist", "right_wrist")


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def molmo():
    return _load("molmoact2_yam_server", MOLMO / "server.py")


@pytest.fixture(scope="module")
def gsrv(molmo):
    return sys.modules["gr00t_wam_server"]


@pytest.fixture(scope="module")
def pi05():
    sys.path.insert(0, str(GR00T))
    try:
        return _load("pi05_backend", GR00T / "pi05_backend.py")
    finally:
        sys.path.remove(str(GR00T))


def _http(gsrv, service):
    server = gsrv.make_http_server(service, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _molmo_service(molmo, station=None):
    prof = molmo.load_profile(MOLMO_PROFILE, station)
    backend = molmo.MolmoAct2Backend(
        prof, model=molmo.FakeMolmoAct2Model(prof["action"]["horizon"])
    )
    return molmo.VlaService(backend)


def _input(profile, q_station):
    obs_state = np.asarray(q_station, dtype=np.float64)
    return {
        "images": {r: np.full((24, 32, 3), 9 * i, np.uint8) for i, r in enumerate(ROLES)},
        "image_order": list(ROLES),
        "state": profile.to_model(obs_state),
        "state_names": [s.name for s in profile.state],
        "t": 1.0,
        "profile": profile.name,
    }


Q = np.array([2.0, 0.3, 0.4, -0.2, 0.1, 0.05, 0.9, 1.0, -0.3, 0.2, 0.1, 0.0, -0.1, 0.8])


# -- profiles: backend, stations, one sign ------------------------------------------------------


def test_profile_backend_and_station_override():
    twin = vla.load_profile(MOLMO_PROFILE)
    real = vla.load_profile(MOLMO_PROFILE, station="real")
    assert twin.backend == real.backend == "molmoact2"
    assert twin.station == "twin" and real.station == "real" and twin.stations == ("real",)
    assert np.allclose(twin.joint_offset_rad["left"], 0)
    assert real.joint_offset_rad["left"][0] == pytest.approx(math.pi / 2)
    assert real.joint_offset_rad["right"][0] == pytest.approx(math.pi / 2)
    with pytest.raises(vla.ProfileError, match="no station 'mars'"):
        vla.load_profile(MOLMO_PROFILE, station="mars")
    with pytest.raises(vla.ProfileError, match="may not override"):
        vla.load_profile({**_raw(), "stations": {"x": {"profile": "other"}}}, station="x")
    pi = vla.load_profile(PI05_PROFILE)
    assert pi.backend == "pi05" and np.allclose(pi.joint_offset_rad["left"], 0)
    assert vla.load_profile(PI05_PROFILE, "real").joint_offset_rad["left"][0] == pytest.approx(
        math.pi / 2
    )


def _raw():
    import yaml

    return yaml.safe_load(MOLMO_PROFILE.read_text())


def test_offset_sign_agrees_everywhere(molmo, pi05):
    """station = model + offset: roboskillhub.vla, molmoact2-yam's server, pi05's FrameMap, gr00t.tools."""
    p = vla.load_profile(MOLMO_PROFILE, station="real")
    m = p.to_model(Q)
    assert m[0] == pytest.approx(Q[0] - math.pi / 2) and m[7] == pytest.approx(Q[7] - math.pi / 2)
    assert np.allclose(molmo.station_to_model(molmo.load_profile(MOLMO_PROFILE, "real"), Q), m)
    fm = pi05.FrameMap(pi05.load_profile(pi05.DEFAULT_PROFILE, "real"), "station")
    assert np.allclose(fm.to_model(Q), m, atol=1e-6)
    assert np.allclose(p.to_station(m), Q)
    src = (GR00T / "tools.py").read_text()
    assert 'out[0:6] -= offset["left"]' in src and 'out[..., 0:6] += offset["left"]' in src


# -- the client against real servers ------------------------------------------------------------


def test_http_vla_yam_station_real_matches_server_side_mapping(molmo, gsrv):
    """Client-side mapping (frame model) == the server's own --station real mapping (frame station)."""
    server, url = _http(gsrv, _molmo_service(molmo, "real"))
    try:
        p = vla.load_profile(MOLMO_PROFILE, station="real")
        client = vla.VlaYamClient(url, p, codec="npy")
        assert client.info()["protocol"] == "vla-yam/1" and client.wire_frame == "model"
        reply = client.act(_input(p, Q), "pick up the cup")
        assert reply["frame"] == "model" and reply["actions"].shape == (30, 14)
        ours = p.to_station(reply["actions"])
        theirs = client.call(
            "act",
            {
                "instruction": "pick up the cup",
                "images": client.request(_input(p, Q), "x")["images"],
                "joints": Q,
                "frame": "station",
            },
        )
        assert np.allclose(ours, np.asarray(theirs["actions"]), atol=1e-5)
        assert ours[0, 0] == pytest.approx(
            Q[0] + 0.001, abs=1e-5
        )  # the fake nudges +1 mrad per row
    finally:
        server.shutdown()


def test_profile_mismatch_and_unknown_protocol_are_refused(molmo, gsrv):
    server, url = _http(gsrv, _molmo_service(molmo))
    try:
        other = vla.load_profile({**_raw(), "profile": "someone-else"})
        with pytest.raises(vla.PolicyServerError, match="runs profile"):
            vla.VlaYamClient(url, other, codec="npy").info()
        assert vla.VlaYamClient(url, other, codec="npy", check_profile=False).info()["profile"]
    finally:
        server.shutdown()
    with pytest.raises(vla.PolicyServerError, match="not one of"):
        vla.VlaYamClient("http://x", transport=lambda e, d: {"protocol": "other/9"}).info()


def test_gr00t_wam_station_wire_sends_station_joints(pi05):
    """A pi05 server started with --wire-frame station maps itself: the client sends station joints."""
    p = vla.load_profile(PI05_PROFILE, "real")
    sent = {}

    def transport(endpoint, data):
        if endpoint == "info":
            return {"protocol": "gr00t-wam/1", "wire_frame": "station", "profile": p.name}
        sent.update(data)
        return {"actions": np.tile(np.asarray(data["joints"]), (30, 1))}

    client = vla.VlaYamClient("http://x", p, transport=transport)
    reply = client.act(_input(p, Q), "pick")
    assert np.allclose(sent["joints"], Q) and sent["frame"] == "station"
    assert reply["frame"] == "station" and np.allclose(reply["actions"][0], Q)


def test_http_errors_become_policy_server_errors(molmo, gsrv):
    server, url = _http(gsrv, _molmo_service(molmo))
    try:
        p = vla.load_profile(MOLMO_PROFILE)
        client = vla.VlaYamClient(url, p, codec="npy")
        inp = _input(p, Q)
        del inp["images"]["top"]
        with pytest.raises(vla.PolicyServerError, match="500|400"):
            client.act(inp, "pick")
    finally:
        server.shutdown()
    with pytest.raises(vla.PolicyServerError):
        vla.VlaYamClient("http://127.0.0.1:9", p, timeout_s=1).info()


def test_zmq_vla_yam(molmo, gsrv):
    pytest.importorskip("zmq")
    pytest.importorskip("msgpack_numpy")
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    threading.Thread(
        target=gsrv.serve_zmq, args=(_molmo_service(molmo), "127.0.0.1", port), daemon=True
    ).start()
    p = vla.load_profile(MOLMO_PROFILE)
    client = vla.VlaYamClient(f"zmq://127.0.0.1:{port}", p, timeout_s=5)
    reply = client.act(_input(p, Q), "pick up the cup")
    assert reply["actions"].shape == (30, 14) and reply["profile"] == p.name


# -- run_policy end to end through HTTP --------------------------------------------------------


class Station:
    """Two arms that land where commanded; jaws follow open / close."""

    def __init__(self, q):
        self.q = {"left": np.asarray(q[0:6], float), "right": np.asarray(q[7:13], float)}
        self.grip = {"left": 1.0, "right": 1.0}
        self.t = 50.0
        self.streamed = 0

    def tool(self, name, **kw):
        if name == "robot.get_observation":
            return {
                "timestamp": self.t,
                "cameras": [
                    {"name": n, "rgb": np.zeros((24, 32, 3), np.uint8), "timestamp": self.t}
                    for n in ("top_cam", "wrist_left", "wrist_right")
                ],
                "arms": [
                    {
                        "joint_state": {"positions": self.q[a].copy()},
                        "gripper_fraction": self.grip[a],
                        "timestamp": self.t,
                    }
                    for a in ("left", "right")
                ],
            }
        if name == "robot.stream_dual":
            self.streamed += 1
            for arm_id, wps in kw["tracks"].items():
                self.q[("left", "right")[int(arm_id)]] = np.asarray(wps[-1])
            return {}
        if name in ("robot.open_gripper", "robot.close_gripper"):
            self.grip[("left", "right")[kw.get("arm_id", 0)]] = 1.0 if "open" in name else 0.0
            return {}
        raise RuntimeError(f"unknown tool {name}")


def test_run_policy_through_the_http_client(molmo, gsrv):
    server, url = _http(gsrv, _molmo_service(molmo, "real"))
    try:
        p = vla.load_profile(MOLMO_PROFILE, station="real")
        q = Q.copy()
        q[6] = q[13] = 1.0
        robot = Station(q)
        res = vla.run_policy(
            vla.VlaYamClient(url, p, codec="npy"),
            p,
            "pick up the cup",
            max_steps=50,
            robot=robot,
            clock=lambda: robot.t,
        )
        assert res["route"] == "max_steps" and res["chunks"] == 2 and robot.streamed >= 2
        assert res["chunk_log"][0]["frame"] == "model" and res["chunk_log"][0]["profile"] == p.name
        # The fake model nudges +1 mrad per row in the model frame; 50 rows later the base joint sits
        # 50 mrad above where it started, in the station frame (pi/2 offset not applied twice).
        assert robot.q["left"][0] == pytest.approx(Q[0] + 0.025 + 0.025, abs=1e-4)
    finally:
        server.shutdown()
