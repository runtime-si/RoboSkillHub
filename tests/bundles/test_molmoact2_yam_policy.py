"""policies/molmoact2-yam with the model faked: the profile, station <-> model mapping, checkpoint pinning,
the vla-yam/1 service over gr00t-wam/1's HTTP / ZMQ transports, and the gr00t client against it."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from .conftest import SKILLS_ROOT

BUNDLE = SKILLS_ROOT / "policies" / "molmoact2-yam"
PROFILE = BUNDLE / "profiles" / "yam-molmoact2-native.yaml"
IMG = np.zeros((48, 64, 3), np.uint8)


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("molmoact2_yam_server", BUNDLE / "server.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["molmoact2_yam_server"] = mod
    spec.loader.exec_module(mod)
    return mod


def _service(srv, station=None):
    prof = srv.load_profile(PROFILE, station)
    backend = srv.MolmoAct2Backend(prof, model=srv.FakeMolmoAct2Model(prof["action"]["horizon"]))
    return srv.VlaService(backend), backend


def _req(**kw):
    return {
        "instruction": "pick up the cup",
        "images": dict.fromkeys(("top", "left_wrist", "right_wrist"), IMG),
        "joints": [0.1] * 6 + [1.0] + [0.2] * 6 + [0.0],
        **kw,
    }


def test_profile_matches_spec(srv):
    p = srv.load_profile(PROFILE)
    assert p["checkpoint"]["repo"] == "allenai/MolmoAct2-BimanualYAM"
    assert len(p["checkpoint"]["revision"]) == 40 and len(p["checkpoint"]["sha256"]) == 5
    assert p["cameras"] == ["top", "left_wrist", "right_wrist"]
    assert p["action"] == {
        "type": "abs_joint",
        "dim": 14,
        "horizon": 30,
        "execute_steps": 25,
        "hz": 30,
    }
    assert p["extra"]["norm_tag"] == "yam_dual_molmoact2" and p["extra"]["dtype"] == "bfloat16"
    assert p["station"] == "twin" and p["joint_offset_rad"]["left"] == [0.0] * 6
    real = srv.load_profile(PROFILE, "real")
    assert real["joint_offset_rad"]["left"][0] == pytest.approx(np.pi / 2)
    with pytest.raises(ValueError, match="no station"):
        srv.load_profile(PROFILE, "mars")


def test_bad_profiles_refused(srv, tmp_path):
    import yaml

    base = yaml.safe_load(PROFILE.read_text())
    for patch, msg in (
        ({"cameras": ["left_wrist", "top", "right_wrist"]}, "cameras"),
        (
            {
                "gripper": {
                    "station_open": 1,
                    "station_closed": 1,
                    "model_open": 1,
                    "model_closed": 0,
                }
            },
            "differ",
        ),
        ({"action": {**base["action"], "execute_steps": 31}}, "execute_steps"),
        ({"joint_offset_rad": {"left": [0] * 5, "right": [0] * 6}}, "6 values"),
    ):
        f = tmp_path / "p.yaml"
        f.write_text(yaml.safe_dump({**base, **patch}))
        with pytest.raises(ValueError, match=msg):
            srv.load_profile(f)


def test_station_model_roundtrip(srv):
    p = srv.load_profile(PROFILE, "real")
    p["gripper"] = {
        "station_open": 0.0,
        "station_closed": 1.0,
        "model_open": 0.9,
        "model_closed": 0.1,
    }
    q = np.array([2.0, 0.5, 0.4, -0.3, 0.2, 0.1, 0.0, 1.0, 0.5, 0.4, -0.3, 0.2, 0.1, 1.0])
    m = srv.station_to_model(p, q)
    assert m[0] == pytest.approx(2.0 - np.pi / 2) and m[7] == pytest.approx(1.0 - np.pi / 2)
    assert m[6] == pytest.approx(0.9) and m[13] == pytest.approx(
        0.1
    )  # station open 0 -> model open 0.9
    assert np.allclose(srv.model_to_station(p, np.stack([m, m])), q)


def test_act_maps_frames_and_reports(srv):
    service, backend = _service(srv, "real")
    out = service.handle("act", _req())
    a = np.asarray(out["actions"])
    assert a.shape == (30, 14) and out["horizon"] == 30 and out["execute_steps"] == 25
    sent = backend.model.calls[-1]
    assert sent["state"][0] == pytest.approx(0.1 - np.pi / 2)  # model frame on the way in
    assert [i.shape for i in sent["images"]] == [IMG.shape] * 3
    assert (
        sent["norm_tag"] == "yam_dual_molmoact2" and sent["inference_action_mode"] == "continuous"
    )
    assert sent["num_steps"] == 10 and sent["enable_depth_reasoning"] is False
    assert a[0, 0] == pytest.approx(0.1 + 0.001, abs=1e-5)  # station frame on the way out
    assert a[0, 6] == pytest.approx(1.0) and a[0, 13] == pytest.approx(0.0)
    assert (
        out["frame"] == "station"
        and out["profile"] == "yam-molmoact2-native"
        and out["station"] == "real"
    )
    assert out["dt_ms"] >= 0 and out["dt"] == pytest.approx(1 / 30) and out["steering"] == "none"
    raw = service.handle("act", _req(frame="model", num_steps=4))
    assert backend.model.calls[-1]["state"][0] == pytest.approx(0.1) and raw["num_steps"] == 4


def test_camera_order_is_top_left_right(srv):
    service, backend = _service(srv)
    imgs = {
        r: np.full((8, 8, 3), i, np.uint8)
        for i, r in enumerate(("top", "left_wrist", "right_wrist"))
    }
    service.handle("act", _req(images=imgs))
    assert [int(i[0, 0, 0]) for i in backend.model.calls[-1]["images"]] == [0, 1, 2]


@pytest.mark.parametrize(
    "patch,msg",
    [
        ({"images": {"top": IMG, "left_wrist": IMG}}, "missing"),
        ({"frame": "world"}, "frame"),
        ({"num_steps": 0}, "num_steps"),
        ({"joints": [0.0] * 13}, "14"),
        ({"instruction": ""}, "instruction"),
    ],
)
def test_bad_requests_refused(srv, patch, msg):
    service, _ = _service(srv)
    with pytest.raises(ValueError, match=msg):
        service.handle("act", _req(**patch))


def test_marks_accepted_and_ignored(srv):
    service, backend = _service(srv)
    out = service.handle("act", _req(marks=[{"kind": "grasp", "camera": "top", "uv": [3, 4]}]))
    assert out["marks_ignored"] == 1 and "marks" not in backend.model.calls[-1]


def test_info_announces_vla_yam_and_gr00t_compat(srv):
    service, _ = _service(srv)
    info = service.info()
    assert info["protocol"] == "vla-yam/1" and info["compatible"] == ["gr00t-wam/1"]
    assert info["roles"] == ["top", "left_wrist", "right_wrist"] and info["horizon"] == 30


def test_verify_checkpoint(srv, tmp_path):
    blob = tmp_path / "blobs"
    blob.mkdir()
    data = b"weights"
    sha = hashlib.sha256(data).hexdigest()
    (blob / sha).write_bytes(data)
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").symlink_to(blob / sha)
    assert srv.verify_checkpoint(snap, {"model.safetensors": sha}, "cache") == {
        "model.safetensors": "ok"
    }
    assert srv.verify_checkpoint(snap, {"model.safetensors": sha}, "full") == {
        "model.safetensors": "ok"
    }
    assert srv.verify_checkpoint(snap, {}, "off") == {}
    with pytest.raises(ValueError, match="pinned"):
        srv.verify_checkpoint(snap, {"model.safetensors": "0" * 64}, "full")
    with pytest.raises(FileNotFoundError):
        srv.verify_checkpoint(snap, {"other.safetensors": sha}, "cache")
    with pytest.raises(ValueError, match="empty"):
        srv.verify_checkpoint(snap, {}, "cache")


def _http(service, srv):
    httpd = srv.g.make_http_server(service, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_http_roundtrip(srv):
    service, _ = _service(srv)
    httpd, url = _http(service, srv)
    try:
        body = _req(
            images={
                r: {
                    "npy": __import__("base64").b64encode(IMG.tobytes()).decode(),
                    "shape": list(IMG.shape),
                    "dtype": "uint8",
                }
                for r in ("top", "left_wrist", "right_wrist")
            }
        )
        r = urllib.request.urlopen(urllib.request.Request(url + "/act", json.dumps(body).encode()))
        out = json.loads(r.read())
        assert (
            np.asarray(out["actions"]).shape == (30, 14)
            and out["profile"] == "yam-molmoact2-native"
        )
        bad = urllib.request.Request(url + "/act", json.dumps({**body, "frame": "x"}).encode())
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 400
        assert json.loads(urllib.request.urlopen(url + "/info").read())["protocol"] == "vla-yam/1"
    finally:
        httpd.shutdown()


def test_zmq_roundtrip(srv):
    zmq = pytest.importorskip("zmq")
    msgpack = pytest.importorskip("msgpack")
    mnp = pytest.importorskip("msgpack_numpy")
    service, _ = _service(srv)
    ctx = zmq.Context()
    probe = ctx.socket(zmq.REP)
    port = probe.bind_to_random_port("tcp://127.0.0.1")
    probe.close()
    threading.Thread(target=srv.g.serve_zmq, args=(service, "127.0.0.1", port), daemon=True).start()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.connect(f"tcp://127.0.0.1:{port}")
    sock.send(msgpack.packb({"endpoint": "act", "data": _req()}, default=mnp.encode))
    out = msgpack.unpackb(sock.recv(), object_hook=mnp.decode, raw=False)
    assert np.asarray(out["actions"]).shape == (30, 14) and out["frame"] == "station"
    sock.close()


def test_gr00t_client_talks_to_it(srv, skills_registry, monkeypatch):
    """gr00t.act (gr00t-wam/1 client) against this server: same request, 30-row chunk back."""
    g = skills_registry.get("gr00t").tools_module
    service, backend = _service(srv)
    httpd, url = _http(service, srv)
    monkeypatch.setenv("GAP_GR00T_URL", url)
    monkeypatch.setenv("GAP_GR00T_IMAGE_CODEC", "npy")
    monkeypatch.delenv("GAP_GR00T_JOINT_OFFSET", raising=False)
    g._clients.clear()
    cam = {
        "rgb": np.full((96, 128, 3), 90, np.uint8),
        "intrinsics": np.eye(3),
        "pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 1.0},
            "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0},
        },
    }
    arm = {
        "joint_state": {"positions": np.full(6, 0.3)},
        "gripper_fraction": 1.0,
        "ee_pose": {
            "position": {"x": 0.3, "y": 0.1, "z": 0.2},
            "rotation": {"w": 1.0, "x": 0, "y": 0, "z": 0},
        },
    }
    obs = {
        "cameras": [{**cam, "name": n} for n in ("top", "left_wrist", "right_wrist")],
        "arms": [arm, arm],
    }
    try:
        out = g.act(obs, "Pick up the cup.")
        assert np.asarray(out["actions"]).shape == (30, 14)
        assert backend.model.calls[-1]["state"][0] == pytest.approx(0.3)
    finally:
        httpd.shutdown()
        g._clients.clear()
