"""A :class:`~roboskillhub.vla.loop.PolicyClient` for vla-yam/1 and gr00t-wam/1 servers over HTTP or ZMQ.

    client = VlaYamClient("http://gpu-host:5560", profile)          # or "zmq://gpu-host:5560"
    run_policy(client, profile, "pick up the red cup", robot=ctx, ...)

One client for policies/molmoact2-yam (vla-yam/1), policies/gr00t's ``n1.7`` / ``pi05`` / ``fake``
backends (gr00t-wam/1) and anything else that speaks the request of VLA_SKILL_SPEC section 3: images by
role, ``joints[14]`` = ``[left j1..j6, left opening, right j1..j6, right opening]``, an instruction,
optional marks; the reply's ``actions`` are ``[T, 14]`` in the same layout.

Frames. The client's profile does the station <-> model mapping (``station = model + offset``, the
gripper map), so the wire carries the **model** frame and the server maps nothing:

- vla-yam/1 (``/info`` ``protocol``): ``frame: "model"`` per request; the server's own profile and
  ``--station`` are not used for this request.
- gr00t-wam/1: the wire is the policy (= model) frame already, unless ``/info`` reports
  ``wire_frame: "station"`` (pi05 ``--wire-frame station``), which ignores a per-request frame and
  maps with *its* profile; then the client sends station joints and takes station actions back.

Either way the reply says which frame its actions are in (``frame``), and ``execute_chunk`` maps model
rows with the client's profile. The server's profile name must match the client's (``/info``
``profile``) unless ``check_profile=False``: two profiles disagreeing on the offset or the gripper map
would move the arm somewhere else.

HTTP needs nothing beyond the standard library (and Pillow for jpeg / png images; ``codec="npy"`` sends
raw arrays); ZMQ needs pyzmq + msgpack-numpy (Isaac-GR00T's framing, ``{"endpoint", "data"}``).
"""

from __future__ import annotations

import base64
import io
import json
import os
from collections.abc import Callable
from typing import Any

import numpy as np

from .profile import ARMS, JOINTS, Profile, Slot

PROTOCOLS = ("vla-yam/1", "gr00t-wam/1")
WIRE_LAYOUT = tuple(Slot(arm, j) for arm in ARMS for j in (*range(JOINTS), None))
"""The 14 wire columns: left j1..j6, left gripper, right j1..j6, right gripper."""
DEFAULT_URL = "http://127.0.0.1:5560"
URL_ENV = ("ROBOSKILLHUB_VLA_URL", "GAP_GR00T_URL")


class PolicyServerError(RuntimeError):
    """The server answered with an error (HTTP 4xx / 5xx or ``{"error": ...}``) or not at all."""


def default_url() -> str:
    for k in URL_ENV:
        if os.environ.get(k):
            return str(os.environ[k])
    return DEFAULT_URL


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def encode_image(image: np.ndarray, codec: str = "jpeg") -> dict[str, Any]:
    """An RGB uint8 image as gr00t-wam/1's JSON blob (``{"jpeg"|"png": b64}`` or ``{"npy", shape, dtype}``)."""
    a = np.ascontiguousarray(image)
    if codec == "npy":
        return {
            "npy": base64.b64encode(a.tobytes()).decode(),
            "shape": list(a.shape),
            "dtype": str(a.dtype),
        }
    from PIL import Image

    buf = io.BytesIO()
    if codec == "png":
        Image.fromarray(a).save(buf, "PNG")
        return {"png": base64.b64encode(buf.getvalue()).decode()}
    if codec != "jpeg":
        raise ValueError(f"codec is jpeg | png | npy, not {codec!r}")
    Image.fromarray(a).save(buf, "JPEG", quality=95)
    return {"jpeg": base64.b64encode(buf.getvalue()).decode()}


class HttpTransport:
    """``GET /info``, ``POST /act`` | ``/reset``; JSON bodies, images as blobs."""

    def __init__(self, url: str, timeout_s: float = 30.0, codec: str = "jpeg") -> None:
        self.url, self.timeout_s, self.codec = url.rstrip("/"), float(timeout_s), codec

    def __call__(self, endpoint: str, data: dict[str, Any] | None) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        body = None
        if data is not None:
            wire = dict(data)
            if "images" in wire:
                wire["images"] = {k: encode_image(v, self.codec) for k, v in wire["images"].items()}
            body = json.dumps(_jsonable(wire)).encode()
        req = urllib.request.Request(
            f"{self.url}/{endpoint}",
            data=body,
            method="POST" if body is not None else "GET",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                reply = json.loads(e.read())
            except Exception:
                reply = {}
            raise PolicyServerError(f"HTTP {e.code}: {reply.get('error', e.reason)}") from None
        except (urllib.error.URLError, OSError) as e:
            raise PolicyServerError(f"{self.url}: {e}") from e


class ZmqTransport:
    """A REQ socket with Isaac-GR00T's msgpack framing; a missed reply resets the socket."""

    def __init__(self, url: str, timeout_s: float = 30.0) -> None:
        self.address = "tcp://" + url.split("://", 1)[1]
        self.timeout_s = float(timeout_s)
        self._sock: Any = None

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __call__(self, endpoint: str, data: dict[str, Any] | None) -> dict[str, Any]:
        import msgpack
        import msgpack_numpy as mnp
        import zmq

        if self._sock is None:
            self._sock = zmq.Context.instance().socket(zmq.REQ)
            ms = int(self.timeout_s * 1000)
            for opt, v in ((zmq.RCVTIMEO, ms), (zmq.SNDTIMEO, ms), (zmq.LINGER, 0)):
                self._sock.setsockopt(opt, v)
            self._sock.connect(self.address)
        try:
            self._sock.send(
                msgpack.packb({"endpoint": endpoint, "data": data or {}}, default=mnp.encode)
            )
            return msgpack.unpackb(self._sock.recv(), object_hook=mnp.decode, raw=False)
        except zmq.error.Again as e:
            self.close()  # a REQ socket that missed its reply cannot send again
            raise PolicyServerError(f"{self.address}: no reply in {self.timeout_s} s") from e


def make_transport(
    url: str, timeout_s: float = 30.0, codec: str = "jpeg"
) -> Callable[..., dict[str, Any]]:
    if url.startswith(("zmq://", "tcp://")):
        return ZmqTransport(url, timeout_s)
    if url.startswith(("http://", "https://")):
        return HttpTransport(url, timeout_s, codec)
    raise ValueError(f"policy server url must be http(s):// or zmq://, got {url!r}")


class VlaYamClient:
    """``act(model_input, instruction)`` for :func:`~roboskillhub.vla.run_policy` against a vla-yam/1 or
    gr00t-wam/1 server. ``transport`` (``endpoint, data -> reply``) replaces HTTP / ZMQ in tests."""

    def __init__(
        self,
        url: str | None = None,
        profile: Profile | None = None,
        *,
        timeout_s: float | None = None,
        codec: str | None = None,
        marks: list[dict[str, Any]] | None = None,
        num_steps: int | None = None,
        check_profile: bool = True,
        transport: Callable[[str, dict[str, Any] | None], dict[str, Any]] | None = None,
    ) -> None:
        self.url = (url or default_url()).rstrip("/")
        self.profile = profile
        self.marks = list(marks or [])
        self.num_steps = num_steps
        self.check_profile = check_profile
        timeout = float(timeout_s or os.environ.get("GAP_GR00T_TIMEOUT_S") or 30.0)
        self.transport = transport or make_transport(
            self.url, timeout, codec or os.environ.get("GAP_GR00T_IMAGE_CODEC", "jpeg")
        )
        self._info: dict[str, Any] | None = None

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        reply = self.transport(endpoint, data)
        if not isinstance(reply, dict):
            raise PolicyServerError(f"{endpoint}: reply is {type(reply).__name__}, not a mapping")
        if "error" in reply:
            raise PolicyServerError(f"{endpoint}: {reply['error']}")
        return reply

    def info(self, refresh: bool = False) -> dict[str, Any]:
        """The server's ``/info``, checked once: a known protocol and (with a profile) the same profile."""
        if self._info is None or refresh:
            info = self.call("info")
            proto = str(info.get("protocol") or "")
            if proto not in PROTOCOLS and not set(PROTOCOLS) & set(info.get("compatible") or []):
                raise PolicyServerError(
                    f"server speaks {proto or 'no protocol'!r}, not one of {list(PROTOCOLS)}"
                )
            served = info.get("profile")
            if (
                self.check_profile
                and self.profile is not None
                and served
                and served != self.profile.name
            ):
                raise PolicyServerError(
                    f"server runs profile {served!r}, the client maps with {self.profile.name!r} "
                    "(different offsets / gripper maps; pass check_profile=False only if they agree)"
                )
            self._info = info
        return self._info

    @property
    def wire_frame(self) -> str:
        """``model`` unless a gr00t-wam/1 server maps every request itself (``wire_frame: station``)."""
        info = self.info()
        if info.get("protocol") != "vla-yam/1" and info.get("wire_frame") == "station":
            return "station"
        return "model"

    def reset(self) -> dict[str, Any]:
        return self.call("reset", {})

    def request(self, model_input: dict[str, Any], instruction: str) -> dict[str, Any]:
        """The ``act`` body for one model input (state in model frame, profile layout)."""
        names = list(model_input.get("state_names") or [s.name for s in WIRE_LAYOUT])
        state = np.asarray(model_input["state"], dtype=np.float64).reshape(-1)
        if len(names) != state.size:
            raise ValueError(f"state has {state.size} values for {len(names)} names")
        by_name = dict(zip(names, state, strict=True))
        missing = [s.name for s in WIRE_LAYOUT if s.name not in by_name]
        if missing:
            raise ValueError(f"the wire needs all 14 columns; the state lacks {missing}")
        joints = np.array([by_name[s.name] for s in WIRE_LAYOUT])
        frame = self.wire_frame
        if frame == "station":
            if self.profile is None:
                raise ValueError(
                    "this server takes station-frame joints; give the client the profile"
                )
            joints = self.profile.to_station(joints, WIRE_LAYOUT)
        body: dict[str, Any] = {
            "instruction": str(instruction),
            "images": {
                str(r): np.asarray(img) for r, img in (model_input.get("images") or {}).items()
            },
            "joints": joints,
            "frame": frame,
        }
        if model_input.get("t") is not None:
            body["t"] = float(model_input["t"])
        if self.marks:
            body["marks"] = self.marks
        if self.num_steps is not None:
            body["num_steps"] = int(self.num_steps)
        return body

    def act(self, model_input: dict[str, Any], instruction: str) -> dict[str, Any]:
        sent_frame = self.wire_frame
        reply = self.call("act", self.request(model_input, instruction))
        actions = np.asarray(reply.get("actions"), dtype=np.float64)
        if (
            actions.ndim != 2
            or actions.shape[1] != len(WIRE_LAYOUT)
            or not np.all(np.isfinite(actions))
        ):
            raise PolicyServerError(
                f"act: actions are {actions.shape}, want finite [T, {len(WIRE_LAYOUT)}]"
            )
        frame = str(reply.get("frame") or sent_frame)
        if frame == "policy":
            frame = "model"
        if frame not in ("model", "station"):
            raise PolicyServerError(f"act: unknown action frame {frame!r}")
        if self.profile is not None and self.profile.action.layout != WIRE_LAYOUT:
            actions = _reorder(actions, self.profile.action.layout)
        return {
            **{k: v for k, v in reply.items() if k != "actions"},
            "actions": actions,
            "frame": frame,
        }


def _reorder(actions: np.ndarray, layout: tuple[Slot, ...]) -> np.ndarray:
    cols = {s: i for i, s in enumerate(WIRE_LAYOUT)}
    missing = [s.name for s in layout if s not in cols]
    if missing:
        raise PolicyServerError(f"the profile's action layout names {missing}, not on the wire")
    return actions[:, [cols[s] for s in layout]]
