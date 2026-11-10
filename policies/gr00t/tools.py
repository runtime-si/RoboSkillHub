"""gr00t policy bundle - hand a contact-rich step of a GaP graph to a GR00T model on the YAM station.

The graph decides what and where (perception, planning); this bundle does the how: it sends the
station's three camera views, the 14-D joint/gripper state, an instruction and optional grasp /
release marks to a GR00T server (``server.py``, gr00t-wam/1 over HTTP or ZMQ) and executes the action
chunks it returns through the robot connector until a success check passes or the step budget
runs out. The marks are GR00T 2's visual steering (points on the policy's camera view); the N1.7
backend emulates them by drawing them into the image.

Tools:

- ``gr00t.act(observation, instruction, marks=None)``: one chunk, ``[T, 14]`` absolute joint +
  opening targets in the station's joint frame.
- ``gr00t.run_skill(instruction, marks, max_steps, success_check)``: closed loop through
  ``robot.get_observation`` / ``robot.execute_trajectory`` / ``robot.open_gripper`` /
  ``robot.close_gripper`` (``robot.stream_dual`` on request, for lockstep bimanual chunks).
- ``gr00t.marks_from_points(observation, points)``: world points (a grasp, a release) to pixel marks
  on a camera's view.
- ``gr00t.info()``: what the server runs.

Env knobs: ``GAP_GR00T_URL`` (``http://host:port`` or ``zmq://host:port``; default
``http://127.0.0.1:5555``), ``GAP_GR00T_TIMEOUT_S`` (30), ``GAP_GR00T_IMAGE_CODEC`` (``jpeg`` |
``png`` | ``npy``; jpeg), ``GAP_GR00T_JOINT_OFFSET`` (JSON ``{"left": [6], "right": [6]}``:
station joint = policy joint + offset), ``GAP_GR00T_CAMERAS`` (JSON role -> camera name).
Importing this module pulls neither torch nor the model; everything heavy runs in the server.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import tool

ROLES = ("top", "left_wrist", "right_wrist")
DOF = 14
ARM_COLS = {"left": slice(0, 7), "right": slice(7, 14)}
ARM_ID = {"left": 0, "right": 1}

GRIP_CLOSE_BELOW = 0.35
GRIP_OPEN_ABOVE = 0.65
"""Hysteresis on the predicted opening (0 closed, 1 open): the connector's jaws are open or closed."""

MAX_JOINT_STEP_RAD = 0.35
"""Largest change per joint between the measured arm and the first target, or between two targets
(1/30 s apart). The MolmoAct2-YAM station adapter guards the same jump (its README: a 63 deg base
jump from a clipped normaliser); 0.35 rad in 33 ms is beyond the YAM's joint speed."""

DEFAULT_URL = "http://127.0.0.1:5555"


class ActResult(TypedDict):
    actions: np.ndarray  # float32 [T, 14], station joint frame, opening 0 closed .. 1 open
    dt: float  # seconds between rows
    horizon: int
    steering: str  # "emulated-overlay" (N1.7 draws the marks) | "native" | "none"
    latency_ms: float  # round trip, client side
    server_ms: dict[str, float]
    gpu_mem_mb: dict[str, float]
    backend: str
    model: str
    marks_sent: list[dict[str, Any]]


class MarksResult(TypedDict):
    marks: list[dict[str, Any]]
    dropped: list[dict[str, Any]]


class RunResult(TypedDict):
    route: str  # succeeded | max_steps | unsafe_action
    steps: int  # action rows executed
    chunks: int
    success: bool
    check: dict[str, Any]  # the last success-check reply
    latency_ms_mean: float
    latency_ms_p95: float
    server_inference_ms_mean: float
    gpu_mem_mb: dict[str, float]
    gripper_events: list[dict[str, Any]]
    detail: str


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class Client:
    """gr00t-wam/1 over HTTP (stdlib only) or ZMQ (pyzmq + msgpack-numpy, when installed)."""

    def __init__(self, url: str | None = None, timeout_s: float | None = None) -> None:
        self.url = (url or os.environ.get("GAP_GR00T_URL") or DEFAULT_URL).rstrip("/")
        self.timeout_s = float(timeout_s or os.environ.get("GAP_GR00T_TIMEOUT_S") or 30.0)
        self.codec = os.environ.get("GAP_GR00T_IMAGE_CODEC", "jpeg")
        self.zmq = self.url.startswith(("zmq://", "tcp://"))
        self._sock: Any = None

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._zmq(endpoint, data) if self.zmq else self._http(endpoint, data)

    def _http(self, endpoint: str, data: dict[str, Any] | None) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        body = None
        if data is not None:
            wire = dict(data)
            if "images" in wire:
                wire["images"] = {k: self._encode(v) for k, v in wire["images"].items()}
            body = json.dumps(_jsonable(wire)).encode()
        req = urllib.request.Request(
            f"{self.url}/{endpoint}",
            data=body,
            method="POST" if body is not None else "GET",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                reply = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                reply = json.loads(e.read())
            except Exception:
                reply = {"error": f"HTTP {e.code}"}
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"gr00t server: {reply['error']}")
        return reply

    def _zmq(self, endpoint: str, data: dict[str, Any] | None) -> dict[str, Any]:
        import msgpack
        import msgpack_numpy as mnp
        import zmq

        if self._sock is None:
            self._sock = zmq.Context.instance().socket(zmq.REQ)
            self._sock.setsockopt(zmq.RCVTIMEO, int(self.timeout_s * 1000))
            self._sock.setsockopt(zmq.SNDTIMEO, int(self.timeout_s * 1000))
            self._sock.setsockopt(zmq.LINGER, 0)
            self._sock.connect("tcp://" + self.url.split("://", 1)[1])
        try:
            self._sock.send(msgpack.packb({"endpoint": endpoint, "data": data or {}}, default=mnp.encode))
            reply = msgpack.unpackb(self._sock.recv(), object_hook=mnp.decode, raw=False)
        except zmq.error.Again:
            self._sock.close()
            self._sock = None  # a REQ socket that missed its reply cannot send again
            raise
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"gr00t server: {reply['error']}")
        return reply

    def _encode(self, image: np.ndarray) -> dict[str, Any]:
        a = np.ascontiguousarray(image)
        if self.codec == "npy":
            return {"npy": base64.b64encode(a.tobytes()).decode(), "shape": list(a.shape), "dtype": str(a.dtype)}
        from PIL import Image

        buf = io.BytesIO()
        if self.codec == "png":
            Image.fromarray(a).save(buf, "PNG")
            return {"png": base64.b64encode(buf.getvalue()).decode()}
        Image.fromarray(a).save(buf, "JPEG", quality=95)
        return {"jpeg": base64.b64encode(buf.getvalue()).decode()}


_clients: dict[str, Client] = {}


def _client() -> Client:
    url = os.environ.get("GAP_GR00T_URL") or DEFAULT_URL
    if url not in _clients:
        _clients[url] = Client(url)
    return _clients[url]


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


# ---------------------------------------------------------------------------
# Observation -> policy inputs
# ---------------------------------------------------------------------------


def joint_offset(override: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """station joint = policy joint + offset, per arm (6 values). Identity unless configured.

    The real YAM station needs ``{"left": [pi/2, 0, 0, 0, 0, 0], "right": [pi/2, 0, 0, 0, 0, 0]}``
    (MolmoAct2-YAM configs/molmo-yam-joint-mapping.json: hardware base joint at 90 deg is the
    MolmoAct2 data's 0); the station twin is already in the data's frame (SKILL.md, Measured)."""
    raw = override if override is not None else json.loads(os.environ.get("GAP_GR00T_JOINT_OFFSET") or "{}")
    out = {}
    for side in ("left", "right"):
        v = np.asarray(raw.get(side, [0.0] * 6), dtype=np.float64).reshape(-1)
        if v.shape != (6,):
            raise ValueError(f"joint_offset[{side!r}] needs 6 values, got {v.shape}")
        out[side] = v
    return out


def _camera_name(cam: dict[str, Any]) -> str:
    return str(cam.get("name") or "").lower()


def camera_roles(observation: dict[str, Any], cameras: dict[str, str] | None = None) -> dict[str, dict]:
    """The observation's camera frames by role (top, left_wrist, right_wrist).

    ``cameras`` (or ``GAP_GR00T_CAMERAS``) names them explicitly; otherwise by name: top / base /
    scene / overhead for the top view, a name with left (or a lone "wrist") for the left wrist, one
    with right for the right wrist. Roles not found are left out (the server sends black)."""
    frames = [c for c in (observation.get("cameras") or []) if isinstance(c, dict)]
    explicit = cameras if cameras is not None else json.loads(os.environ.get("GAP_GR00T_CAMERAS") or "{}")
    by_name = {_camera_name(c): c for c in frames}
    out: dict[str, dict] = {}
    for role, name in (explicit or {}).items():
        if role not in ROLES:
            raise ValueError(f"camera role {role!r} is not one of {list(ROLES)}")
        if str(name).lower() not in by_name:
            raise ValueError(f"no camera {name!r} in the observation ({sorted(by_name)})")
        out[role] = by_name[str(name).lower()]
    for c in frames:
        n = _camera_name(c)
        if "wrist" in n or "hand" in n:
            role = "right_wrist" if "right" in n else "left_wrist"
        elif any(k in n for k in ("top", "base", "scene", "overhead")):
            role = "top"
        else:
            continue
        out.setdefault(role, c)
    if "top" not in out:
        rest = [c for c in frames if c not in out.values()]
        if rest:
            out["top"] = rest[0]
    return out


def _rgb(cam: dict[str, Any]) -> np.ndarray:
    a = cam.get("rgb")
    if a is None:
        a = (cam.get("images") or {}).get("rgb")
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"camera {cam.get('name')!r}: rgb is {a.dtype} {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def _xyz(v: Any) -> np.ndarray:
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    return np.asarray(v, dtype=np.float64).reshape(3)


def _quat_wxyz(q: Any) -> np.ndarray:
    if isinstance(q, dict):
        return np.array([float(q["w"]), float(q["x"]), float(q["y"]), float(q["z"])])
    return np.asarray(q, dtype=np.float64).reshape(4)


def _pose(p: Any) -> tuple[np.ndarray, np.ndarray]:
    """(position, wxyz) from an Se3Pose dict or a 7-vector [x, y, z, qw, qx, qy, qz]."""
    if isinstance(p, dict):
        return _xyz(p["position"]), _quat_wxyz(p["rotation"])
    a = np.asarray(p, dtype=np.float64).reshape(-1)
    return a[:3], a[3:7]


def _rot(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz / (np.linalg.norm(q_wxyz) or 1.0)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def measured_joints(observation: dict[str, Any]) -> np.ndarray:
    """The station's 14-D [left q1..q6, left opening, right q1..q6, right opening] (arm 0 is left).

    A one-armed observation fills the other arm with zeros and an open jaw."""
    arms = observation.get("arms") or []
    if not arms:
        raise ValueError("the observation has no arms")
    out = np.zeros(DOF)
    out[6] = out[13] = 1.0
    for side, arm in zip(("left", "right"), arms, strict=False):
        q = np.asarray((arm.get("joint_state") or {}).get("positions"), dtype=np.float64).reshape(-1)
        if q.size < 6:
            raise ValueError(f"{side} arm reports {q.size} joints; the YAM has 6")
        cols = ARM_COLS[side]
        out[cols.start : cols.start + 6] = q[:6]
        out[cols.stop - 1] = float(arm.get("gripper_fraction", 1.0))
    return out


def to_policy(joints: np.ndarray, offset: dict[str, np.ndarray]) -> np.ndarray:
    out = np.array(joints, dtype=np.float64)
    out[0:6] -= offset["left"]
    out[7:13] -= offset["right"]
    return out


def to_station(actions: np.ndarray, offset: dict[str, np.ndarray]) -> np.ndarray:
    out = np.array(actions, dtype=np.float64)
    out[..., 0:6] += offset["left"]
    out[..., 7:13] += offset["right"]
    return out


def project(point: np.ndarray, cam: dict[str, Any]) -> tuple[float, float, float]:
    """(u, v, depth) of a world point in a posed pinhole camera (OpenCV axes: x right, y down, z
    forward - the convention geometry.pixel_to_world_point back-projects with)."""
    k = np.asarray(cam.get("intrinsics"), dtype=np.float64).reshape(3, 3)
    pos, q = _pose(cam.get("pose"))
    p = _rot(q).T @ (np.asarray(point, dtype=np.float64) - pos)
    if p[2] <= 1e-6:
        return float("nan"), float("nan"), float(p[2])
    return float(k[0, 0] * p[0] / p[2] + k[0, 2]), float(k[1, 1] * p[1] / p[2] + k[1, 2]), float(p[2])


def marks_for(observation: dict[str, Any], marks: list[dict] | None, cameras: dict[str, str] | None = None) -> tuple[list[dict], list[dict]]:
    """Wire marks (pixels on a role's view) from pixel marks and world-point marks; off-view ones dropped."""
    roles = camera_roles(observation, cameras)
    out, dropped = [], []
    for m in marks or []:
        kind = str(m.get("kind"))
        if kind not in ("grasp", "release"):
            raise ValueError(f"mark kind {kind!r} is not grasp or release")
        role = str(m.get("camera") or "top")
        cam = roles.get(role)
        if "uv" in m:
            u, v = (float(c) for c in m["uv"])
        elif "point" in m:
            if cam is None or cam.get("intrinsics") is None or cam.get("pose") is None:
                dropped.append({**m, "why": f"no posed {role} camera to project onto"})
                continue
            u, v, _ = project(_xyz(m["point"]), cam)
        else:
            raise ValueError("a mark needs uv (pixels) or point (world xyz)")
        if cam is not None:
            h, w = _rgb(cam).shape[:2]
            if not (0 <= u < w and 0 <= v < h):
                dropped.append({**m, "why": f"({u:.0f}, {v:.0f}) is off the {w}x{h} {role} view"})
                continue
        wire = {"kind": kind, "camera": role, "uv": [round(u, 1), round(v, 1)]}
        if m.get("arm"):
            wire["arm"] = str(m["arm"])
        out.append(wire)
    return out, dropped


def request(
    observation: dict[str, Any],
    instruction: str,
    marks: list[dict] | None = None,
    cameras: dict[str, str] | None = None,
    offset: dict[str, np.ndarray] | None = None,
    mark_text: bool = False,
) -> dict[str, Any]:
    """The gr00t-wam/1 ``act`` body for one observation."""
    off = offset if offset is not None else joint_offset()
    roles = camera_roles(observation, cameras)
    wire_marks, _ = marks_for(observation, marks, cameras)
    ee = {}
    for side, arm in zip(("left", "right"), observation.get("arms") or [], strict=False):
        if arm.get("ee_pose") is not None:
            pos, q = _pose(arm["ee_pose"])
            ee[side] = [*pos.tolist(), *q.tolist()]
    return {
        "instruction": instruction,
        "images": {role: _rgb(cam) for role, cam in roles.items()},
        "joints": to_policy(measured_joints(observation), off).tolist(),
        "ee": ee or None,
        "marks": wire_marks,
        "t": time.time(),
        "mark_text": bool(mark_text),
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="gr00t.info",
    summary="What the GR00T server runs: protocol, backend (n1.7 / gr00t2), checkpoint, horizon, rate, steering mode, GPU memory.",
    tags=("policy",),
)
def info() -> dict[str, Any]:
    try:
        return _client().call("info")
    except Exception as e:
        raise ToolError("gr00t.info", f"server at {_client().url} did not answer: {e}") from e


@tool(
    name="gr00t.marks_from_points",
    summary="Grasp / release marks for gr00t.act from world points: each point projected onto a camera's view (pixels), off-view points dropped.",
    tags=("policy", "perception"),
)
def marks_from_points(
    observation: dict[str, Any], points: list[dict[str, Any]], camera: str = "top"
) -> MarksResult:
    """``points``: ``[{"kind": "grasp" | "release", "point": Vec3 | [x, y, z], "arm": ...}]``."""
    try:
        marks, dropped = marks_for(observation, [{**p, "camera": p.get("camera", camera)} for p in points])
    except ValueError as e:
        raise ToolError("gr00t.marks_from_points", str(e)) from e
    return {"marks": marks, "dropped": dropped}


@tool(
    name="gr00t.act",
    summary="One GR00T action chunk for the YAM station: [T, 14] absolute joint + jaw-opening targets (left 7, right 7) from the three camera views, the joint state, an instruction and optional grasp/release marks.",
    tags=("policy",),
)
def act(
    observation: dict[str, Any],
    instruction: str,
    marks: list[dict[str, Any]] | None = None,
    cameras: dict[str, str] | None = None,
    joint_offset_rad: dict[str, Any] | None = None,
    mark_text: bool = False,
) -> ActResult:
    """``marks``: ``[{"kind": "grasp" | "release", "uv": [u, v] | "point": [x, y, z], "camera": role}]``
    (role top / left_wrist / right_wrist; default top). ``joint_offset_rad`` overrides
    ``GAP_GR00T_JOINT_OFFSET``. The actions come back in the station's joint frame."""
    name = "gr00t.act"
    if not isinstance(instruction, str) or not instruction.strip():
        raise ToolError(name, "instruction must be a non-empty sentence")
    try:
        off = joint_offset(joint_offset_rad)
        body = request(observation, instruction.strip(), marks, cameras, off, mark_text)
    except ValueError as e:
        raise ToolError(name, str(e)) from e
    client = _client()
    t0 = time.perf_counter()
    try:
        reply = client.call("act", body)
    except Exception as e:
        raise ToolError(name, f"server at {client.url}: {e}") from e
    latency = (time.perf_counter() - t0) * 1e3
    actions = np.asarray(reply["actions"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != DOF:
        raise ToolError(name, f"server returned actions of shape {actions.shape}")
    return {
        "actions": to_station(actions, off).astype(np.float32),
        "dt": float(reply.get("dt", 1.0 / 30.0)),
        "horizon": int(actions.shape[0]),
        "steering": str(reply.get("steering", "none")),
        "latency_ms": round(latency, 2),
        "server_ms": dict(reply.get("timing_ms") or {}),
        "gpu_mem_mb": dict(reply.get("gpu_mem_mb") or {}),
        "backend": str(reply.get("backend", "")),
        "model": str(reply.get("model", "")),
        "marks_sent": body["marks"],
    }


def _check(ctx: Any, spec: Any) -> tuple[bool, dict[str, Any]]:
    """Evaluate a success check: a callable ``(ctx) -> bool | dict``, or ``{"tool": name, "args":
    {...}, "field": key, "equals": value}`` (``field`` truthy when ``equals`` is left out)."""
    if callable(spec):
        r = spec(ctx)
        return (bool(r.get("success")) if isinstance(r, dict) else bool(r)), (r if isinstance(r, dict) else {"value": r})
    reply = ctx.tool(str(spec["tool"]), **dict(spec.get("args") or {}))
    value: Any = reply
    for part in str(spec.get("field") or "success").split("."):
        value = value.get(part) if isinstance(value, dict) else None
    ok = value == spec["equals"] if "equals" in spec else bool(value)
    return ok, reply if isinstance(reply, dict) else {"value": reply}


def _resample(rows: np.ndarray, factor: int) -> np.ndarray:
    """Linear interpolation to ``factor`` waypoints per row, starting after ``rows[0]``'s predecessor."""
    if factor <= 1 or len(rows) < 2:
        return rows
    out = [rows[0]]
    for a, b in zip(rows[:-1], rows[1:], strict=True):
        out.extend(a + (b - a) * (i / factor) for i in range(1, factor + 1))
    return np.asarray(out)


@tool(
    name="gr00t.run_skill",
    summary="Run GR00T closed-loop on the YAM station: query a chunk from the current cameras + joints (+ grasp/release marks), execute its first rows, repeat until a success check passes or max_steps rows ran.",
    tags=("policy", "execute"),
)
def run_skill(
    ctx: Any,
    instruction: str,
    marks: list[dict[str, Any]] | None = None,
    max_steps: int = 480,
    success_check: Any = None,
    arms: str = "both",
    execute_steps: int = 8,
    check_every: int = 1,
    control_hz: float = 0.0,
    cameras: dict[str, str] | None = None,
    joint_offset_rad: dict[str, Any] | None = None,
    mark_text: bool = False,
    max_joint_step_rad: float = MAX_JOINT_STEP_RAD,
    tolerance: float = 0.0,
    executor: str = "trajectory",
    on_chunk: Callable[[dict[str, Any]], None] | None = None,
) -> RunResult:
    """Closed-loop GR00T on the station's arms, chunk by chunk.

    - ``marks``: as for ``gr00t.act``; world-point marks are re-projected every chunk, so they stay
      put on a wrist view while the wrist moves.
    - ``success_check``: ``{"tool": ..., "args": {...}, "field": ..., "equals": ...}`` (or a callable
      when called from Python), checked every ``check_every`` chunks and once at the end; ``None``
      runs the budget out (route ``max_steps``).
    - ``arms``: ``both``, ``left`` or ``right`` - the arms that move; the other holds.
    - ``execute_steps``: rows of each chunk executed before re-querying (receding horizon; a jaw
      switch ends the segment early so the jaws act at the predicted time).
    - ``control_hz``: the connector's waypoint rate; rows (``dt`` apart) are interpolated to it.
      0 sends the rows as they are.
    - ``executor``: ``trajectory`` (``robot.execute_trajectory`` per arm, one after the other),
      ``stream_dual`` (both arms in lockstep through ``robot.stream_dual``, which only some
      connectors have - the station twin does), ``auto`` (stream_dual when both arms move and the
      connector has it, else trajectory) or ``joints`` (``robot.move_to_joints`` per row).
    """
    name = "gr00t.run_skill"
    if arms not in ("both", "left", "right"):
        raise ToolError(name, f"arms must be both, left or right, not {arms!r}")
    if executor not in ("auto", "trajectory", "stream_dual", "joints"):
        raise ToolError(name, f"unknown executor {executor!r}")
    sides = ["left", "right"] if arms == "both" else [arms]
    steps = chunks = 0
    latencies: list[float] = []
    inference: list[float] = []
    events: list[dict[str, Any]] = []
    jaw: dict[str, bool | None] = {s: None for s in sides}  # True = commanded closed
    last_check: dict[str, Any] = {}
    gpu: dict[str, float] = {}
    use_dual = executor in ("auto", "stream_dual") and len(sides) == 2

    def finish(route: str, ok: bool, detail: str) -> RunResult:
        lat = np.asarray(latencies or [0.0])
        return {
            "route": route,
            "steps": steps,
            "chunks": chunks,
            "success": ok,
            "check": last_check,
            "latency_ms_mean": round(float(lat.mean()), 2),
            "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
            "server_inference_ms_mean": round(float(np.mean(inference)) if inference else 0.0, 2),
            "gpu_mem_mb": gpu,
            "gripper_events": events,
            "detail": detail,
        }

    while steps < max_steps:
        obs = ctx.tool("robot.get_observation")
        current = measured_joints(obs)
        out = act(obs, instruction, marks, cameras, joint_offset_rad, mark_text)
        chunks += 1
        latencies.append(out["latency_ms"])
        inference.append(float(out["server_ms"].get("inference", out["server_ms"].get("total", 0.0))))
        gpu = out["gpu_mem_mb"] or gpu
        rows = np.asarray(out["actions"], dtype=np.float64)[: max(1, min(execute_steps, max_steps - steps))]

        # Guard: no joint may jump further than a YAM can move in one row.
        prev = np.concatenate([current[None], rows[:-1]])
        joint_cols = [c for s in sides for c in range(ARM_COLS[s].start, ARM_COLS[s].start + 6)]
        jump = np.abs(rows[:, joint_cols] - prev[:, joint_cols])
        if float(jump.max()) > max_joint_step_rad:
            i, j = np.unravel_index(int(np.argmax(jump)), jump.shape)
            return finish(
                "unsafe_action",
                False,
                f"chunk {chunks} row {i}: joint {joint_cols[j]} would move {jump[i, j]:.3f} rad (limit {max_joint_step_rad})",
            )

        # End the segment at the first jaw switch so the jaws act when predicted.
        switch_at, switches = len(rows), {}
        for s in sides:
            col = ARM_COLS[s].stop - 1
            state = jaw[s] if jaw[s] is not None else current[col] < 0.5
            for i, g in enumerate(rows[:, col]):
                want = True if g < GRIP_CLOSE_BELOW else False if g > GRIP_OPEN_ABOVE else state
                if want != state:
                    if i + 1 < switch_at:
                        switch_at, switches = i + 1, {s: want}
                    elif i + 1 == switch_at:
                        switches[s] = want
                    break
        rows = rows[:switch_at]

        if on_chunk is not None:
            on_chunk({"chunk": chunks, "rows": rows, **{k: out[k] for k in ("latency_ms", "server_ms", "marks_sent")}})

        hz = float(control_hz or 0.0)
        factor = max(1, round(hz * out["dt"])) if hz > 0 else 1
        tracks = {}
        for s in sides:
            arm_rows = np.concatenate([current[None, ARM_COLS[s].start : ARM_COLS[s].start + 6], rows[:, ARM_COLS[s].start : ARM_COLS[s].start + 6]])
            tracks[s] = _resample(arm_rows, factor)[1:]
        tol = {"tolerance": float(tolerance)} if tolerance > 0 else {}
        if executor == "joints":
            for i in range(len(rows)):
                for s in sides:
                    q = rows[i, ARM_COLS[s].start : ARM_COLS[s].start + 6]
                    ctx.tool("robot.move_to_joints", joint_config={"positions": q.tolist()}, arm_id=ARM_ID[s], **tol)
        elif use_dual:
            try:
                ctx.tool(
                    "robot.stream_dual",
                    tracks={ARM_ID[s]: tracks[s].tolist() for s in sides},
                    **({"tolerance": float(tolerance)} if tolerance > 0 else {}),
                )
            except Exception as e:
                if executor == "stream_dual":
                    raise
                use_dual = False
                events.append({"chunk": chunks, "note": f"robot.stream_dual unavailable ({type(e).__name__}); executing per arm"})
        if executor != "joints" and not use_dual:
            for s in sides:
                kw = {"arm_id": ARM_ID[s]} if ARM_ID[s] else {}
                ctx.tool(
                    "robot.execute_trajectory",
                    trajectory={"waypoints": [{"positions": q.tolist()} for q in tracks[s]]},
                    **kw,
                    **tol,
                )
        steps += len(rows)

        for s, closed in switches.items():
            kw = {"arm_id": ARM_ID[s]} if ARM_ID[s] else {}
            ctx.tool("robot.close_gripper" if closed else "robot.open_gripper", **kw)
            jaw[s] = closed
            events.append({"chunk": chunks, "step": steps, "arm": s, "jaw": "close" if closed else "open"})

        if success_check is not None and chunks % max(1, int(check_every)) == 0:
            ok, last_check = _check(ctx, success_check)
            if ok:
                return finish("succeeded", True, f"success check passed after {steps} rows ({chunks} chunks)")

    if success_check is not None:
        ok, last_check = _check(ctx, success_check)
        if ok:
            return finish("succeeded", True, f"success check passed after {steps} rows ({chunks} chunks)")
    return finish("max_steps", False, f"{steps} rows ({chunks} chunks) without the success check passing")
