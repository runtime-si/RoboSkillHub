"""GR00T policy server for the YAM station: one backend behind the gr00t-wam/1 protocol.

    python server.py --backend n1.7 --profile yam-molmoact2 --port 5555
    python server.py --backend n1.7 --profile xdof --port 5556          # base N1.7, zero-shot tag
    python server.py --backend fake --port 5557                          # CPU, no model (tests)

Runs on a GPU host in the Isaac-GR00T environment (this bundle's pyproject). The client is
``tools.py`` (``gr00t.act`` / ``gr00t.run_skill``); it speaks only YAM terms - three camera roles,
the 14-D joint/gripper vector, an instruction and optional grasp/release marks - so the backend
behind this server can change (N1.7 now, GR00T 2 when it is released) without touching a graph.

Transports (pick with --transport):

- ``http`` (default): ``GET /info``, ``POST /act``, ``POST /reset``, JSON bodies. Images travel as
  ``{"jpeg": b64}``, ``{"png": b64}`` or ``{"npy": b64, "shape": [...], "dtype": "uint8"}``.
- ``zmq``: a REP socket with Isaac-GR00T's framing (``gr00t/policy/server_client.py``): msgpack
  ``{"endpoint": "act" | "info" | "reset" | "ping", "data": {...}}`` with numpy arrays through
  msgpack-numpy; an error comes back as ``{"error": "..."}``.

The ``act`` request (both transports):

    instruction: str
    images: {"top" | "left_wrist" | "right_wrist": uint8 [H, W, 3]}
    joints: float[14]  [left q1..q6, left opening, right q1..q6, right opening], policy frame,
            opening 0 closed .. 1 open
    ee: optional {"left" | "right": [x, y, z, qw, qx, qy, qz]} (the xdof profile reads it)
    marks: optional [{"kind": "grasp" | "release", "camera": role, "uv": [u, v], "arm": ...}]
    t: optional client time [s] (frame history for profiles that read past frames)
    mark_text: optional bool, append a sentence naming the marks to the instruction

and the reply: ``actions`` float32 [T, 14] (same layout, absolute), ``dt`` [s], ``steering``
("emulated-overlay" for N1.7: the marks are drawn into the image; "native" for a backend that takes
them as input), ``timing_ms``, ``gpu_mem_mb`` and the model's identity.

Profiles map the roles onto a checkpoint's modality keys:

- ``yam-molmoact2``: robocurve/gr00t-n1.7-yam-molmoact2 (tag new_embodiment; base_view /
  left_wrist_view / right_wrist_view; left_arm, left_gripper, right_arm, right_gripper; 16 x 14
  absolute joints at 30 Hz; experiment_cfg/conf.yaml of the checkpoint).
- ``xdof``: base nvidia/GR00T-N1.7-3B with its pretraining tag xdof_relative_eef_relative_joint
  (three 320x240 views at t-1 s and t, wrist EEF xyz+rot6d, gripper and joint state; 40 steps of
  relative joints the processor turns absolute; processor_config.json of the checkpoint). The EEF
  frame XDOF recorded in is not published: the client sends the arm's world-frame pose.

Backends:

- ``n1.7``: Isaac-GR00T's ``Gr00tPolicy`` (github.com/NVIDIA/Isaac-GR00T, Apache-2.0). Weights
  under the NVIDIA Open Model License. ``--backbone qwen3-vl-2b`` builds the VLM from the ungated
  Qwen/Qwen3-VL-2B-Instruct (Apache-2.0, same architecture) instead of the gated
  nvidia/Cosmos-Reason2-2B; every backbone tensor is then overwritten by the checkpoint's own, but
  the tokenizer / chat template / image-processor configs are Qwen's (see SKILL.md, Measured).
- ``gr00t2``: placeholder for GR00T 2 (announced 2026-11-09, no weights or code yet);
  :class:`Gr00t2Backend` documents what it must provide.
- ``fake``: holds the current joints and walks the grippers to the marks' intent; no model.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("gr00t.server")

PROTOCOL = "gr00t-wam/1"
ROLES = ("top", "left_wrist", "right_wrist")
DOF = 14
LEFT_Q, LEFT_G, RIGHT_Q, RIGHT_G = slice(0, 6), slice(6, 7), slice(7, 13), slice(13, 14)

#: Pinned checkpoints (Hugging Face repo -> commit), as evaluated in SKILL.md.
CHECKPOINTS = {
    "robocurve/gr00t-n1.7-yam-molmoact2": "f04fe258e6a5d84c939547cd49043fbc3b1e8985",
    "nvidia/GR00T-N1.7-3B": "2fc962b973bccdd5d8ce4f67cc63b264d6886495",
}
#: Ungated stand-in for the gated nvidia/Cosmos-Reason2-2B backbone (same Qwen3-VL-2B architecture).
QWEN_BACKBONE = ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203")

MARK_COLOURS = {"grasp": (0, 255, 0), "release": (0, 96, 255)}
"""RGB: a green disc where to grasp, a blue disc where to release."""


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------


def decode_image(value: Any) -> np.ndarray:
    """An image from the wire: an ndarray (zmq) or a JSON blob (http)."""
    if isinstance(value, np.ndarray):
        a = value
    elif isinstance(value, dict) and "npy" in value:
        raw = base64.b64decode(value["npy"])
        a = np.frombuffer(raw, dtype=np.dtype(value.get("dtype", "uint8"))).reshape(value["shape"])
    elif isinstance(value, dict) and ("jpeg" in value or "png" in value):
        from PIL import Image

        raw = base64.b64decode(value.get("jpeg") or value.get("png"))
        a = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    else:
        raise ValueError(f"cannot read an image from {type(value).__name__}")
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError(f"images are RGB uint8 [H, W, 3], got {a.dtype} {a.shape}")
    return np.ascontiguousarray(a)


def draw_marks(image: np.ndarray, marks: list[dict], radius: int | None = None) -> np.ndarray:
    """A copy of ``image`` with each mark drawn as a filled disc in a white ring.

    The disc is the visual-steering cue GR00T 2 reads from its camera view; a model never trained
    on it (N1.7) just sees a coloured dot. Radius defaults to 2.5 % of the shorter side."""
    out = image.copy()
    h, w = out.shape[:2]
    r = int(radius or max(3, round(0.025 * min(h, w))))
    yy, xx = np.mgrid[0:h, 0:w]
    for m in marks:
        u, v = (float(c) for c in m["uv"])
        d2 = (xx - u) ** 2 + (yy - v) ** 2
        out[(d2 <= (r + 2) ** 2) & (d2 > r**2)] = (255, 255, 255)
        out[d2 <= r**2] = MARK_COLOURS.get(str(m.get("kind")), (255, 0, 255))
    return out


def mark_sentence(marks: list[dict]) -> str:
    kinds = [str(m.get("kind")) for m in marks]
    parts = []
    if "grasp" in kinds:
        parts.append("grasp at the green dot")
    if "release" in kinds:
        parts.append("release at the blue dot")
    return (" " + " and ".join(parts).capitalize() + ".") if parts else ""


def rot6d_from_quat(q_wxyz: np.ndarray) -> np.ndarray:
    """The first two rows of the rotation matrix (Isaac-GR00T's rot6d, state_action/pose.py)."""
    w, x, y, z = (float(c) for c in q_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return r[:2].reshape(6)


# ---------------------------------------------------------------------------
# Profiles: YAM roles -> a checkpoint's modality keys
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    name: str
    tag: str
    video: dict[str, str]  # modality key -> role
    state: dict[
        str, str
    ]  # modality key -> source (left_q, left_g, right_q, right_g, left_ee, right_ee)
    action: dict[str, str]  # modality key -> destination slice name
    language_key: str
    fps: float = 30.0
    image_size: tuple[int, int] | None = (
        None  # (W, H) the views are resized to, when the keys name it
    )
    default_checkpoint: str = ""


PROFILES = {
    "yam-molmoact2": Profile(
        name="yam-molmoact2",
        tag="new_embodiment",
        video={
            "base_view": "top",
            "left_wrist_view": "left_wrist",
            "right_wrist_view": "right_wrist",
        },
        state={
            "left_arm": "left_q",
            "left_gripper": "left_g",
            "right_arm": "right_q",
            "right_gripper": "right_g",
        },
        action={
            "left_arm": "left_q",
            "left_gripper": "left_g",
            "right_arm": "right_q",
            "right_gripper": "right_g",
        },
        language_key="annotation.human.task_description",
        default_checkpoint="robocurve/gr00t-n1.7-yam-molmoact2",
    ),
    "xdof": Profile(
        name="xdof",
        tag="xdof_relative_eef_relative_joint",
        video={
            "top_camera-images-rgb_320_240": "top",
            "left_camera-images-rgb_320_240": "left_wrist",
            "right_camera-images-rgb_320_240": "right_wrist",
        },
        state={
            "left_wrist_eef": "left_ee",
            "right_wrist_eef": "right_ee",
            "left_gripper_pos": "left_g",
            "right_gripper_pos": "right_g",
            "left_joint_pos": "left_q",
            "right_joint_pos": "right_q",
        },
        action={
            "left_joint_pos": "left_q",
            "left_gripper_pos": "left_g",
            "right_joint_pos": "right_q",
            "right_gripper_pos": "right_g",
        },
        language_key="annotation.task",
        image_size=(320, 240),
        default_checkpoint="nvidia/GR00T-N1.7-3B",
    ),
    "yam-roles": Profile(
        name="yam-roles",
        tag="new_embodiment",
        video={"top": "top", "left_wrist": "left_wrist", "right_wrist": "right_wrist"},
        state={
            "left_arm": "left_q",
            "left_gripper": "left_g",
            "right_arm": "right_q",
            "right_gripper": "right_g",
        },
        action={
            "left_arm": "left_q",
            "left_gripper": "left_g",
            "right_arm": "right_q",
            "right_gripper": "right_g",
        },
        language_key="annotation.human.task_description",
    ),
    "yam-left-roles": Profile(
        name="yam-left-roles",
        tag="new_embodiment",
        video={"top": "top", "left_wrist": "left_wrist"},
        state={"left_arm": "left_q", "left_gripper": "left_g"},
        action={"left_arm": "left_q", "left_gripper": "left_g"},
        language_key="annotation.human.task_description",
    ),
}
"""``yam-roles`` / ``yam-left-roles``: a fine-tune whose LeRobot dataset names its views and joint groups after
the roles (the station's own recorder, ``~/yam_gap/real/record_lerobot.py``, and the hub's sim demos): pass its
checkpoint directory with ``--checkpoint``."""

_SLICES = {"left_q": LEFT_Q, "left_g": LEFT_G, "right_q": RIGHT_Q, "right_g": RIGHT_G}


def state_value(source: str, joints: np.ndarray, ee: dict[str, Any] | None) -> np.ndarray:
    if source in _SLICES:
        return joints[_SLICES[source]].astype(np.float32)
    side = source.split("_")[0]
    pose = np.asarray((ee or {}).get(side, [0, 0, 0, 1, 0, 0, 0]), dtype=np.float64).reshape(-1)
    return np.concatenate([pose[:3], rot6d_from_quat(pose[3:7])]).astype(np.float32)


def assemble_actions(
    profile: Profile, action: dict[str, np.ndarray], joints: np.ndarray
) -> np.ndarray:
    """``[T, 14]`` from the model's per-key ``[1, T, d]`` arrays; keys a profile leaves out hold."""
    horizon = max(np.asarray(v).shape[-2] for v in action.values())
    out = np.tile(joints.astype(np.float32), (horizon, 1))
    for key, dest in profile.action.items():
        if key in action:
            out[:, _SLICES[dest]] = np.asarray(action[key], dtype=np.float32).reshape(horizon, -1)
    return out


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


@dataclass
class ActRequest:
    instruction: str
    images: dict[str, np.ndarray]
    joints: np.ndarray
    ee: dict[str, Any] | None = None
    marks: list[dict] = field(default_factory=list)
    t: float | None = None
    mark_text: bool = False

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> ActRequest:
        instruction = str(data.get("instruction") or "").strip()
        if not instruction:
            raise ValueError("instruction is required")
        joints = np.asarray(data.get("joints"), dtype=np.float64).reshape(-1)
        if joints.shape != (DOF,) or not np.all(np.isfinite(joints)):
            raise ValueError(f"joints must be {DOF} finite numbers, got shape {joints.shape}")
        images = {str(k): decode_image(v) for k, v in (data.get("images") or {}).items()}
        unknown = sorted(set(images) - set(ROLES))
        if unknown:
            raise ValueError(f"unknown camera roles {unknown}; roles are {list(ROLES)}")
        marks = []
        for m in data.get("marks") or []:
            kind, camera = str(m.get("kind")), str(m.get("camera") or "top")
            if kind not in MARK_COLOURS:
                raise ValueError(f"mark kind {kind!r} is not one of {sorted(MARK_COLOURS)}")
            if camera not in ROLES:
                raise ValueError(f"mark camera {camera!r} is not one of {list(ROLES)}")
            uv = [float(c) for c in m.get("uv")]
            marks.append({**m, "kind": kind, "camera": camera, "uv": uv})
        t = data.get("t")
        return cls(
            instruction=instruction,
            images=images,
            joints=joints,
            ee=data.get("ee"),
            marks=marks,
            t=None if t is None else float(t),
            mark_text=bool(data.get("mark_text", False)),
        )


class FakeBackend:
    """No model: holds the arms where they are; a grasp mark closes the left jaw over the chunk's first
    six rows, a release mark (alone) opens it. For protocol tests and dry runs of a graph."""

    steering = "emulated-overlay"

    def __init__(self, horizon: int = 16, fps: float = 30.0) -> None:
        self.horizon, self.dt = horizon, 1.0 / fps
        self.calls = 0

    def info(self) -> dict[str, Any]:
        return {
            "backend": "fake",
            "model": "fake",
            "revision": "",
            "profile": "fake",
            "horizon": self.horizon,
        }

    def reset(self) -> None:
        self.calls = 0

    def act(self, req: ActRequest) -> dict[str, Any]:
        self.calls += 1
        actions = np.tile(req.joints.astype(np.float32), (self.horizon, 1))
        kinds = {m["kind"] for m in req.marks}
        target = 0.0 if "grasp" in kinds else 1.0 if "release" in kinds else None
        if target is not None:
            ramp = np.linspace(req.joints[6], target, 6)
            actions[:, 6] = np.concatenate([ramp, np.full(self.horizon - 6, target)])
        return {"actions": actions, "prompt": req.instruction, "marks_drawn": len(req.marks)}


class N17Backend:
    """GR00T N1.7 through Isaac-GR00T's ``Gr00tPolicy`` (Apache-2.0); marks drawn into the views."""

    steering = "emulated-overlay"

    def __init__(
        self,
        profile: Profile,
        checkpoint: str = "",
        revision: str | None = None,
        device: str = "cuda",
        backbone: str = "canonical",
        fps: float | None = None,
    ) -> None:
        import torch
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.torch = torch
        self.profile = profile
        self.checkpoint = checkpoint or profile.default_checkpoint
        self.revision = revision if revision is not None else CHECKPOINTS.get(self.checkpoint)
        self.backbone = backbone
        t0 = time.time()
        path = self._local(self.checkpoint, self.revision)
        if backbone != "canonical":
            path = self._with_backbone(path, backbone)
        self.policy = Gr00tPolicy(
            embodiment_tag=profile.tag, model_path=str(path), device=device, strict=True
        )
        self.load_s = time.time() - t0
        self.repaired = self._repair_empty_action_dims()
        self.modality = self.policy.get_modality_config()
        self._check_profile()
        self.horizon = len(self.modality["action"].delta_indices)
        self.video_offsets = list(self.modality["video"].delta_indices)
        self.history: deque[tuple[float, dict[str, np.ndarray]]] = deque(maxlen=256)
        self.dt = 1.0 / (fps or profile.fps)

    @staticmethod
    def _local(checkpoint: str, revision: str | None) -> Path:
        if os.path.isdir(checkpoint):
            return Path(checkpoint)
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=checkpoint, revision=revision))

    @staticmethod
    def _with_backbone(path: Path, backbone: str) -> Path:
        """A shadow of the checkpoint whose configs name another VLM to build the backbone from.

        Only ``model_name`` changes (config.json, processor_config.json); the weights are symlinks,
        so every backbone tensor still comes from the checkpoint."""
        if backbone != "qwen3-vl-2b":
            raise ValueError(f"unknown --backbone {backbone!r} (canonical | qwen3-vl-2b)")
        repo, rev = QWEN_BACKBONE
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=repo, revision=rev)
        shadow = Path(tempfile.mkdtemp(prefix="gr00t-shadow-"))
        for item in path.iterdir():
            if item.name == "config.json":
                cfg = json.loads(item.read_text())
                cfg["model_name"] = repo
                (shadow / item.name).write_text(json.dumps(cfg, indent=1))
            elif item.name == "processor_config.json":
                # The base checkpoint leaves model_name out; the loader then falls back to Cosmos.
                cfg = json.loads(item.read_text())
                cfg.setdefault("processor_kwargs", {})["model_name"] = repo
                (shadow / item.name).write_text(json.dumps(cfg, indent=1))
            else:
                (shadow / item.name).symlink_to(item.resolve())
        os.environ.setdefault("GR00T_SHADOW_DIR", str(shadow))
        return shadow

    def _repair_empty_action_dims(self) -> list[str]:
        """Action groups whose released absolute statistics are empty but whose relative ones exist.

        nvidia/GR00T-N1.7-3B@2fc962b ships ``statistics.json`` with empty state / action stats for the
        XDOF tag's left/right_wrist_eef and (40, 9) ``relative_action`` stats; the processor takes a
        group's width from the empty absolute stats (0), so ``decode_action`` slices the chunk wrongly
        and fails. The width is set from the relative stats; no statistic is made up."""
        sap = self.policy.processor.state_action_processor
        tag = self.profile.tag
        params = sap.norm_params.get(tag, {}).get("action", {})
        relative = (sap.statistics.get(tag) or {}).get("relative_action") or {}
        fixed = []
        for key, p in params.items():
            if int(np.asarray(p["dim"])) == 0 and key in relative:
                p["dim"] = np.array(np.asarray(relative[key]["min"]).shape[-1])
                fixed.append(key)
        if fixed:
            dims = getattr(self.policy.processor, "action_dim", None)
            if isinstance(dims, dict):
                dims[tag] = sap.get_action_dim(tag)
            logger.warning(
                "action widths taken from relative stats (absolute stats empty): %s", fixed
            )
        return fixed

    def _check_profile(self) -> None:
        for kind, keys in (
            ("video", self.profile.video),
            ("state", self.profile.state),
            ("action", self.profile.action),
        ):
            have = set(self.modality[kind].modality_keys)
            missing = sorted(set(keys) - have)
            if missing:
                raise ValueError(
                    f"profile {self.profile.name}: {kind} keys {missing} not in the checkpoint ({sorted(have)})"
                )
        if self.profile.language_key not in self.modality["language"].modality_keys:
            raise ValueError(
                f"profile {self.profile.name}: language key {self.profile.language_key!r} not in the checkpoint"
            )

    def info(self) -> dict[str, Any]:
        torch = self.torch
        return {
            "backend": "n1.7",
            "model": self.checkpoint,
            "revision": self.revision or "",
            "profile": self.profile.name,
            "embodiment_tag": self.profile.tag,
            "horizon": self.horizon,
            "video_offsets": self.video_offsets,
            "backbone": QWEN_BACKBONE[0]
            if self.backbone == "qwen3-vl-2b"
            else "nvidia/Cosmos-Reason2-2B",
            "load_s": round(self.load_s, 1),
            "repaired_action_widths": self.repaired,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        }

    def reset(self) -> None:
        self.history.clear()
        self.policy.reset()

    def _frames(self, role_images: dict[str, np.ndarray], t: float) -> dict[str, np.ndarray]:
        """``[1, T, H, W, 3]`` per video key: the frame nearest each offset in the kept history."""
        self.history.append((t, role_images))
        out = {}
        for key, role in self.profile.video.items():
            frames = []
            for off in self.video_offsets:
                want = t + off * self.dt
                _, imgs = min(self.history, key=lambda h, want=want: abs(h[0] - want))
                frames.append(imgs[role])
            out[key] = np.stack(frames)[None]
        return out

    def act(self, req: ActRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        ref = next(iter(req.images.values()), np.zeros((480, 640, 3), np.uint8))
        role_images = {}
        for role in ROLES:
            img = req.images.get(role)
            if img is None:
                img = np.zeros_like(ref)
            marks = [m for m in req.marks if m["camera"] == role]
            if marks:
                img = draw_marks(img, marks)
            if self.profile.image_size is not None:
                from PIL import Image

                img = np.asarray(
                    Image.fromarray(img).resize(self.profile.image_size, Image.BILINEAR)
                )
            role_images[role] = img
        prompt = req.instruction + (mark_sentence(req.marks) if req.mark_text else "")
        obs = {
            "video": self._frames(role_images, req.t if req.t is not None else time.time()),
            "state": {
                key: state_value(src, req.joints, req.ee)[None, None]
                for key, src in self.profile.state.items()
            },
            "language": {self.profile.language_key: [[prompt]]},
        }
        torch = self.torch
        t1 = time.perf_counter()
        action, _ = self.policy.get_action(obs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        return {
            "actions": assemble_actions(self.profile, action, req.joints),
            "prompt": prompt,
            "marks_drawn": len(req.marks),
            "missing_cameras": sorted(set(ROLES) - set(req.images)),
            "timing_ms": {"preprocess": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3},
        }


class Gr00t2Backend:
    """GR00T 2 (world-action model, research.nvidia.com/labs/gear/gr00t-2/) - not released yet.

    Announced 2026-11-09 with no weights, model card or code (code reported for December 2026).
    What a backend here must do, so ``gr00t.act`` / ``gr00t.run_skill`` and every graph stay as
    they are:

    - ``act(req)``: take the three YAM views, the 14-D joint state, the instruction and the marks
      *as input* (the WAM API's visual steering: grasp / release points on its camera view, not
      pixels painted into the image) and return ``{"actions": [T, 14]}`` in the same layout and
      frame, plus optionally ``"predicted_frames": {role: uint8 [K, H, W, 3]}`` - the world model's
      imagined future, which a graph can show or check before executing.
    - ``steering = "native"``; ``info()`` names the checkpoint, its YAM head, horizon and rate.
    - Per-robot heads: pick the YAM (bimanual, 2 x 6 joints + 2 jaws) head; check its joint frame
      against the station's ``joint_offset`` (SKILL.md) the same way as for N1.7.
    - Few-shot post-training (5-shot LoRA) produces a checkpoint directory; ``--checkpoint`` takes it.
    """

    steering = "native"

    def __init__(self, **_: Any) -> None:
        raise NotImplementedError(
            "GR00T 2 has no public weights or code yet (announced 2026-11-09); use --backend n1.7. "
            "Gr00t2Backend's docstring lists what this backend must provide."
        )


def make_backend(args: argparse.Namespace) -> Any:
    if args.backend == "fake":
        return FakeBackend()
    if args.backend == "gr00t2":
        return Gr00t2Backend(checkpoint=args.checkpoint)
    if args.backend == "n1.7":
        return N17Backend(
            PROFILES[args.profile],
            checkpoint=args.checkpoint,
            revision=args.revision,
            device=args.device,
            backbone=args.backbone,
            fps=args.fps,
        )
    raise ValueError(f"unknown backend {args.backend!r}")


# ---------------------------------------------------------------------------
# Service (transport-independent)
# ---------------------------------------------------------------------------


class Service:
    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.lock = threading.Lock()
        self.calls = 0

    def gpu_mem_mb(self) -> dict[str, float]:
        torch = getattr(self.backend, "torch", None)
        if torch is None or not torch.cuda.is_available():
            return {}
        mb = 1024.0 * 1024.0
        return {
            "allocated": round(torch.cuda.memory_allocated() / mb, 1),
            "reserved": round(torch.cuda.memory_reserved() / mb, 1),
            "peak": round(torch.cuda.max_memory_allocated() / mb, 1),
        }

    def info(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "roles": list(ROLES),
            "layout": "left q1..q6, left opening, right q1..q6, right opening (opening 0 closed, 1 open)",
            "steering": getattr(self.backend, "steering", "none"),
            "dt": float(getattr(self.backend, "dt", 1.0 / 30.0)),
            "calls": self.calls,
            "gpu_mem_mb": self.gpu_mem_mb(),
            **self.backend.info(),
        }

    def handle(self, endpoint: str, data: dict[str, Any] | None) -> dict[str, Any]:
        if endpoint == "ping":
            return {"status": "ok"}
        if endpoint == "info":
            return self.info()
        if endpoint == "reset":
            with self.lock:
                self.backend.reset()
            return {"status": "ok"}
        if endpoint == "act":
            req = ActRequest.from_wire(data or {})
            t0 = time.perf_counter()
            with self.lock:
                out = self.backend.act(req)
                self.calls += 1
            actions = np.asarray(out.pop("actions"), dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != DOF or not np.all(np.isfinite(actions)):
                raise RuntimeError(f"backend returned actions {actions.dtype} {actions.shape}")
            timing = dict(out.pop("timing_ms", {}))
            timing["total"] = (time.perf_counter() - t0) * 1e3
            return {
                "actions": actions,
                "dt": float(getattr(self.backend, "dt", 1.0 / 30.0)),
                "horizon": int(actions.shape[0]),
                "steering": getattr(self.backend, "steering", "none"),
                "timing_ms": {k: round(float(v), 2) for k, v in timing.items()},
                "gpu_mem_mb": self.gpu_mem_mb(),
                **{
                    k: v
                    for k, v in self.backend.info().items()
                    if k in ("backend", "model", "revision", "profile")
                },
                **out,
            }
        raise ValueError(f"unknown endpoint {endpoint!r}")


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


def make_http_server(service: Service, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, body: dict[str, Any]) -> None:
            raw = json.dumps(_jsonable(body)).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _serve(self, endpoint: str, data: dict[str, Any] | None) -> None:
            try:
                self._reply(200, service.handle(endpoint, data))
            except ValueError as e:
                self._reply(400, {"error": str(e)})
            except Exception as e:  # the client sees why; the server keeps serving
                logger.exception("request failed")
                self._reply(500, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self) -> None:  # noqa: N802
            self._serve(self.path.strip("/") or "info", None)

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError as e:
                self._reply(400, {"error": f"body is not JSON: {e}"})
                return
            self._serve(self.path.strip("/"), data)

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug(fmt, *args)

    return ThreadingHTTPServer((host, port), Handler)


def serve_zmq(service: Service, host: str, port: int) -> None:
    import msgpack
    import msgpack_numpy as mnp
    import zmq

    def safe_decode(obj: Any) -> Any:
        if (
            isinstance(obj, dict)
            and obj.get(b"nd", obj.get("nd"))
            and obj.get(b"kind", obj.get("kind")) in (b"O", "O")
        ):
            raise ValueError("refusing an object-dtype ndarray (pickle) payload")
        return mnp.decode(obj)

    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    logger.info("gr00t-wam/1 zmq on %s:%d", host, port)
    while True:
        msg = sock.recv()
        try:
            req = msgpack.unpackb(msg, object_hook=safe_decode, raw=False)
            reply = service.handle(str(req.get("endpoint", "act")), req.get("data"))
        except Exception as e:
            logger.exception("request failed")
            reply = {"error": f"{type(e).__name__}: {e}"}
        sock.send(msgpack.packb(reply, default=mnp.encode))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--backend", default="n1.7", choices=["n1.7", "gr00t2", "fake"])
    p.add_argument("--profile", default="yam-molmoact2", choices=sorted(PROFILES))
    p.add_argument("--checkpoint", default="", help="HF repo or local dir (default: the profile's)")
    p.add_argument("--revision", default=None, help="HF commit (default: the pinned one)")
    p.add_argument("--backbone", default="canonical", choices=["canonical", "qwen3-vl-2b"])
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="the checkpoint's control rate (default: the profile's, 30)",
    )
    p.add_argument("--transport", default="http", choices=["http", "zmq"])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="dummy act calls before binding (CUDA kernels, caches)",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    service = Service(make_backend(args))
    for _ in range(args.warmup if args.backend != "fake" else 0):
        blank = np.zeros((480, 640, 3), np.uint8)
        service.handle(
            "act",
            {
                "instruction": "warm up",
                "images": dict.fromkeys(ROLES, blank),
                "joints": [0.0] * DOF,
            },
        )
    service.backend.reset()
    logger.info("ready: %s", json.dumps(_jsonable(service.info())))
    if args.transport == "zmq":
        serve_zmq(service, args.host, args.port)
    else:
        server = make_http_server(service, args.host, args.port)
        logger.info("gr00t-wam/1 http on %s:%d", args.host, args.port)
        try:
            server.serve_forever()
        finally:
            shadow = os.environ.get("GR00T_SHADOW_DIR")
            if shadow:
                shutil.rmtree(shadow, ignore_errors=True)


if __name__ == "__main__":
    main()
