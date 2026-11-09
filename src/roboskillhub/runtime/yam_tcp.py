# SPDX-License-Identifier: Apache-2.0
"""YAM workcell wire clients for the versioned YAM bridge and existing RGB-D server.

Motion requires the explicit ``hub_*`` server extension. The legacy move/grip
protocol has no request deadline, speed contract, or cancellable confirmation;
this client rejects it rather than falling back. It never starts motors, changes
operating mode, invokes finish/reset, or retries an action after a transport loss.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from .contracts import EvidenceMissing, ExecutionFailed, Limits, Rejected, number
from .wire import Client, RemoteError, WireError, decode
from .yam import YamBackend

PROTOCOL = "roboskillhub-yam"
PROTOCOL_VERSION = 1


class YamTcpBackend(YamBackend):
    """Fresh connections, versioned preflight, request IDs and explicit cancellation."""

    def __init__(
        self,
        *,
        limits: Limits,
        port: int,
        host: str = "127.0.0.1",
        connect_timeout_s: float = 3.0,
        stop_timeout_s: float = 3.0,
        object_provider=None,
        expected_kind: str = "real",
    ):
        if expected_kind not in ("real", "dry-run"):
            raise Rejected("expected_kind must explicitly be 'real' or 'dry-run'")
        self._expected_kind = expected_kind
        self._client = Client(host, port, timeout_s=connect_timeout_s)
        self._stop_timeout = number(stop_timeout_s, "stop_timeout_s", 0.01, 10)
        self._last_command_id: str | None = None
        self._owner_id: str | None = None
        self._lease_claimed = False
        self._lease_uncertain = False
        self._uncertain_action = False
        self._unsafe_stop = False
        self._lease_lock = threading.Lock()
        self._remote: dict = {}
        # Connection is read-only. A successful factory never establishes a real
        # station validation result; it only checks the server's declared contract.
        self._preflight(limits, connect_timeout_s)
        super().__init__(
            read_state=self._read_rpc,
            move_cartesian=self._move_rpc,
            set_gripper_width=self._grip_rpc,
            stop=self._stop_rpc,
            limits=limits,
            object_provider=object_provider,
        )

    def _preflight(self, limits: Limits, timeout_s: float) -> dict:
        try:
            description, arrays = self._client.call("hub_describe", timeout_s=timeout_s)
        except (RemoteError, WireError) as exc:
            raise Rejected(
                "YAM server lacks the versioned hub_* bridge; legacy motion is unsupported"
            ) from exc
        if arrays or not isinstance(description, dict):
            raise Rejected("hub_describe must return a header-only description")
        required = {"pose_feedback", "cartesian_motion", "gripper", "stop"}
        capabilities = description.get("capabilities")
        if (
            description.get("protocol") != PROTOCOL
            or type(description.get("protocol_version")) is not int
            or description["protocol_version"] != PROTOCOL_VERSION
            or description.get("kind") != self._expected_kind
            or description.get("mode")
            not in (("confirm", "auto") if self._expected_kind == "real" else ("dry",))
            or (
                "virtual" in description
                and description["virtual"] is not (self._expected_kind == "dry-run")
            )
            or description.get("embodiment") != "yam_i2rt"
            or description.get("frame_id") != limits.frame_id
            or not isinstance(capabilities, list)
            or not all(isinstance(item, str) for item in capabilities)
            or not required <= set(capabilities)
            or any(
                description.get(flag) is not True
                for flag in (
                    "supports_request_ids",
                    "supports_deadlines",
                    "supports_speed_limits",
                    "supports_invocation_lease",
                )
            )
        ):
            raise Rejected(
                "YAM bridge must match expected_kind/mode and frame, with request IDs, deadlines, speed limits, stop and an invocation lease"
            )
        self._remote = self._label_result(description)
        return description

    def _label_result(self, result: dict) -> dict:
        """Preserve explicit server mode tags; never relabel dry as physical."""
        if result.get("kind", self._expected_kind) != self._expected_kind:
            raise EvidenceMissing("YAM response kind changed from the explicitly selected backend")
        virtual = self._expected_kind == "dry-run"
        if "virtual" in result and result["virtual"] is not virtual:
            raise EvidenceMissing("YAM response virtual flag contradicts its selected backend kind")
        return {**result, "kind": self._expected_kind, "virtual": virtual}

    def describe(self) -> dict:
        description = super().describe()
        description.update(
            backend="yam-tcp",
            protocol=PROTOCOL,
            protocol_version=1,
            kind=self._expected_kind,
            virtual=self._expected_kind == "dry-run",
            mode=self._remote["mode"],
            connection="YAM workcell wire / explicit hub_* extension",
            invocation_lease_required=True,
            server_description=dict(self._remote),
        )
        return description

    def begin_invocation(self, *, timeout_s: float) -> dict:
        """Claim one exclusive whole-skill lease, without commanding motion.

        Every invocation gets a new owner nonce. A lost claim/release response
        retains its nonce and blocks new invocations until ownership is resolved;
        it is never silently replaced with another owner or an automatic takeover.
        """
        timeout = self._timeout(timeout_s)
        with self._lease_lock:
            if self._owner_id is not None:
                raise Rejected(
                    "YAM invocation already owns a lease or its previous lease outcome is unresolved"
                )
            owner = str(uuid.uuid4())
            self._owner_id = owner
            self._lease_uncertain = True
        try:
            reply, arrays = self._client.call(
                "hub_claim", timeout_s=timeout, fields={"owner_id": owner}
            )
        except (OSError, ConnectionError, TimeoutError, RemoteError, WireError) as exc:
            raise ExecutionFailed(
                "YAM lease claim outcome is uncertain; nonce retained and no motion sent"
            ) from exc
        if (
            not arrays
            and reply.get("owner_id") == owner
            and reply.get("ok") is False
            and reply.get("status") == "busy"
        ):
            with self._lease_lock:
                self._owner_id = None
                self._lease_uncertain = False
            raise Rejected("YAM invocation lease is held by another owner")
        if (
            arrays
            or reply.get("owner_id") != owner
            or reply.get("ok") is not True
            or reply.get("status") not in {"claimed", "already_owned"}
        ):
            raise ExecutionFailed(
                "YAM lease claim was not confirmed; nonce retained and no motion sent"
            )
        result = self._label_result(reply)
        with self._lease_lock:
            self._lease_claimed = True
            self._lease_uncertain = False
            self._uncertain_action = False
            self._unsafe_stop = False
            self._last_command_id = None
        return result

    def end_invocation(self) -> dict:
        """Release only a known, quiescent lease; failures retain its owner nonce."""
        with self._lease_lock:
            if self._owner_id is None or not self._lease_claimed:
                raise Rejected("no confirmed YAM invocation lease is available to release")
            if self._uncertain_action or self._unsafe_stop:
                raise ExecutionFailed(
                    "cannot release YAM lease while an action or its stop remains unconfirmed"
                )
            if self._lease_uncertain:
                raise ExecutionFailed(
                    "previous YAM lease release outcome is unresolved; operator recovery is required"
                )
            owner = self._owner_id
            self._lease_uncertain = True
        try:
            reply, arrays = self._client.call(
                "hub_release", timeout_s=self._stop_timeout, fields={"owner_id": owner}
            )
        except (OSError, ConnectionError, TimeoutError, RemoteError, WireError) as exc:
            raise ExecutionFailed(
                "YAM lease release outcome is uncertain; owner nonce retained"
            ) from exc
        if (
            arrays
            or reply.get("owner_id") != owner
            or reply.get("ok") is not True
            or reply.get("status") != "released"
        ):
            raise ExecutionFailed("YAM lease release was not confirmed; owner nonce retained")
        result = self._label_result(reply)
        with self._lease_lock:
            self._owner_id = None
            self._lease_claimed = False
            self._lease_uncertain = False
            self._last_command_id = None
        return result

    def _require_lease(self) -> str:
        with self._lease_lock:
            if self._owner_id is None or not self._lease_claimed or self._lease_uncertain:
                raise Rejected(
                    "begin_invocation must confirm the whole-skill YAM lease before actuation"
                )
            return self._owner_id

    def _read_rpc(self, *, timeout_s: float) -> dict:
        state, arrays = self._client.call("hub_state", timeout_s=timeout_s)
        if arrays:
            raise EvidenceMissing("hub_state must contain normalized header-only state")
        if state.get("mode", self._remote["mode"]) not in (
            ("confirm", "auto") if self._expected_kind == "real" else ("dry",)
        ):
            raise EvidenceMissing(
                "YAM state mode differs from the explicitly selected backend kind"
            )
        return self._label_result(state)

    def _action(self, op: str, *, timeout_s: float, fields: dict) -> dict:
        owner = self._require_lease()
        if self._uncertain_action or self._unsafe_stop:
            raise Rejected(
                "previous YAM action must be stopped and resolved before another command"
            )
        deadline = time.monotonic() + timeout_s
        self._preflight(self.limits, min(timeout_s, self._client.timeout_s))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ExecutionFailed("YAM preflight consumed the action deadline; no action sent")
        command_id = str(uuid.uuid4())
        self._last_command_id = command_id
        self._uncertain_action = True
        try:
            reply, arrays = self._client.call(
                op,
                timeout_s=remaining,
                fields={
                    **fields,
                    "command_id": command_id,
                    "owner_id": owner,
                    "timeout_s": remaining,
                },
            )
            if (
                arrays
                or reply.get("command_id") != command_id
                or reply.get("owner_id") != owner
                or type(reply.get("ok")) is not bool
            ):
                raise WireError(
                    "action acknowledgement is missing its owner/command ID or boolean ok"
                )
            # Keep ID until the next command so Runtime.stop() can cancel/hold
            # even if later measured verification fails after this ACK.
            result = self._label_result(reply)
            self._uncertain_action = reply["ok"] is not True
            return result
        except (
            OSError,
            ConnectionError,
            TimeoutError,
            RemoteError,
            WireError,
            EvidenceMissing,
        ) as exc:
            cancellation = ""
            try:
                self._cancel(command_id)
            except Exception as cancel_error:
                cancellation = f"; cancellation could not be confirmed: {cancel_error}"
            raise ExecutionFailed(
                f"{op} outcome uncertain; action was not retried: {exc}{cancellation}"
            ) from exc

    def _move_rpc(self, pose: dict, *, speed_m_s: float, timeout_s: float, linear: bool) -> dict:
        return self._action(
            "hub_move",
            timeout_s=timeout_s,
            fields={
                "frame_id": pose["frame_id"],
                "position_m": pose["position_m"],
                "quat_xyzw": pose["quaternion_xyzw"],
                "speed_m_s": speed_m_s,
                "linear": linear,
            },
        )

    def _grip_rpc(self, width_m: float, *, timeout_s: float) -> dict:
        return self._action("hub_grip", timeout_s=timeout_s, fields={"width_m": width_m})

    def _cancel(self, command_id: str) -> None:
        owner = self._require_lease()
        self._uncertain_action = True
        try:
            self._cancel_rpc(command_id, owner)
        except BaseException:
            # Even a lost response may hide a failed physical hold. Retain the
            # lease and require operator recovery instead of admitting a new owner.
            self._unsafe_stop = True
            raise

    def _cancel_rpc(self, command_id: str, owner: str) -> None:
        # A separate connection can bypass a blocked motion/confirmation worker.
        reply, arrays = self._client.call(
            "hub_stop",
            timeout_s=self._stop_timeout,
            fields={"command_id": command_id, "owner_id": owner},
        )
        if (
            arrays
            or reply.get("ok") is not True
            or reply.get("command_id") != command_id
            or reply.get("owner_id") != owner
        ):
            raise ExecutionFailed("hub_stop did not confirm this owner and command ID")
        self._label_result(reply)
        held = (
            reply.get("status") == "held"
            and reply.get("hold_confirmed") is True
            and reply.get("other_command_active") is False
            and (self._expected_kind == "dry-run" or reply.get("powered_hold") is True)
        )
        never_committed = (
            reply.get("status") == "cancelled_before_commit"
            and reply.get("cancel_confirmed") is True
            and reply.get("command_committed") is False
            and reply.get("other_command_active") is False
        )
        if not held and not never_committed:
            self._uncertain_action = True
            raise ExecutionFailed(
                "hub_stop neither confirmed hold nor proved this command was cancelled before commit"
            )
        self._uncertain_action = False
        if self._last_command_id == command_id:
            self._last_command_id = None

    def _stop_rpc(self) -> None:
        self._require_lease()
        if self._last_command_id is not None:
            self._cancel(self._last_command_id)


def create_backend(
    *, limits: Limits, port: int, host: str = "127.0.0.1", expected_kind: str = "real", **kwargs
) -> YamTcpBackend:
    """Read-only connection factory; virtual station use requires explicit dry-run opt-in."""
    return YamTcpBackend(limits=limits, host=host, port=port, expected_kind=expected_kind, **kwargs)


class CameraClient:
    """Read the existing Orbbec ``frame`` operation, preserving its host timestamp."""

    def __init__(
        self,
        *,
        port: int = 9102,
        host: str = "127.0.0.1",
        timeout_s: float = 5,
        max_age_s: float = 1,
    ):
        self.client = Client(host, port, timeout_s=timeout_s)
        self.max_age_s = number(max_age_s, "max_age_s", 0.01, 5)

    def read_frame(self, *, timeout_s: float | None = None) -> dict:
        import numpy as np

        header, payload = self.client.call("frame", timeout_s=timeout_s)
        arrays = decode(payload)
        try:
            timestamp = number(header.get("time"), "camera time", 1e-6, 1e12)
        except Rejected as exc:
            raise EvidenceMissing(str(exc)) from exc
        age = time.time() - timestamp
        if age > self.max_age_s or age < -0.1:
            raise EvidenceMissing("camera frame is stale or its host clock is not synchronized")
        if not isinstance(header.get("camera"), str) or not header["camera"]:
            raise EvidenceMissing("camera reply must identify its stream")
        rgb, depth, intrinsic = (arrays.get(k) for k in ("rgb", "depth_m", "K"))
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[-1] != 3
        ):
            raise EvidenceMissing("camera rgb must be uint8 [H,W,3]")
        if (
            not isinstance(depth, np.ndarray)
            or depth.shape != rgb.shape[:2]
            or depth.dtype.kind != "f"
            or not np.isfinite(depth).all()
            or (depth < 0).any()
        ):
            raise EvidenceMissing("camera depth_m must be aligned finite nonnegative metre depths")
        if (
            not isinstance(intrinsic, np.ndarray)
            or intrinsic.shape != (3, 3)
            or not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0
            or intrinsic[1, 1] <= 0
            or not np.allclose(intrinsic[2], [0, 0, 1])
        ):
            raise EvidenceMissing("camera K must be a valid 3x3 pinhole intrinsics matrix")
        return {
            "rgb": rgb,
            "depth_m": depth,
            "K": intrinsic,
            "timestamp_s": timestamp,
            "camera": header["camera"],
            "source": "orbbec_frame_rpc",
            "timestamp_semantics": "camera host frame receipt Unix seconds",
            "depth_invalid_value_m": 0.0,
        }


def read_frame(
    *, host: str = "127.0.0.1", port: int = 9102, timeout_s: float = 5, max_age_s: float = 1
) -> dict:
    return CameraClient(host=host, port=port, timeout_s=timeout_s, max_age_s=max_age_s).read_frame()


def capture_clip(
    camera: CameraClient,
    *,
    frame_count: int = 8,
    interval_s: float = 0.1,
    timeout_s: float = 10,
    output_path: str | Path | None = None,
    max_bytes: int = 128 << 20,
) -> dict:
    """Capture a finite RGB clip with strictly increasing original timestamps.

    Writes an uncompressed .npy plus .metadata.json when output_path is supplied;
    otherwise returns the NumPy frames in the result. Duplicate cached camera
    frames are skipped inside the same finite deadline, never counted twice.
    No object identity or world-coordinate observation is synthesized.
    """
    import numpy as np

    if type(frame_count) is not int or not 2 <= frame_count <= 256:
        raise Rejected("frame_count must be an integer in [2,256]")
    interval = number(interval_s, "interval_s", 0, 5)
    budget = number(timeout_s, "timeout_s", 0.05, 120)
    if type(max_bytes) is not int or not 0 < max_bytes <= 512 << 20:
        raise Rejected("max_bytes must be a positive integer no larger than 512 MiB")
    path = Path(output_path).expanduser() if output_path is not None else None
    if path is not None and path.suffix != ".npy":
        raise Rejected("output_path must end in .npy for the tracking API")
    deadline, next_sample = time.monotonic() + budget, time.monotonic()
    frames, timestamps = [], []
    camera_name, shape, total = None, None, 0
    while len(frames) < frame_count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ExecutionFailed(
                "camera clip deadline exceeded before enough distinct frames arrived"
            )
        wait = next_sample - time.monotonic()
        if wait > 0:
            time.sleep(min(wait, remaining))
            continue
        record = camera.read_frame(timeout_s=remaining)
        if time.monotonic() >= deadline:
            raise ExecutionFailed("camera clip deadline exceeded")
        stamp, rgb = record["timestamp_s"], record["rgb"]
        if timestamps and stamp < timestamps[-1]:
            raise EvidenceMissing("camera timestamps moved backwards")
        if timestamps and stamp == timestamps[-1]:
            next_sample = time.monotonic() + max(0.01, interval)
            continue
        if shape is not None and (rgb.shape != shape or record["camera"] != camera_name):
            raise EvidenceMissing("camera stream or image shape changed during capture")
        total += rgb.nbytes
        if total > max_bytes:
            raise Rejected("captured clip exceeds the configured memory budget")
        frames.append(rgb.copy())
        timestamps.append(stamp)
        shape, camera_name = rgb.shape, record["camera"]
        next_sample = time.monotonic() + interval
    stacked = np.stack(frames)
    metadata = {
        "camera": camera_name,
        "timestamps_s": timestamps,
        "frame_count": len(frames),
        "shape": list(stacked.shape),
        "dtype": "uint8",
        "color_order": "RGB",
        "timestamp_semantics": "camera host frame receipt Unix seconds",
    }
    if path is not None:
        np.save(path, stacked, allow_pickle=False)
        metadata_path = path.with_suffix(".metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return {
            **metadata,
            "frames_path": str(path.resolve()),
            "metadata_path": str(metadata_path.resolve()),
        }
    return {**metadata, "frames": stacked}
