# SPDX-License-Identifier: Apache-2.0
"""Bounded implementation of YAM workcell's JSON + NumPy TCP framing protocol.

Independently implemented for RoboSkillHub, compatible with the Apache-2.0
``yam_gap/real/wire.py`` protocol inspected on 2026-11-06. Each request uses a
new connection. No operation is retried; an interrupted action may have run.
"""

from __future__ import annotations

import io
import json
import math
import socket
import struct
import time
from typing import Any

MAX_HEADER = 1 << 20
MAX_ARRAY_BYTES = 64 << 20
MAX_TOTAL_BYTES = 128 << 20
MAX_IMAGE_PIXELS = 4096 * 4096
_LENGTH = struct.Struct("!I")


class WireError(ValueError):
    """Malformed, unsupported, or oversized peer message."""


class RemoteError(RuntimeError):
    """The peer explicitly reported an operation error."""


def _timeout(sock: socket.socket, deadline: float | None) -> None:
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("wire request deadline exceeded")
        sock.settimeout(remaining)


def _read_exact(sock: socket.socket, count: int, deadline: float | None) -> bytes:
    data = bytearray()
    while len(data) < count:
        _timeout(sock, deadline)
        chunk = sock.recv(min(count - len(data), 1 << 20))
        if not chunk:
            raise ConnectionError("peer disconnected before a complete response")
        data.extend(chunk)
    return bytes(data)


def _array_specs(specs: Any) -> list[tuple[str, Any, tuple[int, ...], int]]:
    import numpy as np

    if not isinstance(specs, list) or len(specs) > 16:
        raise WireError("arrays must be a list of at most 16 descriptors")
    names, result, total = set(), [], 0
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) != {"name", "dtype", "shape"}:
            raise WireError("array descriptors require exactly name, dtype and shape")
        name, dtype_name, shape = spec["name"], spec["dtype"], spec["shape"]
        if not isinstance(name, str) or not name or len(name) > 128 or name in names:
            raise WireError("array names must be nonempty, short and unique")
        if not isinstance(dtype_name, str) or len(dtype_name) > 32:
            raise WireError("invalid array dtype")
        try:
            dtype = np.dtype(dtype_name)
        except (TypeError, ValueError) as exc:
            raise WireError("invalid array dtype") from exc
        if (
            dtype.fields
            or dtype.hasobject
            or dtype.kind not in "buif"
            or dtype.itemsize not in (1, 2, 4, 8)
        ):
            raise WireError("only plain boolean/integer/float array dtypes are supported")
        if (
            not isinstance(shape, list)
            or len(shape) > 8
            or any(type(dim) is not int or not 0 <= dim <= 1_048_576 for dim in shape)
        ):
            raise WireError("array shape must contain at most eight bounded nonnegative dimensions")
        size = math.prod(shape) * dtype.itemsize
        total += size
        if size > MAX_ARRAY_BYTES or total > MAX_TOTAL_BYTES:
            raise WireError("array payload exceeds the message byte limit")
        result.append((name, dtype, tuple(shape), size))
        names.add(name)
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WireError("duplicate JSON object key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise WireError(f"nonfinite JSON constant {value}")


def recv(sock: socket.socket, *, deadline: float | None = None) -> tuple[dict, dict]:
    import numpy as np

    (length,) = _LENGTH.unpack(_read_exact(sock, _LENGTH.size, deadline))
    if not 0 < length <= MAX_HEADER:
        raise WireError("JSON header exceeds the allowed byte limit")
    try:
        header = json.loads(
            _read_exact(sock, length, deadline),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise WireError("invalid JSON header") from exc
    if not isinstance(header, dict):
        raise WireError("JSON header must be an object")
    specs = _array_specs(header.pop("arrays", []))
    arrays = {}
    for name, dtype, shape, size in specs:
        arrays[name] = np.frombuffer(_read_exact(sock, size, deadline), dtype=dtype).reshape(shape)
    return header, arrays


def send(
    sock: socket.socket, header: dict, arrays: dict | None = None, *, deadline: float | None = None
) -> None:
    import numpy as np

    if not isinstance(header, dict) or "arrays" in header:
        raise WireError("header must be an object without a caller-supplied arrays field")
    if arrays is not None and not isinstance(arrays, dict):
        raise WireError("arrays must be a name-to-array mapping")
    payload = {name: np.ascontiguousarray(array) for name, array in (arrays or {}).items()}
    specs = [
        {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)}
        for name, array in payload.items()
    ]
    _array_specs(specs)
    try:
        encoded = json.dumps({**header, "arrays": specs}, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise WireError("header must contain JSON-compatible finite values") from exc
    if not 0 < len(encoded) <= MAX_HEADER:
        raise WireError("JSON header exceeds the allowed byte limit")
    _timeout(sock, deadline)
    sock.sendall(_LENGTH.pack(len(encoded)) + encoded)
    for array in payload.values():
        _timeout(sock, deadline)
        sock.sendall(memoryview(array).cast("B"))


class Client:
    """One TCP connection per call, with an absolute timeout and no retries."""

    def __init__(self, host: str, port: int, timeout_s: float = 10.0):
        if not isinstance(host, str) or not host.strip():
            raise ValueError("host must be nonempty")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("port must be an integer in [1,65535]")
        self.host, self.port = host, port
        self.timeout_s = self._validate_timeout(timeout_s)

    @staticmethod
    def _validate_timeout(value: Any) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= 300
        ):
            raise ValueError("timeout_s must be a finite number in (0,300]")
        return float(value)

    def call(
        self,
        op: str,
        *,
        timeout_s: float | None = None,
        fields: dict | None = None,
        arrays: dict | None = None,
    ) -> tuple[dict, dict]:
        if not isinstance(op, str) or not op or (fields and ("op" in fields or "arrays" in fields)):
            raise ValueError("op must be nonempty and fields cannot override op/arrays")
        budget = self._validate_timeout(self.timeout_s if timeout_s is None else timeout_s)
        deadline = time.monotonic() + budget
        with socket.create_connection((self.host, self.port), timeout=budget) as sock:
            send(sock, {"op": op, **(fields or {})}, arrays, deadline=deadline)
            header, payload = recv(sock, deadline=deadline)
        if "error" in header:
            raise RemoteError(f"{op}: {header['error']}")
        return header, payload


def decode(arrays: dict) -> dict:
    """Decode existing Orbbec JPEG/16-bit-PNG fields without changing raw K."""
    import numpy as np

    result = dict(arrays)
    if "rgb_jpeg" not in result and "depth_mm_png" not in result:
        return result
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Compressed camera frames require Pillow.") from exc
    for encoded_name, decoded_name in (("rgb_jpeg", "rgb"), ("depth_mm_png", "depth_m")):
        if encoded_name not in result:
            continue
        if decoded_name in result:
            raise WireError("camera reply contains both compressed and raw copies")
        buffer = np.asarray(result.pop(encoded_name))
        if buffer.dtype != np.uint8 or buffer.ndim != 1:
            raise WireError("compressed image must be a one-dimensional uint8 buffer")
        with Image.open(io.BytesIO(buffer.tobytes())) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise WireError("decoded image exceeds the pixel limit")
            if encoded_name == "rgb_jpeg":
                if image.format != "JPEG":
                    raise WireError("rgb_jpeg is not a JPEG image")
                result[decoded_name] = np.asarray(image.convert("RGB")).copy()
            else:
                if image.format != "PNG" or image.mode not in {"I;16", "I;16B", "I;16L", "I"}:
                    raise WireError("depth_mm_png must be a 16-bit/integer PNG")
                millimetres = np.asarray(image)
                if millimetres.ndim != 2 or millimetres.min() < 0 or millimetres.max() > 65535:
                    raise WireError("depth PNG is outside uint16 millimetre range")
                result[decoded_name] = millimetres.astype(np.float32) / 1000.0
    return result
