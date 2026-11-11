"""Ground truth for tracking a pen cap among look-alike caps in real wrist-camera clips.

The caps of the marker pens in the holder (pens task, ~/yam_gap/pens/runs/*/videos) are labelled by
hand on one frame (circles read off a 50 px grid, +-5 px); the holder is rigid and the pens stay in
it during these segments, so each later frame's boxes are the labels mapped through a homography
fitted (ORB + RANSAC) on features within 40 px of the caps, against the labelled frame, the gripper
fingers (fixed in the wrist view) masked out. Frames whose fit has under MIN_INLIERS inliers are not
scored. Writes tests/bundles/data/pens_lookalike_tracks.json.

    python tests/bundles/_pens_lookalike_gt.py PENS_RUNS_DIR [--draw OUT_DIR]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / "data/pens_lookalike_tracks.json"
FIRST, LAST, STRIDE = 100, 400, 2
MIN_INLIERS = 25

CLIPS = {
    "g4c_rw001": {
        "video": "real-g4c/videos/20261109_195031/right_wrist_001.mp4",
        "caps": {
            "blue_top": [117, 0, 167, 50], "blue_mid": [75, 37, 125, 87],
            "black": [38, 73, 82, 117], "yellow": [155, 52, 205, 102],
            "orange": [105, 90, 155, 140], "magenta": [187, 104, 233, 150],
            "brown": [71, 120, 115, 164], "purple": [148, 138, 192, 182],
            "blue_low": [103, 163, 147, 207], "red": [221, 150, 265, 194],
            "green": [178, 178, 222, 222],
        },
    },
    "g4a_rw001": {
        "video": "real-g4a/videos/20261109_171630/right_wrist_001.mp4",
        "caps": {
            "blue_light": [80, 37, 150, 107], "blue": [52, 98, 112, 158],
            "black": [7, 152, 63, 208], "yellow": [150, 98, 210, 158],
            "orange": [105, 150, 165, 210], "magenta": [205, 140, 255, 190],
            "red": [258, 185, 308, 235], "purple": [173, 188, 217, 232],
            "lime": [220, 225, 270, 275], "green": [185, 260, 235, 310],
        },
    },
}  # fmt: skip


def finger_mask(h: int, w: int) -> np.ndarray:
    """Pixels away from the two gripper fingers (dark wedges fixed in the lower corners)."""
    m = np.full((h, w), 255, np.uint8)
    cv2.fillPoly(m, [np.array([[0, 300], [150, 210], [190, 480], [0, 480]])], 0)
    cv2.fillPoly(m, [np.array([[490, 200], [640, 230], [640, 480], [520, 480]])], 0)
    return m


def read(path: Path, first: int, last: int, stride: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    out = []
    for i in range(first, last + 1):
        ok, f = cap.read()
        if not ok:
            break
        if (i - first) % stride == 0:
            out.append(f)
    return out


def project(box, H) -> list[float]:
    x1, y1, x2, y2 = box
    pts = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], np.float32)
    p = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    c = p.mean(axis=0)
    w = (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2
    h = (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2
    return [round(float(c[0] - w / 2), 1), round(float(c[1] - h / 2), 1),
            round(float(c[0] + w / 2), 1), round(float(c[1] + h / 2), 1)]  # fmt: skip


def main() -> None:
    runs = Path(sys.argv[1]).expanduser()
    draw = Path(sys.argv[sys.argv.index("--draw") + 1]) if "--draw" in sys.argv else None
    orb = cv2.ORB_create(4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    res = {"_about": __doc__.split("\n\n")[0] + " " + __doc__.split("\n\n")[1].split("\n\n")[0],
           "first_frame": FIRST, "stride": STRIDE, "clips": {}}  # fmt: skip
    for name, spec in CLIPS.items():
        frames = read(runs / spec["video"], FIRST, LAST, STRIDE)
        g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        h, w = g0.shape
        fm = finger_mask(h, w)
        roi = np.zeros_like(g0)
        for b in spec["caps"].values():
            x1, y1, x2, y2 = b
            roi[max(0, y1 - 40) : y2 + 40, max(0, x1 - 40) : x2 + 40] = 255
        k0, d0 = orb.detectAndCompute(g0, roi & fm)
        boxes: dict[str, list] = {c: [] for c in spec["caps"]}
        inl = []
        for f in frames:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            k, d = orb.detectAndCompute(g, fm)
            H, n = None, 0
            if d is not None and len(k) > 10:
                ms = [m for m, m2 in (p for p in bf.knnMatch(d0, d, k=2) if len(p) == 2)
                      if m.distance < 0.75 * m2.distance]  # fmt: skip
                if len(ms) >= 10:
                    src = np.float32([k0[m.queryIdx].pt for m in ms]).reshape(-1, 1, 2)
                    dst = np.float32([k[m.trainIdx].pt for m in ms]).reshape(-1, 1, 2)
                    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
                    n = int(mask.sum()) if mask is not None else 0
            inl.append(n)
            for c, b in spec["caps"].items():
                boxes[c].append(project(b, H) if H is not None and n >= MIN_INLIERS else None)
        res["clips"][name] = {"video": spec["video"], "n_frames": len(frames), "inliers": inl,
                              "boxes": boxes}  # fmt: skip
        ok = sum(x >= MIN_INLIERS for x in inl)
        print(
            f"{name}: {len(frames)} frames, {ok} with a fit (median inliers {np.median(inl):.0f})"
        )
        if draw:
            draw.mkdir(parents=True, exist_ok=True)
            tiles = []
            for i in np.linspace(0, len(frames) - 1, 4).astype(int):
                im = frames[i].copy()
                for c, bs in boxes.items():
                    if bs[i] is not None:
                        x1, y1, x2, y2 = (int(v) for v in bs[i])
                        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv2.putText(im, c[:3], (x1, y1 + 10), 0, 0.35, (255, 255, 255), 1)
                tiles.append(cv2.resize(im, (480, 360)))
            cv2.imwrite(str(draw / f"{name}.jpg"), np.concatenate(tiles, axis=1))
    OUT.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
