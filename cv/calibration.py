"""Real-world size of a detection: calibrated, estimated, or unknown.

calibrated  The pole has data/calibration/<pole>.json, written by the Phase 6
            calibration tool: a homography H that maps image pixels to
            ground centimetres (the fixed camera looks at a flat road, so one
            3x3 matrix converts any point of the image to the road plane).
estimate    No calibration, but config/poles.json gives approx_view_m: the
            rough width and height of road the camera sees. Size is the box's
            share of the frame times that area. Ignores perspective, so it is
            always labelled "uncalibrated estimate".
unknown     Neither. Real size is not reported (pixel area is never presented
            as real-world area); growth still works because it is a ratio.

Boxes are given as fractions of the image (0..1), so the camera resolution
does not matter.

Calibration file format:
    {"image_size": [width, height], "H": [[...], [...], [...]], "note": "..."}
"""

import json
from functools import lru_cache

import cv2
import numpy as np

from config import settings

CALIBRATION_DIR = settings.DATA_DIR / "calibration"


@lru_cache(maxsize=64)
def _load(pole_id: str, mtime: float):
    path = CALIBRATION_DIR / f"{pole_id}.json"
    data = json.loads(path.read_text())
    return np.array(data["H"], dtype=np.float64), tuple(data["image_size"])


def load(pole_id: str):
    """(H, (width, height)) for a calibrated pole, else None."""
    path = CALIBRATION_DIR / f"{pole_id}.json"
    if not path.exists():
        return None
    return _load(pole_id, path.stat().st_mtime)   # mtime: reload if re-calibrated


def _polygon_area(points) -> float:
    """Shoelace formula for the area of a polygon."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def measure(pole_id: str, box, approx_view_m=None) -> dict:
    """Size of one box (x1, y1, x2, y2 as fractions of the image).

    Returns area_cm2, length_cm (longest side, used for cracks), basis, and the
    box's own share of the frame (rel_area, rel_len) for growth ratios.
    """
    x1, y1, x2, y2 = box
    out = {"rel_area": (x2 - x1) * (y2 - y1), "rel_len": float(np.hypot(x2 - x1, y2 - y1)),
           "area_cm2": None, "length_cm": None, "basis": "unknown"}

    calib = load(pole_id)
    if calib is not None:
        H, (w, h) = calib
        corners = np.array([[[x1 * w, y1 * h]], [[x2 * w, y1 * h]],
                            [[x2 * w, y2 * h]], [[x1 * w, y2 * h]]], dtype=np.float64)
        ground = cv2.perspectiveTransform(corners, H).reshape(4, 2)
        edges = np.linalg.norm(ground - np.roll(ground, -1, axis=0), axis=1)
        out.update(area_cm2=_polygon_area(ground), length_cm=float(edges.max()), basis="calibrated")
    elif approx_view_m:
        w_cm = (x2 - x1) * approx_view_m[0] * 100
        h_cm = (y2 - y1) * approx_view_m[1] * 100
        out.update(area_cm2=w_cm * h_cm, length_cm=max(w_cm, h_cm), basis="estimate")
    return out


def position_text(pole_id: str, box) -> str:
    """Where the damage is inside the pole's view, for the repair crew."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    calib = load(pole_id)
    if calib is not None:
        H, (w, h) = calib
        gx, gy = cv2.perspectiveTransform(np.array([[[cx * w, cy * h]]]), H)[0, 0]
        return f"{gx:.0f} cm across, {gy:.0f} cm along the road from the calibration origin"
    horiz = "left" if cx < 0.4 else "right" if cx > 0.6 else "centre"
    vert = "far" if cy < 0.4 else "near" if cy > 0.6 else "middle"
    return f"{vert} {horiz} of the camera view ({cx:.0%} across, {cy:.0%} down)"
