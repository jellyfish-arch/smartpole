"""Shared code for the "Test an image" demo mode.

Used by the dashboard's "Test an image" tab and by `python -m tools.test_image`,
so both behave exactly the same way.

A manual test goes through the SAME path as a camera photo:
    prepare_jpeg()  ->  POST /upload (server)  ->  worker  ->  database
The only differences are the headers: pole ID "MANUAL" and X-Source: manual,
so test images never mix with real camera data.
"""

import io
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from config import settings
from server import db

CLASSES = ["D00", "D10", "D20", "D40"]
CLASS_NAMES = {"D00": "Longitudinal crack", "D10": "Transverse crack",
               "D20": "Alligator crack", "D40": "Pothole"}
YOLO_DIR = settings.DATASET_DIR / "yolo"
GT_COLOR = (0, 200, 0)   # BGR green for ground-truth boxes

# Why a train or val image is not a fair test of the model.
SPLIT_WARNING = {
    "train": "this image is in the TRAINING split: the model learned from it, so a "
             "good result here does not show it works on new images.",
    "val": "this image is in the VALIDATION split: it was used to choose the best "
           "epoch, so it is not a fully independent test.",
    "test": "this image is in the TEST split: the model never saw it during training, "
            "so this is a fair test.",
}


class DemoError(Exception):
    """A problem the user can fix (server or worker not running, bad file...)."""


# ---------------------------------------------------------------- preparing

def prepare_jpeg(data: bytes) -> tuple[bytes, int, int]:
    """Turn any JPEG or PNG (e.g. a WhatsApp photo) into an upload-ready JPEG.

    - Applies the phone's rotation flag (EXIF), otherwise portrait photos
      can arrive sideways.
    - Shrinks the long side to MANUAL_MAX_SIDE. The model looks at 640 px
      anyway, and a 4000 px phone photo would exceed the server's size limit.
    - Re-encodes as a plain JPEG, the format the ESP32 sends.
    """
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        raise DemoError(f"Not a readable image ({e.__class__.__name__}). Use a JPEG or PNG.")
    if max(img.size) > settings.MANUAL_MAX_SIDE:
        img.thumbnail((settings.MANUAL_MAX_SIDE, settings.MANUAL_MAX_SIDE), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=92)
    return out.getvalue(), img.width, img.height


# ---------------------------------------------------------------- server and worker

def worker_running() -> bool:
    """True if a worker has reported in within the last WORKER_TIMEOUT_S seconds."""
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=settings.WORKER_TIMEOUT_S)).isoformat()
    with db.connection() as conn:
        return conn.execute("SELECT 1 FROM workers WHERE last_seen >= ? LIMIT 1", (cutoff,)).fetchone() is not None


def upload(jpeg: bytes, original_name: str) -> int:
    """Send the JPEG to the server's /upload, exactly like the ESP32. Returns the image id."""
    req = urllib.request.Request(
        f"{settings.LOCAL_SERVER_URL}/upload", data=jpeg, method="POST",
        headers={"Content-Type": "image/jpeg", "X-Pole-ID": settings.MANUAL_POLE_ID,
                 "X-Seq": "0", "X-Source": "manual", "X-Original-Name": original_name})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())["image_id"]
    except urllib.error.HTTPError as e:
        raise DemoError(f"The server rejected the image: HTTP {e.code} {e.read().decode()}")
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        raise DemoError("The server is not running. Start it with:  python -m server.app")


def wait_for_result(image_id: int, timeout_s: float = settings.MANUAL_WAIT_S) -> dict:
    """Wait until the worker has processed the image; return the image row."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with db.connection() as conn:
            row = dict(conn.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone())
        if row["status"] in ("done", "error"):
            return row
        if not worker_running():
            raise DemoError(
                f"The worker is not running, so image #{image_id} is saved but waiting in the "
                "queue. Start the worker with:  python -m worker.worker   "
                "(it will process the image as soon as it starts).")
        time.sleep(0.3)
    raise DemoError(f"No result after {timeout_s:.0f} s. The worker may be busy with a long "
                    f"queue; image #{image_id} will still be processed.")


def detections(image_id: int) -> list[dict]:
    with db.connection() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT class_name, confidence, x1, y1, x2, y2 FROM detections "
            "WHERE image_id=? ORDER BY confidence DESC", (image_id,))]


# ---------------------------------------------------------------- dataset images

def random_test_image(country: str = "India") -> Path:
    """A random image from our held-out test split (never used in training)."""
    split_file = YOLO_DIR / "splits" / f"{country}_test.txt"
    if not split_file.exists():
        raise DemoError(f"{split_file} not found. Run training.split_dataset first.")
    return Path(random.choice(split_file.read_text().split()))


def dataset_split(name: str) -> str | None:
    """Which split ('train', 'val' or 'test') a dataset image belongs to, if any."""
    stem = Path(name).stem
    for split_file in (YOLO_DIR / "splits").glob("*_*.txt"):
        if any(Path(line).stem == stem for line in split_file.read_text().split()):
            return split_file.stem.rsplit("_", 1)[1]
    return None


def ground_truth(name: str) -> list[dict] | None:
    """Ground-truth boxes for a dataset image, as fractions of the image size.

    Returns None when the image is not from the dataset (e.g. a WhatsApp
    photo): there is no correct answer to compare against. Returns [] when
    it is a dataset image with no damage in it.
    """
    if not name:
        return None
    matches = list((YOLO_DIR / "labels").glob(f"*/{Path(name).stem}.txt"))
    if not matches:
        return None
    boxes = []
    for line in matches[0].read_text().split("\n"):
        if line.strip():
            c, cx, cy, w, h = line.split()
            boxes.append({"class_name": CLASSES[int(c)], "cx": float(cx), "cy": float(cy),
                          "w": float(w), "h": float(h)})
    return boxes


def gt_to_pixels(gt: list[dict], width: int, height: int) -> list[dict]:
    return [{"class_name": b["class_name"],
             "x1": (b["cx"] - b["w"] / 2) * width, "y1": (b["cy"] - b["h"] / 2) * height,
             "x2": (b["cx"] + b["w"] / 2) * width, "y2": (b["cy"] + b["h"] / 2) * height}
            for b in gt]


def draw_ground_truth(image_path: Path, gt_px: list[dict]) -> np.ndarray:
    """The image with ground-truth boxes drawn in green. Returns RGB for display."""
    img = cv2.imread(str(image_path))
    thickness = max(2, round(max(img.shape[:2]) / 400))
    for b in gt_px:
        p1, p2 = (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"]))
        cv2.rectangle(img, p1, p2, GT_COLOR, thickness)
        cv2.putText(img, f"GT {b['class_name']}", (p1[0], max(18, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * thickness / 2, GT_COLOR, thickness)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def iou(a: dict, b: dict) -> float:
    """Intersection over Union: overlap area / combined area (1 = identical boxes)."""
    ix = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    union = ((a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
             + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter)
    return inter / union if union > 0 else 0.0


def compare(preds: list[dict], gt_px: list[dict], iou_threshold: float = 0.5) -> dict:
    """Match predictions to ground truth: same class and IoU >= 0.5 (the mAP50 rule).

    Highest-confidence predictions are matched first; each real box can be
    matched only once.
    """
    unmatched_gt = list(range(len(gt_px)))
    hits = 0
    for p in sorted(preds, key=lambda d: -d["confidence"]):
        best, best_iou = None, iou_threshold
        for i in unmatched_gt:
            if gt_px[i]["class_name"] == p["class_name"]:
                score = iou(p, gt_px[i])
                if score >= best_iou:
                    best, best_iou = i, score
        if best is not None:
            unmatched_gt.remove(best)
            hits += 1
    return {"found": hits, "real": len(gt_px), "missed": len(gt_px) - hits,
            "false_alarms": len(preds) - hits}


# ---------------------------------------------------------------- whole test

def run_test(data: bytes, original_name: str, timeout_s: float = settings.MANUAL_WAIT_S) -> dict:
    """Prepare, upload, wait for the worker, and collect everything to show."""
    start = time.monotonic()
    jpeg, width, height = prepare_jpeg(data)
    image_id = upload(jpeg, original_name)
    row = wait_for_result(image_id, timeout_s)
    return {"image_id": image_id, "row": row, "total_s": time.monotonic() - start}


def describe(image_id: int) -> dict:
    """Everything the dashboard or CLI needs to show one finished manual test."""
    with db.connection() as conn:
        row = dict(conn.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone())
    preds = detections(image_id)
    gt = ground_truth(row["original_name"])
    result = {"row": row, "preds": preds, "gt": None, "comparison": None,
              "split": dataset_split(row["original_name"]) if gt is not None else None}
    if gt is not None:
        result["gt"] = gt_to_pixels(gt, row["width"], row["height"])
        result["comparison"] = compare(preds, result["gt"])
    return result
