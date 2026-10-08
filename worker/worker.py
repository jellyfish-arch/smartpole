"""YOLO worker: turns 'pending' uploads into detections.

Run from the repository root, in its own terminal:
    python -m worker.worker                  # device and weights from config/settings.py
    python -m worker.worker --device 0       # use the GPU (once training is finished)

It runs as a separate process from the server so slow inference never makes
the ESP32's upload wait. It checks the database (on this laptop) for new
rows; it never contacts the ESP32.

How it makes sure an image is never processed twice
  1. Claiming is one atomic SQL statement:
         UPDATE images SET status='processing' WHERE id = <oldest pending> RETURNING ...
     A row can only move from 'pending' to 'processing' once, so even two
     workers running at the same time could never get the same image.
  2. The detections and the 'done' status are written in ONE transaction.
     Either both are saved or neither is, so an image can never end up with
     two sets of detections. The claim stores this worker's id (claimed_by),
     and 'done' is only written if the row is still claimed by us.
  3. If the worker is killed while an image is 'processing' (step 1 done,
     step 2 not), nothing about that image was saved. On the next start
     those rows go back to 'pending' and are tried again. After
     WORKER_MAX_ATTEMPTS failures an image is marked 'error' so one broken
     file cannot block the queue forever.
  Rows marked 'done' are never selected again, so restarting the worker
  does not redo finished work.
"""

import argparse
import json
import os
import shutil
import socket
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import settings
from server import db


# Unique id for this run of the worker, stored on every row it claims.
WORKER_ID = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


class ClaimLost(Exception):
    """Another worker took this row over; our results must not be saved."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelHolder:
    """Loads the YOLO weights and reloads them when the file changes.

    During training, last.pt is rewritten after every epoch. We copy it to a
    temporary file before loading, so we never read a half-written file. If
    loading fails (we caught it mid-write), the previous model is kept.
    """

    def __init__(self, weights: Path, device: str):
        self.weights = weights
        self.device = device
        self.model = None
        self.loaded_mtime = None
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="smartpole_worker_"))

    def refresh(self):
        if not self.weights.exists():
            if self.model is None:
                raise SystemExit(f"Weights not found: {self.weights}")
            return
        mtime = self.weights.stat().st_mtime
        if mtime == self.loaded_mtime:
            return
        snapshot = self.tmp_dir / f"snapshot_{int(mtime)}.pt"
        try:
            shutil.copy2(self.weights, snapshot)
            model = YOLO(str(snapshot))
        except Exception as e:
            print(f"[worker] could not load {self.weights.name} yet ({e.__class__.__name__}); keeping current model")
            snapshot.unlink(missing_ok=True)
            return
        # Delete older snapshots (21 MB each) once the new one is loaded.
        for old in self.tmp_dir.glob("snapshot_*.pt"):
            if old != snapshot:
                old.unlink(missing_ok=True)
        # Warm-up: the very first inference is slow (~4 s on the GPU) because
        # CUDA sets itself up. Do it now on a blank image, so the first real
        # photo is processed at full speed.
        try:
            model.predict(np.zeros((settings.WORKER_IMGSZ, settings.WORKER_IMGSZ, 3), np.uint8),
                          imgsz=settings.WORKER_IMGSZ, device=self.device, verbose=False)
        except Exception as e:
            print(f"[worker] warm-up failed ({e.__class__.__name__}); continuing")
        self.model, self.loaded_mtime = model, mtime
        when = datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
        print(f"[worker] loaded {self.weights} (saved {when}) on device={self.device}")


STARTED_AT = None


def beat(holder: "ModelHolder") -> None:
    """Tell the dashboard this worker is alive (one row per worker run)."""
    with db.connection() as conn:
        conn.execute(
            """INSERT INTO workers (worker_id, started_at, last_seen, device, weights)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(worker_id) DO UPDATE SET last_seen=excluded.last_seen""",
            (WORKER_ID, STARTED_AT, utc_now(), holder.device, str(holder.weights)))


def remove_beat() -> None:
    """On a clean stop, remove our row so the dashboard shows 'stopped' at once."""
    with db.connection() as conn:
        conn.execute("DELETE FROM workers WHERE worker_id=?", (WORKER_ID,))


def recover_interrupted() -> None:
    """Rows left in 'processing' by a worker that was killed go back to the queue."""
    with db.connection() as conn:
        n = conn.execute(
            "UPDATE images SET status='pending' WHERE status='processing'").rowcount
    if n:
        print(f"[worker] {n} image(s) were interrupted last time; queued again")


def claim_next():
    """Atomically take the oldest pending image. Returns the row or None."""
    with db.connection() as conn:
        return conn.execute(
            """UPDATE images
               SET status='processing', claimed_at=?, claimed_by=?, attempts=attempts+1
               WHERE id = (SELECT id FROM images WHERE status='pending' ORDER BY id LIMIT 1)
               RETURNING id, pole_id, seq, path, received_at, attempts""",
            (utc_now(), WORKER_ID),
        ).fetchone()


def process(row, holder: ModelHolder) -> None:
    image_path = settings.PROJECT_ROOT / row["path"]
    day = row["received_at"][:10]
    annotated_path = settings.ANNOTATED_DIR / day / f"{row['id']:06d}.jpg"
    result_path = settings.RESULTS_DIR / day / f"{row['id']:06d}.json"
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    result = holder.model.predict(
        source=str(image_path), conf=settings.WORKER_CONF, imgsz=settings.WORKER_IMGSZ,
        device=holder.device, verbose=False)[0]
    inference_ms = (time.perf_counter() - start) * 1000

    detections = []
    for (x1, y1, x2, y2), conf, cls in zip(result.boxes.xyxy.tolist(),
                                           result.boxes.conf.tolist(),
                                           result.boxes.cls.tolist()):
        detections.append({
            "class_id": int(cls), "class_name": result.names[int(cls)],
            "confidence": round(conf, 4),
            "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)})

    # Files first. If the worker dies right here, they are simply overwritten
    # on the retry, because their names come from the image id.
    cv2.imwrite(str(annotated_path), result.plot())
    result_path.write_text(json.dumps({
        "image_id": row["id"], "pole_id": row["pole_id"], "seq": row["seq"],
        "image": row["path"], "model": str(holder.weights), "model_saved_at":
            datetime.fromtimestamp(holder.loaded_mtime, timezone.utc).isoformat(),
        "conf_threshold": settings.WORKER_CONF, "inference_ms": round(inference_ms, 1),
        "detections": detections}, indent=2))

    rel = lambda p: p.relative_to(settings.PROJECT_ROOT).as_posix()
    # One transaction: 'done' + detections are saved together or not at all.
    # The status update only matches if this worker still holds the claim;
    # if not, raising ClaimLost rolls the whole transaction back.
    with db.connection() as conn:
        updated = conn.execute(
            """UPDATE images SET status='done', processed_at=?, annotated_path=?,
               result_path=?, model_path=?, inference_ms=?, error=NULL
               WHERE id=? AND status='processing' AND claimed_by=?""",
            (utc_now(), rel(annotated_path), rel(result_path), str(holder.weights),
             round(inference_ms, 1), row["id"], WORKER_ID)).rowcount
        if updated != 1:
            raise ClaimLost(f"image #{row['id']} is no longer claimed by this worker")
        conn.executemany(
            "INSERT INTO detections (image_id, class_id, class_name, confidence, x1, y1, x2, y2)"
            " VALUES (:image_id, :class_id, :class_name, :confidence, :x1, :y1, :x2, :y2)",
            [{"image_id": row["id"], **d} for d in detections])

    summary = ", ".join(f"{d['class_name']} {d['confidence']:.2f}" for d in detections) or "no damage"
    print(f"[worker] #{row['id']} {row['pole_id']} seq={row['seq']}: "
          f"{len(detections)} detection(s) [{summary}] in {inference_ms:.0f} ms")


def mark_failed(row, error: Exception) -> None:
    final = row["attempts"] >= settings.WORKER_MAX_ATTEMPTS
    with db.connection() as conn:
        conn.execute("UPDATE images SET status=?, error=? WHERE id=? AND claimed_by=?",
                     ("error" if final else "pending", f"{error.__class__.__name__}: {error}",
                      row["id"], WORKER_ID))
    print(f"[worker] #{row['id']} FAILED (attempt {row['attempts']}): {error}"
          + ("  -> marked error" if final else "  -> will retry"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=settings.WORKER_WEIGHTS)
    parser.add_argument("--device", default=settings.WORKER_DEVICE, help='"cpu" or "0" for the GPU')
    args = parser.parse_args()

    if args.device == "cpu":
        # Leave most CPU cores to the training dataloader.
        torch.set_num_threads(2)

    global STARTED_AT
    db.init_db()
    recover_interrupted()
    holder = ModelHolder(args.weights, args.device)
    holder.refresh()
    STARTED_AT = utc_now()
    print(f"[worker] {WORKER_ID} waiting for pending images (Ctrl+C to stop)")

    last_beat = 0.0
    try:
        while True:
            if time.monotonic() - last_beat >= settings.WORKER_HEARTBEAT_S:
                beat(holder)
                last_beat = time.monotonic()
            row = claim_next()
            if row is None:
                holder.refresh()          # pick up a newer checkpoint while idle
                time.sleep(settings.WORKER_POLL_S)
                continue
            try:
                process(row, holder)
            except ClaimLost as e:
                print(f"[worker] {e}; results discarded")
            except Exception as e:
                mark_failed(row, e)
    except KeyboardInterrupt:
        remove_beat()
        print("[worker] stopped")


if __name__ == "__main__":
    main()
