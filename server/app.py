"""SmartPole ingest server.

Run from the repository root:
    python -m server.app

Endpoints
    POST /upload     raw JPEG bytes in the body (not a multipart form),
                     headers X-Pole-ID and X-Seq
    POST /heartbeat  small JSON status message from the node
    GET  /           quick "is the server up" check

The upload handler only checks, saves and records the image, then replies.
YOLO runs in a separate worker process: inference takes far longer than
the ESP32 is willing to wait for an HTTP reply.
"""

import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from config import settings
from server import db, discovery


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts, and the part after `yield` on shutdown.
    db.init_db()
    settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    stop_discovery = discovery.start_in_background()
    yield
    stop_discovery.set()


app = FastAPI(title="SmartPole ingest server", lifespan=lifespan)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def clean_pole_id(raw: str | None) -> str:
    """Pole IDs end up in file names, so allow only letters, digits, - and _."""
    pole_id = (raw or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", pole_id):
        raise HTTPException(400, "X-Pole-ID header missing or invalid")
    return pole_id


@app.get("/")
def root():
    return {"service": "smartpole", "status": "running", "time": utc_now().isoformat()}


@app.post("/upload")
async def upload(request: Request):
    pole_id = clean_pole_id(request.headers.get("X-Pole-ID"))
    try:
        seq = int(request.headers.get("X-Seq", ""))
    except ValueError:
        seq = None   # still accept the image; seq is only for diagnostics

    body = await request.body()

    # ---- Check that the bytes really are a complete JPEG ----
    if len(body) == 0:
        raise HTTPException(400, "empty body")
    if len(body) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"image larger than {settings.MAX_UPLOAD_BYTES} bytes")
    # Every JPEG starts with FF D8 and ends with FF D9. A missing end marker
    # usually means the upload was cut off by a Wi-Fi drop.
    if body[:2] != b"\xff\xd8" or body.rstrip(b"\x00")[-2:] != b"\xff\xd9":
        raise HTTPException(400, "not a complete JPEG")
    image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "JPEG could not be decoded")
    height, width = image.shape[:2]

    # ---- Save with a unique name: pole, UTC time, seq, random suffix ----
    received = utc_now()
    day_dir = settings.UPLOADS_DIR / received.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    filename = (f"{pole_id}_{received.strftime('%Y%m%dT%H%M%S_%fZ')}"
                f"_seq{seq if seq is not None else 'NA'}_{uuid.uuid4().hex[:6]}.jpg")
    file_path = day_dir / filename
    file_path.write_bytes(body)

    # Store the path relative to the project root so the database stays
    # valid if the project folder is moved.
    rel_path = file_path.relative_to(settings.PROJECT_ROOT).as_posix()
    image_id = db.insert_image(pole_id, seq, received.isoformat(), rel_path,
                               len(body), width, height)

    print(f"[upload] #{image_id} {pole_id} seq={seq} {width}x{height} {len(body)} B -> {rel_path}")
    return {"status": "ok", "image_id": image_id, "width": width, "height": height}


@app.post("/heartbeat")
async def heartbeat(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(400, "heartbeat body must be JSON")
    pole_id = clean_pole_id(data.get("pole_id") or request.headers.get("X-Pole-ID"))

    db.insert_heartbeat(
        pole_id=pole_id,
        received_at=utc_now().isoformat(),
        node_ip=data.get("ip") or (request.client.host if request.client else None),
        uptime_s=data.get("uptime_s"),
        rssi=data.get("rssi"),
        free_heap=data.get("free_heap"),
        free_psram=data.get("free_psram"),
        seq=data.get("seq"),
    )
    return {"status": "ok"}


if __name__ == "__main__":
    # host 0.0.0.0 = reachable from other devices on the hotspot, not just this laptop.
    uvicorn.run(app, host=settings.SERVER_HOST, port=settings.SERVER_PORT)
