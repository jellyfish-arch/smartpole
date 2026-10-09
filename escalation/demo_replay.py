"""Escalation demo with chosen images: each image becomes one DEMO pole.

Run from the repository root, with the server and worker running (the
dashboard's "Test an image" tab can also start it with one click):
    python -m escalation.demo_replay --random Japan 3        # 3 random Japan test images
    python -m escalation.demo_replay --files demo_images\\a.jpg demo_images\\b.jpg
    python -m escalation.demo_replay --image-ids 41 42        # images already tested
    python -m escalation.demo_replay --stop                   # stop a running demo
    python -m escalation.demo_replay --clear                  # delete all demo data

What happens
  Each image is treated as the fixed view of one DEMO pole. Once per demo
  hour (1 real second = 1 hour, so 1 day = 24 s) the image is sent again as
  that pole's "camera photo", with optional realistic disturbances:
    * a passing vehicle covering part of the view in an occasional frame
    * small lighting changes (and darker frames at dawn/dusk)
    * night: photos between 20:00 and 06:00 are almost black (unlit camera)
    * small camera jitter (a few pixels)
  Every frame goes through the REAL server and the REAL YOLO worker, and the
  detections go into the REAL escalation engine with the real thresholds.
  Only the clock is faster (the pole's speed is 3600: one second = one hour).

  Growth is NOT faked: the same photo is replayed, so the damage cannot grow
  and growth comes out ~0 %. Real growth needs real photos of the same spot
  taken days apart.

  Size is reported as "unknown": a replayed photo has no known scale, so the
  damage escalates on proven persistence alone (Repair requested after 3
  demo days). URGENT needs a real size (calibrated pole) or real growth.

DEMO poles are kind='demo' (shown as DEMO everywhere), are ignored by the
camera escalation service, and their frames are stored as manual uploads
under the pole's own ID. --clear (or the button) removes all of it.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from config import settings
from escalation import engine, rules as R, store
from server import db
from tools import manual_test as mt

SECONDS_PER_HOUR = 1.0               # real seconds per demo hour
SPEED = R.HOUR / SECONDS_PER_HOUR    # 3600: the demo poles' clock speed
MAX_POLES = 6
LOG_FILE = settings.DATA_DIR / "demo_replay.log"


# ---------------------------------------------------------------- status

def get_status() -> dict:
    store.init()
    with db.connection() as conn:
        raw = store.get_state(conn, "demo_status")
    return json.loads(raw) if raw else {"state": "idle"}


def _set_status(**fields) -> None:
    with db.connection() as conn:
        store.set_state(conn, "demo_status", json.dumps(fields))


def request_stop() -> None:
    with db.connection() as conn:
        store.set_state(conn, "demo_stop", "1")


def _stop_requested() -> bool:
    with db.connection() as conn:
        return store.get_state(conn, "demo_stop") == "1"


def clear_demo_data() -> int:
    """Stop a running demo, then delete every demo pole, damage and frame."""
    if get_status().get("state") == "running":
        request_stop()
        for _ in range(50):                       # wait up to ~5 s for it to stop
            if get_status().get("state") != "running":
                break
            time.sleep(0.1)
    with db.connection() as conn:
        files = store.clear_demo(conn)
        store.set_state(conn, "demo_status", json.dumps({"state": "idle"}))
    for f in files:
        Path(f).unlink(missing_ok=True)
    return len(files)


def start_detached(args: list[str]) -> None:
    """Start the replay as its own background process (used by the dashboard)."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        subprocess.Popen([sys.executable, "-u", "-m", "escalation.demo_replay", *args],
                         cwd=settings.PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                         creationflags=flags)


# ---------------------------------------------------------------- disturbances

def make_frame(base, rng, hour: int, opts) -> tuple[np.ndarray, list[str]]:
    """One 'camera photo' of the pole's view at this hour of the day."""
    img, notes = base.astype(np.float32), []
    h, w = img.shape[:2]
    night = hour < 6 or hour >= 20
    if opts.night and night:
        # An unlit camera at night: almost black, with sensor noise.
        img = img * 0.03 + rng.normal(0, 6, img.shape)
        notes.append("night")
    elif opts.lighting:
        dim = 0.7 if hour < 8 or hour >= 18 else 1.0                    # dawn / dusk
        img = img * dim * rng.uniform(0.88, 1.12)
    if opts.jitter:
        dx, dy = rng.integers(-4, 5, size=2)
        img = cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                             borderMode=cv2.BORDER_REPLICATE)
    if opts.vehicle and not night and rng.random() < 1 / 12:
        vw, vh = int(w * rng.uniform(0.3, 0.45)), int(h * rng.uniform(0.25, 0.4))
        x, y = int(rng.uniform(0, w - vw)), int(rng.uniform(h * 0.45, h - vh))
        color = rng.uniform(30, 200, 3)
        cv2.rectangle(img, (x, y), (x + vw, y + vh), color.tolist(), -1)              # body
        cv2.rectangle(img, (x + vw // 6, y + vh // 8), (x + vw * 5 // 6, y + vh // 2),
                      (color * 0.4 + 60).tolist(), -1)                                # windows
        notes.append("passing vehicle")
    return np.clip(img, 0, 255).astype(np.uint8), notes


# ---------------------------------------------------------------- the replay

def load_sources(args) -> list[tuple[bytes, str]]:
    if args.random:
        country, n = args.random[0], int(args.random[1])
        return [(p.read_bytes(), p.name) for p in mt.random_test_images(country, n)]
    if args.files:
        return [(Path(f).read_bytes(), Path(f).name) for f in args.files]
    out = []
    with db.connection() as conn:
        for i in args.image_ids:
            r = conn.execute("SELECT path, original_name FROM images WHERE id=?", (i,)).fetchone()
            if r:
                out.append(((settings.PROJECT_ROOT / r["path"]).read_bytes(),
                            r["original_name"] or f"image_{i}.jpg"))
    return out


def run(sources, opts) -> None:
    store.init()
    sources = sources[:MAX_POLES]
    if not sources:
        raise SystemExit("No images to replay.")
    clear_demo_data()                               # one demo at a time
    with db.connection() as conn:
        store.set_state(conn, "demo_stop", "0")

    rng = np.random.default_rng(7)
    poles = []
    with db.connection() as conn:
        for i, (data, name) in enumerate(sources, 1):
            pole_id = f"DEMO-{i:02d}"
            jpeg = mt.prepare_jpeg(data)[0]
            base = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            store.ensure_pole(conn, pole_id, {
                "name": f"DEMO pole {i}: replay of {name}",
                "lat": 16.4958 + 0.0006 * i, "lon": 80.4985 + 0.0005 * (i % 3),
                # No approx_view_m: a photo from the internet or a dashcam has no
                # known scale (it shows many metres of road in perspective), so the
                # demo pole's size is honestly "unknown".
                "address": f"DEMO - not a real location (replayed photo {name})"},
                kind="demo", speed=SPEED)
            poles.append({"pole": pole_id, "name": name, "base": base})

    rounds = opts.days * 24
    names = {p["pole"]: p["name"] for p in poles}
    _set_status(state="running", round=0, rounds=rounds, poles=names, pid=os.getpid(),
                message="starting")
    print(f"DEMO: {len(poles)} pole(s), {opts.days} demo days, 1 s = 1 h. "
          f"Disturbances: vehicle={opts.vehicle} lighting={opts.lighting} "
          f"jitter={opts.jitter} night={opts.night}")

    seen_event = 0
    start = time.monotonic()
    try:
        for r in range(rounds):
            if _stop_requested():
                _set_status(state="stopped", round=r, rounds=rounds, poles=names,
                            message="stopped by user")
                print("DEMO: stopped")
                return
            hour = (6 + r) % 24
            uploads = []
            for p in poles:
                frame, notes = make_frame(p["base"], rng, hour, opts)
                jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
                image_id = mt.upload(jpeg, f"{p['pole']}_hour{r:03d}.jpg", pole_id=p["pole"])
                uploads.append((p["pole"], image_id, notes))
            # Wait for the worker BEFORE opening our write transaction: the worker
            # must be able to write its results while we wait.
            rows = {image_id: mt.wait_for_result(image_id) for _, image_id, _ in uploads}
            with db.connection() as conn:
                at = engine.utc_now()
                for pole_id, image_id, notes in uploads:
                    row = rows[image_id]
                    dets = [{"class_name": d["class_name"], "confidence": d["confidence"],
                             "x1": d["x1"] / row["width"], "y1": d["y1"] / row["height"],
                             "x2": d["x2"] / row["width"], "y2": d["y2"] / row["height"]}
                            for d in conn.execute("SELECT * FROM detections WHERE image_id=?", (image_id,))]
                    engine.observe(conn, pole_id, at, dets, image_id=image_id)
                    engine.evaluate(conn, pole_id, at, SPEED)
                    if notes:
                        print(f"  day {r // 24 + 1} {hour:02d}:00 {pole_id}: {', '.join(notes)} "
                              f"-> {len(dets)} detection(s)")
                for e in conn.execute(
                        """SELECT e.*, t.pole_id FROM track_events e JOIN tracks t ON t.id = e.track_id
                           WHERE e.id > ? AND t.pole_id LIKE 'DEMO-%' ORDER BY e.id""", (seen_event,)):
                    seen_event = e["id"]
                    if e["to_level"] != R.WATCHING:
                        print(f"day {r // 24 + 1} {hour:02d}:00 {e['pole_id']} damage #{e['track_id']}: "
                              f"{e['from_level']} -> {e['to_level']}. {e['reason']}")
            _set_status(state="running", round=r + 1, rounds=rounds, poles=names, pid=os.getpid(),
                        message=f"demo day {r // 24 + 1}, {hour:02d}:00")
            time.sleep(max(0.0, start + (r + 1) * SECONDS_PER_HOUR - time.monotonic()))
    except mt.DemoError as e:
        _set_status(state="error", round=r, rounds=rounds, poles=names, message=str(e))
        print(f"DEMO stopped: {e}")
        return
    _set_status(state="finished", round=rounds, rounds=rounds, poles=names,
                message=f"finished {opts.days} demo days")
    print("DEMO: finished")


def main():
    parser = argparse.ArgumentParser(description="Replay chosen images as DEMO poles.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--random", nargs=2, metavar=("COUNTRY", "N"))
    src.add_argument("--files", nargs="+")
    src.add_argument("--image-ids", nargs="+", type=int)
    src.add_argument("--stop", action="store_true", help="stop a running demo")
    src.add_argument("--clear", action="store_true", help="delete all demo poles and frames")
    parser.add_argument("--days", type=int, default=4)
    for flag in ("vehicle", "lighting", "jitter", "night"):
        parser.add_argument(f"--no-{flag}", dest=flag, action="store_false")
    args = parser.parse_args()

    if args.stop:
        request_stop()
        print("Stop requested.")
    elif args.clear:
        print(f"Demo data cleared ({clear_demo_data()} files removed).")
    else:
        run(load_sources(args), args)


if __name__ == "__main__":
    main()
