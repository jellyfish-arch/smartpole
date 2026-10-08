"""Scripted demo: three weeks on two simulated poles, in under a minute.

Run from the repository root (the authority portal can be open meanwhile):
    python -m escalation.simulate              # plays at a watchable pace
    python -m escalation.simulate --fast       # no pauses

What is real and what is simulated
  REAL       the escalation engine, rules and thresholds (the same code the
             camera service uses), the database, the outbox, the portal.
  SIMULATED  the detections: instead of camera + YOLO, a script says "a
             pothole of this size is at this spot", with realistic noise
             (missed frames, box jitter, a wrong class now and then).
             Timestamps are spread over three real weeks (daylight only),
             so no speed-up is needed at all.

Simulated poles are called SIM-01 / SIM-02, are flagged as simulated in the
database, and every request they create is labelled SIMULATION. Re-running
replaces the previous simulation; real camera data is never touched.

The story
  SIM-01 (calibrated)
    A  pothole, 250 cm2, stable, then grows ~30 %/week after day 5 (rain)
       -> watched, confirmed, REPAIR REQUESTED on day 3, URGENT later by growth
    B  a passing vehicle, one frame                         -> expires
    C  a shadow sliding across the road one afternoon       -> expires
    D  big pothole (840 cm2) appearing on day 13            -> URGENT at once
  SIM-02 (not calibrated: sizes are estimates)
    E  small longitudinal crack, 20 cm, stable              -> stays MONITORING (Low)
    F  transverse crack, 55 cm                              -> REPAIR REQUESTED
    G  manhole cover the model mistakes for a pothole       -> REPAIR REQUESTED
       (dismiss it in the authority portal during the demo)
"""

import argparse
import json
import random
import time
from datetime import datetime, timedelta

from cv import calibration
from escalation import engine, rules as R, store
from server import db

DAYS = 21
POLES = {
    "SIM-01": {"name": "SIMULATED pole 1 (calibrated)", "lat": 16.4952, "lon": 80.5012,
               "address": "SIMULATION - example position near VIT-AP"},
    "SIM-02": {"name": "SIMULATED pole 2 (not calibrated)", "lat": 16.4981, "lon": 80.4978,
               "address": "SIMULATION - example position near VIT-AP", "approx_view_m": [3.0, 2.0]},
}
# SIM-01's simulated calibration: the camera view covers 400 cm x 300 cm of road.
SIM01_CALIBRATION = {"image_size": [640, 480], "H": [[400 / 640, 0, 0], [0, 300 / 480, 0], [0, 0, 1]],
                     "note": "SIMULATED calibration for the demo pole SIM-01"}


def objects_at(day: float, at: datetime):
    """Scripted damages visible at this moment: (pole, class, cx, cy, w, h, detect_prob, label)."""
    out = []
    # A: 250 cm2 = 20 cm x 12.5 cm on SIM-01; grows ~30 %/week (area) after day 5.
    k = 1.0 if day < 5 else (1 + 0.30 * (day - 5) / 7) ** 0.5
    out.append(("SIM-01", "D40", 0.30, 0.62, 0.05 * k, 12.5 / 300 * k, 0.85, "A"))
    if int(day) == 1 and at.hour == 10:                        # B: passing vehicle, one frame
        out.append(("SIM-01", "D40", 0.65, 0.45, 0.08, 0.08, 1.0, "B"))
    if int(day) == 2 and 14 <= at.hour <= 18:                  # C: shadow slides along
        out.append(("SIM-01", "D40", 0.20 + 0.06 * (at.hour - 14), 0.40, 0.07, 0.06, 1.0, "C"))
    if day >= 13:                                              # D: big pothole, 840 cm2
        out.append(("SIM-01", "D40", 0.72, 0.30, 0.10, 21 / 300, 0.85, "D"))
    # SIM-02, view 300 cm x 200 cm
    out.append(("SIM-02", "D00", 0.25, 0.35, 20 / 300, 0.015, 0.80, "E"))   # 20 cm crack
    if day >= 2:
        out.append(("SIM-02", "D10", 0.55, 0.70, 55 / 300, 0.02, 0.80, "F"))  # 55 cm crack
    out.append(("SIM-02", "D40", 0.80, 0.30, 0.07, 0.10, 0.90, "G"))         # manhole cover
    return out


def noisy(rng, cls, cx, cy, w, h):
    """What YOLO would report: jittered box, varying confidence, sometimes a wrong class."""
    w, h = w * rng.uniform(0.97, 1.03), h * rng.uniform(0.97, 1.03)
    cx, cy = cx + rng.uniform(-0.004, 0.004), cy + rng.uniform(-0.004, 0.004)
    if cls == "D40" and rng.random() < 0.1:
        cls = "D20"
    conf = min(0.95, max(0.30, rng.gauss(0.6, 0.08)))
    return {"class_name": cls, "confidence": conf,
            "x1": cx - w / 2, "y1": cy - h / 2, "x2": cx + w / 2, "y2": cy + h / 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="no pauses between steps")
    parser.add_argument("--delay", type=float, default=0.08, help="seconds per simulated photo")
    args = parser.parse_args()

    store.init()
    calibration.CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    (calibration.CALIBRATION_DIR / "SIM-01.json").write_text(json.dumps(SIM01_CALIBRATION, indent=2))
    with db.connection() as conn:
        store.clear(conn, simulated=True)
        for pole_id, info in POLES.items():
            store.ensure_pole(conn, pole_id, info, kind="simulation")
            conn.execute("UPDATE poles SET last_frame_at=NULL WHERE pole_id=?", (pole_id,))

    rng = random.Random(42)                       # same story every run
    today = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
    start = today - timedelta(days=DAYS)          # ends yesterday evening
    seen = 0
    print(f"Simulating {DAYS} days on SIM-01 and SIM-02 (a photo every hour, 06:00-20:00)...")
    for d in range(DAYS):
        for hour in range(6, 21):
            at = start + timedelta(days=d, hours=hour)
            dets = {"SIM-01": [], "SIM-02": []}
            for pole, cls, cx, cy, w, h, p, _ in objects_at(d + hour / 24, at):
                if rng.random() < p:
                    dets[pole].append(noisy(rng, cls, cx, cy, w, h))
            with db.connection() as conn:
                for pole in dets:
                    engine.observe(conn, pole, at, dets[pole])
                    engine.evaluate(conn, pole, at)
                for e in conn.execute(
                        "SELECT e.*, t.pole_id, t.class_name FROM track_events e JOIN tracks t "
                        "ON t.id = e.track_id WHERE e.id > ? AND t.pole_id LIKE 'SIM-%' ORDER BY e.id",
                        (seen,)):
                    seen = e["id"]
                    if e["from_level"] is None and e["to_level"] == R.WATCHING:
                        continue                                    # too chatty
                    print(f"day {d + 1:2d} {at:%H:%M}  {e['pole_id']} damage #{e['track_id']} "
                          f"({e['class_name']}): {e['from_level']} -> {e['to_level']}. {e['reason']}")
            if not args.fast:
                time.sleep(args.delay)

    with db.connection() as conn:
        print("\nFinal state:")
        for t in conn.execute("SELECT * FROM tracks WHERE pole_id LIKE 'SIM-%' ORDER BY pole_id, id"):
            print(f"  {t['pole_id']} #{t['id']:<4} {t['class_name']}  {R.LABEL[t['level']]:<17} "
                  f"severity {t['severity'] or '-':<6} {engine.size_text(dict(t))}")
    print("\nOpen the authority portal to see the requests (streamlit run dashboard/authority.py).")


if __name__ == "__main__":
    main()
