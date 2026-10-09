"""Automated checks of the escalation rules, on a throwaway database.

Run from the repository root:
    python -m escalation.check_scenarios

Each scenario feeds scripted detections with chosen timestamps into the real
engine and checks the outcome against docs/ESCALATION.md. The real database
and outbox are not touched.
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import settings

TMP = Path(tempfile.mkdtemp(prefix="smartpole_escalation_test_"))
settings.DB_PATH = TMP / "test.db"                       # redirect BEFORE using the db

from cv import calibration                               # noqa: E402
from escalation import engine, rules as R, store         # noqa: E402
from server import db                                    # noqa: E402

store.OUTBOX_DIR = TMP / "outbox"
calibration.CALIBRATION_DIR = TMP / "calibration"

T0 = datetime(2026, 10, 1, 6, 0, tzinfo=timezone.utc)
FAILURES = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILURES.append(msg)


def det(cls, cx, cy, w, h, conf=0.6):
    return {"class_name": cls, "confidence": conf,
            "x1": cx - w / 2, "y1": cy - h / 2, "x2": cx + w / 2, "y2": cy + h / 2}


def daytime_frames(days, start=T0, every_h=2):
    """Photo times 06:00-20:00 every 2 h (no frames at night)."""
    for d in range(days):
        for h in range(6, 21, every_h):
            yield start.replace(hour=0) + timedelta(days=d, hours=h)


def run(conn, pole, frames, make_dets, speed=1.0):
    for at in frames:
        engine.observe(conn, pole, at, make_dets(at))
        engine.evaluate(conn, pole, at, speed)


def track(conn, pole, level=None):
    sql, params = "SELECT * FROM tracks WHERE pole_id=?", [pole]
    if level:
        sql, params = sql + " AND level=?", params + [level]
    return [dict(r) for r in conn.execute(sql + " ORDER BY id", params)]


def setup():
    store.init()
    calibration.CALIBRATION_DIR.mkdir(parents=True)
    # Calibrated test pole: the view is 400 cm x 300 cm (simple scale homography).
    (calibration.CALIBRATION_DIR / "T-CAL.json").write_text(json.dumps(
        {"image_size": [640, 480], "H": [[400 / 640, 0, 0], [0, 300 / 480, 0], [0, 0, 1]]}))
    with db.connection() as conn:
        store.ensure_pole(conn, "T-CAL", {"name": "calibrated", "lat": 16.5, "lon": 80.5})
        store.ensure_pole(conn, "T-EST", {"name": "estimate", "lat": 16.5, "lon": 80.5,
                                          "approx_view_m": [3.0, 2.0]})
        store.ensure_pole(conn, "T-UNK", {"name": "unknown size"})


# 250 cm2 on T-CAL: 20 cm x 12.5 cm  ->  w = 20/400, h = 12.5/300
MEDIUM = (0.05, 12.5 / 300)
BIG = (0.10, 21 / 300)          # 40 cm x 21 cm = 840 cm2


def main() -> int:
    setup()
    with db.connection() as conn:
        print("1. A passing vehicle (one frame) never becomes a request")
        engine.observe(conn, "T-CAL", T0 + timedelta(hours=4), [det("D40", 0.8, 0.8, 0.1, 0.1)])
        for h in range(5, 40):
            engine.evaluate(conn, "T-CAL", T0 + timedelta(hours=h))
        t = track(conn, "T-CAL")[-1]
        check(t["level"] == R.EXPIRED, f"vehicle ends EXPIRED (got {t['level']})")
        check(t["request_id"] is None, "vehicle never got a request id")

        print("2. A shadow that moves across the hours is never confirmed")
        shadow_times = [T0 + timedelta(days=1, hours=h) for h in (9, 10, 11, 12, 13, 14)]
        for i, at in enumerate(shadow_times):
            engine.observe(conn, "T-EST", at, [det("D40", 0.2 + 0.09 * i, 0.5, 0.06, 0.06)])
            engine.evaluate(conn, "T-EST", at)
        engine.evaluate(conn, "T-EST", T0 + timedelta(days=3))
        levels = {t["level"] for t in track(conn, "T-EST")}
        check(levels == {R.EXPIRED}, f"moving shadow only produced EXPIRED tracks (got {levels})")

        print("3. A real medium pothole: confirmed, monitored 3 days, then requested")
        frames = list(daytime_frames(6, T0 + timedelta(days=2)))
        cal_before = len(track(conn, "T-CAL"))
        for at in frames:
            engine.observe(conn, "T-CAL", at, [det("D40", 0.3, 0.6, *MEDIUM)])
            engine.evaluate(conn, "T-CAL", at)
            t = track(conn, "T-CAL")[cal_before]
            if at == frames[3]:
                check(t["level"] == R.WATCHING, "still WATCHING after 4 sightings / 6 h")
            if at == frames[4]:
                check(t["level"] == R.MONITORING, f"MONITORING at 5th sighting (got {t['level']})")
            if at == frames[4] + timedelta(days=2):
                check(t["level"] == R.MONITORING, "not requested after only 2 days of monitoring")
        t = track(conn, "T-CAL")[cal_before]
        check(t["level"] == R.REPAIR_REQUESTED, f"REPAIR_REQUESTED after 3 days (got {t['level']})")
        check(t["severity"] == "Medium" and abs(t["area_cm2"] - 250) < 1,
              f"severity Medium, area 250 cm2 (got {t['severity']}, {t['area_cm2']:.0f})")
        check(t["size_basis"] == "calibrated", "size basis is calibrated")

        print("4. A big pothole goes URGENT as soon as it is confirmed")
        cal_before = len(track(conn, "T-CAL"))
        start = T0 + timedelta(days=10)
        for at in list(daytime_frames(1, start))[:5]:
            engine.observe(conn, "T-CAL", at, [det("D40", 0.7, 0.3, *BIG)])
            engine.evaluate(conn, "T-CAL", at)
        t = track(conn, "T-CAL")[cal_before]
        check(t["level"] == R.URGENT, f"URGENT on the day it is confirmed (got {t['level']})")
        check(t["priority"] == 1, "priority 1")

        print("5. Growth > 10 %/week upgrades a request to URGENT; levels never go down")
        cal_before = len(track(conn, "T-CAL"))
        start = T0 + timedelta(days=12)
        frames = list(daytime_frames(16, start))

        def growing(at):
            days = (at - start).total_seconds() / 86400
            scale = 1.0 if days < 5 else (1 + 0.40 * (days - 5) / 7) ** 0.5   # area +40 %/week after day 5
            return [det("D40", 0.5, 0.75, MEDIUM[0] * scale, MEDIUM[1] * scale)]
        levels_seen = []
        for at in frames:
            engine.observe(conn, "T-CAL", at, growing(at))
            engine.evaluate(conn, "T-CAL", at)
            levels_seen.append(track(conn, "T-CAL")[cal_before]["level"])
        t = track(conn, "T-CAL")[cal_before]
        check(R.REPAIR_REQUESTED in levels_seen, "was REPAIR_REQUESTED first")
        check(t["level"] == R.URGENT, f"upgraded to URGENT by growth (got {t['level']}, "
              f"growth {t['growth_per_week']})")
        check(t["growth_per_week"] is not None and t["growth_per_week"] > 0.10,
              f"measured growth {t['growth_per_week']:+.0%}/week")
        ranks = [R.RANK[l] for l in levels_seen]
        check(ranks == sorted(ranks), "level never went down")
        # A frame where a car hides half of it must not lower anything.
        at = frames[-1] + timedelta(hours=2)
        engine.observe(conn, "T-CAL", at, [det("D40", 0.5, 0.75, MEDIUM[0] * 0.5, MEDIUM[1] * 0.5)])
        engine.evaluate(conn, "T-CAL", at)
        check(track(conn, "T-CAL")[cal_before]["level"] == R.URGENT, "one small (occluded) frame changes nothing")

        print("6. A small stable crack stays on the watch list (Low)")
        est_before = len(track(conn, "T-EST"))
        for at in daytime_frames(6, T0 + timedelta(days=5)):
            engine.observe(conn, "T-EST", at, [det("D00", 0.4, 0.4, 20 / 300, 0.01)])   # 20 cm long
            engine.evaluate(conn, "T-EST", at)
        t = track(conn, "T-EST")[est_before]
        check(t["level"] == R.MONITORING and t["severity"] == "Low",
              f"MONITORING / Low after 6 days (got {t['level']} / {t['severity']})")
        check(t["size_basis"] == "estimate", "size labelled as uncalibrated estimate")

        print("7. Unknown size (no calibration, no approx view) still escalates on persistence")
        for at in daytime_frames(5, T0 + timedelta(days=5)):
            engine.observe(conn, "T-UNK", at, [det("D40", 0.5, 0.5, 0.1, 0.1)])
            engine.evaluate(conn, "T-UNK", at)
        t = track(conn, "T-UNK")[0]
        check(t["level"] == R.REPAIR_REQUESTED and t["severity"] is None,
              f"REPAIR_REQUESTED with unknown severity (got {t['level']} / {t['severity']})")
        check("size unknown" in engine.size_text(t), "shown as 'size unknown (uncalibrated)'")

        print("8. Dismissed (e.g. a manhole cover) stays dismissed")
        est_before = len(track(conn, "T-EST"))
        frames = list(daytime_frames(5, T0 + timedelta(days=20)))
        for at in frames:
            engine.observe(conn, "T-EST", at, [det("D40", 0.7, 0.7, 0.08, 0.1)])
            engine.evaluate(conn, "T-EST", at)
        t = track(conn, "T-EST")[est_before]
        check(t["level"] == R.REPAIR_REQUESTED, f"manhole was requested (got {t['level']})")
        notes_before = conn.execute("SELECT COUNT(*) FROM notifications").fetchone()[0]
        engine.close(conn, t["id"], R.DISMISSED, "inspector", "manhole cover", frames[-1])
        for at in daytime_frames(5, T0 + timedelta(days=26)):
            engine.observe(conn, "T-EST", at, [det("D40", 0.7, 0.7, 0.08, 0.1)])
            engine.evaluate(conn, "T-EST", at)
        check(len(track(conn, "T-EST")) == est_before + 1, "no new damage created at the dismissed spot")
        check(conn.execute("SELECT COUNT(*) FROM notifications").fetchone()[0] == notes_before,
              "no new notification after dismissal")
        check(track(conn, "T-EST")[est_before]["sightings"] > t["sightings"],
              "dismissed damage absorbed the new detections")

        print("9. Damage seen again after REPAIR is a new damage, marked 'reappeared'")
        old = track(conn, "T-UNK")[0]
        engine.close(conn, old["id"], R.REPAIRED, "inspector", "patched", T0 + timedelta(days=11))
        engine.observe(conn, "T-UNK", T0 + timedelta(days=30), [det("D40", 0.5, 0.5, 0.1, 0.1)])
        new = track(conn, "T-UNK")[-1]
        check(new["id"] != old["id"] and new["level"] == R.WATCHING, "new WATCHING damage")
        check(new["reappeared_after"] == old["id"], "linked to the repaired one")

        print("10. Demo speed: same rules, a day lasts a minute")
        start = T0 + timedelta(days=40)
        frames = [start + timedelta(seconds=15 * i) for i in range(20)]   # a photo every 15 s
        before = len(track(conn, "T-CAL"))
        confirmed_at = requested_at = None
        for at in frames:
            engine.observe(conn, "T-CAL", at, [det("D20", 0.2, 0.2, *MEDIUM)])
            engine.evaluate(conn, "T-CAL", at, speed=1440)
            lvl = track(conn, "T-CAL")[before]["level"]
            if lvl == R.MONITORING and confirmed_at is None:
                confirmed_at = at
            if lvl == R.REPAIR_REQUESTED and requested_at is None:
                requested_at = at
        check(confirmed_at is not None and (confirmed_at - start).total_seconds() <= 75,
              f"confirmed after {(confirmed_at - start).total_seconds():.0f} s real time")
        check(requested_at is not None and 170 <= (requested_at - confirmed_at).total_seconds() <= 200,
              f"requested {(requested_at - confirmed_at).total_seconds():.0f} s (= 3 'days') after confirmation")

        print("11. Camera photos only: manual tests never create damages; order is kept")
        ids = {}
        for name, source, status in (("cam1", "camera", "done"), ("man1", "manual", "done"),
                                     ("cam2", "camera", "pending"), ("cam3", "camera", "done")):
            pole = settings.MANUAL_POLE_ID if source == "manual" else "T-CAM"
            ids[name] = conn.execute(
                """INSERT INTO images (pole_id, seq, received_at, path, size_bytes, width, height,
                   status, source, original_name) VALUES (?, 1, ?, ?, 1, 640, 480, ?, ?, ?)""",
                (pole, (T0 + timedelta(days=50)).isoformat(), f"x/{name}.jpg", status, source, name)).lastrowid
            conn.execute("INSERT INTO detections (image_id, class_id, class_name, confidence, x1, y1, x2, y2)"
                         " VALUES (?, 3, 'D40', 0.7, 100, 100, 200, 200)", (ids[name],))
        engine.process_camera_images(conn, T0 + timedelta(days=50))
        check(not track(conn, settings.MANUAL_POLE_ID), "manual upload created no damage")
        check(track(conn, "T-CAM")[0]["sightings"] == 1, "only cam1 used; stopped at pending cam2")
        conn.execute("UPDATE images SET status='done' WHERE id=?", (ids["cam2"],))
        engine.process_camera_images(conn, T0 + timedelta(days=50))
        check(track(conn, "T-CAM")[0]["sightings"] == 3, "cam2 and cam3 used once the worker finished")

        print("12. Every request is in the outbox")
        n = conn.execute("SELECT COUNT(*) FROM notifications").fetchone()[0]
        files = list(store.OUTBOX_DIR.glob("*.json"))
        check(n > 0 and len(files) == n, f"{n} notifications, {len(files)} outbox files")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED")
        return 1
    print("All escalation checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
