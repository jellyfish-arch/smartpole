"""The escalation engine: tracking, measurement and level changes.

observe()   one photo's detections at one pole -> update or create damages
evaluate()  apply the time rules (confirm, expire, request, urgent)
process_camera_images()  feed new camera photos from the worker into observe()
close()     the authority marks a damage repaired or dismissed

The rules and thresholds live in escalation/rules.py; the reasons in
docs/ESCALATION.md. `speed` multiplies elapsed time for demos (1 = real time).
"""

import json
import statistics
from collections import Counter
from datetime import datetime, timezone

from config import settings
from cv import calibration, tracking
from escalation import rules as R
from escalation import store


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse(text: str) -> datetime:
    return datetime.fromisoformat(text)


def eff(start: datetime, end: datetime, speed: float) -> float:
    """Effective seconds from start to end: real seconds times the demo speed."""
    return (end - start).total_seconds() * speed


def _speed_note(speed: float) -> str:
    return f" [demo speed x{speed:g}]" if speed != 1 else ""


def _view(pole: dict):
    return (pole["view_w_m"], pole["view_h_m"]) if pole.get("view_w_m") else None


def _box(row) -> tuple:
    return (row["x1"], row["y1"], row["x2"], row["y2"])


# ======================================================================
# Observing: which damage does each detection belong to?
# ======================================================================

def observe(conn, pole_id: str, at: datetime, detections: list[dict], image_id=None) -> None:
    """Record one photo's detections (boxes as fractions of the image)."""
    pole = store.ensure_pole(conn, pole_id)
    conn.execute("UPDATE poles SET last_frame_at=? WHERE pole_id=? "
                 "AND (last_frame_at IS NULL OR last_frame_at < ?)",
                 (at.isoformat(), pole_id, at.isoformat()))

    # Open damages, plus dismissed ones that still "absorb" detections at their spot.
    candidates = [dict(r) for r in conn.execute(
        f"""SELECT * FROM tracks WHERE pole_id=? AND
            (level IN ({','.join('?' * len(R.OPEN_LEVELS))}) OR (level=? AND suppress=1))""",
        (pole_id, *R.OPEN_LEVELS, R.DISMISSED))]
    det_boxes = [(d["x1"], d["y1"], d["x2"], d["y2"]) for d in detections]
    matches = tracking.match(det_boxes, [_box(t) for t in candidates], R.MATCH_IOU)

    for di, det in enumerate(detections):
        box = det_boxes[di]
        if di in matches:
            track = candidates[matches[di]]
        else:
            track = _new_track(conn, pole_id, at, det)
        size = calibration.measure(pole_id, box, _view(pole))
        conn.execute(
            """INSERT INTO track_observations (track_id, image_id, observed_at, class_name,
               confidence, x1, y1, x2, y2, area_cm2, length_cm, rel_area, rel_len)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (track["id"], image_id, at.isoformat(), det["class_name"], det["confidence"], *box,
             size["area_cm2"], size["length_cm"], size["rel_area"], size["rel_len"]))
        _refresh_track(conn, track["id"], at, image_id)


def _new_track(conn, pole_id, at, det) -> dict:
    box = (det["x1"], det["y1"], det["x2"], det["y2"])
    # Was a damage at this exact spot repaired before? Then it has come back.
    repaired = [dict(r) for r in conn.execute(
        "SELECT * FROM tracks WHERE pole_id=? AND level=?", (pole_id, R.REPAIRED))]
    again = next((r["id"] for r in repaired if tracking.iou(box, _box(r)) >= R.REAPPEAR_IOU), None)
    cur = conn.execute(
        """INSERT INTO tracks (pole_id, level, class_name, first_seen, last_seen, x1, y1, x2, y2,
           reappeared_after, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (pole_id, R.WATCHING, det["class_name"], at.isoformat(), at.isoformat(), *box,
         again, at.isoformat()))
    reason = f"First seen: {det['class_name']} (confidence {det['confidence']:.2f})"
    if again:
        reason += f". Same spot as damage #{again}, which was marked repaired: it has reappeared"
    _event(conn, cur.lastrowid, at, None, R.WATCHING, reason)
    return dict(conn.execute("SELECT * FROM tracks WHERE id=?", (cur.lastrowid,)).fetchone())


def _refresh_track(conn, track_id, at, image_id) -> None:
    """Update sightings, last seen, reference box (median of recent) and class (majority)."""
    recent = conn.execute(
        "SELECT * FROM track_observations WHERE track_id=? ORDER BY observed_at DESC LIMIT 20",
        (track_id,)).fetchall()
    ref = [statistics.median(o[k] for o in recent[:5]) for k in ("x1", "y1", "x2", "y2")]
    cls = Counter(o["class_name"] for o in recent).most_common(1)[0][0]
    conn.execute(
        """UPDATE tracks SET sightings = sightings + 1,
               last_seen = CASE WHEN last_seen < ? THEN ? ELSE last_seen END,
               last_image_id = COALESCE(?, last_image_id),
               x1=?, y1=?, x2=?, y2=?, class_name=? WHERE id=?""",
        (at.isoformat(), at.isoformat(), image_id, *ref, cls, track_id))


# ======================================================================
# Evaluating: apply the time rules
# ======================================================================

def evaluate(conn, pole_id: str, now: datetime, speed: float = 1.0) -> None:
    pole = store.ensure_pole(conn, pole_id)
    tracks = [dict(r) for r in conn.execute(
        f"SELECT * FROM tracks WHERE pole_id=? AND level IN ({','.join('?' * len(R.OPEN_LEVELS))})",
        (pole_id, *R.OPEN_LEVELS))]
    for t in tracks:
        if t["level"] == R.WATCHING:
            t = _evaluate_watching(conn, t, now, speed)
        if t and t["level"] != R.WATCHING and t["level"] in R.OPEN_LEVELS:
            _evaluate_confirmed(conn, pole, t, now, speed)


def _evaluate_watching(conn, t, now, speed):
    obs = conn.execute("SELECT observed_at, confidence FROM track_observations WHERE track_id=?",
                       (t["id"],)).fetchall()
    first = parse(t["first_seen"])
    hours = {int(eff(first, parse(o["observed_at"]), speed) // R.HOUR) for o in obs}
    span = eff(first, parse(t["last_seen"]), speed)
    mean_conf = statistics.mean(o["confidence"] for o in obs) if obs else 0.0

    if (len(obs) >= R.CONFIRM_MIN_SIGHTINGS and len(hours) >= R.CONFIRM_MIN_HOURS
            and span >= R.CONFIRM_MIN_SPAN and mean_conf >= R.CONFIRM_MIN_MEAN_CONF):
        _set_level(conn, t, R.MONITORING, now, speed,
                   f"Confirmed real: {len(obs)} sightings in {len(hours)} different hours "
                   f"over {span / R.HOUR:.1f} h, average confidence {mean_conf:.2f}")
        return dict(conn.execute("SELECT * FROM tracks WHERE id=?", (t["id"],)).fetchone())

    unseen = eff(parse(t["last_seen"]), now, speed)
    if unseen >= R.WATCH_EXPIRE:
        _set_level(conn, t, R.EXPIRED, now, speed,
                   f"Not seen for {unseen / R.HOUR:.0f} h while still unconfirmed "
                   f"({len(obs)} sighting(s) in {len(hours)} hour(s)): probably a passing "
                   f"vehicle, a shadow or a bad frame")
        return None
    return t


def _measure_track(pole, t, obs_rows):
    """Median size of observations, recomputed with the CURRENT calibration."""
    sizes = [calibration.measure(pole["pole_id"], _box(o), _view(pole)) for o in obs_rows]
    areas = [s["area_cm2"] for s in sizes if s["area_cm2"] is not None]
    lengths = [s["length_cm"] for s in sizes if s["length_cm"] is not None]
    basis = sizes[0]["basis"] if sizes else "unknown"
    return (statistics.median(areas) if areas else None,
            statistics.median(lengths) if lengths else None, basis)


def _growth_value(pole, t, o):
    """The number whose change we track: real size if known, else share of the frame."""
    s = calibration.measure(pole["pole_id"], _box(o), _view(pole))
    crack = t["class_name"] in R.CRACK_CLASSES
    if s["basis"] != "unknown":
        return s["length_cm"] if crack else s["area_cm2"]
    return s["rel_len"] if crack else s["rel_area"]


def _median_and_error(values):
    """Median and its standard error, from the spread of the values.

    The spread (median absolute deviation) measures the frame-to-frame jitter;
    more sightings make the median more precise (divide by sqrt(n)).
    """
    med = statistics.median(values)
    spread = 1.4826 * statistics.median(abs(v - med) for v in values)
    return med, 1.253 * spread / len(values) ** 0.5


def _growth(conn, pole, t, speed):
    """(growth, uncertainty) as fractions per week, or (None, None).

    Growth = median size in the first 24 h of monitoring vs the last 24 h,
    scaled to one week. The uncertainty is GROWTH_CONFIDENCE standard errors.
    """
    window = R.GROWTH_WINDOW / speed                     # real seconds
    confirmed, last = parse(t["confirmed_at"]), parse(t["last_seen"])
    obs = [dict(o) for o in conn.execute(
        "SELECT * FROM track_observations WHERE track_id=? AND observed_at >= ? ORDER BY observed_at",
        (t["id"], t["confirmed_at"]))]
    first_w = [o for o in obs if (parse(o["observed_at"]) - confirmed).total_seconds() <= window]
    last_w = [o for o in obs if (last - parse(o["observed_at"])).total_seconds() <= window]
    if len(first_w) < R.GROWTH_MIN_SAMPLES or len(last_w) < R.GROWTH_MIN_SAMPLES:
        return None, None
    mid = lambda w: parse(w[len(w) // 2]["observed_at"])
    gap = eff(mid(first_w), mid(last_w), speed)
    if gap < R.GROWTH_MIN_GAP:
        return None, None
    before, se_before = _median_and_error([_growth_value(pole, t, o) for o in first_w])
    after, se_after = _median_and_error([_growth_value(pole, t, o) for o in last_w])
    if before <= 0:
        return None, None
    per_week = R.WEEK / gap
    growth = (after - before) / before * per_week
    error = R.GROWTH_CONFIDENCE * (se_before ** 2 + se_after ** 2) ** 0.5 / before * per_week
    return growth, error


def _evaluate_confirmed(conn, pole, t, now, speed):
    recent = conn.execute(
        "SELECT * FROM track_observations WHERE track_id=? ORDER BY observed_at DESC LIMIT ?",
        (t["id"], R.SIZE_MEDIAN_OF)).fetchall()
    area, length, basis = _measure_track(pole, t, recent)
    monitored = eff(parse(t["confirmed_at"]), now, speed)
    growth, err = _growth(conn, pole, t, speed) if monitored >= R.MONITOR_PERIOD else (None, None)
    sev = R.severity(t["class_name"], area, length, None if growth is None else growth - err)
    conn.execute("""UPDATE tracks SET area_cm2=?, length_cm=?, size_basis=?, growth_per_week=?,
                    growth_err=?, severity=?, updated_at=? WHERE id=?""",
                 (area, length, basis, growth, err, sev, now.isoformat(), t["id"]))
    t.update(area_cm2=area, length_cm=length, size_basis=basis, growth_per_week=growth,
             growth_err=err, severity=sev)

    target, reason = t["level"], None
    size_txt = size_text(t)
    if R.high_by_size(t["class_name"], area):
        target = R.URGENT
        reason = (f"URGENT: {size_txt} is above {R.HIGH_AREA_CM2} cm2 (High severity by size). "
                  "A damage this large is a safety risk, so the request is not delayed.")
    elif monitored >= R.MONITOR_PERIOD:
        days = monitored / R.DAY
        growth_txt = growth_text(t)
        if sev == "High":
            target = R.URGENT
            reason = (f"URGENT: {growth_txt}; even the low end of that range is above "
                      f"{R.HIGH_GROWTH_PER_WEEK:.0%}/week (High severity by growth) after "
                      f"{days:.1f} days of monitoring. Size {size_txt}.")
        elif sev in ("Medium", None):
            target = R.REPAIR_REQUESTED
            why = "severity Medium" if sev == "Medium" else "real size unknown (pole not calibrated)"
            reason = (f"Repair requested: {why}, {size_txt}, {growth_txt}, "
                      f"after {days:.1f} days of monitoring.")
    if R.RANK[target] > R.RANK[t["level"]]:
        _set_level(conn, t, target, now, speed, reason)


def growth_text(t) -> str:
    if t.get("growth_per_week") is None:
        return "growth not measured yet"
    return f"growth {t['growth_per_week']:+.0%} +/- {t.get('growth_err') or 0:.0%} per week"


def size_text(t) -> str:
    crack = t["class_name"] in R.CRACK_CLASSES
    value, unit = (t["length_cm"], "cm long") if crack else (t["area_cm2"], "cm2")
    if t["size_basis"] == "calibrated" and value is not None:
        return f"{value:.0f} {unit}"
    if t["size_basis"] == "estimate" and value is not None:
        return f"~{value:.0f} {unit} (uncalibrated estimate)"
    return "size unknown (uncalibrated)"


# ======================================================================
# Level changes, audit trail, outbox
# ======================================================================

def _event(conn, track_id, at, frm, to, reason, actor="system") -> None:
    conn.execute("INSERT INTO track_events (track_id, at, from_level, to_level, reason, actor) "
                 "VALUES (?, ?, ?, ?, ?, ?)", (track_id, at.isoformat(), frm, to, reason, actor))


def _set_level(conn, t, level, now, speed, reason) -> None:
    fields = {"level": level, "updated_at": now.isoformat()}
    if level == R.MONITORING:
        fields["confirmed_at"] = now.isoformat()
    if level in R.CLOSED_LEVELS:
        fields["closed_at"] = now.isoformat()
    if level in R.PRIORITY:
        fields["priority"] = R.PRIORITY[level]
        fields["requested_at"] = t.get("requested_at") or now.isoformat()
        fields["request_id"] = t.get("request_id") or f"SP-{now.year}-{t['id']:05d}"
    conn.execute(f"UPDATE tracks SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                 (*fields.values(), t["id"]))
    _event(conn, t["id"], now, t["level"], level, reason + _speed_note(speed))
    if level in R.PRIORITY:
        t = dict(conn.execute("SELECT * FROM tracks WHERE id=?", (t["id"],)).fetchone())
        _notify(conn, t, now, reason)


def _notify(conn, t, now, reason) -> None:
    """Put a repair request (or upgrade) in the outbox for the authority."""
    pole = dict(conn.execute("SELECT * FROM poles WHERE pole_id=?", (t["pole_id"],)).fetchone())
    upgrade = t["level"] == R.URGENT and t["requested_at"] != now.isoformat()
    kind = "URGENT repair" if t["level"] == R.URGENT else "Repair request"
    tag = {"simulation": "SIMULATION ", "demo": "DEMO "}.get(pole.get("kind"), "")
    subject = (f"[{tag}{kind}] {t['request_id']}: "
               f"{CLASS_NAMES.get(t['class_name'], t['class_name'])} at {pole['pole_id']}")
    maps = (f"https://www.google.com/maps?q={pole['lat']},{pole['lon']}"
            if pole["lat"] is not None else "location not registered")
    body = "\n".join([
        f"Request: {t['request_id']}   Priority: {t['priority']} ({R.LABEL[t['level']]})"
        + ("   (upgraded)" if upgrade else ""),
        f"Damage: {CLASS_NAMES.get(t['class_name'], t['class_name'])} ({t['class_name']})",
        f"Size: {size_text(t)}   Severity: {t['severity'] or 'unknown'}   {growth_text(t)}",
        f"Pole: {pole['pole_id']} - {pole['name']}",
        f"Location: {pole['address']}  ({maps})",
        f"Position in camera view: {calibration.position_text(pole['pole_id'], _box(t))}",
        f"First seen: {t['first_seen']}   Last seen: {t['last_seen']}   Sightings: {t['sightings']}",
        f"Reason: {reason}",
        f"Evidence photo: image #{t['last_image_id']}" if t["last_image_id"] else "Evidence photo: none (simulated)",
    ])
    cur = conn.execute(
        "INSERT INTO notifications (track_id, created_at, request_id, priority, subject, body) "
        "VALUES (?, ?, ?, ?, ?, ?)", (t["id"], now.isoformat(), t["request_id"], t["priority"], subject, body))
    store.OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    (store.OUTBOX_DIR / f"{cur.lastrowid:05d}_{t['request_id']}_{t['level']}.json").write_text(
        json.dumps({"subject": subject, "body": body, "created_at": now.isoformat()}, indent=2))


CLASS_NAMES = {"D00": "Longitudinal crack", "D10": "Transverse crack",
               "D20": "Alligator crack", "D40": "Pothole"}


# ======================================================================
# Authority actions
# ======================================================================

def close(conn, track_id: int, action: str, by: str, note: str = "", now: datetime | None = None) -> None:
    """Authority marks a damage REPAIRED or DISMISSED. Closes it for good."""
    if action not in (R.REPAIRED, R.DISMISSED):
        raise ValueError("action must be REPAIRED or DISMISSED")
    now = now or utc_now()
    t = conn.execute("SELECT * FROM tracks WHERE id=?", (track_id,)).fetchone()
    if t is None or t["level"] not in R.OPEN_LEVELS:
        raise ValueError(f"damage #{track_id} is not open")
    conn.execute("""UPDATE tracks SET level=?, closed_at=?, closed_by=?, close_note=?, suppress=?,
                    updated_at=? WHERE id=?""",
                 (action, now.isoformat(), by, note, int(action == R.DISMISSED), now.isoformat(), track_id))
    what = ("Marked repaired" if action == R.REPAIRED else
            "Dismissed as not road damage; future detections at this spot will be ignored")
    _event(conn, track_id, now, t["level"], action, f"{what}. Note: {note or '-'}", f"authority:{by}")


# ======================================================================
# Camera photos from the worker
# ======================================================================

def process_camera_images(conn, now: datetime, speed: float = 1.0) -> int:
    """Feed newly processed CAMERA photos into the engine, then apply the time rules.

    Only source='camera' photos are used: manual tests ("Test an image") can
    never create a damage or a request. Photos are taken strictly in order and
    the engine never skips past one the worker has not finished yet.
    """
    last = int(store.get_state(conn, "last_image_id", 0))
    rows = conn.execute(
        "SELECT id, pole_id, received_at, width, height, status FROM images "
        "WHERE id > ? AND source = 'camera' AND pole_id != ? ORDER BY id",
        (last, settings.MANUAL_POLE_ID)).fetchall()
    done = 0
    for r in rows:
        if r["status"] in ("pending", "processing"):
            break                                   # wait for the worker; keep order
        if r["status"] == "done":
            dets = [{"class_name": d["class_name"], "confidence": d["confidence"],
                     "x1": d["x1"] / r["width"], "y1": d["y1"] / r["height"],
                     "x2": d["x2"] / r["width"], "y2": d["y2"] / r["height"]}
                    for d in conn.execute("SELECT * FROM detections WHERE image_id=?", (r["id"],))]
            observe(conn, r["pole_id"], parse(r["received_at"]), dets, image_id=r["id"])
            done += 1
        last = r["id"]
    store.set_state(conn, "last_image_id", last)
    for p in conn.execute("SELECT pole_id FROM poles WHERE simulated=0").fetchall():
        evaluate(conn, p["pole_id"], now, speed)
    return done
