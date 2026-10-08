"""SmartPole dashboard.

Run from the repository root:
    streamlit run dashboard/app.py

Read-only view of the database the server and worker write to. The live
part re-runs every REFRESH_S seconds, so new uploads appear without
reloading the page.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# `streamlit run` puts only dashboard/ on the import path. Add the project
# root so `from config import settings` works like it does everywhere else.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from config import settings
from server import db

REFRESH_S = 3
CLASS_NAMES = {"D00": "Longitudinal crack", "D10": "Transverse crack",
               "D20": "Alligator crack", "D40": "Pothole"}


def parse_utc(text):
    return datetime.fromisoformat(text) if text else None


def local_time(dt):
    return dt.astimezone().strftime("%d %b %Y, %H:%M:%S") if dt else "never"


def age(dt):
    if dt is None:
        return ""
    s = int((datetime.now(timezone.utc) - dt).total_seconds())
    if s < 60:
        return f"{s} s ago"
    if s < 3600:
        return f"{s // 60} min ago"
    if s < 86400:
        return f"{s // 3600} h ago"
    return f"{s // 86400} days ago"


def query(sql, params=()):
    with db.connection() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


st.set_page_config(page_title="SmartPole", page_icon="🛣️", layout="wide")
st.title("SmartPole: road damage monitor")

db.init_db()   # creates empty tables if the server has never run yet
poles = [r["pole_id"] for r in query(
    "SELECT pole_id FROM images UNION SELECT pole_id FROM heartbeats ORDER BY pole_id")]
if not poles:
    st.info("No data yet. Start the server and the node (or `python -m server.send_test_image`).")
    st.stop()
pole = st.selectbox("Camera node", poles, index=len(poles) - 1)


@st.fragment(run_every=REFRESH_S)
def live_view():
    # ---------------- status row ----------------
    hb = query("SELECT * FROM heartbeats WHERE pole_id=? ORDER BY id DESC LIMIT 1", (pole,))
    hb = hb[0] if hb else None
    hb_time = parse_utc(hb["received_at"]) if hb else None
    online = bool(hb_time) and (datetime.now(timezone.utc) - hb_time).total_seconds() < settings.HEARTBEAT_TIMEOUT_S

    last_upload = query("SELECT received_at FROM images WHERE pole_id=? ORDER BY id DESC LIMIT 1", (pole,))
    last_upload_time = parse_utc(last_upload[0]["received_at"]) if last_upload else None

    status = {r["status"]: r["n"] for r in query(
        "SELECT status, COUNT(*) AS n FROM images WHERE pole_id=? GROUP BY status", (pole,))}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Camera", "🟢 Online" if online else "🔴 Offline",
              f"heartbeat {age(hb_time)}" if hb_time else "no heartbeat yet",
              delta_color="off")
    c2.metric("Last upload", local_time(last_upload_time).split(", ")[-1] if last_upload_time else "never",
              age(last_upload_time), delta_color="off")
    c3.metric("Images received", sum(status.values()))
    c4.metric("Waiting for worker", status.get("pending", 0) + status.get("processing", 0))
    c5.metric("Failed", status.get("error", 0))
    if hb:
        st.caption(f"Last heartbeat {local_time(hb_time)}. Node IP {hb['node_ip']}, "
                   f"signal {hb['rssi']} dBm, uptime {hb['uptime_s']} s, "
                   f"free heap {hb['free_heap']:,} B, free PSRAM {hb['free_psram']:,} B")

    # ---------------- latest image ----------------
    latest = query("SELECT * FROM images WHERE pole_id=? AND status='done' ORDER BY id DESC LIMIT 1", (pole,))
    if not latest:
        st.info("No processed images yet. Is the worker running?")
        return
    img = latest[0]
    newer = query("SELECT COUNT(*) AS n FROM images WHERE pole_id=? AND id>? AND status!='done'",
                  (pole, img["id"]))[0]["n"]

    st.subheader(f"Latest image #{img['id']} (seq {img['seq']}), "
                 f"received {local_time(parse_utc(img['received_at']))}")
    if newer:
        st.caption(f"{newer} newer image(s) still waiting for the worker.")
    left, right = st.columns(2)
    left.image(str(settings.PROJECT_ROOT / img["path"]), caption="Original", width="stretch")
    right.image(str(settings.PROJECT_ROOT / img["annotated_path"]),
                caption=f"YOLO detections ({img['inference_ms']:.0f} ms, "
                        f"model {Path(img['model_path']).parent.parent.name}/{Path(img['model_path']).name})",
                width="stretch")

    # ---------------- detections table ----------------
    dets = pd.DataFrame(query(
        "SELECT class_name, confidence, x1, y1, x2, y2 FROM detections WHERE image_id=? "
        "ORDER BY confidence DESC", (img["id"],)))
    st.subheader(f"Detections in this image: {len(dets)}")
    if dets.empty:
        st.write("No damage detected above the confidence threshold "
                 f"({settings.WORKER_CONF}).")
    else:
        dets.insert(1, "type", dets["class_name"].map(CLASS_NAMES))
        dets["box (x1, y1, x2, y2)"] = dets.apply(
            lambda r: f"({r.x1:.0f}, {r.y1:.0f}, {r.x2:.0f}, {r.y2:.0f})", axis=1)
        dets["size (px)"] = dets.apply(lambda r: f"{r.x2 - r.x1:.0f} x {r.y2 - r.y1:.0f}", axis=1)
        st.dataframe(dets[["class_name", "type", "confidence", "box (x1, y1, x2, y2)", "size (px)"]],
                     hide_index=True, width="stretch",
                     column_config={"confidence": st.column_config.ProgressColumn(
                         "confidence", min_value=0.0, max_value=1.0, format="%.2f")})
        st.caption("Sizes are in pixels. Real-world centimetres need the camera "
                   "calibration (Phase 6).")

    # ---------------- counts ----------------
    st.subheader("Counts by damage type")
    counts = pd.DataFrame(query(
        """SELECT d.class_name AS class,
                  SUM(d.image_id = ?) AS "latest image",
                  SUM(i.received_at >= ?) AS "last 24 h",
                  COUNT(*) AS "all time"
           FROM detections d JOIN images i ON i.id = d.image_id
           WHERE i.pole_id = ? GROUP BY d.class_name ORDER BY d.class_name""",
        (img["id"], (datetime.now(timezone.utc) - pd.Timedelta(hours=24)).isoformat(), pole)))
    if counts.empty:
        st.write("No detections yet.")
    else:
        counts.insert(1, "type", counts["class"].map(CLASS_NAMES))
        st.dataframe(counts, hide_index=True, width="stretch")

    # ---------------- recent images ----------------
    with st.expander("Recent images"):
        recent = pd.DataFrame(query(
            """SELECT i.id, i.seq, i.received_at, i.status, i.width || 'x' || i.height AS size,
                      i.inference_ms, COUNT(d.id) AS detections
               FROM images i LEFT JOIN detections d ON d.image_id = i.id
               WHERE i.pole_id = ? GROUP BY i.id ORDER BY i.id DESC LIMIT 20""", (pole,)))
        recent["received_at"] = recent["received_at"].map(lambda t: local_time(parse_utc(t)))
        st.dataframe(recent, hide_index=True, width="stretch")

    st.caption(f"Auto-refreshes every {REFRESH_S} s. Page updated "
               f"{datetime.now().strftime('%H:%M:%S')}.")


live_view()
