"""SmartPole dashboard.

Run from the repository root:
    streamlit run dashboard/app.py

Two tabs:
  * Live camera   - read-only view of what the ESP32 node has sent. Re-runs
                    every REFRESH_S seconds, so new uploads appear by themselves.
  * Test an image - demo mode: upload any road photo, or pick a random image
                    from the test split. It goes through the same server and
                    worker as a camera photo, but is stored as a manual test
                    (pole "MANUAL", source 'manual') so it never mixes with
                    camera data.
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
from tools import manual_test as mt

REFRESH_S = 3
CLASS_NAMES = mt.CLASS_NAMES
WORKER_HELP = "Start it in a terminal with:  python -m worker.worker"


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


def detections_table(rows):
    """Detections as a table with a confidence bar."""
    df = pd.DataFrame(rows)
    df.insert(1, "type", df["class_name"].map(CLASS_NAMES))
    df["box (x1, y1, x2, y2)"] = df.apply(
        lambda r: f"({r.x1:.0f}, {r.y1:.0f}, {r.x2:.0f}, {r.y2:.0f})", axis=1)
    df["size (px)"] = df.apply(lambda r: f"{r.x2 - r.x1:.0f} x {r.y2 - r.y1:.0f}", axis=1)
    st.dataframe(df[["class_name", "type", "confidence", "box (x1, y1, x2, y2)", "size (px)"]],
                 hide_index=True, width="stretch",
                 column_config={"confidence": st.column_config.ProgressColumn(
                     "confidence", min_value=0.0, max_value=1.0, format="%.2f")})


st.set_page_config(page_title="SmartPole", page_icon="🛣️", layout="wide")
st.title("SmartPole: road damage monitor")
db.init_db()   # creates empty tables if the server has never run yet

live_tab, test_tab = st.tabs(["📷 Live camera", "🧪 Test an image"])


# ======================================================================
# Live camera
# ======================================================================

@st.fragment(run_every=REFRESH_S)
def live_view(pole):
    # ---------------- status row ----------------
    hb = query("SELECT * FROM heartbeats WHERE pole_id=? ORDER BY id DESC LIMIT 1", (pole,))
    hb = hb[0] if hb else None
    hb_time = parse_utc(hb["received_at"]) if hb else None
    online = bool(hb_time) and (datetime.now(timezone.utc) - hb_time).total_seconds() < settings.HEARTBEAT_TIMEOUT_S

    last_upload = query("SELECT received_at FROM images WHERE pole_id=? ORDER BY id DESC LIMIT 1", (pole,))
    last_upload_time = parse_utc(last_upload[0]["received_at"]) if last_upload else None

    status = {r["status"]: r["n"] for r in query(
        "SELECT status, COUNT(*) AS n FROM images WHERE pole_id=? GROUP BY status", (pole,))}

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Camera", "🟢 Online" if online else "🔴 Offline",
              f"heartbeat {age(hb_time)}" if hb_time else "no heartbeat yet",
              delta_color="off", delta_arrow="off")
    c2.metric("Last upload", local_time(last_upload_time).split(", ")[-1] if last_upload_time else "never",
              age(last_upload_time), delta_color="off", delta_arrow="off")
    c3.metric("Worker", "🟢 Running" if mt.worker_running() else "🔴 Stopped")
    c4.metric("Images received", sum(status.values()))
    c5.metric("Waiting for worker", status.get("pending", 0) + status.get("processing", 0))
    c6.metric("Failed", status.get("error", 0))
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
    dets = query("SELECT class_name, confidence, x1, y1, x2, y2 FROM detections WHERE image_id=? "
                 "ORDER BY confidence DESC", (img["id"],))
    st.subheader(f"Detections in this image: {len(dets)}")
    if not dets:
        st.write(f"No damage detected above the confidence threshold ({settings.WORKER_CONF}).")
    else:
        detections_table(dets)
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


with live_tab:
    # Camera nodes only (manual tests are excluded), most recently active first.
    poles = [r["pole_id"] for r in query(
        """SELECT pole_id, MAX(t) AS last_seen FROM (
               SELECT pole_id, received_at AS t FROM images WHERE source = 'camera'
               UNION ALL SELECT pole_id, received_at FROM heartbeats)
           WHERE pole_id != ? GROUP BY pole_id ORDER BY last_seen DESC""",
        (settings.MANUAL_POLE_ID,))]
    if not poles:
        st.info("No camera data yet. Start the server and the node (or `python -m server.send_test_image`).")
    else:
        live_view(st.selectbox("Camera node", poles))


# ======================================================================
# Test an image (demo mode)
# ======================================================================

def run_and_remember(data: bytes, name: str):
    """Send one image through the system and keep its id for display."""
    with st.spinner(f"Sending {name} to the server and waiting for the worker..."):
        try:
            test = mt.run_test(data, name)
            st.session_state["manual_id"] = test["image_id"]
            st.session_state["manual_total_s"] = test["total_s"]
        except mt.DemoError as e:
            st.session_state.pop("manual_id", None)
            st.error(str(e))


def show_manual_result(image_id: int):
    info = mt.describe(image_id)
    row = info["row"]
    name = row["original_name"] or f"image #{image_id}"
    st.subheader(f"Result: {name}  (image #{image_id})")

    if row["status"] == "error":
        st.error(f"The worker could not process this image: {row['error']}")
        return

    # Where the image came from, and whether it is a fair test.
    if info["gt"] is None:
        st.info("**No ground truth.** This image is not from RDD2022 (for example a "
                "WhatsApp photo), so there is no correct answer to compare with. "
                "Only the model's prediction is shown.")
    elif info["split"] == "test":
        st.success(f"RDD2022 image: {mt.SPLIT_WARNING['test']}")
    elif info["split"] in ("train", "val"):
        st.warning(f"RDD2022 image: {mt.SPLIT_WARNING[info['split']]}")

    # Numbers.
    preds = info["preds"]
    m = st.columns(5 if info["comparison"] else 3)
    m[0].metric("Damages found by model", len(preds))
    m[1].metric("Model processing time", f"{row['inference_ms']:.0f} ms")
    total = st.session_state.get("manual_total_s")
    m[2].metric("Total (upload + queue + model)", f"{total:.1f} s" if total else "-")
    if info["comparison"]:
        c = info["comparison"]
        if c["real"]:
            m[3].metric("Real damages found", f"{c['found']} of {c['real']}")
        else:
            m[3].metric("Real damages", "none (clean road)")
        m[4].metric("False alarms", c["false_alarms"])

    # Images side by side.
    original = settings.PROJECT_ROOT / row["path"]
    cols = st.columns(3 if info["gt"] is not None else 2)
    cols[0].image(str(original), caption="Original", width="stretch")
    cols[1].image(str(settings.PROJECT_ROOT / row["annotated_path"]),
                  caption="Model prediction", width="stretch")
    if info["gt"] is not None:
        cols[2].image(mt.draw_ground_truth(original, info["gt"]),
                      caption=f"Ground truth (real answer): {len(info['gt'])} box(es)",
                      width="stretch")

    # Tables.
    st.markdown(f"**Model detections** (confidence ≥ {settings.WORKER_CONF})")
    if preds:
        detections_table(preds)
    else:
        st.write("No damage detected.")
    if info["gt"] is not None:
        st.markdown("**Ground truth labels**")
        if info["gt"]:
            st.dataframe(pd.DataFrame([{"class_name": b["class_name"], "type": CLASS_NAMES[b["class_name"]]}
                                       for b in info["gt"]]), hide_index=True, width="stretch")
        else:
            st.write("The label file says this image has no damage.")
        st.caption("A real damage counts as found when the model predicts the same class "
                   "with a box overlapping it by IoU ≥ 0.5 (overlap area / combined area), "
                   "the same rule mAP50 uses.")


with test_tab:
    st.markdown("Run any road photo through SmartPole, the same way a camera photo is "
                "processed. Test images are stored as **manual tests** and never mixed "
                "with camera data.")
    if mt.worker_running():
        st.caption("🟢 Worker is running.")
    else:
        st.error(f"🔴 The worker is not running, so images cannot be processed. {WORKER_HELP}")

    up_col, rand_col = st.columns([2, 1])
    with up_col:
        uploaded = st.file_uploader("Upload a road photo (JPEG or PNG, e.g. saved from WhatsApp)",
                                    type=["jpg", "jpeg", "png"])
        if st.button("Test this image", type="primary", disabled=uploaded is None):
            run_and_remember(uploaded.getvalue(), uploaded.name)
    with rand_col:
        st.write("Or let the system choose:")
        if st.button("🎲 Pick a random test image"):
            path = mt.random_test_image()
            run_and_remember(path.read_bytes(), path.name)
        st.caption("A random image from the India **test split**, which the model "
                   "never saw during training.")

    if "manual_id" in st.session_state:
        st.divider()
        show_manual_result(st.session_state["manual_id"])
