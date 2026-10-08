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
import time
from datetime import datetime, timezone
from pathlib import Path

# `streamlit run` puts only dashboard/ on the import path. Add the project
# root so `from config import settings` works like it does everywhere else.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from config import settings
from server import db
from escalation import demo_replay
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

live_tab, test_tab = st.tabs(["📷 Live camera", "🧪 Test images"])


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
# Test images (demo mode): batch testing + escalation demo
# ======================================================================

def run_batch_and_remember(items: list[tuple[bytes, str]]):
    """Send several images through the system and keep their ids for display."""
    with st.spinner(f"Sending {len(items)} image(s) to the server and waiting for the worker..."):
        try:
            start = time.monotonic()
            ids = mt.run_batch(items)
            st.session_state["batch_ids"] = ids
            st.session_state["batch_total_s"] = time.monotonic() - start
        except mt.DemoError as e:
            st.error(str(e))


def generalisation_note(countries: set):
    others = sorted(c for c in countries if c not in (mt.TRAINED_ON, "not RDD2022"))
    if others:
        st.info(f"**The model was trained on {mt.TRAINED_ON} roads only.** Images from "
                f"{', '.join(others)} test how well it **generalises** to roads, cameras and "
                "road markings it has never seen, so lower accuracy there is expected. "
                "Training on all countries is planned for Phase 7.")
    if "not RDD2022" in countries:
        st.caption("Uploaded photos that are not from RDD2022 have no ground truth: only the "
                   "model's prediction is shown for them.")


def show_manual_result(image_id: int, total_s: float | None = None):
    info = mt.describe(image_id)
    row = info["row"]
    name = row["original_name"] or f"image #{image_id}"
    country = mt.country_of(row["original_name"])
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
        st.success(f"RDD2022 image from {country}: {mt.SPLIT_WARNING['test']}")
    elif info["split"] in ("train", "val"):
        st.warning(f"RDD2022 image from {country}: {mt.SPLIT_WARNING[info['split']]}")

    # Numbers.
    preds = info["preds"]
    m = st.columns(5 if info["comparison"] else 3)
    m[0].metric("Damages found by model", len(preds))
    m[1].metric("Model processing time", f"{row['inference_ms']:.0f} ms")
    m[2].metric("Total (upload + queue + model)", f"{total_s:.1f} s" if total_s else "-")
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


def show_batch(ids: list[int]):
    rows, totals = mt.summarise(ids)
    st.subheader(f"Results for {len(ids)} image(s)")
    generalisation_note({r["country"] for r in rows})

    # ---- totals ----
    m = st.columns(5)
    m[0].metric("Images", totals["images"])
    m[1].metric("Real damages (ground truth)", totals["real damages"] if totals["with ground truth"] else "-")
    m[2].metric("Found", f"{totals['found']} of {totals['real damages']}" if totals["real damages"] else "-")
    m[3].metric("Missed", totals["missed"] if totals["with ground truth"] else "-")
    m[4].metric("False alarms", totals["false alarms"] if totals["with ground truth"] else "-")
    if totals["recall"] is not None:
        st.caption(f"On these {totals['with ground truth']} labelled image(s): the model found "
                   f"{totals['recall']:.0%} of the real damages (recall) and "
                   f"{(totals['precision'] or 0):.0%} of its boxes were real damages (precision). "
                   "A handful of images is a demonstration, not the official evaluation "
                   "(that uses the whole test set, Phase 7).")

    # ---- grid: model boxes + ground truth in green ----
    st.markdown("**Model prediction** (coloured boxes) with the **ground truth in green**")
    per_row = 3
    for start in range(0, len(rows), per_row):
        cols = st.columns(per_row)
        for col, r in zip(cols, rows[start:start + per_row]):
            info = mt.describe(r["image #"])
            annotated = settings.PROJECT_ROOT / info["row"]["annotated_path"]
            picture = mt.draw_overlay(annotated, info["gt"]) if info["gt"] is not None else str(annotated)
            if r["real damages"] is None:
                caption = f"#{r['image #']} {r['file']} · {r['model detections']} detection(s) · no ground truth"
            elif r["real damages"] == 0:
                caption = (f"#{r['image #']} {r['country']} · clean road · "
                           f"{r['false alarms']} false alarm(s)")
            else:
                caption = (f"#{r['image #']} {r['country']} · found {r['found']} of "
                           f"{r['real damages']} · {r['false alarms']} false alarm(s)")
            col.image(picture, caption=caption, width="stretch")

    # ---- summary table with a totals row ----
    table = pd.DataFrame(rows)
    total_row = {"image #": "TOTAL", "file": "", "country": "", "split": "",
                 "model detections": table["model detections"].sum(),
                 "real damages": totals["real damages"], "found": totals["found"],
                 "missed": totals["missed"], "false alarms": totals["false alarms"],
                 "inference ms": round(table["inference ms"].mean())}
    table = pd.concat([table, pd.DataFrame([total_row])], ignore_index=True).astype(str)
    st.dataframe(table.replace({"None": "-", "nan": "-"}), hide_index=True, width="stretch")

    # ---- one image in full detail ----
    with st.expander("Show one image in detail (original, prediction, ground truth side by side)"):
        chosen = st.selectbox("Image", ids, format_func=lambda i: next(
            f"#{r['image #']} {r['file']}" for r in rows if r["image #"] == i))
        show_manual_result(chosen)


@st.fragment(run_every=2)
def demo_status():
    status = demo_replay.get_status()
    state = status.get("state", "idle")
    if state == "idle":
        st.caption("No escalation demo running.")
        return
    poles = ", ".join(f"{p}: {n}" for p, n in (status.get("poles") or {}).items())
    icon = {"running": "⏳", "finished": "✅", "stopped": "⏹", "error": "❌"}.get(state, "")
    st.markdown(f"{icon} **Escalation demo {state}**: {status.get('message', '')}")
    if status.get("rounds"):
        st.progress(min(1.0, status["round"] / status["rounds"]),
                    text=f"{status['round']} of {status['rounds']} demo hours")
    st.caption(f"Demo poles: {poles}")


def escalation_demo(ids: list[int]):
    st.subheader("Simulate escalation with these images")
    st.markdown(
        "Each chosen image becomes the fixed camera view of one **DEMO pole**. Once per demo "
        "hour (**1 second = 1 hour**, so one day takes 24 s) the image is sent again as that "
        "pole's camera photo, through the real server, the real YOLO worker and the real "
        "escalation engine with its real thresholds. Watch it go Watching → Monitoring → "
        "Repair requested (or Urgent) on the **authority portal**.")
    st.warning("**Growth is not simulated.** The same photo is replayed, so a damage cannot "
               "grow: growth will show about 0 %. Real growth needs real photos of the same "
               "spot taken days apart.")
    names = {r["image #"]: r["file"] for r in mt.summarise(ids)[0]}
    chosen = st.multiselect(f"Images to replay (one DEMO pole each, at most {demo_replay.MAX_POLES})",
                            ids, default=ids[:demo_replay.MAX_POLES],
                            format_func=lambda i: f"#{i} {names.get(i, '')}",
                            max_selections=demo_replay.MAX_POLES)
    c1, c2, c3, c4, c5 = st.columns(5)
    vehicle = c1.checkbox("Passing vehicles", value=True, help="Now and then a vehicle covers part of the view for one frame")
    lighting = c2.checkbox("Lighting changes", value=True, help="Brightness varies; darker at dawn and dusk")
    jitter = c3.checkbox("Camera jitter", value=True, help="The view shifts by a few pixels (wind on the pole)")
    night = c4.checkbox("Night", value=True, help="Photos between 20:00 and 06:00 are almost black (unlit camera)")
    days = c5.number_input("Demo days", min_value=3, max_value=7, value=4,
                           help="3 days of monitoring are needed before a normal request")

    running = demo_replay.get_status().get("state") == "running"
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("▶ Start escalation demo", type="primary", disabled=running or not chosen):
        if not mt.worker_running():
            st.error(f"The worker is not running. {WORKER_HELP}")
        else:
            args = ["--image-ids", *map(str, chosen), "--days", str(int(days))]
            args += [f"--no-{name}" for name, on in (("vehicle", vehicle), ("lighting", lighting),
                                                     ("jitter", jitter), ("night", night)) if not on]
            demo_replay.start_detached(args)
            time.sleep(1.5)
            st.rerun()
    if b2.button("⏹ Stop demo", disabled=not running):
        demo_replay.request_stop()
    if b3.button("🧹 Clear demo data", help="Deletes every DEMO pole, its damages, requests and "
                 "replayed photos. Camera poles and normal manual tests are not touched."):
        n = demo_replay.clear_demo_data()
        st.success(f"Demo data cleared ({n} files removed).")
    b4.link_button("Open authority portal", "http://localhost:8502")
    demo_status()


with test_tab:
    st.markdown("Run any road photos through SmartPole, the same way a camera photo is "
                "processed. Test images are stored as **manual tests**: they never mix with "
                "camera data and never create repair requests.")
    if mt.worker_running():
        st.caption("🟢 Worker is running.")
    else:
        st.error(f"🔴 The worker is not running, so images cannot be processed. {WORKER_HELP}")

    up_col, rand_col = st.columns([3, 2])
    with up_col:
        uploaded = st.file_uploader("Upload road photos (JPEG or PNG, e.g. saved from WhatsApp). "
                                    "You can select several at once.",
                                    type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if st.button(f"Test {len(uploaded) or ''} uploaded image(s)", type="primary", disabled=not uploaded):
            run_batch_and_remember([(f.getvalue(), f.name) for f in uploaded])
    with rand_col:
        st.write("Or pick random images from a country's **test split**:")
        country = st.selectbox("Country", mt.COUNTRIES,
                               format_func=lambda c: f"{c} (model trained on this)" if c == mt.TRAINED_ON
                               else f"{c} (unseen country)")
        n = st.number_input("How many", min_value=1, max_value=12, value=6)
        if st.button(f"🎲 Pick {int(n)} random test images"):
            try:
                paths = mt.random_test_images(country, int(n))
                run_batch_and_remember([(p.read_bytes(), p.name) for p in paths])
            except mt.DemoError as e:
                st.error(str(e))
        st.caption("Test-split images were held out before training, and every one has "
                   "ground truth labels to compare with.")

    if st.session_state.get("batch_ids"):
        st.divider()
        show_batch(st.session_state["batch_ids"])
        st.caption(f"Whole batch: {st.session_state.get('batch_total_s', 0):.1f} s "
                   "(upload + queue + model).")
        st.divider()
        escalation_demo(st.session_state["batch_ids"])
    else:
        st.divider()
        st.subheader("Simulate escalation with images")
        st.write("Test some images first (above). You can then replay them as DEMO poles.")
        demo_status()
