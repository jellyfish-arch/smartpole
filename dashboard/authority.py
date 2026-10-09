"""Authority portal: repair requests for the municipal road department.

Run from the repository root (a separate app from the camera dashboard):
    streamlit run dashboard/authority.py --server.port 8502

Shows open repair requests by priority with their location, the damages
still being watched, each damage's history and evidence, and the outbox of
notifications. The authority closes a damage here: Repaired or Dismissed.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import pandas as pd
import streamlit as st

from config import settings
from cv import calibration
from escalation import demo_replay, engine, rules as R, store
from server import db

REFRESH_S = 5
COLORS = {R.URGENT: "#d62728", R.REPAIR_REQUESTED: "#ff7f0e",
          R.MONITORING: "#1f77b4", R.WATCHING: "#7f7f7f"}
ICONS = {R.URGENT: "🔴", R.REPAIR_REQUESTED: "🟠", R.MONITORING: "🔵", R.WATCHING: "⚪",
         R.REPAIRED: "✅", R.DISMISSED: "🚫", R.EXPIRED: "⌛"}


def query(sql, params=()):
    with db.connection() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def parse(t):
    return datetime.fromisoformat(t) if t else None


def local(t):
    return parse(t).astimezone().strftime("%d %b %H:%M") if t else "-"


def ago(t, speed=1.0):
    """'3 h ago'. For DEMO poles (fast clock) the age is shown in demo time."""
    if not t:
        return "-"
    s = (datetime.now(timezone.utc) - parse(t)).total_seconds() * speed
    for unit, size in (("d", 86400), ("h", 3600), ("min", 60)):
        if s >= size:
            return f"{s / size:.0f} {unit} ago" + (" (demo time)" if speed != 1 else "")
    return "just now"


TAG = {"simulation": "SIMULATION · ", "demo": "DEMO · ", "camera": ""}
SUFFIX = {"simulation": " (SIM)", "demo": " (DEMO)", "camera": ""}


def pole_speed(t) -> float:
    """Clock speed for this damage's pole: demo poles run fast, camera poles use the service's speed."""
    if t["kind"] == "camera":
        return demo_speed()
    return t["pole_speed"] or 1.0


def demo_speed() -> float:
    rows = query("SELECT value FROM engine_state WHERE key='speed'")
    return float(rows[0]["value"]) if rows else 1.0


def not_seen_flag(t) -> str:
    """Warn when an open damage has not been seen although the camera is sending photos."""
    if not t["pole_last_frame"]:
        return ""
    gap = engine.eff(parse(t["last_seen"]), parse(t["pole_last_frame"]), pole_speed(t))
    return f"⚠ not seen for {gap / R.DAY:.1f} days" if gap >= R.NOT_SEEN_FLAG else ""


def open_tracks(show_sim: bool):
    rows = query(
        f"""SELECT t.*, p.name AS pole_name, p.lat, p.lon, p.address, p.simulated,
                   p.kind, p.speed AS pole_speed, p.last_frame_at AS pole_last_frame
            FROM tracks t JOIN poles p ON p.pole_id = t.pole_id
            WHERE t.level IN ({','.join('?' * len(R.OPEN_LEVELS))})
            {'' if show_sim else 'AND p.simulated = 0'}
            ORDER BY COALESCE(t.priority, 9), t.requested_at, t.first_seen""", R.OPEN_LEVELS)
    return rows


def describe(t) -> str:
    ref = t["request_id"] or f"damage #{t['id']}"
    return (f"{ICONS[t['level']]} {TAG[t['kind']]}{ref} · {R.LABEL[t['level']]} · {t['pole_id']} · "
            f"{engine.CLASS_NAMES.get(t['class_name'], t['class_name'])}")


# ======================================================================
st.set_page_config(page_title="SmartPole Authority", page_icon="🏛️", layout="wide")
store.init()
st.title("🏛️ Road repair requests")
st.caption("SmartPole: automated pothole detection and location mapping, municipal authority view")

with st.sidebar:
    show_sim = st.checkbox("Show SIMULATION and DEMO poles", value=True)
    inspector = st.text_input("Your name (recorded with every action)", value="")
    st.caption("Requests are raised automatically by the escalation engine "
               "(see docs/ESCALATION.md). Only you can close them.")
    st.divider()
    status = demo_replay.get_status()
    if status.get("state") not in (None, "idle"):
        st.markdown(f"**Escalation demo:** {status['state']}")
        if status.get("rounds"):
            st.progress(min(1.0, status["round"] / status["rounds"]),
                        text=f"{status.get('message', '')} ({status['round']}/{status['rounds']} demo hours)")
    if st.button("🧹 Clear demo data", help="Deletes every DEMO pole, its damages, requests "
                 "and replayed photos. Camera and simulation poles are not touched."):
        n = demo_replay.clear_demo_data()
        st.success(f"Demo data cleared ({n} files removed).")
        st.rerun()

speed = demo_speed()
if speed != 1:
    st.warning(f"⏩ DEMO SPEED ×{speed:g}: one day lasts {1440 / speed:.1f} minutes for the "
               "camera poles. The rules and thresholds are unchanged; only the clock runs faster.")


@st.fragment(run_every=REFRESH_S)
def overview():
    tracks = open_tracks(show_sim)
    counts = {lvl: sum(t["level"] == lvl for t in tracks) for lvl in R.OPEN_LEVELS}
    c = st.columns(4)
    c[0].metric("🔴 Urgent", counts[R.URGENT])
    c[1].metric("🟠 Repair requested", counts[R.REPAIR_REQUESTED])
    c[2].metric("🔵 Monitoring", counts[R.MONITORING])
    c[3].metric("⚪ Watching (unconfirmed)", counts[R.WATCHING])

    # ---- map: one dot per pole, coloured by its most serious open damage ----
    poles = {}
    for t in tracks:
        if t["lat"] is None or t["level"] == R.WATCHING:
            continue
        cur = poles.get(t["pole_id"])
        if cur is None or R.RANK[t["level"]] > R.RANK[cur["level"]]:
            poles[t["pole_id"]] = {"lat": t["lat"], "lon": t["lon"], "level": t["level"]}
    if poles:
        df = pd.DataFrame([{"lat": p["lat"], "lon": p["lon"], "color": COLORS[p["level"]]}
                           for p in poles.values()])
        st.map(df, latitude="lat", longitude="lon", color="color", size=25, zoom=15)
        st.caption("🔴 urgent · 🟠 repair requested · 🔵 monitoring (one dot per pole). "
                   "Map tiles need an internet connection; the table below has map links.")

    # ---- open requests ----
    requests = [t for t in tracks if t["level"] in R.PRIORITY]
    st.subheader(f"Open repair requests ({len(requests)})")
    if requests:
        st.dataframe(pd.DataFrame([{
            "": ICONS[t["level"]],
            "request": t["request_id"] + SUFFIX[t["kind"]],
            "priority": t["priority"],
            "pole": t["pole_id"],
            "location": t["address"],
            "damage": engine.CLASS_NAMES.get(t["class_name"], t["class_name"]),
            "size": engine.size_text(t),
            "severity": t["severity"] or "unknown",
            "growth / week": (f"{t['growth_per_week']:+.0%} ± {t['growth_err'] or 0:.0%}"
                              if t["growth_per_week"] is not None else "-"),
            "requested": ago(t["requested_at"], pole_speed(t)),
            "last seen": ago(t["last_seen"], pole_speed(t)) + (f"  {not_seen_flag(t)}" if not_seen_flag(t) else ""),
            "map": (f"https://www.google.com/maps?q={t['lat']},{t['lon']}" if t["lat"] is not None else None),
        } for t in requests]), hide_index=True, width="stretch",
            column_config={"map": st.column_config.LinkColumn("map", display_text="open map")})
    else:
        st.write("No open requests.")

    # ---- watch list ----
    watch = [t for t in tracks if t["level"] in (R.MONITORING, R.WATCHING)]
    with st.expander(f"Watch list: damages not (yet) requested ({len(watch)})"):
        if watch:
            st.dataframe(pd.DataFrame([{
                "": ICONS[t["level"]], "damage #": t["id"], "level": R.LABEL[t["level"]],
                "pole": t["pole_id"] + SUFFIX[t["kind"]],
                "damage": engine.CLASS_NAMES.get(t["class_name"], t["class_name"]),
                "size": engine.size_text(t) if t["level"] == R.MONITORING else "-",
                "severity": t["severity"] or "-", "sightings": t["sightings"],
                "first seen": ago(t["first_seen"], pole_speed(t)),
                "last seen": ago(t["last_seen"], pole_speed(t)),
            } for t in watch]), hide_index=True, width="stretch")
            st.caption("Watching = seen but not yet proven real (needs 5 sightings in 3 different "
                       "hours over 6 h). Monitoring = real, being measured; Low-severity damages stay here.")
        else:
            st.write("Nothing being watched.")
    st.caption(f"Refreshes every {REFRESH_S} s. Updated {datetime.now():%H:%M:%S}.")


overview()

# ======================================================================
# One damage in detail, with the authority's actions
# ======================================================================
st.divider()
st.subheader("Damage details and actions")
tracks = [t for t in open_tracks(show_sim) if t["level"] != R.WATCHING]
if not tracks:
    st.write("No confirmed open damages.")
else:
    by_id = {t["id"]: t for t in tracks}
    t = by_id[st.selectbox("Choose a damage", list(by_id), format_func=lambda i: describe(by_id[i]))]
    left, right = st.columns([3, 2])

    with left:
        if t["last_image_id"]:
            img = query("SELECT path, width, height FROM images WHERE id=?", (t["last_image_id"],))
            if img and (settings.PROJECT_ROOT / img[0]["path"]).exists():
                frame = cv2.imread(str(settings.PROJECT_ROOT / img[0]["path"]))
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (int(t["x1"] * w), int(t["y1"] * h)),
                              (int(t["x2"] * w), int(t["y2"] * h)), (0, 0, 255), max(2, w // 200))
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch",
                         caption=f"Evidence: latest photo (image #{t['last_image_id']}), damage boxed in red")
        else:
            st.info("Simulated damage: no photo. (Camera damages show the latest photo with the damage boxed.)")

        obs = pd.DataFrame(query(
            "SELECT observed_at, area_cm2, length_cm, rel_area, rel_len FROM track_observations "
            "WHERE track_id=? ORDER BY observed_at", (t["id"],)))
        if not obs.empty:
            crack = t["class_name"] in R.CRACK_CLASSES
            col, label = ("length_cm", "length (cm)") if crack else ("area_cm2", "area (cm²)")
            if obs[col].isna().all():
                col, label = ("rel_len", "length (share of frame)") if crack else ("rel_area", "area (share of frame)")
            chart = obs.assign(time=pd.to_datetime(obs["observed_at"], utc=True).dt.tz_convert(
                datetime.now().astimezone().tzinfo)).set_index("time")[[col]].rename(columns={col: label})
            st.markdown(f"**Size over time** ({t['size_basis']})")
            st.line_chart(chart)

    with right:
        st.markdown(f"### {describe(t)}")
        if t["kind"] == "demo":
            st.warning("**DEMO pole: one photo replayed as hourly camera photos** (1 s = 1 h). "
                       "Detection, tracking and escalation are real. **Growth is ~0 % because "
                       "every frame is the same photo**: a replay cannot show a damage growing. "
                       "Real growth needs real photos of the same spot taken days apart.")
        facts = {
            "Level": R.LABEL[t["level"]] + (f" (priority {t['priority']})" if t["priority"] else ""),
            "Damage": f"{engine.CLASS_NAMES.get(t['class_name'], t['class_name'])} ({t['class_name']})",
            "Size": engine.size_text(t),
            "Severity": t["severity"] or "unknown (no calibration)",
            "Growth": engine.growth_text(t),
            "Pole": f"{t['pole_id']}: {t['pole_name']}",
            "Location": t["address"],
            "GPS": f"{t['lat']}, {t['lon']}" if t["lat"] is not None else "not registered",
            "Position in view": calibration.position_text(t["pole_id"], (t["x1"], t["y1"], t["x2"], t["y2"])),
            "First seen": f"{local(t['first_seen'])} ({ago(t['first_seen'], pole_speed(t))})",
            "Last seen": f"{local(t['last_seen'])} ({ago(t['last_seen'], pole_speed(t))}) {not_seen_flag(t)}",
            "Sightings": t["sightings"],
        }
        if t["reappeared_after"]:
            facts["Note"] = f"Reappeared at the spot of damage #{t['reappeared_after']}, which was marked repaired"
        st.table(pd.DataFrame([(k, str(v)) for k, v in facts.items()], columns=["", "value"]).set_index(""))
        if t["lat"] is not None:
            st.link_button("Open location in Google Maps",
                           f"https://www.google.com/maps?q={t['lat']},{t['lon']}")

        with st.form(f"close_{t['id']}"):
            st.markdown("**Close this damage**")
            note = st.text_input("Note (e.g. 'patched by ward 4 crew', 'manhole cover')")
            b1, b2 = st.columns(2)
            repaired = b1.form_submit_button("✅ Mark repaired")
            dismissed = b2.form_submit_button("🚫 Dismiss: not road damage")
        if repaired or dismissed:
            if not inspector.strip():
                st.error("Enter your name in the sidebar first: every action is recorded with a name.")
            else:
                with db.connection() as conn:
                    engine.close(conn, t["id"], R.REPAIRED if repaired else R.DISMISSED,
                                 inspector.strip(), note.strip())
                st.success(f"{t['request_id'] or 'Damage #' + str(t['id'])} closed as "
                           f"{'repaired' if repaired else 'dismissed'}.")
                st.rerun()

    st.markdown("**History: every level change and why**")
    st.dataframe(pd.DataFrame([{"when": local(e["at"]), "from": R.LABEL.get(e["from_level"], "new"),
                                "to": R.LABEL[e["to_level"]], "by": e["actor"], "reason": e["reason"]}
                               for e in query("SELECT * FROM track_events WHERE track_id=? ORDER BY id",
                                              (t["id"],))]),
                 hide_index=True, width="stretch")

# ======================================================================
st.divider()
with st.expander("📨 Outbox: notifications sent to the authority"):
    st.caption("Prototype: notifications are stored here and as JSON files in data/outbox/. "
               "A real deployment would deliver each one to the municipal complaint system "
               "(API, or email/SMS to the ward engineer) and retry until acknowledged.")
    notes = query(f"""SELECT n.* FROM notifications n JOIN tracks t ON t.id = n.track_id
                      JOIN poles p ON p.pole_id = t.pole_id
                      {'' if show_sim else 'WHERE p.simulated = 0'}
                      ORDER BY n.id DESC LIMIT 20""")
    for n in notes:
        st.markdown(f"**{n['subject']}**  ·  {local(n['created_at'])}  ·  _{n['status']}_")
        st.code(n["body"], language=None)
    if not notes:
        st.write("No notifications yet.")

with st.expander("Recently closed"):
    closed = query(f"""SELECT t.*, p.simulated FROM tracks t JOIN poles p ON p.pole_id = t.pole_id
                       WHERE t.level IN ('REPAIRED', 'DISMISSED')
                       {'' if show_sim else 'AND p.simulated = 0'}
                       ORDER BY t.closed_at DESC LIMIT 20""")
    if closed:
        st.dataframe(pd.DataFrame([{"": ICONS[c["level"]], "request": c["request_id"] or f"#{c['id']}",
                                    "pole": c["pole_id"], "closed": local(c["closed_at"]),
                                    "as": R.LABEL[c["level"]], "by": c["closed_by"], "note": c["close_note"]}
                                   for c in closed]), hide_index=True, width="stretch")
    else:
        st.write("Nothing closed yet.")
