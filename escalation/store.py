"""Database tables for tracking and escalation.

These tables sit next to the ingest tables (images, detections) in the same
SQLite file, but are created here, so the camera path (server and worker)
does not change at all.
"""

import json

from config import settings
from server import db

POLES_FILE = settings.PROJECT_ROOT / "config" / "poles.json"
OUTBOX_DIR = settings.DATA_DIR / "outbox"

SCHEMA = """
CREATE TABLE IF NOT EXISTS poles (
    pole_id        TEXT PRIMARY KEY,
    name           TEXT,
    lat            REAL,
    lon            REAL,
    address        TEXT,
    view_w_m       REAL,          -- approx_view_m from config/poles.json
    view_h_m       REAL,
    simulated      INTEGER NOT NULL DEFAULT 0,
    last_frame_at  TEXT           -- newest photo seen from this pole
);

-- One row per damage the system follows over time.
CREATE TABLE IF NOT EXISTS tracks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    pole_id         TEXT NOT NULL,
    level           TEXT NOT NULL,
    class_name      TEXT,
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    confirmed_at    TEXT,
    requested_at    TEXT,
    closed_at       TEXT,
    sightings       INTEGER NOT NULL DEFAULT 0,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,      -- reference box, fractions of the image
    area_cm2        REAL,
    length_cm       REAL,
    size_basis      TEXT,                    -- calibrated / estimate / unknown
    growth_per_week REAL,                    -- 0.15 = +15 % per week
    growth_err      REAL,                    -- uncertainty: growth_per_week +/- growth_err
    severity        TEXT,                    -- Low / Medium / High / NULL (unknown)
    request_id      TEXT,
    priority        INTEGER,                 -- 1 urgent, 2 normal
    suppress        INTEGER NOT NULL DEFAULT 0,  -- dismissed: absorb detections at this spot
    reappeared_after INTEGER,                -- id of a repaired damage at the same spot
    last_image_id   INTEGER,
    closed_by       TEXT,
    close_note      TEXT,
    updated_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_tracks_pole_level ON tracks(pole_id, level);

CREATE TABLE IF NOT EXISTS track_observations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id    INTEGER NOT NULL REFERENCES tracks(id),
    image_id    INTEGER,                     -- NULL for simulated sightings
    observed_at TEXT NOT NULL,
    class_name  TEXT,
    confidence  REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    area_cm2    REAL,
    length_cm   REAL,
    rel_area    REAL,                        -- box share of the frame (for growth ratios)
    rel_len     REAL
);
CREATE INDEX IF NOT EXISTS idx_obs_track_time ON track_observations(track_id, observed_at);

-- Audit trail: every level change and why. "Why was this escalated?" must
-- always have an answer.
CREATE TABLE IF NOT EXISTS track_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id   INTEGER NOT NULL REFERENCES tracks(id),
    at         TEXT NOT NULL,
    from_level TEXT,
    to_level   TEXT,
    reason     TEXT,
    actor      TEXT                          -- system, or authority:<name>
);

-- Outbox: messages for the authority. Written in the same transaction as
-- the escalation, so none can be lost. A real sender would deliver them.
CREATE TABLE IF NOT EXISTS notifications (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id   INTEGER NOT NULL REFERENCES tracks(id),
    created_at TEXT NOT NULL,
    request_id TEXT,
    priority   INTEGER,
    subject    TEXT,
    body       TEXT,
    status     TEXT NOT NULL DEFAULT 'simulated (outbox)'
);

CREATE TABLE IF NOT EXISTS engine_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def init() -> None:
    db.init_db()
    with db.connection() as conn:
        conn.executescript(SCHEMA)
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(tracks)")}
        if "growth_err" not in columns:          # tables created before this column existed
            conn.execute("ALTER TABLE tracks ADD COLUMN growth_err REAL")


def registry() -> dict:
    data = json.loads(POLES_FILE.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def ensure_pole(conn, pole_id: str, info: dict | None = None, simulated: bool = False) -> dict:
    """Return the pole row; create it from config/poles.json on first sight."""
    row = conn.execute("SELECT * FROM poles WHERE pole_id=?", (pole_id,)).fetchone()
    if row is None or info is not None:
        info = info or registry().get(pole_id, {})
        view = info.get("approx_view_m") or (None, None)
        conn.execute(
            """INSERT INTO poles (pole_id, name, lat, lon, address, view_w_m, view_h_m, simulated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(pole_id) DO UPDATE SET name=excluded.name, lat=excluded.lat,
                 lon=excluded.lon, address=excluded.address, view_w_m=excluded.view_w_m,
                 view_h_m=excluded.view_h_m, simulated=excluded.simulated""",
            (pole_id, info.get("name", f"{pole_id} (not registered)"), info.get("lat"),
             info.get("lon"), info.get("address", "location not registered in config/poles.json"),
             view[0], view[1], int(simulated)))
        row = conn.execute("SELECT * FROM poles WHERE pole_id=?", (pole_id,)).fetchone()
    return dict(row)


def get_state(conn, key, default=None):
    row = conn.execute("SELECT value FROM engine_state WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_state(conn, key, value) -> None:
    conn.execute("INSERT INTO engine_state (key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))


def clear(conn, simulated: bool | None = None) -> None:
    """Delete tracking data. simulated=True: only SIM poles; None: everything."""
    where, params = "", ()
    if simulated is not None:
        where = "WHERE pole_id IN (SELECT pole_id FROM poles WHERE simulated=?)"
        params = (int(simulated),)
    ids = [r["id"] for r in conn.execute(f"SELECT id FROM tracks {where}", params)]
    for table in ("track_observations", "track_events", "notifications"):
        conn.executemany(f"DELETE FROM {table} WHERE track_id=?", [(i,) for i in ids])
    conn.executemany("DELETE FROM tracks WHERE id=?", [(i,) for i in ids])
    if simulated is None:
        conn.execute("DELETE FROM engine_state")
