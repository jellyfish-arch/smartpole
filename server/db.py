"""SQLite database: table definitions and small helper functions.

The server, the worker and the dashboard all open the same database file.
Each caller opens its own short-lived connection, which is the simple and
safe way to share SQLite between processes.
"""

import sqlite3
from contextlib import contextmanager

from config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    pole_id      TEXT    NOT NULL,
    seq          INTEGER,               -- capture counter from the node (resets on reboot)
    received_at  TEXT    NOT NULL,      -- UTC, ISO 8601
    path         TEXT    NOT NULL,      -- relative to the project root
    size_bytes   INTEGER NOT NULL,
    width        INTEGER NOT NULL,
    height       INTEGER NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'pending',  -- pending / processing / done / error
    source       TEXT    NOT NULL DEFAULT 'camera',   -- 'camera' (ESP32) or 'manual' (demo test)
    original_name TEXT,                 -- file name of a manual test upload
    -- filled in by the worker:
    attempts       INTEGER NOT NULL DEFAULT 0,
    claimed_at     TEXT,
    claimed_by     TEXT,                -- id of the worker run that claimed it
    processed_at   TEXT,
    annotated_path TEXT,
    result_path    TEXT,
    model_path     TEXT,
    inference_ms   REAL,
    error          TEXT
);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);

CREATE TABLE IF NOT EXISTS heartbeats (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    pole_id      TEXT    NOT NULL,
    received_at  TEXT    NOT NULL,
    node_ip      TEXT,
    uptime_s     INTEGER,
    rssi         INTEGER,               -- Wi-Fi signal strength in dBm
    free_heap    INTEGER,
    free_psram   INTEGER,
    seq          INTEGER
);
CREATE INDEX IF NOT EXISTS idx_heartbeats_pole_time ON heartbeats(pole_id, received_at);

CREATE TABLE IF NOT EXISTS detections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id     INTEGER NOT NULL REFERENCES images(id),
    class_id     INTEGER NOT NULL,
    class_name   TEXT    NOT NULL,
    confidence   REAL    NOT NULL,
    x1 REAL NOT NULL, y1 REAL NOT NULL,      -- box corners in pixels
    x2 REAL NOT NULL, y2 REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_detections_image ON detections(image_id);

-- One row per running worker, refreshed every few seconds, so the dashboard
-- can tell whether a worker is running.
CREATE TABLE IF NOT EXISTS workers (
    worker_id    TEXT PRIMARY KEY,
    started_at   TEXT NOT NULL,
    last_seen    TEXT NOT NULL,
    device       TEXT,
    weights      TEXT
);
"""

# Columns added after the first version of a table. CREATE TABLE IF NOT
# EXISTS does not touch a table that already exists, so an older database
# file gets these columns added here instead of having to be deleted.
ADDED_COLUMNS = {
    "images": {
        "source": "TEXT NOT NULL DEFAULT 'camera'",
        "original_name": "TEXT",
        "attempts": "INTEGER NOT NULL DEFAULT 0",
        "claimed_at": "TEXT",
        "claimed_by": "TEXT",
        "processed_at": "TEXT",
        "annotated_path": "TEXT",
        "result_path": "TEXT",
        "model_path": "TEXT",
        "inference_ms": "REAL",
        "error": "TEXT",
    },
}


def connect() -> sqlite3.Connection:
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    # WAL mode lets the dashboard read while the server or worker writes,
    # instead of everyone waiting for one lock.
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def connection():
    """Open a connection, commit if the block succeeds, and always close it."""
    conn = connect()
    try:
        with conn:          # commits on success, rolls back on error
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    with connection() as conn:
        conn.executescript(SCHEMA)
        for table, columns in ADDED_COLUMNS.items():
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            for name, definition in columns.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def insert_image(pole_id, seq, received_at, path, size_bytes, width, height,
                 source="camera", original_name=None) -> int:
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO images (pole_id, seq, received_at, path, size_bytes, width, height,"
            " source, original_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pole_id, seq, received_at, path, size_bytes, width, height, source, original_name),
        )
        return cur.lastrowid


def insert_heartbeat(pole_id, received_at, node_ip, uptime_s, rssi,
                     free_heap, free_psram, seq) -> None:
    with connection() as conn:
        conn.execute(
            "INSERT INTO heartbeats (pole_id, received_at, node_ip, uptime_s, rssi,"
            " free_heap, free_psram, seq) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pole_id, received_at, node_ip, uptime_s, rssi, free_heap, free_psram, seq),
        )
