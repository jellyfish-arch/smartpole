"""Escalation service: runs the engine on new camera photos, every few seconds.

Run from the repository root, alongside the server and worker:
    python -m escalation.service                        # real time
    python -m escalation.service --speed 1440 --fresh   # demo: 1 day = 1 minute

--speed  multiplies elapsed time for the time rules (1440: a day lasts a
         minute). Rules and thresholds stay exactly the same.
--fresh  forget all camera tracking data and start from the next photo.
         Use it whenever you change --speed: old timestamps measured at a
         different speed would not be comparable.
"""

import argparse
import time

from escalation import engine, store
from server import db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between runs")
    args = parser.parse_args()

    store.init()
    with db.connection() as conn:
        if args.fresh:
            store.clear(conn, simulated=False)
            newest = conn.execute("SELECT COALESCE(MAX(id), 0) AS m FROM images").fetchone()["m"]
            store.set_state(conn, "last_image_id", newest)
            print(f"[escalation] fresh start: ignoring the {newest} photo(s) already in the database")
        store.set_state(conn, "speed", args.speed)
        seen = conn.execute("SELECT COALESCE(MAX(id), 0) AS m FROM track_events").fetchone()["m"]

    print(f"[escalation] running, speed x{args.speed:g}"
          + (" (DEMO: 1 day lasts %.1f min)" % (1440 / args.speed) if args.speed != 1 else "")
          + ". Ctrl+C to stop.")
    try:
        while True:
            with db.connection() as conn:
                engine.process_camera_images(conn, engine.utc_now(), args.speed)
                for e in conn.execute(
                        """SELECT e.*, t.pole_id FROM track_events e JOIN tracks t ON t.id=e.track_id
                           JOIN poles p ON p.pole_id=t.pole_id
                           WHERE e.id > ? AND p.simulated = 0 ORDER BY e.id""", (seen,)):
                    print(f"[escalation] damage #{e['track_id']} at {e['pole_id']}: "
                          f"{e['from_level'] or 'new'} -> {e['to_level']}. {e['reason']}")
                    seen = e["id"]
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("[escalation] stopped")


if __name__ == "__main__":
    main()
