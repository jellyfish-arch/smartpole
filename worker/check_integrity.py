"""Check that no image has been processed twice (or left half-processed).

Run from the repository root, any time (also while everything is running):
    python -m worker.check_integrity

Checks
  * every 'done' image has its annotated image and its JSON result file
  * the detections stored in the database for each image match its JSON file
    exactly. If an image had been processed twice, it would have two sets
    of detections in the database, and the counts would not match.
  * images that are not 'done' have no detections in the database
"""

import json
import sys
from collections import Counter

from config import settings
from server import db


def main() -> int:
    problems = []
    with db.connection() as conn:
        status = Counter(r["status"] for r in conn.execute("SELECT status FROM images"))
        done = conn.execute("SELECT id, annotated_path, result_path FROM images WHERE status='done'").fetchall()
        db_counts = Counter(r["image_id"] for r in conn.execute("SELECT image_id FROM detections"))
        not_done_with_dets = conn.execute(
            "SELECT DISTINCT d.image_id FROM detections d JOIN images i ON i.id=d.image_id "
            "WHERE i.status != 'done'").fetchall()

    for row in done:
        annotated = settings.PROJECT_ROOT / (row["annotated_path"] or "")
        result = settings.PROJECT_ROOT / (row["result_path"] or "")
        if not annotated.is_file():
            problems.append(f"#{row['id']}: annotated image missing")
        if not result.is_file():
            problems.append(f"#{row['id']}: result JSON missing")
            continue
        n_json = len(json.loads(result.read_text())["detections"])
        if db_counts[row["id"]] != n_json:
            problems.append(f"#{row['id']}: {db_counts[row['id']]} detections in DB but {n_json} in JSON")
    for r in not_done_with_dets:
        problems.append(f"#{r['image_id']}: has detections but is not 'done'")

    print(f"images by status: {dict(status)}; detections: {sum(db_counts.values())}")
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    print(f"OK: all {len(done)} processed images have exactly one set of results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
