"""A4 calibration: turn pixels into real centimetres.

    ONLY CALIBRATE AFTER THE CAMERA IS FIXED IN ITS FINAL POSITION.
    If the camera moves even slightly, the calibration is wrong: calibrate again.

Put an A4 sheet (21 x 29.7 cm) flat on the surface under the camera, take a
photo, then run (from the repository root):

    python -m cv.calibrate                                  # latest POLE01 camera photo
    python -m cv.calibrate --pole POLE01 --image path\\to\\photo.jpg
    python -m cv.calibrate --pole POLE01 --delete           # remove the calibration

A window opens. Click the 4 corners of the A4 sheet, going around the sheet
(clockwise or anticlockwise), and make the FIRST TWO clicks along a SHORT
edge (the 21 cm side). Keys: r = start again, s or Enter = save, q or Esc = quit.

Why 4 corners: the camera is tilted, so the rectangle appears as a trapezium
(the far edge looks shorter). Four point pairs are exactly what is needed to
compute a homography: a 3x3 matrix that maps every pixel of the flat surface
to its true position in centimetres, undoing the perspective. After the 4th
click a second window shows the photo "from above": the A4 sheet must look
like a proper rectangle there. If it does not, press r and click again.

The result is saved to data/calibration/<pole>.json. cv/calibration.py reads
it, so detection sizes and the escalation engine switch from "estimate" to
real cm automatically. Without the file, everything keeps working with
sizes labelled as estimates.

For testing without a mouse, give the corners directly:
    python -m cv.calibrate --pole POLE01 --image photo.jpg --points 120,80 420,85 470,400 70,395
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from config import settings
from cv import calibration

A4_CM = (21.0, 29.7)                      # short side, long side
WARNING = "ONLY CALIBRATE AFTER THE CAMERA IS FIXED IN ITS FINAL POSITION"
PREVIEW_PX_PER_CM = 12                    # scale of the "seen from above" check window


def latest_camera_photo(pole_id: str) -> Path | None:
    from server import db
    with db.connection() as conn:
        row = conn.execute("SELECT path FROM images WHERE pole_id=? AND source='camera' "
                           "ORDER BY id DESC LIMIT 1", (pole_id,)).fetchone()
    return settings.PROJECT_ROOT / row["path"] if row else None


def homography(corners_px) -> np.ndarray:
    """3x3 matrix mapping image pixels to centimetres on the surface.

    corners_px: the 4 clicked corners, in order around the sheet, starting
    with a short (21 cm) edge.
    """
    w, h = A4_CM
    world_cm = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    return cv2.getPerspectiveTransform(np.float32(corners_px), world_cm)


def check_corners(corners_px) -> str | None:
    """Return a problem description, or None if the 4 points look usable."""
    pts = np.float32(corners_px)
    if len(pts) != 4:
        return "exactly 4 corners are needed"
    if min(np.linalg.norm(pts[i] - pts[j]) for i in range(4) for j in range(i + 1, 4)) < 10:
        return "two corners are almost on top of each other"
    if not cv2.isContourConvex(pts.reshape(-1, 1, 2)):
        return "the corners are not in order around the sheet (the shape crosses itself)"
    return None


def top_down_view(image, H) -> np.ndarray:
    """The photo as seen from above, with a 5 cm grid and the A4 outline."""
    margin_cm = 6
    w_cm, h_cm = A4_CM[0] + 2 * margin_cm, A4_CM[1] + 2 * margin_cm
    s = PREVIEW_PX_PER_CM
    to_px = np.array([[s, 0, margin_cm * s], [0, s, margin_cm * s], [0, 0, 1]], dtype=np.float64)
    view = cv2.warpPerspective(image, to_px @ H, (int(w_cm * s), int(h_cm * s)))
    for x in range(0, int(w_cm) + 1, 5):
        cv2.line(view, (x * s, 0), (x * s, view.shape[0]), (80, 80, 80), 1)
    for y in range(0, int(h_cm) + 1, 5):
        cv2.line(view, (0, y * s), (view.shape[1], y * s), (80, 80, 80), 1)
    cv2.rectangle(view, (margin_cm * s, margin_cm * s),
                  (int((margin_cm + A4_CM[0]) * s), int((margin_cm + A4_CM[1]) * s)), (0, 255, 0), 2)
    cv2.putText(view, "A4 should fill the green box", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(view, "grid = 5 cm", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return view


def save(pole_id: str, image_path: Path, image, corners_px, H) -> Path:
    calibration.CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    out = calibration.CALIBRATION_DIR / f"{pole_id}.json"
    out.write_text(json.dumps({
        "pole_id": pole_id,
        "image_size": [image.shape[1], image.shape[0]],
        "H": H.tolist(),
        "corners_px": [list(map(float, p)) for p in corners_px],
        "reference": "A4 sheet, 21 x 29.7 cm, first two corners along a short edge",
        "image": str(image_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": WARNING + ". Valid only while the camera stays in this exact position.",
    }, indent=2))
    return out


def click_corners(image) -> list | None:
    """Let the user click 4 corners. Returns them, or None if they quit."""
    points, title = [], "SmartPole calibration: click the 4 corners of the A4 sheet"
    scale = min(1.0, 1100 / image.shape[1])          # keep big photos on screen

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / scale, y / scale))

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    preview_shown = False
    while True:
        canvas = cv2.resize(image, None, fx=scale, fy=scale)
        cv2.putText(canvas, WARNING, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        hint = (f"Click corner {len(points) + 1} of 4 (first 2 clicks = a SHORT 21 cm edge)"
                if len(points) < 4 else "s/Enter = save   r = redo   q = quit")
        cv2.putText(canvas, hint, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        for i, (px, py) in enumerate(points):
            p = (int(px * scale), int(py * scale))
            cv2.circle(canvas, p, 5, (0, 255, 0), -1)
            cv2.putText(canvas, str(i + 1), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if i:
                q = (int(points[i - 1][0] * scale), int(points[i - 1][1] * scale))
                cv2.line(canvas, q, p, (0, 255, 0), 1)
        if len(points) == 4:
            cv2.line(canvas, (int(points[3][0] * scale), int(points[3][1] * scale)),
                     (int(points[0][0] * scale), int(points[0][1] * scale)), (0, 255, 0), 1)
            problem = check_corners(points)
            if problem:
                cv2.putText(canvas, f"Problem: {problem}. Press r.", (8, 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif not preview_shown:
                cv2.imshow("Top-down check (A4 must look rectangular)", top_down_view(image, homography(points)))
                preview_shown = True
        cv2.imshow(title, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return None
        if key == ord("r"):
            points.clear()
            if preview_shown:
                cv2.destroyWindow("Top-down check (A4 must look rectangular)")
            preview_shown = False
        if key in (ord("s"), 13) and len(points) == 4 and not check_corners(points):
            cv2.destroyAllWindows()
            return points


def main() -> int:
    parser = argparse.ArgumentParser(description="A4 calibration (pixels -> cm). " + WARNING)
    parser.add_argument("--pole", default="POLE01")
    parser.add_argument("--image", type=Path, help="photo to use (default: latest camera photo of the pole)")
    parser.add_argument("--points", nargs=4, metavar="X,Y", help="give the 4 corners instead of clicking")
    parser.add_argument("--delete", action="store_true", help="remove this pole's calibration")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  {WARNING}.")
    print("  If the camera moves afterwards, run this tool again.")
    print("=" * 70)

    if args.delete:
        path = calibration.CALIBRATION_DIR / f"{args.pole}.json"
        path.unlink(missing_ok=True)
        print(f"Calibration for {args.pole} removed. Sizes are estimates again.")
        return 0

    image_path = args.image or latest_camera_photo(args.pole)
    if image_path is None or not Path(image_path).exists():
        print(f"No photo found for {args.pole}. Pass one with --image path\\to\\photo.jpg")
        return 1
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read {image_path}")
        return 1
    print(f"Photo: {image_path} ({image.shape[1]}x{image.shape[0]})")

    if args.points:
        corners = [tuple(float(v) for v in p.split(",")) for p in args.points]
    else:
        print("Click the 4 corners of the A4 sheet in the window (first two along a SHORT edge).")
        corners = click_corners(image)
        if corners is None:
            print("Quit without saving.")
            return 1
    problem = check_corners(corners)
    if problem:
        print(f"Cannot calibrate: {problem}.")
        return 1

    H = homography(corners)
    out = save(args.pole, Path(image_path), image, corners, H)
    preview = out.with_name(f"{args.pole}_top_down.jpg")
    cv2.imwrite(str(preview), top_down_view(image, H))
    print(f"Saved {out}")
    print(f"Top-down check image: {preview} (the A4 sheet must fill the green box)")
    print("Sizes for this pole are now in real centimetres (perspective-corrected).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
