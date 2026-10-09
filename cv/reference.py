"""Reference-frame differencing: a second, simple signal next to YOLO.

The camera never moves, so a photo of the clean road can be kept as a
"reference". Comparing a new photo with it shows WHERE the scene changed.
A new pothole is a change; so are a leaf, a shadow or a car, which is why
this is only a hint shown next to YOLO, never a detection on its own.

Optional. Without a saved reference nothing changes anywhere.

    python -m cv.reference --pole POLE01 --save          # latest camera photo = reference
    python -m cv.reference --pole POLE01 --save --image path\\to\\clean_road.jpg
    python -m cv.reference --pole POLE01 --delete
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from config import settings

REFERENCE_DIR = settings.DATA_DIR / "reference"
DIFF_THRESHOLD = 35          # grey-level difference that counts as "changed"
MIN_REGION_SHARE = 0.002     # ignore changed blobs smaller than 0.2 % of the photo


def reference_path(pole_id: str) -> Path:
    return REFERENCE_DIR / f"{pole_id}.jpg"


def changed_regions(pole_id: str, image_path: Path):
    """(picture with changed regions boxed in yellow, number of regions), or None.

    None when the pole has no reference photo.
    """
    ref_file = reference_path(pole_id)
    if not ref_file.exists():
        return None
    current = cv2.imread(str(image_path))
    ref = cv2.imread(str(ref_file))
    if current is None or ref is None:
        return None
    ref = cv2.resize(ref, (current.shape[1], current.shape[0]))

    a = cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (7, 7), 0).astype(np.float32)
    b = cv2.GaussianBlur(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), (7, 7), 0).astype(np.float32)
    b *= a.mean() / max(b.mean(), 1)            # cancel an overall brightness change
    mask = (cv2.absdiff(a, b) > DIFF_THRESHOLD).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # remove speckles

    out = current.copy()
    min_area = MIN_REGION_SHARE * mask.size
    regions = 0
    for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) >= min_area:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
            regions += 1
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), regions


def main():
    parser = argparse.ArgumentParser(description="Save or delete a pole's clean-road reference photo.")
    parser.add_argument("--pole", default="POLE01")
    parser.add_argument("--image", type=Path, help="photo to use (default: latest camera photo)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save", action="store_true")
    group.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    if args.delete:
        reference_path(args.pole).unlink(missing_ok=True)
        print(f"Reference photo for {args.pole} removed.")
        return
    source = args.image
    if source is None:
        from cv.calibrate import latest_camera_photo
        source = latest_camera_photo(args.pole)
    if source is None or not Path(source).exists():
        print(f"No photo found for {args.pole}. Pass one with --image.")
        return
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, reference_path(args.pole))
    print(f"Saved {source} as the clean-road reference for {args.pole}. "
          "Take it when the road (or sheet) has no damage and the camera is in its final position.")


if __name__ == "__main__":
    main()
