"""Test one image from the VS Code terminal.

Run from the repository root, with the server and worker running:
    python -m tools.test_image demo_images\\whatsapp_photo.jpg
    python -m tools.test_image --random            # random image from the India test split

It sends the image through the same path as the ESP32 (/upload -> worker)
as a manual test, waits for the result and prints the detections. If the
image is from RDD2022, it also compares against the ground-truth labels.
"""

import argparse
import sys
from pathlib import Path

from config import settings
from tools import manual_test as mt


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one image through SmartPole.")
    parser.add_argument("image", nargs="?", type=Path, help="JPEG or PNG file")
    parser.add_argument("--random", action="store_true", help="use a random India test-split image")
    args = parser.parse_args()

    if args.random:
        path = mt.random_test_image()
    elif args.image:
        path = args.image
    else:
        parser.error("give an image path, or --random")
    if not path.is_file():
        print(f"File not found: {path}")
        return 1

    print(f"Testing {path.name} ...")
    try:
        if not mt.worker_running():
            print("Warning: the worker is not running (python -m worker.worker). "
                  "The image will be queued but not processed.")
        test = mt.run_test(path.read_bytes(), path.name)
    except mt.DemoError as e:
        print(f"ERROR: {e}")
        return 1

    info = mt.describe(test["image_id"])
    row = info["row"]
    if row["status"] == "error":
        print(f"The worker could not process image #{row['id']}: {row['error']}")
        return 1

    print(f"\nImage #{row['id']}  ({row['width']}x{row['height']})")
    print(f"Model inference: {row['inference_ms']:.0f} ms   "
          f"total (upload + queue + inference): {test['total_s']:.1f} s")
    print(f"Model: {row['model_path']}")
    print(f"\nModel found {len(info['preds'])} damage(s) "
          f"(confidence >= {settings.WORKER_CONF}):")
    for d in info["preds"]:
        print(f"  {d['class_name']}  {mt.CLASS_NAMES[d['class_name']]:<20} "
              f"confidence {d['confidence']:.2f}   box ({d['x1']:.0f}, {d['y1']:.0f}, "
              f"{d['x2']:.0f}, {d['y2']:.0f})")

    if info["gt"] is None:
        print("\nGround truth: none. This image is not from RDD2022, so there is no "
              "correct answer to compare with.")
    else:
        print(f"\nGround truth ({info['split'] or 'unknown'} split): {len(info['gt'])} real damage(s)")
        for b in info["gt"]:
            print(f"  {b['class_name']}  {mt.CLASS_NAMES[b['class_name']]}")
        c = info["comparison"]
        print(f"Found {c['found']} of {c['real']} (same class, IoU >= 0.5); "
              f"missed {c['missed']}; false alarms {c['false_alarms']}")
        if info["split"] in ("train", "val"):
            print(f"Note: {mt.SPLIT_WARNING[info['split']]}")

    print(f"\nAnnotated image: {settings.PROJECT_ROOT / row['annotated_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
