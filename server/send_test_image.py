"""Pretend to be the ESP32: send JPEGs and heartbeats to the running server.

Start the server first (python -m server.app), then in a second terminal:
    python -m server.send_test_image                       # 3 uploads of a sample road photo
    python -m server.send_test_image --count 10 --interval 2
    python -m server.send_test_image --image path\\to\\photo.jpg
    python -m server.send_test_image --url http://192.168.43.57:8000

It sends exactly what the firmware sends: raw JPEG bytes in the body with
X-Pole-ID and X-Seq headers, plus a JSON heartbeat. It also sends one bad
upload to check that the server rejects data that is not an image.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from config import settings


def sample_jpeg(image_path: Path | None) -> bytes:
    if image_path:
        return image_path.read_bytes()
    # Use a real road photo from the dataset if it is there.
    india = settings.RDD2022_DIR / "India" / "train" / "images"
    if india.is_dir():
        first = next(india.glob("*.jpg"), None)
        if first:
            print(f"Using sample image {first.name}")
            img = cv2.imread(str(first))
            img = cv2.resize(img, (320, 240))   # same size as the node's QVGA frames
            return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    # Otherwise a generated grey test image.
    img = np.full((240, 320, 3), 120, np.uint8)
    cv2.putText(img, "SmartPole test", (40, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


def post(url: str, body: bytes, headers: dict) -> tuple[int, str]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=f"http://127.0.0.1:{settings.SERVER_PORT}")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--pole-id", default="TEST01")
    args = parser.parse_args()

    jpeg = sample_jpeg(args.image)

    for seq in range(1, args.count + 1):
        code, text = post(f"{args.url}/upload", jpeg, {
            "Content-Type": "image/jpeg", "X-Pole-ID": args.pole_id, "X-Seq": str(seq)})
        print(f"upload seq={seq} {len(jpeg)} bytes -> HTTP {code} {text}")
        time.sleep(args.interval)

    hb = {"pole_id": args.pole_id, "uptime_s": 123, "rssi": -55,
          "free_heap": 150000, "free_psram": 3000000, "seq": args.count, "ip": "test"}
    code, text = post(f"{args.url}/heartbeat", json.dumps(hb).encode(),
                      {"Content-Type": "application/json"})
    print(f"heartbeat -> HTTP {code} {text}")

    # The server must refuse anything that is not a complete JPEG.
    code, text = post(f"{args.url}/upload", b"this is not an image",
                      {"Content-Type": "image/jpeg", "X-Pole-ID": args.pole_id, "X-Seq": "0"})
    print(f"bad upload (should be HTTP 400) -> HTTP {code} {text}")
    code, text = post(f"{args.url}/upload", jpeg[: len(jpeg) // 2],
                      {"Content-Type": "image/jpeg", "X-Pole-ID": args.pole_id, "X-Seq": "0"})
    print(f"cut-off JPEG (should be HTTP 400) -> HTTP {code} {text}")


if __name__ == "__main__":
    main()
