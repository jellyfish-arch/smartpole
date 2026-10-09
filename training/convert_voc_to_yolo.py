"""Convert RDD2022 Pascal VOC XML annotations to YOLO format.

Run from the repository root:
    python -m training.convert_voc_to_yolo --countries India
    python -m training.convert_voc_to_yolo            # all countries

Output (inside the gitignored dataset/ folder):
    dataset/yolo/images/<Country>/<name>.jpg
    dataset/yolo/labels/<Country>/<name>.txt   one line per box:
        <class> <x_center> <y_center> <width> <height>   (all 0-1, relative to image size)

Decisions made here (see training/reports/rdd2022_inspection.md):
  * Only D00, D10, D20, D40 are kept. Every other label (D44, D50, D43,
    Repair, D01, D11, "Block crack", the typo "D0w0") is dropped on purpose.
    Dropped counts are saved in training/reports/conversion_<countries>.json.
  * Images left with no target boxes still get an EMPTY label file. YOLO
    treats these as "background" images. How many of them are used is
    decided later, in split_dataset.py.
  * Images whose longer side is above MAX_SIDE (Norway, ~4040x2035) are
    shrunk to MAX_SIDE. YOLO labels are fractions of the image size, so
    resizing does not change them. Training at imgsz=640 shrinks every
    image anyway; doing it once here instead of on every epoch makes
    loading ~10x faster and saves ~9 GB. 1280 keeps room to try imgsz=1280.
"""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter

from PIL import Image

from config import settings

CLASSES = ["D00", "D10", "D20", "D40"]          # index in this list = YOLO class id
MAX_SIDE = 1280
YOLO_DIR = settings.DATASET_DIR / "yolo"
REPORT_DIR = settings.PROJECT_ROOT / "training" / "reports"


def voc_to_yolo_lines(xml_path, img_w, img_h, stats):
    """Read one XML file and return YOLO label lines for the target classes."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is not None:
        xml_w = int(float(size.findtext("width", "0")))
        xml_h = int(float(size.findtext("height", "0")))
        if (xml_w, xml_h) != (img_w, img_h):
            stats["xml_size_mismatch"] += 1

    lines = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in CLASSES:
            stats[f"dropped_{name}"] += 1
            continue
        box = obj.find("bndbox")
        x1, y1, x2, y2 = (float(box.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax"))
        # Keep the box inside the image; a few annotations poke past the edge.
        x1, x2 = max(0.0, x1), min(float(img_w), x2)
        y1, y2 = max(0.0, y1), min(float(img_h), y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            stats["dropped_degenerate_box"] += 1
            continue
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{CLASSES.index(name)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        stats[f"kept_{name}"] += 1
    return lines


def convert_country(country):
    src = settings.RDD2022_DIR / country / "train"
    img_out = YOLO_DIR / "images" / country
    lbl_out = YOLO_DIR / "labels" / country
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    xml_files = sorted((src / "annotations" / "xmls").glob("*.xml"))
    for i, xml_path in enumerate(xml_files, 1):
        img_path = src / "images" / f"{xml_path.stem}.jpg"
        if not img_path.exists():
            stats["missing_image"] += 1
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size
            lines = voc_to_yolo_lines(xml_path, img_w, img_h, stats)
            dst = img_out / img_path.name
            if max(img_w, img_h) > MAX_SIDE:
                scale = MAX_SIDE / max(img_w, img_h)
                im.convert("RGB").resize((round(img_w * scale), round(img_h * scale)),
                                         Image.Resampling.LANCZOS).save(dst, quality=95)
                stats["images_resized"] += 1
            else:
                shutil.copy2(img_path, dst)

        (lbl_out / f"{xml_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["images"] += 1
        stats["images_with_boxes" if lines else "images_background"] += 1
        if i % 1000 == 0:
            print(f"  {country}: {i}/{len(xml_files)}")

    return dict(sorted(stats.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="+", default=None,
                        help="default: every country with a train folder")
    args = parser.parse_args()
    countries = args.countries or sorted(
        p.name for p in settings.RDD2022_DIR.iterdir() if (p / "train").is_dir())

    report = {}
    for country in countries:
        print(f"Converting {country}...")
        report[country] = convert_country(country)
        print(f"  {country}: {json.dumps(report[country])}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"conversion_{'_'.join(countries)}.json"
    out.write_text(json.dumps({"classes": CLASSES, "max_side": MAX_SIDE, "countries": report}, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
