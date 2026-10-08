"""Inspect RDD2022 before converting anything.

Run from the repository root:
    python -m training.inspect_dataset

For every country's annotated "train" folder it counts:
  - every label name that appears in the XML files (not just the 4 targets)
  - images whose XML has no objects at all
  - images that would have no objects left after dropping non-target labels
  - image sizes, missing image/XML pairs and broken boxes

Results are printed and saved to training/reports/rdd2022_inspection.md
(readable) and .json (for later scripts). The "test" folders are skipped:
they have no annotations.
"""

import json
import xml.etree.ElementTree as ET
from collections import Counter

from config import settings

TARGET_CLASSES = ["D00", "D10", "D20", "D40"]
REPORT_DIR = settings.PROJECT_ROOT / "training" / "reports"


def inspect_country(country_dir):
    xml_dir = country_dir / "train" / "annotations" / "xmls"
    img_dir = country_dir / "train" / "images"
    xml_stems = {p.stem for p in xml_dir.glob("*.xml")}
    img_stems = {p.stem for p in img_dir.glob("*.jpg")}

    labels = Counter()
    sizes = Counter()
    no_objects = 0
    no_target_objects = 0
    bad_boxes = 0

    for xml_path in sorted(xml_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        w = int(float(size.findtext("width", "0"))) if size is not None else 0
        h = int(float(size.findtext("height", "0"))) if size is not None else 0
        sizes[f"{w}x{h}"] += 1

        names = []
        for obj in root.iter("object"):
            name = (obj.findtext("name") or "").strip()
            names.append(name)
            box = obj.find("bndbox")
            if box is not None:
                x1, y1, x2, y2 = (float(box.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax"))
                if x2 <= x1 or y2 <= y1:
                    bad_boxes += 1
        labels.update(names)
        if not names:
            no_objects += 1
        if not any(n in TARGET_CLASSES for n in names):
            no_target_objects += 1

    return {
        "xml_files": len(xml_stems),
        "images": len(img_stems),
        "images_without_xml": len(img_stems - xml_stems),
        "xml_without_image": len(xml_stems - img_stems),
        "labels": dict(labels.most_common()),
        "images_with_no_objects": no_objects,
        "images_with_no_target_objects": no_target_objects,
        "boxes_with_zero_or_negative_size": bad_boxes,
        "image_sizes": dict(sizes.most_common()),
    }


def write_markdown(results, path):
    countries = list(results)
    all_labels = Counter()
    for r in results.values():
        all_labels.update(r["labels"])

    lines = ["# RDD2022 inspection (annotated train folders only)", ""]
    lines += ["## Label counts (number of boxes)", ""]
    lines += ["| Label | Kept? | " + " | ".join(countries) + " | Total |"]
    lines += ["|---|---|" + "---|" * len(countries) + "---|"]
    for label, total in all_labels.most_common():
        keep = "**yes**" if label in TARGET_CLASSES else "dropped"
        row = [str(results[c]["labels"].get(label, 0)) for c in countries]
        lines += [f"| {label} | {keep} | " + " | ".join(row) + f" | {total} |"]

    kept = sum(v for k, v in all_labels.items() if k in TARGET_CLASSES)
    dropped = sum(v for k, v in all_labels.items() if k not in TARGET_CLASSES)
    lines += ["", f"Boxes kept: **{kept}**. Boxes dropped (non-target labels): **{dropped}** "
              f"({dropped / max(kept + dropped, 1):.1%}).", ""]

    lines += ["## Images per country", ""]
    lines += ["| Country | Images | XMLs | Unpaired | No objects in XML | No target objects | Bad boxes | Most common size |"]
    lines += ["|---|---|---|---|---|---|---|---|"]
    for c, r in results.items():
        unpaired = r["images_without_xml"] + r["xml_without_image"]
        common = ", ".join(f"{s} ({n})" for s, n in list(r["image_sizes"].items())[:2])
        lines += [f"| {c} | {r['images']} | {r['xml_files']} | {unpaired} | "
                  f"{r['images_with_no_objects']} | {r['images_with_no_target_objects']} | "
                  f"{r['boxes_with_zero_or_negative_size']} | {common} |"]
    lines += ["", "*No target objects* = images that end up with an empty label file "
              "once non-target labels are dropped.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    countries = sorted(p for p in settings.RDD2022_DIR.iterdir() if (p / "train").is_dir())

    results = {}
    for country_dir in countries:
        print(f"Inspecting {country_dir.name}...")
        results[country_dir.name] = inspect_country(country_dir)

    (REPORT_DIR / "rdd2022_inspection.json").write_text(json.dumps(results, indent=2))
    write_markdown(results, REPORT_DIR / "rdd2022_inspection.md")
    print((REPORT_DIR / "rdd2022_inspection.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
