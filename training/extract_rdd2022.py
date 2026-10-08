"""Unpack the RDD2022 figshare download into dataset/RDD2022/<Country>/.

Run from the repository root:
    python -m training.extract_rdd2022

The figshare file (21431547.zip) is a zip inside a zip inside a zip:
    21431547.zip
      RDD2022_released_through_CRDDC2022.zip
        RDD2022/Japan.zip, RDD2022/India.zip, ...   (one zip per country)

This script peels the layers one at a time and deletes each temporary
zip as soon as it has been unpacked, so the disk only ever holds the
original download plus the extracted images. A country that is already
extracted is skipped, so the script is safe to re-run if it gets
interrupted.
"""

import shutil
import zipfile
from pathlib import Path

from config import settings

INNER_ZIP_NAME = "RDD2022_released_through_CRDDC2022.zip"


def country_done(country: str) -> bool:
    """A country counts as extracted once its train/images folder has files."""
    images = settings.RDD2022_DIR / country / "train" / "images"
    return images.is_dir() and any(images.iterdir())


def main() -> None:
    out_dir = settings.RDD2022_DIR
    tmp_dir = settings.DATASET_DIR / "_extract_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Layer 1: the figshare zip -> the big RDD2022 zip plus three text files.
    inner_zip = tmp_dir / INNER_ZIP_NAME
    with zipfile.ZipFile(settings.RDD2022_ARCHIVE) as outer:
        for name in outer.namelist():
            if name.endswith(".txt") or name.endswith(".pbtxt"):
                outer.extract(name, out_dir)
        if not inner_zip.exists():
            print(f"Extracting {INNER_ZIP_NAME} (13 GB, takes a few minutes)...")
            outer.extract(INNER_ZIP_NAME, tmp_dir)

    # Layer 2 and 3: the big zip -> one zip per country -> image folders.
    with zipfile.ZipFile(inner_zip) as rdd:
        for member in rdd.namelist():
            if not member.endswith(".zip"):
                continue
            country = member.split("/")[-1].removesuffix(".zip")
            if country_done(country):
                print(f"{country}: already extracted, skipping")
                continue

            print(f"{country}: unpacking...")
            country_zip = Path(rdd.extract(member, tmp_dir))
            with zipfile.ZipFile(country_zip) as cz:
                # Most country zips contain "<Country>/train/...". If one
                # instead starts directly at "train/...", give it its own folder.
                if cz.namelist()[0].startswith(f"{country}/"):
                    cz.extractall(out_dir)
                else:
                    cz.extractall(out_dir / country)
            # The country zip is no longer needed once its files are out.
            # Deleting it now frees up to 10 GB (mostly Norway).
            country_zip.unlink()

            if not country_done(country):
                raise RuntimeError(
                    f"{country}: expected {out_dir / country / 'train' / 'images'} "
                    "after extraction. The zip layout is different from what "
                    "this script assumes; check it by hand."
                )
            n = len(list((out_dir / country / "train" / "images").iterdir()))
            print(f"{country}: done, {n} train images")

    shutil.rmtree(tmp_dir)
    print(f"\nAll countries extracted to {out_dir}")


if __name__ == "__main__":
    main()
