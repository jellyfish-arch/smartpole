"""Train a YOLO11 model on a converted RDD2022 split.

Run from the repository root:
    python -m training.train                                   # India baseline, yolo11n
    python -m training.train --name india_yolo11n --resume     # continue after a crash or reboot
    python -m training.train --data dataset/yolo/all6.yaml --name all6_yolo11n

Outputs go to training/runs/<name>/ (gitignored):
    weights/best.pt   best model on the validation set  <- BACK THIS UP (see README)
    weights/last.pt   latest epoch, used to resume
    results.csv, results.png, confusion_matrix.png, PR curves, args.yaml

Starting values: batch=8, imgsz=640 on the 8 GB RTX 4060. If CUDA runs out
of memory, lower --batch before lowering --imgsz: cracks are thin, and a
smaller image size removes exactly the detail needed to see them.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from config import settings

PRETRAINED_DIR = settings.PROJECT_ROOT / "training" / "weights"


def keep_windows_awake():
    """Stop Windows from going to sleep while this process runs.

    Same request a video player makes. It does not change any power
    settings and ends by itself when training finishes. Closing the laptop
    lid still puts it to sleep.
    """
    import sys
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=settings.DATASET_DIR / "yolo" / "india.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="pretrained COCO weights to start from")
    parser.add_argument("--name", default="india_yolo11n")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=50,
                        help="stop early if val mAP has not improved for this many epochs")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="continue runs/<name>/weights/last.pt")
    args = parser.parse_args()

    run_dir = settings.TRAINING_RUNS_DIR / args.name
    keep_windows_awake()

    if args.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"Nothing to resume: {last} does not exist")
        print(f"Resuming from {last} with workers={args.workers}")
        # Every setting (data, epochs, batch...) is read back from the checkpoint.
        # Ultralytics lets a few memory-related ones change on resume; we pass
        # --workers, because each dataloader worker is a full Python process
        # and fewer of them is the quickest fix if the laptop runs out of RAM.
        YOLO(str(last)).train(resume=True, workers=args.workers)
        return

    if run_dir.exists():
        raise SystemExit(f"{run_dir} already exists. Use --resume, or pick a new --name.")

    # Start from COCO-pretrained weights (transfer learning). The network
    # already knows edges and textures, so it learns road damage much faster
    # than from random weights. Ultralytics downloads the file (~5 MB) once.
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(PRETRAINED_DIR / args.model))

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=0,                       # the RTX 4060; fail loudly rather than fall back to CPU
        project=str(settings.TRAINING_RUNS_DIR),
        name=args.name,
        seed=42,
        save_period=10,                 # also keep a checkpoint every 10 epochs
        plots=True,
    )


# Required on Windows: dataloader workers start fresh Python processes that
# re-import this file. Without the guard, each worker would start training too.
if __name__ == "__main__":
    main()
