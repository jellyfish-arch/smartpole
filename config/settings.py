"""Central settings for SmartPole.

Every path and port used anywhere in the project is defined here, so the
rest of the code never contains a hard-coded C:\\ string. Paths are built
with pathlib relative to the repository root, which means the project keeps
working if the folder is moved or cloned onto another laptop.
"""

from pathlib import Path

# Repository root = the folder that contains config/, server/, worker/, ...
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- Runtime data (gitignored) ----
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"        # raw JPEGs from the ESP32
ANNOTATED_DIR = DATA_DIR / "annotated"    # images with YOLO boxes drawn on
RESULTS_DIR = DATA_DIR / "results"        # one JSON file per processed image
DB_PATH = DATA_DIR / "smartpole.db"       # SQLite database

# ---- Dataset (gitignored, ~13 GB zipped) ----
DATASET_DIR = PROJECT_ROOT / "dataset"
RDD2022_ARCHIVE = DATASET_DIR / "archive" / "21431547.zip"   # figshare download
RDD2022_DIR = DATASET_DIR / "RDD2022"                        # extracted countries

# ---- Training ----
TRAINING_RUNS_DIR = PROJECT_ROOT / "training" / "runs"

# ---- Network ----
# 0.0.0.0 means "listen on every network interface", so the ESP32 on the
# phone hotspot can reach the server. 127.0.0.1 would only accept
# connections from this laptop itself.
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
