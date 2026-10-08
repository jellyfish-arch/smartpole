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

# The server announces itself with a UDP broadcast so the ESP32 can find the
# laptop even when the hotspot gives the laptop a new IP address.
# Must match DISCOVERY_PORT in the firmware.
DISCOVERY_PORT = 50000
DISCOVERY_INTERVAL_S = 2.0

# Uploads larger than this are rejected. An SVGA JPEG is ~30-80 KB, so 2 MB
# leaves plenty of room while stopping accidental huge uploads.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024

# The dashboard shows the node as offline if no heartbeat arrived recently.
HEARTBEAT_TIMEOUT_S = 30

# ---- Worker ----
# While a training run is in progress, point at its last.pt (newest epoch).
# Once it has finished, use that run's best.pt (best validation mAP).
WORKER_WEIGHTS = TRAINING_RUNS_DIR / "india_yolo11n" / "weights" / "best.pt"
# "0" = the RTX 4060. Use "cpu" while a training run is using the GPU,
# so inference does not compete with training.
# Warning: with 15 GB of RAM, running the worker DURING training pushed
# Windows out of memory and crashed the training run (2026-10-09). Only run
# the worker while training if the training uses --workers 2 or fewer.
WORKER_DEVICE = "0"
WORKER_CONF = 0.25          # ignore detections below this confidence
WORKER_IMGSZ = 640          # same size the model was trained at
WORKER_POLL_S = 1.0         # how often to look for new pending images
WORKER_MAX_ATTEMPTS = 3     # give up on an image after this many failed tries
# The worker writes "I'm alive" to the database this often. If the newest
# note is older than WORKER_TIMEOUT_S, the dashboard reports it as stopped.
WORKER_HEARTBEAT_S = 5
WORKER_TIMEOUT_S = 15

# ---- Manual tests ("Test an image" demo mode) ----
# Programs on this laptop (dashboard, tools/test_image.py) reach the server here.
LOCAL_SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
# Manual test uploads are stored under this pole ID and with source='manual',
# so they never mix with real camera data or with growth tracking.
MANUAL_POLE_ID = "MANUAL"
# Phone photos are often 4000 px wide and several MB. They are shrunk to this
# long side before upload (the model works at 640 px anyway), which also
# keeps them under MAX_UPLOAD_BYTES.
MANUAL_MAX_SIDE = 1280
MANUAL_WAIT_S = 30          # how long to wait for the worker's result
