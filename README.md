# SmartPole

**Automated Pot Hole Detection and Location Mapping System** (VIT-AP University).

A fixed ESP32-CAM photographs one patch of road on its own schedule and
pushes each JPEG over Wi-Fi to a FastAPI server on the laptop. A separate
YOLO11 worker detects cracks and potholes, SQLite stores the results, and a
Streamlit dashboard shows them. Because the camera never moves, the system
can measure defects in centimetres and track how each one grows over time.

```
ESP32-CAM ──HTTP POST JPEG──▶ FastAPI ingest ──▶ YOLO worker ──▶ SQLite ──▶ Streamlit
```

## Folder layout

| Folder | Contents |
|---|---|
| `firmware/smartpole_node/` | The single Arduino sketch for the ESP32-CAM |
| `server/` | FastAPI ingest app and UDP discovery broadcaster |
| `worker/` | YOLO worker process |
| `cv/` | Homography, frame differencing, defect tracking |
| `training/` | Dataset inspection, conversion, training, evaluation |
| `dashboard/` | Streamlit app |
| `config/` | `settings.py`: every path and port in one place |
| `data/` | Runtime files: uploads, annotated images, database (not in git) |

The Python folders contain an empty `__init__.py` so they are packages.
Always run scripts **from the repository root** (for example
`python -m server.app`) so that `from config import settings` works everywhere.

## Setup on a new Windows laptop

Requires Python 3.12 and an NVIDIA GPU.

```powershell
py install 3.12                      # Python install manager (skip if already installed)
py -V:3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python check_env.py
```

If PowerShell refuses to run `Activate.ps1`, run this once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. It lets you run
scripts you created on your own machine. Scripts downloaded from the
internet still need a signature.

`requirements.txt` points pip at PyTorch's CUDA 13.2 package index. That
index matters: plain `pip install torch` from PyPI on Windows installs a
**CPU-only** build, and training would not use the GPU.

## Firmware (ESP32-CAM)

1. Copy `firmware/smartpole_node/secrets.example.h` to `secrets.h` and fill in
   the hotspot name and password (`secrets.h` is gitignored).
2. Open `firmware/smartpole_node/smartpole_node.ino` in Arduino IDE.
3. Tools: Board **ESP32 Dev Module**, PSRAM **Enabled**, Partition Scheme
   **Huge APP (3MB No OTA/1MB SPIFFS)**, Upload Speed **115200**, Port = the
   MB board's COM port.
4. Upload, then open Serial Monitor at 115200 baud.

## Server

```powershell
python -m server.app                  # listens on 0.0.0.0:8000, broadcasts on UDP 50000
python -m server.send_test_image      # second terminal: fake ESP32 uploads
python -m server.listen_discovery     # second terminal: check the UDP broadcast
```

## Worker and dashboard

Each in its own terminal, alongside the server:

```powershell
python -m worker.worker               # YOLO on new uploads (weights/device in config/settings.py)
streamlit run dashboard/app.py        # http://localhost:8501, refreshes every 3 s
python -m worker.check_integrity      # any time: proves no image was processed twice
```

Don't run the worker on the GPU while a training run is going: on this
laptop (15 GB RAM) the extra PyTorch process ran Windows out of memory and
crashed training. Use `--device cpu` and train with `--workers 2` if both
must run together.

## Demo: test any image

The dashboard's **Test an image** tab (and `python -m tools.test_image <image>`)
runs any road photo, such as one sent on WhatsApp or a random RDD2022 test image,
through the same server and worker as a camera photo. Step-by-step guide:
[docs/DEMO.md](docs/DEMO.md).

## Dataset and training

```powershell
python -m training.extract_rdd2022                         # unzip figshare archive (once)
python -m training.inspect_dataset                         # label counts -> training/reports/
python -m training.convert_voc_to_yolo --countries India   # VOC XML -> YOLO txt
python -m training.split_dataset --countries India --name india
python -m training.train                                   # India baseline, yolo11n
python -m training.train --name india_yolo11n --resume     # continue an interrupted run
```

## Backing up model weights (do this every time training finishes)

`.pt` files are excluded from git. They are too large for GitHub, and
losing them is how the previous version of this project died. Every
training run writes its weights to
`training/runs/<run name>/weights/best.pt` (and `last.pt`).

After **every** run that you want to keep:

1. Install **Google Drive for desktop** (drive.google.com/download) and sign
   in. Your Drive then appears as a drive letter, usually `G:\My Drive`.
2. Create a folder there, for example `G:\My Drive\SmartPole\weights\`.
3. Copy `best.pt` into it and **rename it to describe the run**, for example
   `2026-10-20_yolo11n_india_640_best.pt`. Every run produces a file named
   `best.pt`, so without a rename the next copy overwrites the previous one.
4. Next to it, also copy that run's `args.yaml` and `results.csv`. Together
   they record exactly how the model was trained and how well it did.
5. Open drive.google.com in a browser and check that the file is actually
   there and has the right size. Don't trust the sync icon alone.

Copy from PowerShell (adjust the run name and the date):

```powershell
Copy-Item training\runs\india_yolo11n\weights\best.pt "G:\My Drive\SmartPole\weights\2026-10-20_yolo11n_india_640_best.pt"
```

A second copy on a USB stick or a teammate's laptop is cheap insurance.
