# SmartPole demo: "Test an image"

Run any road photo through SmartPole and show the result on the dashboard:
a photo sent on WhatsApp, or a random image from the RDD2022 test split
(with the real answer next to the model's answer).

Test images go through exactly the same path as a photo from the ESP32-CAM,
but they are stored as **manual tests** (pole ID `MANUAL`, source `manual`),
so they never mix with real camera data.

---

## 1. Open the project in VS Code

1. **File → Open Folder…** → choose `C:\Users\Karthikeya\Dev\SmartPole`.
2. Press **Ctrl+Shift+P**, type **Python: Select Interpreter**, and pick the
   one marked **`.venv`** (Python 3.12). Needs the Python extension.
3. Open a terminal with **Ctrl+`** (the key left of 1). Click the **+** in the
   terminal panel two more times, so you have **three terminals**. Right-click
   each tab → **Rename** → call them `server`, `worker`, `dashboard`.

The commands below call `.venv\Scripts\python.exe` directly, so they work
even if the venv is not "activated" in that terminal.

## 2. Start the three parts (one per terminal, in this order)

**Terminal `server`** - receives images (from the ESP32 or from the demo):
```powershell
.venv\Scripts\python.exe -m server.app
```
Ready when it prints `Uvicorn running on http://0.0.0.0:8000`.

**Terminal `worker`** - runs YOLO on every new image:
```powershell
.venv\Scripts\python.exe -m worker.worker
```
Ready when it prints `waiting for pending images`. It warms up the GPU at
start, so even the first image is fast.

**Terminal `dashboard`** - the camera dashboard and the test page:
```powershell
.venv\Scripts\streamlit.exe run dashboard\app.py
```
The browser opens at **http://localhost:8501**. (The very first time,
Streamlit may ask for an email in the terminal: just press Enter.)

To stop any part: click its terminal and press **Ctrl+C**.

## 3. Test a photo sent on WhatsApp

**Save the photo**
- *WhatsApp Desktop:* open the photo → click the **download / save** icon →
  save it into `SmartPole\demo_images\`.
- *WhatsApp Web:* open the photo → **download** icon → it lands in
  `Downloads`. Move it into `SmartPole\demo_images\` (or upload it straight
  from Downloads).

`demo_images\` is excluded from git, so your teacher's photos never end up
on GitHub. JPEG and PNG both work. Large, sideways (portrait) phone photos
are fine: they are turned upright and shrunk automatically.

**On the dashboard**
1. Open the **🧪 Test images** tab.
2. Click **Upload** (or drag the files onto the box) and choose one or
   **several** photos (Ctrl+click to select more than one).
3. Click **Test N uploaded image(s)**.

**Or in the VS Code terminal** (a 4th terminal, while the other three run).
Use quotes, because WhatsApp file names contain spaces:
```powershell
.venv\Scripts\python.exe -m tools.test_image "demo_images\WhatsApp Image 2026-10-09 at 10.15.32.jpeg"
```

You see the original and the model's prediction side by side, a table of
detections (class, confidence, box), and the processing time. There is **no
ground truth** for a WhatsApp photo: nobody has labelled it, so there is no
correct answer to compare with. The dashboard says so.

## 4. Pick random test images (any country)

On the **🧪 Test images** tab, choose a **Country** and **How many**
(1-12), then click **🎲 Pick N random test images**. They come from that
country's **test split**: the 10% of its images held out with a fixed seed,
each with ground-truth labels.

| Country | What it tests |
|---|---|
| **India** | the roads the model was trained on (fair test of the model) |
| Japan, Czech, Norway, United_States, China_Drone, China_MotorBike | **generalisation**: roads, cameras and markings the model has never seen |

The dashboard says this on screen: **the current model was trained on India
only, so lower accuracy on other countries is expected**. In our test, the
India model found 0 of 18 damages on 6 Japan images: it does see something
there, but never above 22 % confidence, below the 0.25 cut-off. That gap is
exactly why Phase 7 trains on all countries.

You get:
* a **grid**: each image with the model's boxes and the **ground truth in
  green**, captioned e.g. "found 2 of 3 · 1 false alarm(s)";
* **totals**: real damages, found, missed, false alarms, plus recall and
  precision for the batch (a small sample, not the official evaluation);
* a **summary table** with one row per image and a TOTAL row;
* **"Show one image in detail"**: the three-picture view below.

The detail view shows three pictures:

| Original | Model prediction | Ground truth |
|---|---|---|
| the photo | boxes the model drew | **green** boxes from the human labels (the real answer) |

and these numbers:

| Number | Meaning |
|---|---|
| Damages found by model | boxes with confidence ≥ 0.25 |
| Model processing time | YOLO inference only (about 40-60 ms on the GPU) |
| Total | upload + waiting in the queue + inference (about 1 s) |
| Real damages found | e.g. "3 of 4": a real damage counts as found when the model predicted the **same class** with a box overlapping it by **IoU ≥ 0.5** (overlap area ÷ combined area), the same rule mAP50 uses |
| False alarms | model boxes that did not match any real damage |

About 1 in 10 test images has no damage at all ("none (clean road)"). That
is deliberate: it checks the model does not invent damage.

Terminal version: `.venv\Scripts\python.exe -m tools.test_image --random`

If you upload a dataset image yourself, the dashboard tells you which split
it came from. A **training** image is not a fair test (the model learned
from it), and the dashboard warns about that. Ground truth is found by file
name, so a renamed dataset image is treated like an unknown photo.

## 5. Simulate escalation with the teacher's images

After testing a batch (uploaded photos or random test images), scroll down
to **Simulate escalation with these images**. Needs the authority portal
running in a 4th terminal:
```powershell
.venv\Scripts\streamlit.exe run dashboard\authority.py --server.port 8502
```

1. Choose up to **6 images**. Each becomes the fixed camera view of one
   **DEMO pole** (DEMO-01, DEMO-02, ...).
2. Keep or untick the disturbances:
   * **Passing vehicles**: now and then a vehicle covers part of the view
     for one frame (the damage may be hidden in that frame)
   * **Lighting changes**: brightness varies, darker at dawn and dusk
   * **Camera jitter**: the view shifts by a few pixels (wind on the pole)
   * **Night**: photos from 20:00 to 06:00 are very dark
3. Click **▶ Start escalation demo**, then **Open authority portal**.

Once per demo hour (**1 second = 1 hour**, a day takes 24 s) every image is
sent again as its pole's camera photo, through the real server, the real
YOLO worker and the real escalation engine with its real thresholds. On the
portal you watch: **Watching** (first seconds) → **Monitoring** (after 5
sightings in 3 different hours, about 6-10 s) → **Repair requested** (3 demo
days later, about 75 s). A 4-day demo takes about 100 s.

What to expect, honestly:
* **Size is "unknown"** for DEMO poles. A photo from the internet or a
  dashcam has no known scale (it shows many metres of road in perspective),
  so the system does not pretend to know the damage's area in cm². It
  escalates on proven persistence alone: **Repair requested**. Urgent needs
  a real size (a calibrated pole) or real growth.
* **Growth is ~0 %.** The same photo is replayed, so nothing can grow. The
  portal says so on every DEMO damage.
* **Night frames and hidden frames** simply have fewer detections. The rules
  count hours with sightings, so the damage is still confirmed.
* **A Japan photo may never escalate**, because the India-only model does not
  detect anything there. That is a real result, not a bug.

DEMO poles are marked **DEMO** everywhere, run on their own fast clock, and
are ignored by the camera escalation service. **🧹 Clear demo data** (on the
dashboard or in the portal's sidebar) deletes all DEMO poles, their requests
and the replayed photos. Camera poles and normal manual tests are not
touched, and normal manual tests never create repair requests.

Terminal version:
```powershell
.venv\Scripts\python.exe -m escalation.demo_replay --random Czech 3
.venv\Scripts\python.exe -m escalation.demo_replay --files "demo_images\photo1.jpg" "demo_images\photo2.jpg"
.venv\Scripts\python.exe -m escalation.demo_replay --clear
```

## 6. If something goes wrong

| Message | Fix |
|---|---|
| "The server is not running" | start it in the `server` terminal |
| "The worker is not running, so image #N is saved but waiting" | start it in the `worker` terminal; the image is processed as soon as it starts |
| "Not a readable image" | the file is not a JPEG/PNG (e.g. a HEIC iPhone photo). Send it again on WhatsApp as a normal photo, which converts it to JPEG |
| A detection appears in one test but not another | detections close to the 0.25 confidence cut-off can flip when the same photo is compressed again |

## 7. How it works (for your teacher)

> The dashboard does not run the model itself. It sends the photo to our
> FastAPI server's `/upload` endpoint, exactly the way the ESP32-CAM does,
> but marked as a manual test so it is kept apart from camera data. The
> server checks it is a real JPEG, saves it, records it in SQLite as
> "pending" and replies immediately. A separate worker process picks up the
> pending image, runs our YOLO11 model on the GPU, draws the boxes, and
> saves the detections in the database in one transaction, so an image can
> never be processed twice. The dashboard reads the result from the
> database. For RDD2022 test images it also draws the human-labelled boxes
> and counts how many real damages the model found, using the same overlap
> rule (IoU ≥ 0.5) as the mAP50 metric.

Be honest about the current model: it is the India-only yolo11n baseline
(validation mAP50 0.43). It was trained on dashboard-camera photos, so it
works best on similar road views. Training on all countries and
fine-tuning on our own ESP32-CAM photos (Phase 7) should improve it.
