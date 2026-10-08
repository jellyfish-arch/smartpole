# Automatic escalation to the authority

SmartPole nodes sit on street-light poles; each watches its own patch of road.
The system follows every damage over time and, when it is proven real and
serious enough, raises a repair request to the municipal authority on its own.

```
camera photos -> YOLO detections -> TRACKING (same place = same damage)
              -> MEASUREMENT (size, growth) -> ESCALATION LEVELS -> authority
```

## 1. The levels

| Level | Meaning | Who moves it here |
|---|---|---|
| **WATCHING** | Seen, not yet proven real | system |
| **MONITORING** | Confirmed real; being measured for size and growth | system |
| **REPAIR REQUESTED** | Repair request sent (normal priority) | system |
| **URGENT** | Repair request sent or upgraded (high priority) | system |
| *Closed:* **REPAIRED** | Authority fixed it | authority |
| *Closed:* **DISMISSED** | Authority says it is not road damage (e.g. a manhole cover) | authority |
| *Closed:* **EXPIRED** | Was only ever WATCHING, then disappeared (vehicle, shadow, bad frame) | system |

Levels only go **up** automatically. The system never lowers a level on its
own, because a smaller measurement is more likely a partly blocked view than
a pothole healing itself. Only the authority closes a damage.

## 2. The rules

| From → To | Rule | Why |
|---|---|---|
| (nothing) → WATCHING | A detection that matches no open damage at that pole | Anything new deserves a look, nothing more |
| WATCHING → MONITORING | ≥ **5** sightings, in ≥ **3** different hours, spread over ≥ **6 h**, average confidence ≥ **0.35** | A passing vehicle or a bad frame fills one hour at most. A shadow moves within the hour, so it does not stay at the same position across 3 hours. Real damage stays put all day. Night gaps don't matter: the rule counts hours with sightings, not consecutive frames |
| WATCHING → EXPIRED | Not seen for **24 h** | Short-lived objects clean themselves up |
| MONITORING → URGENT | Severity **High by size** (area > 500 cm²) → **immediately** | A big pothole is a safety risk; it has already been proven real, so do not wait |
| MONITORING → URGENT | After **3 days** of monitoring, growth > **10 % per week** (low end of its error bar) | Fast growth is the project's definition of High severity |
| MONITORING → REPAIR REQUESTED | After **3 days**, severity **Medium** (100–500 cm², or crack ≥ 30 cm), or size unknown | 3 days gives 3 daylight periods of measurements: enough for a reliable size and a first growth figure, and the authority still hears within the week |
| MONITORING stays | Severity **Low** and not growing | Small, stable cracks go on the authority's watch list, not its repair queue |
| REPAIR REQUESTED → URGENT | Becomes High (size or growth) | Requests are upgraded when things get worse |
| any open → REPAIRED / DISMISSED | Authority action in the authority portal | A human closes it |

Open damages not seen for **3 days** are flagged "not seen recently" (maybe
repaired without telling us, or the view is blocked), but not closed.

### Severity (the project's fixed definition)

| Severity | Rule |
|---|---|
| Low | area < 100 cm², or crack < 30 cm |
| Medium | 100–500 cm² (or crack ≥ 30 cm) |
| High | > 500 cm², or growth > 10 % per week |

Cracks (D00 longitudinal, D10 transverse) are judged by **length**; alligator
cracking (D20) and potholes (D40) by **area**.

### Measurements are robust, not single-frame

* **Size** = the **median** of the last 10 sightings. One frame with a car
  covering half the pothole cannot change it.
* **Growth** = median size in the first 24 h of monitoring vs the last 24 h,
  as % per week. Because both are measured at the same spot by the same fixed
  camera, the *ratio* is valid even before calibration.
* **Growth comes with an error bar**, e.g. "+14 % ± 3 % per week". The
  error is measured from how much the sightings jitter (2 standard errors).
  Growth counts as High only if **even the low end** is above 10 %/week.
  *Why:* the first simulation showed a perfectly stable crack measuring
  "+8 %/week" from frame-to-frame jitter alone, close to the 10 % line. With
  the error bar it reads "−1 % ± 6 %", clearly not growing, while a real
  growing pothole reads "+13 % ± 3 %" and is escalated.

### Real size: calibrated, estimated, or unknown

| Basis | When | Shown as |
|---|---|---|
| **calibrated** | The pole has a homography file (Phase 6 calibration tool) | "312 cm²" |
| **estimate** | No calibration, but the installer entered the approximate ground area the camera sees (`approx_view_m` in `config/poles.json`) | "≈ 310 cm² (uncalibrated estimate)" |
| **unknown** | Neither | "size unknown (uncalibrated)"; escalates on persistence and growth only |

Pixel area is never presented as real-world area.

## 3. Matching ("is this the same damage?")

The camera never moves, so the same place in the image is the same place on
the road. A new detection belongs to an open damage at the same pole when
their boxes overlap with **IoU ≥ 0.3** (boxes jitter between frames, so 0.5
would split one pothole into several). Boxes are stored as fractions of the
image, so changing camera resolution (QVGA → VGA) does not break tracking.

* A **dismissed** damage keeps "absorbing" detections at its spot, so a
  manhole cover that the authority dismissed never comes back as a new request.
* A **repaired** damage does not: damage seen there again is new damage
  (marked "reappeared after repair").

## 4. What never creates a request

* **Manual test uploads** ("Test an image", pole `MANUAL`, source `manual`):
  the engine only reads `source = 'camera'` images.
* **Simulated poles** (`SIM-…`) are labelled SIMULATION everywhere and can be
  hidden in the authority portal.

## 5. Demo speed without faking the logic

All time rules run on a clock that can be sped up: `--speed 1440` makes one
day last one minute (like a time-lapse). The rules, thresholds and code are
unchanged; only the length of a day is. The authority portal shows a banner
whenever demo speed is on.

* **Live camera demo:** `python -m escalation.service --speed 1440`. Point
  the camera at a printed "pothole": WATCHING → MONITORING in ~1.5 min →
  REPAIR REQUESTED after 3 "days" (3 min). Swap in a bigger print and it
  turns URGENT from the growth rule.
* **Scripted demo:** `python -m escalation.simulate` replays three weeks on
  two simulated poles in about a minute, using the real engine and real
  thresholds: a growing pothole, a passing vehicle, a stable small crack, a
  big pothole and a manhole cover (for you to dismiss live).

## 6. Notifying the authority

**Prototype:** every escalation writes a notification into an **outbox**
(database table + a JSON file in `data/outbox/`) in the same transaction as
the level change, and the authority portal shows it.

**Real deployment:** a small sender process would deliver each outbox entry
to the municipality's complaint/ticketing system through its API (or by
email and SMS to the ward engineer), retry until it gets an acknowledgement,
and store the authority's ticket number. Writing to the outbox first means
no request is lost if the network or the municipal server is down.

Each request carries: request ID, priority, pole ID, GPS location (with a map
link), where in the camera view the damage is, type, size (with its basis),
growth, first/last seen, and the evidence photo with the box drawn.

## 7. How this fits Phase 6

| Phase 6 item | Status |
|---|---|
| Position matching and growth tracking | **Built here** (`cv/tracking.py`, `escalation/engine.py`), stored in the `tracks` / `track_observations` tables |
| Calibration (homography, cm) | The engine already reads `data/calibration/<pole>.json`; the click-four-corners tool that writes it is still Phase 6 |
| Severity in cm | Built here, following the Phase 6 table exactly |
| Reference-frame differencing | Still Phase 6. It can later become an extra confirmation condition ("YOLO *and* the scene changed") |
| Size-over-time charts | In the authority portal per damage; the camera dashboard gets them in Phase 6 |

## 8. Running it

| What | Command (from the project root) |
|---|---|
| Escalation service (with server + worker running) | `.venv\Scripts\python.exe -m escalation.service` |
| Same, demo speed (1 day = 1 min) | `.venv\Scripts\python.exe -m escalation.service --speed 1440 --fresh` |
| Authority portal | `.venv\Scripts\streamlit.exe run dashboard\authority.py --server.port 8502` → http://localhost:8502 |
| Scripted 3-week demo | `.venv\Scripts\python.exe -m escalation.simulate` |
| Automated rule checks (32) | `.venv\Scripts\python.exe -m escalation.check_scenarios` |

Use `--fresh` whenever you change `--speed`: timestamps recorded at one speed
cannot be compared at another. It only clears camera tracking data, never
photos or detections.

**10-minute viva plan:** (1) start the portal and run `escalation.simulate`
(~40 s) while explaining the levels; (2) open the URGENT pothole: show the
size chart and the history (why it was escalated); (3) dismiss the manhole
cover live; (4) optionally show the live camera at `--speed 1440` pointed at
a printed pothole: confirmed after ~1 min, requested ~3 min later.

## 9. Known limits (honest list)

* A shadow cast at the same spot at the same time **every day** could
  eventually be confirmed. The authority's Dismiss suppresses it for good,
  and frame differencing (Phase 6) can be added as a second signal.
* The uncalibrated estimate ignores perspective (far parts of the view look
  smaller), so it is only an estimate until calibration.
* Location is per pole (GPS) plus position in the camera view, not GPS per
  damage. Each pole only sees a few metres of road, so that is precise
  enough to find the damage.
