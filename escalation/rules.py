"""Every threshold of the escalation design in one place.

The reasons for each number are in docs/ESCALATION.md. Durations are in
"effective" seconds: real seconds multiplied by the demo speed (1 normally).
"""

HOUR = 3600
DAY = 24 * HOUR
WEEK = 7 * DAY

# ---- Levels ----
WATCHING = "WATCHING"
MONITORING = "MONITORING"
REPAIR_REQUESTED = "REPAIR_REQUESTED"
URGENT = "URGENT"
REPAIRED = "REPAIRED"
DISMISSED = "DISMISSED"
EXPIRED = "EXPIRED"

OPEN_LEVELS = (WATCHING, MONITORING, REPAIR_REQUESTED, URGENT)
CLOSED_LEVELS = (REPAIRED, DISMISSED, EXPIRED)
RANK = {WATCHING: 0, MONITORING: 1, REPAIR_REQUESTED: 2, URGENT: 3}   # levels only go up
PRIORITY = {URGENT: 1, REPAIR_REQUESTED: 2}                          # 1 = most urgent
LABEL = {WATCHING: "Watching", MONITORING: "Monitoring", REPAIR_REQUESTED: "Repair requested",
         URGENT: "URGENT", REPAIRED: "Repaired", DISMISSED: "Dismissed", EXPIRED: "Expired"}

# ---- Matching: is a new detection the same damage? ----
MATCH_IOU = 0.3                # boxes jitter between frames; 0.5 would split one pothole in two
REAPPEAR_IOU = 0.3             # overlap with a repaired damage -> "reappeared after repair"

# ---- WATCHING -> MONITORING (proof that it is real) ----
CONFIRM_MIN_SIGHTINGS = 5
CONFIRM_MIN_HOURS = 3          # sightings in at least this many different hours...
CONFIRM_MIN_SPAN = 6 * HOUR    # ...spread over at least this long
CONFIRM_MIN_MEAN_CONF = 0.35

# ---- WATCHING -> EXPIRED ----
WATCH_EXPIRE = 24 * HOUR       # not seen for this long while still unproven

# ---- MONITORING -> request ----
MONITOR_PERIOD = 3 * DAY       # measure this long before a normal (non-urgent) request
NOT_SEEN_FLAG = 3 * DAY        # open damage not seen this long gets a warning flag

# ---- Measurement ----
SIZE_MEDIAN_OF = 10            # size = median of the last N sightings
GROWTH_WINDOW = 24 * HOUR      # growth: first 24 h of monitoring vs last 24 h
GROWTH_MIN_SAMPLES = 3         # sightings needed in each window
GROWTH_MIN_GAP = 2 * DAY       # windows' midpoints must be at least this far apart
GROWTH_CONFIDENCE = 2          # growth counts only if (growth - 2 x its standard error) > threshold

# ---- Severity: the project's fixed definition (project brief, Phase 6) ----
LOW_AREA_CM2 = 100             # area < 100 cm2 -> Low
HIGH_AREA_CM2 = 500            # area > 500 cm2 -> High
CRACK_LOW_CM = 30              # crack < 30 cm -> Low
HIGH_GROWTH_PER_WEEK = 0.10    # growth > 10 % per week -> High
CRACK_CLASSES = {"D00", "D10"} # judged by length; D20 and D40 by area


def severity(class_name, area_cm2, length_cm, growth_low):
    """Return 'Low', 'Medium', 'High', or None when the size is unknown.

    growth_low is the LOW end of the measured growth range (growth minus its
    measurement uncertainty), so frame-to-frame jitter cannot fake growth.
    Growth works even without calibration (it is a ratio), so a fast-growing
    damage is High even when its size is unknown.
    """
    if growth_low is not None and growth_low > HIGH_GROWTH_PER_WEEK:
        return "High"
    if class_name in CRACK_CLASSES:
        if length_cm is None:
            return None
        return "Low" if length_cm < CRACK_LOW_CM else "Medium"
    if area_cm2 is None:
        return None
    if area_cm2 > HIGH_AREA_CM2:
        return "High"
    return "Low" if area_cm2 < LOW_AREA_CM2 else "Medium"


def high_by_size(class_name, area_cm2):
    """High because of SIZE alone (not growth): triggers URGENT immediately."""
    return class_name not in CRACK_CLASSES and area_cm2 is not None and area_cm2 > HIGH_AREA_CM2
