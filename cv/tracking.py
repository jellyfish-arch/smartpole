"""Position matching: decide which new detections are the same damage.

The camera never moves, so the same place in the image is the same place on
the road. A detection belongs to a known damage when their boxes overlap
enough (IoU). Boxes are fractions of the image (x1, y1, x2, y2 in 0..1).
"""


def iou(a, b) -> float:
    """Intersection over Union of two boxes: overlap area / combined area."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(det_boxes, track_boxes, threshold):
    """Greedy one-to-one matching, best overlaps first.

    Returns {detection index: track index}. Each detection and each track is
    used at most once, so two detections in one photo never merge into one
    damage, and one detection never feeds two damages.
    """
    pairs = sorted(((iou(d, t), di, ti)
                    for di, d in enumerate(det_boxes)
                    for ti, t in enumerate(track_boxes)), reverse=True)
    matched, used_tracks = {}, set()
    for score, di, ti in pairs:
        if score < threshold:
            break
        if di in matched or ti in used_tracks:
            continue
        matched[di] = ti
        used_tracks.add(ti)
    return matched
