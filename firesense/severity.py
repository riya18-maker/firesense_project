"""Severity estimate from the fraction of the frame covered by fire / smoke boxes."""
import numpy as np


def union_area_pct(dets, name, shape):
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    for d in dets:
        if d["cls_name"] != name:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
        m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 1
    return float(m.mean() * 100.0)


def estimate_severity(dets, shape, cfg):
    s = cfg["severity"]
    fire_pct = union_area_pct(dets, "fire", shape)
    smoke_pct = union_area_pct(dets, "smoke", shape)
    if fire_pct == 0 and smoke_pct == 0:
        level = "None"
    else:
        score = max(fire_pct, 0.5 * smoke_pct)
        idx = sum(score >= t for t in s["thresholds"])
        level = s["levels"][idx]
    return {"fire_pct": fire_pct, "smoke_pct": smoke_pct, "level": level}
