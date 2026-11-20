"""Stage 8a: hybrid fusion of detector confidence with classical colour / flicker / texture cues."""
import numpy as np


def box_cue(mask, box):
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((mask[y1:y2, x1:x2] > 0).mean())


def fuse_detections(dets, analysis, cfg):
    """fused = w_dl * detector_conf + w_classical * classical_cue_inside_box."""
    f = cfg["fusion"]
    min_conf = cfg["detector"]["decision_conf"]
    out = []
    for d in dets:
        name = d["cls_name"]
        if name == "fire":
            cue = min(1.0, box_cue(analysis["fire_mask"], d["xyxy"]) / f["fire_cue_sat"])
            if analysis["has_prev"]:
                cue = 0.7 * cue + 0.3 * min(1.0, analysis["flicker"] / f["flicker_sat"])
        elif name == "smoke":
            cue = min(1.0, box_cue(analysis["smoke_mask"], d["xyxy"]) / f["smoke_cue_sat"])
        else:
            cue = 0.0
        nd = dict(d)
        nd["dl_conf"] = float(d["conf"])
        nd["cue"] = float(cue)
        nd["conf"] = float(f["w_dl"] * d["conf"] + f["w_classical"] * cue)
        if nd["conf"] >= min_conf:
            out.append(nd)
    return out
