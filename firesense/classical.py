"""Stage 5: classical OpenCV feature branch.

Fire  : HSV + YCrCb colour rules, flicker (mask change between frames).
Smoke : low-saturation grey regions that are smooth (low Laplacian energy), moving,
        losing edge energy, with an LBP texture measure.
Produces masks + scores, and simple box proposals (the classical-only baseline).
"""
import cv2
import numpy as np

_KERNEL3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_KERNEL5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def lbp_image(gray):
    """Basic 8-neighbour Local Binary Pattern (output is 2 px smaller per side)."""
    g = gray.astype(np.int16)
    h, w = g.shape
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(offs):
        nb = g[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        code |= ((nb >= c).astype(np.uint8) << i)
    return code


def lbp_entropy(gray, mask=None):
    """Normalised (0-1) entropy of the LBP histogram, optionally inside a mask."""
    codes = lbp_image(gray)
    if mask is not None:
        m = mask[1:-1, 1:-1] > 0
        codes = codes[m]
    if codes.size == 0:
        return 0.0
    hist = np.bincount(codes.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / 8.0)


class ClassicalAnalyzer:
    def __init__(self, cfg):
        self.c = cfg["classical"]
        self.reset()

    def reset(self):
        self.prev_gray = None
        self.prev_fire = None
        self.prev_edge = None

    def analyze(self, bgr, update=True):
        c = self.c
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        H, S, V = cv2.split(hsv)
        Y, Cr, Cb = cv2.split(ycc)

        # ---------- fire colour mask ----------
        fire = ((H <= c["fire_h_max"]) & (S >= c["fire_s_min"]) & (V >= c["fire_v_min"])
                & ((Cr.astype(np.int16) - Cb.astype(np.int16)) >= c["fire_crcb_diff"])
                & (Y >= c["fire_y_min"]))
        fire = fire.astype(np.uint8) * 255
        fire = cv2.morphologyEx(fire, cv2.MORPH_OPEN, _KERNEL3)
        fire = cv2.morphologyEx(fire, cv2.MORPH_CLOSE, _KERNEL5)
        fire_ratio = float((fire > 0).mean())

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gblur = cv2.GaussianBlur(gray, (5, 5), 0)

        # ---------- motion ----------
        has_prev = self.prev_gray is not None and self.prev_gray.shape == gblur.shape
        motion = None
        if has_prev:
            diff = cv2.absdiff(gblur, self.prev_gray)
            motion = (diff >= c["motion_thresh"]).astype(np.uint8) * 255
            motion = cv2.dilate(motion, _KERNEL5, iterations=2)

        # ---------- fire flicker ----------
        flicker = 0.0
        if has_prev and self.prev_fire is not None and self.prev_fire.shape == fire.shape:
            a, b = fire > 0, self.prev_fire > 0
            union = np.logical_or(a, b).sum()
            if union > 0:
                flicker = float(1.0 - np.logical_and(a, b).sum() / union)

        # ---------- smoke candidate ----------
        grayish = (S <= c["smoke_s_max"]) & (V >= c["smoke_v_min"]) & (V <= c["smoke_v_max"])
        lap = cv2.Laplacian(gblur, cv2.CV_32F, ksize=3)
        energy = cv2.boxFilter(lap * lap, -1, (15, 15))
        cand = (grayish & (energy <= c["smoke_energy_max"])).astype(np.uint8) * 255
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, _KERNEL5)
        smoke_mask = cv2.bitwise_and(cand, motion) if motion is not None else cand
        smoke_ratio = float((smoke_mask > 0).mean())

        cand_px = int((cand > 0).sum())
        motion_in_cand = 0.0
        if motion is not None and cand_px > 0:
            motion_in_cand = float((cv2.bitwise_and(cand, motion) > 0).sum() / cand_px)

        gx = cv2.Sobel(gblur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gblur, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        edge_now = float(mag[cand > 0].mean()) if cand_px > 0 else 0.0
        edge_loss = 0.0
        if self.prev_edge is not None and self.prev_edge > 1e-6:
            edge_loss = max(0.0, (self.prev_edge - edge_now) / self.prev_edge)

        ent = lbp_entropy(gray, cand) if cand_px > 0 else 0.0

        # ---------- scores (0-1) ----------
        fire_base = min(1.0, fire_ratio / c["fire_ratio_sat"])
        fire_motion = 0.5 + 0.5 * min(1.0, flicker / 0.3) if has_prev else 0.8
        fire_score = float(min(1.0, fire_base * fire_motion))

        smoke_base = min(1.0, smoke_ratio / c["smoke_ratio_sat"]) if motion is not None \
            else min(1.0, (cand_px / cand.size) / c["smoke_ratio_sat"]) * 0.6
        f_motion = 0.5 + 0.5 * min(1.0, motion_in_cand / 0.2) if motion is not None else 1.0
        f_edge = 0.7 + 0.3 * min(1.0, edge_loss / 0.2)
        f_tex = 1.0 - 0.3 * ent
        smoke_score = float(min(1.0, smoke_base * f_motion * f_edge * f_tex))

        if update:
            self.prev_gray, self.prev_fire, self.prev_edge = gblur, fire, edge_now

        return {
            "fire_mask": fire, "smoke_mask": smoke_mask,
            "fire_ratio": fire_ratio, "smoke_ratio": smoke_ratio,
            "flicker": flicker, "motion_in_cand": motion_in_cand,
            "edge_loss": edge_loss, "lbp_entropy": ent, "has_prev": has_prev,
            "fire_score": fire_score, "smoke_score": smoke_score,
        }


def classical_detections(analysis, cfg, shape):
    """Turn classical masks into box proposals (the classical-only baseline detector)."""
    c = cfg["classical"]
    h, w = shape[:2]
    min_area = c["min_region_frac"] * h * w
    dets = []
    for name, mkey, skey, merge in (("fire", "fire_mask", "fire_score", 9),
                                    ("smoke", "smoke_mask", "smoke_score", 25)):
        score = analysis[skey]
        if score <= 0:
            continue
        m = analysis[mkey]
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge, merge))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            boxes.append((bw * bh, [float(x), float(y), float(x + bw), float(y + bh)]))
        boxes.sort(key=lambda t: -t[0])
        for _, box in boxes[:5]:
            dets.append({"cls_name": name, "conf": float(score), "xyxy": box, "source": "classical"})
    return dets
