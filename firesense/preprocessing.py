"""Stage 4: OpenCV preprocessing (resize, denoise, white balance, CLAHE, dehaze) + synthetic haze."""
from collections import OrderedDict
import cv2
import numpy as np


def resize_max(img, max_side):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    s = max_side / m
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def letterbox(img, size=640, color=(114, 114, 114)):
    """Aspect-preserving resize with padding. Returns (image, scale, (pad_left, pad_top))."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), color, np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, (left, top)


def denoise(img):
    return cv2.bilateralFilter(img, 5, 35, 35)


def white_balance_gray_world(img, gain_min=0.8, gain_max=1.25):
    """Gray-world white balance with limited gain so fire colours are not destroyed."""
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(0) + 1e-6
    gain = np.clip(means.mean() / means, gain_min, gain_max)
    return np.clip(f * gain, 0, 255).astype(np.uint8)


def clahe_enhance(img, clip=2.0, grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


# ---------------- Dark Channel Prior dehazing ----------------
def dark_channel(img_f, patch=15):
    dc = np.min(img_f, axis=2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(dc, k)


def estimate_atmosphere(img_f, dark, top_frac=0.001):
    n = max(1, int(dark.size * top_frac))
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    return img_f.reshape(-1, 3)[idx].mean(0)


def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))


def guided_filter(guide, src, r=30, eps=1e-3):
    mean_i, mean_p = _box(guide, r), _box(src, r)
    cov_ip = _box(guide * src, r) - mean_i * mean_p
    var_i = _box(guide * guide, r) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return _box(a, r) * guide + _box(b, r)


def dehaze(img, patch=15, omega=0.85, t0=0.2):
    f = img.astype(np.float32) / 255.0
    dark = dark_channel(f, patch)
    A = np.maximum(estimate_atmosphere(f, dark), 0.3)
    t = 1.0 - omega * dark_channel(f / A, patch)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = np.clip(guided_filter(gray, t.astype(np.float32)), t0, 1.0)
    j = (f - A) / t[..., None] + A
    return (np.clip(j, 0, 1) * 255).astype(np.uint8)


def add_haze(img, strength=0.5, rng=None):
    """Synthetic haze (augmentation for smoke robustness)."""
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    noise = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), max(h, w) / 8)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    t = 1.0 - strength * (0.5 + 0.5 * noise)
    A = rng.uniform(0.75, 0.95)
    f = img.astype(np.float32) / 255.0
    out = f * t[..., None] + A * (1 - t[..., None])
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def preprocess_stages(img, cfg):
    """Ordered dict of every intermediate stage (used for the report figure)."""
    p = cfg
    stages = OrderedDict()
    cur = resize_max(img, p["max_side"])
    stages["1 resized"] = cur
    if p["denoise"]:
        cur = denoise(cur)
        stages["2 denoised"] = cur
    if p["white_balance"]:
        cur = white_balance_gray_world(cur)
        stages["3 white balance"] = cur
    if p["clahe"]:
        cur = clahe_enhance(cur, p["clahe_clip"], p["clahe_grid"])
        stages["4 CLAHE"] = cur
    if p["dehaze"]:
        cur = dehaze(cur, p["dehaze_patch"], p["dehaze_omega"], p["dehaze_t0"])
        stages["5 dehazed"] = cur
    return stages


def preprocess(img, cfg):
    """Full preprocessing chain for the detector input. cfg = config['preprocess']."""
    return list(preprocess_stages(img, cfg).values())[-1]
