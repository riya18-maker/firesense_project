import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from firesense.config import load_config, class_ids
from firesense.preprocessing import (letterbox, resize_max, preprocess, preprocess_stages,
                                     dehaze, add_haze, white_balance_gray_world, clahe_enhance)
from firesense.classical import ClassicalAnalyzer, lbp_entropy, classical_detections
from firesense.temporal import TemporalConfirmer
from firesense.fusion import fuse_detections
from firesense.severity import estimate_severity
from firesense.pipeline import FireSensePipeline
from firesense.explain import cam_from_activations, overlay_heatmap
from firesense.alerts import AlertLogger

CFG = load_config()


def fire_frame(shift=0):
    img = np.full((480, 640, 3), (30, 30, 30), np.uint8)
    cv2.ellipse(img, (320 + shift, 300), (70, 110), 0, 0, 360, (0, 140, 255), -1)   # orange (BGR)
    cv2.ellipse(img, (320 + shift, 320), (35, 60), 0, 0, 360, (80, 230, 255), -1)   # yellow core
    return img


def smoke_frame(shift=0):
    rng = np.random.default_rng(0)
    img = np.full((480, 640, 3), (60, 90, 40), np.uint8)
    noise = (rng.random((480, 640, 1)) * 40).astype(np.uint8)
    img = cv2.add(img, np.repeat(noise, 3, 2))
    cv2.rectangle(img, (200 + shift, 120), (440 + shift, 330), (170, 170, 170), -1)
    return cv2.GaussianBlur(img, (21, 21), 0)


class FakeDetector:
    def __init__(self, dets):
        self.dets = dets
    def predict(self, bgr):
        return [dict(d) for d in self.dets]


def test_config_classes():
    ids = class_ids(CFG)
    assert ids == {"smoke": 0, "fire": 1}


def test_preprocessing_shapes():
    img = fire_frame()
    lb, r, (l, t) = letterbox(img, 640)
    assert lb.shape == (640, 640, 3) and r > 0
    assert resize_max(np.zeros((2000, 1000, 3), np.uint8), 960).shape[0] == 960
    out = preprocess(img, CFG["preprocess"])
    assert out.shape == img.shape and out.dtype == np.uint8
    assert white_balance_gray_world(img).shape == img.shape
    assert clahe_enhance(img).shape == img.shape
    assert len(preprocess_stages(img, CFG["preprocess"])) >= 3


def test_dehaze_and_haze():
    img = smoke_frame()
    hazy = add_haze(img, 0.6, np.random.default_rng(1))
    assert hazy.shape == img.shape
    assert hazy.mean() > img.mean()            # haze brightens / lowers contrast
    out = dehaze(hazy)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert hazy.std() < img.std()              # haze reduces contrast


def test_fire_colour_detection():
    an = ClassicalAnalyzer(CFG)
    r = an.analyze(fire_frame())
    assert r["fire_ratio"] > 0.05 and r["fire_score"] > 0.5
    an2 = ClassicalAnalyzer(CFG)
    blue = np.full((480, 640, 3), (200, 60, 20), np.uint8)
    assert an2.analyze(blue)["fire_ratio"] == 0
    red_cloth = np.full((480, 640, 3), (0, 0, 190), np.uint8)   # dark red cloth must NOT trigger
    assert ClassicalAnalyzer(CFG).analyze(red_cloth)["fire_ratio"] == 0


def test_flicker_and_motion():
    an = ClassicalAnalyzer(CFG)
    an.analyze(fire_frame(0))
    r = an.analyze(fire_frame(12))
    assert r["has_prev"] and r["flicker"] > 0


def test_smoke_cue_and_lbp():
    an = ClassicalAnalyzer(CFG)
    r = an.analyze(smoke_frame(0))
    assert r["smoke_score"] >= 0
    r2 = an.analyze(smoke_frame(10))
    assert 0 <= r2["smoke_score"] <= 1 and r2["has_prev"]
    g = cv2.cvtColor(smoke_frame(), cv2.COLOR_BGR2GRAY)
    assert 0 <= lbp_entropy(g) <= 1
    noise = np.random.default_rng(0).integers(0, 255, (100, 100), dtype=np.uint8)
    assert lbp_entropy(noise) > lbp_entropy(np.full((100, 100), 128, np.uint8))


def test_classical_boxes():
    an = ClassicalAnalyzer(CFG)
    r = an.analyze(fire_frame())
    dets = classical_detections(r, CFG, (480, 640, 3))
    assert any(d["cls_name"] == "fire" for d in dets)
    x1, y1, x2, y2 = dets[0]["xyxy"]
    assert 200 < x1 < 320 and 320 < x2 < 450


def test_temporal_voting_and_hysteresis():
    t = TemporalConfirmer(CFG["temporal"])
    states = [t.update({"fire": True, "smoke": False})["fire"] for _ in range(6)]
    assert states[:5] == [False] * 5 and states[5] is True          # needs 6 hits
    for _ in range(10):
        s = t.update({"fire": False, "smoke": False})
    assert s["fire"] is False                                          # clears again
    t2 = TemporalConfirmer(CFG["temporal"])
    for i in range(30):                                                # isolated blips never alarm
        assert t2.update({"fire": i % 5 == 0, "smoke": False})["fire"] is False


def test_fusion_boosts_with_cue():
    an = ClassicalAnalyzer(CFG)
    a = an.analyze(fire_frame())
    on_fire = {"cls_name": "fire", "conf": 0.40, "xyxy": [250, 190, 390, 420]}
    off_fire = {"cls_name": "fire", "conf": 0.40, "xyxy": [0, 0, 100, 100]}
    out = fuse_detections([on_fire, off_fire], a, CFG)
    assert len(out) == 1 and out[0]["conf"] > 0.40 and out[0]["xyxy"] == on_fire["xyxy"]


def test_severity_levels():
    box = lambda f: {"cls_name": "fire", "conf": 0.9, "xyxy": [0, 0, 640 * f ** 0.5, 480 * f ** 0.5]}
    assert estimate_severity([], (480, 640, 3), CFG)["level"] == "None"
    assert estimate_severity([box(0.005)], (480, 640, 3), CFG)["level"] == "Low"
    assert estimate_severity([box(0.10)], (480, 640, 3), CFG)["level"] == "High"
    assert estimate_severity([box(0.50)], (480, 640, 3), CFG)["level"] == "Critical"


def test_pipeline_with_fake_detector_and_temporal(tmp_path):
    det = FakeDetector([{"cls_name": "fire", "conf": 0.8, "xyxy": [250.0, 190.0, 390.0, 420.0]}])
    pipe = FireSensePipeline(CFG, det)
    alarms = []
    for i in range(10):
        res = pipe.process(fire_frame(i % 3))
        alarms.append(res.alarms["fire"])
    assert alarms[-1] and not alarms[0]
    assert res.annotated.shape == (480, 640, 3) and res.severity["level"] != "None"
    log = AlertLogger(tmp_path, cooldown_s=0)
    ev = log.update(res.annotated, {"fire": True, "smoke": False}, res.detections, res.severity)
    assert len(ev) == 1 and (tmp_path / "alerts_log.csv").exists()
    assert len(list((tmp_path / "snapshots").glob("*.jpg"))) == 1
    assert log.update(res.annotated, {"fire": True, "smoke": False}, res.detections, res.severity) == []


def test_pipeline_ablation_switches():
    det = FakeDetector([{"cls_name": "smoke", "conf": 0.5, "xyxy": [0.0, 0.0, 50.0, 50.0]}])
    p = FireSensePipeline(CFG, det, use_preproc=False, use_fusion=False, use_temporal=False)
    r = p.process(smoke_frame())
    assert r.flags["smoke"] and r.alarms["smoke"]
    p2 = FireSensePipeline(CFG, det, use_fusion=True, use_temporal=False)
    assert p2.process(smoke_frame()).extra == {}


def test_pipeline_classical_only():
    pipe = FireSensePipeline(CFG, None, use_temporal=False)
    r = pipe.process(fire_frame())
    assert r.flags["fire"]
    assert not FireSensePipeline(CFG, None, use_temporal=False).process(
        np.full((480, 640, 3), (200, 60, 20), np.uint8)).flags["fire"]


def test_eigencam_numpy_core():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(32, 20, 20)) * 0.1
    a[:, 5:12, 5:12] += 2.0
    cam = cam_from_activations(a)
    assert cam.shape == (20, 20) and 0 <= cam.min() and cam.max() <= 1
    assert cam[5:12, 5:12].mean() > cam[14:, 14:].mean() + 0.5
    ov = overlay_heatmap(np.zeros((40, 40, 3), np.uint8), cv2.resize(cam, (40, 40)))
    assert ov.shape == (40, 40, 3)
