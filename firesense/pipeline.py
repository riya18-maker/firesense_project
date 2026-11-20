"""End-to-end FireSense pipeline: preprocess -> classical cues -> detector -> fusion -> temporal -> severity."""
import time
from dataclasses import dataclass, field

from .classical import ClassicalAnalyzer, classical_detections
from .fusion import fuse_detections
from .preprocessing import preprocess, resize_max
from .severity import estimate_severity
from .temporal import TemporalConfirmer
from .viz import draw_results


@dataclass
class FrameResult:
    annotated: object
    detections: list
    flags: dict
    alarms: dict
    severity: dict
    analysis: dict
    ms: float = 0.0
    extra: dict = field(default_factory=dict)


class FireSensePipeline:
    """Toggles (use_preproc / use_fusion / use_temporal) make the ablation study trivial.
    detector=None runs the classical-only baseline (also the app's demo mode)."""

    def __init__(self, cfg, detector=None, use_preproc=True, use_fusion=True, use_temporal=True):
        self.cfg = cfg
        self.detector = detector
        self.use_preproc, self.use_fusion, self.use_temporal = use_preproc, use_fusion, use_temporal
        self.analyzer = ClassicalAnalyzer(cfg)
        self.temporal = TemporalConfirmer(cfg["temporal"])

    def reset(self):
        self.analyzer.reset()
        self.temporal.reset()

    def process(self, frame, draw=True):
        t0 = time.perf_counter()
        cfg = self.cfg
        frame = resize_max(frame, cfg["preprocess"]["max_side"])
        # Classical cues use the ORIGINAL colours (white balance would shift fire colours).
        analysis = self.analyzer.analyze(frame)

        if self.detector is not None:
            inp = preprocess(frame, cfg["preprocess"]) if self.use_preproc else frame
            dets = self.detector.predict(inp)
            if self.use_fusion:
                dets = fuse_detections(dets, analysis, cfg)
            else:
                dets = [d for d in dets if d["conf"] >= cfg["detector"]["decision_conf"]]
        else:
            dets = [d for d in classical_detections(analysis, cfg, frame.shape)
                    if d["conf"] >= cfg["classical"]["decision_score"]]

        flags = {"fire": any(d["cls_name"] == "fire" for d in dets),
                 "smoke": any(d["cls_name"] == "smoke" for d in dets)}
        alarms = self.temporal.update(flags) if self.use_temporal else dict(flags)
        severity = estimate_severity(dets, frame.shape, cfg)
        ms = (time.perf_counter() - t0) * 1000.0
        annotated = draw_results(frame, dets, alarms, severity, 1000.0 / ms if ms > 0 else None) if draw else None
        return FrameResult(annotated, dets, flags, alarms, severity, analysis, ms)
