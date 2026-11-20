"""Run FireSense on a video file or webcam (CLI). Press q to quit when --show is used."""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT  # noqa: E402
from firesense.alerts import AlertLogger  # noqa: E402
from firesense.config import load_config, resolve  # noqa: E402
from firesense.pipeline import FireSensePipeline  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video path or webcam index (e.g. 0)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default=None, help="write annotated video here (mp4)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--frame-skip", type=int, default=1)
    ap.add_argument("--no-preproc", action="store_true")
    ap.add_argument("--no-fusion", action="store_true")
    ap.add_argument("--no-temporal", action="store_true")
    ap.add_argument("--classical-only", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    det = None
    if not a.classical_only:
        w = resolve(a.weights or cfg["detector"]["weights"])
        if w.exists():
            from firesense.detector import YoloDetector
            det = YoloDetector(w, cfg["detector"]["imgsz"], cfg["detector"]["conf"], cfg["detector"]["iou"],
                               cfg["detector"]["device"])
        else:
            print(f"[warn] weights not found at {w}; running classical-only mode")
    pipe = FireSensePipeline(cfg, det, not a.no_preproc, not a.no_fusion, not a.no_temporal)
    logger = AlertLogger(resolve(cfg["paths"]["alerts_dir"]), cfg["alerts"]["cooldown_s"])

    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {a.source}")
    writer, i, t0 = None, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % a.frame_skip:
            continue
        res = pipe.process(frame)
        for ev in logger.update(res.annotated, res.alarms, res.detections, res.severity):
            print("[ALERT]", ev["timestamp"], ev["label"], ev["severity"], ev["snapshot"])
        if a.out and writer is None:
            h, w = res.annotated.shape[:2]
            fps = (cap.get(cv2.CAP_PROP_FPS) or 25.0) / a.frame_skip
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if writer is not None:
            writer.write(res.annotated)
        if a.show:
            cv2.imshow("FireSense", res.annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer is not None:
        writer.release()
    if a.show:
        cv2.destroyAllWindows()
    print(f"Processed {i} frames in {time.time() - t0:.1f}s. Alerts: {logger.csv_path}")


if __name__ == "__main__":
    main()
