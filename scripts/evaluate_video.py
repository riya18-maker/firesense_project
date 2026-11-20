"""Stage 9b: evaluate temporal confirmation on videos.

VIDEOS_DIR must contain sub-folders  fire/  smoke/  normal/  with video files (ground truth = folder).
For each video we run the pipeline WITHOUT and WITH temporal voting and report:
alarm raised?, first-alarm latency (s), number of alarm events (false alarm count for normal videos).
"""
import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import progress  # noqa: E402
from firesense.config import load_config, resolve  # noqa: E402
from firesense.pipeline import FireSensePipeline  # noqa: E402

VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def run_video(path, pipe, frame_skip=2, max_frames=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    pipe.reset()
    i = n = events = 0
    first = None
    prev = False
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        i += 1
        if i % frame_skip:
            continue
        n += 1
        res = pipe.process(frame, draw=False)
        on = res.alarms["fire"] or res.alarms["smoke"]
        if on and not prev:
            events += 1
            if first is None:
                first = i / fps
        prev = on
    cap.release()
    return {"alarm": events > 0, "first_alarm_s": first, "alarm_events": events, "frames": n}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--classical-only", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    det = None
    if not a.classical_only:
        from firesense.detector import YoloDetector
        det = YoloDetector(resolve(a.weights or cfg["detector"]["weights"]), cfg["detector"]["imgsz"],
                           cfg["detector"]["conf"], cfg["detector"]["iou"], cfg["detector"]["device"])
    rows = []
    vids = [(p.parent.name.lower(), p) for p in sorted(Path(a.videos_dir).rglob("*")) if p.suffix.lower() in VID_EXT]
    for truth, p in progress(vids, "videos"):
        for label, temporal in (("no temporal", False), ("temporal voting", True)):
            pipe = FireSensePipeline(cfg, det, True, True, temporal)
            r = run_video(p, pipe, a.frame_skip, a.max_frames)
            rows.append({"video": p.name, "truth": truth, "mode": label, **r})
    df = pd.DataFrame(rows)
    out = resolve(cfg["paths"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "video_eval.csv", index=False)
    print(df.to_string(index=False))
    if len(df):
        normal = df[df.truth == "normal"].groupby("mode")["alarm_events"].mean()
        print("\nMean false-alarm events per NORMAL video:\n", normal.to_string())
        pos = df[df.truth != "normal"].groupby("mode")["alarm"].mean()
        print("\nDetection rate on fire/smoke videos:\n", pos.to_string())


if __name__ == "__main__":
    main()
