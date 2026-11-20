"""Stage 9: evaluation.

A) Detection metrics on the test split (Ultralytics): P, R, mAP@0.5, mAP@0.5:0.95, per-class AP,
   confusion matrix and PR curves (saved by Ultralytics).
B) Image-level ABLATION on the test split (same decision rule for every row):
      1 classical only | 2 YOLO only | 3 YOLO + preprocessing | 4 full hybrid (+ classical fusion)
   reporting precision / recall / F1 per class, false-alarm rate on normal images and speed.
   Temporal voting is evaluated on videos (scripts/evaluate_video.py).
C) Failure cases (false positives / false negatives of the hybrid) saved as images.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, list_images, progress  # noqa: E402
from firesense.config import load_config, class_ids, resolve  # noqa: E402
from firesense.pipeline import FireSensePipeline  # noqa: E402


def read_gt(label_path, ids):
    gt = {"fire": False, "smoke": False}
    inv = {v: k for k, v in ids.items()}
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if parts and int(float(parts[0])) in inv:
                gt[inv[int(float(parts[0]))]] = True
    return gt


def prf(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def run_config(name, pipe, images, gts, save_failures=None, max_fail=30):
    rows, ms = [], []
    fails = {"fp": 0, "fn": 0}
    for img_path, gt in progress(list(zip(images, gts)), name):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        pipe.reset()
        res = pipe.process(frame, draw=save_failures is not None)
        ms.append(res.ms)
        pred = res.flags
        rows.append((gt["fire"], gt["smoke"], pred["fire"], pred["smoke"]))
        if save_failures is not None:
            any_gt, any_pred = gt["fire"] or gt["smoke"], pred["fire"] or pred["smoke"]
            kind = "fp" if (any_pred and not any_gt) else "fn" if (any_gt and not any_pred) else None
            if kind and fails[kind] < max_fail:
                d = Path(save_failures) / kind
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{Path(img_path).stem}.jpg"), res.annotated)
                fails[kind] += 1
    a = np.array(rows, bool)
    out = {"config": name}
    for i, cls in enumerate(("fire", "smoke")):
        p, r, f1 = prf(a[:, i], a[:, 2 + i])
        out.update({f"{cls}_precision": p, f"{cls}_recall": r, f"{cls}_f1": f1})
    any_t, any_p = a[:, 0] | a[:, 1], a[:, 2] | a[:, 3]
    p, r, f1 = prf(any_t, any_p)
    neg = ~any_t
    out.update({"alarm_precision": p, "alarm_recall": r, "alarm_f1": f1,
                "false_alarm_rate_on_normal": float(any_p[neg].mean()) if neg.any() else float("nan"),
                "ms_per_image": float(np.mean(ms)), "fps": float(1000 / np.mean(ms))})
    return out


def run_ablation(cfg, detector, images_dir, labels_dir, out_dir, max_images=None):
    ids = class_ids(cfg)
    images = list_images(images_dir)
    if max_images:
        images = images[:max_images]
    gts = [read_gt(Path(labels_dir) / (p.stem + ".txt"), ids) for p in images]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [("1 Classical only", FireSensePipeline(cfg, None, use_temporal=False))]
    if detector is not None:
        configs += [
            ("2 YOLO only", FireSensePipeline(cfg, detector, False, False, False)),
            ("3 YOLO + preprocessing", FireSensePipeline(cfg, detector, True, False, False)),
            ("4 Full hybrid (preproc + fusion)", FireSensePipeline(cfg, detector, True, True, False)),
        ]
    results = []
    for name, pipe in configs:
        fail_dir = out_dir / "failure_cases" if name == configs[-1][0] else None
        results.append(run_config(name, pipe, images, gts, save_failures=fail_dir))
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "ablation.csv", index=False)
    (out_dir / "ablation.md").write_text(df.round(3).to_markdown(index=False)
                                         if _has_tabulate() else df.round(3).to_string(index=False))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4.5))
        w = 0.25
        for k, col in enumerate(("fire_f1", "smoke_f1", "alarm_f1")):
            ax.bar(np.arange(len(df)) + k * w, df[col], w, label=col.replace("_f1", " F1"))
        ax.set_xticks(np.arange(len(df)) + w)
        ax.set_xticklabels([c.split(" ", 1)[1] for c in df["config"]], rotation=12, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title("Ablation study: image-level F1 on the test split")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "ablation.png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)
    return df


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except Exception:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--data", required=True, help="data.yaml from prepare_dataset.py")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-images", type=int, default=None, help="limit images in the ablation (speed)")
    ap.add_argument("--skip-map", action="store_true", help="skip the Ultralytics mAP evaluation")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)

    import yaml
    from firesense.detector import YoloDetector
    cfg = load_config(a.config)
    weights = resolve(a.weights or cfg["detector"]["weights"])
    out_dir = resolve(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    d = yaml.safe_load(open(a.data))
    root = Path(d["path"])
    det = YoloDetector(weights, cfg["detector"]["imgsz"], cfg["detector"]["conf"], cfg["detector"]["iou"],
                       cfg["detector"]["device"])

    if not a.skip_map:
        m = det.model.val(data=a.data, split=a.split, imgsz=cfg["detector"]["imgsz"], conf=0.001, iou=0.6,
                          plots=True, project=str(out_dir), name="detection_eval", exist_ok=True)
        names = det.model.names
        metrics = {"precision": float(m.box.mp), "recall": float(m.box.mr), "mAP50": float(m.box.map50),
                   "mAP50-95": float(m.box.map),
                   "per_class_AP50": {str(names[int(c)]): float(v) for c, v in zip(m.box.ap_class_index, m.box.ap50)}}
        (out_dir / "detection_metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))

    df = run_ablation(cfg, det, root / d[a.split].replace("/images", "") / "images",
                      root / d[a.split].replace("/images", "") / "labels", out_dir, a.max_images)
    print("\n", df.round(3).to_string(index=False))
    print(f"\nSaved to {out_dir} (ablation.csv / ablation.png / failure_cases / detection_eval)")


if __name__ == "__main__":
    main()
