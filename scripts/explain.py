"""Stage 10: EigenCAM explainability heatmaps for a folder of images (or a single image)."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import list_images  # noqa: E402
from firesense.config import load_config, resolve  # noqa: E402
from firesense.explain import eigen_cam, overlay_heatmap  # noqa: E402
from firesense.viz import draw_results  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="image file or folder")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--layer", type=int, default=-2, help="layer index to hook (default: last neck layer before Detect)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)

    from firesense.detector import YoloDetector
    cfg = load_config(a.config)
    det = YoloDetector(resolve(a.weights or cfg["detector"]["weights"]), cfg["detector"]["imgsz"],
                       cfg["detector"]["conf"], cfg["detector"]["iou"], cfg["detector"]["device"])
    p = Path(a.images)
    paths = [p] if p.is_file() else list_images(p)[: a.n]
    out = resolve(cfg["paths"]["results_dir"]) / "explainability"
    out.mkdir(parents=True, exist_ok=True)
    sev = {"level": "None", "fire_pct": 0.0, "smoke_pct": 0.0}
    for ip in paths:
        img = cv2.imread(str(ip))
        if img is None:
            continue
        cam = eigen_cam(det.model, img, layer=a.layer, imgsz=cfg["detector"]["imgsz"])
        dets = [d for d in det.predict(img) if d["conf"] >= cfg["detector"]["decision_conf"]]
        left = draw_results(img, dets, {"fire": False, "smoke": False}, sev)
        panel = np.hstack([left, overlay_heatmap(img, cam)])
        cv2.imwrite(str(out / f"{ip.stem}_cam.jpg"), panel)
    print(f"Saved {len(paths)} explainability panels (detections | heatmap) to {out}")


if __name__ == "__main__":
    main()
