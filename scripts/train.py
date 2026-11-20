"""Stage 7: train the YOLO detector (transfer learning from COCO-pretrained weights)."""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml from prepare_dataset.py")
    ap.add_argument("--model", default="yolov8s.pt", help="yolov8n.pt (fast) | yolov8s.pt | yolo11s.pt ...")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0, help="use only this fraction of the training set (speed)")
    ap.add_argument("--project", default=str(ROOT / "runs"))
    ap.add_argument("--name", default="firesense")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, fraction=a.fraction,
        project=a.project, name=a.name, device=a.device, workers=a.workers,
        patience=a.patience, seed=a.seed, exist_ok=True,
        # augmentation (Stage 6): colour jitter, flips, scale, mosaic, mixup
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, fliplr=0.5, translate=0.1, scale=0.5,
        mosaic=1.0, mixup=0.1, close_mosaic=5,
    )
    best = Path(a.project) / a.name / "weights" / "best.pt"
    if best.exists():
        dst = ROOT / "models" / "best.pt"
        dst.parent.mkdir(exist_ok=True)
        shutil.copy2(best, dst)
        print(f"\nBest weights: {best}\nCopied to: {dst}")


if __name__ == "__main__":
    main()
