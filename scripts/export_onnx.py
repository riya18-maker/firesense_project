"""Stage 11b: export the trained detector to ONNX and benchmark speed (optional)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT  # noqa: E402,F401  (adds the project root to sys.path)
from firesense.config import load_config, resolve  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    a = ap.parse_args(argv)
    from ultralytics import YOLO
    cfg = load_config()
    w = resolve(a.weights or cfg["detector"]["weights"])
    imgsz = a.imgsz or cfg["detector"]["imgsz"]
    onnx_path = YOLO(str(w)).export(format="onnx", imgsz=imgsz)
    print("Exported:", onnx_path)
    img = np.random.randint(0, 255, (imgsz, imgsz, 3), np.uint8)
    for name, path in (("PyTorch .pt", str(w)), ("ONNX", str(onnx_path))):
        m = YOLO(path)
        m.predict(img, imgsz=imgsz, verbose=False)            # warm-up
        t0 = time.perf_counter()
        for _ in range(20):
            m.predict(img, imgsz=imgsz, verbose=False)
        print(f"{name}: {(time.perf_counter() - t0) / 20 * 1000:.1f} ms/frame")


if __name__ == "__main__":
    main()
