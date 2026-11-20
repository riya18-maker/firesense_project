"""Figure for the report: every preprocessing stage + the classical fire/smoke masks for one image."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT  # noqa: E402,F401  (adds the project root to sys.path)
from firesense.classical import ClassicalAnalyzer  # noqa: E402
from firesense.config import load_config, resolve, clone  # noqa: E402
from firesense.preprocessing import preprocess_stages, dehaze, letterbox  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config()
    pcfg = clone(cfg["preprocess"])
    pcfg["dehaze"] = True                       # show dehazing in the figure
    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"Cannot read {a.image}")
    stages = dict(preprocess_stages(img, pcfg))
    stages["letterbox 640 (YOLO input)"] = letterbox(stages["1 resized"], 640)[0]
    an = ClassicalAnalyzer(cfg).analyze(stages["1 resized"])
    panels = list(stages.items())
    panels.append(("fire colour mask", cv2.cvtColor(an["fire_mask"], cv2.COLOR_GRAY2BGR)))
    panels.append(("smoke candidate mask", cv2.cvtColor(an["smoke_mask"], cv2.COLOR_GRAY2BGR)))
    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (title, im) in zip(np.ravel(axes), panels):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"fire score {an['fire_score']:.2f} | smoke score {an['smoke_score']:.2f} | "
                 f"LBP entropy {an['lbp_entropy']:.2f}", fontsize=11)
    fig.tight_layout()
    out = Path(a.out) if a.out else resolve(cfg["paths"]["results_dir"]) / "preprocessing_demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print("Saved", out)


if __name__ == "__main__":
    main()
