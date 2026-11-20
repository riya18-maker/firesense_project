"""Stage 6 (extra): synthetic-haze copies of training images that contain smoke."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import list_images, progress  # noqa: E402
from firesense.config import load_config, class_ids  # noqa: E402
from firesense.preprocessing import add_haze  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="prepared dataset folder (contains train/)")
    ap.add_argument("--ratio", type=float, default=0.3, help="fraction of smoke images to duplicate with haze")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    smoke_id = class_ids(load_config())["smoke"]
    rng = np.random.default_rng(a.seed)
    img_dir, lbl_dir = Path(a.data) / "train" / "images", Path(a.data) / "train" / "labels"
    n = 0
    for img_path in progress([p for p in list_images(img_dir) if not p.stem.endswith("_haze")], "haze"):
        lbl = lbl_dir / (img_path.stem + ".txt")
        if not lbl.exists():
            continue
        text = lbl.read_text()
        has_smoke = any(l.split() and int(float(l.split()[0])) == smoke_id for l in text.splitlines())
        if not has_smoke or rng.random() > a.ratio:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        hazy = add_haze(img, strength=float(rng.uniform(0.3, 0.7)), rng=rng)
        cv2.imwrite(str(img_dir / f"{img_path.stem}_haze{img_path.suffix}"), hazy)
        (lbl_dir / f"{img_path.stem}_haze.txt").write_text(text)
        n += 1
    print(f"Added {n} synthetic-haze images to {img_dir}")


if __name__ == "__main__":
    main()
