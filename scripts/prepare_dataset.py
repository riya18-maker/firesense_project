"""Stages 2-3: dataset audit, cleaning, de-duplication and leakage-safe split (YOLO format).

Accepted input layouts (e.g. D-Fire):
    SRC/{train,val|valid,test}/{images,labels}      or      SRC/{images,labels}
Optional hard negatives: a folder of images with NO fire/smoke (sunsets, fog, steam, red clothes,
car lights ...). They are added as background images (empty labels) to train/val.
"""
import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, list_images, progress  # noqa: E402
from firesense.config import load_config  # noqa: E402


def find_pairs(src):
    src = Path(src)
    pairs = []
    found = False
    for split in ("train", "val", "valid", "test"):
        d = src / split
        if (d / "images").is_dir():
            found = True
            lbl_dir = d / "labels"
            for img in list_images(d / "images"):
                pairs.append({"img": img, "lbl": lbl_dir / (img.stem + ".txt"),
                              "origin": "test" if split == "test" else "trainval"})
    if not found and (src / "images").is_dir():
        found = True
        for img in list_images(src / "images"):
            pairs.append({"img": img, "lbl": src / "labels" / (img.stem + ".txt"), "origin": "flat"})
    if not found:
        raise SystemExit(f"No 'images' folders found under {src}. Expected SRC/train/images ... or SRC/images")
    return pairs


def parse_label(path, n_classes):
    """Return (clean boxes [(cls,cx,cy,w,h)], n_bad_lines, label_file_missing)."""
    boxes, bad = [], 0
    if path is None or not Path(path).exists():
        return boxes, bad, True
    for line in Path(path).read_text(errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            if len(parts) != 5:
                raise ValueError
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            bad += 1
            continue
        x1, x2 = max(0.0, x - w / 2), min(1.0, x + w / 2)
        y1, y2 = max(0.0, y - h / 2), min(1.0, y + h / 2)
        if cls < 0 or cls >= n_classes or (x2 - x1) <= 1e-4 or (y2 - y1) <= 1e-4:
            bad += 1
            continue
        boxes.append((cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return boxes, bad, False


def dhash(gray):
    s = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def strat_key(boxes, n_classes):
    return "".join(str(int(any(b[0] == c for b in boxes))) for c in range(n_classes))


def split_list(items, keys, frac, seed):
    """Stratified split of indices; falls back to a plain split if a stratum is too small."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(items))
    if len(items) < 2 or frac <= 0:
        return list(idx), []
    counts = Counter(keys)
    strat = keys if min(counts.values()) >= 2 and len(counts) > 1 else None
    try:
        a, b = train_test_split(idx, test_size=frac, random_state=seed, stratify=strat)
    except ValueError:
        a, b = train_test_split(idx, test_size=frac, random_state=seed)
    return list(a), list(b)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="raw dataset folder")
    ap.add_argument("--out", default=None, help="output folder (default: data/firesense)")
    ap.add_argument("--hard-negatives", default=None, help="folder of images without fire/smoke")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.10, help="only used when the source has no test split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    n_classes = len(cfg["classes"])
    names = {int(k): str(v) for k, v in cfg["classes"].items()}
    out = Path(a.out) if a.out else ROOT / cfg["paths"]["data_dir"]
    report_dir = ROOT / cfg["paths"]["results_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(a.src)
    n_hard = 0
    if a.hard_negatives:
        for img in list_images(a.hard_negatives):
            pairs.append({"img": img, "lbl": None, "origin": "hardneg"})
            n_hard += 1
    # test images first so that duplicates are dropped from train/val, never from test (no leakage)
    order = {"test": 0, "flat": 1, "trainval": 1, "hardneg": 2}
    pairs.sort(key=lambda p: order[p["origin"]])

    audit = Counter()
    seen_md5, seen_dh = set(), set()
    kept = []
    for p in progress(pairs, "auditing"):
        audit["total_found"] += 1
        img = cv2.imread(str(p["img"]), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        if img is None or img.size == 0:
            audit["removed_corrupt_image"] += 1
            continue
        if min(img.shape[:2]) * 2 < 32:
            audit["removed_too_small"] += 1
            continue
        md5 = hashlib.md5(p["img"].read_bytes()).hexdigest()
        dh = dhash(img)
        if md5 in seen_md5 or dh in seen_dh:
            audit["removed_duplicate"] += 1
            continue
        seen_md5.add(md5)
        seen_dh.add(dh)
        boxes, bad, missing = parse_label(p["lbl"], n_classes) if p["origin"] != "hardneg" else ([], 0, False)
        audit["bad_label_lines_removed"] += bad
        if missing and p["origin"] != "hardneg":
            audit["missing_label_file_treated_as_background"] += 1
        p["boxes"] = boxes
        kept.append(p)
    audit["kept"] = len(kept)
    if not kept:
        raise SystemExit("No valid images left after cleaning.")

    # ---------- split ----------
    test_items = [p for p in kept if p["origin"] == "test"]
    rest = [p for p in kept if p["origin"] != "test"]
    if test_items:
        keys = [strat_key(p["boxes"], n_classes) for p in rest]
        tr_i, va_i = split_list(rest, keys, a.val_frac, a.seed)
        splits = {"train": [rest[i] for i in tr_i], "val": [rest[i] for i in va_i], "test": test_items}
    else:
        keys = [strat_key(p["boxes"], n_classes) for p in rest]
        trv_i, te_i = split_list(rest, keys, a.test_frac, a.seed)
        trv = [rest[i] for i in trv_i]
        k2 = [strat_key(p["boxes"], n_classes) for p in trv]
        tr_i, va_i = split_list(trv, k2, a.val_frac / max(1e-9, 1 - a.test_frac), a.seed)
        splits = {"train": [trv[i] for i in tr_i], "val": [trv[i] for i in va_i], "test": [rest[i] for i in te_i]}

    # ---------- write ----------
    if out.exists():
        shutil.rmtree(out)
    stats = {}
    for split, items in splits.items():
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        box_counts, img_counts = Counter(), Counter()
        for i, p in enumerate(progress(items, f"writing {split}")):
            stem = f"{i:06d}_{p['img'].stem}"
            shutil.copy2(p["img"], out / split / "images" / (stem + p["img"].suffix.lower()))
            lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in p["boxes"]]
            (out / split / "labels" / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            for b in p["boxes"]:
                box_counts[names[b[0]]] += 1
            for c in {b[0] for b in p["boxes"]}:
                img_counts[names[c]] += 1
            if not p["boxes"]:
                img_counts["background"] += 1
        stats[split] = {"images": len(items), "boxes": dict(box_counts), "images_with_class": dict(img_counts)}

    yaml_text = (f"path: {out.resolve().as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
                 f"names:\n" + "".join(f"  {k}: {v}\n" for k, v in sorted(names.items())))
    (out / "data.yaml").write_text(yaml_text)
    report = {"audit": dict(audit), "hard_negatives_added": n_hard, "splits": stats,
              "note": "Duplicates (exact MD5 or identical dHash) removed before splitting; test images processed first."}
    (report_dir / "dataset_audit.json").write_text(json.dumps(report, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        cats = list(names.values())
        for j, (title, key) in enumerate((("Boxes per class", "boxes"), ("Images containing class", "images_with_class"))):
            width = 0.25
            for k, split in enumerate(splits):
                vals = [stats[split][key].get(c, 0) for c in cats]
                ax[j].bar(np.arange(len(cats)) + k * width, vals, width, label=split)
            ax[j].set_xticks(np.arange(len(cats)) + width)
            ax[j].set_xticklabels(cats)
            ax[j].set_title(title)
            ax[j].legend()
        fig.tight_layout()
        fig.savefig(report_dir / "class_balance.png", dpi=150)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)

    print(json.dumps(report, indent=2))
    print(f"\nDataset ready: {out / 'data.yaml'}")
    return report


if __name__ == "__main__":
    main()
