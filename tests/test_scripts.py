import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from test_core import fire_frame, smoke_frame, FakeDetector, CFG  # noqa: E402


def make_raw_dataset(root):
    """D-Fire style: train/test with images+labels, plus defects the audit must catch."""
    rng = np.random.default_rng(0)
    for split, n in (("train", 30), ("test", 10)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(n):
            kind = i % 3
            if kind == 0:
                img, lbl = fire_frame(i), "1 0.5 0.6 0.3 0.5\n"
            elif kind == 1:
                img, lbl = smoke_frame(i), "0 0.5 0.4 0.4 0.4\n"
            else:
                img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
                lbl = ""
            img = cv2.add(img, rng.integers(0, 8, img.shape, dtype=np.uint8))  # make each unique
            cv2.imwrite(str(root / split / "images" / f"{split}{i}.jpg"), img)
            (root / split / "labels" / f"{split}{i}.txt").write_text(lbl)
    tr = root / "train"
    (tr / "images" / "corrupt.jpg").write_bytes(b"not an image")
    (tr / "labels" / "corrupt.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    data = (tr / "images" / "train0.jpg").read_bytes()
    (tr / "images" / "dup_of_train0.jpg").write_bytes(data)
    (tr / "labels" / "dup_of_train0.txt").write_text("1 0.5 0.6 0.3 0.5\n")
    (root / "test" / "images" / "leak.jpg").write_bytes(data)           # duplicate of a TRAIN image
    (root / "test" / "labels" / "leak.txt").write_text("1 0.5 0.6 0.3 0.5\n")
    (tr / "labels" / "train1.txt").write_text("0 0.5 0.4 0.4 0.4\n7 0.5 0.5 0.1 0.1\nbad line\n0 0.99 0.5 0.5 0.5\n")


def _prepare(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    make_raw_dataset(raw)
    hn = tmp_path / "hn"
    hn.mkdir()
    for i in range(4):
        cv2.imwrite(str(hn / f"sunset{i}.jpg"), np.full((200, 300, 3), (40 + 30 * i, 120, 250), np.uint8))
    import prepare_dataset
    rep = prepare_dataset.main(["--src", str(raw), "--out", str(out), "--hard-negatives", str(hn)])
    a = rep["audit"]
    assert a["removed_corrupt_image"] == 1
    assert a["removed_duplicate"] >= 1
    assert a["bad_label_lines_removed"] >= 2
    assert rep["hard_negatives_added"] == 4
    assert (out / "data.yaml").exists()
    # leakage check: no image content shared between splits
    import hashlib
    sets = {s: {hashlib.md5(p.read_bytes()).hexdigest() for p in (out / s / "images").glob("*")}
            for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["test"]) and not (sets["train"] & sets["val"]) and not (sets["val"] & sets["test"])
    # every image has a label file and all label lines are valid YOLO boxes
    for s in ("train", "val", "test"):
        for img in (out / s / "images").glob("*"):
            lab = out / s / "labels" / (img.stem + ".txt")
            assert lab.exists()
            for line in lab.read_text().splitlines():
                c, *v = line.split()
                assert int(c) in (0, 1) and all(0 <= float(x) <= 1 for x in v)
    assert json.loads((ROOT / "results" / "dataset_audit.json").read_text())["audit"]["kept"] > 0
    assert (out / "data.yaml").read_text().count("train") >= 1
    return out


def test_prepare_dataset_audit_and_split(tmp_path):
    _prepare(tmp_path)


def test_haze_augmentation(tmp_path):
    out = _prepare(tmp_path)
    import augment_haze
    n_before = len(list((out / "train" / "images").glob("*")))
    augment_haze.main(["--data", str(out), "--ratio", "1.0"])
    n_after = len(list((out / "train" / "images").glob("*")))
    assert n_after > n_before
    for h in (out / "train" / "images").glob("*_haze.*"):
        assert (out / "train" / "labels" / (h.stem + ".txt")).exists()


def test_ablation_with_fake_detector(tmp_path):
    out = _prepare(tmp_path)
    import evaluate
    det = FakeDetector([{"cls_name": "fire", "conf": 0.8, "xyxy": [150.0, 100.0, 300.0, 300.0]}])
    df = evaluate.run_ablation(CFG, det, out / "test" / "images", out / "test" / "labels", tmp_path / "res")
    assert list(df["config"])[0].startswith("1") and len(df) == 4
    assert (tmp_path / "res" / "ablation.csv").exists() and (tmp_path / "res" / "ablation.png").exists()
    yolo = df[df.config.str.startswith("2")].iloc[0]
    assert yolo["fire_recall"] == 1.0                       # fake detector always says fire
    assert 0 <= yolo["alarm_f1"] <= 1 and yolo["fps"] > 0
    assert (tmp_path / "res" / "failure_cases").exists()
    p, r, f1 = evaluate.prf([1, 1, 0, 0], [1, 0, 1, 0])
    assert (p, r) == (0.5, 0.5)


def make_video(path, n=40):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 480))
    assert w.isOpened()
    for i in range(n):
        w.write(fire_frame((i % 4) * 5))
    w.release()


def test_cli_video_and_eval_video(tmp_path, monkeypatch):
    vid = tmp_path / "videos" / "fire" / "f.mp4"
    vid.parent.mkdir(parents=True)
    make_video(vid)
    norm = tmp_path / "videos" / "normal" / "n.mp4"
    norm.parent.mkdir(parents=True)
    wr = cv2.VideoWriter(str(norm), cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 480))
    for _ in range(30):
        wr.write(np.full((480, 640, 3), (200, 60, 20), np.uint8))
    wr.release()
    import run_video, evaluate_video
    out_vid = tmp_path / "out.mp4"
    run_video.main(["--source", str(vid), "--out", str(out_vid), "--classical-only"])
    assert out_vid.exists() and out_vid.stat().st_size > 1000
    evaluate_video.main(["--videos-dir", str(tmp_path / "videos"), "--classical-only"])
    import pandas as pd
    df = pd.read_csv(ROOT / "results" / "video_eval.csv")
    fire_t = df[(df.truth == "fire") & (df["mode"] == "temporal voting")].iloc[0]
    assert bool(fire_t["alarm"]) and fire_t["first_alarm_s"] > 0
    assert df[(df.truth == "normal")]["alarm_events"].sum() == 0


def test_demo_preprocessing(tmp_path):
    img = tmp_path / "a.jpg"
    cv2.imwrite(str(img), fire_frame())
    import demo_preprocessing
    demo_preprocessing.main(["--image", str(img), "--out", str(tmp_path / "fig.png")])
    assert (tmp_path / "fig.png").stat().st_size > 5000


# ---------- wiring tests with a stub Ultralytics (real YOLO needs PyTorch; not available in CI sandbox) ----------
class _StubBox:
    mp, mr, map50, map = 0.8, 0.7, 0.75, 0.5
    ap_class_index = [0, 1]
    ap50 = [0.6, 0.9]


class _StubModel:
    names = {0: "smoke", 1: "fire"}

    def val(self, **kw):
        assert kw["split"] == "test" and kw["data"].endswith("data.yaml")
        return type("M", (), {"box": _StubBox()})()


def test_evaluate_main_wiring(tmp_path, monkeypatch):
    out = _prepare(tmp_path)
    import types, evaluate
    stub = types.ModuleType("firesense.detector")

    class StubDet(FakeDetector):
        def __init__(self, *a, **k):
            super().__init__([{"cls_name": "smoke", "conf": 0.7, "xyxy": [100.0, 80.0, 400.0, 350.0]}])
            self.model = _StubModel()
    stub.YoloDetector = StubDet
    monkeypatch.setitem(sys.modules, "firesense.detector", stub)
    evaluate.main(["--data", str(out / "data.yaml"), "--weights", "models/best.pt", "--max-images", "12"])
    res = ROOT / "results"
    m = json.loads((res / "detection_metrics.json").read_text())
    assert m["mAP50"] == 0.75 and m["per_class_AP50"] == {"smoke": 0.6, "fire": 0.9}
    assert (res / "ablation.csv").exists()


def test_train_wiring(tmp_path, monkeypatch):
    import types, train
    calls = {}

    class StubYOLO:
        def __init__(self, name):
            calls["model"] = name

        def train(self, **kw):
            calls.update(kw)
            w = Path(kw["project"]) / kw["name"] / "weights"
            w.mkdir(parents=True)
            (w / "best.pt").write_bytes(b"x")
    stub = types.ModuleType("ultralytics")
    stub.YOLO = StubYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", stub)
    backup = ROOT / "models" / "best.pt"
    existed = backup.exists()
    train.main(["--data", "d.yaml", "--model", "yolov8n.pt", "--epochs", "1", "--project", str(tmp_path), "--name", "t"])
    assert calls["model"] == "yolov8n.pt" and calls["epochs"] == 1 and calls["data"] == "d.yaml"
    assert backup.exists()
    if not existed:
        backup.unlink()
