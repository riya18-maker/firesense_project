# 🔥 FireSense: Hybrid Fire & Smoke Detection System

A computer-vision pipeline that combines **classical OpenCV analysis** with **deep-learning detection (YOLO)**
and **temporal confirmation**, wrapped in a Streamlit dashboard with alert logging.

```
frame → resize/denoise/white-balance/CLAHE (+dehaze) → classical cues (colour, flicker, texture, motion)
      → YOLO detector → hybrid fusion → N-frame temporal voting → severity → alert log + snapshot
```

## What makes it more than a basic classifier
| Layer | What it does |
|---|---|
| Hybrid fusion | detector confidence + classical colour / flicker / texture cue inside each box |
| Smoke handling | CLAHE, dark-channel dehazing, synthetic-haze augmentation, smoothness + LBP + edge-loss cues |
| Hard negatives | sunsets, red clothes, fog, steam, lights added as background images |
| Temporal confirmation | alarm only after N detections in a window (with hysteresis) |
| Ablation study | classical only vs YOLO vs +preprocessing vs full hybrid, plus temporal on videos |
| Explainability | EigenCAM heatmaps + saved false-positive / false-negative images |
| Application | Streamlit dashboard (image / video / webcam), severity estimate, alert CSV + snapshots |

## Folder structure
```
app.py                    Streamlit dashboard
configs/default.yaml      ALL parameters (class ids, thresholds, fusion weights, temporal window ...)
firesense/                library: preprocessing, classical, detector, fusion, temporal, severity, alerts, explain, pipeline
scripts/                  prepare_dataset, augment_haze, train, evaluate, evaluate_video, run_video, explain, demo_preprocessing, export_onnx
notebooks/colab_train.ipynb   run training on a free Colab GPU
tests/                    automated tests (pytest)
docs/                     PIPELINE.md (explain every stage), VIVA_QA.md, PROJECT_LOG.md
models/                   put best.pt here
```

## 1. Setup (VS Code, Windows)
```powershell
cd firesense_project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests          # should pass without any dataset or model
```
If your MSI laptop has an NVIDIA GPU you can install the CUDA build of PyTorch from pytorch.org *before* the
requirements; otherwise everything still works on CPU (training should be done on Colab anyway).

## 2. Dataset
Use **D-Fire** (YOLO format, classes `0 = smoke, 1 = fire`). Expected layout:
```
D-Fire/train/images  D-Fire/train/labels  D-Fire/test/images  D-Fire/test/labels
```
**Verify the class order** in the dataset's own documentation. If yours is reversed, edit `classes:` in
`configs/default.yaml` (the whole project reads it from there).

Hard negatives (strongly recommended, ~200+ images): collect pictures with **no** fire/smoke that look similar:
sunsets, orange/red clothing, fog, steam, car brake lights, lamps, chimneys with clear air. Put them in one folder.

```bash
python scripts/prepare_dataset.py --src PATH/TO/D-Fire --out data/firesense --hard-negatives PATH/TO/hard_negatives
```
This removes corrupt files, drops exact and re-encoded duplicates (before splitting, so there is no train/test leakage),
cleans invalid label lines, keeps the original test split, carves a validation split, and writes
`data/firesense/data.yaml`, `results/dataset_audit.json` and `results/class_balance.png`.

## 3. Train (Colab) , Stage 7
Open `notebooks/colab_train.ipynb` in Google Colab (GPU runtime) and follow the cells. Fast profile first, full profile if time allows.
Download `best.pt` into `models/best.pt`.

## 4. Evaluate and explain (Colab or local)
```bash
python scripts/evaluate.py --data data/firesense/data.yaml --weights models/best.pt
python scripts/evaluate_video.py --videos-dir PATH/TO/videos        # folders: fire/ smoke/ normal/
python scripts/explain.py --images data/firesense/test/images --n 12
python scripts/demo_preprocessing.py --image some_image.jpg         # figure for the report
```
Outputs land in `results/`: `detection_metrics.json`, `detection_eval/` (confusion matrix, PR curves),
`ablation.csv/.png`, `failure_cases/`, `explainability/`, `video_eval.csv`.

## 5. Run the app
```bash
streamlit run app.py
```
Works even without weights (classical-only demo mode). With `models/best.pt` it uses the full hybrid pipeline.
Command-line alternative: `python scripts/run_video.py --source 0 --show` (webcam) or `--source video.mp4 --out result.mp4`.

## Suggested 4-day schedule
| Day | Work |
|---|---|
| 1 | Setup, tests, dataset download, `prepare_dataset`, `demo_preprocessing` figure, start Colab training early |
| 2 | Finish training, `evaluate.py` (mAP + ablation), tune `decision_conf` and fusion weights in `configs/default.yaml` |
| 3 | `explain.py`, failure-case analysis, video tests with `evaluate_video.py`, run the dashboard |
| 4 | Report + slides (use `docs/PIPELINE.md` and `docs/VIVA_QA.md`), rehearse the demo |

## Honest notes
- **No results are included.** Numbers (mAP, F1, ablation) come from *your* training run; report what you measure.
- The hybrid is not guaranteed to beat YOLO-only on every metric. The ablation table tells you, and a clear,
  honest ablation is a strength in a viva either way. Dehazing is **off by default** for the detector
  because it can also wash out real smoke; test it with `dehaze: true`.
- Fusion weights and classical thresholds are sensible defaults, not tuned values. Tune them on the **validation** split, not the test split.
- Near-identical frames from the same video can still exist across splits in public datasets (a known limitation).
- Ultralytics YOLO is licensed AGPL-3.0, which is fine for coursework; check licensing before commercial use.

## Troubleshooting
| Problem | Fix |
|---|---|
| `No images folders found` | point `--src` at the folder that directly contains `train/` and `test/` |
| Fire/smoke labels look swapped in the app | class order mismatch, edit `classes:` in `configs/default.yaml` and retrain/re-prepare |
| Too many false alarms | raise `detector.decision_conf`, raise `temporal.min_hits`, add more hard negatives |
| Missed fires | lower `decision_conf`, lower `min_hits`, train longer / larger model |
| Slow on CPU | use `yolov8n`, larger `frame skip` in the app, or export ONNX (`scripts/export_onnx.py`) |
| Webcam does not open | change the camera index (0, 1, 2) |
