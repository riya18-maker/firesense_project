# FireSense: pipeline explanation (use this for the report and viva)

## Problem
Detect **fire** and **smoke** in images, video and live camera streams, localise them with a bounding box,
estimate severity, and raise a **reliable** alert (few false alarms). Fire is easy (bright, saturated colour);
smoke is hard (translucent, grey, low contrast, looks like fog/steam/clouds).

## Architecture
```
Input (image/video/webcam)
  │
  ├─► [Classical branch, on original colours]  HSV+YCrCb fire mask · flicker · smoke smoothness · motion · LBP · edge loss
  │
  ├─► [Preprocessing]  resize → denoise → white balance → CLAHE → (optional dark-channel dehaze)
  │          └─► YOLO detector (boxes + confidence)
  │
  ├─► [Hybrid fusion]  fused_conf = 0.7 · detector_conf + 0.3 · classical_cue_in_box
  ├─► [Temporal confirmation]  alarm only if ≥ min_hits detections in last window frames (hysteresis to clear)
  ├─► [Severity]  % of frame covered by fire/smoke boxes → Low / Medium / High / Critical
  └─► Alert log (CSV) + snapshot + annotated output
```

## Stage by stage

| # | Stage | What we do | Why | Code |
|---|---|---|---|---|
| 1 | Scope | classes fire, smoke, normal; output box + label + confidence + alert | clear task definition | `configs/default.yaml` |
| 2 | Data | D-Fire (YOLO format) + hard negatives (sunsets, red clothes, fog, steam, lights) | hard negatives reduce false alarms | `scripts/prepare_dataset.py` |
| 3 | Audit & clean | remove corrupt images, drop exact and re-encoded duplicates (MD5 + dHash) *before* splitting, clean invalid labels, report class balance, keep official test split, stratified val split | prevents leakage and bad labels | `scripts/prepare_dataset.py` |
| 4 | Preprocessing | letterbox, bilateral denoise, gentle gray-world white balance, CLAHE on LAB lightness, dark-channel-prior dehazing with guided filter | improve contrast/visibility, esp. for smoke | `firesense/preprocessing.py` |
| 5 | Classical features | fire colour rules (HSV and YCrCb), flicker (mask change between frames), smoke candidate (low saturation + smooth + moving), edge-energy loss, LBP texture entropy | interpretable baseline + cues for fusion | `firesense/classical.py` |
| 6 | Augmentation | colour jitter, flip, scale, mosaic, mixup (Ultralytics) + synthetic haze copies of smoke images | robustness to lighting and haze | `scripts/train.py`, `scripts/augment_haze.py` |
| 7 | Deep model | YOLO (transfer learning from COCO weights) | real-time detection + localisation | `scripts/train.py` |
| 8 | Fusion + temporal | combine detector confidence with classical cue; N-of-M voting with hysteresis | fewer false alarms, stable alerts | `firesense/fusion.py`, `firesense/temporal.py` |
| 9 | Evaluation | P, R, mAP@0.5, mAP@0.5:0.95, per-class AP, confusion matrix, PR curves, image-level ablation, video false alarms and latency, FPS | prove each component's value | `scripts/evaluate.py`, `scripts/evaluate_video.py` |
| 10 | Explainability | EigenCAM heatmaps, saved FP/FN images | show *why* it fires, analyse failures | `firesense/explain.py`, `scripts/explain.py` |
| 11 | Application | Streamlit dashboard, severity, alert log + snapshots, optional ONNX export | usable system | `app.py`, `scripts/export_onnx.py` |

## Design decisions worth mentioning
- **Classical cues use the original colours**, not the white-balanced frame, because white balance would shift fire colours.
- **White balance gain is clipped (0.8-1.25)** so a frame that is mostly orange is not "corrected" into grey.
- **Dehazing is optional and off by default**: it can remove real smoke along with haze. We test it in the ablation.
- **EigenCAM instead of Grad-CAM**: Grad-CAM needs one class score to back-propagate; a detector has many outputs, EigenCAM is gradient-free and standard for YOLO.
- **Duplicates removed before splitting**, with test images processed first, so no test image appears in train.
- **Hysteresis in temporal voting** (raise at `min_hits`, clear at `clear_hits`) prevents the alarm from flickering on/off.
- **Image-level ablation uses the same decision rule for every row**, so differences come from the components, not the thresholds.

## Evaluation protocol (what to write in the report)
1. Detection metrics on the held-out test split (mAP, P, R, per-class AP, confusion matrix, PR curves).
2. Ablation: classical only / YOLO / YOLO+preprocessing / full hybrid. Report F1 per class, false-alarm rate on normal images, FPS.
3. Temporal voting on videos: detection rate, first-alarm latency, false-alarm events per normal video, with vs without voting.
4. Failure analysis: look at `results/failure_cases/fp` and `fn`; group them (sunset, red clothing, fog, small distant fire, thin smoke).
5. Limitations: dataset bias, near-duplicate frames, thin smoke at distance, night/IR, single-camera view.
6. Future work: night/thermal imagery, tracking, edge deployment (Jetson/ONNX/TensorRT), semi-supervised learning on unlabelled camera footage.
