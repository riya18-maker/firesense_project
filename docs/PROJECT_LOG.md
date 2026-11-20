# FireSense: project log

## Status: code complete. Remaining work is yours: dataset + training + evaluation + report.

### Built and tested (automated tests, `pytest tests`)
- [x] Config-driven pipeline (`configs/default.yaml`)
- [x] Preprocessing: letterbox, denoise, white balance, CLAHE, dark-channel dehaze, synthetic haze
- [x] Classical branch: fire colour (HSV + YCrCb), flicker, smoke smoothness/motion/edge-loss/LBP, box proposals
- [x] Hybrid fusion, temporal voting with hysteresis, severity estimate, alert logger
- [x] Dataset audit / clean / dedupe / split script (tested with deliberately corrupted data)
- [x] Ablation + failure-case evaluation script (tested with a stub detector)
- [x] Video evaluation, CLI runner, preprocessing figure script
- [x] Streamlit dashboard (tested headlessly; falls back to classical demo mode without weights)
- [x] EigenCAM (numerical core tested)

### Written but NOT run here (needs PyTorch + a trained model, so run on Colab/your machine)
- [ ] `scripts/train.py`, `firesense/detector.py`, `scripts/evaluate.py` mAP part, `scripts/explain.py` (hooking a real YOLO), `scripts/export_onnx.py`

### Your checklist
- [ ] Day 1: venv + tests pass · download D-Fire · collect hard negatives · `prepare_dataset.py` · start Colab training
- [ ] Day 2: evaluate · ablation table · tune thresholds on validation split
- [ ] Day 3: explain · failure analysis · video test · dashboard demo
- [ ] Day 4: report + slides + rehearsal

### Decisions (explain in viva)
Hybrid design · classical branch on original colours · gain-limited white balance · dehazing optional (tested in ablation) ·
EigenCAM for detectors · duplicates removed before splitting · temporal voting with hysteresis · same decision rule across ablation rows.

### Results (fill in after running)
| Item | Value |
|---|---|
| mAP@0.5 | |
| mAP@0.5:0.95 | |
| Fire F1 / Smoke F1 (hybrid) | |
| False-alarm rate on normal images (YOLO vs hybrid) | |
| False alarms per normal video (no temporal vs temporal) | |
| FPS (GPU / CPU) | |
