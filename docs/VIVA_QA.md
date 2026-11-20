# Viva preparation: questions and answers

**Q1. What problem does your project solve?** Early fire and smoke detection from camera video, with localisation, severity, and alerts, designed to keep false alarms low.

**Q2. Why hybrid instead of only a CNN?** A single-frame CNN confuses sunsets, red clothing, fog and lights with fire/smoke. Classical cues (fire colour, flicker, smoothness, motion) and multi-frame confirmation add independent evidence. The ablation table measures what each part adds.

**Q3. Why YOLO?** It localises objects and runs near real-time, which suits video; transfer learning from COCO weights works well with limited data.

**Q4. Why is smoke harder than fire?** It is translucent, low-contrast, grey/white, with no fixed shape, and resembles fog, steam and clouds.

**Q5. What preprocessing did you use and why?** Resize/letterbox (consistent input), bilateral denoise, gray-world white balance with limited gain, CLAHE on lightness (local contrast), optional dark-channel dehazing (haze removal). Smoke needs contrast; fire colour must not be destroyed, so gain is clipped and the classical branch uses the original colours.

**Q6. Explain CLAHE.** Contrast Limited Adaptive Histogram Equalisation: equalises histograms per tile, with a clip limit to avoid amplifying noise. Applied on the L channel in LAB so colours are preserved.

**Q7. Explain dark channel prior.** In haze-free outdoor patches, at least one colour channel has very low intensity. Haze raises it, so the dark channel estimates haze thickness. We estimate atmospheric light A and transmission t = 1 − ω·dark(I/A), refine t with a guided filter, and recover J = (I − A)/max(t, t0) + A.

**Q8. How do you detect fire with classical methods?** HSV rule (red-orange-yellow hue, high saturation and value) AND YCrCb rule (Cr − Cb large, Y high), cleaned by morphology. Flicker is measured as change of the fire mask between frames.

**Q9. How do you detect smoke classically?** Low-saturation grey regions with low local Laplacian energy (smooth), moving between frames, losing edge energy, with LBP texture entropy as a weak cue. It is weaker than fire, which is why the deep model leads.

**Q10. What is LBP?** Local Binary Pattern: each pixel is encoded by comparing it with its 8 neighbours; the histogram (and its entropy) describes texture. Smoke regions have low texture complexity.

**Q11. How does fusion work?** fused = 0.7·detector_confidence + 0.3·classical_cue_inside_the_box. Detections below the decision threshold are dropped. A weak detector score can be rescued by strong colour/flicker evidence, and a confident detector is not killed by a missing cue.

**Q12. Why temporal voting?** Single-frame false positives are usually isolated blips. Requiring `min_hits` of the last `window` frames removes them; hysteresis (clear at `clear_hits`) avoids flickering alarms. Trade-off: a small delay before the alarm.

**Q13. How did you prevent data leakage?** Exact (MD5) and visually identical (dHash) duplicates are removed before splitting; test images are processed first so duplicates are removed from train, not test; the official test split is preserved. Limitation: near-identical video frames can still exist.

**Q14. Which metrics and why?** Precision (false alarms), recall (missed fires), F1, mAP@0.5 and mAP@0.5:0.95 (localisation quality), confusion matrix, PR curves, FPS. For a safety system recall matters, but false alarms make the system unusable, so we also report false-alarm rate on normal images and per video.

**Q15. What is mAP?** Mean over classes of the area under the precision-recall curve (average precision), at IoU threshold 0.5, or averaged over 0.5-0.95.

**Q16. What is the ablation study?** Removing/adding components one at a time under the same decision rule (classical only, YOLO, YOLO+preprocessing, full hybrid) to measure each component's contribution.

**Q17. What is EigenCAM? Why not Grad-CAM?** EigenCAM shows the first principal component of a late feature map: the regions the features respond to most. Grad-CAM needs a single class score to back-propagate; a detector has many outputs, and EigenCAM is gradient-free.

**Q18. How is severity computed?** Percentage of the frame covered by the union of fire (and half-weighted smoke) boxes, mapped to Low/Medium/High/Critical with thresholds in the config. It is an estimate from 2D coverage, not real-world fire size.

**Q19. Limitations?** Dataset bias; thin or distant smoke; night and infrared; lookalikes; severity from 2D area only; classical thresholds are hand-set; fusion weights are tuned on a small validation set.

**Q20. Future work?** Object tracking across frames, night/thermal imagery, edge deployment (ONNX/TensorRT/Jetson), learning the fusion weights, semi-supervised training on camera footage, multi-camera correlation.

**Q21. What would you do about the false alarms you observed?** Show `results/failure_cases/fp` grouped by cause; add more matching hard negatives and retrain; raise the threshold or `min_hits`; add a size/persistence filter.

**Q22. Why does the alarm need a delay?** Confirmation needs several frames. At 25 FPS with frame skip 2, `min_hits = 6` means roughly half a second; it can be tuned.
