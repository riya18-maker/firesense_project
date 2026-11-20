"""Alert logging: CSV log + snapshot image on every new confirmed alarm."""
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2


class AlertLogger:
    FIELDS = ["timestamp", "label", "confidence", "severity", "fire_area_pct", "snapshot"]

    def __init__(self, out_dir, cooldown_s=10):
        self.dir = Path(out_dir)
        self.snap_dir = self.dir / "snapshots"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / "alerts_log.csv"
        self.cooldown = float(cooldown_s)
        self._prev = {"fire": False, "smoke": False}
        self._last = {"fire": 0.0, "smoke": 0.0}
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.FIELDS)

    def update(self, frame_bgr, alarms, dets, severity):
        """Log a row on the rising edge of a confirmed alarm (respecting the cooldown)."""
        events = []
        now = time.time()
        for label in ("fire", "smoke"):
            rising = alarms.get(label, False) and not self._prev[label]
            self._prev[label] = bool(alarms.get(label, False))
            if not rising or now - self._last[label] < self.cooldown:
                continue
            self._last[label] = now
            confs = [d["conf"] for d in dets if d["cls_name"] == label]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap = self.snap_dir / f"{ts}_{label}.jpg"
            cv2.imwrite(str(snap), frame_bgr)
            row = [datetime.now().isoformat(timespec="seconds"), label,
                   f"{max(confs) if confs else 0.0:.3f}", severity["level"],
                   f"{severity['fire_pct']:.2f}", str(snap)]
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
            events.append(dict(zip(self.FIELDS, row)))
        return events
