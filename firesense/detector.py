"""YOLO detector wrapper (Ultralytics). Works with .pt and exported .onnx weights."""
from pathlib import Path


class YoloDetector:
    def __init__(self, weights, imgsz=640, conf=0.25, iou=0.45, device=None):
        from ultralytics import YOLO  # imported lazily so the classical parts work without torch
        if not Path(weights).exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        self.model = YOLO(str(weights))
        self.imgsz, self.conf, self.iou, self.device = imgsz, conf, iou, device

    @property
    def names(self):
        return {int(k): str(v).lower() for k, v in self.model.names.items()}

    def predict(self, bgr):
        kw = dict(imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False)
        if self.device not in (None, "", "auto"):
            kw["device"] = self.device
        res = self.model.predict(bgr, **kw)[0]
        names = self.names
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            for b, c, k in zip(xyxy, conf, cls):
                dets.append({"cls_id": int(k), "cls_name": names.get(int(k), str(k)),
                             "conf": float(c), "xyxy": [float(v) for v in b], "source": "yolo"})
        return dets
