"""Stage 10: explainability. EigenCAM heatmaps for YOLO (gradient-free, works for detectors).

EigenCAM takes the first principal component of a late feature map, which shows the regions
the network's features respond to most. (Grad-CAM needs a classification score to back-propagate
from; a YOLO detector has no single one, so EigenCAM is the standard choice for detectors.)
"""
import cv2
import numpy as np

from .preprocessing import letterbox


def cam_from_activations(a):
    """a: (C, H, W) activations -> (H, W) heatmap in [0, 1] from the first principal component."""
    c, h, w = a.shape
    flat = a.reshape(c, -1).T.astype(np.float64)
    flat = flat - flat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    cam = (flat @ vt[0]).reshape(h, w)
    # The sign of a principal component is arbitrary: orient it so it correlates
    # positively with the raw activation strength (mean over channels).
    strength = np.abs(a).mean(axis=0)
    if cam.std() > 1e-12 and strength.std() > 1e-12 and np.corrcoef(cam.ravel(), strength.ravel())[0, 1] < 0:
        cam = -cam
    cam = np.maximum(cam, 0)
    rng = cam.max() - cam.min()
    return (cam - cam.min()) / rng if rng > 1e-9 else np.zeros_like(cam)


def eigen_cam(yolo_model, bgr, layer=-2, imgsz=640):
    """Return an (H, W) float heatmap in [0, 1] aligned to the input image."""
    import torch
    net = yolo_model.model if hasattr(yolo_model, "model") else yolo_model
    net.eval()
    store = {}
    handle = net.model[layer].register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    lb, r, (left, top) = letterbox(bgr, imgsz)
    x = torch.from_numpy(lb[:, :, ::-1].copy().transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    p = next(net.parameters())
    try:
        with torch.no_grad():
            net(x.to(p.device, p.dtype))
    finally:
        handle.remove()
    act = store["a"]
    act = act[0] if not isinstance(act, (list, tuple)) else act[0][0]
    cam = cam_from_activations(act.float().cpu().numpy())
    cam = cv2.resize(cam, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    h, w = bgr.shape[:2]
    nh, nw = int(round(h * r)), int(round(w * r))
    cam = cam[top:top + nh, left:left + nw]
    return cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)


def overlay_heatmap(bgr, cam, alpha=0.45):
    heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1 - alpha, heat, alpha, 0)
