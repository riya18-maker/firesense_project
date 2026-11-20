"""FireSense dashboard (Streamlit). Run:  streamlit run app.py"""
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from firesense.alerts import AlertLogger
from firesense.config import clone, load_config, resolve
from firesense.pipeline import FireSensePipeline

st.set_page_config(page_title="FireSense", page_icon="🔥", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.2rem;}
.fs-title {font-size: 2.0rem; font-weight: 800; margin-bottom: 0;}
.fs-sub {color: #8a8f98; margin-top: -0.3rem; margin-bottom: 0.8rem;}
</style>""", unsafe_allow_html=True)
st.markdown('<p class="fs-title">🔥 FireSense</p><p class="fs-sub">Hybrid fire &amp; smoke detection: '
            'OpenCV preprocessing + classical cues + YOLO + temporal confirmation</p>', unsafe_allow_html=True)

BASE_CFG = load_config()


@st.cache_resource(show_spinner="Loading detector...")
def load_detector(weights: str, imgsz: int, conf: float, iou: float, device):
    from firesense.detector import YoloDetector
    return YoloDetector(weights, imgsz, conf, iou, device)


# ------------------------------ sidebar ------------------------------
with st.sidebar:
    st.header("Settings")
    weights = st.text_input("Model weights (.pt / .onnx)", BASE_CFG["detector"]["weights"])
    decision = st.slider("Decision confidence", 0.10, 0.90, float(BASE_CFG["detector"]["decision_conf"]), 0.01)
    st.subheader("Pipeline stages")
    use_pre = st.checkbox("OpenCV preprocessing (denoise, WB, CLAHE)", True)
    use_dehaze = st.checkbox("Dark-channel dehazing", BASE_CFG["preprocess"]["dehaze"])
    use_fus = st.checkbox("Hybrid fusion (classical cues)", True)
    use_tmp = st.checkbox("Temporal confirmation (video / webcam)", True)
    window = st.slider("Voting window (frames)", 3, 30, int(BASE_CFG["temporal"]["window"]))
    min_hits = st.slider("Detections needed in window", 1, window, min(int(BASE_CFG["temporal"]["min_hits"]), window))
    frame_skip = st.slider("Process every N-th frame", 1, 10, 2)
    st.subheader("Alerts")
    cooldown = st.slider("Alert cooldown (s)", 0, 60, int(BASE_CFG["alerts"]["cooldown_s"]))

cfg = clone(BASE_CFG)
cfg["detector"]["decision_conf"] = decision
cfg["preprocess"]["dehaze"] = use_dehaze
cfg["temporal"]["window"], cfg["temporal"]["min_hits"] = window, min_hits
cfg["temporal"]["clear_hits"] = min(int(BASE_CFG["temporal"]["clear_hits"]), max(0, min_hits - 1))

detector, det_msg = None, None
wpath = resolve(weights)
if wpath.exists():
    try:
        detector = load_detector(str(wpath), cfg["detector"]["imgsz"], cfg["detector"]["conf"],
                                 cfg["detector"]["iou"], cfg["detector"]["device"])
    except Exception as e:  # missing ultralytics, bad file ...
        det_msg = f"Could not load the model ({e}). Running classical-only demo mode."
else:
    det_msg = f"No weights at `{weights}`. Running **classical-only demo mode** (colour / motion / texture)."
if det_msg:
    st.warning(det_msg)
else:
    st.success(f"Detector loaded: {wpath.name}")

logger = AlertLogger(resolve(cfg["paths"]["alerts_dir"]), cooldown)


def make_pipeline(temporal: bool):
    return FireSensePipeline(cfg, detector, use_pre, use_fus, temporal)


def show_status(cols, res):
    a = res.alarms
    cols[0].metric("Fire", "ALARM" if a["fire"] else ("detected" if res.flags["fire"] else "clear"))
    cols[1].metric("Smoke", "ALARM" if a["smoke"] else ("detected" if res.flags["smoke"] else "clear"))
    cols[2].metric("Severity", res.severity["level"])
    cols[3].metric("Fire area", f"{res.severity['fire_pct']:.1f}%")
    cols[4].metric("Speed", f"{1000 / max(res.ms, 1e-3):.1f} FPS")


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


tab_live, tab_alerts, tab_about = st.tabs(["Detection", "Alert log", "How it works"])

# ------------------------------ detection ------------------------------
with tab_live:
    mode = st.radio("Input", ["Image", "Video", "Webcam"], horizontal=True)

    if mode == "Image":
        up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        want_cam = st.checkbox("Show explainability heatmap (EigenCAM)", False, disabled=detector is None)
        if up is not None:
            frame = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                st.error("Could not read that image.")
            else:
                res = make_pipeline(False).process(frame)
                cols = st.columns(5)
                show_status(cols, res)
                c1, c2 = st.columns(2)
                c1.image(to_rgb(frame), caption="Input", width="stretch")
                c2.image(to_rgb(res.annotated), caption="FireSense output", width="stretch")
                if want_cam and detector is not None:
                    try:
                        from firesense.explain import eigen_cam, overlay_heatmap
                        cam = eigen_cam(detector.model, frame, imgsz=cfg["detector"]["imgsz"])
                        st.image(to_rgb(overlay_heatmap(frame, cam)), caption="EigenCAM: where the model looks",
                                 width="stretch")
                    except Exception as e:
                        st.error(f"Heatmap failed: {e}")
                if res.detections:
                    st.dataframe(pd.DataFrame([{"class": d["cls_name"], "confidence": round(d["conf"], 3),
                                                "detector conf": round(d.get("dl_conf", d["conf"]), 3),
                                                "classical cue": round(d.get("cue", 0.0), 3),
                                                "box": [int(v) for v in d["xyxy"]]} for d in res.detections]),
                                 width="stretch")
                else:
                    st.info("No fire or smoke detected.")
                logger.update(res.annotated, {"fire": res.flags["fire"], "smoke": res.flags["smoke"]},
                              res.detections, res.severity)

    elif mode == "Video":
        up = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"])
        if up is not None and st.button("Run detection", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as tf:
                tf.write(up.read())
            cap = cv2.VideoCapture(tf.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            pipe = make_pipeline(use_tmp)
            cols, img_slot, bar = st.columns(5), st.empty(), st.progress(0.0)
            i, first_alarm = 0, None
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                i += 1
                if i % frame_skip:
                    continue
                res = pipe.process(frame)
                if first_alarm is None and (res.alarms["fire"] or res.alarms["smoke"]):
                    first_alarm = i / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
                logger.update(res.annotated, res.alarms, res.detections, res.severity)
                show_status(cols, res)
                img_slot.image(to_rgb(res.annotated), width="stretch")
                bar.progress(min(1.0, i / total))
            cap.release()
            bar.progress(1.0)
            st.success("Finished." + (f" First confirmed alarm at {first_alarm:.1f}s." if first_alarm is not None
                                      else " No confirmed alarm."))

    else:  # Webcam
        cam_index = st.number_input("Camera index", 0, 5, 0)
        run = st.toggle("Start camera")
        cols, img_slot = st.columns(5), st.empty()
        if run:
            cap = cv2.VideoCapture(int(cam_index))
            if not cap.isOpened():
                st.error("Could not open the camera.")
            else:
                pipe, i = make_pipeline(use_tmp), 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    i += 1
                    if i % max(1, frame_skip // 2):
                        continue
                    res = pipe.process(frame)
                    logger.update(res.annotated, res.alarms, res.detections, res.severity)
                    show_status(cols, res)
                    img_slot.image(to_rgb(res.annotated), width="stretch")
                cap.release()

# ------------------------------ alert log ------------------------------
with tab_alerts:
    if logger.csv_path.exists():
        df = pd.read_csv(logger.csv_path)
        if len(df):
            st.dataframe(df.iloc[::-1], width="stretch")
            st.download_button("Download alert log (CSV)", df.to_csv(index=False), "alerts_log.csv", "text/csv")
            snaps = [Path(p) for p in df["snapshot"].tolist()[-6:][::-1] if Path(p).exists()]
            if snaps:
                st.subheader("Latest snapshots")
                for col, p in zip(st.columns(min(3, len(snaps))), snaps):
                    col.image(str(p), caption=p.name, width="stretch")
        else:
            st.info("No alerts yet. Alerts are logged when a confirmed alarm starts.")

with tab_about:
    st.markdown("""
**Pipeline:** input → resize / denoise / white-balance / CLAHE (optional dehaze) → classical cues
(HSV+YCrCb fire colour, flicker, smoke smoothness, motion, LBP texture) → YOLO detection →
**fusion** (detector confidence + classical cue) → **temporal voting** (N of M frames) →
severity from covered area → alert log + snapshot.

*Why hybrid?* Single-frame detectors raise false alarms on sunsets, red clothing, fog, steam and lights.
Colour/motion cues and multi-frame confirmation remove most of them; the ablation study (see the report)
quantifies each component.
""")
