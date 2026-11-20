"""Drawing helpers for annotated output frames."""
import cv2

COLORS = {"fire": (0, 90, 255), "smoke": (200, 160, 60)}


def draw_results(frame, dets, alarms, severity, fps=None):
    out = frame.copy()
    for d in dets:
        name = d["cls_name"]
        x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
        col = COLORS.get(name, (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 3 if alarms.get(name) else 1)
        label = f"{name} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), col, -1)
        cv2.putText(out, label, (x1 + 2, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    active = [k.upper() for k, v in alarms.items() if v]
    banner_col = (0, 0, 200) if active else (60, 120, 40)
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), banner_col, -1)
    text = ("ALARM: " + " + ".join(active)) if active else "Monitoring - no confirmed alarm"
    if severity["level"] != "None":
        text += f" | severity {severity['level']} | fire area {severity['fire_pct']:.1f}%"
    if fps is not None:
        text += f" | {fps:.1f} FPS"
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if active:
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (0, 0, 220), 4)
    return out
