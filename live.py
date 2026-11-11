"""
Live gesture recognition from the ESP32 stream.

  python live.py

Controls:
  c      recalibrate skin colour (mask/both modes only)
  m      toggle right-panel-only view
  q/ESC  quit

Put your actions in on_gesture() at the top: it fires once each time the
recognised gesture changes (after smoothing), not on every frame.
"""

import os
import sys
from collections import deque

import cv2
import joblib
import numpy as np

from gesture_common import (
    CALIBRATION_FILE, DISPLAY_SCALE, MODEL_FILE, FrameReceiver,
    calibrate, calibration_box, debug_panel, feature_length, frame_features,
    load_calibration, put_lines, save_calibration, two_panel, big,
)

SMOOTH_FRAMES = 6         # average class probabilities over this many frames
CONF_THRESHOLD = 0.55     # below this, show "?" and don't trigger
HOLD_FRAMES = 3           # gesture must persist this many frames to trigger
CALIBRATION_FRAMES = 10


def on_gesture(name, confidence):
    """Called once whenever the stable gesture changes. Add your actions here."""
    print(f">>> {name}  ({confidence:.2f})")
    # Examples:
    # if name == "fist":  ...
    # if name == "point": ...


def main():
    if not os.path.exists(MODEL_FILE):
        sys.exit(f"{MODEL_FILE} not found. Run train.py first.")

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    # Single-frame prediction: thread pool overhead costs more than it saves
    # (measured ~58 ms with n_jobs=-1 vs ~15 ms with n_jobs=1).
    model.n_jobs = 1
    gestures = bundle["gestures"]
    mode = bundle.get("feature_mode", "mask")
    if bundle["feature_length"] != feature_length(mode):
        sys.exit("Feature settings changed since training (ROI/HOG/grid). Re-run train.py.")
    print("Feature mode:", mode)

    # Prefer the current calibration file (lighting may have changed);
    # fall back to the one the model was trained with.
    cal = load_calibration() if os.path.exists(CALIBRATION_FILE) else bundle["calibration"]
    classes = list(model.classes_)

    rx = FrameReceiver().start()

    probs_hist = deque(maxlen=SMOOTH_FRAMES)
    candidate, candidate_count = None, 0
    stable = None
    last_id = -1
    panel_only = False
    calibrating, cal_frames = 0, []

    while True:
        frame, fid = rx.latest()
        if frame is None:
            blank = np.zeros((240, 640, 3), dtype=np.uint8)
            put_lines(blank, ["Waiting for ESP32 ..."])
            cv2.imshow("live", blank)
            if cv2.waitKey(100) & 0xFF in (ord("q"), 27):
                break
            continue

        if fid == last_id:
            if cv2.waitKey(5) & 0xFF in (ord("q"), 27):
                break
            continue
        last_id = fid

        if calibrating:
            cal_frames.append(frame.copy())
            calibrating -= 1
            if calibrating == 0:
                cal = {**cal, **calibrate(cal_frames)}
                save_calibration(cal)
                print("Calibration saved:", cal)
                cal_frames = []

        feats = frame_features(frame, cal, mode).reshape(1, -1)

        p = np.zeros(len(gestures))
        p[classes] = model.predict_proba(feats)[0]
        probs_hist.append(p)
        avg = np.mean(probs_hist, axis=0)

        best = int(np.argmax(avg))
        conf = float(avg[best])
        current = gestures[best] if conf >= CONF_THRESHOLD else None

        # require the gesture to persist before triggering
        if current == candidate:
            candidate_count += 1
        else:
            candidate, candidate_count = current, 1
        if candidate is not None and candidate_count >= HOLD_FRAMES and candidate != stable:
            stable = candidate
            on_gesture(stable, conf)
        elif candidate is None and candidate_count >= HOLD_FRAMES:
            stable = None

        # ---- display ----
        panel = debug_panel(frame, cal, mode)
        if panel_only:
            view = big(panel)
        else:
            view = two_panel(frame, panel)

        x, y, w, h = calibration_box()
        s = DISPLAY_SCALE
        if calibrating:
            cv2.rectangle(view, (x * s, y * s), ((x + w) * s, (y + h) * s), (0, 0, 255), 1)

        put_lines(view, [
            f"{current or '?'}  {conf:.2f}",
            f"stream {rx.fps:.1f} fps",
        ], color=(0, 255, 0) if current else (0, 200, 255))

        # probability bars
        bar_x = view.shape[1] - 230
        for i, g in enumerate(gestures):
            yy = 20 + i * 20
            cv2.rectangle(view, (bar_x + 90, yy - 12), (bar_x + 90 + int(avg[i] * 130), yy + 2),
                          (0, 255, 0) if i == best else (120, 120, 120), -1)
            put_lines(view, [g[:10]], x=bar_x, y=yy, color=(255, 255, 255))

        cv2.imshow("live", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("c"):
            calibrating, cal_frames = CALIBRATION_FRAMES, []
        elif key == ord("m"):
            panel_only = not panel_only

    rx.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
