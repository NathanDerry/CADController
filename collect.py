"""
Collect labelled gesture images from the ESP32 stream.

Controls (click the video window first so it gets key presses):
  0-9    select gesture label (see GESTURES in gesture_common.py)
  SPACE  start / stop recording the selected gesture
  c      calibrate skin colour (only matters in mask/both modes)
  r      reset calibration to defaults
  m      toggle right-panel-only view (mask, or what HOG sees)
  q/ESC  quit

Saves RAW frames (not masks) to data/<gesture>/<session>_<n>.png, so you can
change the mask or features later and retrain without re-recording.
"""

import os
import time

import cv2
import numpy as np

from gesture_common import (
    DATA_DIR, DISPLAY_SCALE, FEATURE_MODE, GESTURES, FrameReceiver, calibrate,
    calibration_box, debug_panel, load_calibration, put_lines, save_calibration,
    two_panel, big, DEFAULT_CALIBRATION,
)

MAX_SAVE_RATE_HZ = 8          # cap saves so bursts aren't all near-duplicates
CALIBRATION_FRAMES = 10


def count_existing():
    counts = {}
    for g in GESTURES:
        d = os.path.join(DATA_DIR, g)
        counts[g] = len([f for f in os.listdir(d) if f.endswith(".png")]) if os.path.isdir(d) else 0
    return counts


def main():
    for g in GESTURES:
        os.makedirs(os.path.join(DATA_DIR, g), exist_ok=True)

    session = time.strftime("%Y%m%d_%H%M%S")
    counts = count_existing()
    session_count = 0

    cal = load_calibration()
    rx = FrameReceiver().start()

    label_idx = 0
    recording = False
    mask_only = False
    calibrating = 0
    cal_frames = []
    last_id = -1
    last_save = 0.0
    redraw = True

    print(f"Session: {session}")
    print("Gestures:", ", ".join(f"{i}={g}" for i, g in enumerate(GESTURES)))

    while True:
        frame, fid = rx.latest()

        if frame is None:
            blank = np.zeros((240, 640, 3), dtype=np.uint8)
            put_lines(blank, ["Waiting for ESP32 ..."])
            cv2.imshow("collect", blank)
            if cv2.waitKey(100) & 0xFF in (ord("q"), 27):
                break
            continue

        new_frame = fid != last_id
        last_id = fid

        if new_frame or redraw:
            redraw = False
            # ---- calibration capture ----
            if calibrating and new_frame:
                cal_frames.append(frame.copy())
                calibrating -= 1
                if calibrating == 0:
                    cal = {**cal, **calibrate(cal_frames)}
                    save_calibration(cal)
                    print("Calibration saved:", cal)
                    cal_frames = []

            panel = debug_panel(frame, cal, FEATURE_MODE)

            # ---- recording ----
            now = time.time()
            if recording and new_frame and now - last_save >= 1.0 / MAX_SAVE_RATE_HZ:
                g = GESTURES[label_idx]
                path = os.path.join(DATA_DIR, g, f"{session}_{counts[g]:05d}.png")
                cv2.imwrite(path, frame)
                counts[g] += 1
                session_count += 1
                last_save = now

            # ---- display ----
            if mask_only:
                view = big(panel)
            else:
                view = two_panel(frame, panel)

            # calibration box on the left panel
            x, y, w, h = calibration_box()
            s = DISPLAY_SCALE
            box_color = (0, 0, 255) if calibrating else (255, 255, 255)
            cv2.rectangle(view, (x * s, y * s), ((x + w) * s, (y + h) * s), box_color, 1)

            g = GESTURES[label_idx]
            status = "REC" if recording else "paused"
            lines = [
                f"[{label_idx}] {g}   {status}   saved: {counts[g]}",
                f"stream {rx.fps:.1f} fps   session total: {session_count}",
            ]
            if calibrating:
                lines.append("Calibrating - keep hand in box ...")
            put_lines(view, lines, color=(0, 0, 255) if recording else (0, 255, 255))
            cv2.imshow("collect", view)

        # ---- keys ----
        # Only process/redraw when a new frame arrives; otherwise idle briefly
        # so the receiver thread isn't starved of CPU.
        key = cv2.waitKey(1 if new_frame else 10) & 0xFF
        if key != 255:
            redraw = True
        if key in (ord("q"), 27):
            break
        elif ord("0") <= key <= ord("9") and key - ord("0") < len(GESTURES):
            label_idx = key - ord("0")
            recording = False
            print("Selected:", GESTURES[label_idx])
        elif key == ord(" "):
            recording = not recording
            print("Recording" if recording else "Paused", GESTURES[label_idx])
        elif key == ord("c"):
            calibrating = CALIBRATION_FRAMES
            cal_frames = []
        elif key == ord("r"):
            cal = dict(DEFAULT_CALIBRATION)
            save_calibration(cal)
            print("Calibration reset to defaults")
        elif key == ord("m"):
            mask_only = not mask_only

    rx.stop()
    cv2.destroyAllWindows()
    print("\nSamples per gesture:")
    for g, n in count_existing().items():
        print(f"  {g:12s} {n}")


if __name__ == "__main__":
    main()
