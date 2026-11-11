"""
Just look at the HOG features live. No recording, no model.

  python view_hog.py

Panels:  camera | what HOG gets (gray + contrast boost) | HOG edges
In the HOG panel each cell shows the edge directions found there;
brighter line = stronger edge. Finger outlines should stand out.

Keys:  q/ESC quit,  s save a screenshot
"""

import time

import cv2
import numpy as np

from gesture_common import (
    HEIGHT, ROI, WIDTH, FrameReceiver, big, hog_image, preprocess_gray,
    put_lines, roi_pixels,
)


def main():
    rx = FrameReceiver().start()
    last_id = -1

    while True:
        frame, fid = rx.latest()
        if frame is None:
            blank = np.zeros((HEIGHT * 4, WIDTH * 4, 3), np.uint8)
            put_lines(blank, ["Waiting for ESP32 ..."])
            cv2.imshow("HOG", blank)
            if cv2.waitKey(100) & 0xFF in (ord("q"), 27):
                break
            continue

        if fid == last_id:
            if cv2.waitKey(5) & 0xFF in (ord("q"), 27):
                break
            continue
        last_id = fid

        gray = preprocess_gray(frame)

        cam = frame.copy()
        if ROI != (0.0, 0.0, 1.0, 1.0):
            x0, y0, x1, y1 = roi_pixels()
            cv2.rectangle(cam, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 255), 1)

        view = np.hstack([
            big(cam),
            big(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
            big(hog_image(gray)),
        ])
        put_lines(view, [f"camera  {rx.fps:.1f} fps"])
        put_lines(view, ["HOG input"], x=WIDTH * 4 + 10)
        put_lines(view, ["HOG edges"], x=WIDTH * 8 + 10)
        cv2.imshow("HOG", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            name = time.strftime("hog_%H%M%S.png")
            cv2.imwrite(name, view)
            print("Saved", name)

    rx.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
