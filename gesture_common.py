"""
Shared code for the wrist-camera gesture pipeline.

  - FrameReceiver : TCP server for the ESP32 stream (background thread)
  - decode_frame  : RGB565 (38,400 bytes) or JPEG (any other size) -> BGR
  - skin_mask     : YCrCb skin segmentation + cleanup
  - calibrate     : learn skin thresholds from a box in the frame
  - extract_features : fixed-length feature vector for the random forest

Everything else (collect.py, train.py, live.py) imports from here, so a
change to the mask or features automatically applies everywhere.
"""

import json
import os
import socket
import struct
import threading
import time

import cv2
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================

HOST = "0.0.0.0"
PORT = 5000

WIDTH = 160
HEIGHT = 120
RGB565_SIZE = WIDTH * HEIGHT * 2

# esp32-camera sends RGB565 big-endian. If colours look wrong
# (blue skin, etc.), flip this.
RGB565_BIG_ENDIAN = True

# Gesture classes. Index = number key used in collect.py.
# Keep "none" (no gesture / hand relaxed / random motion): without it the
# classifier is forced to pick a gesture for every frame.
GESTURES = [
    "none",        # key 0
    "gesture_1",   # key 1
    # "gesture_2", # key 2  (uncomment to add a second gesture later)
]

DATA_DIR = "data"
CALIBRATION_FILE = "calibration.json"
MODEL_FILE = "gesture_model.joblib"

# Default YCrCb skin box (widely used values). Calibration replaces these.
DEFAULT_CALIBRATION = {
    "cr_min": 133, "cr_max": 173,
    "cb_min": 77,  "cb_max": 127,
    "y_min": 30,   "y_max": 255,
    # cleanup (tune with tune_mask.py)
    "blur": 5,          # Gaussian blur size, odd; 0 = off
    "kernel": 5,        # morphology kernel size
    "open_iter": 1,     # removes speckle
    "close_iter": 2,    # fills small gaps
    "keep_largest": 1,  # 1 = keep only the biggest blob
    "fill_holes": 1,
}

COLOR_KEYS = ("y_min", "y_max", "cr_min", "cr_max", "cb_min", "cb_max")

# Feature mode:
#   "hog"  - grayscale edge-shape features, NO skin mask / colour needed (default)
#   "mask" - skin mask silhouette features
#   "both" - concatenation of the two
# train.py can override this per run (python train.py mask); live.py uses
# whatever mode the saved model was trained with.
FEATURE_MODE = "hog"

# Region of the frame to use, as fractions (x0, y0, x1, y1).
# The camera is wrist-mounted, so the hand is always in roughly the same
# place: crop to it to cut background. (0, 0, 1, 1) = full frame.
ROI = (0.0, 0.0, 1.0, 1.0)

# HOG settings
HOG_SIZE = (64, 48)              # ROI is resized to this (w, h) before HOG
HOG_CELL = 8
HOG_BINS = 9
THUMB_SIZE = (16, 12)            # small normalised intensity image, added to HOG

# Mask feature settings
GRID_W, GRID_H = 20, 15          # mask downsampled to this grid
MIN_DEFECT_DEPTH = 6.0           # px, for counting finger gaps

DISPLAY_SCALE = 4


# ==========================================
# FRAME RECEIVER
# ==========================================

def _recv_exact(sock, size):
    buf = bytearray(size)
    view = memoryview(buf)
    got = 0
    while got < size:
        n = sock.recv_into(view[got:], size - got)
        if n == 0:
            raise ConnectionError("ESP32 disconnected")
        got += n
    return buf


class FrameReceiver:
    """
    Accepts the ESP32 connection in a background thread and keeps only the
    newest decoded frame, so slow UI/processing never backs up the socket.

        rx = FrameReceiver(); rx.start()
        frame, frame_id = rx.latest()
    """

    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._running = False
        self.connected = False
        self.fps = 0.0

    def start(self):
        self._running = True
        threading.Thread(target=self._serve, daemon=True).start()
        return self

    def stop(self):
        self._running = False

    def latest(self):
        with self._lock:
            return self._frame, self._frame_id

    def _serve(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
        server.bind((self.host, self.port))
        server.listen(1)
        server.settimeout(1.0)
        print(f"Listening on port {self.port} ...")

        while self._running:
            try:
                client, addr = server.accept()
            except socket.timeout:
                continue

            print("ESP32 connected:", addr)
            self.connected = True
            client.settimeout(15)
            t_last = time.perf_counter()
            n = 0

            try:
                while self._running:
                    size = struct.unpack("!I", _recv_exact(client, 4))[0]
                    if size == 0 or size > 500_000:
                        print("Bad frame size:", size)
                        break
                    payload = _recv_exact(client, size)
                    frame = decode_frame(payload)
                    if frame is None:
                        continue

                    with self._lock:
                        self._frame = frame
                        self._frame_id += 1

                    n += 1
                    now = time.perf_counter()
                    if now - t_last >= 1.0:
                        self.fps = n / (now - t_last)
                        n = 0
                        t_last = now

            except (ConnectionError, OSError) as e:
                print("Connection ended:", e)
            finally:
                client.close()
                self.connected = False
                print("Waiting for ESP32 ...")

        server.close()


# ==========================================
# DECODING
# ==========================================

def decode_rgb565(data, width=WIDTH, height=HEIGHT, big_endian=RGB565_BIG_ENDIAN):
    dtype = ">u2" if big_endian else "<u2"
    px = np.frombuffer(data, dtype=dtype).reshape(height, width).astype(np.uint16)
    r = ((px >> 11) & 0x1F).astype(np.uint8)
    g = ((px >> 5) & 0x3F).astype(np.uint8)
    b = (px & 0x1F).astype(np.uint8)
    # expand to 8-bit with bit replication
    r = (r << 3) | (r >> 2)
    g = (g << 2) | (g >> 4)
    b = (b << 3) | (b >> 2)
    return np.dstack([b, g, r])  # BGR for OpenCV


def decode_frame(payload):
    """RGB565 if the size matches, otherwise assume JPEG."""
    if len(payload) == RGB565_SIZE:
        return decode_rgb565(payload)
    arr = np.frombuffer(payload, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None and img.shape[:2] != (HEIGHT, WIDTH):
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return img


# ==========================================
# SKIN MASK
# ==========================================

def load_calibration(path=CALIBRATION_FILE):
    if os.path.exists(path):
        with open(path) as f:
            cal = json.load(f)
        merged = dict(DEFAULT_CALIBRATION)
        merged.update(cal)
        return merged
    return dict(DEFAULT_CALIBRATION)


def save_calibration(cal, path=CALIBRATION_FILE):
    with open(path, "w") as f:
        json.dump(cal, f, indent=2)


def calibration_box(width=WIDTH, height=HEIGHT, frac=0.3):
    """Centre box the user fills with skin during calibration."""
    bw, bh = int(width * frac), int(height * frac)
    x0, y0 = (width - bw) // 2, (height - bh) // 2
    return x0, y0, bw, bh


def calibrate(frames, margin=6):
    """
    Learn YCrCb thresholds from the centre box of one or more frames.
    Uses 3rd-97th percentiles (robust to a few non-skin pixels) plus a margin.
    """
    x, y, w, h = calibration_box()
    samples = []
    for f in frames:
        roi = cv2.cvtColor(f[y:y + h, x:x + w], cv2.COLOR_BGR2YCrCb)
        samples.append(roi.reshape(-1, 3))
    s = np.concatenate(samples).astype(np.float32)

    lo = np.percentile(s, 3, axis=0)
    hi = np.percentile(s, 97, axis=0)

    cal = {
        "y_min":  int(max(0, lo[0] - 2 * margin)),
        "y_max":  255,
        "cr_min": int(max(0, lo[1] - margin)),
        "cr_max": int(min(255, hi[1] + margin)),
        "cb_min": int(max(0, lo[2] - margin)),
        "cb_max": int(min(255, hi[2] + margin)),
    }
    return cal


def raw_skin_threshold(bgr, cal):
    """Colour threshold only, no cleanup (useful for tuning)."""
    b = int(cal.get("blur", 5))
    if b >= 3:
        b |= 1
        bgr = cv2.GaussianBlur(bgr, (b, b), 0)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([cal["y_min"], cal["cr_min"], cal["cb_min"]], dtype=np.uint8)
    upper = np.array([cal["y_max"], cal["cr_max"], cal["cb_max"]], dtype=np.uint8)
    return cv2.inRange(ycrcb, lower, upper)


def skin_mask(bgr, cal):
    """
    Returns a uint8 mask (0/255) of the skin region.
    Steps: blur -> YCrCb threshold -> open/close -> keep largest blob -> fill holes.
    All steps are controlled by the calibration dict (see DEFAULT_CALIBRATION).
    """
    mask = raw_skin_threshold(bgr, cal)

    ks = max(1, int(cal.get("kernel", 5)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    if int(cal.get("open_iter", 1)) > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=int(cal["open_iter"]))
    if int(cal.get("close_iter", 2)) > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=int(cal["close_iter"]))

    if int(cal.get("keep_largest", 1)):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return np.zeros_like(mask)
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    if int(cal.get("fill_holes", 1)):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        mask = filled

    return mask


# ==========================================
# FEATURES
# ==========================================

N_SHAPE_FEATURES = 11 + 7  # shape stats + Hu moments
FEATURE_LENGTH = GRID_W * GRID_H + GRID_H + GRID_W + N_SHAPE_FEATURES


def extract_features(mask):
    """
    Fixed-length vector describing the hand silhouette:
      1. Coarse grid  : mask downsampled to GRID_W x GRID_H (fraction of skin per cell)
      2. Projections  : skin fraction per row and per column
      3. Shape stats  : area, centroid, bbox, extent, solidity, compactness,
                        convexity-defect count (~finger gaps), orientation
      4. Hu moments   : 7 rotation/scale-invariant shape moments (log scaled)
    """
    h, w = mask.shape
    m = (mask > 0).astype(np.float32)

    grid = cv2.resize(m, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA).ravel()
    rows = cv2.resize(m.mean(axis=1, keepdims=True), (1, GRID_H),
                      interpolation=cv2.INTER_AREA).ravel()
    cols = cv2.resize(m.mean(axis=0, keepdims=True), (GRID_W, 1),
                      interpolation=cv2.INTER_AREA).ravel()

    shape = np.zeros(N_SHAPE_FEATURES, dtype=np.float32)
    area = float(m.sum())

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if area > 20 and contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(mask, binaryImage=True)
        cx = M["m10"] / M["m00"] / w
        cy = M["m01"] / M["m00"] / h
        x, y, bw, bh = cv2.boundingRect(c)

        hull = cv2.convexHull(c)
        hull_area = max(cv2.contourArea(hull), 1.0)
        c_area = max(cv2.contourArea(c), 1.0)
        perim = cv2.arcLength(c, True)

        n_defects = 0
        if len(c) >= 5:
            hull_idx = cv2.convexHull(c, returnPoints=False)
            if hull_idx is not None and len(hull_idx) > 3:
                try:
                    defects = cv2.convexityDefects(c, hull_idx)
                    if defects is not None:
                        depths = defects.reshape(-1, 4)[:, 3] / 256.0
                        n_defects = int(np.sum(depths > MIN_DEFECT_DEPTH))
                except cv2.error:
                    pass

        mu20 = M["mu20"] / M["m00"]
        mu02 = M["mu02"] / M["m00"]
        mu11 = M["mu11"] / M["m00"]
        angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

        shape[:11] = [
            area / (w * h),             # area fraction
            cx, cy,                     # centroid
            bw / w, bh / h,             # bbox size
            c_area / (bw * bh),         # extent
            c_area / hull_area,         # solidity (low = fingers spread)
            perim ** 2 / (4 * np.pi * c_area) / 10.0,  # compactness
            n_defects / 5.0,            # finger gaps
            np.sin(2 * angle),          # orientation (wrap-safe)
            np.cos(2 * angle),
        ]

        hu = cv2.HuMoments(M).ravel()
        shape[11:] = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30) / 30.0

    feats = np.concatenate([grid, rows, cols, shape]).astype(np.float32)
    assert feats.shape[0] == FEATURE_LENGTH
    return feats


# ==========================================
# HOG FEATURES (no skin mask)
# ==========================================

_hog = cv2.HOGDescriptor(
    HOG_SIZE,
    (HOG_CELL * 2, HOG_CELL * 2),   # block = 2x2 cells
    (HOG_CELL, HOG_CELL),           # block stride
    (HOG_CELL, HOG_CELL),           # cell
    HOG_BINS,
)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def roi_pixels(width=WIDTH, height=HEIGHT):
    x0, y0, x1, y1 = ROI
    return int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)


def preprocess_gray(bgr):
    """Grayscale -> crop ROI -> resize -> local contrast boost (CLAHE).
    CLAHE evens out exposure so dim/bright frames look alike to HOG."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = roi_pixels(gray.shape[1], gray.shape[0])
    gray = gray[y0:y1, x0:x1]
    gray = cv2.resize(gray, HOG_SIZE, interpolation=cv2.INTER_AREA)
    return _clahe.apply(gray)


def hog_features(bgr):
    """
    HOG of the preprocessed ROI (edge directions per 8x8 cell - finger and
    knuckle outlines) + a tiny brightness-normalised thumbnail (coarse layout).
    Colour is never used.
    """
    g = preprocess_gray(bgr)
    h = _hog.compute(g).ravel()
    thumb = cv2.resize(g, THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    thumb = (thumb - thumb.mean()) / (thumb.std() + 1e-6)
    return np.concatenate([h, thumb.ravel()]).astype(np.float32)


def frame_features(bgr, cal, mode=FEATURE_MODE):
    """Single entry point used by train.py and live.py."""
    if mode == "hog":
        return hog_features(bgr)
    if mode == "mask":
        return extract_features(skin_mask(bgr, cal))
    if mode == "both":
        return np.concatenate([hog_features(bgr), extract_features(skin_mask(bgr, cal))])
    raise ValueError(f"Unknown feature mode: {mode}")


def feature_length(mode=FEATURE_MODE):
    dummy = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    return len(frame_features(dummy, DEFAULT_CALIBRATION, mode))


def hog_image(gray, cell=HOG_CELL, bins=HOG_BINS, cell_px=40):
    """
    Classic HOG visualisation: for each cell, a star of lines showing the
    EDGE directions in that cell, brightness = edge strength.
    (Edges are drawn perpendicular to the gradient, so they follow outlines.)
    Returns a BGR image (cells_x*cell_px by cells_y*cell_px).
    """
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = ang % 180.0
    b = np.minimum((ang / (180.0 / bins)).astype(np.int32), bins - 1)

    cy, cx = g.shape[0] // cell, g.shape[1] // cell
    hist = np.zeros((cy, cx, bins), np.float32)
    for i in range(cy):
        for j in range(cx):
            sl = (slice(i * cell, (i + 1) * cell), slice(j * cell, (j + 1) * cell))
            hist[i, j] = np.bincount(b[sl].ravel(), weights=mag[sl].ravel(), minlength=bins)

    scale = np.percentile(hist, 99) + 1e-6
    hist = np.clip(hist / scale, 0, 1)

    out = np.zeros((cy * cell_px, cx * cell_px), np.uint8)
    r = cell_px * 0.45
    for i in range(cy):
        for j in range(cx):
            ccx, ccy = (j + 0.5) * cell_px, (i + 0.5) * cell_px
            for k in np.argsort(hist[i, j]):          # weakest first, strongest on top
                v = hist[i, j, k]
                if v < 0.15:
                    continue
                edge = np.deg2rad((k + 0.5) * (180.0 / bins) + 90.0)
                dx, dy = r * np.cos(edge), r * np.sin(edge)
                cv2.line(out, (int(ccx - dx), int(ccy - dy)), (int(ccx + dx), int(ccy + dy)),
                         int(255 * v), 2, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def debug_panel(bgr, cal, mode=FEATURE_MODE):
    """What the classifier 'sees': skin mask in mask mode, HOG visualisation otherwise."""
    if mode == "mask":
        return cv2.cvtColor(skin_mask(bgr, cal), cv2.COLOR_GRAY2BGR)
    return hog_image(preprocess_gray(bgr))


def big(img, scale=DISPLAY_SCALE):
    """Resize any panel to the standard display size."""
    return cv2.resize(img, (WIDTH * scale, HEIGHT * scale), interpolation=cv2.INTER_NEAREST)


def two_panel(bgr, panel_bgr, scale=DISPLAY_SCALE, draw_roi=True):
    left = bgr.copy()
    if draw_roi and ROI != (0.0, 0.0, 1.0, 1.0):
        x0, y0, x1, y1 = roi_pixels()
        cv2.rectangle(left, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 255), 1)
    return np.hstack([big(left, scale), big(panel_bgr, scale)])


# ==========================================
# DISPLAY HELPERS
# ==========================================

def side_by_side(bgr, mask, scale=DISPLAY_SCALE):
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = bgr.copy()
    overlay[mask > 0] = (0.5 * overlay[mask > 0] + [0, 90, 0]).astype(np.uint8)
    view = np.hstack([overlay, mask_bgr])
    return cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def put_lines(img, lines, x=10, y=24, dy=24, color=(0, 255, 255)):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * dy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, y + i * dy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 1, cv2.LINE_AA)
