"""
Train a random forest on the collected gesture images.

  python train.py          # uses FEATURE_MODE from gesture_common.py
  python train.py hog      # or force a mode: hog / mask / both

Loads data/<gesture>/*.png, extracts features (HOG and/or skin-mask), evaluates, then trains on everything and saves
gesture_model.joblib.

Evaluation is grouped by recording session: frames from one recording burst
are near-duplicates, so a random split would give fake ~100% accuracy.
Record each gesture in at least 2-3 separate sessions for an honest score.
"""

import os
import sys
import time

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

from gesture_common import (
    DATA_DIR, FEATURE_MODE, GESTURES, MODEL_FILE,
    frame_features, load_calibration,
)

MIN_SAMPLES_PER_CLASS = 20


def session_of(filename):
    # "<session>_<n>.png" where session = YYYYmmdd_HHMMSS
    return filename.rsplit("_", 1)[0]


def load_dataset(cal, mode):
    X, y, groups = [], [], []
    for label, g in enumerate(GESTURES):
        d = os.path.join(DATA_DIR, g)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        for f in files:
            img = cv2.imread(os.path.join(d, f))
            if img is None:
                continue
            X.append(frame_features(img, cal, mode))
            y.append(label)
            groups.append(session_of(f))
        print(f"  {g:12s} {len(files):5d} images")
    return np.array(X), np.array(y), np.array(groups)


def make_model():
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=0,
    )


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else FEATURE_MODE
    if mode not in ("hog", "mask", "both"):
        sys.exit("Mode must be hog, mask or both")
    cal = load_calibration()
    print("Feature mode:", mode)
    if mode != "hog":
        print("Calibration:", cal)
    print("Loading images ...")
    t0 = time.time()
    X, y, groups = load_dataset(cal, mode)
    print(f"Feature extraction: {len(X)} samples x {X.shape[1] if len(X) else 0} features "
          f"in {time.time() - t0:.1f}s")

    if len(X) == 0:
        sys.exit("No data. Run collect.py first.")

    present = sorted(set(y))
    for lbl in present:
        n = int((y == lbl).sum())
        if n < MIN_SAMPLES_PER_CLASS:
            print(f"WARNING: only {n} samples for '{GESTURES[lbl]}'")
    if len(present) < 2:
        sys.exit("Need at least 2 gestures with data.")

    names = [GESTURES[i] for i in present]

    # ---- evaluation ----
    n_sessions = len(set(groups))
    if n_sessions >= 2:
        k = min(5, n_sessions)
        cv = GroupKFold(n_splits=k)
        print(f"\nCross-validating by session ({k} folds, {n_sessions} sessions) ...")
        pred = cross_val_predict(make_model(), X, y, cv=cv, groups=groups)
    else:
        print("\nWARNING: only one recording session - accuracy below will be optimistic.")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        pred = cross_val_predict(make_model(), X, y, cv=cv)

    print("\n" + classification_report(y, pred, labels=present, target_names=names, digits=3))

    cm = confusion_matrix(y, pred, labels=present)
    print("Confusion matrix (rows = true, cols = predicted):")
    w = max(len(n) for n in names) + 2
    print(" " * w + "".join(f"{n[:7]:>8s}" for n in names))
    for n, row in zip(names, cm):
        print(f"{n:<{w}s}" + "".join(f"{v:8d}" for v in row))

    # ---- final model on all data ----
    print("\nTraining final model on all data ...")
    model = make_model().fit(X, y)

    joblib.dump({
        "model": model,
        "gestures": GESTURES,
        "feature_mode": mode,
        "feature_length": int(X.shape[1]),
        "calibration": cal,
        "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, MODEL_FILE)
    print(f"Saved {MODEL_FILE}")


if __name__ == "__main__":
    main()
