"""Funciones compartidas de SP-02.7, las mismas que usa colibri_profundidad_3d.ipynb."""
import cv2
import numpy as np


def read_frame(cap, f):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, fr = cap.read()
    return fr if ok else None


def motion_mask(frame, bg, x0, y0, x1, y1):
    # quita el cambio global de brillo; el umbral se adapta al ruido del anillo alrededor de la caja
    pad = max(10, int(0.5 * max(x1 - x0, y1 - y0)))
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(frame.shape[1], x1 + pad), min(frame.shape[0], y1 + pad)
    diff = cv2.absdiff(frame[Y0:Y1, X0:X1], bg[Y0:Y1, X0:X1]).astype(np.float32)
    diff = np.abs(diff - np.median(diff, axis=(0, 1))).max(axis=2)
    ring = np.ones(diff.shape, bool); ring[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
    thr = max(15.0, float(np.percentile(diff[ring], 95)) + 5) if ring.any() else 15.0
    m = (diff > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0].astype(bool)
