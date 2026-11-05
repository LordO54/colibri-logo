"""Mosaico de 12 frames test_: GT en verde, predicciones (conf >= 0.25) del viejo a la izquierda y del nuevo a la derecha."""
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from evaluar import V2, gt_boxes, iou

OLD = r"C:\Users\ACER\Documents\logo\video-val\resultados\yolo26n_ds2_ds1\best.pt"
NEW = str(V2 / "runs" / "yolo26n_ft_v2" / "weights" / "best.pt")
PER_VIDEO = {"doj1xa6ksqm": 2, "gu30nofexla": 3, "i5befvb5mxg": 3, "j4pkgwgjaus": 4}


def hits(r, gts):
    return sum(any(iou(p, g) >= 0.5 for p in r.boxes.xyxy.cpu().numpy()) for g in gts)


def draw(img, gts, r, label):
    im = img.copy()
    t = max(2, im.shape[1] // 400)
    for g in gts:
        cv2.rectangle(im, tuple(map(int, g[:2])), tuple(map(int, g[2:])), (0, 255, 0), t)
    for c, b in zip(r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
        cv2.rectangle(im, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 0, 255), t)
        cv2.putText(im, f"{c:.2f}", (int(b[0]), max(15, int(b[1]) - 5)), 0, 0.8, (0, 0, 255), 2)
    im = cv2.resize(im, (480, 270))
    cv2.putText(im, label, (6, 22), 0, 0.55, (0, 255, 255), 2)
    return im


def main():
    old, new = YOLO(OLD), YOLO(NEW)
    imgs = (V2 / "test_video.txt").read_text().split()
    picks = []
    for key, k in PER_VIDEO.items():
        cand = []
        for p in [x for x in imgs if key in Path(x).stem]:
            ro = old.predict(p, imgsz=640, conf=0.25, verbose=False)[0]
            rn = new.predict(p, imgsz=640, conf=0.25, verbose=False)[0]
            h, w = ro.orig_shape
            gts = gt_boxes(p, w, h)
            if not gts:
                continue
            # prioriza frames donde los modelos difieren; desempata por tamaño de ave (pequeñas primero)
            diff = abs(hits(rn, gts) - hits(ro, gts)) + abs(len(rn.boxes) - len(ro.boxes))
            area = min((g[2] - g[0]) * (g[3] - g[1]) for g in gts)
            cand.append((-diff, area, p, ro, rn, gts))
        cand.sort(key=lambda z: (z[0], z[1]))
        picks += cand[:k]
    rows = []
    for _, _, p, ro, rn, gts in picks:
        img = cv2.imread(p)
        name = re.sub(r"test_ytdown_com_youtube_media_", "", Path(p).stem)
        rows.append(np.hstack([draw(img, gts, ro, f"VIEJO {name}"), draw(img, gts, rn, f"NUEVO {name}")]))
    cv2.imwrite(str(V2 / "comparacion.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("ok", len(rows))


if __name__ == "__main__":
    main()
