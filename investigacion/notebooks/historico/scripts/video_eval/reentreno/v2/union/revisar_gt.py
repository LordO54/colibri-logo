"""Listas para revisar la GT de humming con yolo26n_ft_v2 (sin entrenar ni reajustar nada).

1) revisar_gt_vacio_train_valid.csv: frames de train/valid (split_manifest, incluidas las vv seleccionadas) con GT vacía
   y alguna detección a conf >= 0,25, ordenados por confianza; hoja de contacto gt_vacio_train_valid.jpg.
2) revisar_gt_caja_sin_det.csv: frames de train/valid/test con caja GT y ninguna detección a conf >= 0,1;
   hoja de contacto gt_caja_sin_det.jpg.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

V2 = Path(r"D:\video_eval\reentreno\v2")
OUT = V2 / "union"
WEIGHTS = r"C:\Users\ACER\Documents\logo\video-val\resultados\yolo26n_ft_v2\best.pt"


def boxes_of(path):
    lab = Path(path.replace("\\images\\", "\\labels\\")).with_suffix(".txt")
    return [list(map(float, l.split()[1:])) for l in lab.read_text().splitlines() if l.strip()]


def sheet(rows, out, color_det=(0, 0, 255), n=24):
    tiles = []
    for r in rows[:n]:
        im = cv2.imread(r["ruta"]); h, w = im.shape[:2]; t = max(2, w // 400)
        for x, y, bw, bh in r["gt"]:
            cv2.rectangle(im, (int((x - bw / 2) * w), int((y - bh / 2) * h)), (int((x + bw / 2) * w), int((y + bh / 2) * h)), (0, 255, 0), t)
        for c, (x1, y1, x2, y2) in r["det"]:
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color_det, t)
            cv2.putText(im, f"{c:.2f}", (int(x1), max(20, int(y1) - 6)), 0, w / 1000, color_det, 2)
        im = cv2.resize(im, (480, 270))
        cv2.putText(im, f"{r['n']:02d} {r['video_id'][-24:]} f{r['frame']} {r['split']}", (5, 18), 0, 0.5, (0, 255, 255), 2)
        tiles.append(im)
    if not tiles:
        return
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    sm = pd.read_csv(V2 / "split_manifest.csv")
    model = YOLO(WEIGHTS)
    empty, missed = [], []
    for r in sm.itertuples():
        gt = boxes_of(r.ruta)
        res = model.predict(r.ruta, imgsz=640, conf=0.1, verbose=False)[0]
        det = [(float(c), b.tolist()) for c, b in zip(res.boxes.conf.cpu().numpy(), res.boxes.xyxy.cpu().numpy())]
        row = dict(video_id=r.video_id, split=r.split, origen=r.origen, frame=r.frame, rf_file=r.rf_file, ruta=r.ruta,
                   gt=gt, det=det, n_det=len(det), conf_max=round(max((c for c, _ in det), default=0.0), 3))
        if not gt and r.split in ("train", "valid"):
            d25 = [d for d in det if d[0] >= 0.25]
            if d25:
                empty.append({**row, "det": d25, "n_det": len(d25)})
        if gt and not det:
            missed.append(row)
    cols = ["n", "video_id", "split", "origen", "frame", "n_det", "conf_max", "rf_file", "ruta"]
    empty.sort(key=lambda z: -z["conf_max"])
    for i, r in enumerate(empty):
        r["n"] = i
    missed.sort(key=lambda z: (z["split"], z["video_id"], z["frame"]))
    for i, r in enumerate(missed):
        r["n"] = i
    pd.DataFrame(empty, columns=cols + ["gt", "det"])[cols].to_csv(OUT / "revisar_gt_vacio_train_valid.csv", index=False)
    pd.DataFrame(missed, columns=cols + ["gt", "det"])[[c for c in cols if c not in ("n_det", "conf_max")]].to_csv(
        OUT / "revisar_gt_caja_sin_det.csv", index=False)
    # hoja de contacto: todas si caben (hasta 48), en orden de confianza
    sheet(empty, OUT / "gt_vacio_train_valid.jpg", n=48)
    sheet(missed, OUT / "gt_caja_sin_det.jpg", n=48)
    e, m = pd.DataFrame(empty), pd.DataFrame(missed)
    tot_empty = sm[(sm.cajas == 0) & sm.split.isin(["train", "valid"])].groupby("video_id").size()
    print("GT vacía train/valid con detección >= 0,25:", len(e), "de", int(tot_empty.sum()))
    if len(e):
        print(e.groupby(["video_id", "split"]).size().to_string())
    print("GT con caja y sin detección >= 0,1:", len(m))
    if len(m):
        print(m.groupby(["video_id", "split"]).size().to_string())


if __name__ == "__main__":
    main()
