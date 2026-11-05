"""Compara el criterio de caja entre los datasets de entrenamiento (ds1, ds2) y la prueba de video propia.

Para cada fuente: predicción de best.pt frente a la etiqueta, emparejadas por distancia de centros.
Si el modelo reproduce el criterio de sus etiquetas de entrenamiento, en ds1/ds2 la razón pred/GT es ~1;
si en la prueba de video sale >>1, las etiquetas de la prueba siguen otro criterio (p. ej. sin alas extendidas).
Uso: D:\\video_eval\\.venv\\Scripts\\python comparar_criterio_cajas.py
"""
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

PROJ = Path(__file__).resolve().parent
DATA = Path(r"D:\video_eval\colibri_work\data\final")
BEST = PROJ / "resultados" / "yolo26n_ds2_ds1" / "best.pt"
OUT = PROJ / "resultados" / "criterio_cajas"; OUT.mkdir(parents=True, exist_ok=True)
random.seed(42)

def sources():
    imgs = sorted(p for s in ("valid", "test") for p in (DATA / s / "images").glob("*.jpg"))
    yield "ds1 (frames de video, eval)", [p for p in imgs if p.name.startswith("ds1__")]
    yield "ds2 (fotos, eval)", [p for p in imgs if p.name.startswith("ds2__")]
    tv = sorted((DATA / "test_video" / "images").glob("*.jpg"))
    for v in ("vv01", "vv02", "vv03"):
        yield f"prueba {v}", [p for p in tv if p.name.startswith(v)]

def read_gt(img: Path, W: int, H: int) -> np.ndarray:
    lbl = img.parent.parent / "labels" / (img.stem + ".txt")
    rows = [list(map(float, l.split()[1:5])) for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
    a = np.array(rows).reshape(-1, 4)
    return np.c_[a[:, 0] * W, a[:, 1] * H, a[:, 2] * W, a[:, 3] * H]        # cx, cy, w, h en px

def iou_cxcywh(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    i = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    return i / (a[2] * a[3] + b[2] * b[3] - i)

model = YOLO(str(BEST))
rows, examples = [], {}
for name, imgs in sources():
    examples[name] = random.sample(imgs, min(4, len(imgs)))
    for p in imgs:
        r = model.predict(str(p), conf=0.25, imgsz=640, verbose=False)[0]
        H, W = r.orig_shape
        gt, pr = read_gt(p, W, H), r.boxes.xywh.cpu().numpy()
        used = set()
        for g in gt:
            if len(pr) == 0: break
            d = np.hypot(pr[:, 0] - g[0], pr[:, 1] - g[1]); d[list(used)] = np.inf
            j = int(np.argmin(d))
            if d[j] > max(g[2], g[3]): continue                          # centro demasiado lejos: no es la misma ave
            used.add(j)
            rows.append({"fuente": name, "img": p.name, "gt_w": g[2], "gt_h": g[3], "pr_w": pr[j, 2], "pr_h": pr[j, 3],
                         "razon_area": pr[j, 2] * pr[j, 3] / (g[2] * g[3]), "razon_w": pr[j, 2] / g[2],
                         "razon_h": pr[j, 3] / g[3], "iou": iou_cxcywh(g, pr[j])})
df = pd.DataFrame(rows)
df.to_csv(OUT / "pares_pred_gt.csv", index=False)
summ = df.groupby("fuente", sort=False).agg(n=("iou", "size"), razon_area_mediana=("razon_area", "median"),
                                            razon_w_mediana=("razon_w", "median"), razon_h_mediana=("razon_h", "median"),
                                            iou_mediana=("iou", "median"), frac_iou_menor_05=("iou", lambda x: (x < .5).mean()))
summ.to_csv(OUT / "resumen_criterio_cajas.csv")
print(summ.round(2).to_string())

# Mosaico: etiqueta en verde, predicción en rojo
tiles = []
for name, imgs in examples.items():
    row = []
    for p in imgs:
        im = cv2.imread(str(p)); H, W = im.shape[:2]
        r = model.predict(str(p), conf=0.25, imgsz=640, verbose=False)[0]
        for boxes, col in ((read_gt(p, W, H), (0, 200, 0)), (r.boxes.xywh.cpu().numpy(), (0, 0, 255))):
            for cx, cy, w, h in boxes:
                cv2.rectangle(im, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), col, max(2, W // 300))
        im = cv2.resize(im, (480, 270))
        cv2.putText(im, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        row.append(im)
    row += [np.zeros((270, 480, 3), np.uint8)] * (4 - len(row))
    tiles.append(np.hstack(row))
cv2.imwrite(str(OUT / "mosaico_criterio_cajas.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("Guardado en", OUT)
