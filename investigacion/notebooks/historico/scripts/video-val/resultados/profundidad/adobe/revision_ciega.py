"""Hojas de revisión a ciegas: frames con el ave cerca de la flor, numerados al azar y SIN mostrar ninguna cifra de profundidad.

Se hacen para los videos que pasan el filtro y para i5bE de test. La selección depende solo de la posición de la caja
frente a la flor, no de dz. El dz de cada id va a clave_revision_ciega.csv, que no se imprime ni se abre hasta tener las
respuestas de Yilber.
"""
import json, random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

V2 = Path(r"D:\video_eval\reentreno\v2")
VIDEOS = Path(r"C:\Users\ACER\Documents\logo\video-val")
ADOBE = V2 / "profundidad" / "adobe"
OUT = ADOBE / "revision_ciega"; OUT.mkdir(exist_ok=True)
N = 20
REF = {"i5bE": (0.54, 0.38, 0.69, 0.58), "A129": (0.60, 0.15, 0.71, 0.68), "A512": (0.71, 0.40, 0.82, 0.50),
       "vv03": (0.42, 0.25, 0.55, 0.58), "A160": (0.60, 0.14, 0.78, 0.48)}
paths = json.loads((ADOBE / "paths.json").read_text(encoding="utf-8"))
ids = {"A129": "vv01_adobe1294608945", "A512": "vv02_adobe512464022", "vv03": "vv03_florisuga_file", "A160": "adobestock_1600207728"}


def source(name):
    if name == "i5bE":
        return V2 / "profundidad" / "salidas" / "stats_i5bE.csv", next(p for p in VIDEOS.glob("test_*") if "i5befvb5mxg" in p.name.lower())
    return ADOBE / "salidas" / f"stats_{name}.csv", Path(paths[ids[name]])


def main():
    passing = pd.read_csv(ADOBE / "salidas" / "filtro.csv").query("pasa_filtro").video.tolist()
    names = passing + ["i5bE"]
    print("videos en la revisión:", names)
    rng = random.Random(27)
    tmpl, key, next_id = [], [], 1
    for name in names:
        f, path = source(name)
        S = pd.read_csv(f)
        S["dz_clave"] = S.c_mask_med - S.ref_t
        cap = cv2.VideoCapture(str(path)); W, H = int(cap.get(3)), int(cap.get(4))
        x0, y0, x1, y1 = REF[name]; rx0, ry0, rx1, ry1 = x0 * W, y0 * H, x1 * W, y1 * H
        # distancia del centro de la caja a la región de la flor, en múltiplos del lado de la caja
        dx = np.maximum(np.maximum(rx0 - S.cx, S.cx - rx1), 0); dy = np.maximum(np.maximum(ry0 - S.cy, S.cy - ry1), 0)
        S["cerca"] = np.hypot(dx, dy) / ((S.w + S.h) / 2)
        near = S[(S.cerca < 0.6) & S.ref_t.notna()].sort_values("frame").reset_index(drop=True)
        sel = near.iloc[np.linspace(0, len(near) - 1, min(N, len(near))).astype(int)].to_dict("records")
        rng.shuffle(sel)  # sin orden temporal, para no inducir la respuesta
        tiles = []
        for r in sel:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"])); _, fr = cap.read()
            bx0, by0, bx1, by1 = int(r["cx"] - r["w"] / 2), int(r["cy"] - r["h"] / 2), int(r["cx"] + r["w"] / 2), int(r["cy"] + r["h"] / 2)
            X0, Y0, X1, Y1 = max(0, min(bx0, int(rx0)) - 40), max(0, min(by0, int(ry0)) - 40), min(W, max(bx1, int(rx1)) + 40), min(H, max(by1, int(ry1)) + 40)
            crop = fr[Y0:Y1, X0:X1]; s = min(380 / crop.shape[1], 300 / crop.shape[0])
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            tile = np.full((320, 400, 3), 255, np.uint8)
            yo, xo = 20 + (300 - crop.shape[0]) // 2, (400 - crop.shape[1]) // 2
            tile[yo:yo + crop.shape[0], xo:xo + crop.shape[1]] = crop
            cv2.putText(tile, f"#{next_id}", (6, 18), 0, 0.6, (0, 0, 0), 2)
            tiles.append(tile)
            tmpl.append(dict(id=next_id, video=name, frame=int(r["frame"]), posicion=""))
            key.append(dict(id=next_id, video=name, frame=int(r["frame"]), dz=r["dz_clave"]))
            next_id += 1
        cap.release()
        while len(tiles) % 5:
            tiles.append(np.full_like(tiles[0], 255))
        cv2.imwrite(str(OUT / f"revision_ciega_{name}.jpg"), np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(name, "cerca de la flor:", len(near), "| en la hoja:", len(sel))
    pd.DataFrame(tmpl).to_csv(OUT / "plantilla_revision_ciega.csv", index=False)
    pd.DataFrame(key).to_csv(OUT / "clave_revision_ciega.csv", index=False)
    print("escritas", len(tmpl), "filas; clave guardada sin mostrar")


if __name__ == "__main__":
    main()
