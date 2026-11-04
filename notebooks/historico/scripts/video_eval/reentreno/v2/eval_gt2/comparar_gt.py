"""Compara la GT de humming v1 frente a v2 por archivo (nombre original) y verifica las listas de revisión."""
import re
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

V2 = Path(r"D:\video_eval\reentreno\v2")
OUT = V2 / "eval_gt2"
WEIGHTS = r"C:\Users\ACER\Documents\logo\video-val\resultados\yolo26n_ft_v2\best.pt"


def load(rf):
    d = {}
    for split in ("train", "valid", "test"):
        for lab in (rf / split / "labels").glob("*.txt"):
            stem = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", lab.stem)
            lines = sorted(l.strip() for l in lab.read_text().splitlines() if l.strip())
            img = next((rf / split / "images").glob(lab.stem + ".*"), None)
            d[stem] = dict(split_rf=split, lines=lines, img=img)
    return d


def main():
    a, b = load(V2 / "rf_humming"), load(V2 / "rf_humming_v2")
    print("v1:", len(a), "v2:", len(b), "solo en v1:", sorted(set(a) - set(b))[:5], "solo en v2:", sorted(set(b) - set(a))[:5])
    sm = pd.read_csv(V2 / "split_manifest.csv")
    stem_of = {Path(r).stem: (s, v) for r, s, v in zip(sm.ruta, sm.split, sm.video_id)}
    rows = []
    for k in sorted(set(a) & set(b)):
        if a[k]["lines"] != b[k]["lines"]:
            split, vid = stem_of.get(k, ("fuera_de_reparticion", k.rsplit("_f", 1)[0]))
            rows.append(dict(stem=k, video_id=vid, split=split, cajas_v1=len(a[k]["lines"]), cajas_v2=len(b[k]["lines"])))
    C = pd.DataFrame(rows)
    C["cambio"] = C.apply(lambda r: "añadida" if r.cajas_v2 > r.cajas_v1 else ("quitada" if r.cajas_v2 < r.cajas_v1 else "ajustada"), axis=1)
    C.to_csv(OUT / "cambios_v1_v2.csv", index=False)
    print("cambiadas:", len(C))
    print(C.groupby(["split", "video_id", "cambio"]).size().to_string())

    pend = []
    for f, need in (("revisar_gt_vacio_con_track.csv", "caja"), ("revisar_gt_vacio_train_valid.csv", "caja"),
                    ("revisar_gt_caja_sin_det.csv", "caja")):
        L = pd.read_csv(V2 / "union" / f)
        for r in L.itertuples():
            stem = f"{r.video_id}_f{int(r.frame):05d}"
            n2 = len(b[stem]["lines"]) if stem in b else None
            pend.append(dict(lista=f, stem=stem, cajas_v1=len(a[stem]["lines"]) if stem in a else None, cajas_v2=n2))
    P = pd.DataFrame(pend)
    P.to_csv(OUT / "verificacion_listas.csv", index=False)
    for f, g in P.groupby("lista", sort=False):
        print(f, "| total", len(g), "| con caja en v2", int((g.cajas_v2 > 0).sum()), "| siguen vacías:",
              g[g.cajas_v2 == 0].stem.tolist())

    # frames test aún vacíos en v2 con detección a conf 0.25
    model = YOLO(WEIGHTS)
    left = []
    for k, v in b.items():
        if stem_of.get(k, ("",))[0] == "test" and not v["lines"]:
            r = model.predict(str(v["img"]), imgsz=640, conf=0.25, verbose=False)[0]
            if len(r.boxes):
                left.append(dict(stem=k, conf_max=round(float(r.boxes.conf.max()), 3), n_det=len(r.boxes), img=str(v["img"])))
    Lf = pd.DataFrame(left, columns=["stem", "conf_max", "n_det", "img"]).sort_values("conf_max", ascending=False)
    Lf.to_csv(OUT / "test_vacios_v2_con_det.csv", index=False)
    n_empty = sum(1 for k, v in b.items() if stem_of.get(k, ("",))[0] == "test" and not v["lines"])
    print("test vacíos en v2:", n_empty, "| con detección >= 0,25:", len(Lf))
    print(Lf[["stem", "conf_max"]].to_string(index=False))


if __name__ == "__main__":
    main()
