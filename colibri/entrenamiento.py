"""Prepara datasets en formato YOLO, entrena un detector y lo compara con otro. Lo usa notebooks/entrenar_detector.ipynb."""
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CARPETAS, cargar_env

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = (("train", ["train"]), ("valid", ["valid", "val"]), ("test", ["test"]))


def read_labels(lbl_path: Path) -> list:
    """Filas (clase, cx, cy, w, h) de una etiqueta YOLO; los polígonos se convierten a su caja envolvente."""
    if not lbl_path.exists():
        return []
    rows = []
    for line in lbl_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            c, x, y, w, h = parts
            rows.append((int(c), float(x), float(y), float(w), float(h)))
        elif len(parts) > 5:
            c, pts = int(parts[0]), np.array(parts[1:], float).reshape(-1, 2)
            (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
            rows.append((c, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return rows


def _raiz_dataset(nombre, d):
    if "carpeta" in d:
        raiz = Path(d["carpeta"])
        return raiz if raiz.is_absolute() else CARPETAS["datasets"] / raiz
    from roboflow import Roboflow
    cargar_env()
    clave = os.environ.get("ROBOFLOW_API_KEY")
    if not clave:
        raise ValueError("Para descargar de Roboflow, copia .env.example como .env y pon tu ROBOFLOW_API_KEY.")
    v = Roboflow(api_key=clave).workspace(d["workspace"]).project(d["project"]).version(d["version"])
    return Path(v.download(d.get("formato", "yolo26"), location=str(CARPETAS["datasets"] / nombre), overwrite=False).location)


def preparar_datos(nombre_modelo, datasets) -> tuple:
    """Copia cada fuente, unifica las clases en 0 y escribe listas y data.yaml. No modifica tus carpetas. Devuelve (data.yaml, tabla)."""
    if not datasets:
        raise ValueError("DATASETS está vacío: indica al menos una carpeta o un proyecto de Roboflow.")
    dst = CARPETAS["salidas"] / "entrenamiento" / nombre_modelo / "datos"; listas, filas = {"train": [], "valid": [], "test": []}, []
    for nombre, d in datasets.items():
        raiz = _raiz_dataset(nombre, d)
        if not raiz.is_dir():
            raise FileNotFoundError(f"{nombre}: no existe la carpeta {raiz}")
        for split, alias in SPLITS:
            img_dir = next((raiz / a / "images" for a in alias if (raiz / a / "images").is_dir()), None)
            if img_dir is None:
                continue
            di, dl = dst / split / "images", dst / split / "labels"; di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
            imgs, cajas, vacias = [], 0, 0
            for p in sorted(x for x in img_dir.iterdir() if x.suffix.lower() in IMG_EXTS):
                out = di / f"{nombre}__{p.name}"
                if not out.exists():
                    shutil.copy2(p, out)
                rows = read_labels(img_dir.parent / "labels" / f"{p.stem}.txt")
                (dl / f"{out.stem}.txt").write_text("\n".join(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for _, x, y, w, h in rows))
                imgs.append(str(out)); cajas += len(rows); vacias += not rows
            listas[split] += imgs * (d.get("repetir", 1) if split == "train" else 1)
            filas.append(dict(dataset=nombre, conjunto=split, imagenes=len(imgs), cajas=cajas, sin_ave=vacias))
    T = pd.DataFrame(filas)
    if not listas["train"] or not listas["valid"]:
        raise ValueError(f"Hacen falta imágenes en train y en valid. Encontrado:\n{T}")
    if T[T.conjunto == "train"].cajas.sum() == 0:
        raise ValueError("train no tiene ninguna caja: revisa las carpetas labels/.")
    for s, l in listas.items():
        (dst / f"{s}.txt").write_text("\n".join(l) + "\n" if l else "")
    prueba = "test" if listas["test"] else "valid"
    if prueba == "valid":
        print("Aviso: no hay test; la comparación se hará en valid, que también se usa para elegir el mejor peso.")
    y = dst / "data.yaml"
    y.write_text(f"train: {dst / 'train.txt'}\nval: {dst / 'valid.txt'}\ntest: {dst / (prueba + '.txt')}\nnames:\n  0: hummingbird\n", encoding="utf-8")
    return y, T


def entrenar(nombre, pesos_ini, data, out, **kw):
    """Entrena o reanuda con la CLI de Ultralytics en otro proceso. Devuelve la ruta de best.pt."""
    run = Path(out) / nombre; done = run / "TRAINING_DONE"; last = run / "weights" / "last.pt"
    if done.exists():
        return run / "weights" / "best.pt"
    if last.exists():
        args = ["train", "resume", f"model={last}"]
    else:
        args = ["detect", "train", f"model={pesos_ini}", f"data={data}", f"project={out}", f"name={nombre}", "exist_ok=True",
                *[f"{k}={v}" for k, v in kw.items()]]
    print("yolo", " ".join(args), flush=True)
    subprocess.run([str(Path(sys.executable).with_name("yolo")), *args], check=True)
    done.write_text(dt.datetime.now().isoformat(timespec="seconds"))
    return run / "weights" / "best.pt"


def comparar_detectores(detectores: dict, data, out) -> pd.DataFrame:
    """Mismas imágenes y umbrales (confianza 0,001, IoU 0,6) para cada detector de {nombre: ruta}."""
    from ultralytics import YOLO
    filas = []
    for tag, w in detectores.items():
        m = YOLO(str(w)).val(data=str(data), split="test", imgsz=640, conf=0.001, iou=0.6, batch=8, plots=False, verbose=False,
                             project=str(Path(out) / "_comparacion"), name=tag, exist_ok=True)
        filas.append(dict(detector=tag, precision=m.box.mp, recall=m.box.mr, mAP50=m.box.map50, mAP50_95=m.box.map))
    return pd.DataFrame(filas)
