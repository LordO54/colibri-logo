"""Evalúa uno o varios best.pt en test_video (global y por video) y en ds2 test.

Métricas de Ultralytics (conf 0.001, iou 0.6) más P y R a conf 0.25 con emparejamiento greedy a IoU 0.5.
Uso: python evaluar.py --weights viejo=RUTA nuevo=RUTA --out tabla.csv
"""
import argparse, csv, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

V2 = Path(r"D:\video_eval\reentreno\v2")
SETS = {"test_video": V2 / "data_test_video.yaml", "ds2_test": V2 / "data_ds2_test.yaml"}
LISTS = {"test_video": V2 / "test_video.txt",
         "ds2_test": Path(r"D:\video_eval\colibri_work\data\final_ds2\test\images")}


def iou(a, b):
    x1, y1 = np.maximum(a[:2], b[:2]); x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def gt_boxes(img, w, h):
    lab = Path(str(img).replace("\\images\\", "\\labels\\")).with_suffix(".txt")
    out = []
    for l in (lab.read_text().splitlines() if lab.exists() else []):
        p = l.split()
        if len(p) == 5:
            _, x, y, bw, bh = map(float, p)
            out.append(np.array([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
    return out


def pr_at(model, imgs, conf=0.25, imgsz=640):
    """TP/FP/FN a conf fija, agrupados por video (prefijo antes de _fNNNNN)."""
    st = defaultdict(lambda: [0, 0, 0])
    for i in range(0, len(imgs), 16):
        for r in model.predict(imgs[i:i + 16], imgsz=imgsz, conf=conf, verbose=False):
            h, w = r.orig_shape
            gts = gt_boxes(r.path, w, h)
            preds = sorted(zip(r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()), key=lambda z: -z[0])
            used, tp = set(), 0
            for _, pb in preds:
                best = max(((iou(pb, g), j) for j, g in enumerate(gts) if j not in used), default=(0, -1))
                if best[0] >= 0.5:
                    used.add(best[1]); tp += 1
            vid = re.sub(r"_f\d{5}$", "", Path(r.path).stem)
            for k in (vid, "GLOBAL"):
                st[k][0] += tp; st[k][1] += len(preds) - tp; st[k][2] += len(gts) - tp
    return st


def per_video_ap50(model, imgs, tmp):
    """mAP50 por video con val() de Ultralytics sobre listas temporales."""
    groups = defaultdict(list)
    for p in imgs:
        groups[re.sub(r"_f\d{5}$", "", Path(p).stem)].append(p)
    res = {}
    for vid, ps in groups.items():
        lst = tmp / f"{vid[:40]}.txt"; lst.write_text("\n".join(ps) + "\n")
        y = tmp / f"{vid[:40]}.yaml"; y.write_text(f"train: {lst}\nval: {lst}\nnames:\n  0: hummingbird\n")
        m = model.val(data=str(y), split="val", imgsz=640, conf=0.001, iou=0.6, batch=8, plots=False, verbose=False,
                      project=str(tmp / "runs"), name="pv", exist_ok=True)
        res[vid] = (m.box.mp, m.box.mr, m.box.map50, m.box.map)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True, help="etiqueta=ruta")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tmp = V2 / "_eval_tmp"; tmp.mkdir(exist_ok=True)
    imgs = {"test_video": LISTS["test_video"].read_text().split(),
            "ds2_test": sorted(str(p) for p in LISTS["ds2_test"].glob("*.jpg"))}
    rows = []
    for spec in a.weights:
        tag, w = spec.split("=", 1)
        model = YOLO(w)
        for name, y in SETS.items():
            m = model.val(data=str(y), split="test", imgsz=640, conf=0.001, iou=0.6, batch=8, plots=False,
                          verbose=False, project=str(tmp / "runs"), name=f"{tag}_{name}", exist_ok=True)
            st = pr_at(model, imgs[name])
            tp, fp, fn = st["GLOBAL"]
            rows.append(dict(modelo=tag, conjunto=name, video="GLOBAL", P=m.box.mp, R=m.box.mr, mAP50=m.box.map50,
                             mAP50_95=m.box.map, P_c025=tp / max(1, tp + fp), R_c025=tp / max(1, tp + fn),
                             n_img=len(imgs[name]), n_gt=tp + fn))
            if name == "test_video":
                for vid, (p, r, m50, m95) in per_video_ap50(model, imgs[name], tmp).items():
                    tp, fp, fn = st[vid]
                    rows.append(dict(modelo=tag, conjunto=name, video=vid, P=p, R=r, mAP50=m50, mAP50_95=m95,
                                     P_c025=tp / max(1, tp + fp), R_c025=tp / max(1, tp + fn),
                                     n_img=sum(1 for x in imgs[name] if Path(x).stem.startswith(vid)), n_gt=tp + fn))
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, list(rows[0])); wr.writeheader()
        for r in rows:
            wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    for r in rows:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
