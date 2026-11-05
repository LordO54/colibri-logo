# Repite el gate del notebook 02 y el eval S2 con las IMAGENES ORIGINALES de S4/S2 para aislar la causa de los X.
# Copia los originales a runs/_verif_gate (no escribe caches en las carpetas originales).
import json, re, shutil, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from ultralytics import YOLO
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf8")
    Z = Path(r"D:\video_eval"); REPO = Z / "colibri-logo"; OUT = Z / "runs/_verif_gate"; OUT.mkdir(parents=True, exist_ok=True)
    nb = json.load(open(REPO / "notebooks/02_detector.ipynb", encoding="utf8"))
    src = lambda i: "".join(nb["cells"][i]["source"])
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    def read_labels(lbl_path):
        rows = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                p = line.split()
                if len(p) == 5:
                    rows.append((int(p[0]), *map(float, p[1:])))
        return rows
    g = globals()
    exec(src(3), g)   # evaluate, yolo_pred_fn
    exec(src(11).split("REF4 =")[0], g)  # iou, gt_boxes, pr_at, per_video_ap50
    W = {"viejo": Z / "modelos/yolo26n_ds2_ds1/best.pt", "nuevo": REPO / "modelos/yolo26n_ft_v2/best.pt"}
    K = ["modelo", "conjunto", "video"]
    for v, raiz, ref, regen in (("v1", Z / "reentreno/v2/humming/test", "comparacion.csv", "gate_v1.csv"),
                                ("v2", Z / "reentreno/v2/eval_gt2/humming/test", "deteccion_gt2.csv", "gate_v2.csv")):
        d = OUT / f"humming_{v}/test"
        if not d.exists():
            shutil.copytree(raiz, d, ignore=shutil.ignore_patterns("*.cache"))
        imgs = sorted(str(p) for p in (d / "images").glob("*.jpg"))
        # mismo ORDEN que el notebook y que S4: test_video.txt
        L = OUT / f"lista_{v}"; L.mkdir(exist_ok=True)
        (L / "test_video.txt").write_text("\n".join(imgs) + "\n")
        y = L / "d.yaml"; y.write_text(f"train: {L/'test_video.txt'}\nval: {L/'test_video.txt'}\ntest: {L/'test_video.txt'}\nnames:\n  0: hummingbird\n")
        rows = []; tmp = OUT / "_tmp"; tmp.mkdir(exist_ok=True)
        for tag, w in W.items():
            model = YOLO(str(w))
            m = model.val(data=str(y), split="test", imgsz=640, conf=0.001, iou=0.6, batch=8, plots=False, verbose=False, project=str(tmp / "runs"), name=f"{tag}", exist_ok=True)
            st = pr_at(model, imgs); tp, fp, fn = st["GLOBAL"]
            rows.append(dict(modelo=tag, conjunto="test_video", video="GLOBAL", mAP50=m.box.map50, P_c025=tp / max(1, tp + fp), R_c025=tp / max(1, tp + fn)))
            for vid, (p, r, m50, m95) in per_video_ap50(model, imgs, tmp).items():
                tp, fp, fn = st[vid]
                rows.append(dict(modelo=tag, conjunto="test_video", video=vid, mAP50=m50, P_c025=tp / max(1, tp + fp), R_c025=tp / max(1, tp + fn)))
        A = pd.DataFrame(rows); B = pd.read_csv(REPO / "resultados/S4_reentreno_v2" / ref); C = pd.read_csv(Z / "runs/02" / regen)
        M = B.merge(A, on=K, suffixes=("_ref", "_orig")).merge(C[K + ["mAP50", "P_c025", "R_c025"]], on=K)
        for c in ("mAP50", "P_c025", "R_c025"):
            M["orig-ref " + c] = (M[c + "_orig"] - M[c + "_ref"]).round(4); M["regen-ref " + c] = (M[c] - M[c + "_ref"]).round(4)
        print(v); print(M[["modelo", "video"] + [x for x in M if x.startswith(("orig-", "regen-"))]].assign(video=lambda d: d.video.str[-30:-12]).to_string(), flush=True)
    # S2: prueba vv
    ORIG = Z / "colibri_work/data/final/test_video"
    print("existe prueba vv original:", ORIG.exists())
    NUEVA = Z / "datos/frames/vv_prueba"
    for name, w in (("yolo26n_ds2_ds1", Z / "modelos/yolo26n_ds2_ds1/best.pt"), ("yolo26n_ds2", Z / "modelos/yolo26n_ds2/best.pt")):
        for tag, d in (("original", ORIG), ("regenerada", NUEVA)):
            if d.exists():
                r, _, _ = evaluate(yolo_pred_fn(YOLO(str(w)), imgsz=640), d)
                print(name, tag, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()}, flush=True)
