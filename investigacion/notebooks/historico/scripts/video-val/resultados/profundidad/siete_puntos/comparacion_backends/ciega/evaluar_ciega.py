"""Juez entre modelos: dz en los frames de la revisión a ciegas de i5bE frente a las etiquetas de Yilber.

Interpretación, fijada antes de abrir la clave: "posada delante" = delante, con posado = 1; "detras" = detrás; "al lado" se excluye
del acierto de signo y se reporta aparte. delante = dz < 0, detrás = dz > 0. El p-valor es binomial bilateral frente a 0,5 e ignora
que los frames vienen del mismo video y que se comparan 4 modelos. La clave antigua solo se usa como control de cordura de yolo26s.
"""
import glob, os, sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

V2 = Path(r"D:\video_eval\reentreno\v2")
SP = V2 / "profundidad" / "siete_puntos"; os.chdir(SP)
RC = V2 / "profundidad" / "adobe" / "revision_ciega"
PLANTILLA, CLAVE = RC / "plantilla_revision_ciega.csv", RC / "clave_revision_ciega.csv"
OUT = SP / "salidas" / "comparacion_backends" / "ciega"; OUT.mkdir(parents=True, exist_ok=True)
RES = Path(r"C:\Users\ACER\Documents\logo\video-val\resultados\profundidad\siete_puntos\comparacion_backends\ciega"); RES.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(V2 / "profundidad")]
from comun_profundidad import motion_mask, read_frame
from backends_profundidad import make_backend

BACKENDS = [("yolo26", "yolo26s-depth"), ("yolo26", "yolo26x-depth"), ("yolo26_crop", "yolo26s-depth"), ("vda", "vda-small")]
FLOR, FONDO = (0.54, 0.38, 0.69, 0.58), (0.30, 0.03, 0.62, 0.16)
ETQ = {"delante": ("delante", 0), "posada delante": ("delante", 1), "detras": ("detras", 0), "detrás": ("detras", 0), "al lado": ("al lado", 0)}
COLOR = {"delante": "#1f77b4", "detras": "#d62728", "al lado": "#7f7f7f"}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n; d = 1 + z ** 2 / n; c = p + z ** 2 / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (c - h) / d, (c + h) / d


def dz_by_backend(O):
    path = glob.glob(r"C:\Users\ACER\Documents\logo\video-val\test_*i5bEfvb5mxg*")[0]
    t = pd.read_csv(glob.glob(str(V2 / "eval_gt2" / "union" / "prudente" / "trayectoria_*i5befvb5mxg*.csv"))[0]).set_index("frame")
    cap = cv2.VideoCapture(path); n, W, H = int(cap.get(7)), int(cap.get(3)), int(cap.get(4))
    bg = np.median(np.stack([f for f in (read_frame(cap, i) for i in np.linspace(int(0.03 * n), n - 1, 60).astype(int)) if f is not None]), axis=0).astype(np.uint8)
    rect = lambda r: (int(r[0] * W), int(r[1] * H), int(r[2] * W), int(r[3] * H)); ref, far = rect(FLOR), rect(FONDO)
    res = []
    for kind, model in BACKENDS:
        tag = f"{kind}_{model}"; crop = None; fa = far
        if kind == "yolo26_crop":   # el mismo recorte que el notebook
            ob = t[~t.es_puente.astype(bool)]
            x0, x1 = min((ob.cx_px - ob.w_px / 2).min(), FLOR[0] * W) - .05 * W, max((ob.cx_px + ob.w_px / 2).max(), FLOR[2] * W) + .05 * W
            y0, y1 = min((ob.cy_px - ob.h_px / 2).min(), FONDO[1] * H) - .05 * H, max((ob.cy_px + ob.h_px / 2).max(), FLOR[3] * H) + .05 * H
            crop = (max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1)))
            fa = (max(far[0], crop[0]), max(far[1], crop[1]), min(far[2], crop[2]), min(far[3], crop[3]))
        be = make_backend(kind, model, "i5bE", (H, W), crop=crop, cache_dir=SP / "cache_vda")
        for r in O.itertuples():
            fr = read_frame(cap, r.frame); b = t.loc[r.frame]
            box = (max(0, int(b.cx_px - b.w_px / 2)), max(0, int(b.cy_px - b.h_px / 2)), min(W, int(b.cx_px + b.w_px / 2) + 1), min(H, int(b.cy_px + b.h_px / 2) + 1))
            m = motion_mask(fr, bg, *box); d = be.map(r.frame, fr)
            z = float(np.nanmedian(d[box[1]:box[3], box[0]:box[2]][m])) if m.sum() >= 30 else np.nan

            def out(rc):
                x0, y0, x1, y1 = rc; k = np.ones((y1 - y0, x1 - x0), bool); a0, b0, a1, b1 = max(box[0], x0), max(box[1], y0), min(box[2], x1), min(box[3], y1)
                if a1 > a0 and b1 > b0:
                    k[b0 - y0:b1 - y0, a0 - x0:a1 - x0] = False
                return float(np.nanmedian(d[y0:y1, x0:x1][k])) if k.mean() > .2 else float(np.nanmedian(d[y0:y1, x0:x1]))
            fl, fo = out(ref), out(fa)
            res.append(dict(backend=tag, id=r.id, frame=r.frame, respuesta=r.posicion, etiqueta=r.etq, posado=r.posado, z_ave=z, z_flor=fl, z_fondo=fo, dz=(z - fl) / (fo - fl), dz_sin_normalizar=z - fl))
    cap.release()
    return pd.DataFrame(res)


def metrics(g):
    d = g[g.etiqueta.isin(["delante", "detras"])].copy()
    d["ok"] = ((d.etiqueta == "delante") & (d.dz < 0)) | ((d.etiqueta == "detras") & (d.dz > 0))
    k, n = int(d.ok.sum()), len(d); lo, hi = wilson(k, n)
    dl, dt = d[d.etiqueta == "delante"].dz, d[d.etiqueta == "detras"].dz
    auc = float(np.mean([(b > a) + 0.5 * (b == a) for a in dl for b in dt]))
    no, si = d[d.posado == 0], d[d.posado == 1]
    return dict(aciertos=k, n=n, acierto=k / n, ic95_inf=lo, ic95_sup=hi, p_binomial=binomtest(k, n, 0.5).pvalue, auc=auc,
                verd_delante_pred_delante=int((dl < 0).sum()), verd_delante_pred_detras=int((dl >= 0).sum()), verd_detras_pred_delante=int((dt <= 0).sum()), verd_detras_pred_detras=int((dt > 0).sum()),
                aciertos_no_posados=int(no.ok.sum()), n_no_posados=len(no), aciertos_posados=int(si.ok.sum()), n_posados=len(si),
                dz_med_delante_no_posado=float(no[no.etiqueta == "delante"].dz.median()), dz_med_posado=float(si.dz.median()), dz_med_detras=float(dt.median()), dz_med_al_lado=float(g[g.etiqueta == "al lado"].dz.median()))


def main():
    T = pd.read_csv(PLANTILLA, dtype={"posicion": str})
    if T.posicion.notna().sum() == 0:
        print("FALTA: la plantilla sigue vacía:", PLANTILLA); return
    T[["etq", "posado"]] = [ETQ[x.strip().lower()] for x in T.posicion]
    c = T.etq.value_counts().to_dict(); print("recuento:", c, "| posados:", int(T.posado.sum()))
    assert c == {"delante": 12, "detras": 5, "al lado": 3} and T.posado.sum() == 6, "el recuento no cuadra"
    R = dz_by_backend(T); R.to_csv(OUT / "evaluacion_ciega_por_frame.csv", index=False)
    S = pd.DataFrame([dict(backend=b, **metrics(g)) for b, g in R.groupby("backend", sort=False)]); S.to_csv(OUT / "evaluacion_ciega.csv", index=False)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
    print(S.round(3).T.to_string())
    K = pd.read_csv(CLAVE); s = R[R.backend == "yolo26_yolo26s-depth"].merge(K[["id", "dz"]].rename(columns={"dz": "dz_clave"}), on="id")
    s["dz_sin_normalizar_vs_clave"] = s.dz_sin_normalizar - s.dz_clave; s.to_csv(OUT / "control_clave.csv", index=False)
    print("control clave (yolo26s): dif. absoluta mediana", round(float(s.dz_sin_normalizar_vs_clave.abs().median()), 4), "| máx.", round(float(s.dz_sin_normalizar_vs_clave.abs().max()), 4), "| mismo signo:", int((np.sign(s.dz_sin_normalizar) == np.sign(s.dz_clave)).sum()), "de", len(s))
    P = R.pivot(index=["id", "frame", "respuesta"], columns="backend", values="dz").round(3); P.to_csv(OUT / "dz_por_frame.csv"); print(P.reset_index().to_string(index=False))
    fig, axes = plt.subplots(1, len(BACKENDS), figsize=(5.4 * len(BACKENDS), 6))
    for a, (b, g) in zip(axes, R.groupby("backend", sort=False)):
        g = g.sort_values("dz").reset_index(drop=True); m = S[S.backend == b].iloc[0]
        a.axvline(0, color="k", lw=1)
        for i, r in g.iterrows():
            a.scatter(r.dz, i, c=COLOR[r.etiqueta], marker="s" if r.posado else "o", s=70, edgecolors="k", linewidths=0.4, zorder=3)
            a.annotate(f"#{r.id}", (r.dz, i), xytext=(6, -3), textcoords="offset points", fontsize=8)
        a.set_title(f"{b}\nsigno {m.aciertos}/{m.n} ({m.ic95_inf:.2f}-{m.ic95_sup:.2f}) | p {m.p_binomial:.3f} | AUC {m.auc:.2f}", fontsize=10); a.set_yticks([]); a.grid(axis="x", alpha=0.3); a.set_xlabel("dz (negativo = delante según el modelo)")
    for lab, col in COLOR.items():
        axes[0].scatter([], [], c=col, s=70, edgecolors="k", linewidths=0.4, label=lab)
    axes[0].scatter([], [], c="white", marker="s", s=70, edgecolors="k", label="posada"); axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("i5bE: dz de cada frame con la etiqueta a ciegas de Yilber. Bien separado = azules a la izquierda de 0 y rojos a la derecha", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "evaluacion_ciega.jpg", dpi=90); plt.close(fig)
    for f in OUT.iterdir():
        (RES / f.name).write_bytes(f.read_bytes())
    (RES / "evaluar_ciega.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
