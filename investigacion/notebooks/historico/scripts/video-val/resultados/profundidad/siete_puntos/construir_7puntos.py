"""Genera colibri_3d_7puntos.ipynb con el modelo de profundidad como parámetro de la primera celda, y una copia por backend
en salidas\\<backend>_<modelo>\\. La v1 (8 videos, solo YOLO) queda en colibri_3d_7puntos_v1_ejecutado.ipynb y salidas\\."""
from pathlib import Path

import nbformat as nbf

HERE = Path(r"D:\video_eval\reentreno\v2\profundidad\siete_puntos")
BACKENDS = [("yolo26", "yolo26s-depth"), ("yolo26", "yolo26x-depth"), ("yolo26_crop", "yolo26s-depth"), ("vda", "vda-small")]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md(r"""# Trayectoria 3D de 7 puntos con modelo de profundidad seleccionable (SP-02.7)

Para cada punto se lee la profundidad del ave en el mapa de su frame y se normaliza dentro de ese mapa:

`z_norm = (z_ave − z_flor) / (z_fondo − z_flor)`: 0 = flor, 1 = fondo, negativo = delante, positivo = detrás.

**Selector:** `DEPTH_BACKEND` y `DEPTH_MODEL` en la primera celda. Todos entran por `backends_profundidad.py`, que devuelve mapas alineados con el frame original.

| Backend | Modelo | Qué hace | Dominio |
|---|---|---|---|
| `yolo26` | `yolo26{n,s,m,l,x}-depth` | frame completo, sin memoria | profundidad |
| `yolo26_crop` | `yolo26s-depth` | recorte fijo con flor, recorrido y franja de fondo | profundidad |
| `vda` | `vda-small` (Apache-2.0) | video completo, con consistencia temporal (`vda_inferir.py` en `.venv_vda`) | disparidad |

**Comparabilidad:** el signo de z_norm significa lo mismo en todos, pero la magnitud no: profundidad y disparidad no son lineales entre sí. Las variantes de la barra de error también difieren: `yolo26` usa original, volteo y `yolo26m-depth`; `vda` solo original y volteo, porque Base y Large son CC-BY-NC.

No hay verdad de terreno 3D. El juez entre backends es la revisión a ciegas (`evaluar_ciega.py`)."""))

cells.append(code(r"""DEPTH_BACKEND = "@@BACKEND@@"     # "yolo26" | "yolo26_crop" | "vda"
DEPTH_MODEL = "@@MODEL@@"         # "yolo26s-depth" | "yolo26x-depth" | "vda-small"
VIDEO_LIST = ["i5bE"]             # lista de videos; vda necesita haber pasado antes por vda_inferir.py
POINT_SETS = ["b_salientes", "delante"]   # también "a_uniforme_tiempo"; "delante" solo existe en i5bE
"""))

cells.append(code(r"""import json, os, sys
from pathlib import Path

import cv2, numpy as np, pandas as pd, ultralytics
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from IPython.display import display

V2 = Path(r"D:\video_eval\reentreno\v2")
SP = V2 / "profundidad" / "siete_puntos"; os.chdir(SP)   # aquí están los pesos de YOLO
VIDEOS = Path(r"C:\Users\ACER\Documents\logo\video-val")
ADOBE = V2 / "profundidad" / "adobe"
TAG = f"{DEPTH_BACKEND}_{DEPTH_MODEL}"
OUT = SP / "salidas" / TAG; OUT.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(V2 / "trayectorias_7"), str(V2 / "profundidad")]
from trayectorias_7pts import catmull_rom
from comun_profundidad import motion_mask, read_frame
from backends_profundidad import make_backend

IMGSZ, N_BG, MIN_MASK_PX, GRID, SMOOTH_S = 640, 60, 30, (16, 9), 0.2
PATHS = json.loads((ADOBE / "paths.json").read_text(encoding="utf-8"))
# video: clave o id, referencia (nombre, rectángulo x0, y0, x1, y1) y región de fondo lejana y estable, sin ave
SCENES = {
    "i5bE": ("i5befvb5mxg", ("monarda", (0.54, 0.38, 0.69, 0.58)), ("bosque desenfocado arriba", (0.30, 0.03, 0.62, 0.16))),
    "DoJ1": ("doj1xa6ksqm", ("flor de aloe", (0.42, 0.10, 0.54, 0.72)), ("fondo magenta", (0.72, 0.10, 0.95, 0.50))),
    "Gu30": ("gu30nofexla", ("bráctea", (0.36, 0.20, 0.44, 0.70)), ("ladera verde", (0.68, 0.40, 0.95, 0.65))),
    "J4PK": ("j4pkgwgjaus", ("tronco", (0.58, 0.20, 0.90, 0.80)), ("detrás de la reja", (0.09, 0.05, 0.20, 0.42))),
    "A129": ("vv01_adobe1294608945", ("bráctea roja", (0.60, 0.15, 0.71, 0.68)), ("bosque desenfocado", (0.05, 0.10, 0.28, 0.60))),
    "A512": ("vv02_adobe512464022", ("flor tubular", (0.71, 0.40, 0.82, 0.50)), ("cielo", (0.04, 0.10, 0.22, 0.50))),
    "vv03": ("vv03_florisuga_file", ("flores moradas", (0.42, 0.25, 0.55, 0.58)), ("fondo negro", (0.75, 0.10, 0.96, 0.50))),
    "A160": ("adobestock_1600207728", ("hibisco", (0.60, 0.14, 0.78, 0.48)), ("bosque oscuro", (0.04, 0.05, 0.28, 0.40))),
}
print("backend", TAG, "| ultralytics", ultralytics.__version__)

def video_info(name):
    key = SCENES[name][0]
    if key in PATHS:
        return Path(PATHS[key]), ADOBE / "trayectorias" / f"trayectoria_{key}.csv", ADOBE / "puntos7"
    return (next(p for p in VIDEOS.glob("test_*") if key in p.name.lower()),
            next((V2 / "eval_gt2" / "union" / "prudente").glob(f"trayectoria_*{key}*.csv")), VIDEOS / "resultados" / "trayectorias_7pts")

def load_points(name, s):
    if s == "delante":
        return pd.read_csv(SP / "salidas" / "puntos_i5bE_salientes_delante.csv")
    return pd.read_csv(video_info(name)[2] / f"{name}_{s}.csv")

def rect_px(r, W, H):
    return int(r[0] * W), int(r[1] * H), int(r[2] * W), int(r[3] * H)

def background(cap, n):
    fs = [read_frame(cap, i) for i in np.linspace(int(0.03 * n), n - 1, N_BG).astype(int)]
    fs = [f for f in fs if f is not None and cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() >= 5]
    return np.median(np.stack(fs), axis=0).astype(np.uint8)

def crop_for(t, far, W, H, margin=0.05):
    # recorte fijo a partir de la trayectoria, no de z: recorrido del ave + flor + una franja de la región de fondo
    o = t[~t.es_puente.astype(bool)]; ref = SCENES_NOW[1][1]
    x0, x1 = min((o.cx_px - o.w_px / 2).min(), ref[0] * W) - margin * W, max((o.cx_px + o.w_px / 2).max(), ref[2] * W) + margin * W
    y0, y1 = min((o.cy_px - o.h_px / 2).min(), far[1] * H) - margin * H, max((o.cy_px + o.h_px / 2).max(), ref[3] * H) + margin * H
    return max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
"""))

cells.append(md("## 1. Lectura por punto\n\nMismas reglas para todos los backends: mediana en la máscara de movimiento (recorte central si tiene menos de 30 px), flor y fondo en el propio mapa, y el mismo control de vecinos ±2. En `yolo26_crop`, la región de fondo se recorta a lo que cae dentro del recorte."))

cells.append(code(r"""def median_outside(d, rect, box):
    x0, y0, x1, y1 = rect; m = np.ones((y1 - y0, x1 - x0), bool)
    ax0, ay0, ax1, ay1 = max(box[0], x0), max(box[1], y0), min(box[2], x1), min(box[3], y1)
    if ax1 > ax0 and ay1 > ay0:
        m[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = False
    return float(np.nanmedian(d[y0:y1, x0:x1][m])) if m.mean() > 0.2 else float(np.nanmedian(d[y0:y1, x0:x1]))

def read_point(d, box, mask, ref, far):
    x0, y0, x1, y1 = box; b = d[y0:y1, x0:x1]; h, w = b.shape
    use = int(mask.sum()) >= MIN_MASK_PX
    z = float(np.nanmedian(b[mask])) if use else float(np.nanmedian(b[h // 4:h - h // 4 or None, w // 4:w - w // 4 or None]))
    fl, fo = median_outside(d, ref, box), median_outside(d, far, box)
    return dict(z_ave=z, z_flor=fl, z_fondo=fo, z_norm=(z - fl) / (fo - fl), den=fo - fl, metodo_z="mask_med" if use else "recorte")

def box_of(r, W, H):
    return (max(0, int(r.cx_px - r.w_px / 2)), max(0, int(r.cy_px - r.h_px / 2)), min(W, int(r.cx_px + r.w_px / 2) + 1), min(H, int(r.cy_px + r.h_px / 2) + 1))

R, INFO, CTX = {}, {}, {}
for name in VIDEO_LIST:
    SCENES_NOW = SCENES[name]
    path, tf, _ = video_info(name); t = pd.read_csv(tf).set_index("frame")
    cap = cv2.VideoCapture(str(path)); fps, n, W, H = cap.get(5), int(cap.get(7)), int(cap.get(3)), int(cap.get(4)); bg = background(cap, n)
    key, (rn, rr), (fn, ff) = SCENES[name]
    crop = crop_for(t.reset_index(), ff, W, H) if DEPTH_BACKEND == "yolo26_crop" else None
    be = make_backend(DEPTH_BACKEND, DEPTH_MODEL, name, (H, W), crop=crop, cache_dir=SP / "cache_vda")
    ref, far = rect_px(rr, W, H), rect_px(ff, W, H)
    if crop:   # el fondo de referencia debe caer dentro del recorte
        far = (max(far[0], crop[0]), max(far[1], crop[1]), min(far[2], crop[2]), min(far[3], crop[3]))
        print(name, "recorte (px):", crop, "| fondo dentro del recorte (px):", far, f"| el recorte ocupa {100 * (crop[2] - crop[0]) / W:.0f} % del ancho")
    CTX[name] = dict(cap=cap, be=be, bg=bg, ref=ref, far=far, t=t, W=W, H=H, fps=fps, n=n, crop=crop)

    def at(f, tta):
        fr = read_frame(cap, f); box = box_of(t.loc[f], W, H); mask = motion_mask(fr, bg, *box)
        o = read_point(be.map(f, fr), box, mask, ref, far); o["n_px"] = int(mask.sum()); o["box"] = box
        if tta:
            o["tta"] = [o["z_norm"], read_point(be.map(f, fr, flip=True), box, mask, ref, far)["z_norm"]]
            if "secondary" in be.variants:
                o["tta"].append(read_point(be.map(f, fr, secondary=True), box, mask, ref, far)["z_norm"])
        return o

    res = {}
    for s in POINT_SETS:
        P = load_points(name, s); rows = []
        for r in P.itertuples():
            o = at(int(r.frame), True); vec = [o["z_norm"]]
            for g in (r.frame - 2, r.frame - 1, r.frame + 1, r.frame + 2):
                if g in t.index and not bool(t.loc[g].es_puente):
                    vec.append(at(int(g), False)["z_norm"])
            rows.append(dict(punto=r.punto, frame=r.frame, t_s=r.t_s, segment_id=r.segment_id, x_norm=r.x_norm, y_norm=r.y_norm, z_ave=o["z_ave"], z_flor=o["z_flor"], z_fondo=o["z_fondo"],
                             z_norm=float(np.mean(o["tta"])), z_norm_std=float(np.std(o["tta"])), z_norm_orig=o["z_norm"], vecinos_std=float(np.std(vec)), n_vecinos=len(vec) - 1,
                             metodo_z=o["metodo_z"], n_px_mascara=o["n_px"], den=o["den"], box=o["box"]))
        res[s] = pd.DataFrame(rows); res[s].drop(columns=["box"]).to_csv(OUT / f"puntos_{name}_{s}.csv", index=False)
    R[name], INFO[name] = res, dict(W=W, H=H, ref=rr, ref_name=rn, path=path)
    print(name, "listo", flush=True)
"""))

cells.append(md("## 2. Control de ruido\n\nRegla fijada: la forma en z no es fiable si la dispersión entre vecinos es mayor que la diferencia típica entre puntos consecutivos."))

cells.append(code(r"""rows = []
for name, r in R.items():
    for s, D in r.items():
        dif = float(np.median(np.abs(np.diff(D.z_norm)))); vec = float(D.vecinos_std.median()); tta = float(D.z_norm_std.median())
        rows.append(dict(backend=TAG, video=name, puntos=s, dif_entre_puntos=dif, rango_z=float(np.ptp(D.z_norm)), ruido_vecinos=vec, ruido_tta=tta, forma_supera_ruido=bool(dif > vec), recortes=int((D.metodo_z != "mask_med").sum()), min_den=float(D.den.abs().min())))
CTRL = pd.DataFrame(rows); CTRL.to_csv(OUT / "control_ruido.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); display(CTRL.round(3))
for name, r in R.items():
    for s, D in r.items():
        print(name, s); display(D[["punto", "frame", "t_s", "x_norm", "y_norm", "z_norm", "z_norm_std", "vecinos_std", "n_px_mascara"]].round(3))
"""))

cells.append(md("## 3. Filtro de estabilidad\n\nSobre todos los frames observados: variación del fondo con cámara fija (celdas sin ave) y autocorrelación a 0,5 s de la lectura suavizada. El filtro fijado pide variación < 5 % y autocorrelación > 0,5. Se mide `z_ave` (la lectura directa, comparable con la prueba anterior) y `z_norm`. En `yolo26_crop`, las celdas son solo las que caen dentro del recorte."))

cells.append(code(r"""def cells_of(d):
    H, W = d.shape; gx, gy = GRID
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        return np.array([[np.nanmedian(d[H * j // gy:H * (j + 1) // gy:4, W * i // gx:W * (i + 1) // gx:4]) for i in range(gx)] for j in range(gy)])

rows, SERIES = [], {}
for name in VIDEO_LIST:
    c = CTX[name]; t, be, cap, W, H = c["t"], c["be"], c["cap"], c["W"], c["H"]
    obs = t[~t.es_puente.astype(bool)]; gx, gy = GRID; A = []; S = []
    for f, r in obs.iterrows():
        fr = read_frame(cap, f); box = box_of(r, W, H); mask = motion_mask(fr, c["bg"], *box); d = be.map(f, fr)
        o = read_point(d, box, mask, c["ref"], c["far"]); cm = cells_of(d)
        free = np.ones((gy, gx), bool); free[max(0, box[1] * gy // H):min(gy, box[3] * gy // H + 1), max(0, box[0] * gx // W):min(gx, box[2] * gx // W + 1)] = False
        A.append(np.where(free, cm, np.nan)); S.append(dict(frame=f, t_s=r.t_s, segment_id=r.segment_id, z_ave=o["z_ave"], z_norm=o["z_norm"], z_flor=o["z_flor"], z_fondo=o["z_fondo"]))
    A, S = np.stack(A), pd.DataFrame(S); S.to_csv(OUT / f"serie_{name}.csv", index=False); SERIES[name] = S
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        cv = np.nanstd(A, axis=0) / np.nanmean(A, axis=0)
    k = max(1, int(0.5 * c["fps"]))
    def lag(col):
        w = max(5, int(round(SMOOTH_S * c["fps"])) | 1); x = savgol_filter(S[col].to_numpy(), w, 2)
        return spearmanr(x[:-k], x[k:]).statistic
    jit = lambda col: float(np.median(np.abs(np.diff(S[col]))[np.diff(S.frame) == 1]) / max(np.percentile(S[col], 95) - np.percentile(S[col], 5), 1e-6))
    rows.append(dict(backend=TAG, video=name, frames=len(S), variacion_fondo_pct=100 * float(np.nanmedian(cv)), autocorr_z_05s=lag("z_ave"), autocorr_znorm_05s=lag("z_norm"), jitter_rel_z=jit("z_ave"), jitter_rel_znorm=jit("z_norm"),
                     pasa_filtro=bool(100 * np.nanmedian(cv) < 5 and lag("z_ave") > 0.5)))
FIL = pd.DataFrame(rows); FIL.to_csv(OUT / "filtro.csv", index=False); display(FIL.round(3))
"""))

cells.append(md("## 4. Trayectoria 3D\n\nEjes con la unidad del ancho del cuadro. z = z_norm sin reescalar. La flor es un cilindro con el radio de la mitad del ancho de su región (supuesto)."))

cells.append(code(r"""AR = 1080 / 1920
def curve(D):
    P = np.c_[D.x_norm, D.y_norm * AR, D.z_norm]; cuts = np.flatnonzero(np.diff(D.segment_id.to_numpy()) != 0) + 1
    return P, [catmull_rom(p) for p in np.split(P, cuts)]

def figure(name, s):
    D, inf = R[name][s], INFO[name]; rr = inf["ref"]; P, C = curve(D); cx, r = (rr[0] + rr[2]) / 2, (rr[2] - rr[0]) / 2; yt, yb = rr[1] * AR, rr[3] * AR
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2)); e = D.z_norm_std
    for a, i, j, tt in ((ax[0], 0, 1, "frontal x-y"), (ax[1], 0, 2, "superior x-z"), (ax[2], 2, 1, "lateral z-y")):
        for c in C: a.plot(c[:, i], c[:, j], color="black", lw=2.2)
        a.errorbar(P[:, i], P[:, j], xerr=e if i == 2 else None, yerr=e if j == 2 else None, fmt="o", color="crimson", mec="w", capsize=3, zorder=5)
        for k, q in enumerate(P, 1): a.annotate(str(k), (q[i], q[j]), xytext=(6, 5), textcoords="offset points", fontsize=11, weight="bold")
        a.set_title(tt, fontsize=11); a.grid(alpha=0.3)
    ax[0].add_patch(plt.Rectangle((rr[0], yt), 2 * r, yb - yt, fc="gold", alpha=0.35)); ax[0].set_ylim(0.42, 0.08); ax[0].set_xlim(0.3, 0.85); ax[0].set_aspect("equal")
    ax[1].add_patch(plt.Circle((cx, 0), r, fc="gold", alpha=0.35)); ax[1].axhline(0, color="goldenrod", ls=":"); ax[1].set_xlim(0.3, 0.85); ax[1].set_aspect("equal", adjustable="datalim")
    ax[2].add_patch(plt.Rectangle((-r, yt), 2 * r, yb - yt, fc="gold", alpha=0.35)); ax[2].axvline(0, color="goldenrod", ls=":"); ax[2].set_ylim(0.42, 0.08)
    fig.suptitle(f"{name} | {s} | {TAG} | z_norm: 0 = {inf['ref_name']}, 1 = fondo", fontsize=12); fig.tight_layout(); fig.savefig(OUT / f"trayectoria3d_{name}_{s}.jpg", dpi=100); plt.show(); plt.close(fig)
    g = go.Figure(); th = np.linspace(0, 2 * np.pi, 48); T, YY = np.meshgrid(th, [yt, yb])
    g.add_trace(go.Surface(x=cx + r * np.cos(T), y=r * np.sin(T), z=-YY, colorscale=[[0, "gold"], [1, "gold"]], showscale=False, opacity=0.4, name="flor", showlegend=True))
    for c in C: g.add_trace(go.Scatter3d(x=c[:, 0], y=c[:, 2], z=-c[:, 1], mode="lines", line=dict(color="black", width=7), name="curva"))
    g.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 2], z=-P[:, 1], mode="markers+text", marker=dict(size=5, color="crimson"), text=[str(i) for i in D.punto], name="7 puntos"))
    g.update_layout(title=f"{name} | {s} | {TAG}", scene=dict(xaxis_title="x", yaxis_title="z_norm", zaxis_title="−y", aspectmode="data")); g.write_html(OUT / f"trayectoria3d_{name}_{s}.html", include_plotlyjs=True)

def strip(name, s):
    D, inf = R[name][s], INFO[name]; cap = CTX[name]["cap"]; W, H = inf["W"], inf["H"]; x0, y0, x1, y1 = rect_px(inf["ref"], W, H)
    fig, axes = plt.subplots(1, len(D), figsize=(3.2 * len(D), 3.3))
    for ax, r in zip(axes, D.itertuples()):
        fr = read_frame(cap, r.frame); b = r.box; cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        X0, Y0, X1, Y1 = max(0, min(b[0], x0) - 40), max(0, min(b[1], y0) - 40), min(W, max(b[2], x1) + 40), min(H, max(b[3], y1) + 40)
        ax.imshow(cv2.cvtColor(fr[Y0:Y1, X0:X1], cv2.COLOR_BGR2RGB)); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(f"{r.punto} | f{r.frame} | {r.z_norm:+.2f} ± {r.z_norm_std:.2f}", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"tira_{name}_{s}.jpg", dpi=100); plt.show(); plt.close(fig)

for name in VIDEO_LIST:
    for s in POINT_SETS:
        figure(name, s); strip(name, s)
    CTX[name]["cap"].release()
"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
base = nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(c.source.replace("@@BACKEND@@", "yolo26").replace("@@MODEL@@", "yolo26s-depth")) if c.cell_type == "code" else c for c in cells], metadata=nb.metadata)
nbf.write(base, HERE / "colibri_3d_7puntos.ipynb")
for be, mo in BACKENDS:
    d = HERE / "salidas" / f"{be}_{mo}"; d.mkdir(parents=True, exist_ok=True)
    c2 = [nbf.v4.new_code_cell(c.source.replace("@@BACKEND@@", be).replace("@@MODEL@@", mo)) if c.cell_type == "code" else c for c in cells]
    nbf.write(nbf.v4.new_notebook(cells=c2, metadata=nb.metadata), d / f"colibri_3d_7puntos_{be}_{mo}.ipynb")
print("notebooks escritos")
