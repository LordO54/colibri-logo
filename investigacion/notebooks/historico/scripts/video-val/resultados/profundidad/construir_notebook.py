"""Genera los notebooks de SP-02.7 desde una sola fuente: colibri_profundidad_3d.ipynb (config test) y
adobe\\colibri_profundidad_3d_adobe.ipynb (config adobe). La decisión se lee de decision_sp027*.md si existe."""
import textwrap
from pathlib import Path

import nbformat as nbf

HERE = Path(r"D:\video_eval\reentreno\v2\profundidad")
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md(r"""# SP-02.7: profundidad monocular y trayectorias 3D del colibrí

Spike opcional del backlog: **¿la profundidad monocular añade información espacial útil para descubrir geometrías relevantes para el logotipo?** No tiene que dar positivo y no bloquea el flujo 2D.

| Sección | Criterio |
|---|---|
| 2. Muestra procesada | CA-02.7.1 |
| 3. Alineación mapa-frame-coordenadas | CA-02.7.2 |
| 4. Métodos para la profundidad del ave | CA-02.7.3 |
| 5. Estabilidad temporal, fondo y escala | CA-02.7.4 |
| 6-7. Trayectoria 3D frente a 2D | CA-02.7.5 |
| 8. Oclusión en i5bE | apoyo a CA-02.7.4 y 02.7.5 |
| 9. Decisión | CA-02.7.6 |

**Entradas (no se modifican):** trayectorias `eval_gt2\union\prudente` (yolo26n_ft_v2, ByteTrack combo, unión prudente) y 7 puntos de `resultados\trayectorias_7pts`.

**Sin verdad de terreno 3D:** todas las validaciones son indirectas. La escala absoluta del modelo es solo orientativa: se entrenó con escenas de otro tipo y la cámara no está calibrada."""))

cells.append(code(r"""import json, re, sys, platform, warnings
from pathlib import Path

import cv2, numpy as np, pandas as pd, torch, ultralytics
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from scipy.stats import spearmanr, mannwhitneyu
from ultralytics import YOLO
from IPython.display import display, Markdown

# ---------- parámetros ----------
DEPTH_MODEL = "yolo26s-depth.pt"   # cualquier yolo26{n,s,m,l,x}-depth.pt
IMGSZ = 640
HFOV_DEG = 60.0                    # FOV horizontal supuesto: la cámara no está calibrada
SMOOTH_S = 0.2                     # misma ventana Savitzky-Golay que la 2D
RECOMPUTE = False                  # True: repite la inferencia aunque existan las estadísticas
N_BG_FRAMES = 60                   # frames para la mediana de fondo
N_BIRDLESS = 60                    # máximo de frames sin ave para medir el parpadeo
N_SAMPLE_MAPS = 12                 # mapas guardados por video (float16, npz)
GRID = (16, 9)                     # celdas para medir el fondo cuadro a cuadro
MIN_MASK_FRAC = 0.05               # fracción mínima de la caja para confiar en la máscara

V2 = Path(r"D:\video_eval\reentreno\v2")
VIDEOS = Path(r"C:\Users\ACER\Documents\logo\video-val")
GATE_CV, GATE_LAG = 5.0, 0.5       # filtro fijado de antemano: variación del fondo (%) y autocorrelación de z a 0,5 s

# Lista de videos y rutas: SP_CONFIG elige el conjunto. La región de referencia se dibuja a mano en coordenadas
# normalizadas (x0, y0, x1, y1) y se revisa en la figura de la sección 3.
SP_CONFIG = "@@CONFIG@@"
ADOBE = V2 / "profundidad" / "adobe"
CONFIGS = {
    "test": dict(traj=V2 / "eval_gt2" / "union" / "prudente", track=V2 / "eval_gt2" / "tracking" / "b_nuevo_bt_combo",
                 pts=VIDEOS / "resultados" / "trayectorias_7pts", out=V2 / "profundidad" / "salidas", dark=12, paths=None,
                 names={"doj1xa6ksqm": "DoJ1", "gu30nofexla": "Gu30", "i5befvb5mxg": "i5bE", "j4pkgwgjaus": "J4PK"},
                 ref={"DoJ1": ("flor de aloe", (0.42, 0.10, 0.54, 0.72)), "Gu30": ("bráctea de banano", (0.36, 0.20, 0.44, 0.70)),
                      "i5bE": ("monarda", (0.54, 0.38, 0.69, 0.58)), "J4PK": ("tronco", (0.58, 0.20, 0.90, 0.80))}),
    "adobe": dict(traj=ADOBE / "trayectorias", track=ADOBE / "tracks", pts=ADOBE / "puntos7", out=ADOBE / "salidas", dark=5,  # vv03 tiene fondo negro
                  paths=(ADOBE / "paths.json"),
                  names={"vv01_adobe1294608945": "A129", "vv02_adobe512464022": "A512", "vv03_florisuga_file": "vv03", "adobestock_1600207728": "A160"},
                  ref={"A129": ("bráctea roja", (0.60, 0.15, 0.71, 0.68)), "A512": ("flor tubular", (0.71, 0.40, 0.82, 0.50)),
                       "vv03": ("flores moradas", (0.42, 0.25, 0.55, 0.58)), "A160": ("hibisco", (0.60, 0.14, 0.78, 0.48))},
                  z_fijo=("c_mask_med", "z_al")),
}
CFG = CONFIGS[SP_CONFIG]
TRAJ_DIR, TRACK_DIR, PTS_DIR, OUT, NAMES, REF, DARK = CFG["traj"], CFG["track"], CFG["pts"], CFG["out"], CFG["names"], CFG["ref"], CFG["dark"]
PATHS = json.loads(CFG["paths"].read_text(encoding="utf-8")) if CFG["paths"] else None
OUT.mkdir(parents=True, exist_ok=True)

print("config", SP_CONFIG, "| python", platform.python_version(), "| torch", torch.__version__, "| ultralytics", ultralytics.__version__,
      "| cuda", torch.cuda.is_available(), "| modelo", DEPTH_MODEL)
assert tuple(map(int, ultralytics.__version__.split(".")[:3])) >= (8, 4, 104), "YOLO26-Depth requiere ultralytics >= 8.4.104"
"""))

cells.append(md("## 1. Entradas y fondo de cada video\n\nCon cámara fija, la mediana temporal de frames repartidos por todo el video elimina el ave aunque no haya frames sin ella. Se descartan los fundidos a negro (gris medio < 12) y el primer 3 % del video, donde DoJ1 y Gu30 tienen fundido de entrada."))

cells.append(code(r"""model = YOLO(DEPTH_MODEL)
print("tarea:", model.task)

def depth_of(frame):
    return model.predict(frame, imgsz=IMGSZ, verbose=False)[0].depth.data.cpu().numpy().astype(np.float32)

def read_frame(cap, f):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, fr = cap.read()
    return fr if ok else None

V = {}
for f in sorted(TRAJ_DIR.glob("trayectoria_*.csv")):
    key = next(k for k in NAMES if k in f.stem); name = NAMES[key]
    path = Path(PATHS[key]) if PATHS else next(p for p in VIDEOS.glob("test_*") if key in p.name.lower())
    vid = f.stem.replace("trayectoria_", "")
    cap = cv2.VideoCapture(str(path))
    fps, n, W, H = cap.get(5), int(cap.get(7)), int(cap.get(3)), int(cap.get(4))
    frames = []
    for i in np.linspace(int(0.03 * n), n - 1, N_BG_FRAMES).astype(int):
        fr = read_frame(cap, i)
        if fr is not None and cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).mean() >= DARK:
            frames.append(fr)
    bg = np.median(np.stack(frames), axis=0).astype(np.uint8); del frames
    cap.release()
    x0, y0, x1, y1 = REF[name][1]
    V[name] = dict(vid=vid, path=path, fps=fps, n=n, W=W, H=H, bg=bg, bg_depth=depth_of(bg),
                   traj=pd.read_csv(f), track=pd.read_csv(TRACK_DIR / f"{vid}__detallada.csv"),
                   ref=(int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
    print(name, f"{W}x{H} {fps:.2f} fps, {n} frames; filas de trayectoria {len(V[name]['traj'])}; "
          f"observadas {(~V[name]['traj'].es_puente).sum()}")
"""))

cells.append(md("## 2. Profundidad cuadro a cuadro (CA-02.7.1)\n\nSe procesan **todos los frames observados** de la trayectoria (los tramos `es_puente` no tienen ave real). Por frame se guardan estadísticas y solo una muestra de mapas, para no llenar el disco.\n\nProfundidad del ave:\n- **a_centro**: valor en el centro de la caja;\n- **b_recorte**: mediana del recorte central (50 % de ancho y alto);\n- **c_mask_med / c_mask_p25**: mediana y percentil 25 en la máscara de movimiento (píxeles de la caja que difieren del fondo). Si la máscara cubre menos del 5 % de la caja, se usa el recorte central y se marca.\n\nA cada método se le añade la versión `_al`, alineada por frame: se corrige la escala global con la mediana de `log(mapa/mapa_fondo)` en celdas sin ave. Eso quita el parpadeo global de la escala."))

cells.append(code(r"""def motion_mask(frame, bg, x0, y0, x1, y1):
    # quita el cambio global de brillo; el umbral se adapta al ruido del anillo alrededor de la caja
    pad = max(10, int(0.5 * max(x1 - x0, y1 - y0)))
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(frame.shape[1], x1 + pad), min(frame.shape[0], y1 + pad)
    diff = cv2.absdiff(frame[Y0:Y1, X0:X1], bg[Y0:Y1, X0:X1]).astype(np.float32)
    diff = np.abs(diff - np.median(diff, axis=(0, 1))).max(axis=2)
    ring = np.ones(diff.shape, bool); ring[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
    thr = max(15.0, float(np.percentile(diff[ring], 95)) + 5) if ring.any() else 15.0
    m = (diff > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0].astype(bool)

def cell_medians(d):
    H, W = d.shape; gx, gy = GRID
    return np.array([[np.median(d[H * j // gy:H * (j + 1) // gy:4, W * i // gx:W * (i + 1) // gx:4]) for i in range(gx)] for j in range(gy)])

def process(name):
    v = V[name]
    out_csv, out_cells = OUT / f"stats_{name}.csv", OUT / f"celdas_{name}.npz"
    if out_csv.exists() and out_cells.exists() and not RECOMPUTE:
        z = np.load(out_cells)
        return pd.read_csv(out_csv), z["cells"], z["free"], z["birdless"]
    t = v["traj"]; obs = t[~t.es_puente.astype(bool)].set_index("frame")
    W, H, bg, bgd = v["W"], v["H"], v["bg"], v["bg_depth"]
    gx, gy = GRID; rx0, ry0, rx1, ry1 = v["ref"]
    cells_bg = cell_medians(bgd)
    sample = set(obs.index[np.linspace(0, len(obs) - 1, N_SAMPLE_MAPS).astype(int)])
    rows, C, F, maps = [], [], [], {}
    cap = cv2.VideoCapture(str(v["path"])); idx = 0; last = obs.index.max()
    while idx <= last:
        if not cap.grab(): break
        if idx not in obs.index: idx += 1; continue
        ok, fr = cap.retrieve(); r = obs.loc[idx]
        d = depth_of(fr); assert d.shape == (H, W), "mapa no alineado con el frame"
        x0, y0 = max(0, int(r.cx_px - r.w_px / 2)), max(0, int(r.cy_px - r.h_px / 2))
        x1, y1 = min(W, int(r.cx_px + r.w_px / 2) + 1), min(H, int(r.cy_px + r.h_px / 2) + 1)
        box = d[y0:y1, x0:x1]; bw, bh = x1 - x0, y1 - y0
        crop = box[bh // 4:bh - bh // 4 or None, bw // 4:bw - bw // 4 or None]
        m = motion_mask(fr, bg, x0, y0, x1, y1)
        ok_mask = m.mean() >= MIN_MASK_FRAC
        zs = box[m] if ok_mask else crop.ravel()
        bgbox = bgd[y0:y1, x0:x1]; bgs = bgbox[m] if ok_mask else bgbox[bh // 4:bh - bh // 4 or None, bw // 4:bw - bw // 4 or None].ravel()
        # referencia por frame, sin los píxeles de la caja
        rm = np.ones((ry1 - ry0, rx1 - rx0), bool)
        ax0, ay0, ax1, ay1 = max(x0, rx0), max(y0, ry0), min(x1, rx1), min(y1, ry1)
        ov_area = max(0, ax1 - ax0) * max(0, ay1 - ay0); vis_ov = np.nan
        if ov_area:
            rm[ay0 - ry0:ay1 - ry0, ax0 - rx0:ax1 - rx0] = False
            vis_ov = float(m[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0].mean())
        ref_t = float(np.median(d[ry0:ry1, rx0:rx1][rm])) if rm.mean() > 0.2 else np.nan
        # celdas de fondo sin solape con la caja
        cm = cell_medians(d)
        free = np.ones((gy, gx), bool)
        free[max(0, y0 * gy // H):min(gy, y1 * gy // H + 1), max(0, x0 * gx // W):min(gx, x1 * gx // W + 1)] = False
        alpha = float(np.median(np.log(cm[free] / cells_bg[free])))
        C.append(cm); F.append(free)
        rows.append(dict(frame=idx, t_s=idx / v["fps"], cx=r.cx_px, cy=r.cy_px, w=r.w_px, h=r.h_px, segment_id=int(r.segment_id),
                         a_centro=float(d[min(H - 1, int(r.cy_px)), min(W - 1, int(r.cx_px))]), b_recorte=float(np.median(crop)),
                         c_mask_med=float(np.median(zs)), c_mask_p25=float(np.percentile(zs, 25)),
                         mask_frac=float(m.mean()), mask_ok=bool(ok_mask), z_fondo_detras=float(np.median(bgs)),
                         alpha=alpha, ref_t=ref_t, ov_ref=ov_area / (bw * bh), vis_ov=vis_ov, med_mapa=float(np.median(d))))
        if idx in sample:
            maps[f"f{idx}"] = cv2.resize(d, (W // 4, H // 4), interpolation=cv2.INTER_AREA).astype(np.float16)
        idx += 1
    cap.release()
    S = pd.DataFrame(rows); S.to_csv(out_csv, index=False)
    np.savez_compressed(OUT / f"mapas_{name}.npz", bg=cv2.resize(bgd, (W // 4, H // 4)).astype(np.float16), **maps)
    # frames sin ave (fuera de cualquier track) para medir el parpadeo del fondo
    with_track = set(v["track"].frame) | set(t.frame)
    free_fr = [f for f in range(v["n"]) if f not in with_track]
    B = []
    cap = cv2.VideoCapture(str(v["path"]))
    for f in (np.array(free_fr)[np.linspace(0, len(free_fr) - 1, min(N_BIRDLESS, len(free_fr))).astype(int)] if free_fr else []):
        fr = read_frame(cap, f)
        if fr is not None and cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).mean() >= DARK:
            B.append(cell_medians(depth_of(fr)))
    cap.release()
    C, F, B = np.stack(C).astype(np.float32), np.stack(F), (np.stack(B).astype(np.float32) if B else np.zeros((0, gy, gx), np.float32))
    np.savez_compressed(out_cells, cells=C, free=F, birdless=B)
    return S, C, F, B

METHODS = ["a_centro", "b_recorte", "c_mask_med", "c_mask_p25"]
STATS, CELLS, FREE, BIRDLESS = {}, {}, {}, {}
for name in V:
    S, C, F, B = process(name)
    for m in METHODS:
        S[m + "_al"] = S[m] * np.exp(-S.alpha)     # alineado por frame al mapa de fondo
    v = V[name]; rx0, ry0, rx1, ry1 = v["ref"]
    v["z_ref_bg"] = float(np.median(v["bg_depth"][ry0:ry1, rx0:rx1]))
    S["ref"] = S.ref_t.fillna(v["z_ref_bg"] * np.exp(S.alpha))
    STATS[name], CELLS[name], FREE[name], BIRDLESS[name] = S, C, F, B

display(pd.DataFrame([dict(video=n, filas_traj=len(V[n]["traj"]), frames_procesados=len(STATS[n]), frames_sin_ave=len(BIRDLESS[n]),
                           mascara_ok_pct=round(100 * STATS[n].mask_ok.mean(), 1), t_ini=round(STATS[n].t_s.min(), 2), t_fin=round(STATS[n].t_s.max(), 2))
                      for n in V]).set_index("video"))
"""))

cells.append(md("## 3. Alineación del mapa con el frame y la región de referencia (CA-02.7.2)\n\nEl predictor reescala el mapa a la resolución del frame, y la celda anterior comprueba la forma en cada frame. Aquí se ve si la caja cae sobre la silueta del ave en el mapa. El rectángulo cian es la región de referencia."))

cells.append(code(r"""fig, axes = plt.subplots(len(V), 3, figsize=(18, 3.1 * len(V)))
for i, name in enumerate(V):
    v, S = V[name], STATS[name]
    r = S.iloc[(S.mask_frac - S.mask_frac.median()).abs().argmin()]
    cap = cv2.VideoCapture(str(v["path"])); fr = read_frame(cap, r.frame); cap.release()
    d = depth_of(fr); m = None
    x0, y0, x1, y1 = int(r.cx - r.w / 2), int(r.cy - r.h / 2), int(r.cx + r.w / 2), int(r.cy + r.h / 2)
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(v["W"], x1), min(v["H"], y1)
    m = motion_mask(fr, v["bg"], x0, y0, x1, y1)
    rect = lambda ax, c, b: ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=c, lw=1.5))
    axes[i, 0].imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); rect(axes[i, 0], "red", (x0, y0, x1, y1)); rect(axes[i, 0], "cyan", v["ref"])
    axes[i, 0].set_title(f"{name} f{int(r.frame)} | referencia: {REF[name][0]}", fontsize=10)
    axes[i, 1].imshow(d, cmap="turbo", vmin=np.percentile(d, 1), vmax=np.percentile(d, 99)); rect(axes[i, 1], "white", (x0, y0, x1, y1)); rect(axes[i, 1], "cyan", v["ref"])
    axes[i, 1].set_title(f"profundidad: {d.min():.2f}-{d.max():.2f} m (escala no calibrada)", fontsize=10)
    crop = cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB).copy(); crop[m] = (0.5 * crop[m] + [127, 0, 0]).astype(np.uint8)
    axes[i, 2].imshow(crop); axes[i, 2].set_title(f"máscara de movimiento: {100 * m.mean():.0f} % de la caja", fontsize=10)
    for a in axes[i]: a.axis("off")
fig.tight_layout(); fig.savefig(OUT / "alineacion.jpg", dpi=90); plt.show()
"""))

cells.append(md("## 4. Comparación de métodos para la profundidad del ave (CA-02.7.3)\n\n**Validación indirecta:** si el ave mide más o menos lo mismo, su tamaño aparente es proporcional a 1/distancia, así que `z` debería crecer con `1/sqrt(área)` (Spearman positivo). Es una pista ruidosa: el batido de alas y la pose cambian la caja, y los frames consecutivos no son independientes.\n\n**Fuga del fondo:** si `z` del ave sigue a la profundidad del fondo que tiene detrás (`rho_fondo` alto), el modelo está leyendo el fondo y no el ave. `contraste` es la diferencia relativa con ese fondo; `pct_confundida` es el % de frames donde la diferencia es menor del 5 %."))

cells.append(code(r"""def jitter(s, frames):
    consecutive = np.diff(frames) == 1
    d = np.abs(np.diff(s))[consecutive]
    rng = np.percentile(s, 95) - np.percentile(s, 5)
    return float(np.median(d)), float(rng), float(np.median(d) / max(rng, 1e-6))

rows = []
for name, S in STATS.items():
    size = 1 / np.sqrt(S.w * S.h)
    for m in METHODS:
        for var, col in (("z", m), ("z_al", m + "_al"), ("dz", None)):
            s = S[col] if col else S[m] - S.ref
            d, rng, ratio = jitter(s.to_numpy(), S.frame.to_numpy())
            rows.append(dict(video=name, metodo=m, variante=var, rho_tamano=spearmanr(s, size).statistic,
                             rho_fondo=spearmanr(s, S.z_fondo_detras).statistic, jitter_m=d, rango_m=rng, jitter_rel=ratio,
                             contraste=float(((S.z_fondo_detras - S[m]) / S.z_fondo_detras).median()),
                             pct_confundida=float(100 * (np.abs((S.z_fondo_detras - S[m]) / S.z_fondo_detras) < 0.05).mean())))
M = pd.DataFrame(rows); M.to_csv(OUT / "metodos_z.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.precision", 3)
display(M.pivot_table(index=["metodo", "variante"], columns="video", values="rho_tamano"))
display(M[M.variante == "z"].pivot_table(index="metodo", columns="video", values=["rho_fondo", "pct_confundida", "jitter_rel"]).round(3))

# Elección, regla fijada antes de ver resultados: mayor mediana de rho con el tamaño entre videos; desempata menos jitter relativo
A = M[M.variante != "dz"].groupby(["metodo", "variante"]).agg(rho=("rho_tamano", "median"), rho_min=("rho_tamano", "min"), jitter=("jitter_rel", "median")).reset_index()
A = A.sort_values(["rho", "jitter"], ascending=[False, True]); display(A)
BEST = A.iloc[0]; Z_COL = BEST.metodo + ("_al" if BEST.variante == "z_al" else ""); Z_METHOD = BEST.metodo
print("Método elegido por la regla:", Z_COL, "| rho mediana", round(BEST.rho, 3), "| jitter relativo", round(BEST.jitter, 3))
if CFG.get("z_fijo"):  # mismo método que en test: no se vuelve a elegir sobre estos videos
    BEST = A[(A.metodo == CFG["z_fijo"][0]) & (A.variante == CFG["z_fijo"][1])].iloc[0]; Z_METHOD = BEST.metodo
    Z_COL = BEST.metodo + ("_al" if BEST.variante == "z_al" else "")
    print("Se usa el método fijado en test:", Z_COL, "| rho mediana", round(BEST.rho, 3))
"""))

cells.append(md("## 5. Estabilidad temporal, fondo y escala (CA-02.7.4)\n\nCon cámara fija, la profundidad del **fondo** debería ser constante. Se mide en celdas sin solape con el ave, y en frames sin ave cuando los hay:\n- `cv_celda_pct`: variación de cada celda en el tiempo (mediana de todas las celdas);\n- `salto_celda_pct`: cambio cuadro a cuadro;\n- `alpha_std_pct`: parpadeo de la escala global del mapa.\n\nPara que el ave sea interpretable, el recorrido real de su `z` debe superar con claridad ese parpadeo."))

cells.append(code(r"""rows = []
for name, S in STATS.items():
    C, F, B = CELLS[name], FREE[name], BIRDLESS[name]
    X = np.where(F, C, np.nan)
    cv = np.nanstd(X, axis=0) / np.nanmean(X, axis=0)
    consec = np.diff(S.frame.to_numpy()) == 1
    jump = np.nanmedian(np.abs(np.diff(X, axis=0))[consec] / np.nanmean(X, axis=0))
    z = S[Z_METHOD].to_numpy(); r95 = np.percentile(z, 95) - np.percentile(z, 5)
    fondo_std = np.nanmedian(np.nanstd(X, axis=0))
    rows.append(dict(video=name, cv_celda_pct=100 * np.nanmedian(cv), salto_celda_pct=100 * jump, alpha_std_pct=100 * S.alpha.std(),
                     cv_sin_ave_pct=100 * np.median(B.std(axis=0) / B.mean(axis=0)) if len(B) > 1 else np.nan, frames_sin_ave=len(B),
                     rango_ave_m=r95, std_fondo_m=fondo_std, ratio_ave_fondo=r95 / max(fondo_std, 1e-6),
                     z_ref_m=V[name]["z_ref_bg"], escena_min_m=float(V[name]["bg_depth"].min()), escena_max_m=float(V[name]["bg_depth"].max())))
ST = pd.DataFrame(rows).set_index("video"); ST.to_csv(OUT / "estabilidad_fondo.csv"); display(ST.round(3))

fig, axes = plt.subplots(len(V), 1, figsize=(15, 2.6 * len(V)))
for ax, (name, S) in zip(axes, STATS.items()):
    ax.plot(S.t_s, S[Z_METHOD], lw=0.8, label=f"ave ({Z_METHOD})")
    ax.plot(S.t_s, S.z_fondo_detras, lw=0.8, color="0.6", label="fondo detrás del ave")
    ax.plot(S.t_s, S.ref, lw=1, color="teal", label=REF[name][0])
    ax.set_ylabel("z (m, sin calibrar)"); ax.set_title(name, fontsize=10); ax.legend(fontsize=8, ncol=3, loc="upper right")
axes[-1].set_xlabel("t (s)"); fig.tight_layout(); fig.savefig(OUT / "series_z.jpg", dpi=90); plt.show()
"""))

cells.append(md("## 6. Trayectoria 3D\n\n`z` se interpola en los tramos `es_puente` y se suaviza con el mismo Savitzky-Golay que la 2D, por segmento. Se retroproyecta con una cámara estenopeica de FOV horizontal supuesto:\n\n`X = (u − cx)·z/f`, `Y = (v − cy)·z/f`, `Z = z`.\n\n**La escala es relativa:** cambiar el FOV o el factor de `z` cambia las proporciones del recorrido en 3D. La vista frontal (u, v) coincide con la 2D por construcción, y la celda lo comprueba reproyectando."))

cells.append(code(r"""def sg(a, fps):
    w = max(5, int(round(SMOOTH_S * fps)) | 1)
    return savgol_filter(a, w, 2) if len(a) > w else a.copy()

def load_pts(name):
    out = {}
    for tag, m in (("a", "a_uniforme_tiempo"), ("b", "b_salientes")):
        out[tag] = pd.read_csv(PTS_DIR / f"{name}_{m}.csv")
    return out

T3D = {}
for name, S in STATS.items():
    v = V[name]; t = v["traj"].sort_values("frame"); fps = v["fps"]
    f = v["W"] / 2 / np.tan(np.radians(HFOV_DEG) / 2); cx, cy = v["W"] / 2, v["H"] / 2
    zmap, dmap = S.set_index("frame")[Z_COL], (S[Z_METHOD] - S.ref).set_axis(S.frame)
    parts = []
    for sid, g in t.groupby("segment_id"):
        g = g.copy()
        g["x_px"], g["y_px"] = sg(g.cx_px.to_numpy(), fps), sg(g.cy_px.to_numpy(), fps)
        z = g.frame.map(zmap).astype(float).interpolate(limit_direction="both"); dz = g.frame.map(dmap).astype(float).interpolate(limit_direction="both")
        g["z_crudo"], g["z_m"], g["dz_flor"] = z, sg(z.to_numpy(), fps), sg(dz.to_numpy(), fps)
        parts.append(g)
    g = pd.concat(parts).sort_values("frame")
    g["X"], g["Y"], g["Z"] = (g.x_px - cx) * g.z_m / f, (g.y_px - cy) * g.z_m / f, g.z_m
    u, w = g.X * f / g.Z + cx, g.Y * f / g.Z + cy
    assert np.hypot(u - g.x_px, w - g.y_px).max() < 1e-6      # la vista frontal coincide con la 2D
    pts = load_pts(name)
    fa, fb = set(pts["a"].frame), set(pts["b"].frame)
    g["es_7pt"] = [("a+b" if fr in fa and fr in fb else "a" if fr in fa else "b" if fr in fb else "") for fr in g.frame]
    g["metodo_z"] = Z_COL
    g = g[["frame", "t_s", "x_px", "y_px", "z_m", "dz_flor", "X", "Y", "Z", "segment_id", "metodo_z", "es_7pt", "es_puente", "z_crudo"]]
    g.to_csv(OUT / f"trayectoria3d_{name}.csv", index=False)
    T3D[name] = g
    # flor o referencia en 3D
    rx0, ry0, rx1, ry1 = v["ref"]; zr = v["z_ref_bg"]; ur, vr = (rx0 + rx1) / 2, (ry0 + ry1) / 2
    v["ref3d"] = ((ur - cx) * zr / f, (vr - cy) * zr / f, zr)
    print(name, "CSV 3D:", len(g), "filas | z", round(g.z_m.min(), 2), "-", round(g.z_m.max(), 2), "m")

SEG = plt.get_cmap("tab10")
def views(name):
    g, v = T3D[name], V[name]; pts = load_pts(name); rx, ry, rz = v["ref3d"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for (a, xc, yc, tt) in ((ax[0], "X", "Y", "frontal X-Y (igual que la 2D salvo perspectiva)"), (ax[1], "X", "Z", "superior X-Z"), (ax[2], "Z", "Y", "lateral Z-Y")):
        for sid, s in g.groupby("segment_id"):
            a.plot(s[xc], s[yc], color=SEG(int(sid) % 10), lw=1.3)
        ra = dict(X=rx, Y=ry, Z=rz)
        a.scatter(ra[xc], ra[yc], marker="*", s=220, c="gold", edgecolors="k", zorder=5, label=REF[name][0])
        for tag, c in (("a", "crimson"), ("b", "royalblue")):
            sel = g[g.frame.isin(pts[tag].frame)].sort_values("frame")
            a.scatter(sel[xc], sel[yc], s=38, c=c, edgecolors="w", zorder=6, label=f"7 puntos {tag}")
            for k, (_, r) in enumerate(sel.iterrows(), 1):
                a.annotate(str(k), (r[xc], r[yc]), xytext=(4, 4), textcoords="offset points", fontsize=8, color=c)
        a.set_xlabel(f"{xc} (m relativos)"); a.set_ylabel(f"{yc} (m relativos)"); a.set_title(tt, fontsize=10)
        if yc == "Y": a.invert_yaxis()
        a.grid(alpha=0.3)
    ax[1].invert_yaxis(); ax[0].legend(fontsize=8)
    fig.suptitle(f"{name} | z = {Z_COL} | FOV {HFOV_DEG:.0f}° supuesto | escala relativa", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / f"vistas3d_{name}.jpg", dpi=100); return fig

def html3d(name):
    g, v = T3D[name], V[name]; pts = load_pts(name); rx, ry, rz = v["ref3d"]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=g.X, y=g.Z, z=-g.Y, mode="lines", line=dict(color="lightgray", width=3), name="densa"))
    for sid, s in g.groupby("segment_id"):
        fig.add_trace(go.Scatter3d(x=s.X, y=s.Z, z=-s.Y, mode="lines", line=dict(width=5), name=f"segmento {sid}", text=[f"t={t:.2f}s f{int(f)}" for t, f in zip(s.t_s, s.frame)]))
    for tag, c in (("a", "crimson"), ("b", "royalblue")):
        sel = g[g.frame.isin(pts[tag].frame)].sort_values("frame")
        fig.add_trace(go.Scatter3d(x=sel.X, y=sel.Z, z=-sel.Y, mode="markers+text", marker=dict(size=5, color=c), text=[str(i) for i in range(1, len(sel) + 1)], name=f"7 puntos {tag}"))
    fig.add_trace(go.Scatter3d(x=[rx], y=[rz], z=[-ry], mode="markers", marker=dict(size=8, color="gold", symbol="diamond"), name=REF[name][0]))
    fig.update_layout(title=f"{name} | z = {Z_COL} | escala relativa, FOV {HFOV_DEG:.0f}° supuesto", scene=dict(xaxis_title="X", yaxis_title="Z (profundidad)", zaxis_title="−Y (arriba)", aspectmode="data"))
    fig.write_html(OUT / f"trayectoria3d_{name}.html", include_plotlyjs=True)

for name in V:
    views(name); plt.show(); html3d(name)
"""))

cells.append(md("## 7. Comparación con la trayectoria 2D (CA-02.7.5)\n\n- `pct_long_z`: qué parte de la longitud 3D viene del movimiento en `z`.\n- `snr_z`: recorrido de `z` suavizado frente al temblor que el suavizado quitó. Si es bajo, el 3D añade sobre todo ruido.\n- `rho_z_t_lag`: correlación de `z` suavizado consigo mismo a 0,5 s; alta indica variación lenta y creíble, baja indica ruido.\n- `ratio_z_xy`: recorrido en `z` frente al recorrido en el plano."))

cells.append(code(r"""rows = []
for name, g in T3D.items():
    fps = V[name]["fps"]; L = {}
    per = g.groupby("segment_id")
    dX, dY, dZ = [per[c].diff().fillna(0).to_numpy() for c in ("X", "Y", "Z")]
    lxy, lz, l3 = np.hypot(dX, dY).sum(), np.abs(dZ).sum(), np.sqrt(dX**2 + dY**2 + dZ**2).sum()
    res = (g.z_crudo - g.z_m); rz = np.percentile(g.z_m, 95) - np.percentile(g.z_m, 5)
    k = max(1, int(0.5 * fps)); s = g.z_m.to_numpy()
    rows.append(dict(video=name, filas=len(g), long_xy_m=lxy, long_z_m=lz, long_3d_m=l3, pct_long_z=100 * lz / max(l3, 1e-6), snr_z=rz / max(res.std(), 1e-6),
                     rho_z_t_lag=spearmanr(s[:-k], s[k:]).statistic, rho_dz_t_lag=spearmanr(g.dz_flor.to_numpy()[:-k], g.dz_flor.to_numpy()[k:]).statistic, rango_z_m=rz, rango_X_m=np.ptp(g.X), rango_Y_m=np.ptp(g.Y),
                     ratio_z_xy=rz / max(np.hypot(np.ptp(g.X), np.ptp(g.Y)), 1e-6), dz_mediana_m=g.dz_flor.median(), pct_delante=100 * (g.dz_flor < 0).mean()))
CMP = pd.DataFrame(rows).set_index("video"); CMP.to_csv(OUT / "comparacion_2d_3d.csv"); display(CMP.round(3))

# los 7 puntos: profundidad y posición relativa a la referencia
P = pd.concat([T3D[n][T3D[n].es_7pt != ""].assign(video=n) for n in T3D])[["video", "frame", "t_s", "es_7pt", "x_px", "y_px", "z_m", "dz_flor", "X", "Y", "Z", "segment_id", "es_puente"]]
P.to_csv(OUT / "puntos7_3d.csv", index=False); display(P.round(2))
"""))

cells.append(md("## 7b. Estabilización y filtro fijado de antemano\n\nLa mediana de fondo y la máscara suponen un fondo inmóvil. `deriva_px` es el desplazamiento de cada frame frente al fondo (correlación de fase) y `pct_pix_cambian` la parte de la imagen que difiere más de 15 niveles.\n\n**Filtro:** variación del fondo < 5 % y autocorrelación de z a 0,5 s > 0,5."))

cells.append(code(r"""rows = []
for name, v in V.items():
    g0 = np.float32(cv2.cvtColor(v["bg"], cv2.COLOR_BGR2GRAY)); cap = cv2.VideoCapture(str(v["path"])); sh, ch = [], []
    for f in np.linspace(int(0.03 * v["n"]), v["n"] - 1, 40).astype(int):
        fr = read_frame(cap, f)
        if fr is None or cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).mean() < DARK: continue
        (dx, dy), _ = cv2.phaseCorrelate(g0, np.float32(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))); sh.append(np.hypot(dx, dy))
        ch.append(100 * (cv2.absdiff(fr, v["bg"]).max(axis=2) > 15).mean())
    cap.release()
    rows.append(dict(video=name, deriva_px_mediana=float(np.median(sh)), deriva_px_max=float(np.max(sh)), pct_pix_cambian=float(np.median(ch))))
DR = pd.DataFrame(rows).set_index("video"); DR.to_csv(OUT / "deriva_estabilizacion.csv")

R = M[(M.metodo == BEST.metodo) & (M.variante == BEST.variante)].set_index("video").rho_tamano
FI = pd.DataFrame({"variacion_fondo_pct": ST.cv_celda_pct, "cv_sin_ave_pct": ST.cv_sin_ave_pct, "frames_sin_ave": ST.frames_sin_ave, "autocorr_05s": CMP.rho_z_t_lag, "autocorr_dz_05s": CMP.rho_dz_t_lag, "rho_tamano": R,
                   "pct_long_z": CMP.pct_long_z, "ratio_ave_fondo": ST.ratio_ave_fondo}).join(DR)
FI["pasa_filtro"] = (FI.variacion_fondo_pct < GATE_CV) & (FI.autocorr_05s > GATE_LAG)
FI.to_csv(OUT / "filtro.csv"); display(FI.round(2))
print("Pasan el filtro:", list(FI.index[FI.pasa_filtro]) or "ninguno")
"""))

cells.append(md("## 8. i5bE: ¿el ave pasa delante y detrás de la flor?\n\nUna primera prueba automática (píxeles de movimiento en el solape caja-flor) **se descartó**: marcaba como \"tapada\" frames en que el ave solo estaba al lado de la flor (f165-168). Se sustituye por una revisión a ojo de 24 frames repartidos (`oclusion_revision_manual.jpg`), de la que salen tramos etiquetados a mano. Son pocos y los etiquetó una sola persona (Claude): conviene que Yilber los revise.\n\n| Tramo | Qué se ve | Se espera |\n|---|---|---|\n| 9,5-12,2 s | posado junto a la flor, pico en ella | dz ≈ 0 |\n| 12,7-13,7 s | delante del tallo y la flor, su cuerpo los tapa | dz < 0 |\n| 14,9-16,4 s | a la derecha, cabeza tapada por pétalos | dz > 0 |\n| f80 (2,7 s) | casi oculto tras la flor | dz > 0 |"))

SEC8 = (r"""S = STATS["i5bE"].copy(); S["dz"] = S[Z_METHOD] - S.ref
TRAMOS = {"posado, pico en la flor": (9.5, 12.2, "≈ 0"), "delante de la flor": (12.7, 13.7, "< 0"), "detrás de la flor": (14.9, 16.4, "> 0")}
rows = []
for k, (a, b, e) in TRAMOS.items():
    g = S[(S.t_s >= a) & (S.t_s <= b)]
    rows.append(dict(tramo=k, esperado=e, frames=len(g), dz_mediana=g.dz.median(), dz_p25=g.dz.quantile(.25), dz_p75=g.dz.quantile(.75), pct_dz_positivo=100 * (g.dz > 0).mean()))
OC = pd.DataFrame(rows).set_index("tramo"); OC.to_csv(OUT / "oclusion_i5bE.csv"); display(OC.round(3))
d, de = S[(S.t_s >= 14.9) & (S.t_s <= 16.4)].dz, S[(S.t_s >= 12.7) & (S.t_s <= 13.7)].dz
f80 = float(S.iloc[(S.frame - 80).abs().argmin()].dz)
OCC = dict(dz_detras=float(d.median()), dz_delante=float(de.median()), dz_f80_oculto=f80, salto_fondo_i5bE_m=float(ST.loc["i5bE", "std_fondo_m"]),
           p_unilateral=float(mannwhitneyu(d, de, alternative="greater").pvalue))
print(OCC, "(frames consecutivos y etiquetas a ojo: el p-valor es optimista)")

ov = S[S.ov_ref > 0.05].reset_index(drop=True); sel = ov.iloc[np.linspace(0, len(ov) - 1, 24).astype(int)]
cap = cv2.VideoCapture(str(V["i5bE"]["path"])); tiles = []
for r in sel.itertuples():
    f = read_frame(cap, r.frame); x0, y0, x1, y1 = int(r.cx - r.w / 2), int(r.cy - r.h / 2), int(r.cx + r.w / 2), int(r.cy + r.h / 2)
    cv2.rectangle(f, (x0, y0), (x1, y1), (0, 0, 255), 2); c = cv2.resize(f[230:760, 780:1560], (390, 265))
    cv2.putText(c, f"f{r.frame} {r.t_s:.1f}s dz {r.dz:+.2f}", (4, 18), 0, 0.55, (0, 255, 255), 2); tiles.append(c)
cap.release(); mos = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, 24, 6)]); cv2.imwrite(str(OUT / "oclusion_revision_manual.jpg"), mos)
fig, ax = plt.subplots(figsize=(18, 8)); ax.imshow(cv2.cvtColor(mos, cv2.COLOR_BGR2RGB)); ax.axis("off"); plt.show()

fig, ax = plt.subplots(figsize=(15, 3)); ax.plot(S.t_s, S.dz, lw=0.8); ax.axhline(0, color="k", lw=0.8)
for k, (a, b, e) in TRAMOS.items():
    ax.axvspan(a, b, alpha=0.15, label=f"{k} (esperado {e})")
ax.set_xlabel("t (s)"); ax.set_ylabel("dz (ave − flor, m)"); ax.legend(fontsize=8); ax.set_title("i5bE: dz > 0 significa detrás de la flor")
fig.tight_layout(); fig.savefig(OUT / "dz_i5bE.jpg", dpi=90); plt.show()
""")
cells.append(code("# Esta sección solo aplica si i5bE está en la lista de videos.\nOCC = {}\nif \"i5bE\" in V:\n" + textwrap.indent(SEC8, "    ")))

cells.append(md("## 9. Decisión (CA-02.7.6)"))
DEC_IDX = len(cells); cells.append(md(""))
cells.append(code(r"""resumen = dict(z_elegido=Z_COL, rho_tamano=M[(M.metodo == BEST.metodo) & (M.variante == BEST.variante)].set_index("video").rho_tamano.round(3).to_dict(),
               estabilidad=ST[["cv_celda_pct", "alpha_std_pct", "ratio_ave_fondo"]].round(2).to_dict("index"), filtro=FI[["variacion_fondo_pct", "autocorr_05s", "rho_tamano", "pct_long_z", "deriva_px_max", "pasa_filtro"]].round(2).to_dict("index"),
               comparacion=CMP[["pct_long_z", "snr_z", "ratio_z_xy", "pct_delante"]].round(2).to_dict("index"), oclusion_i5bE=OCC)
(OUT / "resumen_sp027.json").write_text(json.dumps(resumen, indent=2, default=float), encoding="utf-8")
print(json.dumps(resumen, indent=2, default=float))
d = OUT.parent / ("decision_sp027_adobe.md" if SP_CONFIG == "adobe" else "decision_sp027.md")
if d.exists():
    (OUT / ("informe_sp027_adobe.md" if SP_CONFIG == "adobe" else "informe_sp027.md")).write_text("# SP-02.7: profundidad monocular\n\n" + d.read_text(encoding="utf-8") + "\n\n## Cifras\n\n```json\n" + json.dumps(resumen, indent=2, default=float) + "\n```\n", encoding="utf-8")
"""))

for cfg, path, dec in (("test", HERE / "colibri_profundidad_3d.ipynb", HERE / "decision_sp027.md"),
                       ("adobe", HERE / "adobe" / "colibri_profundidad_3d_adobe.ipynb", HERE / "adobe" / "decision_sp027_adobe.md")):
    cs = [nbf.v4.new_code_cell(c.source.replace("@@CONFIG@@", cfg)) if c.cell_type == "code" else nbf.v4.new_markdown_cell(c.source) for c in cells]
    cs[DEC_IDX] = md(dec.read_text(encoding="utf-8") if dec.exists() else "*Pendiente de la primera ejecución.*")
    nbf.write(nbf.v4.new_notebook(cells=cs, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}), path)
print("notebooks escritos")
