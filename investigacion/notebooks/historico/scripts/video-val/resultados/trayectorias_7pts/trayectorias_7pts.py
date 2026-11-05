"""Reduce a 7 puntos EN TOTAL la trayectoria completa de cada video (yolo26n_ft_v2, ByteTrack combo, unión prudente).

Todos los segmentos se unen en orden temporal como un único recorrido. Los huecos sin ave (entre segmentos) no son
recorrido: no reparten puntos ni se dibujan, la curva se corta en ellos y se marcan. Los CSV de eval_gt2\\union\\prudente
no traen posición suavizada, así que se suaviza (cx, cy) por segmento con Savitzky-Golay (ventana ~0,2 s, grado 2).
Métodos: (a) 7 instantes equiespaciados contando solo el tiempo con ave; (b) puntos salientes con Visvalingam-Whyatt.
Un punto que caiga en un tramo es_puente se mueve al observado más cercano y se marca. Por los puntos de cada tramo pasa
una curva Catmull-Rom centrípeta. Forma: distancia media y máxima del recorrido denso suavizado a esas curvas.
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

SRC = Path(r"D:\video_eval\reentreno\v2\eval_gt2\union\prudente")
VIDEOS = Path(r"C:\Users\ACER\Documents\logo\video-val")
OUT = VIDEOS / "resultados" / "trayectorias_7pts"
N = 7
SMOOTH_S = 0.2
NAMES = {"doj1xa6ksqm": "DoJ1", "gu30nofexla": "Gu30", "i5befvb5mxg": "i5bE", "j4pkgwgjaus": "J4PK"}
METHODS = {"a_uniforme_tiempo": "uniforme en el tiempo", "b_salientes": "salientes (Visvalingam-Whyatt)"}


def smooth(xy, fps):
    w = max(5, int(round(SMOOTH_S * fps)) | 1)
    if len(xy) <= w:
        return xy.copy()
    return np.c_[savgol_filter(xy[:, 0], w, 2), savgol_filter(xy[:, 1], w, 2)]


def uniform_time(xy, n=N):
    # el recorrido tiene una fila por frame con ave, así que igual número de filas es igual tiempo con ave
    return np.linspace(0, len(xy) - 1, n).round().astype(int).tolist()


def visvalingam(xy, n=N):
    idx = list(range(len(xy)))
    area = lambda a, b, c: 0.5 * abs((xy[b, 0] - xy[a, 0]) * (xy[c, 1] - xy[a, 1]) - (xy[c, 0] - xy[a, 0]) * (xy[b, 1] - xy[a, 1]))
    A = [np.inf] + [area(idx[i - 1], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)] + [np.inf]
    while len(idx) > n:
        k = int(np.argmin(A))
        floor = A[k]  # área efectiva: una vecina nunca baja de la del punto eliminado
        del idx[k], A[k]
        for j in (k - 1, k):
            if 0 < j < len(idx) - 1:
                A[j] = max(area(idx[j - 1], idx[j], idx[j + 1]), floor)
    return idx


def fix_bridges(idx, bridge):
    """Mueve a un punto observado los que caen en un puente, sin repetir ni desordenar. Devuelve índices y marcas."""
    obs = np.flatnonzero(~bridge)
    out, moved, used = [], [], set()
    for i in idx:
        was = bool(bridge[i])
        cand = sorted(obs, key=lambda o: abs(o - i)) if was else [i]
        j = next((c for c in cand if c not in used), i)
        used.add(j); out.append(int(j)); moved.append(was)
    order = np.argsort(out)
    return [out[o] for o in order], [moved[o] for o in order]


def catmull_rom(P, per_seg=40, alpha=0.5):
    P = np.asarray(P, float)
    if len(P) == 1:
        return P.copy()
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(len(Q) - 3):
        p0, p1, p2, p3 = Q[i:i + 4]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** alpha
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** alpha
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** alpha
        t = np.linspace(t1, t2, per_seg, endpoint=i == len(Q) - 4)[:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    return np.vstack(out)


def curves_by_stretch(P, seg_ids):
    """Una curva por tramo continuo: se corta cuando dos puntos consecutivos están en segmentos distintos."""
    cuts = np.flatnonzero(np.diff(seg_ids) != 0) + 1
    return [catmull_rom(p) for p in np.split(P, cuts)]


def shape_error(dense, curve):
    d = np.sqrt(((dense[:, None, :] - curve[None, :, :]) ** 2).sum(-1)).min(1)
    return float(d.mean()), float(d.max())


def gaps_between(sel):
    """Huecos entre puntos consecutivos: (índice del punto anterior, segundos sin ave)."""
    return [(k, float(sel.t_s.iloc[k + 1] - sel.t_s.iloc[k])) for k in range(len(sel) - 1)
            if sel.segment_id.iloc[k] != sel.segment_id.iloc[k + 1]]


def mark_gaps(ax, P, gaps, color="0.35"):
    for k, secs in gaps:
        mid = (P[k] + P[k + 1]) / 2
        ax.annotate(f"hueco {secs:.1f} s", mid, ha="center", va="center", fontsize=9, color=color, style="italic")


def sketch(ax, curves, P, gaps, title=None, numbers=False):
    for c in curves:
        ax.plot(c[:, 0], c[:, 1], color="black", lw=3, solid_capstyle="round")
    ax.scatter(P[:, 0], P[:, 1], s=26, c="black", zorder=3)
    if numbers:
        for i, (x, y) in enumerate(P, 1):
            ax.annotate(str(i), (x, y), xytext=(6, 6), textcoords="offset points", fontsize=10, color="0.25")
        mark_gaps(ax, P, gaps)
    allp = np.vstack(curves)
    m = 0.08 * max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]), 1)
    ax.set_xlim(allp[:, 0].min() - m, allp[:, 0].max() + m); ax.set_ylim(allp[:, 1].max() + m, allp[:, 1].min() - m)
    ax.set_aspect("equal"); ax.set_facecolor("white"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)


def sketch_norm(ax, curves, P, gaps, W, H, title):
    """Misma escala para todos: el cuadro completo del video, normalizado de 0 a 1 (proporción 16:9)."""
    s = np.array([W, H], float)
    for c in curves:
        ax.plot(c[:, 0] / W, c[:, 1] / H, color="black", lw=3, solid_capstyle="round")
    Pn = P / s
    ax.scatter(Pn[:, 0], Pn[:, 1], s=26, c="black", zorder=3)
    for i, (x, y) in enumerate(Pn, 1):
        ax.annotate(str(i), (x, y), xytext=(6, 6), textcoords="offset points", fontsize=10, color="0.25")
    mark_gaps(ax, Pn, gaps)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec="0.6", lw=1))
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(1.01, -0.01); ax.set_aspect(H / W); ax.axis("off")
    ax.set_title(title, fontsize=12)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("trayectoria_*.csv"))
    fig, axes = plt.subplots(len(files), 2, figsize=(24, 27))
    fig_n, axes_n = plt.subplots(len(files), 2, figsize=(24, 27))
    summary = []
    for i, f in enumerate(files):
        key = next(k for k in NAMES if k in f.stem); name = NAMES[key]
        video = next(p for p in VIDEOS.glob("test_*") if key in p.name.lower())
        cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS); W, H = int(cap.get(3)), int(cap.get(4))
        t = pd.read_csv(f).sort_values("frame").reset_index(drop=True)
        dense = np.vstack([smooth(s[["cx_px", "cy_px"]].to_numpy(), fps) for _, s in t.groupby("segment_id", sort=False)])
        bridge = t.es_puente.astype(bool).to_numpy()
        seg_ids = t.segment_id.to_numpy()
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t.frame.median())); _, fr = cap.read(); cap.release()
        bg = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        presence_s = len(t) / fps
        for j, (m, fn) in enumerate((("a_uniforme_tiempo", uniform_time), ("b_salientes", visvalingam))):
            idx, moved = fix_bridges(fn(dense), bridge)
            P = dense[idx]
            sel = t.iloc[idx]
            curves = curves_by_stretch(P, seg_ids[idx])
            gaps = gaps_between(sel)
            e_mean, e_max = shape_error(dense, np.vstack(curves))
            size = float(np.hypot(*np.ptp(dense, axis=0)))
            summary.append(dict(video=name, metodo=m, segmentos=t.segment_id.nunique(), segmentos_con_punto=sel.segment_id.nunique(),
                                filas_densas=len(t), t_ini=round(float(t.t_s.iloc[0]), 2), t_fin=round(float(t.t_s.iloc[-1]), 2),
                                tiempo_con_ave_s=round(presence_s, 2), puntos_movidos=int(sum(moved)), huecos_entre_puntos=len(gaps),
                                error_medio_px=round(e_mean, 1), error_max_px=round(e_max, 1),
                                error_medio_pct_diagonal=round(100 * e_mean / size, 1)))
            hueco = [False] * len(idx)
            for k, _ in gaps:
                hueco[k + 1] = True
            pd.DataFrame(dict(punto=range(1, len(idx) + 1), frame=sel.frame.to_numpy(), t_s=sel.t_s.round(3).to_numpy(),
                              segment_id=sel.segment_id.to_numpy(), x_px=P[:, 0].round(1), y_px=P[:, 1].round(1),
                              x_norm=(P[:, 0] / W).round(4), y_norm=(P[:, 1] / H).round(4), es_puente_corregido=moved,
                              hueco_antes=hueco)).to_csv(OUT / f"{name}_{m}.csv", index=False)
            # con fondo
            f1, a1 = plt.subplots(figsize=(12, 12 * H / W))
            a1.imshow(bg, alpha=0.55)
            for sid in np.unique(seg_ids):
                d = dense[seg_ids == sid]
                a1.plot(d[:, 0], d[:, 1], color="0.3", lw=1)
            for c in curves:
                a1.plot(c[:, 0], c[:, 1], color="crimson", lw=2.5)
            a1.scatter(P[:, 0], P[:, 1], s=60, c="crimson", edgecolors="white", zorder=5)
            for k, (x, y) in enumerate(P, 1):
                a1.annotate(str(k), (x, y), xytext=(6, 6), textcoords="offset points", fontsize=11, weight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            mark_gaps(a1, P, gaps, color="black")
            a1.set_title(f"{name} | {METHODS[m]} | {sel.t_s.iloc[0]:.1f}-{sel.t_s.iloc[-1]:.1f} s | "
                         f"{len(gaps)} huecos | error medio {e_mean:.0f} px", fontsize=11); a1.axis("off")
            f1.savefig(OUT / f"{name}_{m}_fondo.png", dpi=110, bbox_inches="tight"); plt.close(f1)
            # solo trazo
            f2, a2 = plt.subplots(figsize=(10, 10))
            sketch(a2, curves, P, gaps)
            f2.savefig(OUT / f"{name}_{m}_trazo.png", dpi=110, bbox_inches="tight", facecolor="white"); plt.close(f2)
            sketch(axes[i, j], curves, P, gaps, f"{name} | {METHODS[m]} | {sel.t_s.iloc[0]:.1f}-{sel.t_s.iloc[-1]:.1f} s | "
                   f"{len(gaps)} huecos | error medio {e_mean:.0f} px ({100 * e_mean / size:.1f} %)", numbers=True)
            sketch_norm(axes_n[i, j], curves, P, gaps, W, H, f"{name} | {METHODS[m]} | {sel.t_s.iloc[0]:.1f}-{sel.t_s.iloc[-1]:.1f} s | {len(gaps)} huecos")
    fig.tight_layout()
    fig.savefig(OUT / "trayectorias_7pts.jpg", dpi=100, facecolor="white", pil_kwargs={"quality": 92}); plt.close(fig)
    fig_n.tight_layout()
    fig_n.savefig(OUT / "trayectorias_7pts_misma_escala.jpg", dpi=100, facecolor="white", pil_kwargs={"quality": 92}); plt.close(fig_n)
    S = pd.DataFrame(summary); S.to_csv(OUT / "resumen_7pts.csv", index=False)
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
