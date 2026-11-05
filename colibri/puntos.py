"""Reduce una trayectoria a unos pocos puntos y dibuja la curva que pasa por ellos."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

N = 7
SMOOTH_S = 0.2
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


def puntos_clave(t, fps, n=N):
    """t: trayectoria ordenada por frame. Devuelve el recorrido denso y, por método, índices, puntos, curvas y error."""
    dense = np.vstack([smooth(s[["cx_px", "cy_px"]].to_numpy(), fps) for _, s in t.groupby("segment_id", sort=False)])
    bridge = t.es_puente.astype(bool).to_numpy()
    seg_ids = t.segment_id.to_numpy()
    out = dict(dense=dense, seg_ids=seg_ids, metodos={})
    for m, fn in (("a_uniforme_tiempo", uniform_time), ("b_salientes", visvalingam)):
        idx, moved = fix_bridges(fn(dense, n), bridge)
        pts = dense[idx]
        sel = t.iloc[idx]
        curves = curves_by_stretch(pts, seg_ids[idx])
        gaps = gaps_between(sel)
        e_mean, e_max = shape_error(dense, np.vstack(curves))
        out["metodos"][m] = dict(idx=idx, moved=moved, P=pts, sel=sel, curves=curves, gaps=gaps, e_mean=e_mean, e_max=e_max,
                                 size=float(np.hypot(*np.ptp(dense, axis=0))))
    return out


def tabla_puntos(r, W, H):
    """CSV de 7 puntos de un método: la columna hueco_antes marca dónde se corta la curva."""
    pts, sel = r["P"], r["sel"]
    hueco = [False] * len(r["idx"])
    for k, _ in r["gaps"]:
        hueco[k + 1] = True
    return pd.DataFrame(dict(punto=range(1, len(r["idx"]) + 1), frame=sel.frame.to_numpy(), t_s=sel.t_s.round(3).to_numpy(),
                             segment_id=sel.segment_id.to_numpy(), x_px=pts[:, 0].round(1), y_px=pts[:, 1].round(1),
                             x_norm=(pts[:, 0] / W).round(4), y_norm=(pts[:, 1] / H).round(4), es_puente_corregido=r["moved"],
                             hueco_antes=hueco))


def lamina_fondo(sp, r, bg, W, H, titulo, salida):
    f1, a1 = plt.subplots(figsize=(12, 12 * H / W))
    a1.imshow(bg, alpha=0.55)
    for sid in np.unique(sp["seg_ids"]):
        d = sp["dense"][sp["seg_ids"] == sid]
        a1.plot(d[:, 0], d[:, 1], color="0.3", lw=1)
    for c in r["curves"]:
        a1.plot(c[:, 0], c[:, 1], color="crimson", lw=2.5)
    pts = r["P"]
    a1.scatter(pts[:, 0], pts[:, 1], s=60, c="crimson", edgecolors="white", zorder=5)
    for k, (x, y) in enumerate(pts, 1):
        a1.annotate(str(k), (x, y), xytext=(6, 6), textcoords="offset points", fontsize=11, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    mark_gaps(a1, pts, r["gaps"], color="black")
    a1.set_title(titulo, fontsize=11); a1.axis("off")
    f1.savefig(salida, dpi=110, bbox_inches="tight"); plt.close(f1)


def lamina_trazo(r, salida):
    f2, a2 = plt.subplots(figsize=(10, 10))
    sketch(a2, r["curves"], r["P"], r["gaps"])
    f2.savefig(salida, dpi=110, bbox_inches="tight", facecolor="white"); plt.close(f2)
