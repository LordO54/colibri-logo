"""Variantes de 'delante' (i5bE): escala de z, diagnóstico del signo de 5 y 7 y variante guiada por observación.

Ejes con la misma unidad (ancho del cuadro): x de 0 a 1, y multiplicada por alto/ancho. z_vis = k * z_norm. La escala es una
decisión de diseño. La flor es un cilindro vertical cuyo radio es la mitad del ancho de su región (supuesto a partir de la imagen).
El diagnóstico es posterior: ya se sabe qué se espera, así que elegir un método por este resultado sería ajustarse a él.
"""
import glob, sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

V2 = Path(r"D:\video_eval\reentreno\v2")
SP = V2 / "profundidad" / "siete_puntos"
HERE = SP / "escala_z"
RES = Path(r"C:\Users\ACER\Documents\logo\video-val\resultados\profundidad\siete_puntos\escala_z"); RES.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(V2 / "trayectorias_7"), str(V2 / "profundidad")]
from trayectorias_7pts import catmull_rom
from comun_profundidad import motion_mask, read_frame

AR = 1080 / 1920
FLOR, FONDO = (0.54, 0.38, 0.69, 0.58), (0.30, 0.03, 0.62, 0.16)
ETIQUETA = {5: "delante", 6: "delante", 7: "delante"}   # etiqueta visual de Yilber; el resto queda sin etiqueta
D = pd.read_csv(SP / "salidas" / "puntos_i5bE_salientes_delante.csv")
cx, r = (FLOR[0] + FLOR[2]) / 2, (FLOR[2] - FLOR[0]) / 2
yt, yb = FLOR[1] * AR, FLOR[3] * AR
EXT = max(np.ptp(D.x_norm), np.ptp(D.y_norm) * AR)


def k_for(z, pct):
    return pct * EXT / np.ptp(z)


def coords(z):
    return np.c_[D.x_norm, D.y_norm * AR, z]


def superior(a, z, err, title, zlim):
    P = coords(z); C = catmull_rom(P)
    a.add_patch(plt.Circle((cx, 0), r, fc="gold", ec="goldenrod", alpha=0.4)); a.axhline(0, color="goldenrod", lw=1, ls=":")
    a.plot(C[:, 0], C[:, 2], color="black", lw=2.3); a.errorbar(P[:, 0], P[:, 2], yerr=err, fmt="o", color="crimson", mec="w", capsize=3, zorder=5)
    for i, q in enumerate(P, 1):
        a.annotate(str(i), (q[0], q[2]), xytext=(6, 5), textcoords="offset points", fontsize=11, weight="bold")
    a.set_aspect("equal"); a.set_xlim(0.30, 0.85); a.set_ylim(*zlim); a.grid(alpha=0.3); a.set_xlabel("x"); a.set_ylabel("z (atrás arriba, delante abajo)"); a.set_title(title, fontsize=11)


def views(z, err, title, path):
    P = coords(z); C = catmull_rom(P); zlim = (min(-0.12, P[:, 2].min() - err.max() - 0.03), max(0.12, P[:, 2].max() + err.max() + 0.03))
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    a = ax[0]; a.add_patch(plt.Rectangle((FLOR[0], yt), FLOR[2] - FLOR[0], yb - yt, fc="gold", ec="goldenrod", alpha=0.35, label="flor (cilindro, supuesto)"))
    a.plot(C[:, 0], C[:, 1], color="black", lw=2.3); a.scatter(P[:, 0], P[:, 1], c="crimson", edgecolors="w", s=45, zorder=5)
    for i, q in enumerate(P, 1):
        a.annotate(str(i), (q[0], q[1]), xytext=(6, 5), textcoords="offset points", fontsize=11, weight="bold")
    a.set_aspect("equal"); a.set_xlim(0.30, 0.85); a.set_ylim(0.42, 0.08); a.grid(alpha=0.3); a.set_xlabel("x"); a.set_ylabel("y"); a.set_title("frontal x-y (igual que la 2D)", fontsize=11); a.legend(fontsize=9)
    superior(ax[1], z, err, "superior x-z", zlim)
    a = ax[2]; a.add_patch(plt.Rectangle((-r, yt), 2 * r, yb - yt, fc="gold", ec="goldenrod", alpha=0.35)); a.axvline(0, color="goldenrod", lw=1, ls=":")
    a.plot(C[:, 2], C[:, 1], color="black", lw=2.3); a.errorbar(P[:, 2], P[:, 1], xerr=err, fmt="o", color="crimson", mec="w", capsize=3, zorder=5)
    for i, q in enumerate(P, 1):
        a.annotate(str(i), (q[2], q[1]), xytext=(6, 5), textcoords="offset points", fontsize=11, weight="bold")
    a.set_aspect("equal"); a.set_xlim(*zlim); a.set_ylim(0.42, 0.08); a.grid(alpha=0.3); a.set_xlabel("z (delante a la izquierda, atrás a la derecha)"); a.set_ylabel("y"); a.set_title("lateral z-y", fontsize=11)
    fig.suptitle(title, fontsize=13); fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def html(z, title, path):
    P = coords(z); C = catmull_rom(P); th = np.linspace(0, 2 * np.pi, 48); T, YY = np.meshgrid(th, [yt, yb])
    g = go.Figure()
    g.add_trace(go.Surface(x=cx + r * np.cos(T), y=r * np.sin(T), z=-YY, colorscale=[[0, "gold"], [1, "gold"]], showscale=False, opacity=0.45, name="flor (cilindro, supuesto)", showlegend=True))
    g.add_trace(go.Scatter3d(x=C[:, 0], y=C[:, 2], z=-C[:, 1], mode="lines", line=dict(color="black", width=7), name="curva"))
    g.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 2], z=-P[:, 1], mode="markers+text", marker=dict(size=5, color="crimson"), text=[str(i) for i in D.punto], name="7 puntos", customdata=np.c_[D.frame, D.z_norm], hovertemplate="f%{customdata[0]} z_norm=%{customdata[1]:.3f}"))
    g.update_layout(title=title, scene=dict(xaxis_title="x", yaxis_title="z (delante -, atrás +)", zaxis_title="−y", aspectmode="data"))
    g.write_html(path, include_plotlyjs=True)


def escalas():
    z = D.z_norm.to_numpy(); e = D.z_norm_std.to_numpy()
    K = {"k=1 (actual)": 1.0, "25 %": k_for(z, .25), "50 %": k_for(z, .5), "100 %": k_for(z, 1.0), "200 % (extra)": k_for(z, 2.0)}
    print("extensión x-y:", round(EXT, 3), "| rango z_norm:", round(float(np.ptp(z)), 3), "| k:", {n: round(k, 2) for n, k in K.items()})
    zmax = K["200 % (extra)"] * (z.max() + e.max()) + 0.03; zmin = min(-0.14, K["200 % (extra)"] * z.min() - 0.03)
    fig, ax = plt.subplots(1, len(K), figsize=(5.6 * len(K), 9))
    for a, (n, k) in zip(ax, K.items()):
        tag = n.split()[0].replace("=", "").replace("%", "pct") if "%" in n else "k1"
        superior(a, z * k, e * k, f"{n} | k = {k:.2f} | punto 6: {z[5] * k:+.3f}", (zmin, zmax))
        views(z * k, e * k, f"i5bE | escala {n}, k = {k:.2f} | escala elegida por diseño, no medida", HERE / f"vistas_{tag}.jpg")
        html(z * k, f"i5bE | escala {n}, k = {k:.2f} | diseño, no medida", HERE / f"trayectoria3d_{tag}.html")
    fig.suptitle("i5bE: vista superior con 4 escalas + 1 extra. Mismas unidades en x y z. El círculo es la flor como cilindro (supuesto).", fontsize=14)
    fig.tight_layout(); fig.savefig(HERE / "lamina_escalas.jpg", dpi=85); plt.close(fig)
    return K


def diagnostico():
    A = YOLO(str(SP / "yolo26s-depth.pt"))
    path = glob.glob(r"C:\Users\ACER\Documents\logo\video-val\test_*i5bEfvb5mxg*")[0]
    t = pd.read_csv(glob.glob(str(V2 / "eval_gt2" / "union" / "prudente" / "trayectoria_*i5befvb5mxg*.csv"))[0]).set_index("frame")
    cap = cv2.VideoCapture(path); n, W, H = int(cap.get(7)), int(cap.get(3)), int(cap.get(4))
    bg = np.median(np.stack([f for f in (read_frame(cap, i) for i in np.linspace(int(0.03 * n), n - 1, 60).astype(int)) if f is not None]), axis=0).astype(np.uint8)
    depth = lambda fr: A.predict(fr, imgsz=640, verbose=False)[0].depth.data.cpu().numpy().astype(np.float32)
    bgd = depth(bg)
    R = np.zeros((H, W), bool); R[int(FLOR[1] * H):int(FLOR[3] * H), int(FLOR[0] * W):int(FLOR[2] * W)] = True
    F = np.zeros((H, W), bool); F[int(FONDO[1] * H):int(FONDO[3] * H), int(FONDO[0] * W):int(FONDO[2] * W)] = True
    rows = []
    for p in D.itertuples():
        fr = read_frame(cap, p.frame); b = t.loc[p.frame]; d = depth(fr)
        x0, y0, x1, y1 = max(0, int(b.cx_px - b.w_px / 2)), max(0, int(b.cy_px - b.h_px / 2)), min(W, int(b.cx_px + b.w_px / 2) + 1), min(H, int(b.cy_px + b.h_px / 2) + 1)
        m = motion_mask(fr, bg, x0, y0, x1, y1)
        M = np.zeros((H, W), bool); M[y0:y1, x0:x1] = m
        Box = np.zeros((H, W), bool); Box[y0:y1, x0:x1] = True
        fl, fo = float(np.median(d[R & ~Box])) if (R & ~Box).sum() > 50 else float(np.median(d[R])), float(np.median(d[F & ~Box]))
        den = fo - fl; z_ave = float(np.median(d[M]))
        # b) flor local: anillo de la flor alrededor de la caja
        pad = int(0.5 * max(x1 - x0, y1 - y0)); Ring = np.zeros((H, W), bool); Ring[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = True; Ring &= R & ~Box
        z_b = (z_ave - float(np.median(d[Ring]))) / den if Ring.sum() >= 50 else np.nan
        # c) núcleo del cuerpo: máscara erosionada y percentil 25
        kk = max(3, int(round(0.08 * min(x1 - x0, y1 - y0))) | 1); core = cv2.erode(m.astype(np.uint8), np.ones((kk, kk), np.uint8)).astype(bool)
        z_c = (float(np.percentile(d[y0:y1, x0:x1][core], 25)) - fl) / den if core.sum() >= 30 else np.nan
        # d) el ave frente a lo que tapa de la flor: mismos píxeles, cada mapa normalizado con sus propias referencias
        Over = M & R; flb, fob = float(np.median(bgd[R])), float(np.median(bgd[F]))
        z_d = ((float(np.median(d[Over])) - fl) / den - (float(np.median(bgd[Over])) - flb) / (fob - flb)) if Over.sum() >= 30 else np.nan
        rows.append(dict(punto=p.punto, frame=p.frame, etiqueta_yilber=ETIQUETA.get(p.punto, "sin etiqueta"), a_flor_completa=(z_ave - fl) / den, b_flor_local=z_b, c_nucleo_p25=z_c, d_frente_a_lo_que_tapa=z_d, px_tapa_flor=int(Over.sum()), px_anillo=int(Ring.sum()), px_nucleo=int(core.sum())))
    cap.release()
    T = pd.DataFrame(rows); T.to_csv(HERE / "diagnostico_signo.csv", index=False)
    cols = ["a_flor_completa", "b_flor_local", "c_nucleo_p25", "d_frente_a_lo_que_tapa"]
    pd.set_option("display.width", 200); print(T.round(3).to_string(index=False))
    for c in cols:
        lab = T[T.punto.isin(ETIQUETA)][c]; oth = T[~T.punto.isin(ETIQUETA)][c]
        print(c, "| 5, 6 y 7 negativos:", int((lab < 0).sum()), "de", int(lab.notna().sum()), "con dato (de 3) | 1-4 negativos:", int((oth < 0).sum()), "de", int(oth.notna().sum()), "con dato")


def guiada(K):
    s = np.array([-1.0 if ETIQUETA.get(p) == "delante" else np.sign(z) for p, z in zip(D.punto, D.z_norm)])
    zg = s * np.abs(D.z_norm.to_numpy()); k = k_for(zg, .5); zv = zg * k; e = D.z_norm_std.to_numpy() * k
    title = "i5bE | VARIANTE GUIADA, diseño y no medida: signo de la etiqueta de Yilber en 5, 6 y 7; magnitud del modelo"
    views(zv, e, f"{title} | k = {k:.2f}", HERE / "vistas_guiada.jpg"); html(zv, f"{title} | k = {k:.2f}", HERE / "trayectoria3d_guiada.html")
    G = D[["punto", "frame", "t_s", "x_norm", "y_norm", "z_norm", "z_norm_std"]].copy()
    G["signo_origen"] = ["etiqueta Yilber" if p in ETIQUETA else "modelo" for p in D.punto]
    G["z_guiada"] = zg.round(4); G["k"] = round(k, 3); G["z_vis"] = zv.round(4); G["es_diseno_no_medida"] = True
    G.to_csv(HERE / "puntos_i5bE_delante_guiada.csv", index=False)
    print(G[["punto", "frame", "z_norm", "signo_origen", "z_guiada", "z_vis"]].round(3).to_string(index=False)); print("k guiada:", round(k, 2), "| radio del cilindro:", r)


if __name__ == "__main__":
    K = escalas(); diagnostico(); guiada(K)
    for f in HERE.iterdir():
        if f.is_file():
            (RES / f.name).write_bytes(f.read_bytes())
