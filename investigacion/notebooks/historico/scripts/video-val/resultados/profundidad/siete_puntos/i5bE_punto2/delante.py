"""Variante de i5bE con un punto forzado en 12,7-13,7 s (el ave pasa por delante de la flor). No toca las variantes anteriores.

Uso: delante.py seleccionar   (solo visibilidad, sin calcular profundidad)
     delante.py calcular      (profundidad, curva, figuras)
Filtro de visibilidad: observado, conf >= 0,5, máscara >= 30 px, área >= 60 % de la mediana, solape caja-flor <= 20 %.
"""
import glob, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

V2 = Path(r"D:\video_eval\reentreno\v2")
SP = V2 / "profundidad" / "siete_puntos"
HERE = SP / "i5bE_punto2"; OUT = SP / "salidas"
RES = Path(r"C:\Users\ACER\Documents\logo\video-val\resultados\profundidad\siete_puntos")
sys.path[:0] = [str(V2 / "trayectorias_7"), str(V2 / "profundidad")]
from trayectorias_7pts import catmull_rom, smooth
from comun_profundidad import motion_mask, read_frame

T0, T1 = 12.7, 13.7
FPS = 29.97
FLOR, FONDO = (0.54, 0.38, 0.69, 0.58), (0.30, 0.03, 0.62, 0.16)
VIDEO = glob.glob(r"C:\Users\ACER\Documents\logo\video-val\test_*i5bEfvb5mxg*")[0]
TRAJ = glob.glob(str(V2 / "eval_gt2" / "union" / "prudente" / "trayectoria_*i5befvb5mxg*.csv"))[0]
SEL = HERE / "seleccion_delante.json"


def table():
    t = pd.read_csv(TRAJ).sort_values("frame").reset_index(drop=True)
    S = pd.read_csv(V2 / "profundidad" / "salidas" / "stats_i5bE.csv")[["frame", "mask_frac", "ov_ref", "w", "h"]]
    S["px"] = S.mask_frac * S.w * S.h
    t = t.merge(S[["frame", "px", "ov_ref"]], on="frame", how="left")
    t["frac"] = t.px / S.px.median()
    t["visible"] = ~t.es_puente.astype(bool) & (t.conf >= 0.5) & (t.px >= 30) & (t.frac >= 0.6)
    return t


def vw_protected(xy, keep, n=7):
    idx = list(range(len(xy)))
    area = lambda a, b, c: 0.5 * abs((xy[b, 0] - xy[a, 0]) * (xy[c, 1] - xy[a, 1]) - (xy[c, 0] - xy[a, 0]) * (xy[b, 1] - xy[a, 1]))
    A = [np.inf] + [area(idx[i - 1], idx[i], idx[i + 1]) if idx[i] != keep else np.inf for i in range(1, len(idx) - 1)] + [np.inf]
    while len(idx) > n:
        k = int(np.argmin(A)); floor = A[k]
        del idx[k], A[k]
        for j in (k - 1, k):
            if 0 < j < len(idx) - 1 and idx[j] != keep:
                A[j] = max(area(idx[j - 1], idx[j], idx[j + 1]), floor)
    return idx


def seleccionar():
    t = table(); xy = smooth(t[["cx_px", "cy_px"]].to_numpy(), FPS)
    win = t[(t.t_s >= T0) & (t.t_s <= T1)]
    print("frames en el tramo:", len(win), "| observados:", (~win.es_puente.astype(bool)).sum(), "| conf>=0,5:", (win.conf >= .5).sum(), "| área>=60 %:", (win.frac >= .6).sum(),
          "| solape<=20 %:", (win.ov_ref <= .2).sum(), "| solape<=50 %:", (win.ov_ref <= .5).sum(), "| solape mín:", round(win.ov_ref.min(), 2))
    lim = None
    for L in (0.2, 0.5):
        c = win[win.visible & (win.ov_ref <= L)]
        if len(c):
            lim = L; forced = c.sort_values("px", ascending=False).iloc[0]; break
    assert lim is not None, "ningún frame del tramo pasa, ni con solape <= 50 %"
    print(f"FORZADO: f{int(forced.frame)} t={forced.t_s:.2f}s con solape <= {lim} (solape {forced.ov_ref:.2f}, área {forced.frac:.2f}, conf {forced.conf:.2f}); candidatos {len(c)}")
    keep = int(t.index[t.frame == forced.frame][0])
    idx = vw_protected(xy, keep)
    print("VW restringido (frames):", t.frame.iloc[idx].tolist())
    ok = (t.visible & (t.ov_ref <= 0.2)).to_numpy(); out = []
    for k, i in enumerate(idx):
        if i == keep or ok[i]:
            out.append(dict(i=i, movido=False, de=int(t.frame[i]))); continue
        lo = out[-1]["i"] if out else -1
        hi = idx[k + 1] if k + 1 < len(idx) else len(t)
        cand = [j for j in range(lo + 1, hi) if ok[j]]
        if cand:
            j = min(cand, key=lambda q: abs(q - i)); out.append(dict(i=j, movido=True, de=int(t.frame[i])))
        else:
            out.append(dict(i=i, movido=False, de=int(t.frame[i]), sin_candidato=True))
    final = [dict(punto=k + 1, frame=int(t.frame[o["i"]]), t_s=round(float(t.t_s[o["i"]]), 2), de=o["de"], movido=o["movido"], sin_candidato=o.get("sin_candidato", False), forzado=o["i"] == keep,
                  conf=round(float(t.conf[o["i"]]), 2), solape=round(float(t.ov_ref[o["i"]]), 2), area=round(float(t.frac[o["i"]]), 2)) for k, o in enumerate(out)]
    print(pd.DataFrame(final).to_string(index=False))
    SEL.write_text(json.dumps(dict(limite_solape=lim, puntos=final), indent=2), encoding="utf-8")
    import cv2
    cap = cv2.VideoCapture(VIDEO); ts = []
    for p in final:
        fr = read_frame(cap, p["frame"]); r = t[t.frame == p["frame"]].iloc[0]
        cv2.rectangle(fr, (int(r.cx_px - r.w_px / 2), int(r.cy_px - r.h_px / 2)), (int(r.cx_px + r.w_px / 2), int(r.cy_px + r.h_px / 2)), (0, 0, 255), 2)
        c = cv2.resize(fr[120:800, 800:1560], (500, 448)); cv2.putText(c, f"{p['punto']}: f{p['frame']} {p['t_s']}s" + (" FORZADO" if p["forzado"] else ""), (6, 24), 0, .7, (0, 255, 255), 2); ts.append(c)
    ts.append(np.zeros_like(ts[0])); cap.release()
    cv2.imwrite(str(HERE / "verificacion_delante.jpg"), np.vstack([np.hstack(ts[:4]), np.hstack(ts[4:])]))


def calcular():
    import cv2
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from ultralytics import YOLO
    sel = json.loads(SEL.read_text(encoding="utf-8"))["puntos"]
    A, B = YOLO(str(SP / "yolo26s-depth.pt")), YOLO(str(SP / "yolo26m-depth.pt"))
    t = table(); xy = smooth(t[["cx_px", "cy_px"]].to_numpy(), FPS)
    cap = cv2.VideoCapture(VIDEO); fps, n, W, H = cap.get(5), int(cap.get(7)), int(cap.get(3)), int(cap.get(4))
    bg = np.median(np.stack([f for f in (read_frame(cap, i) for i in np.linspace(int(0.03 * n), n - 1, 60).astype(int)) if f is not None]), axis=0).astype(np.uint8)
    rect = lambda r: (int(r[0] * W), int(r[1] * H), int(r[2] * W), int(r[3] * H)); ref, far = rect(FLOR), rect(FONDO)
    depth = lambda m, fr: m.predict(fr, imgsz=640, verbose=False)[0].depth.data.cpu().numpy().astype(np.float32)

    def outside(d, r, box):
        x0, y0, x1, y1 = r; m = np.ones((y1 - y0, x1 - x0), bool)
        a0, b0, a1, b1 = max(box[0], x0), max(box[1], y0), min(box[2], x1), min(box[3], y1)
        if a1 > a0 and b1 > b0: m[b0 - y0:b1 - y0, a0 - x0:a1 - x0] = False
        return float(np.median(d[y0:y1, x0:x1][m])) if m.mean() > 0.2 else float(np.median(d[y0:y1, x0:x1]))

    def read(d, box, mask):
        x0, y0, x1, y1 = box; z = float(np.median(d[y0:y1, x0:x1][mask])); fl, fo = outside(d, ref, box), outside(d, far, box)
        return dict(z_ave=z, z_flor=fl, z_fondo=fo, z_norm=(z - fl) / (fo - fl))

    def box_of(r):
        return (max(0, int(r.cx_px - r.w_px / 2)), max(0, int(r.cy_px - r.h_px / 2)), min(W, int(r.cx_px + r.w_px / 2) + 1), min(H, int(r.cy_px + r.h_px / 2) + 1))

    rows, frames = [], {}
    for p in sel:
        f = p["frame"]; i = int(t.index[t.frame == f][0]); r = t.loc[i]; fr = read_frame(cap, f); box = box_of(r); mask = motion_mask(fr, bg, *box); frames[f] = (fr.copy(), box)
        o = read(depth(A, fr), box, mask)
        v = [o["z_norm"], read(depth(A, fr[:, ::-1].copy())[:, ::-1], box, mask)["z_norm"], read(depth(B, fr), box, mask)["z_norm"]]
        vec = [o["z_norm"]]
        for g in (i - 2, i - 1, i + 1, i + 2):
            if 0 <= g < len(t) and not bool(t.es_puente[g]) and t.frame[g] == f + g - i:
                f2 = read_frame(cap, t.frame[g]); b2 = box_of(t.loc[g]); vec.append(read(depth(A, f2), b2, motion_mask(f2, bg, *b2))["z_norm"])
        rows.append(dict(punto=p["punto"], frame=f, t_s=round(f / fps, 3), segment_id=int(r.segment_id), x_norm=round(xy[i, 0] / W, 4), y_norm=round(xy[i, 1] / H, 4), z_ave=o["z_ave"], z_flor=o["z_flor"], z_fondo=o["z_fondo"],
                         z_norm=float(np.mean(v)), z_norm_std=float(np.std(v)), z_norm_orig=v[0], z_norm_volteo=v[1], z_norm_mod_b=v[2], vecinos_std=float(np.std(vec)), n_vecinos=len(vec) - 1,
                         metodo_z="mask_med", n_px_mascara=int(mask.sum()), forzado=p["forzado"], movido_desde=p["de"] if p["movido"] else "", solape_flor=p["solape"]))
    D = pd.DataFrame(rows); D.to_csv(OUT / "puntos_i5bE_salientes_delante.csv", index=False)
    O = pd.read_csv(OUT / "puntos_i5bE_b_salientes.csv")
    C0, C1 = [catmull_rom(x[["x_norm", "y_norm", "z_norm"]].to_numpy()) for x in (O, D)]
    cx, cy = (FLOR[0] + FLOR[2]) / 2, (FLOR[1] + FLOR[3]) / 2
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    for a, i, j, tt in ((ax[0], 0, 1, "frontal x-y"), (ax[1], 0, 2, "superior x-z"), (ax[2], 2, 1, "lateral z-y")):
        a.plot(C0[:, i], C0[:, j], color="0.65", lw=1, label="salientes original")
        a.plot(C1[:, i], C1[:, j], color="black", lw=2.2, label="con punto forzado")
        Q = D[["x_norm", "y_norm", "z_norm"]].to_numpy()
        a.errorbar(Q[:, i], Q[:, j], xerr=D.z_norm_std if i == 2 else None, yerr=D.z_norm_std if j == 2 else None, fmt="o", color="crimson", ms=6, mec="w", capsize=3, zorder=5)
        for k, q in enumerate(Q, 1):
            a.annotate(str(k), (q[i], q[j]), xytext=(6, 6), textcoords="offset points", fontsize=11, weight="bold")
        fz = Q[D.forzado.to_numpy()][0]; a.scatter(fz[i], fz[j], s=180, facecolors="none", edgecolors="royalblue", linewidths=2, zorder=6, label="punto forzado")
        a.set_title(tt, fontsize=11); a.grid(alpha=0.3)
        a.set_xlabel({0: "x (0-1)", 2: "z_norm (0 = flor, 1 = fondo)"}[i]); a.set_ylabel({1: "y (0-1)", 2: "z_norm (0 = flor, 1 = fondo)"}[j])
    ax[0].add_patch(plt.Rectangle((FLOR[0], FLOR[1]), FLOR[2] - FLOR[0], FLOR[3] - FLOR[1], fill=False, ec="goldenrod", lw=2)); ax[0].set_xlim(0, 1); ax[0].set_ylim(1, 0); ax[0].legend(fontsize=8)
    ax[1].axhline(0, color="goldenrod", lw=2); ax[1].scatter(cx, 0, marker="*", s=220, c="gold", edgecolors="k", zorder=4); ax[1].set_xlim(0, 1)
    ax[2].axvline(0, color="goldenrod", lw=2); ax[2].scatter(0, cy, marker="*", s=220, c="gold", edgecolors="k", zorder=4); ax[2].set_ylim(1, 0)
    fz = D[D.forzado].iloc[0]
    fig.suptitle(f"i5bE | salientes con punto forzado f{fz.frame} | z_norm forzado {fz.z_norm:+.2f} ± {fz.z_norm_std:.2f} | puntos con z < 0: {(D.z_norm < 0).sum()}", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "trayectoria3d_i5bE_salientes_delante.jpg", dpi=100); plt.close(fig)
    g = go.Figure()
    g.add_trace(go.Scatter3d(x=C0[:, 0], y=C0[:, 2], z=-C0[:, 1], mode="lines", line=dict(color="lightgray", width=3), name="salientes original"))
    g.add_trace(go.Scatter3d(x=C1[:, 0], y=C1[:, 2], z=-C1[:, 1], mode="lines", line=dict(color="black", width=7), name="con punto forzado"))
    g.add_trace(go.Scatter3d(x=D.x_norm, y=D.z_norm, z=-D.y_norm, mode="markers+text", marker=dict(size=5, color="crimson"), text=[str(i) for i in D.punto], name="7 puntos", error_y=dict(type="data", array=D.z_norm_std.to_numpy(), visible=True)))
    g.add_trace(go.Mesh3d(x=[FLOR[0], FLOR[2], FLOR[2], FLOR[0]], y=[0, 0, 0, 0], z=[-FLOR[1], -FLOR[1], -FLOR[3], -FLOR[3]], i=[0, 0], j=[1, 2], k=[2, 3], color="gold", opacity=0.45, name="monarda (z = 0)", showlegend=True))
    g.update_layout(title="i5bE | salientes con punto forzado delante", scene=dict(xaxis_title="x (0-1)", yaxis_title="z_norm", zaxis_title="−y (0-1)", aspectmode="cube"))
    g.write_html(OUT / "trayectoria3d_i5bE_salientes_delante.html", include_plotlyjs=True)
    fig, axes = plt.subplots(1, len(D), figsize=(3.3 * len(D), 3.4))
    for ax_, r in zip(axes, D.itertuples()):
        fr, b = frames[r.frame]; cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        ax_.imshow(cv2.cvtColor(fr[120:800, 800:1560], cv2.COLOR_BGR2RGB)); ax_.set_xticks([]); ax_.set_yticks([])
        ax_.set_xlabel(f"{r.punto} | f{r.frame}{' FORZADO' if r.forzado else ''}\nz_norm {r.z_norm:+.2f} ± {r.z_norm_std:.2f}", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "tira_i5bE_salientes_delante.jpg", dpi=100); plt.close(fig)
    cap.release()
    print(D[["punto", "frame", "t_s", "x_norm", "y_norm", "z_norm", "z_norm_std", "vecinos_std", "z_norm_orig", "z_norm_volteo", "z_norm_mod_b", "n_px_mascara", "forzado", "movido_desde", "solape_flor"]].round(3).to_string(index=False))
    for f in OUT.glob("*i5bE_salientes_delante*"):
        (RES / "salidas" / f.name).write_bytes(f.read_bytes())
    for f in ("delante.py", "seleccion_delante.json", "verificacion_delante.jpg"):
        (RES / "i5bE_punto2" / f).write_bytes((HERE / f).read_bytes())


if __name__ == "__main__":
    {"seleccionar": seleccionar, "calcular": calcular}[sys.argv[1]]()
