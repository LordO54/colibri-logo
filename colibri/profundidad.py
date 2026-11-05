"""Profundidad relativa en los puntos de una trayectoria y figura 3D."""
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .puntos import catmull_rom

N_BG, MIN_MASK_PX = 60, 30


def read_frame(cap, f):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, fr = cap.read()
    return fr if ok else None


def motion_mask(frame, bg, x0, y0, x1, y1):
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


class Yolo26:
    variants = ("orig", "flip", "secondary")

    def __init__(self, model, secondary="yolo26m-depth.pt", imgsz=640):
        from ultralytics import YOLO
        self.m, self.s, self.imgsz = YOLO(model), YOLO(secondary), imgsz

    def _run(self, m, img):
        return m.predict(img, imgsz=self.imgsz, verbose=False)[0].depth.data.cpu().numpy().astype(np.float32)

    def map(self, frame_idx, frame, flip=False, secondary=False):
        m = self.s if secondary else self.m
        return self._run(m, frame[:, ::-1].copy())[:, ::-1] if flip else self._run(m, frame)


class Yolo26Crop(Yolo26):
    def __init__(self, model, crop, secondary="yolo26m-depth.pt", imgsz=640):
        super().__init__(model, secondary, imgsz)
        self.crop = crop  # (x0, y0, x1, y1) en píxeles, fijo para todo el video

    def map(self, frame_idx, frame, flip=False, secondary=False):
        x0, y0, x1, y1 = self.crop
        out = np.full(frame.shape[:2], np.nan, np.float32)
        out[y0:y1, x0:x1] = super().map(frame_idx, frame[y0:y1, x0:x1], flip, secondary)
        return out


class VDA:
    variants = ("orig", "flip")

    def __init__(self, cache_dir, name, shape):
        self.o = np.load(Path(cache_dir) / f"{name}_orig.npz")["maps"]
        self.f = np.load(Path(cache_dir) / f"{name}_flip.npz")["maps"]
        self.H, self.W = shape

    def map(self, frame_idx, frame, flip=False, secondary=False):
        src = self.f if flip else self.o
        return cv2.resize(src[int(frame_idx)].astype(np.float32), (self.W, self.H), interpolation=cv2.INTER_LINEAR)


def make_backend(kind, model, name, shape, crop=None, cache_dir=None):
    if kind == "yolo26":
        return Yolo26(model + ".pt")
    if kind == "yolo26_crop":
        return Yolo26Crop(model + ".pt", crop)
    if kind == "vda":
        return VDA(cache_dir, name, shape)
    raise ValueError(kind)


def rect_px(r, W, H):
    return int(r[0] * W), int(r[1] * H), int(r[2] * W), int(r[3] * H)


def background(cap, n):
    fs = [read_frame(cap, i) for i in np.linspace(int(0.03 * n), n - 1, N_BG).astype(int)]
    fs = [f for f in fs if f is not None and cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() >= 5]
    return np.median(np.stack(fs), axis=0).astype(np.uint8)


def median_outside(d, rect, box):
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


def leer_puntos_z(PT, t, cap, be, bg, ref, far, W, H):
    """Lee z en cada punto de PT: original + volteado (TTA) y vecinos a ±2 frames. t: trayectoria indexada por frame."""
    def at(f, tta):
        fr = read_frame(cap, f); box = box_of(t.loc[f], W, H); mask = motion_mask(fr, bg, *box)
        o = read_point(be.map(f, fr), box, mask, ref, far); o["n_px"] = int(mask.sum()); o["box"] = box
        if tta:
            o["tta"] = [o["z_norm"], read_point(be.map(f, fr, flip=True), box, mask, ref, far)["z_norm"]]
            if "secondary" in be.variants:
                o["tta"].append(read_point(be.map(f, fr, secondary=True), box, mask, ref, far)["z_norm"])
        return o
    rows = []
    for r in PT.itertuples():
        o = at(int(r.frame), True); vec = [o["z_norm"]]
        for g in (r.frame - 2, r.frame - 1, r.frame + 1, r.frame + 2):
            if g in t.index and not bool(t.loc[g].es_puente):
                vec.append(at(int(g), False)["z_norm"])
        rows.append(dict(punto=r.punto, frame=r.frame, t_s=r.t_s, segment_id=r.segment_id, x_norm=r.x_norm, y_norm=r.y_norm, z_ave=o["z_ave"], z_flor=o["z_flor"], z_fondo=o["z_fondo"],
                         z_norm=float(np.mean(o["tta"])), z_norm_std=float(np.std(o["tta"])), z_norm_orig=o["z_norm"], vecinos_std=float(np.std(vec)), n_vecinos=len(vec) - 1,
                         metodo_z=o["metodo_z"], n_px_mascara=o["n_px"], den=o["den"], box=o["box"]))
    return pd.DataFrame(rows)


def control_ruido(D):
    """Regla fijada: la forma en z no es fiable si el ruido entre vecinos supera la diferencia típica entre puntos."""
    dif = float(np.median(np.abs(np.diff(D.z_norm)))); vec = float(D.vecinos_std.median()); tta = float(D.z_norm_std.median())
    return dict(dif_entre_puntos=dif, rango_z=float(np.ptp(D.z_norm)), ruido_vecinos=vec, ruido_tta=tta, forma_supera_ruido=bool(dif > vec),
                recortes=int((D.metodo_z != "mask_med").sum()), min_den=float(D.den.abs().min()))


def figura_3d(D, titulo, ref_rect, ref_nombre, AR, salida, xlim=None, ylim=None):
    """Tres vistas (jpg) y figura interactiva (html). x e y en unidades del ancho del cuadro; z = z_norm sin reescalar.
    La referencia se dibuja como un cilindro con el radio de la mitad del ancho de su región (supuesto)."""
    import plotly.graph_objects as go
    rr = ref_rect; P3 = np.c_[D.x_norm, D.y_norm * AR, D.z_norm]; cuts = np.flatnonzero(np.diff(D.segment_id.to_numpy()) != 0) + 1
    C3 = [catmull_rom(p) for p in np.split(P3, cuts)]
    cx, r = (rr[0] + rr[2]) / 2, (rr[2] - rr[0]) / 2; yt, yb = rr[1] * AR, rr[3] * AR
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2)); e = D.z_norm_std
    for a, i, j, tt in ((ax[0], 0, 1, "frontal x-y"), (ax[1], 0, 2, "superior x-z"), (ax[2], 2, 1, "lateral z-y")):
        for c in C3: a.plot(c[:, i], c[:, j], color="black", lw=2.2)
        a.errorbar(P3[:, i], P3[:, j], xerr=e if i == 2 else None, yerr=e if j == 2 else None, fmt="o", color="crimson", mec="w", capsize=3, zorder=5)
        for k, q in enumerate(P3, 1): a.annotate(str(k), (q[i], q[j]), xytext=(6, 5), textcoords="offset points", fontsize=11, weight="bold")
        a.set_title(tt, fontsize=11); a.grid(alpha=0.3)
    ax[0].add_patch(plt.Rectangle((rr[0], yt), 2 * r, yb - yt, fc="gold", alpha=0.35)); ax[0].set_aspect("equal")
    ax[1].add_patch(plt.Circle((cx, 0), r, fc="gold", alpha=0.35)); ax[1].axhline(0, color="goldenrod", ls=":"); ax[1].set_aspect("equal", adjustable="datalim")
    ax[2].add_patch(plt.Rectangle((-r, yt), 2 * r, yb - yt, fc="gold", alpha=0.35)); ax[2].axvline(0, color="goldenrod", ls=":")
    if xlim:
        ax[0].set_xlim(*xlim); ax[1].set_xlim(*xlim)
    if ylim:
        ax[0].set_ylim(*ylim); ax[2].set_ylim(*ylim)
    else:
        ax[0].autoscale_view(); ax[2].autoscale_view(); ax[0].invert_yaxis(); ax[2].invert_yaxis()
    fig.suptitle(f"{titulo} | z_norm: 0 = {ref_nombre}, 1 = fondo", fontsize=12); fig.tight_layout(); fig.savefig(f"{salida}.jpg", dpi=100); plt.show(); plt.close(fig)
    g = go.Figure(); th = np.linspace(0, 2 * np.pi, 48); T, YY = np.meshgrid(th, [yt, yb])
    g.add_trace(go.Surface(x=cx + r * np.cos(T), y=r * np.sin(T), z=-YY, colorscale=[[0, "gold"], [1, "gold"]], showscale=False, opacity=0.4, name=ref_nombre, showlegend=True))
    for c in C3: g.add_trace(go.Scatter3d(x=c[:, 0], y=c[:, 2], z=-c[:, 1], mode="lines", line=dict(color="black", width=7), name="curva"))
    g.add_trace(go.Scatter3d(x=P3[:, 0], y=P3[:, 2], z=-P3[:, 1], mode="markers+text", marker=dict(size=5, color="crimson"), text=[str(i) for i in D.punto], name="7 puntos"))
    g.update_layout(title=titulo, scene=dict(xaxis_title="x", yaxis_title="z_norm", zaxis_title="−y", aspectmode="data")); g.write_html(f"{salida}.html", include_plotlyjs=True)


def tira_puntos(D, cap, ref_rect, W, H, salida):
    """Recorte de cada punto con su z_norm, para revisar a ojo si la lectura tiene sentido."""
    x0, y0, x1, y1 = rect_px(ref_rect, W, H)
    fig, axes = plt.subplots(1, len(D), figsize=(3.2 * len(D), 3.3))
    for ax, r in zip(axes, D.itertuples()):
        fr = read_frame(cap, r.frame); b = r.box; cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        X0, Y0, X1, Y1 = max(0, min(b[0], x0) - 40), max(0, min(b[1], y0) - 40), min(W, max(b[2], x1) + 40), min(H, max(b[3], y1) + 40)
        ax.imshow(cv2.cvtColor(fr[Y0:Y1, X0:X1], cv2.COLOR_BGR2RGB)); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(f"{r.punto} | f{r.frame} | {r.z_norm:+.2f} ± {r.z_norm_std:.2f}", fontsize=10)
    fig.tight_layout(); fig.savefig(salida, dpi=100); plt.show(); plt.close(fig)


def _leer_video_vda(path, short):
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS); out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        s = short / min(f.shape[:2])
        out.append(cv2.cvtColor(cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out), fps


def mapas_vda(videos_vda, cache, sizes, repo, ckpt):
    """Mapas de VDA Small de {nombre: ruta}, original y volteado, en <cache>/<nombre>_{orig,flip}.npz. Los que existen no se rehacen."""
    import gc, time
    import torch
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    pendientes = [(n, p, t) for n, p in videos_vda.items() for t in ("orig", "flip") if not (cache / f"{n}_{t}.npz").exists()]
    if not pendientes:
        return
    repo, ckpt = Path(repo), Path(ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"Falta el modelo de profundidad en {ckpt}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    cwd = os.getcwd(); os.chdir(repo)
    from video_depth_anything.video_depth import VideoDepthAnything
    os.chdir(cwd)
    gc.collect(); torch.cuda.empty_cache()
    m = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    m = m.to("cuda").eval()
    used = {}
    for name, path, tag in pendientes:
        for size in sizes:
            try:
                t0 = time.time(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                frames, fps = _leer_video_vda(path, size)
                if tag == "flip": frames = frames[:, :, ::-1].copy()
                d, _ = m.infer_video_depth(frames, fps, input_size=size, device="cuda", fp32=False)
                break
            except (torch.OutOfMemoryError, MemoryError) as e:
                print(name, tag, "sin memoria a", size, type(e).__name__, flush=True); frames = d = None; gc.collect(); torch.cuda.empty_cache()
        else:
            raise RuntimeError(f"{name}: no cabe en memoria a ninguna resolución; prueba con un clip más corto")
        d = np.asarray(d, dtype=np.float32)
        if tag == "flip": d = d[:, :, ::-1]
        np.savez_compressed(cache / f"{name}_{tag}.npz", maps=d.astype(np.float16))
        used[f"{name}_{tag}"] = size
        print(name, tag, d.shape, f"input {size}", f"{len(d) / (time.time() - t0):.1f} fps", f"pico GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", "rango", float(d.min()), float(d.max()), flush=True)
        frames = d = None; gc.collect(); torch.cuda.empty_cache()
    del m; gc.collect(); torch.cuda.empty_cache()
    previo = json.loads((cache / "info.json").read_text(encoding="utf-8")).get("input_size", {}) if (cache / "info.json").exists() else {}
    (cache / "info.json").write_text(json.dumps(dict(modelo="vda-small", input_size={**previo, **used}, salida="disparidad relativa, mayor = más cerca"), indent=2), encoding="utf-8")
