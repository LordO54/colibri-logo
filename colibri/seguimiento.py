"""Detección y seguimiento cuadro a cuadro, y unión de fragmentos en una trayectoria."""
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

IMGSZ, CONF_CALL, IOU_MATCH = 640, 0.1, 0.5
TRACKERS = {"bytetrack": "bytetrack.yaml", "ocsort": "ocsort.yaml"}
MIN_SEG_S = 0.5    # los segmentos más cortos se descartan
CUT_THR = 0.5      # correlación de histograma entre cuadros consecutivos por debajo de la cual hay corte de escena


def escribir_tracker(name: str, overrides: dict, destino: Path) -> Path:
    """YAML del tracker: el de Ultralytics con estos cambios y sin compensar movimiento de cámara."""
    from ultralytics.utils import ROOT as ULTRA_ROOT
    src = ULTRA_ROOT / "cfg" / "trackers" / TRACKERS[name]
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg.update({k: v for k, v in overrides.items() if k in cfg})
    if "gmc_method" in cfg:
        cfg["gmc_method"] = "none"
    destino = Path(destino); destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return destino


def track_video(weights: Path, video: Path, tracker: str, cfg_path: Path, conf=CONF_CALL, imgsz=IMGSZ) -> pd.DataFrame:
    from ultralytics import YOLO
    model = YOLO(str(weights))                  # modelo nuevo por video y configuración: estado limpio
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS)
    rows, frame_idx, t0 = [], 0, pd.Timestamp.now()
    while True:
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ok, frame = cap.read()
        if not ok:
            break
        r = model.track(frame, persist=True, tracker=str(cfg_path), conf=conf, imgsz=imgsz, verbose=False)[0]
        if r.boxes is not None and r.boxes.is_track:
            for (cx, cy, w, h), tid, c in zip(r.boxes.xywh.cpu().numpy(), r.boxes.id.int().cpu().tolist(),
                                             r.boxes.conf.cpu().numpy()):
                rows.append(dict(frame=frame_idx, t_ms=t_ms, track_id=tid, cx_px=cx, cy_px=cy,
                                 w_px=w, h_px=h, conf=float(c)))
        frame_idx += 1
    cap.release()
    secs = (pd.Timestamp.now() - t0).total_seconds()
    df = pd.DataFrame(rows, columns=["frame", "t_ms", "track_id", "cx_px", "cy_px", "w_px", "h_px", "conf"])
    df.attrs.update({"fps_declared": fps, "n_frames": frame_idx, "tracker": tracker, "video": str(video),
                     "weights": str(weights), "cfg": str(cfg_path), "fps_proceso": round(frame_idx / max(secs, 1e-6), 1)})
    return df


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2]); br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def to_xyxy(df: pd.DataFrame) -> np.ndarray:
    return np.c_[df.cx_px - df.w_px / 2, df.cy_px - df.h_px / 2, df.cx_px + df.w_px / 2, df.cy_px + df.h_px / 2]


def scene_cuts(path: Path, cache_dir: Path) -> list:
    """Cuadros donde hay un corte de escena. Se guardan en cache_dir para no volver a leer el video."""
    path = Path(path); f = Path(cache_dir) / f"{path.stem[:60]}.json"
    if f.exists():
        return json.loads(f.read_text())
    cap = cv2.VideoCapture(str(path)); prev, cuts, i = None, [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        ok, fr = cap.retrieve()
        hsv = cv2.cvtColor(cv2.resize(fr, (160, 90)), cv2.COLOR_BGR2HSV)
        h = cv2.normalize(cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256]), None).flatten()
        if prev is not None and cv2.compareHist(prev, h, cv2.HISTCMP_CORREL) < CUT_THR:
            cuts.append(i)  # el corte ocurre entre i-1 e i
        prev, i = h, i + 1
    cap.release()
    f.parent.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(cuts))
    return cuts


def accepted_rows(tr):
    """Aceptación de fragmentos del notebook: el mayor primero; fuera los que se solapan en el tiempo con uno aceptado."""
    span = tr.groupby("track_id").frame.agg(["min", "max", "size"]).sort_values("size", ascending=False)
    acc = []
    for tid, s in span.iterrows():
        if any(not (s["max"] < span.loc[a, "min"] or s["min"] > span.loc[a, "max"]) for a in acc):
            continue
        acc.append(tid)
    return tr[tr.track_id.isin(acc)].sort_values("frame").drop_duplicates("frame").reset_index(drop=True)


def union_tuned(tr, max_gap_s, max_dist, cuts):
    fps = tr.attrs["fps_declared"]
    if tr.empty:
        return pd.DataFrame(columns=["frame", "t_s", "cx_px", "cy_px", "w_px", "h_px", "conf", "track_id",
                                     "interpolado", "segment_id"])
    a = accepted_rows(tr)
    cuts = np.array(cuts)
    seg = np.zeros(len(a), int)
    for i in range(1, len(a)):
        p, q = a.iloc[i - 1], a.iloc[i]
        gap = (q.frame - p.frame) / fps
        size = (p.w_px + p.h_px) / 2
        dist = np.hypot(q.cx_px - p.cx_px, q.cy_px - p.cy_px)
        cut = ((cuts > p.frame) & (cuts <= q.frame)).any() if len(cuts) else False
        seg[i] = seg[i - 1] + (not (gap <= max_gap_s + 1e-9 and dist <= max_dist * size and not cut))
    a["segment_id"] = seg
    parts = []
    for sid, g in a.groupby("segment_id"):
        if len(g) < MIN_SEG_S * fps:
            continue
        full = g.set_index("frame").reindex(range(g.frame.min(), g.frame.max() + 1))
        miss = full.cx_px.isna()
        full[["cx_px", "cy_px", "w_px", "h_px"]] = full[["cx_px", "cy_px", "w_px", "h_px"]].interpolate()
        full["interpolado"] = miss
        full["segment_id"] = sid
        parts.append(full.reset_index().rename(columns={"index": "frame"}))
    if not parts:
        return union_tuned(tr.iloc[:0], max_gap_s, max_dist, cuts)
    out = pd.concat(parts, ignore_index=True)
    out["segment_id"] = out.segment_id.rank(method="dense").astype(int) - 1
    out["t_s"] = out.frame / fps
    return out[["frame", "t_s", "cx_px", "cy_px", "w_px", "h_px", "conf", "track_id", "interpolado", "segment_id"]]


def draw(video, traj, g, title, out):
    cap = cv2.VideoCapture(str(video))
    mid = int(traj.frame.median()) if len(traj) else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid); ok, f = cap.read(); cap.release()
    fig, ax = plt.subplots(figsize=(12, 12 * f.shape[0] / f.shape[1]))
    ax.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), alpha=0.5)
    cmap = plt.get_cmap("tab10")
    for sid, s in traj.groupby("segment_id"):
        c = cmap(int(sid) % 10)
        s = s.sort_values("frame")
        real = ~s.interpolado.astype(bool)
        # tramos reales en línea continua, puentes interpolados en discontinua
        brk = (real != real.shift()).cumsum()
        for _, part in s.groupby(brk):
            ax.plot(part.cx_px, part.cy_px, color=c, lw=2, ls="-" if real.loc[part.index[0]] else "--")
        ax.annotate(f"s{sid}", (s.cx_px.iloc[0], s.cy_px.iloc[0]), color=c, fontsize=9, weight="bold")
    gx = [(b[0][0] + b[0][2]) / 2 for b in g.boxes if b]; gy = [(b[0][1] + b[0][3]) / 2 for b in g.boxes if b]
    if gx:   # sin verdad de terreno (videos propios) no se dibuja
        ax.scatter(gx, gy, marker="x", c="red", s=20, label="verdad de terreno")
    ax.plot([], [], "k--", label="puente (interpolado)")
    ax.set_title(title, fontsize=10); ax.axis("off"); ax.legend(loc="lower left", fontsize=8)
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def bridges(traj, W, H):
    """Tramos interpolados: (segmento, frame inicio, frame fin, longitud px); y el número de puentes largos."""
    out = []
    if traj.empty:
        return out, 0
    t = traj.sort_values("frame").reset_index(drop=True)
    interp = t.interpolado.astype(bool).to_numpy()
    i = 0
    while i < len(t):
        if interp[i]:
            j = i
            while j < len(t) and interp[j] and t.segment_id[j] == t.segment_id[i]:
                j += 1
            a, b = max(i - 1, 0), min(j, len(t) - 1)
            L = float(np.hypot(t.cx_px[b] - t.cx_px[a], t.cy_px[b] - t.cy_px[a]))
            out.append(dict(segment_id=int(t.segment_id[i]), frame_ini=int(t.frame[i]), frame_fin=int(t.frame[j - 1]), long_px=L))
            i = j
        else:
            i += 1
    thr = min(W, H) / 4
    return out, sum(1 for o in out if o["long_px"] > thr)
