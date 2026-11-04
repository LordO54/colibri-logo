"""Tracking de los 4 videos test_ con 5 configuraciones fijas (sin barrido) y evaluación en los frames etiquetados.

Reutiliza tal cual las funciones de colibri_tracking.ipynb (tracker_cfg, track_video, evaluate_tracks,
build_trajectory) con IMGSZ 640, CONF_CALL 0.1, IOU_MATCH 0.5, gmc_method none y persist=True.
Cambia solo la verdad de terreno: las etiquetas corregidas de los 216 frames test_ (split_manifest.csv),
guardadas a <= 1280 px; las etiquetas son normalizadas, así que se pasan a píxeles del video original.
"""
import json, re
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from ultralytics import YOLO
from ultralytics.utils import ROOT as ULTRA_ROOT

PROJ = Path(r"C:\Users\ACER\Documents\logo\video-val")
V2 = Path(r"D:\video_eval\reentreno\v2")
OUT = V2 / "tracking"
NEW = PROJ / "resultados" / "yolo26n_ft_v2" / "best.pt"
OLD = PROJ / "resultados" / "yolo26n_ds2_ds1" / "best.pt"
IMGSZ, CONF_CALL, IOU_MATCH = 640, 0.1, 0.5
SHORT_S = 0.5  # tracks más cortos que esto se consideran sospechosos de ser cajas espurias

TRACKERS = {"bytetrack": "bytetrack.yaml", "ocsort": "ocsort.yaml"}
BASE = {"bytetrack": {"track_buffer": 60}, "ocsort": {"track_buffer": 60, "use_byte": True}}
COMBO = {"track_buffer": 300, "match_thresh": 0.95, "new_track_thresh": 0.5}
CONFIGS = {  # nombre: (pesos, tracker, overrides sobre el YAML de Ultralytics)
    "a_nuevo_oc_combo": (NEW, "ocsort", {**BASE["ocsort"], **COMBO}),
    "b_nuevo_bt_combo": (NEW, "bytetrack", {**BASE["bytetrack"], **COMBO}),
    "c_nuevo_oc_base": (NEW, "ocsort", BASE["ocsort"]),
    "d_nuevo_bt_base": (NEW, "bytetrack", BASE["bytetrack"]),
    "e_viejo_oc_combo": (OLD, "ocsort", {**BASE["ocsort"], **COMBO}),
}
DECISION = ["a_nuevo_oc_combo", "b_nuevo_bt_combo", "c_nuevo_oc_base", "d_nuevo_bt_base"]  # e es solo referencia


# ---------- sección 3 del notebook anterior ----------
def tracker_cfg(name: str, cfg_name: str, overrides: dict) -> Path:
    src = ULTRA_ROOT / "cfg" / "trackers" / TRACKERS[name]
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg.update({k: v for k, v in overrides.items() if k in cfg})
    if "gmc_method" in cfg:
        cfg["gmc_method"] = "none"        # cámara fija
    dst = OUT / "cfg" / f"tracker_{cfg_name}.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return dst


def track_video(weights: Path, video: Path, tracker: str, cfg_path: Path, conf=CONF_CALL, imgsz=IMGSZ) -> pd.DataFrame:
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


# ---------- sección 4 del notebook anterior ----------
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2]); br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def to_xyxy(df: pd.DataFrame) -> np.ndarray:
    return np.c_[df.cx_px - df.w_px / 2, df.cy_px - df.h_px / 2, df.cx_px + df.w_px / 2, df.cy_px + df.h_px / 2]


def evaluate_tracks(GT: pd.DataFrame, vid: str, tr: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    g = GT[GT.video_id == vid]
    by_frame = dict(tuple(tr.groupby("frame")))
    per = []
    for r in g.itertuples():
        p = by_frame.get(r.frame, tr.iloc[:0])
        gt = np.array(r.boxes, dtype=float).reshape(-1, 4)
        iou = iou_matrix(gt, to_xyxy(p))
        matched = []
        if iou.size:
            gi, pi = linear_sum_assignment(-iou)
            matched = [(i, j) for i, j in zip(gi, pi) if iou[i, j] >= IOU_MATCH]
        ids = p.track_id.to_numpy()
        per.append({"frame": r.frame, "n_gt": len(gt), "n_pred": len(p), "tp": len(matched),
                    "id": int(ids[matched[0][1]]) if matched else None,           # un ave por video
                    "iou": float(iou[matched[0]]) if matched else np.nan})
    per = pd.DataFrame(per)
    seq = per.id.dropna().astype(int).to_numpy()
    idsw = int((seq[1:] != seq[:-1]).sum()) if len(seq) else 0
    n_gt, n_pred, tp = per.n_gt.sum(), per.n_pred.sum(), per.tp.sum()
    idtp = int(pd.Series(seq).value_counts().iloc[0]) if len(seq) else 0
    m = {"video_id": vid, "tracker": tr.attrs["tracker"], "n_gt": int(n_gt), "TP": int(tp), "FN": int(n_gt - tp),
         "FP": int(n_pred - tp), "IDSW": idsw, "ids_por_ave": int(len(set(seq))),
         "ids_totales": int(tr.track_id.nunique()),
         "recall": tp / max(n_gt, 1), "precision": tp / max(n_pred, 1),
         "MOTA": 1 - (n_gt - tp + n_pred - tp + idsw) / max(n_gt, 1),
         "IDF1": 2 * idtp / max(n_gt + n_pred, 1), "IoU_medio": per.iou.mean(),
         "fps_proceso": tr.attrs.get("fps_proceso"), "idtp": idtp}
    return m, per


# ---------- sección 5 del notebook anterior ----------
INTERP_MAX_GAP = 10
JOIN_MAX_GAP = 30
JOIN_MAX_DIST = 1.5
MIN_FRAG_LEN = 15


def build_trajectory(tr: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    fps = tr.attrs["fps_declared"]
    if tr.empty:
        return tr.iloc[:0], {"ids_unidos": [], "ids_descartados": []}
    span = tr.groupby("track_id").frame.agg(["min", "max", "size"]).sort_values("size", ascending=False)
    main = [span.index[0]]
    discarded = []
    for tid in span.index[1:]:
        s = span.loc[tid]
        used = tr[tr.track_id.isin(main)]
        if ((used.frame >= s["min"]) & (used.frame <= s["max"])).any():
            discarded.append(int(tid)); continue
        before, after = used[used.frame < s["min"]], used[used.frame > s["max"]]
        frag = tr[tr.track_id == tid].sort_values("frame")
        ok, near = False, False
        for ref, ends, gap in ((before.sort_values("frame").tail(1), frag.head(1), s["min"] - before.frame.max() if len(before) else np.inf),
                               (after.sort_values("frame").head(1), frag.tail(1), after.frame.min() - s["max"] if len(after) else np.inf)):
            if len(ref) and gap <= JOIN_MAX_GAP:
                near = True
                size = (ref.w_px.iloc[0] + ref.h_px.iloc[0]) / 2
                d = np.hypot(ref.cx_px.iloc[0] - ends.cx_px.iloc[0], ref.cy_px.iloc[0] - ends.cy_px.iloc[0])
                ok |= d <= JOIN_MAX_DIST * size
        if not near:
            ok = s["size"] >= MIN_FRAG_LEN
        (main if ok else discarded).append(int(tid))
    t = tr[tr.track_id.isin(main)].sort_values("frame").drop_duplicates("frame")
    t = t.set_index("frame")[["cx_px", "cy_px", "w_px", "h_px", "conf"]]
    full = t.reindex(range(t.index.min(), t.index.max() + 1))
    missing = full.cx_px.isna()
    run_id = (~missing).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    fill = missing & (run_len <= INTERP_MAX_GAP)
    interp = full.interpolate(limit_area="inside")
    full.loc[fill, ["cx_px", "cy_px", "w_px", "h_px"]] = interp.loc[fill, ["cx_px", "cy_px", "w_px", "h_px"]]
    full["interpolado"] = fill
    full = full[full.cx_px.notna()].copy()
    full["segmento"] = (np.diff(full.index, prepend=full.index[0]) > 1).cumsum()
    full["t_s"] = full.index / fps
    v = np.full(len(full), np.nan)
    for _, seg in full.groupby("segmento"):
        if len(seg) >= 7:
            w = min(11, len(seg) // 2 * 2 - 1)
            vx = savgol_filter(seg.cx_px, w, 2, deriv=1) * fps
            vy = savgol_filter(seg.cy_px, w, 2, deriv=1) * fps
            v[full.index.get_indexer(seg.index)] = np.hypot(vx, vy)
    full["v_px_s"] = v
    full = full.reset_index().rename(columns={"index": "frame"})
    return full[["frame", "t_s", "cx_px", "cy_px", "w_px", "h_px", "conf", "interpolado", "segmento", "v_px_s"]], \
        {"ids_unidos": [int(x) for x in main], "ids_descartados": discarded}


# ---------- específico de esta prueba ----------
def load_gt(videos: dict) -> pd.DataFrame:
    sm = pd.read_csv(V2 / "split_manifest.csv")
    sm = sm[(sm.split == "test") & (sm.origen == "nuevo")]
    rows = []
    for r in sm.itertuples():
        W, H = videos[r.video_id]["wh"]
        lab = Path(r.ruta.replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        boxes = []
        for l in lab.read_text().splitlines():
            if l.strip():
                _, cx, cy, w, h = map(float, l.split())
                boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        rows.append({"video_id": r.video_id, "frame": int(r.frame), "boxes": boxes})
    GT = pd.DataFrame(rows).sort_values(["video_id", "frame"]).reset_index(drop=True)
    GT["n_gt"] = GT.boxes.str.len()
    return GT


def short_tracks(tr: pd.DataFrame, GT_v: pd.DataFrame) -> pd.DataFrame:
    fps = tr.attrs["fps_declared"]
    g = tr.groupby("track_id").agg(ini=("frame", "min"), fin=("frame", "max"), n=("frame", "size"),
                                   cx=("cx_px", "mean"), cy=("cy_px", "mean"), w=("w_px", "mean"),
                                   h=("h_px", "mean"), conf=("conf", "mean"))
    g["dur_s"] = (g.fin - g.ini + 1) / fps
    g = g[g.dur_s < SHORT_S].reset_index()
    # ¿había ave etiquetada en el frame GT más cercano?
    gf = GT_v.frame.to_numpy()
    g["ave_en_gt_cercano"] = [bool(GT_v.n_gt.iloc[np.abs(gf - (a + b) / 2).argmin()]) if len(gf) else None
                              for a, b in zip(g.ini, g.fin)]
    g["t_ini_s"] = g.ini / fps
    return g


def draw_traj(video: Path, tr, traj, GT_v, title, out):
    cap = cv2.VideoCapture(str(video))
    mid = int(traj.frame.median()) if len(traj) else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid); ok, f = cap.read(); cap.release()
    fig, ax = plt.subplots(figsize=(12, 12 * f.shape[0] / f.shape[1]))
    ax.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), alpha=0.55)
    for tid, g in tr.groupby("track_id"):           # todos los IDs, finos
        ax.plot(g.cx_px, g.cy_px, lw=0.8, alpha=0.7)
    sc = None
    for _, seg in traj.groupby("segmento"):         # trayectoria final, gruesa y coloreada por tiempo
        sc = ax.scatter(seg.cx_px, seg.cy_px, c=seg.t_s, cmap="viridis", s=8, vmin=0, vmax=tr.attrs["n_frames"] / tr.attrs["fps_declared"])
    gx = [(b[0][0] + b[0][2]) / 2 for b in GT_v.boxes if b]; gy = [(b[0][1] + b[0][3]) / 2 for b in GT_v.boxes if b]
    ax.scatter(gx, gy, marker="x", c="red", s=22, label="verdad de terreno")
    ax.set_title(title, fontsize=10); ax.axis("off"); ax.legend(loc="lower left", fontsize=8)
    if sc is not None:
        fig.colorbar(sc, ax=ax, shrink=0.6, label="t (s)")
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def main():
    videos = {}
    for p in sorted(PROJ.glob("test_*.mp4")):
        vid = re.sub(r"[^a-z0-9]+", "_", p.stem.lower()).strip("_")[:48].rstrip("_")
        cap = cv2.VideoCapture(str(p))
        videos[vid] = {"path": p, "wh": (int(cap.get(3)), int(cap.get(4)))}
        cap.release()
    GT = load_gt(videos)
    print("frames GT:", len(GT), "| cajas:", int(GT.n_gt.sum()), "| máx. aves por frame:", int(GT.n_gt.max()))
    print(GT.groupby("video_id").agg(frames=("frame", "size"), cajas=("n_gt", "sum")).to_string())

    rows, short_all, traj_rows, TR = [], [], [], {}
    for cfg_name, (weights, tracker, over) in CONFIGS.items():
        cfg_path = tracker_cfg(tracker, cfg_name, over)
        d = OUT / cfg_name; d.mkdir(parents=True, exist_ok=True)
        per_vid = []
        for vid, info in videos.items():
            out = d / f"{vid}__detallada.csv"
            if out.exists():  # fuente de verdad: no se sobrescribe; se recarga
                tr = pd.read_csv(out); tr.attrs.update(json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8")))
            else:
                print(f"{cfg_name} {vid}", flush=True)
                tr = track_video(weights, info["path"], tracker, cfg_path)
                tr.to_csv(out, index=False)
                out.with_suffix(".meta.json").write_text(json.dumps(tr.attrs, indent=2), encoding="utf-8")
            TR[(cfg_name, vid)] = tr
            m, per = evaluate_tracks(GT, vid, tr)
            traj, tinfo = build_trajectory(tr)
            st = short_tracks(tr, GT[GT.video_id == vid])
            m.update({"config": cfg_name, "segmentos": int(traj.segmento.nunique()) if len(traj) else 0,
                      "frames_traj": len(traj), "tracks_cortos": len(st)})
            per_vid.append(m)
            short_all.append(st.assign(config=cfg_name, video_id=vid))
            traj_rows.append({"config": cfg_name, "video_id": vid, **tinfo, "frames_traj": len(traj),
                              "cobertura_video": len(traj) / tr.attrs["n_frames"],
                              "interpolados": int(traj.interpolado.sum()) if len(traj) else 0,
                              "segmentos": int(traj.segmento.nunique()) if len(traj) else 0})
        pv = pd.DataFrame(per_vid)
        n_gt, tp, fp = pv.n_gt.sum(), pv.TP.sum(), pv.FP.sum()
        rows += per_vid
        rows.append({"config": cfg_name, "video_id": "GLOBAL", "tracker": tracker, "n_gt": int(n_gt), "TP": int(tp),
                     "FN": int(n_gt - tp), "FP": int(fp), "IDSW": int(pv.IDSW.sum()),
                     "ids_por_ave": int(pv.ids_por_ave.sum()), "ids_totales": int(pv.ids_totales.sum()),
                     "recall": tp / n_gt, "precision": tp / max(tp + fp, 1),
                     "MOTA": 1 - (n_gt - tp + fp + pv.IDSW.sum()) / n_gt, "IDF1": 2 * pv.idtp.sum() / (n_gt + tp + fp),
                     "IoU_medio": float(np.average(pv.IoU_medio.fillna(0), weights=pv.TP)) if tp else np.nan,
                     "segmentos": int(pv.segmentos.sum()), "frames_traj": int(pv.frames_traj.sum()),
                     "tracks_cortos": int(pv.tracks_cortos.sum()), "idtp": int(pv.idtp.sum())})
    cols = ["config", "video_id", "tracker", "n_gt", "TP", "FN", "FP", "IDSW", "ids_por_ave", "ids_totales",
            "tracks_cortos", "recall", "precision", "MOTA", "IDF1", "IoU_medio", "segmentos", "frames_traj", "fps_proceso"]
    M = pd.DataFrame(rows)[cols]
    M.to_csv(OUT / "metricas_tracking_test.csv", index=False)
    S = pd.concat(short_all, ignore_index=True)
    S.to_csv(OUT / "tracks_cortos.csv", index=False)
    pd.DataFrame(traj_rows).to_csv(OUT / "resumen_trayectorias.csv", index=False)

    # Elección con el criterio de 4.1 del notebook anterior, solo entre configuraciones del modelo nuevo
    G = M[(M.video_id == "GLOBAL") & M.config.isin(DECISION)].copy()
    G["_k1"] = G.ids_por_ave + G.IDSW
    G = G.sort_values(["_k1", "IDF1", "MOTA"], ascending=[True, False, False])
    best_score = tuple(G.iloc[0][["_k1", "IDF1", "MOTA"]].astype(float).round(3))
    tied = [c for c in G.config if tuple(G[G.config == c].iloc[0][["_k1", "IDF1", "MOTA"]].astype(float).round(3)) == best_score]
    bt = [c for c in tied if "_bt_" in c]
    chosen = bt[0] if bt else G.config.iloc[0]
    print("Configuración elegida:", chosen, "(empate)" if len(tied) > 1 else "")

    fin = OUT / "ganadora"; fin.mkdir(exist_ok=True)
    for vid, info in videos.items():
        tr = TR[(chosen, vid)]
        traj, _ = build_trajectory(tr)
        raw = tr.assign(t_s=tr.frame / tr.attrs["fps_declared"]).rename(
            columns={"cx_px": "cx", "cy_px": "cy", "w_px": "w", "h_px": "h"})
        raw[["frame", "t_s", "track_id", "cx", "cy", "w", "h", "conf"]].to_csv(fin / f"tracks_{vid}.csv", index=False)
        traj.to_csv(fin / f"trayectoria_{vid}.csv", index=False)
        m = M[(M.config == chosen) & (M.video_id == vid)].iloc[0]
        draw_traj(info["path"], tr, traj, GT[GT.video_id == vid],
                  f"{vid} | {chosen} | IDs {m.ids_totales} | IDSW {m.IDSW} | IDF1 {m.IDF1:.2f} | R {m.recall:.2f}",
                  fin / f"trayectoria_{vid}.png")
    (OUT / "eleccion.json").write_text(json.dumps({"elegida": chosen, "empate": tied, "criterio":
        "min(ids_por_ave + IDSW), luego max IDF1, luego max MOTA; ByteTrack si empatan"}, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 260, "display.max_columns", 30, "display.precision", 3):
        print(M.to_string(index=False))
        print(S[["config", "video_id", "track_id", "t_ini_s", "dur_s", "n", "cx", "cy", "w", "h", "conf", "ave_en_gt_cercano"]].to_string(index=False))


if __name__ == "__main__":
    main()
