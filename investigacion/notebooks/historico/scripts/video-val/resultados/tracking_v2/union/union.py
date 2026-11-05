"""Ajuste de la unión de fragmentos de track (build_trajectory) a nivel de visita.

Unidades del notebook anterior (build_trajectory en colibri_tracking.ipynb):
  - JOIN_MAX_GAP = 30 FRAMES entre el fin de un fragmento y el inicio del siguiente (0,5 s a 60 fps, 2 s a 15 fps);
  - JOIN_MAX_DIST = 1,5 en MÚLTIPLOS del lado medio de la caja de referencia ((w + h) / 2, píxeles), distancia
    euclídea entre centros;
  - la unión solo decide si un fragmento entra en la identidad principal; el segmento lo define después el hueco de
    interpolación (INTERP_MAX_GAP = 10 frames), así que dos fragmentos unidos con un hueco > 10 frames quedan en
    segmentos distintos.
Versión ajustable (configuraciones 'g*_d*'): el hueco va en SEGUNDOS y el segmento es la cadena de uniones. Se recorren
las filas aceptadas en orden temporal, y dos filas consecutivas siguen en el mismo segmento si el hueco es
<= max_gap_s, la distancia es <= max_dist * lado medio y no hay un corte de escena entre ellas. Dentro de un segmento
los huecos se interpolan (puente). La aceptación de fragmentos es la del notebook: por tamaño y descartando los que
se solapan en el tiempo con uno aceptado. Los segmentos con menos de 0,5 s de filas reales se descartan
(MIN_FRAG_LEN = 15 frames del notebook, ~0,5 s a 30 fps).
"""
import json, sys, zipfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

V2 = Path(r"D:\video_eval\reentreno\v2")
OUT = V2 / "union"
PROJ = Path(r"C:\Users\ACER\Documents\logo\video-val")
sys.path.insert(0, str(V2 / "tracking"))
import tracking_test as T  # noqa: E402  (funciones del notebook anterior y configuraciones del tracker)

CFG_NAME = "b_nuevo_bt_combo"
WEIGHTS, TRACKER, OVER = T.CONFIGS[CFG_NAME]
IOU_MATCH = 0.5
MIN_SEG_S = 0.5
CUT_THR = 0.5  # correlación de histograma HS entre frames consecutivos por debajo de la cual hay corte de escena
GRID = [(g, d) for g in (0.25, 0.5, 1, 2, 3, 5) for d in (1, 1.5, 2.5, 4)]
TUNE_TRAIN = ["p8uux2kbhh0", "kkimys9gove", "jrsom3z_ows", "vimu4v2jgww"]


def slug(stem, maxlen=48):
    import re
    s = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return s[:maxlen].rstrip("_")


# ---------- videos, verdad de terreno y cortes ----------
def list_videos():
    vv = pd.read_csv(PROJ / "eval_frames" / "videos.csv")
    vids = {}
    for r in vv.itertuples():
        vids[r.video_id] = dict(path=PROJ / r.file, rol="vv")
    for p in PROJ.glob("train_*.mp4"):
        vid = slug(p.stem)
        if any(k in vid for k in TUNE_TRAIN):
            vids[vid] = dict(path=p, rol="train")
    for p in PROJ.glob("test_*.mp4"):
        vids[slug(p.stem)] = dict(path=p, rol="test")
    for v in vids.values():
        cap = cv2.VideoCapture(str(v["path"]))
        v.update(W=int(cap.get(3)), H=int(cap.get(4)), fps=cap.get(cv2.CAP_PROP_FPS)); cap.release()
    return vids


def yolo_boxes(text, W, H):
    out = []
    for l in text.splitlines():
        if l.strip():
            _, cx, cy, w, h = map(float, l.split())
            out.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return out


def load_gt(vids):
    rows = []
    # vv: todos sus frames etiquetados en la versión de Roboflow (rf_humming), menos los 4 de ignorar_frames.txt
    with zipfile.ZipFile(PROJ / "eval_frames" / "prueba_etiquetas_corregidas.zip") as z:
        ignore = set(z.read("ignorar_frames.txt").decode().split())
    sel = pd.read_csv(V2 / "split_manifest.csv")
    sel_vv = {(r.video_id, int(r.frame)): r.split for r in sel[sel.origen == "vv"].itertuples()}
    for lab in (V2 / "rf_humming" / "test" / "labels").glob("vv*.txt"):
        stem = lab.stem.split("_jpg.rf.")[0]
        vid, f = stem.rsplit("_f", 1)
        f = int(f)
        if stem in ignore:
            continue
        v = vids[vid]
        rows.append(dict(video_id=vid, frame=f, boxes=yolo_boxes(lab.read_text(), v["W"], v["H"]),
                         split=sel_vv.get((vid, f), "no_usado")))
    # train_ y test_: los frames de split_manifest
    for r in sel[sel.origen == "nuevo"].itertuples():
        if r.video_id not in vids:
            continue
        v = vids[r.video_id]
        lab = Path(r.ruta.replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        rows.append(dict(video_id=r.video_id, frame=int(r.frame), boxes=yolo_boxes(lab.read_text(), v["W"], v["H"]),
                         split=r.split))
    GT = pd.DataFrame(rows).sort_values(["video_id", "frame"]).reset_index(drop=True)
    GT["n_gt"] = GT.boxes.str.len()
    return GT


def scene_cuts(path: Path) -> list:
    f = OUT / "cortes" / f"{path.stem[:60]}.json"
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


def get_tracks(vid, v):
    d = V2 / "tracking" / CFG_NAME
    out = d / f"{vid}__detallada.csv"
    if not out.exists():
        d = OUT / "tracks"; d.mkdir(parents=True, exist_ok=True); out = d / f"{vid}__detallada.csv"
    if out.exists():
        tr = pd.read_csv(out); tr.attrs.update(json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8")))
        return tr
    print("tracking", vid, flush=True)
    tr = T.track_video(WEIGHTS, v["path"], TRACKER, T.OUT / "cfg" / f"tracker_{CFG_NAME}.yaml")
    tr.to_csv(out, index=False); out.with_suffix(".meta.json").write_text(json.dumps(tr.attrs, indent=2), encoding="utf-8")
    return tr


# ---------- unión ----------
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


def union_reference(tr):
    traj, _ = T.build_trajectory(tr)
    return traj.rename(columns={"segmento": "segment_id"})


# ---------- métricas de visita ----------
def visit_metrics(traj, g):
    """g: frames GT de un video. Frames con 2+ cajas se excluyen (no forman ni separan visitas)."""
    g = g[g.n_gt <= 1].sort_values("frame")
    present = (g.n_gt == 1).to_numpy()
    visit = np.where(present, np.cumsum(np.r_[present[0], present[1:] & ~present[:-1]]) - 1, -1)
    by_frame = traj.set_index("frame") if len(traj) else None
    match, n_pred = [], 0  # (visita, segmento) por caja GT cubierta
    for (f, boxes), vi in zip(zip(g.frame, g.boxes), visit):
        if by_frame is None or f not in by_frame.index:
            continue
        r = by_frame.loc[f]
        n_pred += 1
        if vi >= 0:
            pb = np.array([[r.cx_px - r.w_px / 2, r.cy_px - r.h_px / 2, r.cx_px + r.w_px / 2, r.cy_px + r.h_px / 2]])
            if T.iou_matrix(np.array(boxes), pb)[0, 0] >= IOU_MATCH:
                match.append((vi, int(r.segment_id)))
    n_vis, n_box = int(visit.max() + 1) if len(visit) else 0, int(present.sum())
    M = pd.DataFrame(match, columns=["v", "s"])
    seg_per_visit = M.groupby("v").s.nunique().reindex(range(n_vis), fill_value=0)
    bridges = int((M.groupby("s").v.nunique() - 1).clip(lower=0).sum()) if len(M) else 0
    if len(M):
        C = M.value_counts().unstack(fill_value=0)
        ri, ci = linear_sum_assignment(-C.to_numpy())
        idtp = int(C.to_numpy()[ri, ci].sum())
    else:
        idtp = 0
    return dict(visitas=n_vis, cajas=n_box, cubiertas=len(M), pred_en_gt=n_pred, idtp=idtp,
                seg_por_visita=float(seg_per_visit[seg_per_visit > 0].mean()) if (seg_per_visit > 0).any() else 0.0,
                visitas_sin_segmento=int((seg_per_visit == 0).sum()), puentes_falsos=bridges,
                cobertura=len(M) / max(n_box, 1), IDF1_visita=2 * idtp / max(n_box + n_pred, 1),
                segmentos=int(traj.segment_id.nunique()) if len(traj) else 0)


def aggregate(rows):
    d = pd.DataFrame(rows)
    s = d[["visitas", "cajas", "cubiertas", "pred_en_gt", "idtp", "visitas_sin_segmento", "puentes_falsos", "segmentos"]].sum()
    w = d.visitas - d.visitas_sin_segmento
    return dict(**s.to_dict(), seg_por_visita=float((d.seg_por_visita * w).sum() / max(w.sum(), 1)),
                cobertura=s.cubiertas / max(s.cajas, 1), IDF1_visita=2 * s.idtp / max(s.cajas + s.pred_en_gt, 1))


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
    ax.scatter(gx, gy, marker="x", c="red", s=20, label="verdad de terreno")
    ax.plot([], [], "k--", label="puente (interpolado)")
    ax.set_title(title, fontsize=10); ax.axis("off"); ax.legend(loc="lower left", fontsize=8)
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def evaluate(vids, GT, configs, rol, extra=None):
    rows = []
    for vid, v in vids.items():
        if v["rol"] not in rol:
            continue
        tr = get_tracks(vid, v); cuts = scene_cuts(v["path"]); g = GT[GT.video_id == vid]
        for name, fn in configs.items():
            traj = fn(tr, cuts)
            m = visit_metrics(traj, g)
            rows.append(dict(config=name, video_id=vid, rol=v["rol"], cortes=len(cuts), **m))
            if extra:
                gv = g[g.split == "valid"]
                if len(gv):
                    mv = visit_metrics(traj, gv)
                    rows[-1].update(cobertura_valid=mv["cobertura"], cajas_valid=mv["cajas"])
                extra(name, vid, v, traj, g)
    return pd.DataFrame(rows)


def main():
    vids = list_videos()
    GT = load_gt(vids)
    multi = GT[GT.n_gt >= 2]
    print("GT por video:", GT.groupby("video_id").agg(frames=("frame", "size"), cajas=("n_gt", "sum")).to_string())
    print("frames con 2+ cajas (excluidos de las métricas de visita):", len(multi), multi.groupby("video_id").size().to_dict())
    for vid, v in vids.items():
        print(vid, v["rol"], f"{v['fps']:.2f} fps", "cortes:", scene_cuts(v["path"]), flush=True)

    configs = {"actual_30f_1.5": lambda tr, cuts: union_reference(tr)}
    for gs, dm in GRID:
        configs[f"g{gs}s_d{dm}"] = (lambda gs, dm: lambda tr, cuts: union_tuned(tr, gs, dm, cuts))(gs, dm)

    R = evaluate(vids, GT, configs, {"vv", "train"}, extra=lambda *a: None)
    R.to_csv(OUT / "union_rejilla_por_video.csv", index=False)
    A = pd.DataFrame([dict(config=c, **aggregate(g.to_dict("records"))) for c, g in R.groupby("config", sort=False)])
    # Criterio fijado antes de mirar: mínimos puentes falsos, luego máximo IDF1 de visita, luego menos segmentos por visita
    A = A.sort_values(["puentes_falsos", "IDF1_visita", "seg_por_visita"], ascending=[True, False, True]).reset_index(drop=True)
    cand = A[A.config != "actual_30f_1.5"]
    chosen = cand.iloc[0].config
    A.to_csv(OUT / "union_rejilla.csv", index=False)
    print(A.to_string(index=False))
    print("ELEGIDA:", chosen)

    # Confirmación única en test_
    (OUT / "test").mkdir(exist_ok=True)

    def save(name, vid, v, traj, g):
        tag = "actual" if name.startswith("actual") else "elegida"
        traj.to_csv(OUT / "test" / f"trayectoria_{tag}_{vid}.csv", index=False)
        m = visit_metrics(traj, g)
        draw(v["path"], traj, g, f"{vid[30:]} | {tag} {name} | seg {m['segmentos']} | visitas {m['visitas']} | "
             f"puentes falsos {m['puentes_falsos']} | IDF1v {m['IDF1_visita']:.2f}", OUT / "test" / f"trayectoria_{tag}_{vid}.png")

    test_cfg = {"actual_30f_1.5": configs["actual_30f_1.5"], chosen: configs[chosen]}
    Rt = evaluate(vids, GT, test_cfg, {"test"}, extra=save)
    At = pd.DataFrame([dict(config=c, video_id="GLOBAL", **aggregate(g.to_dict("records"))) for c, g in Rt.groupby("config", sort=False)])
    tune_cmp = A[A.config.isin(test_cfg)].assign(conjunto="ajuste")
    Rt = pd.concat([Rt, At], ignore_index=True)
    Rt.to_csv(OUT / "union_test.csv", index=False)
    (OUT / "eleccion_union.json").write_text(json.dumps({"elegida": chosen, "criterio":
        "min puentes_falsos, luego max IDF1_visita, luego min seg_por_visita (rejilla en vv + train_)"}, indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print(tune_cmp.to_string(index=False))
        print(Rt.to_string(index=False))
    # cobertura sólo en frames valid (lo que el modelo no vio)
    Rv = R[R.config.isin(test_cfg)][["config", "video_id", "cobertura", "cobertura_valid", "cajas_valid"]] \
        if "cobertura_valid" in R else None
    if Rv is not None:
        Rv.to_csv(OUT / "union_ajuste_valid.csv", index=False)
        print(Rv.to_string(index=False))


if __name__ == "__main__":
    main()
