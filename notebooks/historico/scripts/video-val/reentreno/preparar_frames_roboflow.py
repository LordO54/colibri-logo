"""Selecciona frames de los videos train_/test_ (y de vv01-vv03) para el reentrenamiento y los deja listos para Roboflow.

Política: candidatos cada 0,1 s; se guarda si pasaron >= MIN_GAP (0,4 s; 0,2 s en frames difíciles) y la escena cambió >= DIFF_THR
(fracción de píxeles de una miniatura 192x108 en color que cambian más de 12 niveles), o si pasaron >= MAX_GAP (1,5 s). Tope 120/video.
test_: 6 fps fijos. Valid: último 20 % de cada train_ con 1 s de guarda. Fuga: dHash test vs train, distancia hasta 6.
"""
import argparse, csv, re, sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
DARK_THR = 12  # media de gris por debajo de la cual el frame se considera negro


def slug(stem, maxlen=48):
    s = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return s[:maxlen].rstrip("_")


def signature(frame):
    # en color: un ave verde sobre follaje puede tener casi el mismo gris que el fondo
    small = cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (5, 5), 0).astype(np.int16)


def change(a, b, pix_thr=12):
    return float((np.abs(a - b).max(axis=2) > pix_thr).mean())


def dhash(frame):
    g = cv2.cvtColor(cv2.resize(frame, (9, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if x else "0" for x in bits), 2)


def iter_frames(path, step, wanted=None):
    """Decodifica en orden; devuelve (idx, t_s, frame) cada `step` frames o solo los índices de `wanted`."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if (wanted is None and idx % step == 0) or (wanted is not None and idx in wanted):
            ok, frame = cap.retrieve()
            if ok:
                yield idx, idx / fps, frame
        idx += 1
    cap.release()


class Detector:
    def __init__(self, weights, imgsz, conf):
        from ultralytics import YOLO
        self.m, self.imgsz, self.conf = YOLO(weights), imgsz, conf

    def __call__(self, frame):
        r = self.m.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xywhn = r.boxes.xywhn.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        return [(float(c), *map(float, b)) for c, b in zip(cf, xywhn)]


def analyse(path, cand_dt, det, wanted=None):
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    step = max(1, round(fps * cand_dt))
    rows, n_dark = [], 0
    for idx, t, fr in iter_frames(path, step, wanted):
        if wanted is None and cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).mean() < DARK_THR:
            n_dark += 1  # fundido a negro
            continue
        dets = det(fr) if det else None
        rows.append(dict(frame=idx, t=t, sig=signature(fr), dets=dets))
    if det:
        areas = [d[3] * d[4] for r in rows for d in (r["dets"] or [])]
        med = float(np.median(areas)) if areas else 0.0
        for r in rows:
            d = r["dets"]
            r["max_conf"] = max((x[0] for x in d), default=0.0)
            big = med > 0 and any(x[3] * x[4] > 2.5 * med for x in d)
            r["hard"] = (not d) or r["max_conf"] < 0.5 or big
            r["hard_why"] = "sin_det" if not d else ("conf_baja" if r["max_conf"] < 0.5 else ("caja_grande" if big else ""))
    return fps, n, rows, n_dark


def select(rows, a):
    kept, last = [], None
    for r in rows:
        if last is None:
            kept.append(r); last = r; r["why"] = "primero"; continue
        gap = r["t"] - last["t"]
        d = change(r["sig"], last["sig"])
        r["diff"] = d
        min_gap = a.min_gap_hard if r.get("hard") else a.min_gap
        if gap + 1e-6 < min_gap:
            continue
        if d >= a.diff_thr:
            r["why"] = "cambio_dificil" if r.get("hard") else "cambio"
        elif gap + 1e-6 >= a.max_gap:
            r["why"] = "max_gap"
        else:
            continue
        kept.append(r); last = r
    if len(kept) > a.max_per_video:
        ix = np.linspace(0, len(kept) - 1, a.max_per_video).round().astype(int)
        kept = [kept[i] for i in sorted(set(ix))]
    return kept


def assign_val(kept, duration, a, whole_val):
    cut = duration * (1 - a.val_frac)
    for r in kept:
        if whole_val:
            r["split"] = "valid"
        elif r["t"] >= cut:
            r["split"] = "valid"
        elif r["t"] >= cut - a.guard_s:
            r["split"] = None  # banda de guarda: se descarta
        else:
            r["split"] = "train"
    return [r for r in kept if r["split"]]


def write_frame(out, split, name, frame, dets, cls=0):
    (out / split / "images").mkdir(parents=True, exist_ok=True)
    (out / split / "labels").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / split / "images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(out / split / "labels" / f"{name}.txt", "w") as f:
        for c, x, y, w, h in dets or []:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def materialise(path, vid, rows, split_out, out, det):
    by_idx = {r["frame"]: r for r in rows}
    for idx, t, fr in iter_frames(path, 1, set(by_idx)):
        r = by_idx[idx]
        r["dhash"] = dhash(fr)
        if split_out and out is not None:
            write_frame(out, r["split"], f"{vid}_f{idx:05d}", fr, r.get("dets") if det else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model", help="best.pt para pre-etiquetas y frames difíciles (opcional)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--vv-manifest", type=Path, help="manifest_frames.csv de la prueba anterior")
    p.add_argument("--vv-videos", type=Path, help="videos.csv (mapea vv01..vv03 a su archivo)")
    p.add_argument("--cand-dt", type=float, default=0.1)
    p.add_argument("--min-gap", type=float, default=0.4)
    p.add_argument("--min-gap-hard", type=float, default=0.2)
    p.add_argument("--max-gap", type=float, default=1.5)
    p.add_argument("--diff-thr", type=float, default=0.003, help="fracción de píxeles cambiados (miniatura 192x108)")
    p.add_argument("--max-per-video", type=int, default=120)
    p.add_argument("--test-fps", type=float, default=6.0)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--guard-s", type=float, default=1.0)
    p.add_argument("--val-videos", default="", help="ids (slug) de videos train_ que van enteros a valid, separados por coma")
    p.add_argument("--extra-train", default="", help="nombres de archivo (sin prefijo train_) que se tratan como train, separados por coma")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    extra_train = {s.strip() for s in a.extra_train.split(",") if s.strip()}

    det = Detector(a.model, a.imgsz, a.conf) if a.model else None
    upload = a.out / "roboflow_upload"
    val_videos = {s.strip() for s in a.val_videos.split(",") if s.strip()}
    vids = sorted(f for f in a.videos.iterdir() if f.suffix.lower() in VIDEO_EXT)
    manifest, summary = [], []

    vv_files = {}
    if a.vv_videos and a.vv_manifest:
        with open(a.vv_videos, encoding="utf-8") as f:
            vv_files = {row["file"]: row["video_id"] for row in csv.DictReader(f)}
        with open(a.vv_manifest, encoding="utf-8") as f:
            vv_wanted = {}
            for row in csv.DictReader(f):
                vv_wanted.setdefault(row["video_id"], set()).add(int(row["frame"]))

    for v in vids:
        low = v.name.lower()
        if low.startswith("train_") or v.name in extra_train:
            role, vid = "train", slug(v.stem)
        elif low.startswith("test_"):
            role, vid = "test", slug(v.stem)
        elif v.name in vv_files:
            role, vid = "vv", vv_files[v.name]
        else:
            summary.append(f"IGNORADO (sin prefijo train_/test_ ni vv): {v.name}")
            continue
        print(f"[{role}] {v.name}", flush=True)
        if role == "test":
            fps, n, rows, n_dark = analyse(v, 1.0 / a.test_fps, det)
            for r in rows:
                r["split"], r["why"] = "test", "fijo"
            kept = rows
        else:
            wanted = vv_wanted.get(vid) if role == "vv" else None
            fps, n, rows, n_dark = analyse(v, a.cand_dt, det, wanted)
            kept = select(rows, a)
            kept = assign_val(kept, n / fps, a, vid in val_videos)
        materialise(v, vid, kept, role != "vv" and not a.dry_run, upload, det)
        for r in kept:
            manifest.append(dict(video_id=vid, role=role, split=r["split"], frame=r["frame"], t_s=round(r["t"], 3),
                                 motivo=r.get("why", ""), diff=round(r.get("diff", 0.0), 5),
                                 dificil=r.get("hard_why", ""), n_prelabels=len(r["dets"]) if r.get("dets") is not None else "",
                                 max_conf=round(r.get("max_conf", 0.0), 3) if det else "",
                                 dhash=f"{r.get('dhash', 0):016x}", file=f"{vid}_f{r['frame']:05d}.jpg"))
        cnt = {s: sum(1 for r in kept if r["split"] == s) for s in ("train", "valid", "test")}
        summary.append(f"{role:5s} {vid}: {n} frames @ {fps:.2f} fps ({n / fps:.1f} s), candidatos {len(rows)}, "
                       f"negros descartados {n_dark}, elegidos train {cnt['train']} / valid {cnt['valid']} / test {cnt['test']}")

    a.out.mkdir(parents=True, exist_ok=True)
    cols = ["video_id", "role", "split", "frame", "t_s", "motivo", "diff", "dificil", "n_prelabels", "max_conf", "dhash", "file"]
    with open(a.out / "manifest_reentrenamiento.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(m for m in manifest if m["role"] != "vv")
    with open(a.out / "seleccion_vv.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(m for m in manifest if m["role"] == "vv")

    # Fuga: misma escena en entrenamiento y en prueba.
    tr = [m for m in manifest if m["split"] in ("train", "valid")]
    te = [m for m in manifest if m["split"] == "test"]
    alerts = []
    for m in te:
        hm = int(m["dhash"], 16)
        best = min(((bin(hm ^ int(x["dhash"], 16)).count("1"), x) for x in tr), default=None, key=lambda z: z[0])
        if best and best[0] <= 6:
            alerts.append(dict(test_file=m["file"], train_file=best[1]["file"], hamming=best[0]))
    with open(a.out / "alertas_fuga.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["test_file", "train_file", "hamming"]); w.writeheader(); w.writerows(alerts)
    scenes = sorted({(x["test_file"].rsplit("_f", 1)[0], x["train_file"].rsplit("_f", 1)[0]) for x in alerts})
    summary.append(f"ALERTAS de fuga test-train: {len(alerts)} frames; pares de video: {scenes or 'ninguno'}")

    if not a.dry_run and manifest:
        (upload).mkdir(parents=True, exist_ok=True)
        (upload / "data.yaml").write_text("train: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['hummingbird']\n")
    tot = {s: sum(1 for m in manifest if m["split"] == s and m["role"] != "vv") for s in ("train", "valid", "test")}
    vvt = {s: sum(1 for m in manifest if m["split"] == s and m["role"] == "vv") for s in ("train", "valid")}
    summary.append(f"TOTAL a subir: train {tot['train']} / valid {tot['valid']} / test {tot['test']}; "
                   f"vv ya etiquetados que entran: train {vvt['train']} / valid {vvt['valid']}")
    params = {k: v for k, v in vars(a).items() if k not in ("videos", "out")}
    summary.append(f"Parámetros: {params}")
    (a.out / "resumen.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
