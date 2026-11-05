"""Paso 3: yolo26n_ft_v2 frente a yolo26s_v2 con la GT v2 de test_ (216 frames, 177 cajas), sin reajustar nada.

- Detección a imgsz 640 y 960 (solo inferencia), global y por video, y en ds2 test (conf 0.001, iou 0.6; P/R a 0.25).
- Tracking ByteTrack combo + unión prudente (1 s, 2,5x) a imgsz 640: R, P, MOTA, IDF1, IDs, IDF1 de visita,
  segmentos por visita, puentes largos, cobertura y fps de proceso.
Criterio fijado antes de ver resultados: yolo26s se adopta si, a imgsz 640 en test_, el recall de detección sube
>= 3 puntos y la cobertura de trayectoria mejora, sin bajar la precisión más de 2 puntos ni el mAP50 de ds2 test
más de 2 puntos.
"""
import json, re, shutil, sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

V2 = Path(r"D:\video_eval\reentreno\v2")
E2 = V2 / "eval_gt2"
HERE = V2 / "yolo26s"
OUT = HERE / "comparacion"
RES = Path(r"C:\Users\ACER\Documents\logo\video-val\resultados\yolo26s_v2")
MODELS = {"yolo26n_ft_v2": Path(r"C:\Users\ACER\Documents\logo\video-val\resultados\yolo26n_ft_v2\best.pt"),
          "yolo26s_v2": V2 / "runs" / "yolo26s_v2" / "weights" / "best.pt"}
sys.path.insert(0, str(V2)); sys.path.insert(0, str(V2 / "tracking")); sys.path.insert(0, str(V2 / "union")); sys.path.insert(0, str(E2))
import evaluar as EV  # noqa: E402
import tracking_test as T  # noqa: E402
import union as U  # noqa: E402
import union_gt2 as G  # noqa: E402

T.V2 = E2  # GT v2


def detection(tag, w, imgsz):
    model = YOLO(str(w))
    rows = []
    sets = {"test_video": (E2 / "data_test_video.yaml", (E2 / "test_video.txt").read_text().split()),
            "ds2_test": (E2 / "data_ds2_test.yaml",
                         sorted(str(p) for p in Path(r"D:\video_eval\colibri_work\data\final_ds2\test\images").glob("*.jpg")))}
    for name, (y, imgs) in sets.items():
        if name == "ds2_test" and imgsz != 640:
            continue
        m = model.val(data=str(y), split="test", imgsz=imgsz, conf=0.001, iou=0.6, batch=8, plots=False, verbose=False,
                      project=str(OUT / "_val"), name=f"{tag}_{name}_{imgsz}", exist_ok=True)
        st = EV.pr_at(model, imgs, conf=0.25, imgsz=imgsz)
        tp, fp, fn = st["GLOBAL"]
        rows.append(dict(modelo=tag, imgsz=imgsz, conjunto=name, video="GLOBAL", P=m.box.mp, R=m.box.mr, mAP50=m.box.map50,
                         mAP50_95=m.box.map, P_c025=tp / max(1, tp + fp), R_c025=tp / max(1, tp + fn), n_gt=tp + fn))
        if name == "test_video":
            groups = defaultdict(list)
            for p in imgs:
                groups[re.sub(r"_f\d{5}$", "", Path(p).stem)].append(p)
            for vid, ps in groups.items():
                lst = OUT / "_val" / f"{vid[:40]}.txt"; lst.write_text("\n".join(ps) + "\n")
                yy = OUT / "_val" / f"{vid[:40]}.yaml"; yy.write_text(f"train: {lst}\nval: {lst}\nnames:\n  0: hummingbird\n")
                mv = model.val(data=str(yy), split="val", imgsz=imgsz, conf=0.001, iou=0.6, batch=8, plots=False, verbose=False,
                               project=str(OUT / "_val"), name="pv", exist_ok=True)
                tp, fp, fn = st[vid]
                rows.append(dict(modelo=tag, imgsz=imgsz, conjunto=name, video=vid, P=mv.box.mp, R=mv.box.mr, mAP50=mv.box.map50,
                                 mAP50_95=mv.box.map, P_c025=tp / max(1, tp + fp), R_c025=tp / max(1, tp + fn), n_gt=tp + fn))
    return rows


def tracking(tag, w):
    vids = {k: v for k, v in U.list_videos().items() if v["rol"] == "test"}
    GTt = T.load_gt({k: {"wh": (v["W"], v["H"])} for k, v in vids.items()})
    GTu = G.load_test_gt(G.MANIFEST["GT_v2"], vids)
    cfg = V2 / "tracking" / "cfg" / "tracker_b_nuevo_bt_combo.yaml"
    d = (E2 / "tracking" / "b_nuevo_bt_combo") if tag == "yolo26n_ft_v2" else (OUT / "tracks_yolo26s")
    d.mkdir(parents=True, exist_ok=True)
    rows, trajs = [], {}
    for vid, v in vids.items():
        f = d / f"{vid}__detallada.csv"
        if f.exists():
            tr = pd.read_csv(f); tr.attrs.update(json.loads(f.with_suffix(".meta.json").read_text(encoding="utf-8")))
        else:
            tr = T.track_video(w, v["path"], "bytetrack", cfg)
            tr.to_csv(f, index=False); f.with_suffix(".meta.json").write_text(json.dumps(tr.attrs, indent=2), encoding="utf-8")
        m, _ = T.evaluate_tracks(GTt, vid, tr)
        traj = G.CONFIGS["prudente_1s_2.5"](tr, U.scene_cuts(v["path"]))
        vm = U.visit_metrics(traj, GTu[GTu.video_id == vid])
        _, n_long = G.bridges(traj, v["W"], v["H"])
        trajs[vid] = traj
        rows.append(dict(modelo=tag, video_id=vid, **{k: m[k] for k in ("n_gt", "TP", "FN", "FP", "IDSW", "ids_por_ave",
                    "ids_totales", "recall", "precision", "MOTA", "IDF1", "idtp", "fps_proceso")},
                         **{k: vm[k] for k in ("visitas", "cajas", "cubiertas", "pred_en_gt", "seg_por_visita",
                            "visitas_sin_segmento", "puentes_falsos", "cobertura", "IDF1_visita", "segmentos")},
                         idtp_v=vm["idtp"], puentes_largos=n_long))
    D = pd.DataFrame(rows)
    n_gt, tp, fp = D.n_gt.sum(), D.TP.sum(), D.FP.sum()
    a = U.aggregate([{**r, "idtp": r["idtp_v"]} for r in rows])
    glob = dict(modelo=tag, video_id="GLOBAL", n_gt=n_gt, TP=tp, FN=n_gt - tp, FP=fp, IDSW=D.IDSW.sum(),
                ids_por_ave=D.ids_por_ave.sum(), ids_totales=D.ids_totales.sum(), recall=tp / n_gt, precision=tp / max(tp + fp, 1),
                MOTA=1 - (n_gt - tp + fp + D.IDSW.sum()) / n_gt, IDF1=2 * D.idtp.sum() / (n_gt + tp + fp),
                fps_proceso=D.fps_proceso.mean(), seg_por_visita=a["seg_por_visita"], cobertura=a["cobertura"],
                IDF1_visita=a["IDF1_visita"], puentes_falsos=a["puentes_falsos"], segmentos=a["segmentos"],
                puentes_largos=D.puentes_largos.sum())
    return pd.concat([D, pd.DataFrame([glob])], ignore_index=True), trajs, vids, GTu


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    det = pd.DataFrame([r for tag, w in MODELS.items() for s in (640, 960) for r in detection(tag, w, s)])
    det.to_csv(OUT / "deteccion_n_vs_s.csv", index=False)
    trk, TRAJ = [], {}
    for tag, w in MODELS.items():
        D, trajs, vids, GTu = tracking(tag, w)
        trk.append(D); TRAJ[tag] = trajs
    trk = pd.concat(trk, ignore_index=True)
    trk.to_csv(OUT / "tracking_union_n_vs_s.csv", index=False)

    g = lambda m, c="test_video", s=640: det[(det.modelo == m) & (det.conjunto == c) & (det.imgsz == s) & (det.video == "GLOBAL")].iloc[0]
    n, s_ = g("yolo26n_ft_v2"), g("yolo26s_v2")
    nd, sd = g("yolo26n_ft_v2", "ds2_test"), g("yolo26s_v2", "ds2_test")
    tn = trk[(trk.modelo == "yolo26n_ft_v2") & (trk.video_id == "GLOBAL")].iloc[0]
    ts = trk[(trk.modelo == "yolo26s_v2") & (trk.video_id == "GLOBAL")].iloc[0]
    checks = {"recall_test_+3pts": bool(s_.R - n.R >= 0.03), "cobertura_trayectoria_mejora": bool(ts.cobertura > tn.cobertura),
              "precision_test_no_baja_>2pts": bool(n.P - s_.P <= 0.02), "ds2_mAP50_no_baja_>2pts": bool(nd.mAP50 - sd.mAP50 <= 0.02)}
    adopt = all(checks.values())
    verdict = {"adoptar_yolo26s": adopt, "comprobaciones": checks,
               "valores": {"R_test_n": n.R, "R_test_s": s_.R, "P_test_n": n.P, "P_test_s": s_.P, "cobertura_n": tn.cobertura,
                           "cobertura_s": ts.cobertura, "ds2_mAP50_n": nd.mAP50, "ds2_mAP50_s": sd.mAP50}}
    (OUT / "veredicto.json").write_text(json.dumps(verdict, indent=2, default=float), encoding="utf-8")

    if adopt:
        fin = E2 / "union" / "prudente_yolo26s"; fin.mkdir(parents=True, exist_ok=True)
        for vid, traj in TRAJ["yolo26s_v2"].items():
            t = traj.copy(); t["es_puente"] = t.interpolado.astype(bool)
            t.to_csv(fin / f"trayectoria_{vid}.csv", index=False)
            r = trk[(trk.modelo == "yolo26s_v2") & (trk.video_id == vid)].iloc[0]
            U.draw(vids[vid]["path"], t, GTu[GTu.video_id == vid], f"{vid[30:]} | yolo26s prudente | seg {r.segmentos} | "
                   f"largos {r.puentes_largos} | IDF1v {r.IDF1_visita:.2f}", fin / f"trayectoria_{vid}.png")

    RES.mkdir(parents=True, exist_ok=True)
    run = V2 / "runs" / "yolo26s_v2"
    for f in (run / "weights" / "best.pt", run / "results.csv", run / "results.png", run / "args.yaml",
              OUT / "deteccion_n_vs_s.csv", OUT / "tracking_union_n_vs_s.csv", OUT / "veredicto.json", Path(__file__)):
        if f.exists():
            shutil.copy2(f, RES / f.name)
    if adopt:
        shutil.copytree(E2 / "union" / "prudente_yolo26s", RES / "prudente_yolo26s", dirs_exist_ok=True)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print(det.assign(video=det.video.str.replace("test_ytdown_com_youtube_media_", "").str[:11]).to_string(index=False))
        print(trk.assign(video_id=trk.video_id.str.replace("test_ytdown_com_youtube_media_", "").str[:11])[
            ["modelo", "video_id", "recall", "precision", "MOTA", "IDF1", "ids_totales", "IDF1_visita", "seg_por_visita",
             "puentes_largos", "cobertura", "fps_proceso"]].to_string(index=False))
    print(json.dumps(verdict, indent=2, default=float))
    (OUT / "COMPARACION_DONE").write_text("ok")


if __name__ == "__main__":
    main()
