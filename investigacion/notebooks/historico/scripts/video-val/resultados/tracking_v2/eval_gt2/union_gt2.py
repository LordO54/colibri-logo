"""Unión en test_ con la GT v1 (antes) y la GT v2 (después): actual, elegida y prudente, una sola pasada.

Mismas funciones de union.py (union_reference, union_tuned, visit_metrics). Puente largo: tramo interpolado cuya
longitud en línea recta (entre la última posición real antes del hueco y la primera después) supera 1/4 del lado
corto del video. Criterio fijado antes de ver resultados: mínimos puentes falsos, luego mayor IDF1 de visita,
luego menos puentes largos.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

E2 = Path(r"D:\video_eval\reentreno\v2\eval_gt2")
V2 = E2.parent
sys.path.insert(0, str(V2 / "union"))
import union as U  # noqa: E402

OUT = E2 / "union"
CONFIGS = {
    "actual_30f_1.5": lambda tr, cuts: U.union_reference(tr),
    "elegida_0.5s_4": lambda tr, cuts: U.union_tuned(tr, 0.5, 4, cuts),
    "prudente_1s_2.5": lambda tr, cuts: U.union_tuned(tr, 1, 2.5, cuts),
}
MANIFEST = {"GT_v1": V2 / "split_manifest.csv", "GT_v2": E2 / "split_manifest.csv"}


def load_test_gt(manifest, vids):
    sm = pd.read_csv(manifest)
    sm = sm[(sm.split == "test") & (sm.origen == "nuevo")]
    rows = []
    for r in sm.itertuples():
        v = vids[r.video_id]
        lab = Path(r.ruta.replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        rows.append(dict(video_id=r.video_id, frame=int(r.frame), boxes=U.yolo_boxes(lab.read_text(), v["W"], v["H"])))
    GT = pd.DataFrame(rows).sort_values(["video_id", "frame"]).reset_index(drop=True)
    GT["n_gt"] = GT.boxes.str.len()
    return GT


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vids = {k: v for k, v in U.list_videos().items() if v["rol"] == "test"}
    rows, br_rows = [], []
    for gt_name, man in MANIFEST.items():
        GT = load_test_gt(man, vids)
        for vid, v in vids.items():
            tr = U.get_tracks(vid, v); cuts = U.scene_cuts(v["path"]); g = GT[GT.video_id == vid]
            for name, fn in CONFIGS.items():
                traj = fn(tr, cuts)
                m = U.visit_metrics(traj, g)
                br, n_long = bridges(traj, v["W"], v["H"])
                rows.append(dict(gt=gt_name, config=name, video_id=vid, **m, puentes=len(br), puentes_largos=n_long))
                if gt_name == "GT_v2":
                    br_rows += [dict(config=name, video_id=vid, **b, largo=b["long_px"] > min(v["W"], v["H"]) / 4) for b in br]
    R = pd.DataFrame(rows)
    agg = []
    for (gt, c), g in R.groupby(["gt", "config"], sort=False):
        a = U.aggregate(g.to_dict("records"))
        agg.append(dict(gt=gt, config=c, video_id="GLOBAL", **a, puentes=int(g.puentes.sum()), puentes_largos=int(g.puentes_largos.sum())))
    R = pd.concat([R, pd.DataFrame(agg)], ignore_index=True)
    R.to_csv(OUT / "union_test_gt1_gt2.csv", index=False)
    pd.DataFrame(br_rows).to_csv(OUT / "puentes_gt2.csv", index=False)

    G = R[(R["gt"] == "GT_v2") & (R.video_id == "GLOBAL")].sort_values(
        ["puentes_falsos", "IDF1_visita", "puentes_largos"], ascending=[True, False, True])
    chosen = G.config.iloc[0]
    print("GANADORA (GT v2):", chosen)

    fin = OUT / "ganadora"; fin.mkdir(exist_ok=True)
    GT = load_test_gt(MANIFEST["GT_v2"], vids)
    for vid, v in vids.items():
        tr = U.get_tracks(vid, v); cuts = U.scene_cuts(v["path"]); g = GT[GT.video_id == vid]
        traj = CONFIGS[chosen](tr, cuts).copy()
        traj["es_puente"] = traj.interpolado.astype(bool)
        traj.to_csv(fin / f"trayectoria_{vid}.csv", index=False)
        m = R[(R["gt"] == "GT_v2") & (R.config == chosen) & (R.video_id == vid)].iloc[0]
        U.draw(v["path"], traj, g, f"{vid[30:]} | {chosen} | seg {m.segmentos} | visitas {m.visitas} | puentes falsos "
               f"{m.puentes_falsos} | largos {m.puentes_largos} | IDF1v {m.IDF1_visita:.2f}", fin / f"trayectoria_{vid}.png")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        R["video_id"] = R.video_id.str.replace("test_ytdown_com_youtube_media_", "").str[:11]
        print(R[["gt", "config", "video_id", "visitas", "cajas", "puentes_falsos", "IDF1_visita", "seg_por_visita",
                 "cobertura", "visitas_sin_segmento", "segmentos", "puentes", "puentes_largos"]].to_string(index=False))


if __name__ == "__main__":
    main()
