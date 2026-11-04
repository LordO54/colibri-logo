"""Trayectorias y 7 puntos de los videos estables de Adobe, con el mismo método que test_ (yolo26n_ft_v2 + ByteTrack combo
+ unión prudente 1 s, 2,5x; 7 puntos uniforme en el tiempo y salientes). AdobeStock_1600207728 tiene 2 aves: la unión
aplica la regla de una sola identidad (descarta fragmentos solapados), así que su trayectoria sigue sobre todo al ave principal."""
import json, shutil, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

V2 = Path(r"D:\video_eval\reentreno\v2")
HERE = V2 / "profundidad" / "adobe"
VIDEOS = Path(r"C:\Users\ACER\Documents\logo\video-val")
sys.path[:0] = [str(V2 / "union"), str(V2 / "tracking"), str(V2 / "trayectorias_7")]
import union as U  # noqa: E402
import trayectorias_7pts as P7  # noqa: E402

NAMES = {"vv01_adobe1294608945": "A129", "vv02_adobe512464022": "A512", "vv03_florisuga_file": "vv03", "adobestock_1600207728": "A160"}
TRAJ, TRACKS, PTS = HERE / "trayectorias", HERE / "tracks", HERE / "puntos7"


def main():
    for d in (TRAJ, TRACKS, PTS):
        d.mkdir(parents=True, exist_ok=True)
    vids = U.list_videos()
    vids["adobestock_1600207728"] = dict(path=VIDEOS / "AdobeStock_1600207728.mov", rol="train")
    paths = {}
    for vid, name in NAMES.items():
        v = vids[vid]; paths[vid] = str(v["path"])
        src = TRACKS / f"{vid}__detallada.csv"
        cached = U.OUT / "tracks" / f"{vid}__detallada.csv"
        if not src.exists() and cached.exists():
            shutil.copy2(cached, src); shutil.copy2(cached.with_suffix(".meta.json"), src.with_suffix(".meta.json"))
        if src.exists():
            tr = pd.read_csv(src); tr.attrs.update(json.loads(src.with_suffix(".meta.json").read_text(encoding="utf-8")))
        else:
            print("tracking", vid, flush=True)
            tr = U.T.track_video(U.WEIGHTS, v["path"], U.TRACKER, U.T.OUT / "cfg" / f"tracker_{U.CFG_NAME}.yaml")
            tr.to_csv(src, index=False); src.with_suffix(".meta.json").write_text(json.dumps(tr.attrs, indent=2), encoding="utf-8")
        t = U.union_tuned(tr, 1, 2.5, U.scene_cuts(v["path"])).copy()
        t["es_puente"] = t.interpolado.astype(bool)
        t.to_csv(TRAJ / f"trayectoria_{vid}.csv", index=False)
        cap = cv2.VideoCapture(str(v["path"])); fps, W, H = cap.get(5), int(cap.get(3)), int(cap.get(4)); cap.release()
        dense = np.vstack([P7.smooth(s[["cx_px", "cy_px"]].to_numpy(), fps) for _, s in t.groupby("segment_id", sort=False)])
        for m, fn in (("a_uniforme_tiempo", P7.uniform_time), ("b_salientes", P7.visvalingam)):
            idx, moved = P7.fix_bridges(fn(dense), t.es_puente.to_numpy())
            sel, Q = t.iloc[idx], dense[idx]
            pd.DataFrame(dict(punto=range(1, len(idx) + 1), frame=sel.frame.to_numpy(), t_s=sel.t_s.round(3).to_numpy(), segment_id=sel.segment_id.to_numpy(),
                              x_px=Q[:, 0].round(1), y_px=Q[:, 1].round(1), x_norm=(Q[:, 0] / W).round(4), y_norm=(Q[:, 1] / H).round(4),
                              es_puente_corregido=moved)).to_csv(PTS / f"{name}_{m}.csv", index=False)
        print(name, vid, f"{W}x{H} {fps:.2f} fps | IDs track {tr.track_id.nunique()} | filas {len(t)} | observadas {(~t.es_puente).sum()} | "
              f"segmentos {t.segment_id.nunique()} | t {t.t_s.min():.1f}-{t.t_s.max():.1f} s | cortes {len(U.scene_cuts(v['path']))}", flush=True)
    (HERE / "paths.json").write_text(json.dumps(paths, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
