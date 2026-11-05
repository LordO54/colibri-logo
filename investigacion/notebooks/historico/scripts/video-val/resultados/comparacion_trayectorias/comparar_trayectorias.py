"""Lámina de trayectorias yolo26n_ft_v2 frente a yolo26s_ft_v2 (ByteTrack combo + unión prudente 1 s, 2,5x).

Solo para visualizar: no cambia el veredicto. Mismo fondo y escala en cada fila; cada segmento en un color, puentes en
discontinua y centros de las cajas GT v2 como puntos blancos.
"""
import json, shutil, sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

V2 = Path(r"D:\video_eval\reentreno\v2")
E2 = V2 / "eval_gt2"
RES = Path(r"C:\Users\ACER\Documents\logo\video-val\resultados\comparacion_trayectorias")
sys.path.insert(0, str(E2))
import union_gt2 as G  # noqa: E402

U = G.U
PRUDENTE = "prudente_1s_2.5"
MODELS = {"yolo26n_ft_v2": E2 / "tracking" / "b_nuevo_bt_combo",
          "yolo26s_ft_v2": V2 / "yolo26s_ft" / "comparacion" / "tracks_yolo26s_ft"}
SHORT = {"doj1xa6ksqm": "DoJ1", "gu30nofexla": "Gu30", "i5befvb5mxg": "i5bE", "j4pkgwgjaus": "J4PK"}


def load_tracks(d, vid):
    f = d / f"{vid}__detallada.csv"
    tr = pd.read_csv(f); tr.attrs.update(json.loads(f.with_suffix(".meta.json").read_text(encoding="utf-8")))
    return tr


def panel(ax, bg, traj, g, title):
    ax.imshow(bg, alpha=0.6)
    cmap = plt.get_cmap("tab10")
    for sid, s in traj.groupby("segment_id"):
        c = cmap(int(sid) % 10)
        s = s.sort_values("frame")
        real = ~s.interpolado.astype(bool)
        for _, part in s.groupby((real != real.shift()).cumsum()):
            ax.plot(part.cx_px, part.cy_px, color=c, lw=1.8, ls="-" if real.loc[part.index[0]] else "--")
        ax.annotate(f"s{sid}", (s.cx_px.iloc[0], s.cy_px.iloc[0]), color=c, fontsize=9, weight="bold")
    gx = [(b[0][0] + b[0][2]) / 2 for b in g.boxes if b]; gy = [(b[0][1] + b[0][3]) / 2 for b in g.boxes if b]
    ax.scatter(gx, gy, s=16, c="white", edgecolors="black", linewidths=0.5, zorder=5)
    ax.set_title(title, fontsize=11); ax.axis("off")


def main():
    RES.mkdir(parents=True, exist_ok=True)
    vids = {k: v for k, v in U.list_videos().items() if v["rol"] == "test"}
    GT = G.load_test_gt(G.MANIFEST["GT_v2"], vids)
    fin = E2 / "union" / "prudente_yolo26s_ft"; fin.mkdir(parents=True, exist_ok=True)
    (fin / "LEEME.txt").write_text("Solo para visualizar la comparación: yolo26s_ft_v2 no se adoptó (ver yolo26s_ft\\comparacion\\veredicto.json).\n", encoding="utf-8")
    fig, axes = plt.subplots(len(vids), 2, figsize=(24, 3.45 * 2 * len(vids) * 0.98))
    for i, (vid, v) in enumerate(vids.items()):
        name = next(s for k, s in SHORT.items() if k in vid)
        g = GT[GT.video_id == vid]; cuts = U.scene_cuts(v["path"])
        trajs = {tag: G.CONFIGS[PRUDENTE](load_tracks(d, vid), cuts) for tag, d in MODELS.items()}
        cap = cv2.VideoCapture(str(v["path"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(g[g.n_gt > 0].frame.median())); _, f = cap.read(); cap.release()
        bg = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        fig1, ax1 = plt.subplots(1, 2, figsize=(20, 5.9))
        for j, (tag, traj) in enumerate(trajs.items()):
            m = U.visit_metrics(traj, g)
            title = f"{name} | {tag} | seg {m['segmentos']} | cobertura {m['cobertura']:.2f} | IDF1 visita {m['IDF1_visita']:.2f}"
            panel(axes[i, j], bg, traj, g, title); panel(ax1[j], bg, traj, g, title)
            print(title, "| filas", len(traj))
            if tag == "yolo26s_ft_v2":
                t = traj.copy(); t["es_puente"] = t.interpolado.astype(bool)
                t.to_csv(fin / f"trayectoria_{vid}.csv", index=False)
                _, n_long = G.bridges(traj, v["W"], v["H"])
                U.draw(v["path"], t, g, f"{vid[30:]} | yolo26s_ft prudente (solo visual) | seg {m['segmentos']} | "
                       f"largos {n_long} | IDF1v {m['IDF1_visita']:.2f}", fin / f"trayectoria_{vid}.png")
        fig1.tight_layout(); fig1.savefig(RES / f"comparacion_{name}.jpg", dpi=120); plt.close(fig1)
    fig.tight_layout(); fig.savefig(RES / "comparacion_trayectorias_n_vs_s.jpg", dpi=100, pil_kwargs={"quality": 90}); plt.close(fig)
    shutil.copytree(fin, RES / "prudente_yolo26s_ft", dirs_exist_ok=True)
    shutil.copy2(__file__, RES / Path(__file__).name)


if __name__ == "__main__":
    main()
