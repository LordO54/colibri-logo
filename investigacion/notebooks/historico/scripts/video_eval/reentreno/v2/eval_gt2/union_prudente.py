"""Trayectorias test_ con la unión prudente (1 s, 2,5x), mismo formato que union\ganadora\."""
import sys
from pathlib import Path

E2 = Path(r"D:\video_eval\reentreno\v2\eval_gt2")
sys.path.insert(0, str(E2))
import union_gt2 as G  # noqa: E402

NAME = "prudente_1s_2.5"


def main():
    out = G.OUT / "prudente"; out.mkdir(parents=True, exist_ok=True)
    vids = {k: v for k, v in G.U.list_videos().items() if v["rol"] == "test"}
    GT = G.load_test_gt(G.MANIFEST["GT_v2"], vids)
    for vid, v in vids.items():
        tr = G.U.get_tracks(vid, v); cuts = G.U.scene_cuts(v["path"]); g = GT[GT.video_id == vid]
        traj = G.CONFIGS[NAME](tr, cuts).copy()
        traj["es_puente"] = traj.interpolado.astype(bool)
        traj.to_csv(out / f"trayectoria_{vid}.csv", index=False)
        m = G.U.visit_metrics(traj, g); _, n_long = G.bridges(traj, v["W"], v["H"])
        G.U.draw(v["path"], traj, g, f"{vid[30:]} | {NAME} | seg {m['segmentos']} | visitas {m['visitas']} | puentes falsos "
                 f"{m['puentes_falsos']} | largos {n_long} | IDF1v {m['IDF1_visita']:.2f}", out / f"trayectoria_{vid}.png")
        print(vid[30:], "segmentos", m["segmentos"], "puentes largos", n_long, "IDF1v", round(m["IDF1_visita"], 3))


if __name__ == "__main__":
    main()
