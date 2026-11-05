"""Compara los backends en i5bE: z_norm ± std por punto, control de ruido, filtro de estabilidad y lámina en vista superior.

No usa las etiquetas de Yilber de 5, 6 y 7 como criterio. El juez es evaluar_ciega.py. Referencia de escala: en cada panel, z_norm
sin reescalar y la misma unidad en x y z.
"""
from pathlib import Path
import sys, shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V2 = Path(r"D:\video_eval\reentreno\v2")
SP = V2 / "profundidad" / "siete_puntos"
OUT = SP / "salidas" / "comparacion_backends"; OUT.mkdir(parents=True, exist_ok=True)
RES = Path(r"C:\Users\ACER\Documents\logo\video-val\resultados\profundidad\siete_puntos")
sys.path.insert(0, str(V2 / "trayectorias_7"))
from trayectorias_7pts import catmull_rom

TAGS = ["yolo26_yolo26s-depth", "yolo26_yolo26x-depth", "yolo26_crop_yolo26s-depth", "vda_vda-small"]
SETS = {"b_salientes": "salientes original", "delante": "con punto forzado"}
AR = 1080 / 1920
FLOR = (0.54, 0.38, 0.69, 0.58); cx, r = (FLOR[0] + FLOR[2]) / 2, (FLOR[2] - FLOR[0]) / 2


def main():
    P = {t: {s: pd.read_csv(SP / "salidas" / t / f"puntos_i5bE_{s}.csv") for s in SETS} for t in TAGS}
    long = pd.concat([P[t][s].assign(backend=t, puntos=s) for t in TAGS for s in SETS])
    long.to_csv(OUT / "puntos_por_backend.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    for s in SETS:
        T = pd.DataFrame({t: [f"{a:+.3f} ± {b:.3f}" for a, b in zip(P[t][s].z_norm, P[t][s].z_norm_std)] for t in TAGS}, index=[f"{p} f{f}" for p, f in zip(P[TAGS[0]][s].punto, P[TAGS[0]][s].frame)])
        print(s); print(T.to_string()); T.to_csv(OUT / f"tabla_{s}.csv")
    C = pd.concat([pd.read_csv(SP / "salidas" / t / "control_ruido.csv") for t in TAGS]); C.to_csv(OUT / "control_ruido.csv", index=False)
    F = pd.concat([pd.read_csv(SP / "salidas" / t / "filtro.csv") for t in TAGS]); F.to_csv(OUT / "filtro.csv", index=False)
    print(C[["backend", "puntos", "dif_entre_puntos", "rango_z", "ruido_vecinos", "ruido_tta", "forma_supera_ruido", "min_den"]].round(3).to_string(index=False))
    print(F[["backend", "frames", "variacion_fondo_pct", "autocorr_z_05s", "autocorr_znorm_05s", "jitter_rel_z", "jitter_rel_znorm", "pasa_filtro"]].round(3).to_string(index=False))
    fig, axes = plt.subplots(len(SETS), len(TAGS), figsize=(5.8 * len(TAGS), 5.8 * len(SETS)), squeeze=False)
    zs = long.z_norm; lo, hi = min(-0.12, (zs - long.z_norm_std).min() - 0.03), max(0.12, (zs + long.z_norm_std).max() + 0.03)
    for i, s in enumerate(SETS):
        for j, t in enumerate(TAGS):
            D = P[t][s]; a = axes[i, j]; Q = np.c_[D.x_norm, D.y_norm * AR, D.z_norm]; c = catmull_rom(Q)
            a.add_patch(plt.Circle((cx, 0), r, fc="gold", ec="goldenrod", alpha=0.4)); a.axhline(0, color="goldenrod", ls=":", lw=1)
            a.plot(c[:, 0], c[:, 2], color="black", lw=2); a.errorbar(Q[:, 0], Q[:, 2], yerr=D.z_norm_std, fmt="o", color="crimson", mec="w", capsize=3, zorder=5)
            for k, q in enumerate(Q, 1):
                a.annotate(str(k), (q[0], q[2]), xytext=(6, 5), textcoords="offset points", fontsize=11, weight="bold")
            a.set_aspect("equal"); a.set_xlim(0.30, 0.85); a.set_ylim(lo, hi); a.grid(alpha=0.3); a.set_xlabel("x"); a.set_ylabel("z_norm (atrás arriba, delante abajo)")
            a.set_title(f"{t}\n{SETS[s]}", fontsize=11)
    fig.suptitle("i5bE, vista superior por modelo. z_norm sin reescalar; la flor es un cilindro (supuesto). VDA usa disparidad: comparar el signo, no la magnitud.", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / "comparacion_backends_i5bE.jpg", dpi=85); plt.close(fig)
    dest = RES / "comparacion_backends"; dest.mkdir(parents=True, exist_ok=True)
    for f in OUT.iterdir():
        shutil.copy2(f, dest / f.name)


if __name__ == "__main__":
    main()
