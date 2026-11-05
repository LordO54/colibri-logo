"""Ejecuta el notebook colibri_yolo26_tracking.ipynb como script, por etapas y de forma reanudable.

Uso (desde la carpeta que contiene el notebook y eval_frames/):
    set ROBOFLOW_API_KEY=...            (PowerShell: $env:ROBOFLOW_API_KEY="...")
    set LICENSE_DS1=MIT (declarada; videos de terceros)
    set LICENSE_DS2=CC BY 4.0 (con atribución)
    python run_pipeline.py --until prep     # descarga, auditoría, particiones y prueba de video
    python run_pipeline.py                  # todo: prep + línea base + entrenamiento + evaluación
Si se corta, vuelve a lanzar el mismo comando: el entrenamiento se reanuda desde last.pt.
Variables opcionales: EPOCHS (100), BATCH (16), WORKERS (4), RUN_ABLATION (1), SKIP_DOWNLOAD (0),
FRACTION (1.0; p. ej. 0.02 para una prueba de humo rápida).
"""
import argparse, builtins, json, os, sys, traceback
from pathlib import Path

STAGES = {
    "prep":  [3, 5, 7, 8, 10, 11, 13, 15, 17, 18, 20],
    "base":  [22, 23],
    "train": [25],
    "eval":  [27, 28, 33],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default="colibri_yolo26_tracking.ipynb")
    ap.add_argument("--until", choices=list(STAGES), default="eval")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    cells = json.load(open(a.notebook, encoding="utf-8"))["cells"]
    g = {"__name__": "colibri_pipeline", "display": lambda x: print(x.to_string() if hasattr(x, "to_string") else x)}
    if os.environ.get("SKIP_DOWNLOAD") == "1":      # reutiliza descargas previas sin tocar la red
        STAGES["prep"].remove(5)
        STAGES["prep"].remove(7)
    for stage, idxs in STAGES.items():
        print(f"\n==================== etapa {stage} ====================", flush=True)
        for i in idxs:
            src = "".join(cells[i]["source"])
            if stage == "prep" and i == 10 and "RAW_DIRS" not in g:
                g["RAW_DIRS"] = {k: g["DATA_RAW"] / k for k in g["DATASETS"]}
            print(f"--- celda {i}", flush=True)
            exec(compile(src, f"{a.notebook}#celda{i}", "exec"), g)
        if stage == a.until:
            break
    print("\nListo. Resultados en", g.get("REPORTS"), "y", g.get("RUNS"))

if __name__ == "__main__":      # imprescindible en Windows: los workers del DataLoader re-importan este módulo
    main()
