"""Entrena un run adicional con los mismos datos y ajustes que yolo26n_ds2_ds1, cambiando solo modelo y épocas.

Uso (Windows, desde cualquier carpeta):
    D:\\video_eval\\.venv\\Scripts\\python entrenar_run.py --model yolo26s.pt --epochs 25
Reanudable: si existe weights/last.pt del mismo nombre y no hay TRAINING_DONE, continúa desde ahí.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

WORK = Path(r"D:\video_eval\colibri_work")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"{Path(a.model).stem}_ds2_ds1_e{a.epochs}"
    run = WORK / "runs" / name
    last = run / "weights" / "last.pt"
    if (run / "TRAINING_DONE").exists():
        print("Ya entrenado:", run); return
    if last.exists():
        YOLO(str(last)).train(resume=True)
    else:
        # Mismos ajustes que el run de referencia (runs/yolo26n_ds2_ds1/args.yaml); RAM limitada: cache=disk, 3 workers
        YOLO(a.model).train(data=str(WORK / "data" / "final" / "data.yaml"), epochs=a.epochs, imgsz=640, batch=0.7,
                            workers=3, cache="disk", patience=15, seed=42, close_mosaic=min(10, a.epochs // 3),
                            project=str(WORK / "runs"), name=name, exist_ok=True)
    (run / "TRAINING_DONE").write_text("ok")
    print("Listo:", run / "weights" / "best.pt")

if __name__ == "__main__":      # imprescindible en Windows: los workers del DataLoader re-importan este módulo
    main()
