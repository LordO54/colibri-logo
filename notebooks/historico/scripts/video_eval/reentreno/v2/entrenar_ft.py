"""Ajuste fino de yolo26n_ds2_ds1 con ds2+ds1+humming (x3). Reanudable desde last.pt."""
from pathlib import Path

from ultralytics import YOLO

V2 = Path(r"D:\video_eval\reentreno\v2")
BASE = r"C:\Users\ACER\Documents\logo\video-val\resultados\yolo26n_ds2_ds1\best.pt"
RUN = V2 / "runs" / "yolo26n_ft_v2"


def main():
    last = RUN / "weights" / "last.pt"
    if (RUN / "TRAINING_DONE").exists():
        print("Ya entrenado:", RUN); return
    if last.exists():
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(BASE).train(data=str(V2 / "data_ft.yaml"), epochs=20, patience=8, imgsz=640, warmup_epochs=1, lr0=0.002,
                         optimizer="SGD", cache="disk", workers=3, batch=0.70, project=str(V2 / "runs"),
                         name="yolo26n_ft_v2", exist_ok=False, seed=0)
    (RUN / "TRAINING_DONE").write_text("ok")
    print("Listo:", RUN / "weights" / "best.pt")


if __name__ == "__main__":  # imprescindible en Windows: los workers del DataLoader re-importan este módulo
    main()
