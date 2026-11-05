"""Detección viejo frente a nuevo con la GT v2: mismo evaluar.py, solo cambia la lista de test_video."""
import sys
from pathlib import Path

E2 = Path(r"D:\video_eval\reentreno\v2\eval_gt2")
sys.path.insert(0, str(E2.parent))
import evaluar  # noqa: E402

evaluar.SETS["test_video"] = E2 / "data_test_video.yaml"
evaluar.LISTS["test_video"] = E2 / "test_video.txt"

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--weights",
                r"viejo=C:\Users\ACER\Documents\logo\video-val\resultados\yolo26n_ds2_ds1\best.pt",
                r"nuevo=C:\Users\ACER\Documents\logo\video-val\resultados\yolo26n_ft_v2\best.pt",
                "--out", str(E2 / "deteccion_gt2.csv")]
    evaluar.main()
