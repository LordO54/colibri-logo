"""Tracking con la GT v2: mismo tracking_test.py y mismas 5 configuraciones.

Las trayectorias detalladas no dependen de la GT: se copian de v2\tracking y se reutilizan (la regla del notebook es
no sobrescribirlas); solo se recalculan las métricas y las salidas, en eval_gt2\tracking.
"""
import sys
from pathlib import Path

E2 = Path(r"D:\video_eval\reentreno\v2\eval_gt2")
sys.path.insert(0, str(E2.parent / "tracking"))
import tracking_test as T  # noqa: E402

T.V2 = E2                # load_gt lee E2\split_manifest.csv (rutas a E2\humming)
T.OUT = E2 / "tracking"

if __name__ == "__main__":
    T.main()
