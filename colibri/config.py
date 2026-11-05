"""Lee config.yaml de la raíz del repositorio y resuelve las carpetas."""
import os
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
CARPETAS = {k: RAIZ / v for k, v in CFG["carpetas"].items()}
DETECTOR = RAIZ / CFG["detector"]


def cargar_env(path=RAIZ / ".env"):
    """Carga KEY=valor de .env en el entorno del proceso sin mostrar nada."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
