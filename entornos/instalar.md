# Instalar los entornos

**Para la herramienta (`herramienta_trayectorias`) basta un entorno:** `yolo`, secciones 1 a 3 (D-21). El entorno `vda`
(sección 4) solo hace falta para reproducir el notebook 05 con su kernel `colibri-vda`; los demás notebooks usan `colibri-yolo`.

Las rutas son las de la laptop de referencia (Windows, `D:\video_eval`). En otro equipo cambia `zona_trabajo` en
`config.yaml` y crea la misma estructura.

## 1. Zona de trabajo

```
D:\video_eval\
├─ colibri-logo\      este repositorio (git clone)
├─ datos\             mis_videos\ (tus videos, herramienta); el resto lo crea el notebook 01
├─ runs\              salidas de los notebooks
└─ modelos\           pesos fuera de git y repos externos
   ├─ yolo26n_ds2_ds1\best.pt         detector anterior (D-05), base del ajuste fino
   ├─ depth\                          yolo26*-depth.pt (Ultralytics los descarga aquí)
   └─ Video-Depth-Anything\           repo externo + checkpoints\video_depth_anything_vits.pth
```

## 2. Entorno `yolo`

```powershell
cd D:\video_eval
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r colibri-logo\entornos\requirements-yolo.txt
python -m ipykernel install --user --name colibri-yolo --display-name "Python (colibri yolo)"
```

## 3. Video Depth Anything (para la profundidad)

Sin esto, la herramienta llega hasta los 7 puntos 2D; la 3D la necesita. Las rutas se configuran en `config.yaml` → `vda`.

```powershell
cd D:\video_eval\modelos
git clone https://github.com/DepthAnything/Video-Depth-Anything
cd Video-Depth-Anything
git checkout 4f5ae23172ba60fd7bc11ef671cca678842c7072   # commit usado en S8 (2025-10-07)
mkdir checkpoints
curl -L -o checkpoints\video_depth_anything_vits.pth https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
```

## 4. Entorno `vda` (solo para el notebook 05 con su kernel)

```powershell
cd D:\video_eval
py -3.11 -m venv .venv_vda
.venv_vda\Scripts\activate
pip install -r colibri-logo\entornos\requirements-vda.txt   # torch 2.6.0+cu124, easydict, einops...
python -m ipykernel install --user --name colibri-vda --display-name "Python (colibri vda)"
```

No instales el `requirements.txt` del repo de VDA: fija torch 2.1.1, y los mapas de S8 se calcularon con torch 2.6.0.

Solo el modelo **Small** (Apache-2.0). Base y Large son CC-BY-NC (D-17).

**En la laptop de referencia** `.venv_vda` no tiene un torch propio: solo instala `easydict` y `einops`, y reutiliza
los paquetes de `.venv` mediante `.venv_vda\Lib\site-packages\_venv_principal.pth`. Los mapas de S8 se calcularon así,
con torch 2.6.0+cu124 (no con el torch 2.1.1 que fija el `requirements.txt` de VDA) y sin xFormers.
`requirements-vda.txt` es el `pip freeze` de ese entorno, así que sirve para crear uno independiente con las mismas versiones.

## 5. Git

```powershell
cd D:\video_eval\colibri-logo
nbstripout --install          # limpia las salidas de los notebooks al hacer commit
copy .env.example .env        # y pon tu ROBOFLOW_API_KEY
```

## 6. Orden de ejecución

**Herramienta:** solo `herramienta_trayectorias`.

**Línea de investigación:** `01` → `02` → `03` → `04` → `05` (kernel vda) → `06`. En modo `artefactos` (por defecto) 03, 04 y 06 funcionan sin
haber ejecutado 01 ni 02, siempre que estén los videos: usan las trayectorias congeladas del repo.
