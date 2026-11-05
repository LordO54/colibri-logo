# Preparar la investigación

Primero instala la herramienta como dice el README de la raíz: un solo entorno y el kernel `colibri`.
Crea `investigacion/config.local.yaml` con `zona_trabajo: <ruta>` si tus datos pesados no están en `investigacion/trabajo/`.

## 1. Zona de trabajo

Por defecto, `investigacion/trabajo/` (fuera de git):

```
trabajo/
├─ datos/             los crea el notebook 01
├─ runs/              salidas de los notebooks
└─ modelos/           pesos fuera de git y repos externos
   ├─ yolo26n_ds2_ds1/best.pt         detector anterior (D-05), base del ajuste fino
   ├─ depth/                          yolo26*-depth.pt (Ultralytics los descarga aquí)
   └─ Video-Depth-Anything/           repo externo + checkpoints/video_depth_anything_vits.pth
```

## 2. Video Depth Anything (para la profundidad)

Lo necesitan 05 y 06. Las rutas se configuran en `investigacion/config.yaml` → `vda`.

```powershell
cd investigacion	rabajo\modelos
git clone https://github.com/DepthAnything/Video-Depth-Anything
cd Video-Depth-Anything
git checkout 4f5ae23172ba60fd7bc11ef671cca678842c7072   # commit usado en S8 (2025-10-07)
mkdir checkpoints
curl -L -o checkpoints\video_depth_anything_vits.pth https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
```

## 3. Git

```powershell
nbstripout --install          # limpia las salidas de los notebooks al hacer commit
copy .env.example .env        # en la raíz del repositorio, con tu ROBOFLOW_API_KEY
```

## 4. Orden de ejecución

`01` → `02` → `03` → `04` → `05` → `06`. En modo `artefactos` (por defecto) 03, 04 y 06 funcionan sin
haber ejecutado 01 ni 02, siempre que estén los videos: usan las trayectorias congeladas del repo.
