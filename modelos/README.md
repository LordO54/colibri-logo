# Detectores

`config.yaml` → `detector` indica cuál se usa. Cada detector publicado está en `modelos/<nombre>/`, con `best.pt`,
`args.yaml` (parámetros de entrenamiento) y `results.csv` (curvas).

| Detector | Estado | Base | Datos de entrenamiento | Resultado en videos de prueba no vistos |
|---|---|---|---|---|
| `yolo26n_ft_v2` | **actual** | YOLO26n | Fotos y videos públicos de colibríes más cuadros etiquetados a mano | mAP50 0,88 y recall 0,82, en 4 videos y 216 cuadros etiquetados |

**Limitaciones conocidas:** pierde recall con aves pequeñas o de espaldas. La prueba es pequeña: orienta, pero no garantiza
el resultado en tus videos.

## Publicar un detector mejor

1. Fija el criterio de aceptación **antes** de entrenar y usa videos de prueba que el modelo no haya visto en entrenamiento
   ni en validación.
2. Añade `modelos/<nombre>/` con `best.pt`, `args.yaml` y `results.csv`, y su fila en esta tabla.
3. Cambia `detector` en `config.yaml`.

Quien use la herramienta lo recibe con `git pull`. Cualquiera puede proponer uno por pull request.

## Licencias

Los pesos derivan de Ultralytics YOLO (AGPL-3.0). Parte de los datos de entrenamiento es de
[hummingbirds-detect-classify](https://universe.roboflow.com/modelodevisinartificialcolibres/hummingbirds-detect-classify-gcm5s) (CC BY 4.0).
