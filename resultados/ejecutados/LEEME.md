# Validación en la laptop (04-nov-2026)

Notebooks ejecutados en modo `artefactos` con `nbconvert`, cada uno con su kernel. Los HTML de esta carpeta son esas
ejecuciones. En `01_fuentes_y_datasets.html` se quitaron las 4 rejillas de imágenes de dataset (celdas 13 y 15).

| Notebook | Comparaciones | Resultado |
|---|---|---|
| 01 | 2 | ✓ `manifest_sha1` y `manifest_sha1_ds2` iguales a S1 |
| 02 | 6 | 3 ✓ y 3 ✗ (ver abajo) |
| 03 | 8 | ✓ diferencia 0 |
| 04 | 9 | ✓ diferencia 0 |
| 05 | 1 | ✓ (mapas ya calculados, no se recalculan) |
| 06 | 5 | ✓ diferencia 0 |

## Los ✗ del notebook 02

- `yolo26n_ds2` AP50 0,812 (S2: 0,790).
- Gate GT v1 y GT v2: diferencia máxima 0,1019 (`P_c025` de `yolo26n_ft_v2` en Gu30).

**Causa comprobada:** las imágenes de evaluación. Las referencias se midieron con los JPEG originales de S2 y S4, y el
notebook 01 regenera los frames desde los videos, con otra compresión JPEG. Con las imágenes originales y los mismos
pesos, el gate da diferencia 0 en `mAP50`, `P_c025` y `R_c025` (con 4 decimales) y la prueba vv da AP50 0,7952 y
0,7901, como en S2. Detalle en `verificacion_02_imagenes_originales.txt`; script en `gate_orig.py`. Las referencias
no se cambiaron.

**Qué implica:** una ejecución desde cero no reproduce estas cifras al detalle. El mAP50 global de `yolo26n_ft_v2` en
`test_` queda en 0,877 (GT v1) y 0,805 (GT v2), pero las métricas a conf 0,25 por video se mueven hasta 0,10. Con los
frames regenerados `yolo26n_ds2` (0,812) supera a `yolo26n_ds2_ds1` (0,795) en vv, al revés que en S2, así que esa
diferencia de AP50 cae dentro de lo que mueve la recompresión.

## Frames regenerados (notebook 01, sección 6.1)

576 de 594 frames dan dHash ≤ 2 frente a los originales. Los otros 18 (16 de A512, 1 de p8uU, 1 de kkIm) dan dHash de
3 a 5, pero su menor diferencia de píxeles cae en el índice correcto al compararlos con los frames vecinos (±3): no hay
desfase de índice. Script en `frames_check.py`.

## Descarga de Roboflow

El notebook 01 no descargó porque `raw/` ya estaba copiado. Una descarga aparte es idéntica byte a byte a esa copia
(ds1 1267 archivos, ds2 13535, 0 distintos).
