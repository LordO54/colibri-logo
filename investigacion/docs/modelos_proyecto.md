# Modelos

Registro de detectores. Los publicados están en git, en `modelos/<nombre>/best.pt`; los demás pesos se quedan en
la zona de trabajo (ruta `modelos` de `config.yaml`) y aquí queda su ficha. La herramienta y los notebooks usan
`config.yaml` → `vigente.detector`.

| Modelo | Sprint | Entrenado con | Prueba | Estado | Pesos |
|---|---|---|---|---|---|
| `yolo26n.pt` COCO `bird` | S2 | COCO | vv: AP50 0,70, R 0,52 | línea base | Ultralytics |
| `yolo26n_ds2` | S2 | ds2 | vv: AP50 0,790 | ablación | zona de trabajo |
| `yolo26n_ds2_ds1` | S2 | ds2 + ds1 re-partido, 40 épocas | vv: AP50 0,795, P 0,843, R 0,862; `test_` mAP50 0,706 | D-05, sustituido por D-08 | `modelos/yolo26n_ds2_ds1/best.pt` |
| **`yolo26n_ft_v2`** | S4 | ajuste fino de `yolo26n_ds2_ds1` con ds2+ds1+humming ×3 | `test_` mAP50 0,877, R 0,821; ds2 −0,5 pts | **vigente (D-08)** | **en el repo** |
| `yolo26s_v2` | S6 | mismas listas que `ft_v2`, tamaño s | no pasa el gate (R `test_` 0,68 frente a 0,73) | descartado (D-11) | zona de trabajo |
| `yolo26s_ft_v2` | S6 | ajuste fino de s solo en humming | Gu30 mAP50 0,60 frente a 0,97; IDF1 visita 0,46 frente a 0,67 | descartado (D-11) | zona de trabajo |

## Publicar un detector mejor

1. Fija el criterio **antes** de entrenar y pasa el gate del notebook 02 en `test_` (como D-08). Los `test_` nunca entran en
   entrenamiento ni en validación (D-07).
2. Copia los pesos a `modelos/<nombre>/best.pt`, con `args.yaml`, `results.csv` y una `ficha.md`, y añade su fila a esta tabla.
3. Cambia `vigente.detector` en `config.yaml` y actualiza `resultados/VIGENTE.md` y `docs/decisiones.md`.
4. Quien use la herramienta lo recibe con `git pull`. Las cifras de referencia de 03 en adelante son del detector con el que se
   midieron: al cambiarlo, esos notebooks hay que correrlos en modo `recalcular` y guardar sus resultados como un sprint nuevo.

Cualquiera puede proponer uno por PR; lo aceptan los mantenedores si pasa el gate.

Licencia: los pesos derivan de Ultralytics YOLO26 (AGPL-3.0). ds2 es CC BY 4.0 y exige atribución (ver README).
