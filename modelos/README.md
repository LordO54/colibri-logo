# Modelos

Solo el detector **vigente** está en git (`yolo26n_ft_v2/best.pt`, 5 MB). Los demás pesos se quedan en
`D:\video_eval\modelos\` (ruta `modelos` de `config.yaml`); aquí queda su ficha.

| Modelo | Sprint | Entrenado con | Prueba | Estado | Pesos |
|---|---|---|---|---|---|
| `yolo26n.pt` COCO `bird` | S2 | COCO | vv: AP50 0,70, R 0,52 | línea base | Ultralytics |
| `yolo26n_ds2` | S2 | ds2 | vv: AP50 0,790 | ablación | zona de trabajo |
| `yolo26n_ds2_ds1` | S2 | ds2 + ds1 re-partido, 40 épocas | vv: AP50 0,795, P 0,843, R 0,862; `test_` mAP50 0,706 | D-05, sustituido por D-08 | `modelos/yolo26n_ds2_ds1/best.pt` |
| **`yolo26n_ft_v2`** | S4 | ajuste fino de `yolo26n_ds2_ds1` con ds2+ds1+humming ×3 | `test_` mAP50 0,877, R 0,821; ds2 −0,5 pts | **vigente (D-08)** | **en el repo** |
| `yolo26s_v2` | S6 | mismas listas que `ft_v2`, tamaño s | no pasa el gate (R `test_` 0,68 frente a 0,73) | descartado (D-11) | zona de trabajo |
| `yolo26s_ft_v2` | S6 | ajuste fino de s solo en humming | Gu30 mAP50 0,60 frente a 0,97; IDF1 visita 0,46 frente a 0,67 | descartado (D-11) | zona de trabajo |

Licencia: los pesos derivan de Ultralytics YOLO26 (AGPL-3.0). ds2 es CC BY 4.0 y exige atribución (ver README).
