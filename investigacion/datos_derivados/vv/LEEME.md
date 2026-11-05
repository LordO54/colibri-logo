# Conjunto de evaluación propio (US-01.3), generado el 2026-11-01

| video_id | archivo | fps | duración | frames muestreados (cada N) | ave visible |
|---|---|---|---|---|---|
| vv01_adobe1294608945 | AdobeStock_1294608945.mov | 50 | 15,4 s | 96 (cada 8) | 0–12 s; colibrí nuca blanca (*Florisuga mellivora*) |
| vv02_adobe512464022 | AdobeStock_512464022.mov | 29,97 | 16,4 s | 99 (cada 5) | ~3–13 s; colibrí garganta rosada (tipo Anna) |
| vv03_florisuga_file | Flying White-necked Jacobin…mov | 29,97 | 8,5 s | 52 (cada 5) | ~1–7 s; **el contenido es un colibrí rufo sobre fondo negro, no un Florisuga como dice el nombre del archivo** |

Los tres son 1920x1080, con cámara fija, un solo ave y una flor como referencia.

## Contenido
- `<video_id>_frames.zip`: `images/` (JPG a resolución completa), `prelabels/` (formato YOLO, clase 0 = hummingbird) y `classes.txt`.
- `prelabels/` **no es verdad de terreno**: salen de YOLO26s COCO, clase `bird`, conf ≥ 0,25, imgsz 960. Hay que corregirlas a mano (añadir las aves que faltan, ajustar cajas y borrar falsos positivos) antes de usarlas como prueba.
- `<video_id>_revision.jpg`: los primeros 48 frames muestreados de cada video, con las pre-etiquetas dibujadas.
- `manifest_frames.csv`: id de frame, índice de frame, marca temporal del decodificador (t_ms) y número de pre-etiquetas.
- `videos.csv`: metadatos de cada video.
- `deteccion_yolo26s_todos_los_frames.csv`: detecciones de todas las clases en los 1515 frames, conf ≥ 0,05.

## Uso previsto
- Las tres son escenas nuevas respecto al dataset de Roboflow, así que funcionan como **prueba de decisión** (CA-01.5.5). No se usan para entrenar ni para elegir hiperparámetros: la validación sigue saliendo de Roboflow, re-particionado por bloques.
- Los videos completos sirven también para el piloto de tracking (sección 8 del notebook): ByteTrack frente a OC-SORT.

## Etiquetas corregidas (2026-11-01, proyecto Roboflow `yilber-lopez/humming`)
- `prueba_etiquetas_corregidas.zip`: 247 etiquetas YOLO congeladas, con 174 cajas y 73 frames vacíos. Se corrigieron 17 frames: 11 aves añadidas, 2 cajas ajustadas y 3 duplicados borrados.
- `ignorar_frames.txt`: 4 frames de vv02 (f95, f400, f405, f410) donde solo asoma el pico o la punta del ala por el borde. Quedan sin caja y se excluyen de la evaluación.
- `linea_base_coco_bird.csv`: línea base YOLO26 COCO `bird` sin ajustar, a IoU 0,5. Resultados: yolo26n@640 AP50 0,70 (R 0,52); yolo26s@960 AP50 0,94 (P 0,95, R 0,91). **Sesgo:** las pre-etiquetas salieron de yolo26s@960 y el 93 % se aceptó sin cambios, así que su cifra está inflada. Para comparar, usar yolo26n o un modelo distinto.
