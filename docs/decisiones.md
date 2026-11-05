# Registro de decisiones

Cada decisión indica la fecha, quién la tomó, la evidencia y las alternativas descartadas. Cuando una decisión deja sin efecto a otra, se marca como **revertida** y no se borra. Las rutas son las de la estructura nueva; `docs/mapa_rutas.csv` da su ubicación anterior.

| ID | Fecha | Sprint | Decisión | Quién | Evidencia | Alternativas descartadas |
|---|---|---|---|---|---|---|
| D-00 | ≤01-nov | S0 | El centro de la caja es la referencia corporal; primero 2D, la profundidad es opcional | Backlog v0.3 | `docs/historico/backlog_redisenio_colibri_v0_3.md` | Máscara, pose |
| D-01 | 01-nov | S1 | Usar datasets públicos de Roboflow en lugar de etiquetar desde cero | Yilber | Backlog v0.4 | Dataset propio |
| D-02 | 01-nov | S1 | Re-partir ds1 por bloques temporales (40 frames, guarda 15) → 442/83/82 | Auditoría | Ficha ds1: fuga ~100 % | Usar el split de Roboflow |
| D-03 | 01-nov | S1 | Entrenar ds2+ds1, con ablación solo de ds2 | Yilber | Ficha ds2 (6766 fotos, CC BY 4.0) | Solo ds1 |
| D-04 | 01-nov | S1 | vv01-vv03 como prueba de decisión, sin entrar en entrenamiento (**revertida por D-07**) | Yilber | `datasets/d1_eval_vv/` | — |
| D-05 | 01-nov | S2 | Adoptar `yolo26n_ds2_ds1` | Yilber | vv AP50 0,795, P 0,843, R 0,862 frente a COCO 0,70/0,76/0,52; ablación ds2 0,790 | yolo26n_ds2 |
| D-06 | 01-nov | S3 | OC-SORT combo como tracker recomendado (**revertida por D-09**) | Recomendación | `resultados/S3_tracking_piloto/barrido_trackers.csv`: IDF1 0,69 → 0,80 | ByteTrack por defecto |
| D-07 | 02-nov | S3→S4 | vv y `train_` pasan a entrenamiento; los `test_` son la nueva prueba y nunca entran en train ni en valid | Yilber | Fallo de vv01 (ave y flor en una sola caja) | Seguir con vv como prueba |
| D-08 | 02-nov | S4 | **`yolo26n_ft_v2` es el detector vigente** | Criterio fijado + Yilber | test_ mAP50 0,706 → 0,877, R 0,757 → 0,821; ds2 −0,5 pts | Modelo viejo |
| D-09 | 03-nov | S5 | **ByteTrack combo** (`buffer` 300, `match` 0,95, `new` 0,5) | Regla fijada | `resultados/S5_tracking_union_test/eleccion.json` | OC-SORT combo/base, ByteTrack base, viejo + OC-SORT |
| D-10 | 03-nov | S5 | **Unión prudente: 1 s y 2,5×** | Yilber | GT v2: IDF1 de visita 0,667, 0 puentes falsos, 3 puentes largos | Ganadora de la rejilla 0,5 s × 4 (rectas largas en J4PK); unión actual |
| D-11 | 03-nov | S6 | No adoptar yolo26s ni yolo26s_ft | Criterio fijado | `resultados/S6_spike_yolo26s/*/veredicto.json` | yolo26s_v2, yolo26s_ft_v2, imgsz 960 |
| D-12 | 04-nov | S7 | 7 puntos **en total** por video, sobre todos los segmentos; los huecos no se dibujan | Yilber | `resultados/S7_trayectorias_2d/` | 7 por segmento, solo el segmento más largo |
| D-13 | 04-nov | S8 | Descartar YOLO26-Depth cuadro a cuadro como coordenada 3D continua | Filtro fijado | 1 de 8 videos pasa (fondo < 5 %, autocorrelación > 0,5) | Incorporar a las geometrías |
| D-14 | 04-nov | S8 | Profundidad solo en los 7 puntos, con `z_norm = (z_ave − z_flor)/(z_fondo − z_flor)` dentro de cada mapa | Yilber (idea) | `lectura.md`: diferencia ~0,10 frente a ruido ~0,02 en i5bE | z absoluta en metros |
| D-15 | 04-nov | S8 | i5bE: forzar un punto en 12,7-13,7 s, elegido por visibilidad y no por z (variante `salientes_delante`) | Yilber | f384 se ve delante; dz −0,02 | f98 o f54 como punto 2 |
| D-16 | 04-nov | S8 | La escala de z para visualizar es una decisión de diseño, no una medida | Acordado | `escala_z/` | Escalar z como si fuera dato |
| D-17 | 04-nov | S8 | De Video Depth Anything, solo Small (Apache-2.0) | Licencia | LICENSE del repositorio | Base y Large (CC-BY-NC) |
| D-18 | 04-nov | S8 | **SP-02.7: otra prueba** con VDA Small en otro video de cámara fija, con ciegos repartidos; la 2D sigue siendo la base | Juez ciego | VDA 14/17 (p = 0,013) frente a YOLO 8-12/17 | Incorporar ya; descartar la profundidad |
| D-19 | 04-nov | S9 | **Un repositorio git (`colibri-logo`) con dos entornos** (`yolo` y `vda`, un kernel por notebook); el código suelto pasa a notebooks 00-06 con las funciones compartidas en `00_comun`; los datos pesados viven fuera de git en `D:\video_eval\datos` | Yilber | `docs/historico/propuesta_estructura_v2.md` | Dos repos (entrenamiento y 3D); reorganizar carpetas en `logo/` sin git |
| D-20 | 04-nov | S9 | `humming` se publica **solo como etiquetas**, GT v2 y manifiestos en el repo, con un notebook que regenera los frames desde los videos; no se publican imágenes en Roboflow | Yilber | Los frames son de Adobe Stock y YouTube (no redistribuibles) | Hacer público el proyecto de Roboflow con imágenes |
| D-21 | 04-nov | S9 | **El repo pasa a ser una herramienta**: `herramienta_trayectorias` lleva cualquier video hasta la trayectoria 3D con el detector vigente, en un solo entorno (`yolo` + `easydict` y `einops`); los datasets se configuran en `config.yaml` (Roboflow o carpeta local) y ds1, ds2 y humming quedan descritos en el notebook 01; el equipo sigue publicando detectores en `modelos/` y la línea de investigación (01-06) se mantiene | Yilber | El repo debe servir a quien trae sus propios videos, sin depender de los videos de Adobe ni de nuestros datos. `.venv_vda` ya reutilizaba el torch de `.venv` y solo añadía esos dos paquetes. 03, 04, 05 y 06 dan los mismos ✓ tras el cambio | Dos entornos obligatorios; regiones de flor y fondo automáticas (se saltan la 3D si faltan: una referencia automática no sería la flor) |

**Nota a D-05 (05-nov, validación S9):** con los frames de la prueba vv regenerados desde los videos (otra compresión JPEG), la ablación solo ds2 da AP50 0,812 frente a 0,795 de ds2+ds1. Con las imágenes originales se reproducen 0,790 y 0,795. La ventaja de ds2+ds1 sobre ds2 queda dentro de lo que mueve la recompresión: no es significativa. D-05 deja de importar porque D-08 la sustituye.

## Decisiones abiertas

- Método final de 7 puntos para E-03: uniforme, salientes o híbrido (CA-02.6.6).
- Licencia de ds1 (`LICENSE_CONFIRMED`) y lista blanca de videos para el logo (CA-01.1.5 y CA-01.1.6).
- Si la profundidad entra en las geometrías (SP-02.9 → US-03.5).
