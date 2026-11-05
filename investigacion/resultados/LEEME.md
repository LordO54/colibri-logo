# Resultados de referencia por sprint

Cada carpeta guarda las cifras **con las que se tomó una decisión**. Los notebooks en modo `artefactos` las
recalculan en `runs/` y las comparan con estas; nunca las sobrescriben. Las figuras son ligeras (jpg/png).

| Carpeta | Sprint | Contenido | Notebook |
|---|---|---|---|
| `S1_datos/` | S1 | `run_record.json` (SHA1 de las particiones) | 01 |
| `S2_detector_v1/` | S2 | línea base y comparación en vv | 02 |
| `S3_tracking_piloto/` | S3 | `barrido_trackers.csv` (piloto en vv, D-06 revertida) | histórico |
| `S4_reentreno_v2/` | S4 | `comparacion.csv` (GT v1), `deteccion_gt2.csv` (GT v2) | 02 |
| `S5_tracking_union_test/` | S5 | `gt1/`, `gt2/` (5 trackers), rejilla y prueba de la unión, `trayectorias_prudente/` | 03 |
| `S6_spike_yolo26s/` | S6 | `*/veredicto.json` | 02 |
| `S7_trayectorias_2d/` | S7 | `resumen_7pts.csv`, puntos por video y método, láminas | 04 |
| `S8_exploracion_3d/` | S8 | `salidas/<backend>/`, `ciega/`, `adobe/`, informes `.md` | 05, 06 |

El origen de cada archivo está en `docs/mapa_rutas.csv`.
