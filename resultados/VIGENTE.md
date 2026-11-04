# Qué usar hoy

Actualizado: 2026-11-04 (S9). Si cambia algo de esta lista, se actualiza aquí y en `docs/decisiones.md` (DoD-07).

| Pieza | Vigente | Decisión | Dónde |
|---|---|---|---|
| Detector | `yolo26n_ft_v2` | D-08 | `modelos/yolo26n_ft_v2/best.pt` |
| Tracker | ByteTrack combo: `track_buffer` 300, `match_thresh` 0,95, `new_track_thresh` 0,5, `gmc_method` none, conf 0,1, imgsz 640 | D-09 | `config.yaml` → `vigente.tracker`; `00_comun` sección 5 |
| Unión de fragmentos | prudente: 1 s y 2,5× el lado medio de la caja | D-10 | `00_comun` → `union_vigente` |
| Trayectorias de prueba | 4 videos `test_`, con `es_puente` | D-10 | `resultados/S5_tracking_union_test/trayectorias_prudente/` |
| Geometría del logo | 7 puntos en total por video; método aún abierto (uniforme o salientes) | D-12, CA-02.6.6 | `resultados/S7_trayectorias_2d/` |
| Profundidad | **exploratoria**: VDA Small en los 7 puntos, `z_norm` relativo a flor y fondo; pendiente de SP-02.9 | D-14, D-17, D-18 | `resultados/S8_exploracion_3d/` |
| Verdad de terreno | GT v2 de `humming` | D-10 | `datos_derivados/humming/etiquetas_v2/` |

**No usar:** `yolo26s*` (D-11), OC-SORT (D-06 revertida), la unión 0,5 s × 4 de la rejilla, YOLO26-Depth como z continua (D-13),
VDA Base o Large (licencia CC-BY-NC, D-17).
