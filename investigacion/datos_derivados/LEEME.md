# Datos derivados

Archivos pequeños que **creamos** nosotros y que no se pueden (o no conviene) regenerar. Todo lo demás se regenera
desde los videos y Roboflow con el notebook 01.

| Ruta | Qué es | Lo usa |
|---|---|---|
| `fuentes.csv` | los 12 videos: id, archivo, carpeta, origen, URL, fps y frames esperados | todos |
| `vv/` | `videos.csv`, `manifest_frames.csv`, `prueba_etiquetas_corregidas.zip` (247 etiquetas e `ignorar_frames.txt`) | 01, 02 |
| `humming/manifest_reentrenamiento.csv`, `seleccion_vv.csv`, `resumen_seleccion.txt` | frames elegidos en S4 y política | 01 |
| `humming/split_manifest_{v1,v2}.csv` | split de cada frame (train, valid, test) | 00, 01, 03 |
| `humming/etiquetas_{v1,v2}/{split}/*.txt` | etiquetas YOLO de humming (sin imágenes, D-20) | 01, 03 |
| `humming/roboflow_v1_vv/*.txt` | etiquetas de vv tal como salieron de Roboflow (GT de la rejilla de unión) | 03 |
| `humming/cambios_v1_v2.csv` | qué cambió entre GT v1 y GT v2 | 03 |
| `tracks/<config>/*__detallada.{csv,meta.json}` | trayectorias detalladas de S5 (una fila por caja y frame) | 03 |
| `profundidad/puntos_i5bE_salientes_delante.csv`, `seleccion_delante.json` | variante `delante` (D-15) | 06 |
| `profundidad/revision_ciega/` | plantilla con las respuestas de Yilber y clave | 06 |

**Estado:** copiado desde la laptop el 04-nov-2026 (ver `docs/mapa_rutas.csv`). `verificacion_delante.jpg` está en la laptop pero `.gitignore` lo deja fuera del repo.
