# colibri-logo

Trayectorias de colibríes en video para diseñar un logo. Un detector YOLO26 encuentra el ave, un tracker la sigue,
las visitas a la flor se convierten en trayectorias y cada trayectoria se reduce a **7 puntos** que sirven de geometría
base. Ahora se explora si la profundidad monocular aporta una tercera dimensión creíble.

**Estado (S9, 2026-11-04):** pipeline 2D cerrado y vigente; 3D en exploración (SP-02.9 pendiente); el diseño del logo
(E-03 a E-05) todavía no ha empezado. Qué usar hoy: [`resultados/VIGENTE.md`](resultados/VIGENTE.md).

## Cómo navegar

| Si quieres... | Ve a |
|---|---|
| Entender qué se hizo y por qué, sprint a sprint | [`docs/sprints.md`](docs/sprints.md) |
| Ver cada decisión con su evidencia | [`docs/decisiones.md`](docs/decisiones.md) (D-00 a D-20) |
| Ver el backlog, los CA, la DoR y la DoD | [`docs/backlog_v0_5.md`](docs/backlog_v0_5.md) |
| Reproducir un resultado | `notebooks/` (tabla de abajo) |
| Ver las cifras de referencia | [`resultados/`](resultados/LEEME.md), una carpeta por sprint |
| Saber de dónde viene cada archivo | [`docs/mapa_rutas.csv`](docs/mapa_rutas.csv) |

```
colibri-logo/
├─ config.yaml          rutas a la zona de trabajo, modo y configuración vigente
├─ .env.example         plantilla de .env (clave de Roboflow; .env nunca se sube)
├─ entornos/            requisitos de los dos entornos e instalar.md
├─ notebooks/
│  ├─ 00_comun.ipynb    configuración y funciones compartidas (los demás lo cargan con %run)
│  ├─ 01 … 06           un notebook por etapa
│  └─ historico/        notebooks y scripts originales, tal como se ejecutaron
├─ datos_derivados/     lo pequeño que creamos: fuentes, etiquetas de humming, manifiestos, tracks, juicios humanos
├─ modelos/             detector vigente (best.pt) y fichas de todos los modelos
├─ resultados/          cifras y figuras de referencia por sprint, VIGENTE.md
└─ docs/                backlog, sprints, decisiones, mapa de rutas e histórico
```

Lo pesado no está en git: vive en la **zona de trabajo** (`D:\video_eval` en la laptop de referencia), con
`datos/` (videos, Roboflow, frames), `runs/` (salidas) y `modelos/` (pesos descartados, Video Depth Anything).

## Notebooks

| Notebook | Sprint | Kernel | Qué hace | Decisiones |
|---|---|---|---|---|
| `00_comun` | — | ambos | config, rutas y funciones compartidas | — |
| `01_fuentes_y_datasets` | S1, S4 | yolo | verifica los videos, descarga ds1/ds2, auditoría, re-partición, regenera los frames de humming | D-01..D-03, D-07, D-20 |
| `02_detector` | S2, S4, S6 | yolo | evaluador, línea base, entrenamiento (`ENTRENAR`), gate viejo/nuevo, spike `s` | D-05, D-08, D-11 |
| `03_tracking_y_union` | S3, S5 | yolo | 5 trackers en `test_`, rejilla de unión, unión prudente con GT v1/v2 | D-09, D-10 |
| `04_trayectorias_7_puntos` | S7 | yolo | 7 puntos (uniforme y salientes), curvas y láminas | D-12 |
| `05_profundidad_vda` | S8 | **vda** | mapas de Video Depth Anything Small | D-17 |
| `06_profundidad_3d_7puntos` | S8 | yolo | z en los 7 puntos, filtro, figuras 3D, revisión a ciegas | D-13..D-16, D-18 |

Cada notebook empieza con una tabla de lo que reproduce y termina con las conclusiones del sprint. Tiene dos modos
(`modo` en `config.yaml`, o la variable de entorno `COLIBRI_MODO`):

- **`artefactos`** (por defecto): usa los modelos y CSV congelados, recalcula las métricas y las compara con
  `resultados/` dentro de la tolerancia de `config.yaml` (0,01 en métricas, 1 px en posiciones). Cada comparación
  imprime ✓ o ✗.
- **`recalcular`**: rehace también lo caro: entrenar (~7 h), el tracking y los mapas de profundidad.

## Replicar desde cero

1. **Entornos:** sigue [`entornos/instalar.md`](entornos/instalar.md) (dos `venv` y un kernel de Jupyter por entorno).
2. **Clave:** copia `.env.example` como `.env` y pon tu `ROBOFLOW_API_KEY`.
3. **Videos:** consigue los 12 de [`datos_derivados/fuentes.csv`](datos_derivados/fuentes.csv). Los de YouTube se
   descargan con `yt-dlp` (notebook 01, sección 1); los de Adobe Stock necesitan licencia y se bajan a mano de la
   biblioteca del proyecto. Lo que tiene que coincidir es `fps` y `frames`, porque de eso dependen los índices de las
   etiquetas.
4. **Datos:** ejecuta `01`. Descarga ds1 y ds2 y **regenera** los frames de humming desde los videos con las etiquetas
   del repo.
5. **Resultados:** ejecuta `03`, `04`, `05` (kernel vda) y `06` en modo `artefactos`. Para reentrenar, ejecuta `02` con
   `COLIBRI_MODO=recalcular`.

**Límite conocido:** los frames regenerados no son los mismos JPEG que los originales. Las cifras globales se mantienen (mAP50 de `yolo26n_ft_v2` 0,877), pero las métricas de detección **por video** del notebook 02 pueden moverse hasta 0,10. Los notebooks 03 a 06 no dependen de eso y reproducen sus cifras con diferencia 0.

**Qué no se puede regenerar** y por eso está congelado en el repo: las etiquetas corregidas a mano (humming v1 y v2,
prueba vv), la revisión a ciegas de Yilber y la selección de puntos `delante` de i5bE.

## Datos y licencias

- **Videos:** Adobe Stock (con licencia, no redistribuibles) y YouTube (no redistribuibles). El repo solo guarda sus
  identificadores, fps y número de frames.
- **ds1:** [`magic-workspace/hummingbird-dataset-0zvvn`](https://universe.roboflow.com/magic-workspace/hummingbird-dataset-0zvvn) v1,
  licencia declarada MIT; la procedencia de sus videos no está verificada (CA-01.1.5 abierto).
- **ds2:** [`modelodevisinartificialcolibres/hummingbirds-detect-classify-gcm5s`](https://universe.roboflow.com/modelodevisinartificialcolibres/hummingbirds-detect-classify-gcm5s) v1,
  **CC BY 4.0**: se atribuye a sus autores en Roboflow Universe.
- **humming** (nuestro): se publica como etiquetas YOLO, manifiestos y GT v2, sin imágenes (D-20).
- **Modelos:** Ultralytics YOLO26 (AGPL-3.0). Video Depth Anything Small (Apache-2.0); Base y Large son CC-BY-NC y no se usan.

## Reglas del proyecto

- La clave de Roboflow solo vive en `.env`; nunca en notebooks, salidas ni git.
- Las trayectorias detalladas y las cifras de referencia no se sobrescriben: lo nuevo va a `runs/`.
- Los `test_` no entran nunca en entrenamiento ni en validación (D-07).
- Las figuras con cuadros de video (fondos, recortes) no se suben: se regeneran en `runs/` (`.gitignore`).
- Cada función existe una sola vez (`00_comun`). Las salidas de los notebooks se limpian al hacer commit (`nbstripout`).
