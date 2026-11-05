# Propuesta v2: repositorio git del proyecto

Fecha: 2026-11-04. Sustituye a la v1 (que reorganizaba carpetas). **Todavía no se ha movido nada.**

## Lo que pide Yilber

1. Que el código suelto (unos 40 `.py` repartidos entre `video-val\resultados` y `D:\video_eval\reentreno\v2`) pase a ser celdas de notebooks.
2. Un repositorio versionado en git.
3. Valorar si conviene tener 2 repositorios (entrenamiento y 3D), dado que el trabajo se hace en `D:`.

## Recomendación: un solo repositorio con dos entornos

Un solo repositorio, no dos. La separación natural está en los **entornos**, no en el código:

- **La 3D consume lo que produce el 2D.** Usa las trayectorias de la unión prudente, los 7 puntos y el mismo detector. Con dos repositorios, el de 3D dependería de una versión concreta del otro, y habría que sincronizar a mano rutas, CSV y decisiones.
- **El backlog, los sprints y las decisiones son uno solo.** Partirlos en dos repositorios duplica la documentación.
- **Lo que de verdad no se puede mezclar son las dependencias.** El `.venv` tiene Ultralytics y torch para YOLO26; el `.venv_vda` tiene Video Depth Anything y otro torch. Eso se resuelve con dos archivos de requisitos y un kernel de Jupyter por entorno. Cada notebook declara en su primera celda qué kernel usa.

Si más adelante la 3D crece hasta ser otro producto, se puede separar con `git subtree split` sin perder la historia. Hacerlo al revés, juntar dos repositorios, es mucho más incómodo.

## Qué va en git y qué no

| En git | Fuera de git (en D:, ruta definida en `config.yaml`) |
|---|---|
| Notebooks (sin salidas pesadas), `config.yaml`, requisitos, docs, backlog, sprints, decisiones | Videos originales (~400 MB) |
| CSV pequeños: manifiestos, splits, métricas, trayectorias, 7 puntos, revisión ciega | Datasets descargados de Roboflow, frames, el zip de 311 MB |
| `best.pt` del modelo vigente (5 MB) y las fichas de todos los modelos | Pesos descartados (`yolo26s*`, 20 MB cada uno), runs de Ultralytics |
| Figuras finales de cada sprint (jpg/png ligeros) | Mapas de profundidad (`.npz`, cientos de MB), entornos `.venv*`, cachés |
| `.gitignore` y `.env.example` | `.env` con la clave de Roboflow (**nunca** en git) |

Las salidas de los notebooks se limpian al hacer commit (`nbstripout`). La versión ejecutada de cada notebook se exporta a HTML en `resultados\` para poder verla sin ejecutar nada.

## Dónde vive cada cosa

```
D:\video_eval\                        ZONA DE TRABAJO (no versionada)
├─ .venv\          entorno "yolo" (entrenamiento, tracking, 2D)
├─ .venv_vda\      entorno "vda" (profundidad)
├─ datos\
│  ├─ fuentes\videos\{vv,train,test}\     los 12 clips (se mueven desde video-val)
│  ├─ roboflow\    ds1, ds2, humming (lo que hoy está en colibri_work\data y reentreno\v2\rf_*)
│  └─ frames\      frames extraídos y listas para entrenar
├─ runs\           salidas crudas de Ultralytics y de la profundidad
└─ colibri-logo\   ◄ EL REPOSITORIO GIT
```

```
colibri-logo\                         REPOSITORIO
├─ README.md               guía: qué es, cómo navegar, cómo replicar, estado
├─ config.yaml             rutas a D:\video_eval\datos y runs; configuración vigente
├─ .env.example  .gitignore
├─ entornos\
│  ├─ requirements-yolo.txt          (exportado de .venv)
│  ├─ requirements-vda.txt           (exportado de .venv_vda + commit del repo VDA)
│  └─ instalar.md                    cómo crear ambos entornos y registrar los kernels
├─ notebooks\
│  ├─ 00_comun.ipynb                 config, rutas y funciones compartidas (se carga con %run)
│  ├─ 01_fuentes_y_datasets.ipynb    inventario, auditoría ds1/ds2, frames vv y v2, splits   [yolo]
│  ├─ 02_detector.ipynb              línea base, entrenamiento (ENTRENAR=False), gate, spike s [yolo]
│  ├─ 03_tracking_y_union.ipynb      trackers, métricas MOT, unión prudente, GT v2           [yolo]
│  ├─ 04_trayectorias_7_puntos.ipynb uniforme / salientes, láminas                          [yolo]
│  ├─ 05_profundidad_3d.ipynb        DEPTH_BACKEND, filtro, z_norm de 7 puntos, revisión ciega [vda]
│  └─ historico\                     los notebooks originales, tal cual se ejecutaron
├─ datos_derivados\       CSV pequeños que creamos: manifiestos, splits, GT, selecciones
├─ modelos\
│  ├─ yolo26n_ft_v2\       best.pt (VIGENTE) + args.yaml + results.csv + ficha.md
│  └─ fichas\              ficha.md de los demás modelos (los pesos se quedan en D:)
├─ resultados\             un subdirectorio por sprint con CONCLUSIONES.md y figuras ligeras
│  └─ VIGENTE.md
└─ docs\
   ├─ backlog_v0_5.md   sprints.md   decisiones.md   mapa_rutas.csv
   ├─ auditoria\        historico\  (backlog v0.3 y v0.4, propuestas, planes)
```

## Código suelto → celdas de notebook

- **Regla:** todo script pasa a ser una sección de su notebook, con un encabezado Markdown que dice qué historia y qué CA cubre, qué se decidió y con qué cifra.
- **Funciones compartidas** (cargar la config, leer los videos, métricas MOT, la unión de fragmentos, cargar las trayectorias): las usan 3 o 4 notebooks. Copiarlas en cada uno es exactamente la deuda que tenemos hoy (la evaluación de tracking está en 3 sitios distintos). Por eso viven en `00_comun.ipynb`, que los demás cargan con `%run 00_comun.ipynb`. Así sigue siendo todo notebook, pero con una sola copia.
- **Dos modos por notebook:** `MODO = "artefactos"` (por defecto: carga el modelo congelado, recalcula y compara con las cifras del sprint) y `MODO = "recalcular"` (por ejemplo, reentrenar, unas 7 h).
- **Los scripts originales** no se pierden: quedan en `notebooks\historico\scripts\` como evidencia, con sus rutas viejas.

## Cómo lo ejecuto

1. **Tú creas el repositorio en GitHub**, vacío y privado (por ejemplo, `colibri-logo`). Desde aquí no puedo crear repositorios. Cuando exista, lo conecto a esta sesión.
2. **Escribo aquí los notebooks 00-05**, la configuración y los docs, y los subo en una rama con un PR para que lo revises.
3. **En tu laptop, con una sesión de Remote Control:**
   - clono el repositorio en `D:\video_eval\colibri-logo`;
   - muevo los datos a `D:\video_eval\datos` (copiar, verificar y luego mover; nada se borra);
   - exporto los requisitos de los dos entornos y registro los kernels;
   - ejecuto 03, 04 y 05 en modo "artefactos" y compruebo que reproducen las cifras de S5, S7 y S8.
4. La carpeta `C:\Users\ACER\Documents\logo\video-val` queda vacía de originales. Puedes borrarla tú cuando lo hayas comprobado.

Para el push desde la laptop hace falta que git tenga tus credenciales de GitHub allí. Si no las tiene, lo empujo yo desde aquí.
