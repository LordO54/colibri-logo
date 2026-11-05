# Guía para quien clona el repositorio

**¿Quieres trayectorias de tus propios videos o entrenar con tus datos?** No necesitas esta guía: sigue el README de la raíz.

Esta guía es para la **línea de investigación**: tres recorridos, de menos a más esfuerzo, para comprobar lo que hicimos.

| Recorrido | Llegas a | Necesitas | Tiempo aprox. |
|---|---|---|---|
| A. Ver | cifras, decisiones y figuras | solo el navegador | minutos |
| B. Reproducir las trayectorias | las trayectorias y los 7 puntos de S5 y S7, recalculados | 1 entorno + 4 videos de YouTube | 1 h |
| C. Rehacer todo desde cero | datos, reentrenamiento, tracking, 7 puntos y profundidad | 2 entornos + los 12 videos (4 de Adobe Stock con licencia) + clave de Roboflow + GPU | 1-2 días |

## A. Ver sin ejecutar nada

1. `README.md`: qué es el proyecto y cómo está organizado.
2. `resultados/VIGENTE.md`: qué se usa hoy (detector, tracker, unión).
3. `docs/sprints.md` y `docs/decisiones.md`: qué se hizo en cada sprint y por qué.
4. `resultados/ejecutados/*.html`: cada notebook ya ejecutado, con sus ✓ y ✗ (`LEEME.md` los resume).

## B. Reproducir las trayectorias (camino corto)

Las trayectorias dependen del detector vigente (`modelos/yolo26n_ft_v2/best.pt`, en el repo) y de las trayectorias
detalladas congeladas en `datos_derivados/tracks/`. No hace falta Roboflow, ni los videos de Adobe, ni entrenar.

1. **Clona** el repositorio. La *zona de trabajo* es `investigacion/trabajo/` (`zona_trabajo` en `investigacion/config.yaml`);
   ahí se crean `datos\`, `runs\` y `modelos\`.
2. **Entorno `yolo`**: README de la raíz (venv, `requirements.txt`, kernel `colibri`).
3. **Los videos de YouTube**: en el notebook 01, sección 1, pon `DESCARGAR_YOUTUBE = True` y ejecuta **solo esa
   sección**. Descarga con `yt-dlp` los 8 videos de YouTube (los 4 `test_` son los imprescindibles) de `datos_derivados/fuentes.csv` en la resolución de la
   columna `nota` y los deja en `datos/fuentes/videos/`. La tabla final te dice si `fps` y `frames` coinciden.
4. **Ejecuta `03_tracking_y_union`** (modo `artefactos`). Recalcula las métricas de S5 y escribe las trayectorias en
   `runs/03/trayectorias_prudente/`, con ✓ si coinciden con las del repo. La sección 3 (rejilla de unión) usa también
   los videos vv de Adobe Stock: si no los tienes, se omite y avisa; no afecta a las trayectorias.
5. **Ejecuta `04_trayectorias_7_puntos`**. Escribe los 7 puntos, curvas y láminas en `runs/04/`, con ✓ frente a S7.

Si en el paso 3 un video da otro número de frames (YouTube a veces sirve otra codificación), sus etiquetas no
valen para esa copia y su comparación saldrá ✗. Las demás no se ven afectadas.

## C. Rehacer todo desde cero

1. Pasos 1 y 2 de B, y además Video Depth Anything (`entornos/instalar.md`, sección 3: su repo en el commit fijado y
   el checkpoint Small).
2. **`.env`**: copia `.env.example` y pon tu `ROBOFLOW_API_KEY` (cuenta gratuita de Roboflow). ds1 y ds2 son públicos.
3. **Los 12 videos** (`datos_derivados/fuentes.csv`):
   - 8 de YouTube: como en B.3;
   - 4 de Adobe Stock: hay que **licenciarlos**, buscando cada uno por su id (`id_origen`) en Adobe Stock, y guardarlos
     con el nombre de la columna `archivo`. El enlace de la columna `url` es la biblioteca del proyecto y solo lo
     pueden abrir sus miembros.
4. **`01_fuentes_y_datasets`**: verifica los videos, descarga ds1 y ds2, audita, re-parte ds1 (mismo SHA1 que S1) y
   regenera los frames de humming a partir de los videos y las etiquetas del repo.
5. **`02_detector` con `COLIBRI_MODO=recalcular`**: reentrena S2 (~5 h) y el ajuste fino de S4 (~7 h en una GPU de
   6 GB) y pasa el gate. Los pesos nuevos quedan en `runs/02/<modelo>/weights/best.pt`.
6. **`03` a `06` con `COLIBRI_MODO=recalcular`**: tracking con tu detector, unión, 7 puntos, mapas de VDA (05) y profundidad en los 7 puntos.

**Qué esperar:** los frames regenerados no son los mismos JPEG que los originales, así que las métricas por video de
02 pueden moverse hasta 0,10 (las globales se mantienen). Un detector reentrenado tampoco da cajas idénticas, así que
las cifras de 03 en adelante serán parecidas pero no iguales. Para comparar con el proyecto, usa el modo
`artefactos`.

## ¿Puedo reentrenar? ¿Quién cambia el `best.pt`?

Sí, cualquiera puede reentrenar (C.5). El modelo nuevo se queda en `runs/` y **no sustituye** al del repo: los notebooks
siguen usando el que indica `config.yaml` → `vigente.detector`. Para probar el tuyo, cambia esa línea en tu copia.

El `best.pt` del repo solo lo cambian los mantenedores del proyecto, con un PR que:
1. pasa el gate de 02 en `test_` con un criterio fijado antes de entrenar (como D-08);
2. registra la decisión en `docs/decisiones.md`;
3. actualiza `config.yaml`, `modelos/` y `resultados/VIGENTE.md` (DoD-07).

Los `test_` nunca entran en entrenamiento ni en validación (D-07). Si los usas para elegir algo, la cifra deja de
ser una prueba.
