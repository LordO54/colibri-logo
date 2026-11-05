# Sprints del proyecto

Reconstrucción a partir de los hilos de trabajo, la memoria del proyecto y los archivos del 1 al 4 de noviembre de 2026. Los sprints fueron cortos (de horas a un día), porque cada uno terminaba con una decisión de Yilber como PO. Las historias se refieren al backlog v0.5 (`backlog_v0_5.md`) y las decisiones a `decisiones.md`.

**Cómo leer cada sprint:**
- **Objetivo:** la pregunta que había que responder.
- **Incremento:** lo que quedó usable para el sprint siguiente.
- **Decisiones:** qué se decidió y con qué evidencia.
- **Conclusiones:** lo aprendido, incluido lo que salió mal.

## Mapa

| Sprint | Fechas | Etapa | Épica | Incremento clave | Resultado |
|---|---|---|---|---|---|
| S0 | ≤ 01-nov | Planteamiento | todas | Backlog v0.3 | 5 épicas, DoR y DoD |
| S1 | 01-nov | Datos y prueba | E-01 | Fichas ds1/ds2, prueba vv etiquetada, backlog v0.4 | ds1 con fuga ~100 %; se entrena ds2+ds1 |
| S2 | 01-nov | Detector v1 | E-02 | `yolo26n_ds2_ds1` | AP50 en vv 0,70 → 0,795 |
| S3 | 01-02 nov | Tracking piloto | E-02 | Pipeline de tracking y métricas, barrido de 25 corridas | El ave se fusiona con la flor: hacen falta datos del dominio |
| S4 | 02-nov | Reentrenamiento v2 | E-01, E-02 | `yolo26n_ft_v2` (VIGENTE) y prueba `test_` | mAP50 en test_ 0,706 → 0,877 |
| S5 | 03-nov | Tracking y unión en test_ | E-02 | ByteTrack combo + unión prudente, GT v2 | 1 segmento en 3 de 4 videos |
| S6 | 03-nov | Spike: modelo mayor | E-02 | Evidencia negativa | `s` no mejora; el cuello son los datos |
| S7 | 03-04 nov | Trayectorias 2D para el logo | E-02 → E-03 | 7 puntos (uniforme y salientes), láminas | Insumo listo para geometrías |
| S8 | 04-nov | **Exploración 3D** | E-02 (SP-02.7) | z_norm de 7 puntos, selector de backend, revisión ciega | VDA Small 14/17; decisión "otra prueba" |
| S9 | 04-nov → | Consolidación | E-06 | Repo `colibri-logo`, notebooks 00-06, README, backlog v0.5 | **En curso** |

---

## S0 Planteamiento

- **Objetivo:** definir el producto y el camino desde el video hasta el logo.
- **Historias:** backlog v0.1 a v0.3.
- **Incremento:** backlog v0.3 con 5 épicas (E-01 a E-05), DoR, DoD, principios y riesgos. Tracking secuencial como flujo central y profundidad como opción.
- **Decisiones:** D-00 (centro de caja como referencia, el 2D primero).
- **Conclusiones:** el plan suponía "un video del proyecto" que no llegó. El proyecto acabó usando clips de stock y de YouTube, y eso cambió E-01.

## S1 Datos externos y prueba de decisión

- **Objetivo:** ¿con qué datos entrenamos y contra qué medimos?
- **Historias:** US-01.4, US-01.5, US-01.3 (vv), US-01.2 y US-01.1 parcial.
- **Incremento:**
  - Ficha de ds1: 632 imágenes que salen de 2 videos, con fuga del ~100 % entre splits y Stretch 16:9 → 640². Re-partición por bloques: 442/83/82.
  - Ficha de ds2: 6766 fotos, CC BY 4.0, sin fugas.
  - Prueba vv: 3 clips de stock, 243 frames a ~6 fps y 174 cajas corregidas por Yilber en `humming`.
  - Backlog v0.4 (YOLO26, ByteTrack frente a OC-SORT, un notebook).
- **Decisiones:**
  - D-01: adoptar Roboflow y no etiquetar desde cero.
  - D-02: re-partir ds1 por bloques temporales.
  - D-03: entrenar ds2+ds1, con ablación solo de ds2.
  - D-04: los vv son la prueba y no entran en entrenamiento.
- **Conclusiones:**
  - El mAP de 97,9 % que publica Roboflow para ds1 es una fuga; no hay que fiarse de las métricas publicadas.
  - La línea base yolo26s COCO (AP50 0,93) está sesgada porque generó las pre-etiquetas. La referencia justa es yolo26n@640 (0,70).
  - Desde la nube no se llega a app.roboflow.com: las descargas se hacen en la laptop.

## S2 Detector v1

- **Objetivo:** ¿un YOLO26 ajustado detecta mejor que COCO en nuestros videos?
- **Historias:** US-02.2 y US-02.3.
- **Incremento:** `yolo26n_ds2_ds1` (40 épocas, 5,1 h, val mAP50 0,908). En vv: AP50 0,795, P 0,843, R 0,862, frente a 0,70 / 0,76 / 0,52 de COCO. Ablación ds2: 0,790.
- **Decisiones:** D-05, adoptar `yolo26n_ds2_ds1`.
- **Conclusiones:**
  - ds1 aporta poco, porque tiene solo 2 escenas.
  - En Windows, `cache=ram` se duplica por cada worker y provoca OOM: hay que usar `cache=disk` y 2-3 workers. La CPU es el cuello (~5,3 min/época).

## S3 Tracking piloto en vv

- **Objetivo:** ¿qué tracker mantiene una sola identidad por ave?
- **Historias:** US-02.1, US-02.4 y US-02.5 (v1); US-01.2 (criterio de cajas).
- **Incremento:**
  - Pipeline `model.track(persist=True)` con métricas MOT por video y trayectorias CSV.
  - Comparación por defecto: vv01 con 5 IDs en ByteTrack y 4 en OC-SORT; vv02 y vv03 perfectos.
  - Barrido de 25 corridas sobre las detecciones guardadas.
- **Decisiones:**
  - D-06: recomendar OC-SORT combo (`match` 0,95, `buffer` 300, `new` 0,5; IDF1 de 0,69 a 0,80). Era provisional, porque se eligió sobre la propia prueba.
  - D-07: Yilber pasa vv a entrenamiento y crea una prueba nueva `test_`.
- **Conclusiones:**
  - El fallo de vv01 viene del detector: une el ave y la flor en una sola caja. Ni el tracker ni inferir a 960 o 1280 px lo arreglan.
  - Elegir parámetros mirando la prueba da cifras optimistas, y de ahí sale el principio 9.
  - Se lanzó por error un entrenamiento yolo26s de 25 épocas y se detuvo: "25 runs" eran de tracking. Lección: confirmar la interpretación antes de ocupar la GPU varias horas.

## S4 Reentrenamiento v2

- **Objetivo:** ¿los frames de video del dominio mejoran el detector en escenas nuevas?
- **Historias:** US-01.1, US-01.3, US-01.5 (v2), US-02.2 y US-02.3.
- **Incremento:**
  - Revisión de los 12 clips con roles fijos y 0 alertas de fuga.
  - Política de densidad adaptativa.
  - 518 frames subidos a `humming` y corregidos por Yilber; la versión 1 se re-partió según el manifiesto (train 299 ×3, valid 79, test 216).
  - **`yolo26n_ft_v2`**: ajuste fino de 20 épocas.
- **Decisiones:** D-08, `yolo26n_ft_v2` vigente. Cumple el criterio fijado: test_ mAP50 0,706 → 0,877, R 0,757 → 0,821; ds2 0,968 → 0,963.
- **Conclusiones:**
  - Los datos del dominio dan +17 pts: Gu30 0,46 → 0,97 y J4PK 0,66 → 0,82.
  - El precio son cajas de más junto a la flor en i5bE (P@0,25 0,97 → 0,66).
  - El mAP50-95 baja, probablemente porque las pre-etiquetas venían del modelo viejo (sin comprobar).

## S5 Tracking y unión en test_

- **Objetivo:** ¿qué configuración da trayectorias continuas en escenas nunca vistas?
- **Historias:** US-02.4 y US-02.5; US-01.5 (GT v2).
- **Incremento:**
  - 5 configuraciones fijadas de antemano (sin barrido en test_).
  - Unión de fragmentos con puentes marcados (`es_puente`).
  - Lista de aves que faltaban en la GT; Yilber las corrigió (GT v2, 177 cajas).
  - Trayectorias finales en `eval_gt2/union/prudente/`.
- **Decisiones:**
  - D-09: ByteTrack combo, elegido por la regla min(IDs + IDSW) → IDF1 → MOTA. Supera a OC-SORT en `test_`, lo que revierte la recomendación de S3.
  - D-10: unión prudente, 1 s y 2,5× (Yilber), frente a la ganadora de la rejilla (0,5 s × 4), que dibujaba rectas largas.
- **Conclusiones:**
  - Con GT v2: detección mAP50 0,801; tracking R 0,723, P 0,901, IDF1 0,489; IDF1 de visita 0,667 y 0 puentes falsos.
  - Lo que falla es la fragmentación, no las tracks espurias.
  - Una GT incompleta engaña: con la GT v1 los "puentes falsos" eran en realidad aves sin etiquetar.

## S6 Spike: detector más grande

- **Objetivo:** ¿yolo26s resuelve el recall que le falta a la n?
- **Historias:** SP-02.3b.
- **Incremento:** `yolo26s_v2` y `yolo26s_ft_v2` evaluados con un criterio fijado de antemano, con `veredicto.json` y la comparación de trayectorias.
- **Decisiones:** D-11, no adoptar `s`. R 0,678 frente a 0,734 de la n y cobertura 0,52 frente a 0,68; en Gu30 la s se hunde (mAP50 0,41-0,60).
- **Conclusiones:** el cuello de botella son los datos del dominio, no el tamaño del modelo. Inferir a 960 no ayuda. La GPU se estrangula a 85 °C (las épocas pasan de 5 a 27 min).

## S7 Trayectorias 2D para el logo

- **Objetivo:** ¿cómo se reduce una trayectoria a una forma dibujable sin perder su origen?
- **Historias:** US-02.6.
- **Incremento:** 7 puntos en total por video sobre todos los segmentos (los huecos se cortan), con dos métodos (uniforme en el tiempo y salientes con Visvalingam-Whyatt), curva Catmull-Rom y láminas a la misma escala y ampliadas. Error de forma del 3 al 9 % de la diagonal.
- **Decisiones:** D-12, 7 puntos en total y no por segmento (corrección de Yilber). El método final queda abierto.
- **Conclusiones:**
  - i5bE es el más legible, y su lazo alrededor de la flor solo lo conserva el método de salientes.
  - Con el uniforme, J4PK llena el cuadro; Gu30 casi no se mueve, y DoJ1 es el menos fiable porque tiene puntos en puentes.
  - Propuesta pendiente: un híbrido con 7 tramos de tiempo y el punto saliente de cada uno.

## S8 Exploración 3D (etapa actual)

- **Objetivo:** ¿podemos añadir una profundidad creíble a la trayectoria?
- **Historias:** SP-02.7.
- **Incremento:**
  1. Notebook de profundidad con YOLO26-Depth sobre todos los frames, con un filtro de estabilidad fijado de antemano.
  2. Método de 7 puntos con `z_norm` relativo a la flor y al fondo, barra de error por TTA y control con los frames vecinos.
  3. Variantes de i5bE: punto 2 corregido, punto "delante" forzado y escalas de z.
  4. Notebook `colibri_3d_7puntos.ipynb` con selector `DEPTH_BACKEND` (yolo26, yolo26_crop, vda) y entorno `.venv_vda`.
  5. Protocolo de revisión a ciegas y su puntuación.
  6. La trayectoria 3D de i5bE con VDA Small.
- **Decisiones:**
  - D-13: descartar YOLO26-Depth cuadro a cuadro como coordenada continua (1 de 8 videos pasa el filtro).
  - D-14: método de 7 puntos con normalización dentro de cada mapa (idea de Yilber).
  - D-15: forzar un punto delante en i5bE (Yilber).
  - D-16: la escala de z es una decisión de diseño, no una medida.
  - D-17: solo VDA Small, porque Base y Large son no comerciales.
  - D-18: **otra prueba** con VDA Small en otro video, con ciegos repartidos en el tiempo.
- **Conclusiones:**
  - El modelo cuadro a cuadro parpadea incluso con la cámara quieta: el problema es el modelo, no la estabilización.
  - "Estable no es correcto": f83 da una lectura consistente con el signo equivocado.
  - Sin GT 3D, la revisión a ciegas es el único juez fiable: VDA Small acierta 14/17 (p = 0,013), los YOLO entre 8 y 12.
  - Cautela: un solo video, y los "detrás" están todos en los primeros 5 s.

## S9 Consolidación y reproducibilidad (en curso)

- **Objetivo:** que cualquier persona pueda navegar el proyecto y reproducir S1 a S8 desde notebooks.
- **Historias:** US-06.1, US-06.2 y US-06.3.
- **Incremento esperado:**
  - Repositorio git `colibri-logo` con dos entornos (D-19) y la zona de trabajo en `D:\video_eval`.
  - Notebooks 00-06 (código suelto convertido en celdas, funciones compartidas en `00_comun`), verificados en modo `artefactos`.
  - `humming` publicado como etiquetas + manifiestos, con regeneración de frames desde los videos (D-20).
  - README, `VIGENTE.md`, backlog v0.5, este documento y el registro de decisiones.
- **Decisiones:** D-19 (un repo, dos entornos) y D-20 (humming solo como etiquetas).
- **Criterio de cierre:** los notebooks 03, 04 y 06 reproducen las cifras de S5, S7 y S8 dentro de la tolerancia de `config.yaml` (0,01 en métricas, 1 px en posiciones).

---

## Siguiente sprint propuesto (S10): cerrar el 3D y arrancar el diseño

- **Objetivo:** decidir si el 3D entra en las geometrías y empezar a seleccionar formas con un brief.
- **Candidatas:**
  1. **US-04.1 Brief de diseño** (P1). Es el mayor riesgo, porque debía ir en paralelo y no se hizo.
  2. **SP-02.9** segunda prueba de profundidad con VDA Small (un video nuevo, ciego repartido).
  3. **US-03.1 y US-03.2:** clasificar segmentos y montar el atlas de los 12 videos.
  4. **US-01.1 CA-01.1.5:** procedencia y licencias de los clips (lista blanca para el logo).

## Épicas e historias pendientes de explorar

| Épica | Pendiente | Por qué importa |
|---|---|---|
| **E-03 Geometrías** | Toda la épica: US-03.1 a US-03.6 | Es el puente entre los datos y el logo; hoy solo hay insumos |
| **E-04 Propuestas visuales** | Toda la épica, **empezando por US-04.1 (brief)** | Sin brief no hay criterio para elegir geometrías |
| **E-05 Validación y entrega** | Toda la épica | Cierre del proyecto con el stakeholder |
| E-01 | US-01.1 (licencias de los clips de YouTube y Adobe), US-01.4 (`LICENSE_CONFIRMED` de ds1), **US-01.6** (prueba virgen) | Uso legítimo de las trayectorias en el logo y cifras finales sin sesgo |
| E-02 | **US-02.8** (recall en J4PK y aves difíciles), **SP-02.9** (2.ª prueba 3D), CA-02.6.6 (método final de 7 puntos) | Continuidad de las trayectorias y decisión sobre la 3D |
