# Backlog de producto v0.5
## Rediseño de identidad visual a partir de trayectorias de vuelo del colibrí

**Product Owner y stakeholder técnico:** Yilber.
**Stakeholder de diseño:** representante de la organización responsable de la identidad visual (todavía sin brief, ver US-04.1).
**Objetivo del producto:** un logotipo renovado cuyas formas derivan, de manera trazable, de trayectorias de vuelo de colibrí observadas en video.

**Fecha:** 2026-11-04. Sustituye a la v0.4 (01-nov), que queda en `docs/historico/`.

### Qué cambia en la v0.5

1. **El backlog refleja lo que hicimos.** Cada historia lleva su **estado**, el **sprint** donde se trabajó, sus criterios de aceptación marcados (✓ cumplido, ◐ parcial, ✗ no cumplido, ○ pendiente) y la **evidencia** (ruta en la estructura nueva). Las historias de la v0.4 que se resolvieron de otra forma se reescriben.
2. **Las fuentes cambiaron.** Ya no hay "un video del proyecto": hay 12 clips con roles fijos (`vv`, `train`, `test`). Los 4 `test_` son la prueba de decisión del detector y del tracker.
3. **DoR y DoD nuevas** (secciones 3 y 4). Recogen lo que aprendimos: los umbrales se fijan antes de mirar la prueba, las licencias se revisan antes de instalar un modelo, todo resultado debe salir de un notebook reproducible y cada decisión queda registrada.
4. **Historias nuevas:** US-01.6 (integridad de la prueba), US-02.8 (recall en aves difíciles), SP-02.9 (segunda prueba de profundidad con VDA) y la épica transversal **E-06 Reproducibilidad y organización**.
5. **Cambios de alcance:**
   - La talla del detector queda fijada en `n` (el spike de `s` dio negativo).
   - La "densidad configurable" de US-02.6 se concreta en la reducción a **7 puntos**.
   - La profundidad pasa de "opcional abstracto" a un spike con un juez definido: la **revisión a ciegas**.

### Estado de un vistazo

| Épica | Estado | Comentario |
|---|---|---|
| E-01 Evidencia audiovisual y datos | **Hecha, con 2 pendientes** | Falta la procedencia y licencia de los clips de YouTube y una prueba "virgen" |
| E-02 Detección, seguimiento y trayectoria | **Hecha en 2D; 3D en curso** | Configuración vigente congelada; SP-02.7 decidió "otra prueba" |
| E-03 Geometrías | **Sin empezar** (solo insumos) | Las trayectorias de 7 puntos son el insumo |
| E-04 Propuestas visuales | **Sin empezar** | El brief (US-04.1) debía ir en paralelo y no se hizo: es el riesgo principal |
| E-05 Validación y entrega | **Sin empezar** | Depende de E-04 |
| E-06 Reproducibilidad y organización | **En curso (Sprint 9)** | Estructura, notebooks, README, decisiones |

---

# 1. Principios de trabajo

1. **El video es la fuente de evidencia.** Se prefiere una toma estable donde el colibrí sea lo que se mueve.
2. **El objeto de seguimiento es el cuerpo del ave**, representado por el centro de la caja. Las alas y la pose no entran en el alcance.
3. **Detectar y seguir son procesos distintos.** YOLO localiza; el tracker asocia.
4. **Primero el detalle, después la simplificación.** La trayectoria densa es la fuente de verdad; los 7 puntos son derivados trazables.
5. **No se inventan tramos.** Los puentes entre fragmentos (`es_puente`) se marcan como derivación, nunca como observación.
6. **La geometría es una hipótesis de diseño.** No se presupone simetría ni se generaliza a la especie.
7. **Calidad técnica y calidad de diseño se evalúan por separado.**
8. **La profundidad no bloquea el 2D.** Solo entra si un juez independiente la respalda.
9. **(nuevo) La prueba de decisión no se toca para ajustar.** Los umbrales y los criterios se fijan antes de mirarla; los parámetros se ajustan en `train_`/`vv`.
10. **(nuevo) Sin verdad de terreno, el juez es humano y ciego.** Cuando no hay GT (por ejemplo, profundidad), Yilber etiqueta sin ver las cifras del modelo.
11. **(nuevo) Licencia antes de instalar.** Ningún modelo, dataset o video entra al flujo sin revisar su licencia de uso comercial.
12. **(nuevo) Si no está en un notebook, no está hecho.** Un resultado cuenta cuando se puede regenerar desde `notebooks/` con `config.yaml`.

# 2. Priorización

| Prioridad | Significado |
|---|---|
| **P0 Gate** | Resuelve una incertidumbre o autoriza la etapa siguiente. |
| **P1 Core** | Necesario para el resultado principal. |
| **P2 Opcional** | Solo entra al flujo si demuestra valor. |

Estados: **Hecha**, **Hecha con pendiente** (se usa, pero queda un CA abierto que no bloquea), **En curso**, **Pendiente**, **Bloqueada**, **Descartada** (spike con resultado negativo, se conserva como evidencia).

# 3. Definition of Ready (DoR) v0.5

Una historia entra en un sprint cuando cumple **todo** lo siguiente:

- **DoR-01 Valor claro.** Se sabe qué decisión o qué entregable habilita y para quién (análisis, diseño o stakeholder).
- **DoR-02 Criterios verificables y fijados de antemano.** Cada CA se comprueba con una cifra, un archivo o una decisión registrada. Si la historia compara alternativas, el **criterio de elección y sus umbrales se escriben antes de ejecutar** (ejemplo: "adoptar `s` si gana ≥3 pts de recall sin perder >2 pts de precisión").
- **DoR-03 Datos con rol.** Las entradas están identificadas en `config.yaml` con su rol (train, valid, test o exploración). Si la historia usa `test_`, se declara para qué y se confirma que no se ajusta nada con él.
- **DoR-04 Dependencias Hechas.** Las historias previas están Hechas, o bien su pendiente está identificado y no afecta.
- **DoR-05 Coste estimado y cabe en la laptop.** Se estiman las horas de GPU (RTX 2060, 6 GB; la GPU se estrangula a 85 °C) y las horas de trabajo humano. Si hay un entrenamiento de más de 2 h, se planifica en segundo plano y con reanudación.
- **DoR-06 Licencias revisadas.** Cualquier modelo, dataset o video nuevo tiene la licencia leída y compatible con un logo de uso comercial (por ejemplo, VDA Base y Large son CC-BY-NC y se excluyen).
- **DoR-07 Trabajo humano agendado.** Si hace falta que Yilber corrija etiquetas o haga una revisión a ciegas, se sabe cuántos elementos son y el material está preparado.
- **DoR-08 Verificación y evidencia definidas.** Se sabe qué figura, CSV o notebook quedará, y en qué carpeta de `resultados/`.

**DoR adicional para spikes:** la pregunta, la muestra mínima, el límite de esfuerzo, el juez (métrica o revisión ciega), las alternativas y **qué cambia en el backlog con cada resultado posible** (incorporar, otra prueba o descartar).

# 4. Definition of Done (DoD) v0.5

Una historia está Hecha cuando:

- **DoD-01 Aceptación con evidencia.** Cada CA está marcado y apunta a un archivo o a una cifra.
- **DoD-02 Reproducible desde notebook.** El paso está en un notebook de `notebooks/` que se ejecuta de punta a punta en modo "desde artefactos" con `config.yaml`, sin rutas absolutas en el código. Se registran las versiones (`ultralytics`, `torch`), la semilla y los datos de entrada.
- **DoD-03 Cada cosa en su sitio.** El código va en celdas de `notebooks/` (lo compartido, una sola vez, en `00_comun`), los datos derivados pequeños en `datos_derivados/`, las cifras de referencia en `resultados/Sx_*/` y lo pesado fuera de git, en la zona de trabajo. Nada queda suelto en la raíz.
- **DoD-04 Integridad de la prueba.** Se declara si se usó `test_` y para qué. Si se usó para elegir algo, el resultado se marca como optimista.
- **DoD-05 Decisión registrada.** Si la historia decide algo, queda en `docs/decisiones.md`: fecha, quién decide, evidencia, alternativas descartadas.
- **DoD-06 Limitaciones visibles.** Se separan lo medido de lo interpretado, y los errores conocidos y los riesgos para las historias siguientes.
- **DoD-07 Vigente al día.** Si cambia el modelo, el tracker, la unión o las trayectorias de referencia, se actualiza `resultados/VIGENTE.md`.
- **DoD-08 Sin secretos.** La clave de Roboflow solo está en `.env` y no aparece en notebooks, logs ni memoria.
- **DoD-09 Revisada por el PO.** Yilber revisó el resultado frente a los CA.

**DoD adicional por tipo de entregable**

| Tipo | Evidencia adicional |
|---|---|
| Fuente o dataset | Procedencia, licencia, rol, manifiesto con hash o recuento, correspondencia frame-tiempo |
| Modelo | `args.yaml`, `results.csv`, `best.pt`, métricas en `test_` y en ds2 test, mosaico de aciertos y errores |
| Tracking y trayectoria | YAML del tracker, métricas por video, CSV de trayectoria con `es_puente`, figura sobre el frame |
| Profundidad | Filtro de estabilidad, barra de error, resultado de la revisión ciega, licencia del modelo |
| Geometría | Datos de origen, operaciones, visualizaciones y límites interpretativos |
| Diseño | Archivos editables, pruebas a escala y monocromo, justificación formal |

**DoD del incremento (fin de sprint):** el incremento se puede usar en el sprint siguiente sin reconstrucción manual. Además, `docs/sprints.md` tiene su objetivo, incremento, decisiones y conclusiones, y la memoria del proyecto está actualizada.

---

# 5. E-01 Caracterizar y preparar la evidencia audiovisual

**Objetivo:** disponer de fuentes con un rol fijo y de datos etiquetados para entrenar y para decidir.

## US-01.1 Inventariar y caracterizar las fuentes
**P0 · Hecha con pendiente · S1, S4**

Como responsable del análisis, quiero conocer cada clip y su rol para usarlo sin fugas y con permisos.

- ✓ CA-01.1.1 Hay una ficha por video con fps, frames, resolución y duración (`fuentes/fuentes.csv`; en origen, `reentreno/resumen.txt`).
- ✓ CA-01.1.2 Cada video se revisó con una hoja de contacto de 12 frames: especie, escena, cámara, número de aves, flor o comedero, tramos sin ave.
- ✓ CA-01.1.3 Cada video tiene un rol fijo: `vv` (3), `train` (5, con AdobeStock_1600207728) y `test` (4). Los test no comparten escena con train ni con vv (0 alertas dHash, `alertas_fuga.csv`).
- ✓ CA-01.1.4 Se documentaron las rarezas: vv03 es un rufo y no un jacobino; Gu30 tiene fundidos a negro; vv02 deriva 37 px; JrsOM3Z y vIMu4V2Jgww son la misma baranda.
- ○ CA-01.1.5 **Pendiente:** procedencia y licencia de cada clip. Los Adobe tienen licencia de stock por confirmar; los `train_`/`test_` vienen de YouTube (descargados con YTDown) sin licencia registrada. Para entrenar el riesgo es bajo; para un logo derivado de una trayectoria concreta hay que saberlo.
- ○ CA-01.1.6 **Pendiente:** decidir de qué videos puede salir una geometría para el logo (lista blanca).

**Evidencia:** `fuentes/fuentes.csv`, `resultados/S4_reentreno_v2/revision/`.

## US-01.2 Definir la referencia corporal
**P0 · Hecha · S1, S3**

- ✓ CA-01.2.1 La referencia es el centro de la caja `(cx, cy)` en píxeles del video original, con `t_s = frame / fps`.
- ✓ CA-01.2.2 Está documentado que el centro varía con las alas y la postura (35 % de cajas anchas en ds1).
- ✓ CA-01.2.3 Se comparó el criterio de caja entre las predicciones y la GT (`criterio_cajas/`, mosaico): no hizo falta cambiar de referencia.
- ✓ CA-01.2.4 Se suaviza con Savitzky-Golay (~0,2 s, grado 2) solo en los derivados; la trayectoria densa no se toca.

## US-01.3 Extraer frames con densidad controlada
**P1 · Hecha · S1, S4**

- ✓ CA-01.3.1 Prueba vv: frames a ~6 fps (247; 243 evaluables), con manifiesto.
- ✓ CA-01.3.2 Entrenamiento: política adaptativa. Hay candidatos cada 0,1 s y se guarda uno si pasaron ≥0,4 s (0,2 s en frames difíciles) y cambió ≥0,3 % de la miniatura, o si pasaron ≥1,5 s. Tope de 120 por video, valid = último 20 % con 1 s de guarda.
- ✓ CA-01.3.3 Test: 3 fps fijos y se descartan los frames negros (gris < 12).
- ✓ CA-01.3.4 Cada frame lleva video, índice y tiempo (`manifest_reentrenamiento.csv`).
- ✓ CA-01.3.5 El script se puede repetir (`preparar_frames_roboflow.py`). En v0.5 queda congelado en `datos_derivados/humming/` y el notebook 01 regenera los frames.

## US-01.4 Auditar los datasets externos
**P0 · Hecha con pendiente · S1**

- ✓ CA-01.4.1 ds1 (`magic-workspace/hummingbird-dataset-0zvvn` v1): 632 imágenes que son frames de solo 2 videos, con fuga del ~100 % entre splits y "Stretch to 640". El mAP de 97,9 % que publica Roboflow está inflado.
- ✓ CA-01.4.2 ds2 (`modelodevisinartificialcolibres/hummingbirds-detect-classify-gcm5s` v1): 6766 fotos, CC BY 4.0, sin fugas.
- ✓ CA-01.4.3 humming (`yilber-lopez/humming` v1/v2): frames propios corregidos a mano por Yilber.
- ◐ CA-01.4.4 Licencias: ds2 CC BY 4.0 (requiere atribución, que hay que incluir en la entrega). ds1 declara MIT, pero los videos de origen son de terceros. **Falta la decisión `LICENSE_CONFIRMED` de ds1.**

**Evidencia:** `docs/auditoria/ficha_dataset_roboflow_v1.md`, `ficha_dataset_roboflow_hummingbirds_detect_classify_v1.md`.

## US-01.5 Congelar particiones y prueba de decisión
**P1 · Hecha · S1, S4, S5**

- ✓ CA-01.5.1 ds1 re-partido por bloques temporales (40 frames, guarda 15): 442/83/82.
- ✓ CA-01.5.2 Prueba v1 (S1-S3): 243 frames vv con 174 cajas corregidas; 4 frames de vv02 excluidos.
- ✓ CA-01.5.3 Prueba v2 (S4): los 216 frames `test_`; el split sale del manifiesto y no de Roboflow (`split_manifest.csv`).
- ✓ CA-01.5.4 GT v2 (S5): Yilber completó las aves que faltaban, con 177 cajas. Las cifras con GT v1 quedan obsoletas.
- ✓ CA-01.5.5 Mezcla de entrenamiento v2: ds2 + ds1 + humming ×3; humming reducido a 1280 px.

## US-01.6 (nueva) Mantener la integridad de la prueba de decisión
**P0 · Pendiente**

Como PO, quiero que las cifras finales no estén sesgadas por haber mirado la prueba muchas veces.

- ○ CA-01.6.1 Hay un registro de cada uso de `test_`: S4 gate del detector, S5 tracking (GT v1 y v2) y elección de la unión, S6 spike s, S8 profundidad.
- ○ CA-01.6.2 Antes de presentar cifras en la entrega final, hay una **prueba nueva** (2 o 3 clips no vistos, con licencia clara) evaluada **una sola vez** con la configuración vigente congelada.
- ○ CA-01.6.3 Las cifras de `test_` se presentan como "prueba de desarrollo".

---

# 6. E-02 Detectar, seguir y reconstruir la trayectoria

**Objetivo:** trayectorias densas y fiables de una sola ave por video, más sus derivados.

## US-02.1 Protocolo de procesamiento y seguimiento
**P0 · Hecha · S3**

- ✓ CA-02.1.1 Procesamiento secuencial con `model.track(persist=True)`, tracker explícito y YAML versionado, `gmc_method: none`.
- ✓ CA-02.1.2 Detección con conf 0,1 e imgsz 640. Subir la conf a 0,25 empeora (19 IDSW en el piloto).
- ✓ CA-02.1.3 Las métricas por video son TP, FN, FP, IDSW, IDs por ave, recall, precisión, MOTA, IDF1 e IoU, con una sola identidad GT por video.
- ✓ CA-02.1.4 La trayectoria exportada tiene una fila por frame: `frame, t_s, track_id, cx, cy, w, h, conf`, más `segment_id` y `es_puente` tras la unión.

## US-02.2 Entrenar el detector YOLO26
**P1 · Hecha · S2, S4**

- ✓ CA-02.2.1 Línea base COCO `bird`: yolo26n@640 AP50 0,70, P 0,76, R 0,52 en vv.
- ✓ CA-02.2.2 yolo26n_ds2_ds1: 40 épocas, 5,1 h, val mAP50 0,908. En vv: AP50 0,795, P 0,843, R 0,862. La ablación solo con ds2 da 0,790, así que ds1 aporta poco.
- ✓ CA-02.2.3 yolo26n_ft_v2: ajuste fino de 20 épocas desde ds2_ds1 con humming ×3, SGD lr0 0,002, seed 0.
- ✓ CA-02.2.4 Se conservan `args.yaml`, `results.csv`, `best.pt` y `split_manifest.csv` de cada modelo.
- ✓ CA-02.2.5 En Windows hay que usar `cache=disk` y 2 o 3 workers (con `cache=ram`, OOM).

## US-02.3 Evaluar la fiabilidad del detector (gate)
**P0 · Hecha · S2, S4, S5**

- ✓ CA-02.3.1 Criterio fijado antes de entrenar: el nuevo mejora mAP50 y R en `test_` sin perder más de 2 pts de mAP50 en ds2 test.
- ✓ CA-02.3.2 yolo26n_ft_v2 (GT v1): mAP50 0,706 → 0,877 y R 0,757 → 0,821; ds2 0,968 → 0,963. **Cumple.**
- ✓ CA-02.3.3 Con GT v2: mAP50 0,801 frente a 0,587 del viejo, R 0,734, P@0,25 0,827 (P 0,884 en el informe del spike s).
- ✓ CA-02.3.4 Errores conocidos: cajas de más junto a la flor en i5bE (P@0,25 0,66) y aves pequeñas con poca luz en J4PK. El mAP50-95 baja, probablemente por el sesgo de las pre-etiquetas (sin comprobar).

## SP-02.3b Spike: ¿un detector más grande mejora el recall?
**P2 · Descartada (resultado negativo) · S6**

- ✓ Criterio previo: adoptar `s` si gana ≥3 pts de R y mejora la cobertura de la trayectoria, sin perder >2 pts de P ni de mAP50 en ds2.
- ✓ yolo26s_v2: R 0,678, cobertura 0,531. yolo26s_ft_v2: P 0,765, R 0,678, cobertura 0,520. La n: P 0,884, R 0,734, cobertura 0,678. **No se adopta.**
- ✓ Inferir a 960 no ayuda a la n.
- ✓ Conclusión: el cuello de botella son los datos del dominio, no la capacidad del modelo.

## US-02.4 Asociar detecciones entre frames (gate)
**P0 · Hecha · S3, S5**

- ✓ CA-02.4.1 Piloto vv (S3): barrido de 25 corridas. El mejor era OC-SORT combo (IDF1 0,80), con parámetros elegidos en la propia prueba y por tanto optimistas.
- ✓ CA-02.4.2 `test_` (S5): 5 configuraciones fijadas de antemano, sin barrido. **Elegida: ByteTrack combo** (`track_buffer` 300, `match_thresh` 0,95, `new_track_thresh` 0,5).
- ✓ CA-02.4.3 Con GT v2: R 0,723, P 0,901, MOTA 0,554, IDF1 0,489.
- ✓ CA-02.4.4 El fallo principal es la fragmentación, no las tracks espurias. J4PK es el punto débil.

## US-02.5 Unir fragmentos y exportar la trayectoria detallada
**P1 · Hecha · S5**

- ✓ CA-02.5.1 La unión de fragmentos es una derivación explícita: hueco en segundos y distancia en múltiplos de (w+h)/2. Los puentes quedan marcados `es_puente`.
- ✓ CA-02.5.2 La unión se ajustó en `train_`+`vv`. La mejor de la rejilla (0,5 s × 4) no se confirmó en `test_`, porque dibuja rectas largas en J4PK.
- ✓ CA-02.5.3 **Decidido por Yilber: unión prudente (1 s, 2,5×).** IDF1 de visita 0,667, 0 puentes falsos, 1 segmento en DoJ1, Gu30 e i5bE y 4 en J4PK.
- ✓ CA-02.5.4 Hay trayectorias CSV y PNG por video en `resultados/S5_tracking_union_test/eval_gt2/union/prudente/`.

## US-02.6 Derivados de densidad: 7 puntos
**P1 · Hecha con pendiente · S7**

Como diseñador, quiero reducir cada trayectoria a pocos puntos para dibujar una curva limpia sin perder la relación con lo observado.

- ✓ CA-02.6.1 7 puntos **en total** por video, sobre todos los segmentos unidos en orden temporal. Los huecos sin ave no reparten puntos y la curva se corta en ellos.
- ✓ CA-02.6.2 Dos métodos: (a) uniforme en el tiempo con ave y (b) salientes con Visvalingam-Whyatt.
- ✓ CA-02.6.3 Un punto que cae en un puente se mueve al observado más cercano y se marca.
- ✓ CA-02.6.4 Curva Catmull-Rom centrípeta y error de forma frente a la trayectoria densa (`resumen_7pts.csv`: error medio del 3 al 9 % de la diagonal).
- ✓ CA-02.6.5 Lámina a la misma escala y ampliada por video.
- ○ CA-02.6.6 **Pendiente:** elegir el método para E-03. La propuesta abierta es un híbrido: 7 tramos de tiempo iguales y el punto más saliente de cada uno.

## SP-02.7 Spike: profundidad monocular
**P2 · En curso → decisión "otra prueba" · S8**

Pregunta: ¿la profundidad estimada añade una tercera dimensión creíble a la trayectoria?

- ✓ CA-02.7.1 YOLO26-Depth sobre todos los frames de `test_` y Adobe. Filtro fijado de antemano: fondo < 5 % y autocorrelación a 0,5 s > 0,5. **Solo pasa i5bE (1 de 8). Se descarta como coordenada continua.**
- ✓ CA-02.7.2 Método de 7 puntos (idea de Yilber): `z_norm = (z_ave − z_flor)/(z_fondo − z_flor)` dentro del mapa de cada frame, con barra de error por TTA. En i5bE la diferencia entre puntos (~0,10) es unas 5 veces el ruido (~0,02).
- ✓ CA-02.7.3 "Estable no es correcto": f83 sale delante con el ave detrás de la flor. Se crearon variantes (`p2corr`, `p2alt_f98`, `salientes_delante`) y la escala z quedó como decisión de diseño.
- ✓ CA-02.7.4 Selector `DEPTH_BACKEND` = yolo26 / yolo26_crop / vda. Licencias revisadas: VDA Small es Apache-2.0; Base y Large son CC-BY-NC y quedan excluidas.
- ✓ CA-02.7.5 **Juez: revisión a ciegas de Yilber** (20 frames de i5bE). Acierto de signo: yolo26s 8/17, yolo26x 10/17, recorte 12/17, **VDA Small 14/17 (p = 0,013)**.
- ✓ CA-02.7.6 **Decisión: otra prueba** (ver SP-02.9). La 2D sigue siendo la base. Cautela: los 5 "detrás" están todos en los primeros 5 s.

## US-02.8 (nueva) Mejorar el recall en aves difíciles
**P1 · Pendiente**

- ○ CA-02.8.1 Lista de frames difíciles de J4PK, DoJ1 e i5bE (`revisar_gt_caja_sin_det.csv`), clasificados por causa: pequeña, poca luz, borrosa, junto a la flor.
- ○ CA-02.8.2 Se añaden clips de entrenamiento del mismo tipo (no de `test_`), etiquetados en humming.
- ○ CA-02.8.3 Criterio previo: la cobertura de la trayectoria en `test_` sube ≥5 pts y J4PK pasa de 4 a ≤2 segmentos, sin perder >2 pts de P.

## SP-02.9 (nueva) Segunda prueba de profundidad con VDA Small
**P2 · Pendiente (siguiente sprint)**

- ○ CA-02.9.1 Un video nuevo de cámara fija con flor, el ave grande en el cuadro, el fondo con textura y vista lateral, con licencia clara.
- ○ CA-02.9.2 Revisión a ciegas de 20 frames **repartidos a lo largo de todo el video** y balanceados (delante, detrás, al lado), etiquetados antes de calcular nada.
- ○ CA-02.9.3 Criterio previo: VDA Small acierta el signo con p < 0,05 y no hay confusión con el tiempo (el acierto se mantiene en la primera y la segunda mitad).
- ○ CA-02.9.4 Si pasa, se incorpora z_norm de 7 puntos a E-03 (US-03.5). Si no, la profundidad queda solo como recurso de diseño (grosor o color del trazo).

---

# 7. E-03 Investigar y caracterizar las geometrías
**Sin empezar.** El insumo ya existe: trayectorias densas (S5) y 7 puntos (S7), más la 3D exploratoria de i5bE (S8).

## US-03.1 Preparar las trayectorias para el análisis
**P1 · Pendiente (parcialmente cubierta por US-02.5)**
- ○ CA-03.1.1 Cada segmento queda clasificado como apto, condicionado o no apto. Propuesta de regla: apto si dura ≥2 s, tiene <10 % de puentes y no presenta saltos >3·(w+h)/2.
- ○ CA-03.1.2 Las trayectorias en coordenadas normalizadas usan una escala común documentada: cuadro 0-1 o caja envolvente.
- ○ CA-03.1.3 Ningún tratamiento crea patrones nuevos: se comparan contra la trayectoria densa.

## US-03.2 Visualizar las trayectorias
**P1 · Pendiente (con figuras previas en S5 y S7)**
- ○ CA-03.2.1 Hay un atlas con los 12 videos (no solo `test_`) a la misma escala, densa más 7 puntos.
- ○ CA-03.2.2 Se distingue lo observado, lo unido y lo suavizado.

## US-03.3 Caracterizar propiedades geométricas
**P1 · Pendiente**
- ○ CA-03.3.1 Descriptores por segmento: extensión, longitud, curvatura media y máxima, giros, radio de lazo y relación de aspecto.
- ○ CA-03.3.2 Se registran los que no son fiables por fps o ruido.

## US-03.4 Investigar recurrencias y simetrías (gate)
**P0 · Pendiente**
- ○ CA-03.4.1 Hay criterios escritos antes de buscar (por ejemplo, lazo alrededor de la flor, zig-zag de aproximación, cernido).
- ○ CA-03.4.2 Cada patrón candidato apunta a su video, sus frames y su segmento.
- ○ CA-03.4.3 Se separa lo observado de la interpretación.

## US-03.5 Comparar geometrías con y sin profundidad
**P2 · Bloqueada por SP-02.9**

## US-03.6 Seleccionar geometrías candidatas (gate)
**P0 · Pendiente**
- ○ CA-03.6.1 Hay un catálogo de 3 a 6 geometrías con su origen, su derivación y el formato vectorial (SVG) listo para diseño.
- ○ CA-03.6.2 Se documentan las descartadas y por qué.

---

# 8. E-04 Traducir las geometrías a propuestas visuales
**Sin empezar.** El riesgo principal del proyecto es que **US-04.1 (brief) debía ir en paralelo desde el principio y no se ha hecho**. Sin brief, E-03 no tiene un criterio de selección de diseño.

- **US-04.1 Formalizar el brief de diseño** · P1, **adelantar al siguiente sprint**. CA sin cambios respecto a la v0.4: identidad actual a conservar, atributos, soportes, criterios y confirmación del stakeholder.
- **US-04.2 Principios de abstracción visual** · P1 · Pendiente. Depende de US-03.6 y US-04.1.
- **US-04.3 Desarrollar alternativas de logotipo** (al menos 3) · P1 · Pendiente.
- **US-04.4 Legibilidad y escalabilidad preliminares** · P1 · Pendiente.
- **US-04.5 Trazabilidad entre geometría y diseño** · P1 · Pendiente. Nuevo CA-04.5.5: cada propuesta cita el CSV de 7 puntos y el video del que sale, y ese video está en la lista blanca de US-01.1.

# 9. E-05 Validar y entregar la identidad visual
**Sin empezar.** Las historias US-05.1 a US-05.4 no cambian respecto a la v0.4. Nuevos CA de entrega:

- CA-05.4.7 La entrega incluye la atribución CC BY 4.0 de ds2 si se menciona el proceso.
- CA-05.4.8 Si se entrega código, se registra la licencia AGPL-3.0 de Ultralytics.

# 10. E-06 (nueva) Reproducibilidad y organización
**Transversal · En curso (Sprint 9)**

## US-06.1 Ordenar los archivos del proyecto
- CA-06.1.1 El repositorio git `colibri-logo` tiene `notebooks/`, `datos_derivados/`, `modelos/`, `resultados/`, `entornos/` y `docs/`, con un README de navegación (D-19).
- CA-06.1.2 Nada se borra. Cada movimiento queda en `docs/mapa_rutas.csv` (origen → destino).
- CA-06.1.3 `D:\video_eval` queda documentado como zona de trabajo (entornos, `datos/`, `runs/`, `modelos/`), con las rutas en `config.yaml`.
- CA-06.1.4 Desde el repo y los 12 videos, otra persona reconstruye los datos de entrada: ds1 y ds2 por API con su clave, humming con las etiquetas congeladas (D-20).

## US-06.2 Notebooks reproducibles
- CA-06.2.1 Hay una secuencia `00` a `06` que cubre fuentes, detector, tracking y unión, 7 puntos y profundidad (05 con el kernel `vda`).
- CA-06.2.2 El código suelto pasa a celdas de notebook; las funciones compartidas viven una sola vez en `00_comun.ipynb` (`%run`), sin rutas absolutas: todas en `config.yaml`.
- CA-06.2.3 Cada notebook tiene dos modos: "desde artefactos" (por defecto) y "recalcular" (por ejemplo, `ENTRENAR = True`).
- CA-06.2.4 En modo "desde artefactos" cada notebook reproduce las cifras de su sprint dentro de una tolerancia declarada.
- CA-06.2.5 Los entornos quedan exportados en `entornos/requirements-*.txt`, con un kernel por entorno (`entornos/instalar.md`).

## US-06.3 Registro de sprints y decisiones
- CA-06.3.1 `docs/sprints.md` recoge objetivo, historias, incremento, decisiones y conclusiones de cada sprint.
- CA-06.3.2 `docs/decisiones.md` recoge cada decisión con fecha, quién decide y evidencia.
- CA-06.3.3 `resultados/VIGENTE.md` indica qué usar hoy.

---

# 11. Flujo y dependencias

```
Fuentes con rol (US-01.1) → frames y etiquetas (US-01.3, 01.5) → detector (US-02.2/02.3)
  → tracking (US-02.4) → unión prudente (US-02.5) → 7 puntos (US-02.6)
      → [SP-02.9 profundidad, opcional] → geometrías (E-03) → diseño (E-04) → entrega (E-05)
Brief de diseño (US-04.1) ───────────────────────────────────────────┘ (en paralelo, ya)
Prueba virgen (US-01.6) ─────────────────────────────── antes de las cifras finales (E-05)
```

# 12. Riesgos transversales (actualizados)

- **Sin brief de diseño:** se está optimizando técnica sin saber qué necesita el logo. Mitigación: US-04.1 en el próximo sprint.
- **Licencias de los clips de YouTube:** pueden impedir usar su trayectoria en el logo. Mitigación: lista blanca (US-01.1).
- **Prueba desgastada:** `test_` se ha mirado en 5 sprints. Mitigación: US-01.6.
- **Recall en aves difíciles:** J4PK se fragmenta en 4 segmentos. Mitigación: datos del dominio (US-02.8), no un modelo mayor (SP-02.3b).
- **Profundidad:** un solo video y una posible confusión temporal. Mitigación: SP-02.9 con ciego repartido.
- **Hardware:** la GPU de 6 GB se estrangula a 85 °C y el pagefile en C: se llena. Mitigación: entrenamientos con reanudación y Chrome cerrado para VDA.
- **Deuda de código:** unos 40 scripts con rutas fijas. Mitigación: E-06.
- Se mantienen los riesgos de la v0.4: punto corporal inestable, simplificación que oculta detalles, generalización limitada, AGPL de Ultralytics.

# 13. Fuera de alcance

Sin cambios respecto a la v0.4: reconstrucción 3D científicamente validada, afirmaciones sobre la especie, pose de alas, plataforma de generación automática y sistema de marca completo.

# 14. Regla de avance

El avance se decide por evidencia. Las preguntas 1 a 4 de la v0.4 están respondidas en positivo con la configuración vigente, con la salvedad de J4PK. Las preguntas abiertas hoy son:

5. ¿Las geometrías candidatas sirven como identidad visual? (E-03 y E-04)
6. ¿La propuesta final recibe la aprobación del stakeholder? (E-05)

Y como pregunta opcional: ¿la profundidad añade valor creíble? (SP-02.9)
