**Decisión actualizada (test + Adobe): descartar YOLO26-Depth cuadro a cuadro como coordenada 3D de las geometrías. La 2D sigue siendo la base y US-03.5 sigue bloqueada.** Queda abierta una sola comprobación: la revisión a ciegas de i5bE, que decide si `dz` se conserva como atributo exploratorio en ese video.

**Filtro fijado de antemano** (variación del fondo < 5 % y autocorrelación de `z` a 0,5 s > 0,5), con el mismo método en todos (`c_mask_med_al`): **1 de 8 videos lo pasa (i5bE).**

| Video | Fondo (%) | Autocorr. 0,5 s | Spearman con 1/√área | Longitud 3D por z (%) | Deriva máx. (px) | Pasa |
|---|---|---|---|---|---|---|
| A129 (vv01) | 11,5 | 0,40 | −0,25 | 73 | 3,1 | no |
| A512 (vv02) | 28,4 | −0,12 | −0,10 | 88 | 37,3 | no |
| vv03 | 6,8 | −0,15 | 0,14 | 70 | 0,04 | no |
| A160 | 10,7 | 0,26 | −0,11 | 93 | 4,7 | no |
| DoJ1 | 15,1 | 0,05 | 0,27 | 66 | 5,9 | no |
| Gu30 | 3,0 | 0,02 | 0,04 | 99 | 11,6 | no |
| i5bE | 4,7 | 0,64 | 0,15 | 90 | 0,4 | **sí** |
| J4PK | 22,4 | 0,13 | −0,33 | 94 | 10,0 | no |

**Por qué se descarta**

- **La estabilidad de la cámara no era el problema.** vv03 no se mueve (deriva 0,04 px) y aun así no pasa: `z` no tiene autocorrelación. El parpadeo viene del modelo, que estima cada frame por separado.
- **El ave cambia la escala del mapa.** En A129 el fondo varía un 11,5 % con el ave en escena y un 4,5 % sin ella; en vv03, un 6,8 % frente a un 3,0 %. Esto se observa, no se ha demostrado la causa.
- **El tamaño aparente no respalda a `z`:** la correlación es negativa en 4 de 8 videos y nunca pasa de 0,27.
- **La mayor parte del recorrido 3D (66-99 %) viene de `z`**, es decir, sobre todo de ese ruido.

**Otros resultados**

- **A512 no está realmente fijo:** deriva hasta 37 px, así que su mediana de fondo y su máscara no son fiables. Además, el cielo no tiene una profundidad bien definida.
- **A160:** el tracker no mezcla aves. La trayectoria es un solo track (id 1) del ave principal. La segunda ave casi no se detecta (tracks 2 y 3, 44 de 719 frames) y la unión la descarta, así que no tiene trayectoria propia. No hay frames sin ave, y las flores se mueven con el viento.
- **Los mapas son buenos como imagen:** las siluetas de ave, flor y fondo están muy bien separadas. Lo que falla es la estabilidad en el tiempo, no la forma.
- **Detección:** estos videos se usaron para entrenar el detector, así que las cajas son optimistas. Eso no afecta a la profundidad.

**Límites:** no hay verdad de terreno 3D, los metros no son fiables y el FOV de 60° es un supuesto. El filtro usa un solo método y umbrales elegidos antes de ver estos datos.

**Qué queda**

1. Revisión a ciegas de i5bE: 20 frames en `revision_ciega_i5bE.jpg`, que Yilber rellena en `plantilla_revision_ciega.csv`. Si coincide con `dz`, se puede usar `dz` en i5bE para variar el grosor o el color del trazo, no como geometría.
2. Solo si se quiere insistir en el 3D: probar un modelo de profundidad para video, con consistencia temporal. No es un modelo de Ultralytics y no está evaluado aquí.
