**La normalización funciona: la forma en z ya no es solo parpadeo. Pero z mide si el ave está cerca del plano de la flor, y no siempre acierta cuando está tapada.**

- **Regla fijada (diferencia entre puntos mayor que el ruido entre vecinos):** la cumplen 13 de 16 combinaciones de video y método. Fallan DoJ1 con salientes, Gu30 con uniforme y A129 con uniforme.
- **Con más margen:** esa regla es poco exigente: varias la pasan por poco. Como lectura descriptiva, añadida después de ver los datos, solo superan el doble del ruido tanto entre vecinos como entre variantes del mismo frame i5bE (ambos métodos), J4PK (ambos) y Gu30 con salientes.
- **i5bE:** diferencia entre puntos 0,09-0,10 frente a ruido 0,02. El ave se mueve entre −0,09 y +0,17, es decir, muy cerca de la flor en comparación con la distancia flor-fondo.

**Advertencias**

- **Estable no significa correcto.** En i5bE f83 (punto 2 de salientes), el ave está casi oculta detrás de la flor y z_norm da −0,09, delante. Es consistente entre variantes (±0,03) y vecinos (0,04), y aun así tiene el signo contrario: el punto más "delantero" de la curva es probablemente un error. Las barras de error no detectan esto.
- **Punto 4 de i5bE (f125):** tiene mucha dispersión (±0,11); los modelos dan de 0,06 a 0,33.
- **Sin escala común con x e y:** z_norm se mide en distancia flor-fondo, y x e y en ancho y alto del cuadro. Cuánto se estira el eje z es una decisión de diseño, no una medida.
- **Referencias frágiles:** en DoJ1 la flor y el fondo están muy cerca en el mapa (denominador ~0,2), lo que amplifica el ruido. En A512 (cielo) y vv03 (negro) el fondo no tiene textura y su profundidad es poco fiable. Las regiones se dibujaron a mano.
- **El control de vecinos solo mide estabilidad a corto plazo** (±2 frames), no si el valor es verdadero. Sigue sin haber verdad de terreno 3D.
- **Detección:** los videos de Adobe se usaron para entrenar el detector.

**Conclusión para el logo:** i5bE sirve como trayectoria 3D exploratoria, revisando a ojo cada punto, porque son solo 7. El punto 2 de salientes conviene sustituirlo por un frame cercano con el ave visible. Esto no cambia la decisión de SP-02.7 sobre usar la profundidad de forma general en las geometrías.
