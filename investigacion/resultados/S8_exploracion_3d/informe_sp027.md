# SP-02.7: profundidad monocular

**Decisión: realizar otra prueba, acotada. No se incorpora todavía a las geometrías; la 2D sigue siendo la base.**

La profundidad funciona como imagen, pero como coordenada solo es creíble en uno de los cuatro videos.

**Lo que funciona**

- **Alineación (CA-02.7.2):** el mapa tiene la forma del frame y la caja cae sobre la silueta del ave.
- **i5bE, el video de la monarda:** es coherente. El fondo es estable (variación del 3 al 5 %) y `z` varía de forma lenta (correlación consigo mismo a 0,5 s de 0,64). En los tramos revisados a ojo, `dz` = 0,000 m con el pico en la flor, −0,015 m delante de ella y +0,088 m detrás. El orden es el esperado.

**Lo que no funciona**

- **Parpadeo con cámara fija (CA-02.7.4):** en J4PK el fondo varía un 22 % y en DoJ1 un 15 %. Frente a eso, los cambios de `z` del ave no se pueden interpretar.
- **Gu30:** el fondo es estable (3 %), pero `z` del ave oscila entre 0,8 y 1,5 m en décimas de segundo mientras revolotea casi en el sitio. No es un movimiento físico creíble. Correlación a 0,5 s: 0,02.
- **Validación con el tamaño (CA-02.7.3):** débil e inconsistente. El método elegido por la regla, `c_mask_med_al`, da rho de Spearman 0,27 (DoJ1), 0,04 (Gu30), 0,15 (i5bE) y −0,33 (J4PK). Ningún método es positivo en los cuatro videos.
- **3D frente a 2D (CA-02.7.5):** entre el 66 % y el 99 % de la longitud del recorrido 3D viene de `z`, así que en la mayoría de los videos el 3D dibuja sobre todo el ruido de `z`. La vista frontal coincide con la 2D por construcción.
- **Oclusión:** en f80 el ave está casi oculta tras la flor y `dz` = −0,19 m, el signo contrario: el modelo lee la flor, no el ave.

**Límites de esta evaluación**

- No hay verdad de terreno 3D; todo es indirecto.
- La escala en metros no es fiable y el FOV de 60° es un supuesto.
- Los tramos de i5bE se etiquetaron a ojo viendo los valores de `dz`, con riesgo de sesgo de confirmación: hay que repetirlo a ciegas.
- Es un solo video a favor. No basta para generalizar.

**Otra prueba propuesta**

1. Aceptar un video solo si pasa un filtro fijado de antemano: variación del fondo menor del 5 % y correlación de `z` a 0,5 s mayor de 0,5. Hoy solo lo pasa i5bE.
2. Probar 3 a 5 videos más con flor y cámara fija, con los tramos delante/detrás etiquetados por Yilber sin ver `dz`.
3. Probar un modelo mayor (por ejemplo `yolo26l-depth`) y, si sigue parpadeando, un modelo de profundidad para video con consistencia temporal.

Mientras tanto, `dz` se puede usar en i5bE como atributo exploratorio del trazo (grosor o color), no como geometría. US-03.5 sigue bloqueada hasta tener un resultado favorable.


## Cifras

```json
{
  "z_elegido": "c_mask_med_al",
  "rho_tamano": {
    "DoJ1": 0.272,
    "Gu30": 0.041,
    "i5bE": 0.15,
    "J4PK": -0.332
  },
  "estabilidad": {
    "DoJ1": {
      "cv_celda_pct": 15.09000015258789,
      "alpha_std_pct": 13.73,
      "ratio_ave_fondo": 2.82
    },
    "Gu30": {
      "cv_celda_pct": 2.990000009536743,
      "alpha_std_pct": 2.79,
      "ratio_ave_fondo": 17.83
    },
    "i5bE": {
      "cv_celda_pct": 4.670000076293945,
      "alpha_std_pct": 3.92,
      "ratio_ave_fondo": 4.8
    },
    "J4PK": {
      "cv_celda_pct": 22.3799991607666,
      "alpha_std_pct": 19.58,
      "ratio_ave_fondo": 5.0
    }
  },
  "filtro": {
    "DoJ1": {
      "variacion_fondo_pct": 15.09000015258789,
      "autocorr_05s": 0.05,
      "rho_tamano": 0.27,
      "pct_long_z": 65.62,
      "deriva_px_max": 5.92,
      "pasa_filtro": false
    },
    "Gu30": {
      "variacion_fondo_pct": 2.990000009536743,
      "autocorr_05s": 0.02,
      "rho_tamano": 0.04,
      "pct_long_z": 99.19,
      "deriva_px_max": 11.57,
      "pasa_filtro": false
    },
    "i5bE": {
      "variacion_fondo_pct": 4.670000076293945,
      "autocorr_05s": 0.64,
      "rho_tamano": 0.15,
      "pct_long_z": 89.53,
      "deriva_px_max": 0.39,
      "pasa_filtro": true
    },
    "J4PK": {
      "variacion_fondo_pct": 22.3799991607666,
      "autocorr_05s": 0.13,
      "rho_tamano": -0.33,
      "pct_long_z": 93.64,
      "deriva_px_max": 10.03,
      "pasa_filtro": false
    }
  },
  "comparacion": {
    "DoJ1": {
      "pct_long_z": 65.62,
      "snr_z": 7.41,
      "ratio_z_xy": 0.37,
      "pct_delante": 0.0
    },
    "Gu30": {
      "pct_long_z": 99.19,
      "snr_z": 4.94,
      "ratio_z_xy": 4.09,
      "pct_delante": 4.47
    },
    "i5bE": {
      "pct_long_z": 89.53,
      "snr_z": 10.31,
      "ratio_z_xy": 0.65,
      "pct_delante": 28.86
    },
    "J4PK": {
      "pct_long_z": 93.64,
      "snr_z": 11.42,
      "ratio_z_xy": 0.9,
      "pct_delante": 1.7
    }
  },
  "oclusion_i5bE": {
    "dz_detras": 0.087593674659729,
    "dz_delante": -0.01536107063293457,
    "dz_f80_oculto": -0.19113922119140625,
    "salto_fondo_i5bE_m": 0.0729571282863617,
    "p_unilateral": 5.652865293383797e-12
  }
}
```
