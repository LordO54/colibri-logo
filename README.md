# colibri-logo

Extrae la trayectoria de un colibrí a partir de un video. Un detector YOLO encuentra el ave, un tracker la sigue y los
fragmentos de una misma visita se unen en una trayectoria. Esa trayectoria se reduce a unos pocos puntos clave y,
opcionalmente, se les añade una profundidad relativa para obtener una curva 3D.

## Instalación

Necesitas Python 3.10 o 3.11 y Git. Una GPU NVIDIA es muy recomendable: la profundidad y el entrenamiento la necesitan.

```bash
git clone https://github.com/LordO54/colibri-logo.git
cd colibri-logo
python -m venv .venv
```

Activa el entorno: `.venv\Scripts\activate` en Windows, `source .venv/bin/activate` en Linux. Después:

```bash
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name colibri --display-name "Python (colibri)"
```

`requirements.txt` instala PyTorch para CUDA 12.4. Sin esa GPU, o con otro CUDA, instala antes el PyTorch adecuado desde
[pytorch.org](https://pytorch.org/get-started/locally/) y quita de `requirements.txt` las líneas `torch` y `torchvision`.

Si vas a modificar los notebooks y a subir cambios, ejecuta también `nbstripout --install`.

## Sacar la trayectoria de tus videos

1. Copia tus videos a la carpeta `videos/`.
2. Abre `notebooks/trayectorias.ipynb` con el kernel **Python (colibri)**, por ejemplo con `jupyter lab`.
3. Ejecuta las celdas en orden.

Los resultados quedan en `salidas/<video>/`:

| Archivo | Contenido |
|---|---|
| `trayectoria.csv`, `trayectoria.png` | Posición del ave en cada cuadro, y su dibujo sobre el video |
| `puntos.csv`, `puntos.png`, `puntos_sobre_video.png` | Los puntos clave y la curva que pasa por ellos |
| `puntos_3d.csv`, `trayectoria_3d.html`, `trayectoria_3d.jpg`, `revision_3d.jpg` | La lectura de profundidad y la curva 3D, si haces el paso opcional |

Funciona mejor con cámara fija y un ave a la vez. Usa solo videos sobre los que tengas derechos.

## Entrenar un detector con tus datos

Si el detector falla con tus videos, puedes ajustarlo con tus imágenes etiquetadas: `notebooks/entrenar_detector.ipynb`. El
notebook explica el formato de los datos y compara tu modelo con el actual. Para usarlo, indícalo en
`notebooks/trayectorias.ipynb` o en `config.yaml`.

## Actualizar el detector

El mejor detector publicado está en `modelos/` y se recibe con `git pull`. Más detalles en [`modelos/README.md`](modelos/README.md).

## Configuración

Los parámetros del seguimiento, la unión de fragmentos, los puntos clave y la profundidad están explicados en
[`config.yaml`](config.yaml). Los valores por defecto son los que mejor han funcionado.

## Organización

```
config.yaml      parámetros
notebooks/       los dos notebooks de uso
colibri/         el código; no hace falta leerlo para usar la herramienta
modelos/         detectores publicados
videos/          tus videos (fuera de git)
datasets/        tus datasets (fuera de git)
salidas/         resultados (fuera de git)
investigacion/   el proyecto original: experimentos, cifras y decisiones
```

## Licencias

El detector deriva de Ultralytics YOLO (AGPL-3.0) y se entrenó en parte con datos CC BY 4.0; ver
[`modelos/README.md`](modelos/README.md). La profundidad usa Video Depth Anything Small (Apache-2.0).
