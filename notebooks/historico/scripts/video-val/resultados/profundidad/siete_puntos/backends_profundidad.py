"""Interfaz común de los modelos de profundidad de SP-02.7.

Todos los backends ofrecen map(frame_idx, frame, flip=False, secondary=False) y devuelven un mapa alineado con el frame original,
de forma (H, W). Los píxeles sin dato (fuera del recorte) son NaN. El resto del pipeline no sabe qué modelo hay detrás.

- yolo26: Ultralytics sobre el frame completo. Salida: profundidad (mayor = más lejos).
- yolo26_crop: el mismo modelo sobre un recorte fijo, con el mapa recolocado en coordenadas del frame.
- vda: Video Depth Anything; lee los mapas de vda_inferir.py, que procesó el video completo. Salida: disparidad (mayor = más cerca).

z_norm = (z - z_flor) / (z_fondo - z_flor) no cambia de significado con el dominio: 0 = flor, 1 = fondo, negativo = delante.
La magnitud sí: en disparidad y en profundidad la misma escena da valores distintos, así que los valores no son intercambiables.
"""
from pathlib import Path

import cv2
import numpy as np


class Yolo26:
    variants = ("orig", "flip", "secondary")

    def __init__(self, model, secondary="yolo26m-depth.pt", imgsz=640):
        from ultralytics import YOLO
        self.m, self.s, self.imgsz = YOLO(model), YOLO(secondary), imgsz

    def _run(self, m, img):
        return m.predict(img, imgsz=self.imgsz, verbose=False)[0].depth.data.cpu().numpy().astype(np.float32)

    def map(self, frame_idx, frame, flip=False, secondary=False):
        m = self.s if secondary else self.m
        return self._run(m, frame[:, ::-1].copy())[:, ::-1] if flip else self._run(m, frame)


class Yolo26Crop(Yolo26):
    def __init__(self, model, crop, secondary="yolo26m-depth.pt", imgsz=640):
        super().__init__(model, secondary, imgsz)
        self.crop = crop  # (x0, y0, x1, y1) en píxeles, fijo para todo el video

    def map(self, frame_idx, frame, flip=False, secondary=False):
        x0, y0, x1, y1 = self.crop
        out = np.full(frame.shape[:2], np.nan, np.float32)
        out[y0:y1, x0:x1] = super().map(frame_idx, frame[y0:y1, x0:x1], flip, secondary)
        return out


class VDA:
    variants = ("orig", "flip")

    def __init__(self, cache_dir, name, shape):
        self.o = np.load(Path(cache_dir) / f"{name}_orig.npz")["maps"]
        self.f = np.load(Path(cache_dir) / f"{name}_flip.npz")["maps"]
        self.H, self.W = shape

    def map(self, frame_idx, frame, flip=False, secondary=False):
        src = self.f if flip else self.o
        return cv2.resize(src[int(frame_idx)].astype(np.float32), (self.W, self.H), interpolation=cv2.INTER_LINEAR)


def make_backend(kind, model, name, shape, crop=None, cache_dir=None):
    if kind == "yolo26":
        return Yolo26(model + ".pt")
    if kind == "yolo26_crop":
        return Yolo26Crop(model + ".pt", crop)
    if kind == "vda":
        return VDA(cache_dir, name, shape)
    raise ValueError(kind)
