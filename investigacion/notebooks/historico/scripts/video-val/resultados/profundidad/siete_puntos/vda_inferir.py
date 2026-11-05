"""Video Depth Anything Small sobre el video completo (original y volteado). Se ejecuta en .venv_vda.

Guarda los mapas en float16 en cache_vda/<video>_{orig,flip}.npz, ya alineados con el frame original (el volteado se revierte aquí),
a la resolución de entrada: backends_profundidad.VDA los lleva al tamaño del frame. La salida es una disparidad relativa: mayor = más cerca.
Los frames se cargan al tamaño de entrada para gastar poca RAM: la memoria virtual del equipo es limitada.
"""
import gc, glob, json, os, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(r"D:\video_eval\modelos\Video-Depth-Anything")
SP = Path(r"D:\video_eval\reentreno\v2\profundidad\siete_puntos")
CACHE = SP / "cache_vda"; CACHE.mkdir(exist_ok=True)
VIDEOS = {"i5bE": glob.glob(r"C:\Users\ACER\Documents\logo\video-val\test_*i5bEfvb5mxg*")[0]}
SIZES = tuple(int(s) for s in os.environ.get("VDA_SIZES", "448,392,336,280").split(","))   # 518 es la del README; sin xFormers no cabe en 6 GB
sys.path.insert(0, str(REPO)); os.chdir(REPO)
from video_depth_anything.video_depth import VideoDepthAnything


def read_all(path, short):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS); out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        s = short / min(f.shape[:2])
        out.append(cv2.cvtColor(cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out), fps


def main():
    m = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    m.load_state_dict(torch.load(REPO / "checkpoints" / "video_depth_anything_vits.pth", map_location="cpu", weights_only=True), strict=True)
    m = m.to("cuda").eval()
    used = {}
    for name, path in VIDEOS.items():
        for tag in ("orig", "flip"):
            out = CACHE / f"{name}_{tag}.npz"
            if out.exists(): continue
            for size in SIZES:
                try:
                    t0 = time.time(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                    frames, fps = read_all(path, size)
                    if tag == "flip": frames = frames[:, :, ::-1].copy()
                    d, _ = m.infer_video_depth(frames, fps, input_size=size, device="cuda", fp32=False)
                    break
                except (torch.OutOfMemoryError, MemoryError) as e:
                    print(name, tag, "sin memoria a", size, type(e).__name__, flush=True); frames = d = None; gc.collect(); torch.cuda.empty_cache()
            else:
                raise RuntimeError("no cabe a ninguna resolución")
            d = np.asarray(d, dtype=np.float32)
            if tag == "flip": d = d[:, :, ::-1]
            np.savez_compressed(out, maps=d.astype(np.float16))
            used[f"{name}_{tag}"] = size
            print(name, tag, d.shape, f"input {size}", f"{len(d) / (time.time() - t0):.1f} fps", f"pico GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", "rango", float(d.min()), float(d.max()), flush=True)
            frames = d = None; gc.collect(); torch.cuda.empty_cache()
    (CACHE / "info.json").write_text(json.dumps(dict(modelo="vda-small", input_size=used, salida="disparidad relativa, mayor = más cerca"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
