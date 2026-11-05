"""De un video a su trayectoria, sus puntos clave y su lectura de profundidad. Lo usa notebooks/trayectorias.ipynb."""
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CARPETAS, CFG, DETECTOR, RAIZ
from .profundidad import background, control_ruido, figura_3d, leer_puntos_z, make_backend, mapas_vda, read_frame, rect_px, tira_puntos
from .puntos import METHODS, lamina_fondo, lamina_trazo, puntos_clave, tabla_puntos
from .seguimiento import bridges, draw, escribir_tracker, scene_cuts, track_video, union_tuned

EXT_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
METODOS = {"salientes": "b_salientes", "uniforme": "a_uniforme_tiempo"}
VDA_URL = "https://github.com/DepthAnything/Video-Depth-Anything"
VDA_CKPT_URL = "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth"


def _nombre(stem, maxlen=48):
    return re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")[:maxlen].rstrip("_")


def descargar_urls(urls, carpeta=None):
    """Baja cada URL con yt-dlp (solo la imagen: no hace falta ffmpeg). Lo ya descargado no se repite."""
    carpeta = Path(carpeta or CARPETAS["videos"]); carpeta.mkdir(parents=True, exist_ok=True)
    for u in urls or []:
        subprocess.run([sys.executable, "-m", "yt_dlp", "--no-playlist", "--restrict-filenames", "-f", "bv*[ext=mp4]/b[ext=mp4]/b",
                        "-o", str(carpeta / "%(title).60s_%(id)s.%(ext)s"), u], check=True)


def leer_videos(fuente=None, archivos=None) -> dict:
    """{nombre: datos del video} de una carpeta (sin subcarpetas), de un archivo, o de `archivos` dentro de la carpeta."""
    fuente = Path(fuente or CARPETAS["videos"])
    if not fuente.exists():
        raise FileNotFoundError(f"No existe {fuente}")
    if archivos:
        rutas = [fuente / a for a in archivos]
    elif fuente.is_file():
        rutas = [fuente]
    else:
        rutas = sorted(p for p in fuente.iterdir() if p.suffix.lower() in EXT_VIDEO)
    if not rutas:
        raise FileNotFoundError(f"No hay videos en {fuente}. Copia ahí tus archivos ({', '.join(sorted(EXT_VIDEO))}) o usa URLS.")
    out = {}
    for p in rutas:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            print(f"✗ no se puede abrir {p.name}; se omite"); continue
        nombre = _nombre(p.stem)
        if nombre in out:
            nombre = f"{nombre}_{hashlib.sha1(p.name.encode()).hexdigest()[:6]}"
        out[nombre] = dict(path=p, ancho=int(cap.get(3)), alto=int(cap.get(4)), fps=cap.get(cv2.CAP_PROP_FPS), frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
    return out


def carpeta_salida(nombre):
    return CARPETAS["salidas"] / nombre


def _nombre_detector(detector):
    """modelos/x/best.pt -> x; salidas/entrenamiento/x/weights/best.pt -> x."""
    d = Path(detector).parent
    return d.parent.name if d.name == "weights" else d.name


def procesar_video(nombre, v, detector=None, metodo=None, n_puntos=None):
    """Detecta y sigue al ave, une los fragmentos y calcula los puntos clave. Devuelve una fila de resumen.
    Las detecciones de cada detector se guardan y no se recalculan."""
    detector = Path(detector or DETECTOR); d = carpeta_salida(nombre); d.mkdir(parents=True, exist_ok=True)
    metodo = METODOS[metodo or CFG["puntos"]["metodo"]]; n_puntos = n_puntos or CFG["puntos"]["n"]
    seg = CFG["seguimiento"]; tracker = seg["tracker"]; W, H, fps = v["ancho"], v["alto"], v["fps"]
    det = d / f"detecciones_{_nombre_detector(detector)}.csv"
    if det.exists():
        tr = pd.read_csv(det); tr.attrs.update(json.loads(det.with_suffix(".json").read_text(encoding="utf-8")))
    else:
        yaml_tracker = escribir_tracker(tracker, seg["parametros"], CARPETAS["salidas"] / "_cache" / f"tracker_{tracker}.yaml")
        tr = track_video(detector, v["path"], tracker, yaml_tracker, conf=seg["confianza"], imgsz=seg["tamano_imagen"])
        tr.to_csv(det, index=False); det.with_suffix(".json").write_text(json.dumps(tr.attrs, indent=2), encoding="utf-8")
    u = CFG["union"]
    traj = union_tuned(tr, u["hueco_max_s"], u["distancia_max"], scene_cuts(v["path"], CARPETAS["salidas"] / "_cache" / "cortes")).copy()
    traj["es_puente"] = traj.interpolado.astype(bool)
    fila = dict(video=nombre, detector=_nombre_detector(detector), duracion_s=round(v["frames"] / fps, 1), segmentos=int(traj.segment_id.nunique()) if len(traj) else 0,
                tiempo_con_ave_s=round(len(traj) / fps, 1), pct_con_ave=round(100 * len(traj) / max(v["frames"], 1), 1),
                confianza_mediana=round(float(tr.conf.median()), 3) if len(tr) else None, listo=False)
    (d / "trayectoria.csv").unlink(missing_ok=True)      # lo que queda en la carpeta corresponde a este detector
    if len(traj) < n_puntos:
        print(f"✗ {nombre}: solo {len(traj)} cuadros con ave. ¿Sale un colibrí? Prueba con otro video o detector.")
        return fila
    traj = traj.sort_values("frame").reset_index(drop=True); traj.to_csv(d / "trayectoria.csv", index=False)
    br, largos = bridges(traj, W, H); fila.update(puentes=len(br), puentes_largos=largos)
    draw(v["path"], traj, pd.DataFrame(dict(boxes=[])), f"{nombre} | {fila['segmentos']} segmentos | {fila['pct_con_ave']} % de cuadros con ave", d / "trayectoria.png")
    sp = puntos_clave(traj, fps, n_puntos); r = sp["metodos"][metodo]
    tabla_puntos(r, W, H).to_csv(d / "puntos.csv", index=False)
    cap = cv2.VideoCapture(str(v["path"])); cap.set(cv2.CAP_PROP_POS_FRAMES, int(traj.frame.median())); _, fr = cap.read(); cap.release()
    lamina_fondo(sp, r, cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), W, H, f"{nombre} | {METHODS[metodo]} | {len(r['gaps'])} huecos | error medio {r['e_mean']:.0f} px", d / "puntos_sobre_video.png")
    lamina_trazo(r, d / "puntos.png")
    fila.update(error_medio_px=round(r["e_mean"], 1), listo=True)
    return fila


def ver_cuadro(nombre, v, divisiones=10):
    """Muestra el cuadro central de la trayectoria con una rejilla de 0 a 1, para escribir las regiones."""
    f = carpeta_salida(nombre) / "trayectoria.csv"
    i = int(pd.read_csv(f).frame.median()) if f.exists() else v["frames"] // 2
    cap = cv2.VideoCapture(str(v["path"])); fr = read_frame(cap, i); cap.release()
    fig, ax = plt.subplots(figsize=(10, 10 * v["alto"] / v["ancho"])); ax.imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), extent=(0, 1, 1, 0), aspect="auto")
    ax.set_xticks(np.linspace(0, 1, divisiones + 1)); ax.set_yticks(np.linspace(0, 1, divisiones + 1)); ax.grid(color="white", alpha=0.6, lw=0.8)
    ax.set_title(nombre, fontsize=10); plt.show(); plt.close(fig)


def preparar_modelo_profundidad():
    """Descarga Video Depth Anything Small la primera vez que hace falta."""
    prof = CFG["profundidad"]; repo, ckpt = RAIZ / prof["repo"], RAIZ / prof["checkpoint"]
    if not repo.exists():
        print("Descargando el código de Video Depth Anything...", flush=True)
        subprocess.run(["git", "clone", "-q", VDA_URL, str(repo)], check=True)
        subprocess.run(["git", "checkout", "-q", prof["commit"]], cwd=repo, check=True)
    if not ckpt.exists():
        print("Descargando el modelo de profundidad (unos 110 MB)...", flush=True)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(VDA_CKPT_URL, ckpt)
    return repo, ckpt


def calcular_profundidad(videos):
    """Calcula los mapas de profundidad que falten. videos: {nombre: datos del video}."""
    repo, ckpt = preparar_modelo_profundidad()
    cache = CARPETAS["salidas"] / "_cache" / "profundidad"
    mapas_vda({n: v["path"] for n, v in videos.items()}, cache, CFG["profundidad"]["tamanos"], repo, ckpt)
    return cache


def procesar_3d(nombre, v, regiones, cache):
    """Lee la profundidad en los puntos del video y dibuja la trayectoria 3D. regiones: {"flor": rectángulo, "fondo": rectángulo}."""
    d = carpeta_salida(nombre); t = pd.read_csv(d / "trayectoria.csv").set_index("frame")
    cap = cv2.VideoCapture(str(v["path"])); W, H = v["ancho"], v["alto"]
    bg = background(cap, v["frames"]); be = make_backend("vda", "vda-small", nombre, (H, W), cache_dir=cache)
    rr, ff = regiones["flor"], regiones["fondo"]
    D = leer_puntos_z(pd.read_csv(d / "puntos.csv"), t, cap, be, bg, rect_px(rr, W, H), rect_px(ff, W, H), W, H)
    D.drop(columns=["box"]).to_csv(d / "puntos_3d.csv", index=False)
    figura_3d(D, nombre, rr, "flor", H / W, d / "trayectoria_3d")
    tira_puntos(D, cap, rr, W, H, d / "revision_3d.jpg")
    cap.release()
    return dict(video=nombre, **control_ruido(D))
