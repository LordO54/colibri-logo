# Compara TODOS los frames regenerados con los originales de S4 y, en los que superan dHash 2, prueba indices vecinos.
import sys, re
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf8")
Z = Path(r"D:\video_eval")
F = pd.read_csv(Z / "colibri-logo/datos_derivados/fuentes.csv").set_index("video_id")
def dh(fr):
    g = cv2.cvtColor(cv2.resize(fr, (9, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return (g[:, 1:] > g[:, :-1]).flatten()
rows = []
for ver, raiz in (("v1", Z / "reentreno/v2/humming"), ("v2", Z / "reentreno/v2/eval_gt2/humming")):
    for sp in ("train", "valid", "test"):
        for n in sorted((Z / f"datos/frames/humming_{ver}/{sp}/images").glob("*.jpg")):
            o = cv2.imread(str(raiz / sp / "images" / n.name)); a = cv2.imread(str(n))
            if o is None or a is None:
                rows.append(dict(ver=ver, sp=sp, f=n.name, estado="no abre")); continue
            if o.shape != a.shape:
                rows.append(dict(ver=ver, sp=sp, f=n.name, estado=f"forma {o.shape} vs {a.shape}")); continue
            rows.append(dict(ver=ver, sp=sp, f=n.name, estado="ok", dif=float(cv2.absdiff(o, a).mean()), dhash=int((dh(o) ^ dh(a)).sum())))
R = pd.DataFrame(rows); print(R.groupby(["ver", "estado"]).size())
ok = R[R.estado == "ok"].copy(); ok["vid"] = ok.f.str.replace(r"_f\d{5}\.jpg$", "", regex=True)
print(ok.groupby("ver").agg(n=("f", "size"), dif_med=("dif", "median"), dif_max=("dif", "max"), dh_le2=("dhash", lambda s: int((s <= 2).sum())), dh_max=("dhash", "max")))
print(R[R.estado != "ok"].to_string())
mal = ok[(ok.ver == "v1") & (ok.dhash > 2)]
print(mal.groupby("vid").size())
# prueba de desplazamiento en los que fallan: frame original frente a i-3..i+3 del video
for r in mal.itertuples():
    m = re.search(r"_f(\d{5})\.jpg$", r.f); i = int(m[1]); v = F.loc[r.vid]
    p = Z / "datos/fuentes/videos" / v.carpeta / v.archivo
    o = cv2.imread(str(Z / "reentreno/v2/humming" / r.sp / "images" / r.f)); h, w = o.shape[:2]
    cap = cv2.VideoCapture(str(p)); k = 0; res = {}
    while cap.grab():
        if abs(k - i) <= 3:
            _, fr = cap.retrieve(); fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
            res[k - i] = (round(float(cv2.absdiff(o, fr).mean()), 2), int((dh(o) ^ dh(fr)).sum()))
        k += 1
        if k > i + 3: break
    cap.release()
    print(r.f[:30], "f", i, "dif", round(r.dif, 2), "dhash", r.dhash, "| vecinos (dif, dhash):", res)
ok.to_csv(Z / "runs/01/frames_regenerados_vs_originales.csv", index=False)
