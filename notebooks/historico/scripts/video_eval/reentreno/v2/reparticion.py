"""Re-particiona la versión de Roboflow humming según los manifiestos locales y arma los yaml del ajuste fino.

El split sale de manifest_reentrenamiento.csv (frames nuevos) y de seleccion_vv.csv (vv01-vv03), no de Roboflow.
Las imágenes se reducen a un lado máximo de 1280 px; las etiquetas YOLO normalizadas no cambian.
"""
import csv, re, shutil
from collections import Counter
from pathlib import Path

import cv2

V2 = Path(r"D:\video_eval\reentreno\v2")
RF = V2 / "rf_humming"
OUT = V2 / "humming"
RE = Path(r"D:\video_eval\reentreno")
OLD = Path(r"D:\video_eval\colibri_work\data")
IGNORAR = {("vv02_adobe512464022", f) for f in (95, 400, 405, 410)}  # el ave apenas asoma por el borde
MAX_SIDE = 1280
OVERSAMPLE = 3


def main():
    truth = {}
    for f, src in ((RE / "manifest_reentrenamiento.csv", "nuevo"), (RE / "seleccion_vv.csv", "vv")):
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                truth[Path(r["file"]).stem] = (r["split"], src, r["video_id"], int(r["frame"]))

    if OUT.exists():
        shutil.rmtree(OUT)
    rows, unmapped, dropped = [], [], Counter()
    for split_rf in ("train", "valid", "test"):
        for img in sorted((RF / split_rf / "images").iterdir()):
            stem = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", img.stem)
            if stem not in truth:
                if stem.startswith("vv"):
                    dropped["vv_no_seleccionado"] += 1
                else:
                    unmapped.append(img.name)
                continue
            split, src, vid, frame = truth[stem]
            if (vid, frame) in IGNORAR and split in ("train", "valid"):
                dropped["vv02_ignorar"] += 1
                continue
            (OUT / split / "images").mkdir(parents=True, exist_ok=True)
            (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
            im = cv2.imread(str(img))
            h, w = im.shape[:2]
            s = MAX_SIDE / max(h, w)
            if s < 1:
                im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            dst = OUT / split / "images" / f"{stem}.jpg"
            cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lab = RF / split_rf / "labels" / f"{img.stem}.txt"
            text = lab.read_text() if lab.exists() else ""
            (OUT / split / "labels" / f"{stem}.txt").write_text(text)
            n_box = sum(1 for l in text.splitlines() if l.strip())
            rows.append(dict(ruta=str(dst), split=split, origen=src, video_id=vid, frame=frame,
                             split_roboflow=split_rf, cajas=n_box, rf_file=img.name))

    found = {Path(r["ruta"]).stem for r in rows}
    missing = sorted(k for k, v in truth.items() if k not in found
                     and not ((v[2], v[3]) in IGNORAR and v[0] in ("train", "valid")))

    with open(V2 / "split_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)

    def listed(split):
        return [r["ruta"] for r in rows if r["split"] == split]

    old_train = sorted(str(p) for p in (OLD / "final" / "train" / "images").glob("*.jpg"))
    old_val = sorted(str(p) for p in (OLD / "final" / "valid" / "images").glob("*.jpg"))
    (V2 / "train_ft.txt").write_text("\n".join(old_train + listed("train") * OVERSAMPLE) + "\n")
    (V2 / "val_ft.txt").write_text("\n".join(old_val + listed("valid")) + "\n")
    (V2 / "test_video.txt").write_text("\n".join(listed("test")) + "\n")
    (V2 / "data_ft.yaml").write_text(
        f"train: {V2 / 'train_ft.txt'}\nval: {V2 / 'val_ft.txt'}\ntest: {V2 / 'test_video.txt'}\nnames:\n  0: hummingbird\n")
    (V2 / "data_test_video.yaml").write_text(
        f"train: {V2 / 'test_video.txt'}\nval: {V2 / 'test_video.txt'}\ntest: {V2 / 'test_video.txt'}\nnames:\n  0: hummingbird\n")
    (V2 / "data_ds2_test.yaml").write_text(
        f"path: {OLD / 'final_ds2'}\ntrain: test/images\nval: test/images\ntest: test/images\nnames:\n  0: hummingbird\n")

    c = Counter((r["split"], r["origen"]) for r in rows)
    b = Counter(r["split"] for r in rows for _ in range(r["cajas"]))
    e = Counter(r["split"] for r in rows if r["cajas"] == 0)
    print("imágenes por split/origen:", dict(c))
    print("cajas por split:", dict(b), "| imágenes vacías por split:", dict(e))
    print("descartados:", dict(dropped))
    print("no mapeables:", len(unmapped), unmapped[:10])
    print("en manifiesto pero ausentes en Roboflow:", len(missing), missing)
    print(f"train_ft: {len(old_train)} ds2+ds1 + {len(listed('train'))} humming x{OVERSAMPLE} = "
          f"{len(old_train) + OVERSAMPLE * len(listed('train'))}; val_ft: {len(old_val)} + {len(listed('valid'))}; "
          f"test_video: {len(listed('test'))}")


if __name__ == "__main__":
    main()
