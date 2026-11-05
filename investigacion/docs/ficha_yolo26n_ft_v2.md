# yolo26n_ft_v2 (detector vigente, D-08)

- **Base:** `yolo26n_ds2_ds1/best.pt` (S2).
- **Datos:** `listas_v1/train_ft.txt` = ds2 + ds1 (re-partido) + humming v1 train ×3; valid = ds2+ds1 valid + humming valid. Los 4 `test_` nunca entran (D-07). Lo genera el notebook 01.
- **Entrenamiento:** 20 épocas, paciencia 8, imgsz 640, SGD, lr0 0,002, warmup 1, batch 0,70, cache disk, 3 workers, semilla 0 (`args.yaml`, `results.csv`). Notebook 02, sección 3.
- **Gate (S4):** en `test_` con GT v1, mAP50 0,706 → 0,877 y R 0,757 → 0,821; en ds2 test −0,5 puntos de mAP50. Detalle por video en `resultados/S4_reentreno_v2/comparacion.csv`.
- **Debilidad conocida (US-02.8):** recall en aves pequeñas o de espaldas (J4PK).
- **Uso:** tracking con ByteTrack combo (D-09) y unión prudente (D-10), notebook 03.
