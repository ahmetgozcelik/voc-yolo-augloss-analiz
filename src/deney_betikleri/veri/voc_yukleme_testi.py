"""Ultralytics voc_dev.yaml'ı tanıyor mu? train/val etiketlerini YOLODataset ile okur (eğitim yok).
voc_test.yaml yalnız ayrıştırılır (check_det_dataset), test etiketleri OKUNMAZ.
Kullanım: env/.venv/Scripts/python.exe scripts/veri/voc_yukleme_testi.py
"""
from pathlib import Path
import numpy as np
import ultralytics
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.dataset import YOLODataset

if __name__ == "__main__":
    D = Path(__file__).resolve().parents[2] / "data"
    dev = check_det_dataset(str(D / "voc_dev.yaml"))
    print("ultralytics", ultralytics.__version__, "nc", dev["nc"], "names[0,19]", dev["names"][0], dev["names"][19])
    for k in ("train", "val"):
        ds = YOLODataset(img_path=dev[k], data=dev, task="detect", augment=False, imgsz=640, cache=False)
        n_img = len(ds.labels); n_obj = int(sum(len(l["cls"]) for l in ds.labels))
        cls = np.concatenate([l["cls"].ravel() for l in ds.labels]).astype(int)
        print(f"{k}: images={n_img} objects={n_obj} n_classes_seen={len(set(cls))} empty={sum(len(l['cls'])==0 for l in ds.labels)}")
    tst = check_det_dataset(str(D / "voc_test.yaml"))
    print("voc_test.yaml parse ok; val ->", Path(tst["val"]).name, "(etiketler okunmadı, KİLİTLİ)")
