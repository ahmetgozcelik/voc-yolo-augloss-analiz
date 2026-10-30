"""KITTI 2D Object — sürüş dizisine (drive) göre bölme, etiket denetimi, DontCare çıkarımı, sızıntı taraması.

WS kökünden:  env/.venv/Scripts/python.exe scripts/ek_deney/kitti_hazirla.py
Sahne grupları: farklı drive'lardan NCC>=0.95 görüntü çifti olan drive'lar birleştirilir (sabit-kamera kayıtları).
Girdi (data/kitti/):
  yolo/images/{train,val}/NNNNNN.png, yolo/labels/{train,val}/NNNNNN.txt   (Ultralytics kitti.zip)
  devkit/mapping/train_rand.txt, train_mapping.txt                          (resmî devkit_object.zip)
  resmi/training/label_2/NNNNNN.txt                                         (resmî data_object_label_2.zip)
Çıktı:
  data/kitti/splits/{train,val,test}.txt, data/kitti_dev.yaml, data/kitti_test.yaml (KİLİTLİ)
  data/kitti/dontcare.json, data/kitti/bolme_raporu.json
Ön-kayıt: hipotezler/ek_deney/ON-KAYIT-ek-deney.md (Veri bölümü).
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

WS = Path(__file__).resolve().parents[2]
D = WS / "data/kitti"
SINIFLAR = ["car", "van", "truck", "pedestrian", "person_sitting", "cyclist", "tram", "misc"]
RESMI_AD = {"Car": 0, "Van": 1, "Truck": 2, "Pedestrian": 3, "Person_sitting": 4, "Cyclist": 5, "Tram": 6, "Misc": 7}
HEDEF = {"test": 0.20, "val": 0.13}          # görüntü payı
ALT_PAY = {"test": 0.10, "val": 0.05, "train": 0.40}  # her sınıfın nesne payı alt sınırı
MAX_GRUP_PAY = 0.25  # tek bir sahne grubu test/val hedefinin en çok %25'i (birleşik/sabit-kamera dev grupları train'e)
SEED = 0


def goruntuler():
    out = {}
    for p in sorted((D / "yolo/images").glob("*/*.png")):
        out[int(p.stem)] = p
    return out


def drive_esle():
    rand = [int(x) for x in (D / "devkit/mapping/train_rand.txt").read_text().replace("\n", "").split(",") if x.strip()]
    mapping = (D / "devkit/mapping/train_mapping.txt").read_text().splitlines()
    return {i: mapping[r - 1].split()[1] for i, r in enumerate(rand)}


def yolo_etiket(p):
    lp = Path(str(p).replace("images", "labels")).with_suffix(".txt")
    if not lp.exists():
        return []
    return [l.split() for l in lp.read_text().splitlines() if l.strip()]


def resmi_etiket(i):
    return [l.split() for l in (D / f"resmi/training/label_2/{i:06d}.txt").read_text().splitlines() if l.strip()]


def ncc_vec(p):
    a = np.asarray(Image.open(p).convert("L").resize((96, 29), Image.BILINEAR), dtype=np.float32).ravel()
    a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


def sahne_gruplari(img, drv, esik=0.95):
    """Farklı drive'lardan NCC >= esik görüntü çifti olan drive'ları birleştirir (union-find).
    Gerekçe: 2011_09_28 sabit-kamera kayıtları aynı sahneyi farklı drive adlarıyla içerir."""
    ids = sorted(img)
    V = np.stack([ncc_vec(img[i]) for i in ids])
    ebeveyn = {}
    def bul(x):
        ebeveyn.setdefault(x, x)
        while ebeveyn[x] != x:
            ebeveyn[x] = ebeveyn[ebeveyn[x]]
            x = ebeveyn[x]
        return x
    for d in set(drv.values()):
        bul(d)
    bag = 0
    for k in range(0, len(ids), 512):
        C = V[k:k + 512] @ V.T
        ii, jj = np.nonzero(C >= esik)
        for a, b in zip(ii, jj):
            da, db = drv[ids[k + a]], drv[ids[b]]
            if da != db and bul(da) != bul(db):
                ebeveyn[bul(da)] = bul(db)
                bag += 1
    return {d: bul(d) for d in set(drv.values())}, bag, V, ids


def dhash(p, n=8):
    im = Image.open(p).convert("L").resize((n + 1, n), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def main():
    img = goruntuler()
    drv = drive_esle()
    rapor = {"goruntu": len(img), "eslemesiz": sorted(set(img) - set(drv))}

    # 1) Ultralytics bölmesi drive-ayrık mı?
    ul = {i: p.parent.name for i, p in img.items()}
    d_tr = {drv[i] for i in img if ul[i] == "train" and i in drv}
    d_va = {drv[i] for i in img if ul[i] == "val" and i in drv}
    rapor["ultralytics_bolme"] = {"train": sum(v == "train" for v in ul.values()), "val": sum(v == "val" for v in ul.values()),
                                 "ortak_drive": len(d_tr & d_va), "drive_toplam": len(d_tr | d_va)}

    # 2) Etiket denetimi (YOLO ↔ resmî, DontCare hariç) + DontCare çıkarımı (normalize xyxy)
    say_yolo, say_resmi, uyusmaz = Counter(), Counter(), []
    dontcare = {}
    sinif_say = {}
    for i, p in img.items():
        w, h = Image.open(p).size
        y = yolo_etiket(p)
        r = resmi_etiket(i)
        cy = Counter(int(l[0]) for l in y)
        cr = Counter(RESMI_AD[l[0]] for l in r if l[0] in RESMI_AD)
        say_yolo.update(cy)
        say_resmi.update(cr)
        if cy != cr:
            uyusmaz.append(i)
        dc = [[float(l[4]) / w, float(l[5]) / h, float(l[6]) / w, float(l[7]) / h] for l in r if l[0] == "DontCare"]
        if dc:
            dontcare[f"{i:06d}"] = dc
        sinif_say[i] = cy
    rapor["etiket"] = {"yolo": {SINIFLAR[k]: v for k, v in sorted(say_yolo.items())},
                       "resmi": {SINIFLAR[k]: v for k, v in sorted(say_resmi.items())},
                       "uyusmayan_goruntu": len(uyusmaz), "uyusmayan_ornek": uyusmaz[:10],
                       "dontcare_goruntu": len(dontcare), "dontcare_kutu": sum(map(len, dontcare.values()))}

    # 3) Sahne grubu düzeyinde bölme (drive'lar NCC>=0.95 ile birleştirilir; kısıtları sağlayan ilk alt-seed)
    grup, bag, V, V_ids = sahne_gruplari(img, drv)
    rapor["sahne_gruplari"] = {"drive": len(set(drv.values())), "grup": len(set(grup.values())), "birlestirme": bag,
                               "en_buyuk_grup_drive": max(Counter(grup.values()).values()),
                               "en_buyuk_grup_goruntu": max(Counter(grup[drv[i]] for i in img).values())}
    by_drive = defaultdict(list)
    for i in img:
        by_drive[grup[drv[i]]].append(i)
    drives = sorted(d for d in by_drive if not d.startswith("_eslemesiz"))
    toplam_sinif = Counter()
    for c in sinif_say.values():
        toplam_sinif.update(c)
    N = len(img)
    secim = None
    for alt in range(5000):
        rng = random.Random(SEED * 1000 + alt)
        sira = drives[:]
        rng.shuffle(sira)
        atama, sayac = {}, {"test": 0, "val": 0}
        for d in sira:
            n = len(by_drive[d])
            if n > MAX_GRUP_PAY * HEDEF["test"] * N:  # test sınırını aşan grup yalnız train'e
                atama[d] = "train"
                continue
            if n <= MAX_GRUP_PAY * HEDEF["test"] * N and sayac["test"] + n <= HEDEF["test"] * N * 1.05 and sayac["test"] < HEDEF["test"] * N:
                atama[d] = "test"; sayac["test"] += n
            elif n <= MAX_GRUP_PAY * HEDEF["val"] * N and sayac["val"] + n <= HEDEF["val"] * N * 1.05 and sayac["val"] < HEDEF["val"] * N:
                atama[d] = "val"; sayac["val"] += n
            else:
                atama[d] = "train"
        bol = {"train": [], "val": [], "test": []}
        for d, ids in by_drive.items():
            bol[atama.get(d, "train")].extend(ids)
        pay_ok = True
        for b in ("test", "val", "train"):
            if b in HEDEF and abs(len(bol[b]) / N - HEDEF[b]) > 0.02:
                pay_ok = False
            cs = Counter()
            for i in bol[b]:
                cs.update(sinif_say[i])
            if any(cs[k] < ALT_PAY[b] * toplam_sinif[k] for k in toplam_sinif):
                pay_ok = False
        if pay_ok:
            secim = (alt, bol, atama)
            break
    if secim is None:
        raise SystemExit("Ön-kayıtlı kısıtları sağlayan bölme bulunamadı (5000 alt-seed)")
    alt, bol, atama = secim
    rapor["bolme"] = {"alt_seed": alt, "grup_sayisi": {b: sum(v == b for v in atama.values()) for b in ("val", "test")} | {
        "train": len(drives) - sum(v != "train" for v in atama.values())},
        "goruntu": {b: len(v) for b, v in bol.items()},
        "sinif_payi": {b: {SINIFLAR[k]: round(sum(sinif_say[i][k] for i in v) / toplam_sinif[k], 3) for k in sorted(toplam_sinif)}
                       for b, v in bol.items()}}

    # 4) Sızıntı: bölmeler arası NCC (>=0.95) ve dHash (Hamming <= 2)
    pos = {i: k for k, i in enumerate(V_ids)}
    def ncc_es(a, b):
        C = V[[pos[i] for i in a]] @ V[[pos[i] for i in b]].T
        return int((C.max(1) >= 0.95).sum())
    rapor["sizinti_ncc_ge095"] = {"test-train": ncc_es(bol["test"], bol["train"]), "val-train": ncc_es(bol["val"], bol["train"]),
                                  "test-val": ncc_es(bol["test"], bol["val"])}
    hs = {i: dhash(img[i]) for i in img}
    def es(a, b):
        A = np.stack([hs[i] for i in a]); B = np.stack([hs[i] for i in b])
        n = 0
        for k in range(0, len(A), 256):
            x = np.unpackbits(A[k:k + 256][:, None, :] ^ B[None, :, :], axis=-1).sum(-1)
            n += int((x.min(1) <= 2).sum())
        return n
    rapor["sizinti_dhash_le2"] = {"test-train": es(bol["test"], bol["train"]), "val-train": es(bol["val"], bol["train"]),
                                  "test-val": es(bol["test"], bol["val"])}

    # 5) Yaz
    (D / "splits").mkdir(exist_ok=True)
    for b, ids in bol.items():
        (D / f"splits/{b}.txt").write_text("\n".join(img[i].as_posix() for i in sorted(ids)) + "\n")
    ad = "\n".join(f"  {k}: {v}" for k, v in enumerate(SINIFLAR))
    kok = D.as_posix()
    (WS / "data/kitti_dev.yaml").write_text(
        f"# Üretildi: scripts/ek_deney/kitti_hazirla.py (drive-ayrık, seed {SEED}, alt {alt}). Test YOK.\n"
        f"path: {kok}\ntrain: splits/train.txt\nval: splits/val.txt\n\nnames:\n{ad}\n", encoding="utf-8")
    (WS / "data/kitti_test.yaml").write_text(
        f"# KİLİTLİ TEST — tüm eğitimler bitene kadar okunmaz (ON-KAYIT-ek-deney.md).\n"
        f"path: {kok}\ntrain: splits/train.txt\nval: splits/test.txt\n\nnames:\n{ad}\n", encoding="utf-8")
    (D / "dontcare.json").write_text(json.dumps(dontcare))
    (D / "bolme_raporu.json").write_text(json.dumps(rapor, ensure_ascii=False, indent=1))
    print(json.dumps({k: rapor[k] for k in ("goruntu", "ultralytics_bolme", "sahne_gruplari", "bolme", "sizinti_ncc_ge095", "sizinti_dhash_le2")},
                     ensure_ascii=False, indent=1))
    print("eslemesiz:", len(rapor["eslemesiz"]))


if __name__ == "__main__":
    main()
