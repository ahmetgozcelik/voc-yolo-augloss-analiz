"""VOC 07+12 split (seed=0, katmanlı val=1000), sızıntı denetimi (dosya adı + sha1 + dHash),
sınıf/boyut dağılımı, Ultralytics yaml'ları. Kullanım (WS kökünden):
    env/.venv/Scripts/python.exe scripts/veri/voc_split_denetim.py
Test (VOC2007 test) yalnız hash + etiket istatistiği için okunur; başka işlem yapılmaz.
"""
import csv
import hashlib
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

WS = Path(__file__).resolve().parents[2]
D = WS / "data"
VOC = D / "VOC"
DEVKIT = VOC / "images/VOCdevkit"
SPL = D / "splits"
SPL.mkdir(parents=True, exist_ok=True)
SEED, NVAL = 0, 1000
NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
         "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
         "train", "tvmonitor"]


def ids(year, s):
    return (DEVKIT / f"VOC{year}/ImageSets/Main/{s}.txt").read_text().split()


# rec: (year, id, rel_img_path) ; rel = VOC'a göreli
def recs(year, subsets):
    out = []
    for s in subsets:
        out += [(year, i, f"images/{s}{year}/{i}.jpg") for i in ids(year, s)]
    return sorted(out)


pool07 = recs("2007", ["train", "val"])
pool12 = recs("2012", ["train", "val"])
test = recs("2007", ["test"])
assert len(pool07) == 5011 and len(pool12) == 11540 and len(test) == 4952

# ---- katmanlı val: 07 ve 12 havuz oranında ----
n07 = round(NVAL * len(pool07) / (len(pool07) + len(pool12)))
n12 = NVAL - n07
rng = random.Random(SEED)
val = rng.sample(pool07, n07) + rng.sample(pool12, n12)
vset = set(val)
train = [r for r in pool07 + pool12 if r not in vset]

# ---- sızıntı denetimi ----
def sha1(p):
    return hashlib.sha1(p.read_bytes()).hexdigest()


def dhash(p):
    im = Image.open(p)
    im.draft("L", (64, 64))
    a = np.asarray(im.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


allr = train + val + test
H, DH = {}, {}
for r in allr:
    p = VOC / r[2]
    H[r] = sha1(p)
    DH[r] = dhash(p)

pool = train + val
fn_pool = Counter(Path(r[2]).name for r in pool)
fn_test = Counter(Path(r[2]).name for r in test)
fn_overlap = sorted(set(fn_pool) & set(fn_test))
h_test = defaultdict(list)
for r in test:
    h_test[H[r]].append(r)
sha_overlap = [(r, h_test[H[r]]) for r in pool if H[r] in h_test]
# havuz içi (07tv vs 12tv, train vs val) sha1 kopyaları
h_pool = defaultdict(list)
for r in pool:
    h_pool[H[r]].append(r)
pool_dup_groups = [v for v in h_pool.values() if len(v) > 1]
tv_dup = [(a, b) for g in pool_dup_groups for a in g for b in g if a in vset and b not in vset]
test_dup_groups = [v for v in h_test.values() if len(v) > 1]

# dHash yakın-kopya (Hamming) havuz x test
pa = np.array([DH[r] for r in pool], dtype=np.uint64)
ta = np.array([DH[r] for r in test], dtype=np.uint64)
# popcount: bayt tablosu
TBL = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
def ham_row(v, arr):
    x = np.bitwise_xor(arr, np.uint64(v)).view(np.uint8).reshape(-1, 8)
    return TBL[x].sum(1)
near = []  # (pool_rec, test_rec, dist)
for i, r in enumerate(pool):
    d = ham_row(pa[i], ta)
    for j in np.nonzero(d <= 4)[0]:
        near.append((r, test[j], int(d[j])))


def pix_mad(a, b):
    A = np.asarray(Image.open(VOC / a[2]).convert("L").resize((64, 64)), dtype=np.float32)
    B = np.asarray(Image.open(VOC / b[2]).convert("L").resize((64, 64)), dtype=np.float32)
    return float(np.abs(A - B).mean())


near_detail = [{"pool": a[2], "test": b[2], "dhash_hamming": d, "mad64_gray": round(pix_mad(a, b), 2)}
               for a, b, d in near]
near_dist_counts = {f"<= {k}": sum(1 for x in near if x[2] <= k) for k in (0, 2, 4)}

# kesin sızıntı: sha1 eşleşmesi -> train/val'den çıkar
leak = {r for r, _ in sha_overlap}
removed = sorted(leak)
train = [r for r in train if r not in leak]
val = [r for r in val if r not in leak]

# ---- liste dosyaları ----
def abs_path(r):
    return (VOC / r[2]).as_posix()

lists = {
    "train": sorted(train),
    # val azalan sırada yazılır (tarihsel; içerik/sha256 bunu yansıtır). NOT: Ultralytics yolları kendisi
    # sıraladığı için train ve val yine aynı labels/train2007.cache'i paylaşır (hash denetimi yeniden tarar).
    "val": sorted(val, reverse=True),
    "test": sorted(test),
}
meta_lists = {}
for k, rs in lists.items():
    txt = "\n".join(abs_path(r) for r in rs) + "\n"
    (SPL / f"{k}.txt").write_text(txt, encoding="utf-8", newline="\n")
    canon = "\n".join(sorted(f"VOC{r[0]}/{r[1]}" for r in rs)) + "\n"
    meta_lists[k] = {
        "n_images": len(rs),
        "n_voc2007": sum(r[0] == "2007" for r in rs),
        "n_voc2012": sum(r[0] == "2012" for r in rs),
        "file": f"data/splits/{k}.txt",
        "sha256_file": hashlib.sha256(txt.encode()).hexdigest(),
        "sha256_canonical_ids": hashlib.sha256(canon.encode()).hexdigest(),
    }

# ---- dağılım ----
AREA_S, AREA_M = 32 ** 2, 96 ** 2
stats = {}
rows = []
label_mismatch = []
for k, rs in lists.items():
    img_c = Counter(); obj_c = Counter(); dif_c = Counter()
    size_c = defaultdict(Counter); size640 = Counter(); size_all = Counter()
    empty_lbl = 0; widths = []; heights = []; nobj_img = []
    for y, i, rel in rs:
        root = ET.parse(DEVKIT / f"VOC{y}/Annotations/{i}.xml").getroot()
        W = int(root.find("size/width").text); Hh = int(root.find("size/height").text)
        widths.append(W); heights.append(Hh)
        s640 = 640 / max(W, Hh)
        present = set(); n = 0
        for o in root.iter("object"):
            c = o.find("name").text
            if c not in NAMES:
                continue
            if int(o.find("difficult").text) == 1:
                dif_c[c] += 1
                continue
            b = o.find("bndbox")
            x0, x1, y0, y1 = (float(b.find(t).text) for t in ("xmin", "xmax", "ymin", "ymax"))
            a = (x1 - x0) * (y1 - y0)
            bucket = "small" if a < AREA_S else "medium" if a < AREA_M else "large"
            a6 = a * s640 * s640
            b6 = "small" if a6 < AREA_S else "medium" if a6 < AREA_M else "large"
            size_c[c][bucket] += 1; size_all[bucket] += 1; size640[b6] += 1
            obj_c[c] += 1; present.add(c); n += 1
        for c in present:
            img_c[c] += 1
        nobj_img.append(n)
        lbl = VOC / rel.replace("images/", "labels/", 1).replace(".jpg", ".txt")
        nl = len([l for l in lbl.read_text().splitlines() if l.strip()])
        if nl != n:
            label_mismatch.append(rel)
        if nl == 0:
            empty_lbl += 1
    for ci, c in enumerate(NAMES):
        rows.append({"split": k, "class_id": ci, "class": c, "images": img_c[c], "objects": obj_c[c],
                     "difficult_objects_dropped": dif_c[c], "small_lt32sq": size_c[c]["small"],
                     "medium_32_96": size_c[c]["medium"], "large_ge96sq": size_c[c]["large"]})
    tot = sum(obj_c.values())
    stats[k] = {
        "images": len(rs), "objects_non_difficult": tot, "difficult_dropped": sum(dif_c.values()),
        "images_with_empty_label": empty_lbl,
        "objects_per_image_mean": round(tot / len(rs), 3),
        "size_orig_px": dict(size_all), "size_orig_frac": {b: round(v / tot, 4) for b, v in size_all.items()},
        "size_at_imgsz640": dict(size640), "size_at_imgsz640_frac": {b: round(v / tot, 4) for b, v in size640.items()},
        "img_w_min_max": [min(widths), max(widths)], "img_h_min_max": [min(heights), max(heights)],
        "imbalance_objects_max_over_min": round(max(obj_c.values()) / min(obj_c.values()), 2),
        "most_objects": obj_c.most_common(1)[0], "least_objects": min(obj_c.items(), key=lambda x: x[1]),
    }

with open(D / "sinif_dagilimi.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# ---- yaml'lar ----
names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES))
hdr = "# Üretildi: scripts/veri/voc_split_denetim.py (seed=0). Sınıf sırası Ultralytics VOC.yaml ile aynı.\n"
(D / "voc_dev.yaml").write_text(
    hdr + "# GELİŞTİRME (K4 döngüsü): val = train∪val havuzundan ayrılan 1000 görüntü. Test YOK.\n"
    f"path: {D.as_posix()}\ntrain: splits/train.txt\nval: splits/val.txt\n\nnames:\n{names_yaml}\n",
    encoding="utf-8")
(D / "voc_test.yaml").write_text(
    hdr + "# KİLİTLİ — YALNIZ K5 kilit açılışında kullanılır. val = VOC2007 test (4952).\n"
    f"path: {D.as_posix()}\ntrain: splits/train.txt\nval: splits/test.txt\n\nnames:\n{names_yaml}\n",
    encoding="utf-8")

meta = {
    "seed": SEED, "method": "random.Random(0); rng.sample(sorted(VOC2007 trainval), n07) sonra "
              "rng.sample(sorted(VOC2012 trainval), n12); kayıt sıralaması (year,id,relpath)",
    "val_size": NVAL, "val_strata": {"VOC2007": n07, "VOC2012": n12},
    "pool": {"VOC2007_trainval": len(pool07), "VOC2012_trainval": len(pool12), "total": len(pool07) + len(pool12)},
    "ratio_pool_07": round(len(pool07) / (len(pool07) + len(pool12)), 4),
    "test": "VOC2007 test (resmî), KİLİTLİ",
    "removed_for_leakage": [r[2] for r in removed],
    "lists": meta_lists,
}
(SPL / "split_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

leak_rep = {
    "filename_overlap_pool_test": len(fn_overlap),
    "sha1_overlap_pool_test_images": len(sha_overlap),
    "sha1_overlap_detail": [{"pool": r[2], "test": [t[2] for t in ts]} for r, ts in sha_overlap],
    "sha1_dup_groups_within_pool": len(pool_dup_groups),
    "sha1_dup_groups_within_pool_detail": [[r[2] for r in g] for g in pool_dup_groups],
    "sha1_dup_train_val_pairs": len(tv_dup),
    "sha1_dup_groups_within_test": len(test_dup_groups),
    "dhash_near_pool_test_counts": near_dist_counts,
    "dhash_near_pool_test_detail": near_detail,
    "removed_from_train_val": [r[2] for r in removed],
}
(D / "sizinti_raporu.json").write_text(json.dumps(leak_rep, indent=2, ensure_ascii=False), encoding="utf-8")
out = {"splits": meta_lists, "stats": stats, "label_mismatch_count": len(label_mismatch),
       "leakage": {k: v for k, v in leak_rep.items() if not k.endswith("detail")}}
(D / "_split_denetim_ozet.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))
