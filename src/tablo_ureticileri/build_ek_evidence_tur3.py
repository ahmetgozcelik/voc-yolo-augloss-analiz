#!/usr/bin/env python
"""Hakem tur 3 ek kanitlari (B1, B2, B3, B5, B6, O4, O5, O6, O10): evidence.json'a E3574'ten itibaren `ek3.*` kayitlari EKLER.

Mevcut kayitlara (E0001..E3573; ozellikle build_ek_evidence.py'nin E2571-E3560'i ve elle eklenen E3561-E3573) DOKUNMAZ.
Idempotent: onceki calistirmanin `ek3.*` kayitlari silinip yeniden uretilir. YENI egitim/degerlendirme KOSUMU YOKTUR:
degerler WS/sonuclar/ek_deney/*, WS/runs/*/{metrics,val_preds}.json, WS/data/* dosyalarindan okunur ya da ayni dosyalardan
formulle turetilir (CPU; GPU kullanilmaz). Birim: puan = x100. Hicbir sayi elle yazilmaz.
Calistirma (WS venv; pycocotools gerekir): env/.venv/Scripts/python.exe evidence/scripts/build_ek_evidence_tur3.py
Onkosul: build_ek_evidence.py (E2571+) ve (varsa) ek3_kumeli_p.py ciktisi (evidence/ek3_kumeli_p_yolo11s_H01.json).
"""
import json
import math
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
MW = Path(__file__).resolve().parents[2]
ROOT = MW.parents[1]
WS = ROOT / "deney_workspace" / "2-1-enhancing-real-time-object-detection-on-pascal-v"
EK = WS / "sonuclar" / "ek_deney"
EVID = MW / "evidence" / "evidence.json"
START = 3574
sys.path.insert(0, str(WS / "scripts" / "ek_deney"))
import kitti_degerlendir as kd  # noqa: E402  (CPU: pycocotools)


def J(p):
    return json.load(open(p, encoding="utf-8"))


d = J(EVID)
base = [r for r in d["kayitlar"] if not r["anahtar"].startswith("ek3.")]
assert max(int(r["id"][1:]) for r in base) == START - 1, "beklenen son mevcut kayit E3573"
BASE = {r["anahtar"]: r for r in base}
R, BY = [], {}


def ev(k):
    return BASE[k]["deger"]


def eid(k):
    return BASE[k]["id"]


def add(anahtar, metrik, yontem, veri_seti, deger, birim, dosya, konum, etiket, kaynak="yazilim", **kw):
    assert anahtar not in BY and anahtar not in BASE, f"yinelenen anahtar {anahtar}"
    if isinstance(deger, float):
        assert deger == deger and abs(deger) != float("inf"), anahtar
    rec = {"id": f"E{START + len(R):04d}", "anahtar": anahtar, "metrik": metrik, "yontem": yontem, "veri_seti": veri_seti,
           "deger": deger, "birim": birim, "kaynak": kaynak, "kaynak_dosya": dosya, "kaynak_konum": konum,
           "atif": None, "durum": "gecerli", "etiket": etiket}
    rec.update(kw)
    R.append(rec)
    BY[anahtar] = rec
    return rec


MODELS = ["yolo26n", "yolo11s"]
MAD = {"yolo26n": "YOLO26n", "yolo11s": "YOLO11s"}
CARDS = ["H01", "H02", "H03", "H04", "H05", "H06", "H08"]
IYI = ["H01", "H03", "H04", "H06", "H08"]
SINIF = ["car", "van", "truck", "pedestrian", "person_sitting", "cyclist", "tram", "misc"]
N = 3
DS_TEST = "KITTI test (1504 goruntu, 13 sahne grubu; DontCare=COCO crowd)"
DS_VAL = "KITTI val (987 goruntu, 10 sahne grubu)"
F_TA, F_VA = "WS/sonuclar/ek_deney/test-analiz.json", "WS/sonuclar/ek_deney/val-analiz.json"
VA = J(EK / "val-analiz.json")
TA = J(EK / "test-analiz.json")
RUN = VA["kosumlar"]
TEST = {k: J(EK / "test" / f"{rid}.json") for k, rid in RUN.items()}
EPOCH = {k: VA["tani"][k]["en_iyi_epoch"] for k in RUN}
y_ = lambda m, c: f"{c}/{MAD[m]}"  # noqa: E731
tf = lambda k: f"WS/sonuclar/ek_deney/test/{RUN[k]}.json"  # noqa: E731

# ======================================================================================= B1: kumeli GA kayitli mi + sayimlar
ET = "tur3_b1"
bg = [k for k in BASE if k.startswith("ek.test.") and k.endswith(".boot_ga_lo")]
assert len(bg) == 14 and all(k.replace("_lo", "_hi") in BASE for k in bg)
add("ek3.b1.kumeli_ga_kayitli_sayisi", "kumeli_bootstrap_ga_kayitli_kart_sayisi", "tum kartlar/iki model", DS_TEST, len(bg), "adet", F_TA,
    "count(ek.test.<m>.<kart>.boot_ga_lo/hi) = 2 model x 7 kart; E2581-E2817 araligindaki boot_ga_* kayitlari", ET, turetme="count(ek.test.*.boot_ga_lo)")
for m in MODELS:
    dis_b, dis_t = [], []
    for c in CARDS:
        k = f"ek.test.{m}.{c}"
        if ev(f"{k}.boot_ga_lo") > 0 or ev(f"{k}.boot_ga_hi") < 0:
            dis_b.append(c)
        if ev(f"{k}.ga_lo") > 0 or ev(f"{k}.ga_hi") < 0:
            dis_t.append(c)
    tm = f"ek.test.{m}.<kart>.boot_ga_lo/hi (sifiri disliyorsa say)"
    add(f"ek3.b1.{m}.kumeli_ga_sifiri_dislayan_sayi", "kumeli_ga_sifiri_dislayan_kart_sayisi", f"7 kart/{MAD[m]}", DS_TEST, len(dis_b), "adet", F_TA,
        f"{m}.<kart>.kumeli_boot_ga95: alt>0 veya ust<0 olan kart sayisi (7 karttan)", ET, turetme=tm)
    add(f"ek3.b1.{m}.kumeli_ga_sifiri_dislayan_kartlar", "kumeli_ga_sifiri_dislayan_kartlar", f"7 kart/{MAD[m]}", DS_TEST, ",".join(dis_b) or "yok", "", F_TA,
        "yukaridaki sayimin kart listesi (hepsinde alt sinir > 0)", ET, turetme=tm)
    tm2 = f"ek.test.{m}.<kart>.ga_lo/hi (sifiri disliyorsa say)"
    add(f"ek3.b1.{m}.t_ga_sifiri_dislayan_sayi", "tohum_t_ga_sifiri_dislayan_kart_sayisi", f"7 kart/{MAD[m]}", DS_TEST, len(dis_t), "adet", F_TA,
        f"{m}.<kart>.ga95 (tohum-duzeyi t, df=2): sifiri dislayan kart sayisi", ET, turetme=tm2)
    add(f"ek3.b1.{m}.t_ga_sifiri_dislayan_kartlar", "tohum_t_ga_sifiri_dislayan_kartlar", f"7 kart/{MAD[m]}", DS_TEST, ",".join(dis_t) or "yok", "", F_TA,
        "yukaridaki sayimin kart listesi", ET, turetme=tm2)
assert BY["ek3.b1.yolo11s.kumeli_ga_sifiri_dislayan_kartlar"]["deger"] == "H03,H04,H05,H06,H08"
assert BY["ek3.b1.yolo26n.kumeli_ga_sifiri_dislayan_sayi"]["deger"] == 0

# ======================================================================================= B2 (i): DontCare'siz test Δ (tohum-esli)
ET = "tur3_b2"
for m in MODELS:
    for c in CARDS:
        dl = []
        for i in range(N):
            v = (TEST[f"{m}/{c}/s{i}"]["dontcaresiz"]["map50_95"] - TEST[f"{m}/B/s{i}"]["dontcaresiz"]["map50_95"]) * 100
            dl.append(v)
            add(f"ek3.test.{m}.{c}.dcsiz.delta_s{i}", "delta_map50_95_dontcaresiz_seed", y_(m, c), DS_TEST.replace("DontCare=COCO crowd", "DontCare'siz"), v, "puan",
                tf(f"{m}/{c}/s{i}"), f"dontcaresiz.map50_95 x100 (best.pt) - ayni seed B ({RUN[f'{m}/B/s{i}']}.json dontcaresiz.map50_95 x100)", ET, seed=i, n=N,
                turetme="kart - B, ayni seed; betimsel (Holm/karar yok)")
        add(f"ek3.test.{m}.{c}.dcsiz.delta_ort", "delta_map50_95_dontcaresiz_ort", y_(m, c), DS_TEST.replace("DontCare=COCO crowd", "DontCare'siz"), st.mean(dl), "puan",
            f"WS/sonuclar/ek_deney/test/<run_id>.json", "mean(dcsiz.delta_s0..2)", ET, n=N, turetme=f"mean(ek3.test.{m}.{c}.dcsiz.delta_s0..2)")
        add(f"ek3.test.{m}.{c}.dcsiz.delta_sd", "delta_map50_95_dontcaresiz_sd", y_(m, c), DS_TEST.replace("DontCare=COCO crowd", "DontCare'siz"), st.stdev(dl), "puan",
            f"WS/sonuclar/ek_deney/test/<run_id>.json", "stdev(dcsiz.delta_s0..2), ddof=1", ET, n=N, turetme=f"stdev(ek3.test.{m}.{c}.dcsiz.delta_s0..2)")

# B2 (v4): dusuk nesne sayili siniflar -> test siniflari basina GT nesne sayisi (YOLO etiketleri; DontCare yok)
tl = (WS / "data/kitti/splits/test.txt").read_text().split()
cnt = Counter()
for p in tl:
    lp = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
    for ln in lp.read_text().split("\n"):
        if ln.strip():
            cnt[int(ln.split()[0])] += 1
assert sum(cnt.values()) == TEST["yolo26n/B/s0"]["dontcare"]["n_gt"] == 6212, sum(cnt.values())
for k, s in enumerate(SINIF):
    add(f"ek3.b2.test_nesne.{s}", "test_gt_nesne_sayisi_sinif", f"KITTI test/{s}", DS_TEST, cnt[k], "adet", "WS/data/kitti/yolo/labels/<test goruntuleri>.txt",
        f"sinif {k} ({s}) satir sayisi; toplam {sum(cnt.values())} = test/<B>.json dontcare.n_gt", ET, turetme="count(label satirlari, sinif=k; 1504 test goruntusu)")

# B2 (v5 m.3): best-last val farki (egitim-ici en iyi val - son epoch val; Ultralytics val, DontCare'siz)
for m in MODELS:
    fb, fk = [], []
    for c in ["B"] + CARDS:
        for i in range(N):
            t = VA["tani"][f"{m}/{c}/s{i}"]
            f = 100 * (t["egitim_ici_en_iyi"] - t["son_epoch"])
            (fb if c == "B" else fk).append(f)
            add(f"ek3.val.{m}.{c}.s{i}.best_eksi_last", "val_best_eksi_son_epoch", y_(m, c), DS_VAL + " (egitim-ici, Ultralytics val)", f, "puan", F_VA,
                f"tani.{m}/{c}/s{i}: 100 x (egitim_ici_en_iyi - son_epoch); ON-KAYIT v5 m.3", ET, seed=i, run_id=RUN[f"{m}/{c}/s{i}"],
                turetme="100*(egitim_ici_en_iyi - son_epoch)")
    add(f"ek3.val.{m}.B.best_eksi_last_ort", "val_best_eksi_son_epoch_ort", f"B/{MAD[m]}", DS_VAL, st.mean(fb), "puan", F_VA, "mean(B s0..2 best_eksi_last)", ET,
        turetme=f"mean(ek3.val.{m}.B.s0..2.best_eksi_last)", n=N)
    add(f"ek3.val.{m}.kartlar.best_eksi_last_ort", "val_best_eksi_son_epoch_ort", f"7 kart/{MAD[m]}", DS_VAL, st.mean(fk), "puan", F_VA,
        "mean(7 kart x 3 seed = 21 kosum)", ET, turetme=f"mean(ek3.val.{m}.<H01..H08>.s*.best_eksi_last)", n=21)

# ======================================================================================= B3: epoch-1 sayimlari + train1000 tanisi
ET = "tur3_b3"
for m in MODELS:
    for i in range(N):
        arms = [c for c in ["B"] + CARDS if EPOCH[f"{m}/{c}/s{i}"] == 1]
        tm = f"en_iyi_epoch==1 olan kollar (ek.epoch.{m}.<kol>.s{i})"
        add(f"ek3.b3.{m}.s{i}.n_ep1", "en_iyi_epoch_eq_1_sayisi_seed", f"8 kol/{MAD[m]} seed {i}", "KITTI val", len(arms), "adet", "WS/runs/*/metrics.json", "count(en_iyi_epoch==1), seed basina 8 koldan",
            ET, seed=i, turetme=tm)
        add(f"ek3.b3.{m}.s{i}.ep1_kollar", "en_iyi_epoch_eq_1_kollar", f"8 kol/{MAD[m]} seed {i}", "KITTI val", ",".join(arms) or "yok", "", "WS/runs/*/metrics.json",
            "en_iyi_epoch==1 olan kollarin listesi (B + 7 kart)", ET, seed=i, turetme=tm)
    tot = sum(BY[f"ek3.b3.{m}.s{i}.n_ep1"]["deger"] for i in range(N))
    assert tot == ev(f"ek.epoch.{m}.tum.n_ep1"), (m, tot)
    # iki epoch-1 agirligi arasindaki kiyas: B ve kart ayni seed'de ikisi de epoch 1
    both, eitherC = [], []
    for i in range(N):
        for c in CARDS:
            eb, ec = EPOCH[f"{m}/B/s{i}"], EPOCH[f"{m}/{c}/s{i}"]
            if eb == 1 and ec == 1:
                both.append((c, i))
            if eb == 1 or ec == 1:
                eitherC.append((c, i))
    tm = f"count/list over (kart in 7, seed in 0..2): en_iyi_epoch(B,s)==1 and en_iyi_epoch(kart,s)==1 ({m})"
    add(f"ek3.b3.{m}.kiyas_iki_ep1_sayisi", "iki_ep1_agirligi_kiyas_sayisi", f"7 kart x 3 seed/{MAD[m]}", "KITTI val/test", len(both), "adet", "WS/runs/*/metrics.json",
        "toplam 21 (kart, seed) kiyasindan: B ve kartin best.pt'si ikisi de epoch 1", ET, turetme=tm, n_ref=21)
    add(f"ek3.b3.{m}.kiyas_iki_ep1_liste", "iki_ep1_agirligi_kiyas_listesi", f"7 kart x 3 seed/{MAD[m]}", "KITTI val/test", ",".join(f"{c}/s{i}" for c, i in both) or "yok", "",
        "WS/runs/*/metrics.json", "kart/seed listesi", ET, turetme=tm)
    add(f"ek3.b3.{m}.kiyas_en_az_biri_ep1_sayisi", "en_az_bir_tarafi_ep1_kiyas_sayisi", f"7 kart x 3 seed/{MAD[m]}", "KITTI val/test", len(eitherC), "adet", "WS/runs/*/metrics.json",
        "21 kiyastan: B ya da kartin best.pt'si (en az biri) epoch 1", ET, turetme=tm.replace("and", "or"), n_ref=21)
    if both:
        dl = [ev(f"ek.test.{m}.{c}.delta_s{i}") for c, i in both]
        ids = ",".join(eid(f"ek.test.{m}.{c}.delta_s{i}") for c, i in both)
        add(f"ek3.b3.{m}.kiyas_iki_ep1_delta_ort", "iki_ep1_kiyas_test_delta_ort", f"{len(both)} kiyas/{MAD[m]}", DS_TEST, st.mean(dl), "puan", F_TA,
            f"mean(ek.test.{m}.<kart>.delta_s<seed>) kiyas listesi icin; kaynak id'ler: {ids}", ET, turetme="mean(ilgili test delta_seed degerleri)", n=len(both))
        add(f"ek3.b3.{m}.kiyas_iki_ep1_delta_min", "iki_ep1_kiyas_test_delta_min", f"{len(both)} kiyas/{MAD[m]}", DS_TEST, min(dl), "puan", F_TA,
            f"min(ilgili test delta_seed); id'ler: {ids}", ET, turetme="min", n=len(both))
        add(f"ek3.b3.{m}.kiyas_iki_ep1_delta_max", "iki_ep1_kiyas_test_delta_max", f"{len(both)} kiyas/{MAD[m]}", DS_TEST, max(dl), "puan", F_TA,
            f"max(ilgili test delta_seed); id'ler: {ids}", ET, turetme="max", n=len(both))
        # tohum-2 kartlarin bir epoch'luk (kart ep1, B ep1) farklari tek tek: id baglantisi
        for c, i in both:
            add(f"ek3.b3.{m}.kiyas_iki_ep1.{c}_s{i}.delta", "iki_ep1_kiyas_test_delta", y_(m, c), DS_TEST, ev(f"ek.test.{m}.{c}.delta_s{i}"), "puan", F_TA,
                f"{m}.{c}.delta_seed[{i}] (kopya; kaynak {eid(f'ek.test.{m}.{c}.delta_s{i}')}); B ve kart best.pt = epoch 1", ET, seed=i, turetme=f"= ek.test.{m}.{c}.delta_s{i}")
assert BY["ek3.b3.yolo11s.s2.n_ep1"]["deger"] == 6 and BY["ek3.b3.yolo11s.s2.ep1_kollar"]["deger"] == "B,H01,H02,H03,H04,H06"
assert BY["ek3.b3.yolo26n.s0.n_ep1"]["deger"] == 0

# train1000 tanisi: model/koşum README + degerlendirme betiginden (sayilar regex ile okunur)
RD = (EK / "tani" / "README.md").read_text(encoding="utf-8")
SC = (EK / "tani" / "train1000_degerlendir.py").read_text(encoding="utf-8")
rid_t = re.search(r"runs/(\d{8}-\d{6}_ek_(yolo\w+?)_baseline_s(\d))/", SC)
assert rid_t, "train1000 run_id bulunamadi"
RID_T, M_T, S_T = rid_t.group(1), rid_t.group(2), int(rid_t.group(3))
assert M_T == "yolo26n" and S_T == 0 and RUN["yolo26n/B/s0"] == RID_T
mm = re.search(r"best\.pt mAP50-95 ([0-9.]+) \(mAP50 ([0-9.]+)\) \| last\.pt ([0-9.]+) \(([0-9.]+)\)\.\s*Aynı ağırlıklar val'de (0\.\d+) / ≈(0\.\d+)", RD)
assert mm, "README train1000 satiri ayristirilamadi"
b95, b50, l95, l50, vb, vl = (float(x) for x in mm.groups())
sc = re.search(r"Sınıf başı \(best\): (.+)", RD).group(1)
cls = re.findall(r"(car|van|truck|ped|sit|cyc|tram|misc) \.(\d+)", sc)
assert [c for c, _ in cls] == ["car", "van", "truck", "ped", "sit", "cyc", "tram", "misc"], cls
ap_cls = {c: float("0." + v) for c, v in cls}
F_TR = "WS/sonuclar/ek_deney/tani/README.md"
DS_T1 = "KITTI train alt kumesi (1000 goruntu; train1000.txt, seed 0); Ultralytics val, DontCare'siz"
KON = "README madde 1: 'YOLO26n B s0 (20261019-124541) agirliklari ... train1000'"
y_t = "B/YOLO26n seed 0"
add("ek3.b3.train1000.model", "train1000_tani_modeli", y_t, DS_T1, MAD[M_T], "", "WS/sonuclar/ek_deney/tani/train1000_degerlendir.py", f"R=\"runs/{RID_T}/train/weights/\" (run_id -> model ve tohum); README madde 1 ile ayni",
    ET, run_id=RID_T, seed=S_T, uyari="HAKEM TUR 3 B3 metni bu tanıyı 'YOLO11s'e ait gibi okutabilir; kaynakta tanı YOLO26n B s0'dır (YOLO11s için train1000 tanısı YOKTUR).")
add("ek3.b3.train1000.n_goruntu", "train1000_goruntu_sayisi", y_t, DS_T1, len((EK / "tani" / "train1000.txt").read_text().split()), "goruntu", "WS/sonuclar/ek_deney/tani/train1000.txt", "satir sayisi", ET)
add("ek3.b3.train1000.best_map50_95", "train1000_map50_95_best", y_t, DS_T1, b95 * 100, "puan", F_TR, f"{KON}: best.pt mAP50-95 {b95} (x100)", ET, run_id=RID_T)
add("ek3.b3.train1000.best_map50", "train1000_map50_best", y_t, DS_T1, b50 * 100, "puan", F_TR, f"best.pt mAP50 {b50} (x100)", ET, run_id=RID_T)
add("ek3.b3.train1000.last_map50_95", "train1000_map50_95_last", y_t, DS_T1, l95 * 100, "puan", F_TR, f"last.pt mAP50-95 {l95} (x100)", ET, run_id=RID_T)
add("ek3.b3.train1000.last_map50", "train1000_map50_last", y_t, DS_T1, l50 * 100, "puan", F_TR, f"last.pt mAP50 {l50} (x100)", ET, run_id=RID_T)
add("ek3.b3.train1000.val_best", "val_map50_95_ayni_agirlik_best", y_t, DS_VAL + "; Ultralytics val, DontCare'siz", vb * 100, "puan", F_TR,
    f"README: 'Aynı ağırlıklar val'de {vb} / ≈{vl}' (ilk deger = best.pt); ayni sayi val-analiz.json tani.yolo26n/B/s0.yeniden_deg = {VA['tani']['yolo26n/B/s0']['yeniden_deg']}", ET, run_id=RID_T)
assert abs(vb - VA["tani"]["yolo26n/B/s0"]["yeniden_deg"]) < 5e-5
add("ek3.b3.train1000.val_last_readme", "val_map50_95_ayni_agirlik_last", y_t, DS_VAL, vl * 100, "puan", F_TR,
    f"README: '≈{vl}' (yaklasik; last.pt val). Kaynakta olcum yontemi (DontCare kurali) belirtilmemis.", ET, run_id=RID_T,
    uyari="README'de '≈' ile yaklasik yazilmis; kesin deger val-last-tani.json tamamlaninca (su an baska bir surec yaziyor) ayri kayitlanmali. Makalede '≈' korunmali.")
for c, nm in (("car", "car"), ("ped", "pedestrian"), ("cyc", "cyclist")):
    add(f"ek3.b3.train1000.best_ap5095_{nm}", "train1000_ap50_95_sinif_best", y_t, DS_T1, ap_cls[c] * 100, "puan", F_TR, f"'Sınıf başı (best)': {c} .{str(ap_cls[c])[2:]} (x100)", ET, run_id=RID_T)
mt = J(WS / "runs" / RID_T / "metrics.json")
add("ek3.b3.val_car_ap5095_B_s0", "val_ap50_95_car_best", y_t, DS_VAL + " (egitim sonu best.pt; Ultralytics val, DontCare'siz)", mt["alt_grup"]["ap5095_car"] * 100, "puan",
    f"WS/runs/{RID_T}/metrics.json", "alt_grup.ap5095_car x100", ET, run_id=RID_T,
    not_="Hakem tur 3 B3'teki 'car AP50-95 0.571' bu VAL degeridir (train1000 car AP50-95 = ek3.b3.train1000.best_ap5095_car, 0.704); ON-KAYIT v5 m.1 ile ayni.")

# ======================================================================================= B5: H05 geç tepe
ET = "tur3_b5"
n_gec_tum = 0
for m in MODELS:
    gec = 0
    for i in range(N):
        e5, eb = EPOCH[f"{m}/H05/s{i}"], EPOCH[f"{m}/B/s{i}"]
        gec += e5 > eb
        add(f"ek3.b5.{m}.s{i}.h05_eksi_B_en_iyi_epoch", "h05_en_iyi_epoch_eksi_B", f"H05-B/{MAD[m]}", "KITTI val (egitim-ici en iyi epoch)", e5 - eb, "epoch", "WS/runs/*/metrics.json",
            f"ek.epoch.{m}.H05.s{i} ({e5}) - ek.epoch.{m}.B.s{i} ({eb})", ET, seed=i, turetme=f"ek.epoch.{m}.H05.s{i} - ek.epoch.{m}.B.s{i}")
    n_gec_tum += gec
    add(f"ek3.b5.{m}.h05_gec_sayisi", "h05_baseline_den_gec_tepe_seed_sayisi", f"H05 vs B/{MAD[m]}", "KITTI val", gec, "adet", "WS/runs/*/metrics.json",
        "count(seed: en_iyi_epoch(H05) > en_iyi_epoch(B)), 3 tohumdan", ET, turetme=f"count(ek3.b5.{m}.s*.h05_eksi_B_en_iyi_epoch > 0)", n_ref=N)
    for kk, ad in (("delta_ort", "ort"), ("delta_sd", "sd")):
        add(f"ek3.b5.{m}.h05_egri_ep11_90_{ad}", f"val_egri_ep11_90_delta_{ad}", y_("%s" % m, "H05"), DS_VAL, ev(f"ek.val.{m}.H05.egri_ep11_90.{kk}"), "puan", F_VA,
            f"ikincil_betimsel.egri_ep11_90.{m}.H05.{ad} (kopya; kaynak {eid(f'ek.val.{m}.H05.egri_ep11_90.{kk}')})", ET, n=N, turetme=f"= ek.val.{m}.H05.egri_ep11_90.{kk}")
    add(f"ek3.b5.{m}.h05_egri_ep11_90_seed_pozitif", "val_egri_ep11_90_delta_pozitif_seed_sayisi", y_(m, "H05"), DS_VAL,
        sum(ev(f"ek.val.{m}.H05.egri_ep11_90.delta_s{i}") > 0 for i in range(N)), "adet", F_VA, f"count(ek.val.{m}.H05.egri_ep11_90.delta_s0..2 > 0)", ET, n=N,
        turetme="count(delta_seed>0)")
add("ek3.b5.tum.h05_gec_sayisi", "h05_baseline_den_gec_tepe_seed_sayisi", "H05 vs B/iki model", "KITTI val", n_gec_tum, "adet", "WS/runs/*/metrics.json",
    "ek3.b5.yolo26n.h05_gec_sayisi + ek3.b5.yolo11s.h05_gec_sayisi (6 tohumdan)", ET, turetme="sum", n_ref=6)
assert n_gec_tum == 6

# ======================================================================================= B6: baseline mutlak degerler
ET = "tur3_b6"
liste_v = (WS / "data/kitti/splits/val.txt").read_text().split()
dcj = json.loads((WS / "data/kitti/dontcare.json").read_text())
for m in MODELS:
    vdc, vdcs, tb, tl_, tbs, tls = [], [], [], [], [], []
    cb, cl = {s: [] for s in SINIF}, {s: [] for s in SINIF}
    c3b, c7b = [], []
    for i in range(N):
        rid = RUN[f"{m}/B/s{i}"]
        preds = json.loads((WS / "runs" / rid / "val_preds.json").read_text())
        e = kd.degerlendir(preds, liste_v, dcj, 8)
        mt = J(WS / "runs" / rid / "metrics.json")
        t = TEST[f"{m}/B/s{i}"]
        vdc.append(e["map50_95"] * 100)
        vdcs.append(mt["val_map50_95"] * 100)
        tb.append(t["dontcare"]["map50_95"] * 100); tl_.append(t["last"]["dontcare"]["map50_95"] * 100)
        tbs.append(t["dontcaresiz"]["map50_95"] * 100); tls.append(t["last"]["dontcaresiz"]["map50_95"] * 100)
        c3b.append(t["dontcare"]["map50_95_3sinif"] * 100); c7b.append(t["dontcare"]["map50_95_7sinif"] * 100)
        for k, s in enumerate(SINIF):
            cb[s].append(t["dontcare"]["ap50_95_sinif"][k] * 100)
            cl[s].append(t["last"]["dontcare"]["ap50_95_sinif"][k] * 100)
        add(f"ek3.b6.{m}.B.val_dc_s{i}", "val_map50_95_mutlak_dontcare", f"B/{MAD[m]}", DS_VAL + "; DontCare kurali", e["map50_95"] * 100, "puan",
            f"WS/runs/{rid}/val_preds.json", "kitti_degerlendir.degerlendir(val_preds.json, splits/val.txt, dontcare.json) x100 (analiz_ek.py val ile ayni yol; CPU)", ET, seed=i, run_id=rid,
            turetme="scripts/ek_deney/kitti_degerlendir.degerlendir (yeniden hesap, kosum yok)")
        add(f"ek3.b6.{m}.B.val_dcsiz_s{i}", "val_map50_95_mutlak_dontcaresiz", f"B/{MAD[m]}", DS_VAL + "; DontCare'siz (Ultralytics val)", mt["val_map50_95"] * 100, "puan",
            f"WS/runs/{rid}/metrics.json", "val_map50_95 x100 (egitim sonu best.pt, Ultralytics val)", ET, seed=i, run_id=rid)
    ds_t = f"KITTI test (1504 goruntu), {MAD[m]}"
    for ad, lst, dsx, ks in (("val_dc", vdc, DS_VAL + "; DontCare kurali", "val_dc_s0..2"), ("val_dcsiz", vdcs, DS_VAL + "; DontCare'siz", "val_dcsiz_s0..2"),
                             ("test_dcsiz_best", tbs, ds_t + "; DontCare'siz; best.pt", None), ("test_dcsiz_last", tls, ds_t + "; DontCare'siz; last.pt", None),
                             ("test_3sinif_best", c3b, ds_t + "; DontCare kurali; car/ped/cyc", None), ("test_7sinif_best", c7b, ds_t + "; DontCare kurali; person_sitting haric 7 sinif", None)):
        src = "WS/runs/*/val_preds.json|metrics.json" if ad.startswith("val") else "WS/sonuclar/ek_deney/test/<B run_id>.json"
        key = {"test_dcsiz_best": "dontcaresiz.map50_95", "test_dcsiz_last": "last.dontcaresiz.map50_95",
               "test_3sinif_best": "dontcare.map50_95_3sinif", "test_7sinif_best": "dontcare.map50_95_7sinif"}.get(ad, ks)
        for suf, f in (("ort", st.mean), ("sd", st.stdev)):
            add(f"ek3.b6.{m}.B.{ad}_{suf}", f"B_mutlak_{ad}_{suf}", f"B/{MAD[m]}", dsx, f(lst), "puan", src, f"{'mean' if suf == 'ort' else 'stdev (ddof=1)'}(s0..2) of {key} x100", ET, n=N,
                turetme=f"{'mean' if suf == 'ort' else 'stdev'}(B s0..2)")
    for s in ("car", "pedestrian", "cyclist") + tuple(x for x in SINIF if x not in ("car", "pedestrian", "cyclist")):
        for yk, lst, ad in (("best", cb[s], "best.pt"), ("last", cl[s], "last.pt")):
            kon = ("dontcare.ap50_95_sinif" if yk == "best" else "last.dontcare.ap50_95_sinif") + f"[{SINIF.index(s)}] ({s}) x100"
            add(f"ek3.b6.{m}.B.test_ap5095_{s}_{yk}_ort", f"B_test_ap50_95_sinif_{yk}_ort", f"B/{MAD[m]}", ds_t + f"; DontCare kurali; {ad}", st.mean(lst), "puan",
                "WS/sonuclar/ek_deney/test/<B run_id>.json", f"mean(s0..2) of {kon}", ET, n=N, turetme="mean(B s0..2)")
            add(f"ek3.b6.{m}.B.test_ap5095_{s}_{yk}_sd", f"B_test_ap50_95_sinif_{yk}_sd", f"B/{MAD[m]}", ds_t + f"; DontCare kurali; {ad}", st.stdev(lst), "puan",
                "WS/sonuclar/ek_deney/test/<B run_id>.json", f"stdev(s0..2) of {kon}", ET, n=N, turetme="stdev(B s0..2)")
    # kaynak tutarlilik: DontCare kurali test best/last ort = ek.artefakt.* (mevcut)
    assert abs(st.mean(tb) - ev(f"ek.artefakt.{m}.B.best_ort")) < 1e-9 and abs(st.mean(tl_) - ev(f"ek.artefakt.{m}.B.last_ort")) < 1e-9
    add(f"ek3.b6.{m}.B.test_dc_best_ort_ref", "B_test_mutlak_dontcare_best_ort_referans", f"B/{MAD[m]}", ds_t + "; DontCare kurali; best.pt", st.mean(tb), "puan", "WS/sonuclar/ek_deney/test/<B run_id>.json",
        f"= {eid(f'ek.artefakt.{m}.B.best_ort')} (ek.artefakt.{m}.B.best_ort; ayni deger, tek yerde toplamak icin kopya)", ET, n=N, turetme=f"= ek.artefakt.{m}.B.best_ort")
    add(f"ek3.b6.{m}.B.test_dc_last_ort_ref", "B_test_mutlak_dontcare_last_ort_referans", f"B/{MAD[m]}", ds_t + "; DontCare kurali; last.pt", st.mean(tl_), "puan", "WS/sonuclar/ek_deney/test/<B run_id>.json",
        f"= {eid(f'ek.artefakt.{m}.B.last_ort')} (ek.artefakt.{m}.B.last_ort)", ET, n=N, turetme=f"= ek.artefakt.{m}.B.last_ort")

# B6: dis makullugu referansi (YALNIZ kesifsel makullük; kiyas degil)
F_HD = "WS/hipotezler/ek_deney/hakem-tasarim-ek-deney-tur4.md"
HD = (WS / "hipotezler" / "ek_deney" / "hakem-tasarim-ek-deney-tur4.md").read_text(encoding="utf-8")
ln = next(l for l in HD.split("\n") if "dronefreak/kitti-yolo11n" in l)
ln_no = HD.split("\n").index(ln) + 1
mm = re.search(r"1280 px ile eğitilmiş bir YOLO11n, val'de mAP50-95 ([0-9.]+) ve mAP50 ([0-9.]+) raporluyor \(car ([0-9.]+), pedestrian ([0-9.]+), tram ([0-9.]+), person_sitting ([0-9.]+), misc ([0-9.]+)\)", ln)
assert mm, "dronefreak satiri ayristirilamadi"
dv = [float(x) for x in mm.groups()]
pm = re.search(r"patience (\d+) ile (\d+)\. epoch'ta durmuş", ln)
DR = dict(kaynak="dis_referans", atif=None)
KOSUL = ("Koşullar (hakem raporundaki aktarım; model kartı doğrudan açılmadı): YOLO11n, 1280 px, sahne-ayrık 3DOP bölmesi 3712/3769 (val), DontCare işleme biçimi bilinmiyor, "
         "patience 20 / 43. epoch'ta durmuş; hakemli değil. Bizim kurulumla (640 px, kendi sahne-grubu bölmesi, DontCare=COCO crowd) AYNI KOŞULDA ÜRETİLMEDİ: kıyas değil, yalnız büyüklük mertebesi.")
for nm, v_, met in zip(["map50_95", "map50", "car_ap50_95", "pedestrian_ap50_95", "tram_ap50_95", "person_sitting_ap50_95", "misc_ap50_95"], dv,
                       ["val_map50_95", "val_map50", "val_ap50_95_car", "val_ap50_95_pedestrian", "val_ap50_95_tram", "val_ap50_95_person_sitting", "val_ap50_95_misc"]):
    add(f"ek3.b6.dis.dronefreak.{nm}", met, "dronefreak/kitti-yolo11n (HF model karti; YOLO11n, 1280 px)", "KITTI val (3DOP 3712/3769 bolmesi; dis kaynak)", v_ * 100, "puan", F_HD,
        f"satir {ln_no}: 'Dış referans' maddesi; model karti degeri x100 (kartta {v_})", "kesifsel_makullük", kaynak="dis_referans", amac="kesifsel makullük (kıyas değil)", not_=KOSUL)
add("ek3.b6.dis.dronefreak.imgsz", "egitim_cozunurlugu", "dronefreak/kitti-yolo11n", "KITTI (dis kaynak)", 1280, "px", F_HD, f"satir {ln_no}: '1280 px ile eğitilmiş bir YOLO11n'", "kesifsel_makullük", kaynak="dis_referans", not_=KOSUL)
add("ek3.b6.dis.dronefreak.bolme", "bolme", "dronefreak/kitti-yolo11n", "KITTI (dis kaynak)", "3712/3769 (3DOP train/val)", "", F_HD, f"satir {ln_no}: 'Sahne-ayrık 3DOP bölmesinde (3 712/3 769)'", "kesifsel_makullük", kaynak="dis_referans", not_=KOSUL)
add("ek3.b6.dis.dronefreak.durus_epoch", "erken_durdurma_epoch", "dronefreak/kitti-yolo11n", "KITTI (dis kaynak)", int(pm.group(2)), "epoch", F_HD,
    f"satir {ln_no}: 'patience {pm.group(1)} ile {pm.group(2)}. epoch'ta durmuş (en iyi epoch ≈{int(pm.group(2)) - int(pm.group(1))})'", "kesifsel_makullük", kaynak="dis_referans", not_=KOSUL)

# ======================================================================================= Ö4: istatistik
ET = "tur3_o4"
pk = {(m, c): ev(f"ek.test.{m}.{c}.p_tek") for m in MODELS for c in IYI}


def holm_adj(ps):
    k = len(ps)
    sira = np.argsort(ps)
    adj = np.empty(k)
    mx = 0.0
    for r, i in enumerate(sira):
        mx = max(mx, (k - r) * ps[i])
        adj[i] = min(1.0, mx)
    return adj


keys10 = list(pk)
a10 = holm_adj(np.array([pk[k] for k in keys10]))
h10 = float(a10[keys10.index(("yolo11s", "H01"))])
assert abs(h10 - 10 * pk[("yolo11s", "H01")]) < 1e-12
add("ek3.o4.yolo11s.H01.p_holm10", "p_holm_iki_model_birlikte_10_test", y_("yolo11s", "H01"), DS_TEST, h10, "p", F_TA,
    "Holm-10: iyilesme ailesi (H01,H03,H04,H06,H08) x iki model = 10 tek yonlu p (ek.test.<m>.<kart>.p_tek); H01 YOLO11s en kucuk p -> 10 x p_tek", ET,
    turetme="holm_adim_azalan(10 p; ek.test.{yolo26n,yolo11s}.{H01,H03,H04,H06,H08}.p_tek)[YOLO11s H01]", n=N, kaynak_idler=[eid(f"ek.test.{m}.{c}.p_tek") for (m, c) in keys10])
add("ek3.o4.yolo11s.H01.p_holm10_gecer", "p_holm10_lt_010", y_("yolo11s", "H01"), DS_TEST, "evet" if h10 < 0.10 else "hayir", "", F_TA, f"p_holm10 = {h10:.4f} < 0.10 (KITTI kabul esigi)", ET,
    turetme="ek3.o4.yolo11s.H01.p_holm10 < 0.10")
# duyarlilik: 14 testin tamami tek ailede (ON-KAYITLI DEGIL)
pk14 = {(m, c): ev(f"ek.test.{m}.{c}.p_tek") for m in MODELS for c in CARDS}
k14 = list(pk14)
a14 = holm_adj(np.array([pk14[k] for k in k14]))
h14 = float(a14[k14.index(("yolo11s", "H01"))])
add("ek3.o4.yolo11s.H01.p_holm14_duyarlilik", "p_holm_14_test_duyarlilik", y_("yolo11s", "H01"), DS_TEST, h14, "p", F_TA,
    "Holm-14: 7 kart x 2 model tek aile (p_tek; yonlu H02/H05 icin Δ<0 yonu degil kayitli p_tek)", ET,
    turetme="holm(14 p_tek)[YOLO11s H01] = 14 x p_tek", n=N,
    not_="ON-KAYITLI DEGIL (ön-kayıt iki aile, model başına Holm-5/Holm-2). Dürüstlük için duyarlılık: 14 testlik tek aile p<0.10'u sağlamaz.")
d_h01 = [ev(f"ek.test.yolo11s.H01.delta_s{i}") for i in range(N)]
w = stats.wilcoxon(d_h01, alternative="greater", method="exact")
assert abs(w.pvalue - 1 / 2 ** N) < 1e-12
add("ek3.o4.yolo11s.H01.wilcoxon_tek_yonlu_p", "wilcoxon_isaretli_sira_tek_yonlu_p", y_("yolo11s", "H01"), DS_TEST, float(w.pvalue), "p", F_TA,
    f"scipy.stats.wilcoxon({[round(x, 3) for x in d_h01]}, alternative='greater', method='exact') ; n=3 delta_seed tumu pozitif", ET,
    turetme="tum isaretler pozitif: p = 1/2^n = 1/8", n=N, kaynak_idler=[eid(f"ek.test.yolo11s.H01.delta_s{i}") for i in range(N)])
add("ek3.o4.n3.en_kucuk_p_tek_yonlu", "n3_isaret_cevirme_en_kucuk_tek_yonlu_p", "n=3 tohum", "-", 1 / 2 ** N, "p", "(teorik)", "min p = 1/2^n (isaret-cevirme/Wilcoxon, n=3, tek yonlu)", ET,
    turetme="1/2**3", not_="Parametrik olmayan hicbir test n=3'te p<0.10 veremez (min 0.125).")
add("ek3.o4.yolo11s.H01.t_p_tek", "t_testi_p_tek_yonlu_kopya", y_("yolo11s", "H01"), DS_TEST, ev("ek.test.yolo11s.H01.p_tek"), "p", F_TA,
    f"= {eid('ek.test.yolo11s.H01.p_tek')} (tek yonlu seed-esli t, df=2)", ET, turetme="= ek.test.yolo11s.H01.p_tek")
kp = MW / "evidence" / "ek3_kumeli_p_yolo11s_H01.json"
if kp.exists():
    K = J(kp)
    assert K["ga_tutarli"], "kumeli bootstrap GA yeniden hesabi kayitli GA ile esit degil"
    FK = "evidence/ek3_kumeli_p_yolo11s_H01.json (uretici: evidence/scripts/ek3_kumeli_p.py; CPU)"
    add("ek3.o4.yolo11s.H01.kumeli_boot_p_tek", "kumeli_bootstrap_p_tek_yonlu_delta_le_0", y_("yolo11s", "H01"), DS_TEST, K["p_tek_yonlu_le0"], "p", FK,
        f"p_tek_yonlu_le0: B={K['B']} kumeli bootstrap ornekleminde Δ*≤0 orani (13 sahne grubu; rng tohumu 0; analiz_ek.py test() ile ayni cekim). Tohum varyansi icermez; ON-KAYITLI DEGIL",
        ET, turetme="mean(boot<=0)", n=N, uyari="Keşifsel/duyarlılık; ön-kayıtlı karar kuralı değildir (karar: seed-eşli t + Holm).")
    add("ek3.o4.yolo11s.H01.kumeli_boot_q10", "kumeli_bootstrap_yuzdelik_10", y_("yolo11s", "H01"), DS_TEST, K["q10"], "puan", FK, "np.quantile(boot, 0.10)", ET, turetme="quantile 0.10")
    add("ek3.o4.yolo11s.H01.kumeli_boot_ga_tutarlilik", "kumeli_ga_yeniden_hesap_tutarli", y_("yolo11s", "H01"), DS_TEST, "evet", "", FK,
        f"yeniden hesaplanan GA {K['ga95_yeniden']} == test-analiz.json kumeli_boot_ga95 {K['ga95_kaynak_test_analiz']}", ET, turetme="ga_tutarli")
else:
    print("UYARI: ek3_kumeli_p_yolo11s_H01.json yok; kumeli bootstrap p kaydi eklenmedi")

# ======================================================================================= Ö5: goruntu boyutu / 640 px
ET = "tur3_o5"
SZ = {}
for sp in ("train", "val", "test"):
    L = (WS / "data/kitti/splits" / f"{sp}.txt").read_text().split()
    sizes = []
    for p in L:
        with Image.open(p) as im:
            sizes.append(im.size)
    SZ[sp] = (L, sizes)
    c = Counter(sizes)
    W_, H_ = zip(*sizes)
    ds = f"KITTI {sp} ({len(L)} goruntu; PIL ile TUM goruntulerin basligi okundu)"
    add(f"ek3.o5.kitti.{sp}.n_goruntu", "goruntu_sayisi_olculen", f"KITTI {sp}", ds, len(L), "goruntu", f"WS/data/kitti/splits/{sp}.txt", "PIL Image.open(p).size", ET)
    add(f"ek3.o5.kitti.{sp}.n_1242x375", "goruntu_sayisi_1242x375", f"KITTI {sp}", ds, c[(1242, 375)], "goruntu", f"WS/data/kitti/splits/{sp}.txt", "count(size == (1242, 375))", ET, turetme="count")
    add(f"ek3.o5.kitti.{sp}.n_farkli_boyut", "farkli_boyut_sayisi", f"KITTI {sp}", ds, len(c), "adet", f"WS/data/kitti/splits/{sp}.txt", "len(set(size)); " + "; ".join(f"{w}x{h}:{n}" for (w, h), n in c.most_common()), ET)
    for nm, vv in (("w_min", min(W_)), ("w_max", max(W_)), ("h_min", min(H_)), ("h_max", max(H_))):
        add(f"ek3.o5.kitti.{sp}.{nm}", f"goruntu_{nm}", f"KITTI {sp}", ds, vv, "px", f"WS/data/kitti/splits/{sp}.txt", nm, ET)
Wm, Hm = 1242, 375
add("ek3.o5.kitti.en_boy_orani", "en_boy_orani_tipik", "KITTI tipik kare 1242x375", "KITTI", Wm / Hm, "oran", "WS/data/kitti/splits/*.txt", "1242/375 (en sik boyut; ek3.o5.kitti.*.n_1242x375)", ET, turetme="1242/375")
add("ek3.o5.kitti.olcek_640", "letterbox_olcek", "KITTI tipik kare, imgsz 640", "KITTI", 640 / Wm, "oran", "WS/data/kitti/splits/*.txt", "640/max(1242,375) = uzun kenar olcegi (Ultralytics letterbox)", ET, turetme="640/1242")
add("ek3.o5.kitti.etkin_yukseklik_640", "etkin_yukseklik_640", "KITTI tipik kare, imgsz 640", "KITTI", Hm * 640 / Wm, "px", "WS/data/kitti/splits/*.txt", "375 x 640/1242 (etkin en: 640; yatay pad haric)", ET, turetme="375*640/1242")
# nesne boyutu: 640'ta kucuk (<32^2) nesne orani


def nesne_boyut(sp):
    L, sizes = SZ[sp]
    ar, a640, cl = [], [], []
    for p, (W, H) in zip(L, sizes):
        lp = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
        s2 = (640 / max(W, H)) ** 2
        for l in lp.read_text().split("\n"):
            if l.strip():
                t = l.split()
                a = float(t[3]) * W * float(t[4]) * H
                ar.append(a); a640.append(a * s2); cl.append(int(t[0]))
    return np.array(ar), np.array(a640), np.array(cl)


for sp in ("train", "test"):
    ar, a6, cl_ = nesne_boyut(sp)
    pc = np.isin(cl_, [SINIF.index("pedestrian"), SINIF.index("cyclist")])
    ds = f"KITTI {sp} nesneleri (DontCare haric; YOLO etiketleri)"
    for ad, msk in (("tum", np.ones(len(ar), bool)), ("ped_cyc", pc)):
        add(f"ek3.o5.kitti.{sp}.{ad}.n_nesne", "nesne_sayisi", f"{ad}/KITTI {sp}", ds, int(msk.sum()), "adet", f"WS/data/kitti/yolo/labels/*.txt (splits/{sp}.txt)", f"count ({ad})", ET)
        add(f"ek3.o5.kitti.{sp}.{ad}.kucuk_orijinal_oran", "kucuk_nesne_orani_orijinal_px", f"{ad}/KITTI {sp}", ds, float((ar[msk] < 1024).mean()), "oran", f"WS/data/kitti/yolo/labels/*.txt",
            "mean(kutu alani < 32^2 orijinal piksel; COCO small)", ET, turetme="mean(w*W*h*H < 1024)")
        add(f"ek3.o5.kitti.{sp}.{ad}.kucuk_640_oran", "kucuk_nesne_orani_640px", f"{ad}/KITTI {sp}", ds, float((a6[msk] < 1024).mean()), "oran", f"WS/data/kitti/yolo/labels/*.txt",
            "mean(kutu alani x (640/max(W,H))^2 < 32^2); letterbox olcegi goruntu basina", ET, turetme="mean(a*(640/max(W,H))^2 < 1024)")
DM = J(WS / "data" / "data_manifest.json")["stats"]
for sp in ("train", "test"):
    s_ = DM[sp]
    add(f"ek3.o5.voc.{sp}.en_buyuk_kenar", "goruntu_en_buyuk_kenar", f"VOC {sp}", f"VOC07+12 {sp}", max(s_["img_w_min_max"][1], s_["img_h_min_max"][1]), "px", "WS/data/data_manifest.json",
        f"stats.{sp}.img_w_min_max[1], img_h_min_max[1] (manifest .md: 'en büyük kenar 500')", ET)
    add(f"ek3.o5.voc.{sp}.kucuk_orijinal_oran", "kucuk_nesne_orani_orijinal_px", f"VOC {sp}", f"VOC07+12 {sp}", s_["size_orig_frac"]["small"], "oran", "WS/data/data_manifest.json",
        f"stats.{sp}.size_orig_frac.small", ET)
    add(f"ek3.o5.voc.{sp}.kucuk_640_oran", "kucuk_nesne_orani_640px", f"VOC {sp}", f"VOC07+12 {sp}", s_["size_at_imgsz640_frac"]["small"], "oran", "WS/data/data_manifest.json",
        f"stats.{sp}.size_at_imgsz640_frac.small (VOC'ta 640 px BUYUTME: kucuk nesne orani duser)", ET)
add("ek3.o5.voc.olcek_640", "letterbox_olcek", "VOC (en buyuk kenar 500), imgsz 640", "VOC07+12", 640 / BY["ek3.o5.voc.train.en_buyuk_kenar"]["deger"], "oran", "WS/data/data_manifest.json",
    "640/500 (>1: BUYUTME; KITTI'de 0.515: KUCULTME)", ET, turetme="640/ek3.o5.voc.train.en_buyuk_kenar")

# ======================================================================================= Ö6: H07 gecikme motorlari
ET = "tur3_o6"
HS = (WS / "scripts" / "ek_deney" / "h07_kitti.py").read_text(encoding="utf-8").split("\n")
i0 = next(i for i, l in enumerate(HS) if l.startswith("def gecikme(")) + 1
i1 = next(i for i, l in enumerate(HS) if l.startswith("def main("))
F_H7 = "WS/scripts/ek_deney/h07_kitti.py"
add("ek3.o6.yolo11s.gecikme_motoru", "h07_gecikme_olcum_motoru", "H07/YOLO11s", "KITTI val (s0, batch 1)",
    "VOC ölçümünün motoru: H.Motor + H.gecikme_olc (varyantlar/h07_post/degerlendir.py; ön-işleme + çıkarım + son-işlem/NMS ayrı sayaçlı)", "", F_H7,
    f"gecikme() satir {i0}-{i1 - 1}: 'if not end2end: motor = H.Motor(...); return H.gecikme_olc(...)' ; docstring 'YOLO11s: VOC ölçümünün motoru (Motor/gecikme_olc) birebir'", ET)
add("ek3.o6.yolo26n.gecikme_motoru", "h07_gecikme_olcum_motoru", "H07/YOLO26n", "KITTI val (s0, batch 1)",
    "Görüntü başına ultralytics YOLO.predict(batch=1, half=True) + perf_counter + torch.cuda.synchronize (NMS'siz end2end; Python/ön-işleme ek yükü dahil)", "", F_H7,
    f"gecikme() satir {i0}-{i1 - 1}: 'm.predict(source=im, **kw)' dongusu; docstring 'YOLO26n ...: görüntü başına predict, perf_counter + cuda senkronu'", ET)
add("ek3.o6.motor_ayni_mi", "h07_gecikme_motoru_model_arasi_ayni", "H07 iki model", "KITTI val", "hayir", "", F_H7, "YOLO11s=H.Motor/gecikme_olc, YOLO26n=predict dongusu (iki ayri motor)", ET,
    uyari="Mutlak ms degerleri modeller arasi KARSILASTIRILAMAZ (R0: YOLO26n > YOLO11s tersine-sezgisel; ek3.o6.R0_oran_26n_11s).")
add("ek3.o6.R0_oran_26n_11s", "R0_gecikme_orani_26n_11s", "R0 YOLO26n / R0 YOLO11s", "KITTI val (s0, batch 1)", ev("ek.h07.yolo26n.gecikme_R0_ms") / ev("ek.h07.yolo11s.gecikme_R0_ms"), "x",
    "WS/sonuclar/ek_deney/h07_kitti.json", f"{eid('ek.h07.yolo26n.gecikme_R0_ms')} / {eid('ek.h07.yolo11s.gecikme_R0_ms')} (R0 ms: YOLO26n {ev('ek.h07.yolo26n.gecikme_R0_ms'):.3f}, YOLO11s {ev('ek.h07.yolo11s.gecikme_R0_ms'):.3f})", ET,
    turetme="ek.h07.yolo26n.gecikme_R0_ms / ek.h07.yolo11s.gecikme_R0_ms", uyari="nano model (daha az param) small modelden YAVAS gorunuyor: olcum motoru farki (Ö6).")
sec = {m: J(EK / "h07_kitti.json")[m]["secilen_test_satiri"] for m in MODELS}
assert sec == {"yolo26n": "U832", "yolo11s": "W2"}
add("ek3.o6.piksel_orani_U832", "girdi_piksel_orani_832_640", "U832/R0 (imgsz 832 vs 640)", "-", (832 / 640) ** 2, "x", "WS/scripts/ek_deney/h07_kitti.py", "(832/640)^2 (satir adi U832 = test cozunurlugu 832 px)", ET, turetme="(832/640)**2")
add("ek3.o6.gecikme_oran_U832_26n_kopya", "gecikme_orani_U832", "U832/R0 YOLO26n", "KITTI val (s0, batch 1)", ev("ek.h07.yolo26n.gecikme_oran"), "x", "WS/sonuclar/ek_deney/h07_kitti.json",
    f"= {eid('ek.h07.yolo26n.gecikme_oran')}", ET, turetme="= ek.h07.yolo26n.gecikme_oran")
add("ek3.o6.gecikme_oran_W2_11s_kopya", "gecikme_orani_W2", "W2/R0 YOLO11s", "KITTI val (s0, batch 1)", ev("ek.h07.yolo11s.gecikme_oran"), "x", "WS/sonuclar/ek_deney/h07_kitti.json",
    f"= {eid('ek.h07.yolo11s.gecikme_oran')}", ET, turetme="= ek.h07.yolo11s.gecikme_oran")
add("ek3.o6.voc_U832_gecikme_oran_kopya", "gecikme_orani_U832_VOC", "U832/R0 YOLO11n VOC", "VOC val", ev("h07.U832.lat_oran"), "x", "WS/sonuclar/h07_post.csv", f"= {eid('h07.U832.lat_oran')} (T4)", ET,
    turetme="= h07.U832.lat_oran")
H7M = (WS / "hipotezler" / "H07.md").read_text(encoding="utf-8").split("\n")
hb = next((n for n, l in enumerate(H7M, 1) if "gecikme oranı ≤1.10" in l), None)
assert hb
add("ek3.o6.butce_gecikme_orani", "h07_gecikme_butcesi_on_kayit", "KAZANC-A/B gecikme siniri", "-", float(re.search(r"gecikme oranı ≤(\d\.\d+)", H7M[hb - 1]).group(1)), "x", "WS/hipotezler/H07.md",
    f"satir {hb}: 'KAZANÇ-A/B gecikme oranı ≤1.10' (ilk deneyin ön-kaydı; ek ön-kayıt gecikme bütçesini yeniden tanımlamaz: ON-KAYIT-ek-deney.md H07 bölümü)", ET)

# ======================================================================================= Ö10: standart KITTI bolmesi
ET = "tur3_o10"
UB = J(WS / "data" / "kitti" / "bolme_raporu.json")["ultralytics_bolme"]
for k, nm in (("train", "train"), ("val", "val"), ("ortak_drive", "ortak_drive"), ("drive_toplam", "drive_toplam")):
    add(f"ek3.o10.ultralytics_bolme.{nm}", f"ultralytics_bolme_{nm}", "Ultralytics hazir KITTI bolmesi", "KITTI 2D Object (7481)", UB[k], "goruntu" if k in ("train", "val") else "drive", "WS/data/kitti/bolme_raporu.json", f"ultralytics_bolme.{k}", ET)
add("ek3.o10.ultralytics_bolme_kullanilmadi_gerekce", "bolme_karari", "Ultralytics hazir bolmesi", "KITTI", "141 drive'in 106'si train ve val'de ortak -> KULLANILMADI (drive-ayrik degil)", "", "WS/hipotezler/ek_deney/ON-KAYIT-ek-deney.md",
    "satir 29-30: 'Ultralytics hazır bölmesi: 141 drive'ın 106'sı train ve val'de ortak → KULLANILMAZ'; sabit-kamera 2011_09_28 'Ek' maddesi satir 27-29", ET)
add("ek3.o10.standart_bolme_kaynakta", "standart_3712_3769_bolmesi_anilan_dosya", "KITTI 3712/3769 (3DOP) bolmesi", "KITTI", "yalniz hakem-tasarim-ek-deney-tur4.md:%d (dronefreak model karti baglaminda); ON-KAYIT, bolme_raporu, gunluk, data/ anmiyor" % ln_no, "", F_HD,
    f"satir {ln_no}: 'Sahne-ayrık 3DOP bölmesinde (3 712/3 769)'. Chen ve ark. 3DOP kunyesi ve 'standart bolmede test bolumu yoktur' iddiasi KAYNAKTA YOK (dis kaynak arastirilmadi).", ET,
    not_="Ö10(2) için gerekçe cümlesinin ikinci yarısı (sabit-kamera sızıntısı) kaynaktadır (ON-KAYIT satır 27-29); birinci yarısı ('test bölümü yoktur') ve atıf kaynakta yoktur.")

# ----------------------------------------------------------------------------------------------------------------- yaz
d["kayitlar"] = base + R
with open(EVID, "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent=1)
print(f"{len(R)} kayit eklendi: {R[0]['id']} .. {R[-1]['id']}; toplam {len(d['kayitlar'])}")
