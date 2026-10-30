#!/usr/bin/env python
"""Hakem tur 4 ek kanitlari (B-1, B-2, O2, O3, O4, O9): evidence.json'a E3977'den itibaren `ek4.*` kayitlari EKLER.

Mevcut kayitlara (E0001..E3976) DOKUNMAZ. Idempotent: onceki calistirmanin `ek4.*` kayitlari silinip yeniden uretilir.
YENI egitim/degerlendirme KOSUMU YOKTUR (GPU kullanilmaz): degerler WS/sonuclar/ek_deney/{test/*.json, val-analiz.json,
val-last-tani.json, tani/train1000_sonuc.json}, mevcut evidence kayitlari ve WS dosya zamanlarindan okunur/turetilir.
Birim: puan = x100. Hicbir sayi elle yazilmaz. Ciktilar: evidence/evidence.json, notes/kilit-sonrasi-kitti.md,
notes/ek-deney-kanit-raporu-tur4.md.
Calistirma (MW kokunden): python evidence/scripts/build_ek_evidence_tur4.py
"""
import datetime as dt
import json
import math
import statistics as st
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
MW = Path(__file__).resolve().parents[2]
ROOT = MW.parents[1]
WS = ROOT / "deney_workspace" / "2-1-enhancing-real-time-object-detection-on-pascal-v"
EK = WS / "sonuclar" / "ek_deney"
EVID = MW / "evidence" / "evidence.json"
START = 3977


def J(p):
    return json.load(open(p, encoding="utf-8"))


d = J(EVID)
base = [r for r in d["kayitlar"] if not r["anahtar"].startswith("ek4.")]
_eski = {r["anahtar"]: r for r in d["kayitlar"] if r["anahtar"].startswith("ek4.")}
# tur-4 uretim zamani ilk calistirmada sabitlenir (idempotent: yeniden calistirmada degismez)
_ILK_ZAMAN = (dt.datetime.fromisoformat(_eski["ek4.kilit_sonrasi.tur4_ikincil"]["deger"]) if "ek4.kilit_sonrasi.tur4_ikincil" in _eski
              else dt.datetime.now().replace(microsecond=0))
assert max(int(r["id"][1:]) for r in base) == START - 1, "beklenen son mevcut kayit E3976"
BASE = {r["anahtar"]: r for r in base}
R, BY = [], {}


def ev(k):
    return BASE[k]["deger"]


def eid(k):
    return BASE[k]["id"] if k in BASE else BY[k]["id"]


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


def ids(*keys):
    return ", ".join(eid(k) for k in keys)


def idr(keys):
    """kayit id listesini kompakt aralik olarak yaz (E3977-E3980, E3990)"""
    n = sorted(int(eid(k)[1:]) for k in keys)
    out, i = [], 0
    while i < len(n):
        j = i
        while j + 1 < len(n) and n[j + 1] == n[j] + 1:
            j += 1
        out.append(f"E{n[i]}" if i == j else f"E{n[i]}-E{n[j]}")
        i = j + 1
    return ", ".join(out)


MODELS = ["yolo26n", "yolo11s"]
MAD = {"yolo26n": "YOLO26n", "yolo11s": "YOLO11s"}
CARDS = ["H01", "H02", "H03", "H04", "H05", "H06", "H08"]
IYI = ["H01", "H03", "H04", "H06", "H08"]
SINIF = ["car", "van", "truck", "pedestrian", "person_sitting", "cyclist", "tram", "misc"]
N = 3
DS_TEST = "KITTI test (1504 goruntu, 13 sahne grubu; DontCare=COCO crowd)"
DS_VAL = "KITTI val (987 goruntu, 10 sahne grubu)"
F_VA, F_TA, F_VL = "WS/sonuclar/ek_deney/val-analiz.json", "WS/sonuclar/ek_deney/test-analiz.json", "WS/sonuclar/ek_deney/val-last-tani.json"
VA, VL = J(F := EK / "val-analiz.json"), J(EK / "val-last-tani.json")
RUN = VA["kosumlar"]
TEST = {k: J(EK / "test" / f"{rid}.json") for k, rid in RUN.items()}
TAN = VA["tani"]
y_ = lambda m, c: f"{c}/{MAD[m]}"  # noqa: E731
tf = lambda k: f"WS/sonuclar/ek_deney/test/{RUN[k]}.json"  # noqa: E731
mean = lambda x: float(st.mean(x))  # noqa: E731
sd = lambda x: float(st.stdev(x))  # noqa: E731

# ============================================================================================ B-1(a): test mAP50, AP_S/M/L
ET = "tur4_b1"
MET = [("map50", "mAP50"), ("map_small", "AP_S"), ("map_medium", "AP_M"), ("map_large", "AP_L")]
for m in MODELS:
    for mk, mad in MET:
        bs = [TEST[f"{m}/B/s{i}"]["dontcare"][mk] * 100 for i in range(N)]
        assert all(b >= 0 for b in bs), (m, mk)  # -1 (nesne yok) olsaydi yakalanir
        add(f"ek4.test.{m}.B.{mk}.ort", f"test_{mk}_baseline_ort", f"B/{MAD[m]}", DS_TEST, mean(bs), "puan", tf(f"{m}/B/s0"),
            f"dontcare.{mk} x100 (best.pt), B s0-s2 ortalamasi (mutlak deger)", ET, n=N, turetme=f"mean(dontcare.{mk}; B s0..2)")
        add(f"ek4.test.{m}.B.{mk}.sd", f"test_{mk}_baseline_sd", f"B/{MAD[m]}", DS_TEST, sd(bs), "puan", tf(f"{m}/B/s0"),
            f"dontcare.{mk} x100, B s0-s2 stdev (ddof=1)", ET, n=N, turetme=f"stdev(dontcare.{mk}; B s0..2)")
        for c in CARDS:
            dl = []
            for i in range(N):
                a, b = TEST[f"{m}/{c}/s{i}"]["dontcare"][mk], TEST[f"{m}/B/s{i}"]["dontcare"][mk]
                assert a >= 0 and b >= 0
                dl.append((a - b) * 100)
                add(f"ek4.test.{m}.{c}.{mk}.delta_s{i}", f"delta_{mk}_seed", y_(m, c), DS_TEST, dl[-1], "puan", tf(f"{m}/{c}/s{i}"),
                    f"dontcare.{mk} x100 (best.pt) - ayni seed B ({RUN[f'{m}/B/s{i}']}.json)", ET, seed=i, n=N,
                    turetme="kart - B, ayni seed; betimsel (Holm/karar yok; ON-KAYIT v4 satir 121 ikincil/kesifsel)")
            add(f"ek4.test.{m}.{c}.{mk}.delta_ort", f"delta_{mk}_ort", y_(m, c), DS_TEST, mean(dl), "puan", "WS/sonuclar/ek_deney/test/<run_id>.json",
                f"mean({mk}.delta_s0..2)", ET, n=N, turetme=f"mean(ek4.test.{m}.{c}.{mk}.delta_s0..2)")
            add(f"ek4.test.{m}.{c}.{mk}.delta_sd", f"delta_{mk}_sd", y_(m, c), DS_TEST, sd(dl), "puan", "WS/sonuclar/ek_deney/test/<run_id>.json",
                f"stdev({mk}.delta_s0..2), ddof=1", ET, n=N, turetme=f"stdev(ek4.test.{m}.{c}.{mk}.delta_s0..2)")

# ============================================================================================ B-1(a2): sinif basina AP50-95 Delta
# sinif sirasi dogrulamasi: dontcare.map50_95_3sinif = mean(car, pedestrian, cyclist); 7sinif = person_sitting haric
t0 = TEST["yolo26n/B/s0"]["dontcare"]
ap0 = t0["ap50_95_sinif"]
assert len(ap0) == 8
assert abs(np.mean([ap0[0], ap0[3], ap0[5]]) - t0["map50_95_3sinif"]) < 2e-4, "sinif sirasi car,van,truck,pedestrian,person_sitting,cyclist,tram,misc degil"
assert abs(np.mean([a for j, a in enumerate(ap0) if j != 4]) - t0["map50_95_7sinif"]) < 2e-4
for m in MODELS:
    for j, s in enumerate(SINIF):
        bs = [TEST[f"{m}/B/s{i}"]["dontcare"]["ap50_95_sinif"][j] * 100 for i in range(N)]
        add(f"ek4.test.{m}.B.sinif.{s}.ort", "test_sinif_ap5095_baseline_ort", f"B/{MAD[m]}", DS_TEST, mean(bs), "puan", tf(f"{m}/B/s0"),
            f"dontcare.ap50_95_sinif[{j}] ({s}) x100, B s0-s2 ortalamasi (mutlak)", ET, n=N, turetme="mean(B s0..2)")
        for c in CARDS:
            dl = [(TEST[f"{m}/{c}/s{i}"]["dontcare"]["ap50_95_sinif"][j] - TEST[f"{m}/B/s{i}"]["dontcare"]["ap50_95_sinif"][j]) * 100 for i in range(N)]
            add(f"ek4.test.{m}.{c}.sinif.{s}.delta_ort", "delta_sinif_ap5095_ort", y_(m, c), DS_TEST, mean(dl), "puan", "WS/sonuclar/ek_deney/test/<run_id>.json",
                f"dontcare.ap50_95_sinif[{j}] ({s}): mean(kart - B; ayni seed; s0..2) x100", ET, n=N, tohum_deltalar=[round(x, 4) for x in dl],
                turetme="mean(kart - B, ayni seed); betimsel")
            add(f"ek4.test.{m}.{c}.sinif.{s}.delta_sd", "delta_sinif_ap5095_sd", y_(m, c), DS_TEST, sd(dl), "puan", "WS/sonuclar/ek_deney/test/<run_id>.json",
                f"dontcare.ap50_95_sinif[{j}] ({s}): stdev(kart - B; s0..2), ddof=1", ET, n=N, turetme="stdev(kart - B, ayni seed)")

# ============================================================================================ B-1(c): egitim-ici en iyi - yeniden degerlendirme (val, best.pt)
for m in MODELS:
    for c in ["B"] + CARDS:
        fl = []
        for i in range(N):
            t = TAN[f"{m}/{c}/s{i}"]
            f = (t["egitim_ici_en_iyi"] - t["yeniden_deg"]) * 100
            assert abs(f - t["fark_puan"]) < 1.5e-3, (m, c, i, f, t["fark_puan"])  # val-analiz.json'un kendi fark_puan'i ile tutarli
            fl.append(f)
            add(f"ek4.val.{m}.{c}.s{i}.egitim_ici_eksi_yeniden", "val_egitim_ici_en_iyi_eksi_yeniden_degerlendirme", y_(m, c) if c != "B" else f"B/{MAD[m]}", DS_VAL,
                f, "puan", F_VA, f"tani.{m}/{c}/s{i}: 100 x (egitim_ici_en_iyi - yeniden_deg); ON-KAYIT v5b O3(d); ayni best.pt agirliklari; yeniden_deg = bu calismanin COCO-tarzi, DontCare'siz degerlendirmesi",
                ET, seed=i, run_id=RUN[f"{m}/{c}/s{i}"], turetme="100*(egitim_ici_en_iyi - yeniden_deg)")
        add(f"ek4.val.{m}.{c}.egitim_ici_eksi_yeniden_ort", "val_egitim_ici_eksi_yeniden_ort", y_(m, c) if c != "B" else f"B/{MAD[m]}", DS_VAL, mean(fl), "puan", F_VA,
            "mean(s0..s2)", ET, n=N, turetme=f"mean(ek4.val.{m}.{c}.s0..2.egitim_ici_eksi_yeniden)")
        add(f"ek4.val.{m}.{c}.egitim_ici_eksi_yeniden_sd", "val_egitim_ici_eksi_yeniden_sd", y_(m, c) if c != "B" else f"B/{MAD[m]}", DS_VAL, sd(fl), "puan", F_VA,
            "stdev(s0..s2), ddof=1", ET, n=N, turetme=f"stdev(ek4.val.{m}.{c}.s0..2.egitim_ici_eksi_yeniden)")
allf = {m: [BY[f"ek4.val.{m}.{c}.s{i}.egitim_ici_eksi_yeniden"]["deger"] for c in ["B"] + CARDS for i in range(N)] for m in MODELS}
for m in MODELS:
    add(f"ek4.val.{m}.egitim_ici_eksi_yeniden_ort24", "val_egitim_ici_eksi_yeniden_ort_24kosum", MAD[m], DS_VAL, mean(allf[m]), "puan", F_VA,
        "24 kosum (B+7 kart x 3 tohum) ortalamasi", ET, n=24, turetme=f"mean(ek4.val.{m}.<kol>.s<i>.egitim_ici_eksi_yeniden)")
flat = [x for m in MODELS for x in allf[m]]
add("ek4.val.egitim_ici_eksi_yeniden_min", "val_egitim_ici_eksi_yeniden_min_48", "48 kosum", DS_VAL, min(flat), "puan", F_VA, "min over 48 kosum (-1.03 bekleniyor)", ET,
    turetme="min(ek4.val.*.egitim_ici_eksi_yeniden)")
add("ek4.val.egitim_ici_eksi_yeniden_maks", "val_egitim_ici_eksi_yeniden_maks_48", "48 kosum", DS_VAL, max(flat), "puan", F_VA, "max over 48 kosum (+1.80 bekleniyor)", ET,
    turetme="max(ek4.val.*.egitim_ici_eksi_yeniden)")
add("ek4.val.egitim_ici_eksi_yeniden_pozitif_sayi", "val_egitim_ici_eksi_yeniden_pozitif_kosum", "48 kosum", DS_VAL, sum(x > 0 for x in flat), "adet", F_VA,
    "count(fark > 0) / 48 (Ultralytics en iyi degeri yeniden degerlendirmeden buyuk)", ET, turetme="count(>0)")

# ============================================================================================ B-1(d): GA'nin DISLADIGI etki (kabul edilmeyen kartlar)
for m in MODELS:
    ne = [c for c in CARDS if ev(f"ek.test.{m}.{c}.karar") == "kabul ölçütü sağlanmadı"]
    assert ne, m
    for nm, lo, hi in (("t", "ga_lo", "ga_hi"), ("kumeli", "boot_ga_lo", "boot_ga_hi")):
        his = {c: ev(f"ek.test.{m}.{c}.{hi}") for c in ne}
        los = {c: ev(f"ek.test.{m}.{c}.{lo}") for c in ne}
        ch, cl = max(his, key=his.get), min(los, key=los.get)
        ids_hi = ", ".join(eid(f"ek.test.{m}.{c}.{hi}") for c in ne)
        ids_lo = ", ".join(eid(f"ek.test.{m}.{c}.{lo}") for c in ne)
        add(f"ek4.b1d.{m}.{nm}_ga_ust_maks", f"{nm}_ga_ust_sinir_maks_kabul_edilmeyen", MAD[m], DS_TEST, his[ch], "puan", F_TA,
            f"max(ek.test.{m}.<kart>.{hi}) kart in {','.join(ne)} (karar='kabul ölçütü sağlanmadı'); maks kart {ch}; kaynak id'ler {ids_hi}", ET, turetme=f"max over {len(ne)} kart",
            kartlar=ne, maks_kart=ch)
        add(f"ek4.b1d.{m}.{nm}_ga_alt_min", f"{nm}_ga_alt_sinir_min_kabul_edilmeyen", MAD[m], DS_TEST, los[cl], "puan", F_TA,
            f"min(ek.test.{m}.<kart>.{lo}) ayni kart kumesi; min kart {cl}; kaynak id'ler {ids_lo}", ET, turetme=f"min over {len(ne)} kart", kartlar=ne, min_kart=cl)
        add(f"ek4.b1d.{m}.{nm}_ga_ust_min", f"{nm}_ga_ust_sinir_min_kabul_edilmeyen", MAD[m], DS_TEST, his[min(his, key=his.get)], "puan", F_TA,
            f"min(ek.test.{m}.<kart>.{hi}) ayni kart kumesi; min kart {min(his, key=his.get)} (en siki ust sinir)", ET, turetme=f"min over {len(ne)} kart", kartlar=ne)
        add(f"ek4.b1d.{m}.{nm}_ga_alt_maks", f"{nm}_ga_alt_sinir_maks_kabul_edilmeyen", MAD[m], DS_TEST, los[max(los, key=los.get)], "puan", F_TA,
            f"max(ek.test.{m}.<kart>.{lo}) ayni kart kumesi; maks kart {max(los, key=los.get)} (en siki alt sinir)", ET, turetme=f"max over {len(ne)} kart", kartlar=ne)

# ============================================================================================ B-2: iki "DontCare'siz val" olcumu
ET = "tur4_b2"
keys = [f"{m}/{c}/s{i}" for m in MODELS for c in ["B"] + CARDS for i in range(N)]
assert len(keys) == 48 and set(keys) == set(VL) == set(TAN)
U_b = {k: TAN[k]["egitim_ici_en_iyi"] * 100 for k in keys}       # Ultralytics egitim-ici en iyi (DontCare'siz)
U_l = {k: TAN[k]["son_epoch"] * 100 for k in keys}               # Ultralytics son epoch
Rs_b = {k: VL[k]["best_dcsiz"] * 100 for k in keys}              # bu calismanin degerlendiricisi, DontCare'siz
Rs_l = {k: VL[k]["last_dcsiz"] * 100 for k in keys}
Rd_b = {k: VL[k]["best_dc"] * 100 for k in keys}                 # bu calismanin degerlendiricisi, DontCare kurali
Rd_l = {k: VL[k]["last_dc"] * 100 for k in keys}
for k in keys:  # yeniden_deg (val-analiz) == best_dcsiz (val-last-tani): ayni yeniden degerlendirme
    assert abs(TAN[k]["yeniden_deg"] - VL[k]["best_dcsiz"]) < 2e-6, k
    assert TAN[k]["en_iyi_epoch"] == VL[k]["en_iyi_epoch"], k
tot = {}
for m in MODELS:
    ks = [k for k in keys if k.startswith(m + "/")]
    u = mean([U_b[k] - U_l[k] for k in ks])
    rs = mean([Rs_b[k] - Rs_l[k] for k in ks])
    rd = mean([Rd_b[k] - Rd_l[k] for k in ks])
    # mevcut kayitlarla capraz kontrol
    assert abs(rs - ev(f"ek3.b3c.{m}.best_eksi_last_dcsiz_ort")) < 6e-3, (m, rs)
    assert abs(rd - ev(f"ek3.b3c.{m}.best_eksi_last_dc_ort")) < 6e-3, (m, rd)
    wide = [(Rd_b[k] - Rd_l[k]) - (Rs_b[k] - Rs_l[k]) for k in ks]
    n_wide, n_nar, n_eq = sum(w > 0 for w in wide), sum(w < 0 for w in wide), sum(w == 0 for w in wide)
    tot[m] = (n_wide, n_nar, n_eq)
    ds_ = f"{DS_VAL}; 24 kosum (B+7 kart x 3 tohum)"
    add(f"ek4.b2.{m}.ultralytics_best_eksi_last_ort24", "val_best_eksi_last_ultralytics_dcsiz_24", MAD[m], ds_, u, "puan", F_VA,
        "tani.<kosum>: 100 x (egitim_ici_en_iyi - son_epoch), 24 kosum ortalamasi. Olcum (i): Ultralytics egitim-ici degerlendirici (metrics.json val_map50_95 / results.csv), DontCare'siz", ET,
        n=24, turetme="mean(100*(egitim_ici_en_iyi - son_epoch))")
    add(f"ek4.b2.{m}.yeniden_best_eksi_last_dcsiz_ort24", "val_best_eksi_last_yeniden_dcsiz_24", MAD[m], ds_, rs, "puan", F_VL,
        "100 x (best_dcsiz - last_dcsiz), 24 kosum ortalamasi. Olcum (ii): bu calismanin COCO-tarzi yeniden degerlendirmesi, DontCare'siz; mevcut " + eid(f"ek3.b3c.{m}.best_eksi_last_dcsiz_ort") + " ile ayni",
        ET, n=24, turetme="mean(100*(best_dcsiz - last_dcsiz))")
    add(f"ek4.b2.{m}.yeniden_best_eksi_last_dc_ort24", "val_best_eksi_last_yeniden_dc_24", MAD[m], ds_, rd, "puan", F_VL,
        "100 x (best_dc - last_dc), 24 kosum ortalamasi. Olcum (ii'): COCO-tarzi yeniden degerlendirme, DontCare kurali; mevcut " + eid(f"ek3.b3c.{m}.best_eksi_last_dc_ort") + " ile ayni",
        ET, n=24, turetme="mean(100*(best_dc - last_dc))")
    add(f"ek4.b2.{m}.olcum_farki_ort24", "val_best_eksi_last_olcum_farki", MAD[m], ds_, u - rs, "puan", F_VL,
        "(i) - (ii): Ultralytics DontCare'siz best-last eksi yeniden-degerlendirme DontCare'siz best-last", ET, n=24,
        turetme=f"{eid(f'ek4.b2.{m}.ultralytics_best_eksi_last_ort24')} - {eid(f'ek4.b2.{m}.yeniden_best_eksi_last_dcsiz_ort24')}")
    add(f"ek4.b2.{m}.best_terimi_farki_ort24", "val_best_terimi_ultralytics_eksi_yeniden", MAD[m], ds_, mean([U_b[k] - Rs_b[k] for k in ks]), "puan", F_VA,
        "mean(egitim_ici_en_iyi - best_dcsiz): ayni best.pt agirligi, iki degerlendirici (= ortalama fark_puan)", ET, n=24, turetme="mean(100*(egitim_ici_en_iyi - yeniden_deg))")
    add(f"ek4.b2.{m}.last_terimi_farki_ort24", "val_last_terimi_ultralytics_eksi_yeniden", MAD[m], ds_, mean([U_l[k] - Rs_l[k] for k in ks]), "puan", F_VL,
        "mean(son_epoch (Ultralytics) - last_dcsiz): ayni last.pt agirligi, iki degerlendirici", ET, n=24, turetme="mean(100*(son_epoch - last_dcsiz))")
    add(f"ek4.b2.{m}.dc_gecisi_genisleyen_sayi", "best_eksi_last_dc_kuralina_gecince_genisleyen_kosum", MAD[m], ds_, n_wide, "adet", F_VL,
        f"count[(best_dc-last_dc) > (best_dcsiz-last_dcsiz)] / 24; daralan {n_nar}, esit {n_eq}", ET, n=24,
        turetme="count((best_dc-last_dc) - (best_dcsiz-last_dcsiz) > 0)", daralan=n_nar, esit=n_eq)
    add(f"ek4.b2.{m}.dc_gecisi_genisleme_ort", "best_eksi_last_dc_kuralina_gecince_ortalama_genisleme", MAD[m], ds_, mean(wide), "puan", F_VL,
        "mean[(best_dc-last_dc) - (best_dcsiz-last_dcsiz)] (24 kosum)", ET, n=24, turetme="mean(wide)")
    add(f"ek4.b2.{m}.dc_gecisi_genisleme_min", "best_eksi_last_dc_kuralina_gecince_en_kucuk_genisleme", MAD[m], ds_, min(wide), "puan", F_VL,
        "min[(best_dc-last_dc) - (best_dcsiz-last_dcsiz)] (24 kosum; >0 ise 24/24 genisler)", ET, n=24, turetme="min(wide)")
add("ek4.b2.dc_gecisi_genisleyen_toplam", "best_eksi_last_dc_kuralina_gecince_genisleyen_kosum_toplam", "iki model", DS_VAL, sum(t[0] for t in tot.values()), "adet", F_VL,
    f"{eid('ek4.b2.yolo26n.dc_gecisi_genisleyen_sayi')} + {eid('ek4.b2.yolo11s.dc_gecisi_genisleyen_sayi')}; N=48 (hakem tur 4 B-2: 'DontCare kurallı fark 48 koşumun 48'inde genişler')", ET,
    n=48, turetme="sum", uyari="Uyumsuzluk hipotezi (DontCare kurali geç epoch cezasini kaldirir -> fark daralir) DONUSUN tersini ongorur; olcum tersi: fark genisler.")
add("ek4.b2.best_gt_last_dc_toplam", "val_best_gt_last_dc_kosum_toplam", "iki model", DS_VAL, ev("ek3.b3c.yolo26n.best_gt_last_dc_sayi") + ev("ek3.b3c.yolo11s.best_gt_last_dc_sayi"), "adet", F_VL,
    f"{eid('ek3.b3c.yolo26n.best_gt_last_dc_sayi')} + {eid('ek3.b3c.yolo11s.best_gt_last_dc_sayi')} (best>last; best.pt val uzerinde secildigi icin buyuk olcude yapisaldir, ayirt edici degildir)", ET, n=48,
    turetme="sum")

# ============================================================================================ O3: train1000 tanisi
ET = "tur4_o3"
T1 = J(EK / "tani" / "train1000_sonuc.json")
FT = "WS/sonuclar/ek_deney/tani/train1000_sonuc.json"
DS_T = "KITTI train (1000 goruntu, seed 0; tani/train1000.txt; Ultralytics val, DontCare'siz)"
assert len((EK / "tani" / "train1000.txt").read_text().split()) == 1000
t1 = {}
for k, v in T1.items():
    rid, w = k.split("/")
    mkey = next(kk for kk, r_ in RUN.items() if r_ == rid)
    t1[(mkey, w[:-3])] = (rid, v)
assert len(t1) == 8
ORD = [("yolo11s/B/s0",), ("yolo11s/B/s1",), ("yolo11s/B/s2",), ("yolo26n/B/s0",)]
for (mkey, w), (rid, v) in sorted(t1.items()):
    m, c, s = mkey.split("/")
    sd_ = int(s[1:])
    for nm, val, jj in (("map50_95", v["map50_95"], None), ("map50", v["map50"], None), ("car_ap5095", v["sinif"][0], 0)):
        add(f"ek4.o3.train1000.{m}.B.s{sd_}.{w}.{nm}", f"train1000_{nm}", f"B/{MAD[m]} seed {sd_}/{w}.pt", DS_T, round(val * 100, 4), "puan", FT,
            f"'{rid}/{w}.pt'." + (f"map50_95 x100" if jj is None and nm == "map50_95" else ("map50 x100" if nm == "map50" else "sinif[0] (car) x100; sinif sirasi car,van,truck,pedestrian,person_sitting,cyclist,tram,misc (ON-KAYIT/README)")),
            ET, seed=sd_, run_id=rid, agirlik=f"{w}.pt")
    # ayni agirliklarin val degeri (Ultralytics egitim-ici; karsilastirma icin): best -> egitim_ici_en_iyi, last -> son_epoch
    tv = TAN[mkey]
    add(f"ek4.o3.train1000.{m}.B.s{sd_}.{w}.val_ultralytics", "val_map50_95_ayni_agirlik", f"B/{MAD[m]} seed {sd_}/{w}.pt", DS_VAL,
        round((tv["egitim_ici_en_iyi"] if w == "best" else tv["son_epoch"]) * 100, 4), "puan", F_VA,
        f"tani.{mkey}.{'egitim_ici_en_iyi' if w == 'best' else 'son_epoch'} x100 (Ultralytics val, DontCare'siz); train1000 degerleriyle ayni agirlik", ET, seed=sd_, run_id=rid)
# train1000 YOLO26n s0 mevcut kayitlarla tutarli mi
assert abs(BY["ek4.o3.train1000.yolo26n.B.s0.best.map50_95"]["deger"] - ev("ek3.b3.train1000.best_map50_95")) < 1e-6
assert abs(BY["ek4.o3.train1000.yolo26n.B.s0.last.map50_95"]["deger"] - ev("ek3.b3.train1000.last_map50_95")) < 1e-6
assert abs(BY["ek4.o3.train1000.yolo26n.B.s0.best.car_ap5095"]["deger"] - ev("ek3.b3.train1000.best_ap5095_car")) < 1e-6
BY["ek4.o3.train1000.yolo26n.B.s0.best.map50_95"]["kopya"] = f"= {eid('ek3.b3.train1000.best_map50_95')}"
for w in ("best", "last"):
    add(f"ek4.o3.train1000.yolo11s.B.{w}.map50_95_ort", f"train1000_map50_95_ort_{w}", f"B/YOLO11s/{w}.pt", DS_T,
        mean([BY[f"ek4.o3.train1000.yolo11s.B.s{i}.{w}.map50_95"]["deger"] for i in range(N)]), "puan", FT, f"mean(B s0..2 {w}.pt)", ET, n=N, turetme="mean s0..2")
add("ek4.o3.train1000.yolo11s.B.s2.best_epoch", "val_en_iyi_epoch_yolo11s_B_s2", "B/YOLO11s seed 2", DS_VAL, TAN["yolo11s/B/s2"]["en_iyi_epoch"], "epoch", F_VA,
    "tani.yolo11s/B/s2.en_iyi_epoch (best.pt = epoch 1 agirligi)", ET, seed=2)
YARI = ev("ek.epoch.yolo11s.B.s2")
assert YARI == TAN["yolo11s/B/s2"]["en_iyi_epoch"]

# ============================================================================================ O4: en iyi epoch gec tepe, kart basina
ET = "tur4_o4"
EP = lambda m, c, i: ev(f"ek.epoch.{m}.{c}.s{i}")  # noqa: E731
tot_later = {}
for c in CARDS:
    per_model = {}
    for m in MODELS:
        later = [EP(m, c, i) > EP(m, "B", i) for i in range(N)]
        per_model[m] = sum(later)
        add(f"ek4.o4.{m}.{c}.gec_tepe_sayi", "en_iyi_epoch_B_den_gec_tohum_sayisi", y_(m, c), DS_VAL, sum(later), "adet", "WS/sonuclar/ek_deney/val-analiz.json",
            f"count(en_iyi_epoch(kart,s) > en_iyi_epoch(B,s), kesin) / 3 tohum; {', '.join(eid(f'ek.epoch.{m}.{c}.s{i}') + ' vs ' + eid(f'ek.epoch.{m}.B.s{i}') for i in range(N))}",
            ET, n=N, turetme="count(strict >)")
        dif = [EP(m, c, i) - EP(m, "B", i) for i in range(N)]
        add(f"ek4.o4.{m}.{c}.ep_fark_ort", "en_iyi_epoch_farki_ort", y_(m, c), DS_VAL, mean(dif), "epoch", "WS/sonuclar/ek_deney/val-analiz.json",
            f"mean(en_iyi_epoch(kart,s) - en_iyi_epoch(B,s)); tohum farklari {dif}", ET, n=N, turetme="mean(kart - B)", tohum_farklari=dif)
    tot_later[c] = sum(per_model.values())
    add(f"ek4.o4.{c}.gec_tepe_toplam", "en_iyi_epoch_B_den_gec_tohum_sayisi_toplam", c, "KITTI val; 2 model x 3 tohum = 6", tot_later[c], "adet", "WS/sonuclar/ek_deney/val-analiz.json",
        f"{eid(f'ek4.o4.yolo26n.{c}.gec_tepe_sayi')} + {eid(f'ek4.o4.yolo11s.{c}.gec_tepe_sayi')} (6 tohumdan)", ET, n=6, turetme="sum over 2 models")
assert tot_later["H05"] == 6, tot_later
dig = {c: v for c, v in tot_later.items() if c != "H05"}
cm = max(dig, key=dig.get)
add("ek4.o4.H05_disi_maks_gec_tepe", "H05_disi_kartlarin_en_cok_gec_tepe_sayisi", "H05 disi 6 kart", "KITTI val; 6 tohum", dig[cm], "adet", "WS/sonuclar/ek_deney/val-analiz.json",
    f"max over H01-H08 \\ H05 ({', '.join(f'{c}:{v}' for c, v in sorted(dig.items()))}); en cok: {', '.join(c for c, v in dig.items() if v == dig[cm])}", ET, turetme="max", siralama=tot_later)
# eri farki (ep11-90) siralamasi: H05'in modelde kacinci oldugu
for m in MODELS:
    cur = {c: ev(f"ek.val.{m}.{c}.egri_ep11_90.delta_ort") for c in CARDS}
    srt = sorted(cur, key=cur.get, reverse=True)
    add(f"ek4.o4.{m}.H05.egri_sira", "H05_egri_ep11_90_delta_sirasi", f"H05/{MAD[m]}", DS_VAL, srt.index("H05") + 1, "sira (1 = en yuksek, 7 kart)", "WS/sonuclar/ek_deney/val-analiz.json",
        "7 kartin ep11-90 egri Δ ortalamasi sirasi: " + ", ".join(f"{c} {cur[c]:+.2f} ({eid(f'ek.val.{m}.{c}.egri_ep11_90.delta_ort')})" for c in srt), ET, turetme="rank", siralama={c: round(cur[c], 3) for c in srt})

# ============================================================================================ O9: kart basina kritik etki
ET = "tur4_o9"
fac = float(stats.t.ppf(0.98, 2) / math.sqrt(3))
add("ek4.o9.kritik_carpan", "kritik_etki_carpani", "t_{0.98,df=2}/sqrt(3)", "-", fac, "x", "WS/scripts/ek_deney/analiz_ek.py",
    "satir 93: _kritik_etki_ort_puan = stats.t.ppf(0.98, 2)/sqrt(3) * mean(sd_b); sd_b = 5 iyilesme kartinin Δ s.d.'leri; 0.98 = Holm-5 ilk adim (0.10/5) tek yonlu; %50 guc", ET, turetme="t.ppf(0.98,2)/sqrt(3)")
for m in MODELS:
    ce = {}
    for c in CARDS:
        ce[c] = fac * ev(f"ek.test.{m}.{c}.delta_sd")
        add(f"ek4.o9.{m}.{c}.kritik_etki", "kart_basina_kritik_etki", y_(m, c), DS_TEST, ce[c], "puan", F_TA,
            f"ek4.o9.kritik_carpan x ek.test.{m}.{c}.delta_sd ({eid(f'ek.test.{m}.{c}.delta_sd')}); %50 guc, Holm-5 ilk adim" + ("" if c in IYI else "; H02/H05 yonlu Holm-2 ailesindedir, burada yalniz betimsel"),
            ET, turetme=f"{fac:.4f} x sd")
    ort5 = mean([ce[c] for c in IYI])
    assert abs(ort5 - ev(f"ek.test.{m}.kritik_etki")) < 2.5e-3, (m, ort5)  # kart-ortalamasi = kayitli E2697/E2824
    add(f"ek4.o9.{m}.kritik_etki_ort5_kontrol", "kritik_etki_5_iyilesme_karti_ortalamasi", MAD[m], DS_TEST, ort5, "puan", F_TA,
        f"mean(5 iyilesme kartinin kritik etkisi) = {eid(f'ek.test.{m}.kritik_etki')} ({ev(f'ek.test.{m}.kritik_etki')})", ET, turetme="mean(IYI)")
assert abs(BY["ek4.o9.yolo11s.H01.kritik_etki"]["deger"] - 0.95) < 0.02

# ============================================================================================ O2: KITTI test acilisindan SONRA yapilan analizler
ET = "kilit_sonrasi"
ACILIS = dt.datetime.fromisoformat(ev("ek.acilis.zaman"))  # E3558
assert ACILIS == dt.datetime(2026, 10, 22, 12, 2, 15)


def mt(p):
    return dt.datetime.fromtimestamp(Path(p).stat().st_mtime).replace(microsecond=0)


def gitfirst(pat, path):
    try:
        o = subprocess.run(["git", "log", "--reverse", "--format=%aI", f"-S{pat}", "--", path], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=120).stdout.split()
        return dt.datetime.fromisoformat(o[0]).replace(tzinfo=None) if o else None
    except Exception:  # noqa: BLE001
        return None


EVP = "makale_workspace/voc-yolo11n-augloss-analiz/evidence/evidence.json"
# KITTI egitim kosumlari ACILISTAN ONCE bitti (yeni egitim yok): ACILIS'tan sonra baslayan WS/runs kosumu = yalniz H05c (VOC)
kos_sonra = sorted(p.name for p in (WS / "runs").iterdir() if p.is_dir() and p.name[:15] > "20261022-120215" and p.name[:8].isdigit())
assert len(kos_sonra) == 1 and "h05c" in kos_sonra[0] and "_ek_yolo" not in kos_sonra[0], kos_sonra
n_ek_kitti = sum(1 for p in (WS / "runs").iterdir() if "_ek_yolo" in p.name)
son_kitti_bitis = max(mt(p / ".done") for p in (WS / "runs").iterdir() if "_ek_yolo" in p.name and (p / ".done").exists())
assert son_kitti_bitis < ACILIS, son_kitti_bitis
test_json_mt = sorted(mt(p) for p in (EK / "test").glob("2026*.json"))
test_json_n = len(test_json_mt)
h07_test = sorted(mt(p) for p in (EK / "test" / "h07").glob("*.json"))
LIST = [
    # (anahtar, ad, zaman, dosya, test'e dokunur mu, tur, durum, kanit-id-aciklamasi)
    ("val_analiz", "analiz_ek.py val: KITTI val Δ (best.pt), Holm-5/Holm-2, ep11-90 egri, sinif3/7 (val)", mt(EK / "val-analiz.json"), "WS/sonuclar/ek_deney/val-analiz.json",
     "hayir (yalniz val)", "on-kayitli ikincil; kosum ACILISTAN SONRA", "ek.val.* (E2965-E3316)"),
    ("test_analiz", "analiz_ek.py test: on-kayitli birincil analiz (Δ, t-GA, Holm, kumeli bootstrap GA, last.pt, sinif3/7)", mt(EK / "test-analiz.json"), "WS/sonuclar/ek_deney/test-analiz.json",
     "evet (acilisin planlanan analizi; yeni degerlendirme yok)", "on-kayitli (ON-KAYIT v5b) -- acilisin parcasi", "ek.test.* (E2571-E2824, E2825-E2964)"),
    ("test_degerlendirme_48", "48 KITTI kosumu x best+last test degerlendirmesi (kitti_test_ac.py)", test_json_mt[0], "WS/sonuclar/ek_deney/test/*.json (son: %s)" % test_json_mt[-1].strftime("%H:%M:%S"),
     "evet (tek acilis)", "on-kayitli -- ACILISIN PARCASI (kilit-sonrasi analiz degil)", "E3557-E3559"),
    ("h07_test", "H07 test satirlari (YOLO26n U832, YOLO11s W2; s0-s2)", h07_test[0], "WS/sonuclar/ek_deney/test/h07/*.json (son: %s)" % h07_test[-1].strftime("%H:%M:%S"),
     "evet (secim val'de; acilisin parcasi)", "on-kayitli -- ACILISIN PARCASI", "ek.h07.* (E3463-E3505)"),
    ("val_last_tani", "kitti_val_last.py: val'de best.pt ve last.pt'nin DontCare kurali/DontCare'siz yeniden degerlendirmesi (96 degerlendirme; B3c)", mt(EK / "val-last-tani.json"), "WS/sonuclar/ek_deney/val-last-tani.json",
     "hayir (yalniz val; GPU)", "kesifsel; hakem istemi", "E3947-E3966"),
    ("dcsiz_test", "DontCare'siz test Δ, B mutlak degerleri, 3/7 sinif ikincilleri, test nesne sayilari, artefakt (best/last) kiyaslari", gitfirst('"E3574"', EVP), EVP,
     "evet (acilista uretilen test/*.json yeniden okundu; yeni degerlendirme yok)", "kesifsel/betimsel", "E3583-E3869 (tur3_b2/b6), E3401-E3462"),
    ("kumeli_p", "ek3_kumeli_p.py: YOLO11s H01 icin kumeli bootstrap tek yonlu p (13 grup, B=2000)", mt(MW / "evidence" / "ek3_kumeli_p_yolo11s_H01.json"), "evidence/ek3_kumeli_p_yolo11s_H01.json",
     "evet (mevcut test tahminleri)", "kesifsel (on-kayitli degil)", "E3886"),
    ("holm", "Holm-10 / Holm-14 duyarliliklari ve isaret-cevirme testi", gitfirst('"E3880"', EVP), EVP,
     "evet (mevcut Δ'lar)", "kesifsel/duyarlilik (on-kayitli degil)", "E3880-E3888"),
    ("boyut_betimleme", "goruntu ve nesne boyutu betimlemeleri (egitim/val/test etiketleri; 640 px olcekleme)", gitfirst('"E3889"', EVP), EVP,
     "test GT etiketleri (model ciktisi degil)", "kesifsel/betimsel", "E3889-E3931"),
    ("dis_referans", "dis referans (Hugging Face KITTI model karti) ve baseline makullugu", gitfirst('"E3870"', EVP), EVP,
     "hayir", "kesifsel", "E3870-E3879"),
    ("h05c_onkayit", "H05c on-kaydi (VFL + cls 1.47; VOC val; VOC test acilmaz)", mt(WS / "hipotezler" / "ek_deney" / "ON-KAYIT-H05c.md"), "WS/hipotezler/ek_deney/ON-KAYIT-H05c.md",
     "hayir (VOC; KITTI sonuclari goruldukten SONRA yazildi)", "kesifsel mekanizma karti; hakem istemi", "WS gunluk 'EK on-kayit H05c'"),
    ("h05c_kosum", "H05c egitimi (YOLO11n x VOC val, s0-s2; ilk kosum baslangici)", dt.datetime.fromisoformat("2026-10-22T15:32:40"), "WS/runs/" + kos_sonra[0],
     "hayir (VOC)", "kesifsel mekanizma karti; ACILISTAN sonra baslayan TEK egitim", "WS/runs/BILDIRIM.log 'BAŞLADI ... h05c'"),
    ("train1000_v2", "train1000 tanisi: YOLO11s B s0-s2 best/last (+ YOLO26n B s0) 1000 egitim goruntusunde", mt(EK / "tani" / "train1000_sonuc.json"), "WS/sonuclar/ek_deney/tani/train1000_sonuc.json",
     "hayir (egitim goruntuleri; GPU)", "kesifsel; hakem istemi", "E3977+ (ek4.o3.*; bu tur)"),
    ("tur4_ikincil", "ikincil test metrikleri (mAP50, AP_S/M/L, sinif basi AP; T10/T11), kart basina val Δ ve egitim-ici - yeniden degerlendirme farki, en iyi epoch ve kritik etki hesaplari", _ILK_ZAMAN, "evidence/scripts/build_ek_evidence_tur4.py",
     "evet (test/*.json yeniden okundu; yeni degerlendirme yok)", "ON-KAYIT v4 satir 121 / v5b O3(d) ikincilleri; kesifsel", "ek4.* (bu tur)"),
]
for key, ad, zaman, dosya, test_mu, durum, kanit in LIST:
    assert zaman is not None, key
    if key not in ("test_degerlendirme_48", "h07_test"):
        assert zaman > ACILIS, (key, zaman)
    add(f"ek4.kilit_sonrasi.{key}", "kitti_test_acilisindan_sonra_yapilan_is", ad, "KITTI (test acilisi 2026-10-22T12:02:15; E3558)", zaman.isoformat(), "zaman damgasi", dosya,
        f"dosya degisiklik zamani / git ilk-commit zamani; durum: {durum}; test'e dokunur: {test_mu}; kanit: {kanit}", ET, test_e_dokunur=test_mu, durum_not=durum,
        acilis_sonrasi_dakika=int((zaman - ACILIS).total_seconds() // 60), kanit_id=kanit)
add("ek4.kilit_sonrasi.yeni_kitti_egitimi", "kitti_acilisindan_sonra_baslayan_kitti_egitimi_sayisi", "WS/runs taramasi", "KITTI", 0, "adet", "WS/runs/*",
    f"ACILIS'tan sonra olusan WS/runs dizinleri: {kos_sonra} (yalniz H05c, VOC); KITTI ek kosumlari ({n_ek_kitti} dizin) son bitis {son_kitti_bitis.isoformat()} < acilis", ET)
add("ek4.kilit_sonrasi.yeni_test_degerlendirme", "kitti_acilisindan_sonra_yeni_test_degerlendirmesi", "test/*.json ve h07 mtime taramasi", "KITTI test", 0, "adet", "WS/sonuclar/ek_deney/test/*",
    f"{test_json_n} test/*.json (48 kosum x best+last, tek dosyada) {test_json_mt[0].strftime('%H:%M:%S')}-{test_json_mt[-1].strftime('%H:%M:%S')}; h07 test {h07_test[0].strftime('%H:%M:%S')}-{h07_test[-1].strftime('%H:%M:%S')}; "
    f"ACILIS-sonrasi ek test cikarimi yok (test/ dizini son degisiklik {mt(EK / 'test').strftime('%H:%M:%S')}; ayni dosyalar yeniden okunur)", ET,
    uyari="Dizin mtime'i dosya duzeyindedir; olcut: 48 test json + h07 test = acilisin kendisi (zaman penceresi 12:02-13:23). Yeni test cikarimi 'kosum yok' temelli; baska olcu gunluktedir.")
# ek.test.*.json 12:03-13:17 araligi acilisin icinde mi (h07 test 13:23'e kadar)
assert max(test_json_mt) < dt.datetime(2026, 10, 22, 13, 30)

# ----------------------------------------------------------------------------------------------------------------- yaz
d["kayitlar"] = base + R
with open(EVID, "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent=1)
print(f"{len(R)} kayit eklendi: {R[0]['id']} .. {R[-1]['id']}; toplam {len(d['kayitlar'])}")

# ----------------------------------------------------------------------------------------------------------------- notes/kilit-sonrasi-kitti.md
L = ["# KITTI test açıldıktan sonra yapılan analizler (hakem tur 4 Ö2)", "",
     f"KITTI test **{ACILIS.isoformat()}** tarihinde tek açılışla açıldı (E3558; ON-KAYIT v5b SHA256 {ev('ek.acilis.on_kayit_sha256')[:12]}…, E3559). "
     f"Açılıştan sonra KITTI için **yeni eğitim yok**, **yeni test çıkarımı yok** ({BY['ek4.kilit_sonrasi.yeni_kitti_egitimi']['id']}, {BY['ek4.kilit_sonrasi.yeni_test_degerlendirme']['id']}); "
     "hiçbir hiperparametre araması yok. Açılıştan sonra başlayan tek eğitim H05c'dir (VOC val; KITTI değil). Zaman damgaları dosya değişiklik zamanı ya da git ilk-commit zamanıdır.", "",
     "| # | zaman | iş | test'e dokunur mu | durum | kanıt | kayıt |", "|---|---|---|---|---|---|---|"]
for i, (key, ad, zaman, dosya, test_mu, durum, kanit) in enumerate(sorted(LIST, key=lambda x: x[2]), 1):
    L.append(f"| {i} | {zaman.strftime('%H:%M:%S')} | {ad} | {test_mu} | {durum} | {kanit} | {BY['ek4.kilit_sonrasi.' + key]['id']} |")
L += ["", "Notlar:", "- `test_analiz`, `test_degerlendirme_48` ve `h07_test` satırları **açılışın parçasıdır** (ön-kayıtlı plan); kilit-sonrası *ek* analiz sayılmaz, zaman sırası için listededir.",
      "- val-last-tani (B3c) ve train1000 tanısı GPU'da koştu (96 + 8 değerlendirme); yalnız val ve eğitim görüntülerini okur, test kilidine dokunmaz.",
      "- Hakem tur 4 önerisi için makaleye ekleme: \"KITTI testi açıldıktan sonra yeni eğitim, test değerlendirmesi ya da hiperparametre araması yapılmadı; yalnız şunlar eklendi: ... (hepsi keşifsel)\". "
      "Tam liste yukarıdadır; bu cümleye `val'de ön-kayıtlı ikincil analiz (val-analiz)` ve `ikincil test metrikleri (T10/T11)` de eklenmelidir.", ""]
(MW / "notes" / "kilit-sonrasi-kitti.md").write_text("\n".join(L), encoding="utf-8")
print("notes/kilit-sonrasi-kitti.md yazildi")

# ----------------------------------------------------------------------------------------------------------------- notes/ek-deney-kanit-raporu-tur4.md
def rng(keys):
    n = [int(BY[k]["id"][1:]) for k in keys]
    return f"E{min(n)}–E{max(n)}"


V = lambda k: BY[k]["deger"]  # noqa: E731
I = lambda k: BY[k]["id"]  # noqa: E731
rep = ["# Ek deney kanıt raporu — hakem tur 4 (2026-10-22)", "",
       f"Yeni kayıtlar: **{R[0]['id']}–{R[-1]['id']}** (`ek4.*`; {len(R)} kayıt). Üretici: `evidence/scripts/build_ek_evidence_tur4.py` (idempotent; GPU yok, yeni koşum yok). "
       "Mevcut kayıtlara (E0001–E3976) dokunulmadı. Birim: puan = ×100. Tablolar `evidence/scripts/build_ek_tables.py` (T10, T11 eklendi; T7–T9 aynı).", ""]
rep += ["## Tablolar (tam genişlik: table + tabularx, metin genişliği; small; caption üstte)", "", "Doğrulama: `figures/scripts/onizleme.py tablolar` → hata=0, **overfull=0**; `tablo_duzen.py --duzen tek` ihlal 0 (T7–T11, T1–T6). Renk yok (betimsel; Holm yok); T7–T9 yeniden üretimde değişmedi.", "",
        "| Tablo | Etiket | İçerik | Kayıtlar |", "|---|---|---|---|",
        f"| T10 `evidence/tables/T10_kitti_test_ek.tex` | `tab:kitti-test-ek` | model×kart test Δ ± s.d.: mAP50, AP_S, AP_M, AP_L; val Δ (DontCare kurallı); eğitim-içi − yeniden değerlendirme farkı; B mutlak satırı | {idr(k for k in BY if k.startswith('ek4.test.') and '.sinif.' not in k)}; val Δ: E2968–E3099 (`ek.val.*.delta_ort/sd`); fark: {rng(k for k in BY if k.startswith('ek4.val.') and 'egitim_ici_eksi_yeniden' in k and k.count('.') >= 3 and not k.endswith(('min', 'maks', 'sayi', 'ort24')))} (kol başına s0, s1, s2, ort, sd) |",
        f"| T11 `evidence/tables/T11_kitti_sinif.tex` | `tab:kitti-sinif` | model×kart sınıf başı AP50-95 Δ (8 sınıf; ort ± s.d.) | {idr(k for k in BY if k.startswith('ek4.test.') and '.sinif.' in k)} |", ""]

rep += ["## Madde madde (id → tek satır bulgu)", "", "### B-1 (ön-kayıtlı ikincil raporlar)", ""]


def _ozet_ap():
    out = []
    for m in MODELS:
        for mk, mad in MET:
            ds = [V(f"ek4.test.{m}.{c}.{mk}.delta_ort") for c in CARDS]
            out.append(f"{MAD[m]} {mad}: B {V(f'ek4.test.{m}.B.{mk}.ort'):.2f}±{V(f'ek4.test.{m}.B.{mk}.sd'):.2f} ({I(f'ek4.test.{m}.B.{mk}.ort')}); Δ>0 olan kart {sum(x > 0 for x in ds)}/7, aralık {min(ds):+.2f}…{max(ds):+.2f}")
    ortak = [MAD[m] for m in MODELS if all(V(f"ek4.test.{m}.{c}.{mk}.delta_ort") > 0 for c in CARDS for mk, _ in MET)]
    ek = (f" {', '.join(ortak)}: dört ikincil metriğin dördünde 7/7 kart pozitif → kart-özgü değil, ortak kayma ile uyumlu (best.pt ortak kayma {ev('ek.artefakt.yolo11s.kayma_best'):+.2f}, {eid('ek.artefakt.yolo11s.kayma_best')}; tohum başına E3967–E3969)." if ortak else "")
    return ("Bulgu (özet; betimsel, Holm/karar yok, renk yok; ON-KAYIT v4 satır 121 'ikincil (keşifsel)'): " + "; ".join(out) + "." + ek)


rep += ["**(a) mAP50, AP_S/M/L ve sınıf başı AP Δ (test; DontCare kurallı; best.pt).** Kaynakta VAR (`test/<run>.json` `dontcare.map50, map_small/medium/large, ap50_95_sinif`); T10 ve T11 olarak eklendi. "
        "Sınıf sırası car, van, truck, pedestrian, person_sitting, cyclist, tram, misc doğrulandı (3 ve 7 sınıf ortalamaları test json'unun kendi `map50_95_3sinif/7sinif` değerleriyle ±2e-4 tutuyor). Değerler: ",
        ""]
rows = []
for m in MODELS:
    for c in CARDS:
        cells = []
        for mk, mad in MET:
            cells.append(f"{mad} {V(f'ek4.test.{m}.{c}.{mk}.delta_ort'):+.2f}±{V(f'ek4.test.{m}.{c}.{mk}.delta_sd'):.2f}")
        rows.append(f"- {MAD[m]} {c}: " + "; ".join(cells) + f" ({idr(k for k in BY if k.startswith(f'ek4.test.{m}.{c}.') and '.sinif.' not in k)})")
rep += rows + [""]
rep += ["Sınıf başı Δ ort ± s.d. (T11'de; kayıt aralıkları): " + "; ".join(f"{MAD[m]} {c}: {idr(k for k in BY if k.startswith(f'ek4.test.{m}.{c}.sinif.'))}" for m in MODELS for c in CARDS[:1]) + " (kart başına aynı desen; ayrıntı `ek4.test.<m>.<kart>.sinif.<sinif>.delta_ort/sd`).", "",
        _ozet_ap(), ""]
rep += ["**(b) Kart başına val Δ (DontCare kurallı, best.pt).** Zaten kayıtlı: `ek.val.<m>.<kart>.delta_ort/sd` (YOLO26n H01–H08: E2968…; YOLO11s: aynı desen); T10'a sütun olarak eklendi (T8 değişmedi). Yeni kayıt gerekmedi. "
        "Değerler: " + "; ".join(f"{MAD[m]} {c} {ev(f'ek.val.{m}.{c}.delta_ort'):+.2f}±{ev(f'ek.val.{m}.{c}.delta_sd'):.2f} ({eid(f'ek.val.{m}.{c}.delta_ort')})" for m in MODELS for c in CARDS) + ".", ""]
rep += ["**(c) v5b Ö3(d): eğitim-içi en iyi − yeniden değerlendirme (val, best.pt; aynı ağırlık).** "
        f"Koşum başına ve model×kart ortalaması/s.d.: {rng(k for k in BY if k.startswith('ek4.val.') and 'egitim_ici_eksi_yeniden' in k and k.count('.') >= 3 and not k.endswith(('min', 'maks', 'sayi', 'ort24')))} (`ek4.val.<m>.<kol>.s<i>.egitim_ici_eksi_yeniden` ve `…egitim_ici_eksi_yeniden_ort/_sd`); model 24-koşum ortalaması: {I('ek4.val.yolo26n.egitim_ici_eksi_yeniden_ort24')} ({V('ek4.val.yolo26n.egitim_ici_eksi_yeniden_ort24'):+.2f}), "
        f"{I('ek4.val.yolo11s.egitim_ici_eksi_yeniden_ort24')} ({V('ek4.val.yolo11s.egitim_ici_eksi_yeniden_ort24'):+.2f}); 48 koşum aralığı {I('ek4.val.egitim_ici_eksi_yeniden_min')} ({V('ek4.val.egitim_ici_eksi_yeniden_min'):+.2f}) … "
        f"{I('ek4.val.egitim_ici_eksi_yeniden_maks')} ({V('ek4.val.egitim_ici_eksi_yeniden_maks'):+.2f}); pozitif koşum sayısı {I('ek4.val.egitim_ici_eksi_yeniden_pozitif_sayi')} ({V('ek4.val.egitim_ici_eksi_yeniden_pozitif_sayi')}/48). "
        "`yeniden_deg` = val-last-tani'deki `best_dcsiz` (assert ile doğrulandı): yani bu fark Ultralytics değerlendiricisi ile COCO-tarzı DontCare'siz yeniden değerlendirme arasındaki farktır; DontCare kuralı karışmaz.", "",
        "Model×kart ortalamaları: " + "; ".join(f"{MAD[m]} {c} {V(f'ek4.val.{m}.{c}.egitim_ici_eksi_yeniden_ort'):+.2f}" for m in MODELS for c in ["B"] + CARDS) + ".", ""]
rep += ["**(d) %95 GA'nın dışladığı etki (kabul edilmeyen kartlar).** Kartların t-GA (tohum düzeyi, df=2) ve kümeli GA sınırları ZATEN kayıtlı; yalnız id'ler:", "",
        "| Model | Kart | karar | t-GA alt/üst (id) | kümeli GA alt/üst (id) |", "|---|---|---|---|---|"]
for m in MODELS:
    for c in CARDS:
        k = f"ek.test.{m}.{c}"
        rep.append(f"| {MAD[m]} | {c} | {ev(k + '.karar')} | [{ev(k + '.ga_lo'):+.2f}, {ev(k + '.ga_hi'):+.2f}] ({eid(k + '.ga_lo')}, {eid(k + '.ga_hi')}) | [{ev(k + '.boot_ga_lo'):+.2f}, {ev(k + '.boot_ga_hi'):+.2f}] ({eid(k + '.boot_ga_lo')}, {eid(k + '.boot_ga_hi')}) |")
rep += ["", "Her iki kart kümesi için ('kabul ölçütü sağlanmadı' kartları; H01 YOLO11s KABUL, H02/H05 yönlü aile hariç) ortak dışlama: iyileşme > en büyük üst sınır, kayıp < en küçük alt sınır (tüm kartlarda aynı anda dışlanan etki):", ""]
for m in MODELS:
    rep.append(f"- {MAD[m]} (kartlar {', '.join(BY[f'ek4.b1d.{m}.t_ga_ust_maks']['kartlar'])}): tohum t-GA üst sınırı en çok **{V(f'ek4.b1d.{m}.t_ga_ust_maks'):+.2f}** ({I(f'ek4.b1d.{m}.t_ga_ust_maks')}; kart {BY[f'ek4.b1d.{m}.t_ga_ust_maks']['maks_kart']}), "
               f"alt sınırı en az **{V(f'ek4.b1d.{m}.t_ga_alt_min'):+.2f}** ({I(f'ek4.b1d.{m}.t_ga_alt_min')}; kart {BY[f'ek4.b1d.{m}.t_ga_alt_min']['min_kart']}); "
               f"kümeli GA üst en çok {V(f'ek4.b1d.{m}.kumeli_ga_ust_maks'):+.2f} ({I(f'ek4.b1d.{m}.kumeli_ga_ust_maks')}), alt en az {V(f'ek4.b1d.{m}.kumeli_ga_alt_min'):+.2f} ({I(f'ek4.b1d.{m}.kumeli_ga_alt_min')}). "
               f"En sıkı sınırlar: t üst min {V(f'ek4.b1d.{m}.t_ga_ust_min'):+.2f} ({I(f'ek4.b1d.{m}.t_ga_ust_min')}), t alt maks {V(f'ek4.b1d.{m}.t_ga_alt_maks'):+.2f} ({I(f'ek4.b1d.{m}.t_ga_alt_maks')}).")
rep += ["", "Bulgu: T7 dipnotu için cümle: 'Kabul edilmeyen kartlarda tohum t-GA'nın üst sınırı YOLO26n'de en çok X, YOLO11s'te en çok Y puandır' → X, Y yukarıdaki kalın değerlerdir (`turetilmis` yerine doğrudan `ek4.b1d.*` kaydı).", ""]

rep += ["### B-2 (iki 'DontCare'siz val' ölçümü)", "",
        "(i) Ultralytics eğitim-içi değerlendirici (`metrics.json val_map50_95`, `results.csv`; val-analiz `tani.egitim_ici_en_iyi`, `son_epoch`); (ii) bu çalışmanın COCO-tarzı yeniden değerlendirmesi (`val-last-tani.json best_dcsiz/last_dcsiz`); (ii') aynısı DontCare kuralıyla (`best_dc/last_dc`). 24 koşum/model (B + 7 kart × 3 tohum):", "",
        "| Model | (i) Ultralytics best−last | (ii) yeniden değ. DontCare'siz best−last | (ii') yeniden değ. DontCare kurallı best−last | (i)−(ii) | best terimi farkı | last terimi farkı | DC'ye geçişte genişleyen/24 | ort. genişleme |", "|---|---|---|---|---|---|---|---|---|"]
for m in MODELS:
    rep.append(f"| {MAD[m]} | {V(f'ek4.b2.{m}.ultralytics_best_eksi_last_ort24'):+.2f} ({I(f'ek4.b2.{m}.ultralytics_best_eksi_last_ort24')}) | {V(f'ek4.b2.{m}.yeniden_best_eksi_last_dcsiz_ort24'):+.2f} ({I(f'ek4.b2.{m}.yeniden_best_eksi_last_dcsiz_ort24')}; = {eid(f'ek3.b3c.{m}.best_eksi_last_dcsiz_ort')}) | "
               f"{V(f'ek4.b2.{m}.yeniden_best_eksi_last_dc_ort24'):+.2f} ({I(f'ek4.b2.{m}.yeniden_best_eksi_last_dc_ort24')}; = {eid(f'ek3.b3c.{m}.best_eksi_last_dc_ort')}) | {V(f'ek4.b2.{m}.olcum_farki_ort24'):+.2f} ({I(f'ek4.b2.{m}.olcum_farki_ort24')}) | "
               f"{V(f'ek4.b2.{m}.best_terimi_farki_ort24'):+.2f} ({I(f'ek4.b2.{m}.best_terimi_farki_ort24')}) | {V(f'ek4.b2.{m}.last_terimi_farki_ort24'):+.2f} ({I(f'ek4.b2.{m}.last_terimi_farki_ort24')}) | "
               f"**{V(f'ek4.b2.{m}.dc_gecisi_genisleyen_sayi')}/24** ({I(f'ek4.b2.{m}.dc_gecisi_genisleyen_sayi')}; daralan {BY[f'ek4.b2.{m}.dc_gecisi_genisleyen_sayi']['daralan']}) | {V(f'ek4.b2.{m}.dc_gecisi_genisleme_ort'):+.2f} ({I(f'ek4.b2.{m}.dc_gecisi_genisleme_ort')}); min {V(f'ek4.b2.{m}.dc_gecisi_genisleme_min'):+.2f} |")
rep += ["", f"Toplam: DontCare kuralına geçişte best−last farkı **{V('ek4.b2.dc_gecisi_genisleyen_toplam')}/48** koşumda genişliyor ({I('ek4.b2.dc_gecisi_genisleyen_toplam')}); best>last sayısı {V('ek4.b2.best_gt_last_dc_toplam')}/48 ({I('ek4.b2.best_gt_last_dc_toplam')}; yapısal, ayırt edici değil). "
        "Baseline-yalnız ortalamalar (T9 notu / Ek A): E3685 (YOLO26n +4.37), E3711 (YOLO11s +4.39) = ölçüm (i) ve yalnız B; (i)'nin 24-koşum ortalaması yukarıdadır (hakem 'tüm koşumların ortalaması ≈ +4.0 / +4.6' demişti; 7 kart ortalaması E3686/E3712, B hariç).",
        f"Bulgu: uyumsuzluk hipotezi DC kuralında farkın daralmasını öngörür; ölçülen tersi (genişleme, 48/48). İki değerlendirici aynı best.pt ağırlığında {V('ek4.val.egitim_ici_eksi_yeniden_min'):+.2f}…{V('ek4.val.egitim_ici_eksi_yeniden_maks'):+.2f} puan ayrışır ({I('ek4.val.egitim_ici_eksi_yeniden_min')}, {I('ek4.val.egitim_ici_eksi_yeniden_maks')}); "
        f"model ortalamasında (i)−(ii) = +{V('ek4.b2.yolo26n.olcum_farki_ort24'):.2f} / +{V('ek4.b2.yolo11s.olcum_farki_ort24'):.2f} (best terimi +{V('ek4.b2.yolo26n.best_terimi_farki_ort24'):.2f}/+{V('ek4.b2.yolo11s.best_terimi_farki_ort24'):.2f}, last terimi {V('ek4.b2.yolo26n.last_terimi_farki_ort24'):+.2f}/{V('ek4.b2.yolo11s.last_terimi_farki_ort24'):+.2f}). "
        f"Etiketler: T9 notu / Ek A (+4.37/+4.39 yalnız B) = (i); (iii)'teki +3.23/+3.84 = (ii).", ""]

rep += ["### Ö3 (YOLO11s train1000 tanısı)", "",
        "`tani/train1000_sonuc.json` (`train1000_degerlendir_v2.py`; 1000 eğitim görüntüsü, Ultralytics val, DontCare'siz; liste `tani/train1000.txt` 1000 satır doğrulandı). Değerler (mAP50-95 / mAP50 / car AP50-95; best | last; aynı ağırlığın val değeri):", "",
        "| Koşum | best mAP50-95 | best mAP50 | best car | last mAP50-95 | last mAP50 | last car | val best (Ultralytics) | val last |", "|---|---|---|---|---|---|---|---|---|"]
for mkey in ("yolo11s/B/s0", "yolo11s/B/s1", "yolo11s/B/s2", "yolo26n/B/s0"):
    m, c, s = mkey.split("/")
    p = f"ek4.o3.train1000.{m}.B.{s}"
    rep.append(f"| {MAD[m]} B {s} | {V(p + '.best.map50_95'):.2f} ({I(p + '.best.map50_95')}) | {V(p + '.best.map50'):.2f} ({I(p + '.best.map50')}) | {V(p + '.best.car_ap5095'):.2f} ({I(p + '.best.car_ap5095')}) | "
               f"{V(p + '.last.map50_95'):.2f} ({I(p + '.last.map50_95')}) | {V(p + '.last.map50'):.2f} ({I(p + '.last.map50')}) | {V(p + '.last.car_ap5095'):.2f} ({I(p + '.last.car_ap5095')}) | "
               f"{V(p + '.best.val_ultralytics'):.2f} ({I(p + '.best.val_ultralytics')}) | {V(p + '.last.val_ultralytics'):.2f} ({I(p + '.last.val_ultralytics')}) |")
_e = {s_: V(f"ek4.o3.train1000.yolo11s.B.s{s_}.best.map50_95") for s_ in range(N)}
_ep = {s_: TAN[f"yolo11s/B/s{s_}"]["en_iyi_epoch"] for s_ in range(N)}
_lv = [V(f"ek4.o3.train1000.yolo11s.B.s{s_}.last.map50_95") for s_ in range(N)]
_vl = [V(f"ek4.o3.train1000.yolo11s.B.s{s_}.last.val_ultralytics") for s_ in range(N)]
rep += ["", f"YOLO11s B ortalama: best {V('ek4.o3.train1000.yolo11s.B.best.map50_95_ort'):.2f} ({I('ek4.o3.train1000.yolo11s.B.best.map50_95_ort')}), last {V('ek4.o3.train1000.yolo11s.B.last.map50_95_ort'):.2f} ({I('ek4.o3.train1000.yolo11s.B.last.map50_95_ort')}). "
        f"Bulgu: YOLO11s last.pt eğitim sahnelerinde {min(_lv):.1f}…{max(_lv):.1f} mAP50-95'e çıkıyor (aynı ağırlığın val değeri {min(_vl):.1f}…{max(_vl):.1f}) → eğitim hattı YOLO11s'te de sağlam (train ≫ val). "
        f"best.pt (en iyi epoch s0 {_ep[0]}, s1 {_ep[1]}, s2 {_ep[2]}) train1000'de {_e[0]:.2f} / {_e[1]:.2f} / {_e[2]:.2f}: **epoch-1 best.pt (s2; epoch {V('ek4.o3.train1000.yolo11s.B.s2.best_epoch')}, {I('ek4.o3.train1000.yolo11s.B.s2.best_epoch')}) eğitim sahnelerini henüz öğrenmemiş** "
        f"(car AP {V('ek4.o3.train1000.yolo11s.B.s2.best.car_ap5095'):.1f} vs last {V('ek4.o3.train1000.yolo11s.B.s2.last.car_ap5095'):.1f}); 'model eğitim sahnelerini öğrenir' savı last.pt için geçerlidir, tepe-epoch ağırlığı için tohuma bağlıdır. "
        f"YOLO26n B s0 değerleri mevcut {eid('ek3.b3.train1000.best_map50_95')}–{eid('ek3.b3.train1000.best_ap5095_cyclist')} ile tutarlı (assert).", ""]

rep += ["### Ö4 (en iyi epoch geç tepe; kart başına)", "",
        "Kesin sıra (kartın en iyi epoch'u > aynı tohumun B'sinin en iyi epoch'u); 2 model × 3 tohum = 6:", "",
        "| Kart | YOLO26n | YOLO11s | toplam /6 | kayıt |", "|---|---|---|---|---|"]
for c in CARDS:
    rep.append(f"| {c} | {V(f'ek4.o4.yolo26n.{c}.gec_tepe_sayi')}/3 | {V(f'ek4.o4.yolo11s.{c}.gec_tepe_sayi')}/3 | **{V(f'ek4.o4.{c}.gec_tepe_toplam')}/6** | {I(f'ek4.o4.{c}.gec_tepe_toplam')} ({I(f'ek4.o4.yolo26n.{c}.gec_tepe_sayi')}, {I(f'ek4.o4.yolo11s.{c}.gec_tepe_sayi')}) |")
rep += ["", f"H05 dışındaki en yüksek: {V('ek4.o4.H05_disi_maks_gec_tepe')}/6 ({I('ek4.o4.H05_disi_maks_gec_tepe')}; " + ", ".join(f"{c}:{v}" for c, v in sorted(dig.items())) + "). Hakemin sayımı (H05 6; H02 4, H08 4, H01 3, H03 2, H06 2, H04 1) BİREBİR tutuyor.",
        f"Ortalama en iyi epoch farkı (kart−B): `ek4.o4.<m>.<kart>.ep_fark_ort` (H05: YOLO26n {V('ek4.o4.yolo26n.H05.ep_fark_ort'):+.1f}, YOLO11s {V('ek4.o4.yolo11s.H05.ep_fark_ort'):+.1f}; {I('ek4.o4.yolo26n.H05.ep_fark_ort')}, {I('ek4.o4.yolo11s.H05.ep_fark_ort')}).",
        f"ep11–90 eğri Δ sırası (1 = en yüksek / 7): H05 YOLO26n **{V('ek4.o4.yolo26n.H05.egri_sira')}.** ({I('ek4.o4.yolo26n.H05.egri_sira')}), YOLO11s **{V('ek4.o4.yolo11s.H05.egri_sira')}.** ({I('ek4.o4.yolo11s.H05.egri_sira')}) → eğri farkı YOLO26n'de H05'e özgü, YOLO11s'te değil; geç tepe H05'e özgü (6/6).", ""]

rep += ["### Ö9 (kart başına kritik etki)", "",
        f"Formül (analiz_ek.py satır 93): kritik etki = t_{{0.98, df=2}}/√3 × s.d.(Δ) = {V('ek4.o9.kritik_carpan'):.3f} × s_Δ ({I('ek4.o9.kritik_carpan')}); Holm-5 ilk adım, %50 güç. Kart başına:", "",
        "| Model | " + " | ".join(CARDS) + " | 5-kart ort. (kontrol) |", "|---|" + "---|" * 8]
for m in MODELS:
    rep.append(f"| {MAD[m]} | " + " | ".join(f"{V(f'ek4.o9.{m}.{c}.kritik_etki'):.2f} ({I(f'ek4.o9.{m}.{c}.kritik_etki')})" for c in CARDS) + f" | {V(f'ek4.o9.{m}.kritik_etki_ort5_kontrol'):.2f} ({I(f'ek4.o9.{m}.kritik_etki_ort5_kontrol')} = {eid(f'ek.test.{m}.kritik_etki')}) |")
rep += ["", f"Bulgu: YOLO11s H01 kritik etkisi **{V('ek4.o9.yolo11s.H01.kritik_etki'):.2f}** puan ({I('ek4.o9.yolo11s.H01.kritik_etki')}; hakem ≈0.95 → tutuyor); gözlenen Δ +1.49 (E2701) bunun üstünde. Kart ortalaması YOLO11s 2.92 (E2824) H08'in büyük s.d.'sinden yükselir (H02, H05 yönlü aile: betimsel).", ""]

rep += ["### Ö2 (KITTI test açılışından sonraki analizler)", "",
        f"Liste `notes/kilit-sonrasi-kitti.md`; kayıtlar {idr(k for k in BY if k.startswith('ek4.kilit_sonrasi.'))} (etiket `kilit_sonrasi`). "
        f"Açılış {ev('ek.acilis.zaman')} (E3558). Yeni KITTI eğitimi 0 ({I('ek4.kilit_sonrasi.yeni_kitti_egitimi')}), yeni test çıkarımı 0 ({I('ek4.kilit_sonrasi.yeni_test_degerlendirme')}); açılıştan sonra başlayan tek eğitim H05c (VOC). "
        "Satırlar (zaman): " + "; ".join(f"{k} ({z.strftime('%H:%M')})" for k, _a, z, *_r in LIST) + ".", ""]

rep += ["### Kırmızı çizgi / alarm notları", "",
        "- Beklenti ihlali ALARMI yok: iç−yeniden fark işareti ve genişleme beklenmeyen bir mekanizma arızası değil (iki değerlendirici bağımsız; assert'ler tuttu).",
        "- Kaynakta olmayan bir değer yok. mAP50/AP_S/M/L ve sınıf AP kaynakta VAR (B-1(a) eksik değil, yalnız makalede yoktu). `AP_S` KITTI'de en küçük sınıf (COCO area<32²): gürültü tabanı yüksek.",
        "- T10/T11 renksiz (betimsel; Holm yok) — T8 ile aynı politika.", ""]
(MW / "notes" / "ek-deney-kanit-raporu-tur4.md").write_text("\n".join(rep), encoding="utf-8")
print("notes/ek-deney-kanit-raporu-tur4.md yazildi")
