#!/usr/bin/env python
"""Kanit defteri uretici (makale-uret Faz 2): WS/runs/*/{.done,metrics.json,config.json} ve WS/sonuclar/*.csv|json'dan
MW/evidence/evidence.json'u turetir. Hicbir sayi elle yazilmaz; her kayit kaynak_dosya + kaynak_konum tasir.

Birim: "puan" = x100 (mAP 0.6349 -> 63.49). Turetilmis kayitlar `turetme` alaninda formul + girdi anahtarlarini tasir.
Calistirma (MW kokunden):  <WS>/env/.venv/Scripts/python.exe evidence/scripts/build_evidence.py
"""
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
MW = Path(__file__).resolve().parents[2]
ROOT = MW.parents[1]
WS = ROOT / "deney_workspace" / "2-1-enhancing-real-time-object-detection-on-pascal-v"
SON = WS / "sonuclar"

R = []          # kayitlar
BYKEY = {}      # anahtar -> kayit


def val(anahtar):
    return BYKEY[anahtar]["deger"]


def add(anahtar, metrik, yontem, veri_seti, deger, birim, dosya, konum, kaynak="yazilim", atif=None, **kw):
    assert anahtar not in BYKEY, f"yinelenen anahtar {anahtar}"
    if isinstance(deger, (np.floating, np.integer)):
        deger = deger.item()
    if isinstance(deger, float) and (math.isnan(deger) or math.isinf(deger)):
        raise ValueError(f"NaN/inf deger: {anahtar}")
    rec = {"id": f"E{len(R) + 1:04d}", "anahtar": anahtar, "metrik": metrik, "yontem": yontem, "veri_seti": veri_seti,
           "deger": deger, "birim": birim, "kaynak": kaynak, "kaynak_dosya": dosya, "kaynak_konum": konum,
           "atif": atif, "durum": "gecerli"}
    rec.update(kw)
    R.append(rec)
    BYKEY[anahtar] = rec
    return rec


def sd1(x):
    return float(np.std(np.asarray(x, float), ddof=1))


# ------------------------------------------------------------------------------------------- kaynak tablolar
lbv = pd.read_csv(SON / "leaderboard-val.csv")
lbt = pd.read_csv(SON / "leaderboard-test.csv")
h07 = pd.read_csv(SON / "h07_post.csv")
abl = pd.read_csv(SON / "ablasyon.csv")
tj = json.load(open(SON / "test-analiz.json", encoding="utf-8"))
# P1 (n=3 gecisi): leaderboard-val.csv artik 24 satir. CEKIRDEK = ilk 12 kosum (baseline 3, H03 3, diger alti kartin seed 0'i); E0001-E1527 id'leri
# bunlarla uretilir ve DEGISMEZ (metindeki `% ev:` izleri kirilmasin). Seed 1-2 kosumlari (k5-ek-seed) bolum 12'de SONDAN eklenir.
lbv_all = lbv.copy()
lbv = lbv[(lbv["seed"] == 0) | lbv["varyant"].isin(["baseline", "h03_boxcp"])]
assert len(lbv) == 12, len(lbv)

ARM_VAL = {"baseline": "baseline", "h01_multiscale25": "h01", "h02_innerciou": "h02", "h03_boxcp": "h03",
           "h04_scale25": "h04", "h05_vfl": "h05", "h06_box10": "h06", "h08_boxcp_doz": "h08"}
ARM_TEST = {"baseline-test": "baseline", "h03_boxcp-test": "h03", "baseline-tta-test": "tta"}
SEEDS = (0, 1, 2)
N_SEED = {a: 1 for a in ARM_VAL.values()}
N_SEED["baseline"] = 3
N_SEED["h03"] = 3
VAL_COLS = {"map50_95": "val_map50_95", "map50": "val_map50", "ap_s": "val_map_small", "ap_m": "val_map_medium",
            "ap_l": "val_map_large"}
TEST_COLS = {"map50_95": "test_map50_95", "map50": "test_map50", "map75": "test_map75", "ap_s": "test_map_small",
             "ap_m": "test_map_medium", "ap_l": "test_map_large", "precision": "test_precision",
             "recall": "test_recall"}

# ------------------------------------------------------------------------------------------- 1) VAL per-run
VAL_RUN = {}   # (arm, seed) -> run_id
for i, r in lbv.iterrows():
    arm = ARM_VAL[r["varyant"]]
    seed = int(r["seed"])
    VAL_RUN[(arm, seed)] = r["run_id"]
    et = "tam_kosum" if N_SEED[arm] >= 3 else "on_eleme_n1"
    for m, col in VAL_COLS.items():
        add(f"val.{m}.{arm}.s{seed}", f"val_{m}", f"{arm}", "VOC val (1000 goruntu)", float(r[col]) * 100, "puan",
            "WS/sonuclar/leaderboard-val.csv", f"satir {i + 2} (run_id={r['run_id']}), sutun {col}; x100",
            etiket=et, seed=seed, run_id=r["run_id"])
    add(f"val.egitim_sure_sa.{arm}.s{seed}", "egitim_sure", arm, "VOC train", float(r["egitim_sure_sa"]), "sa",
        "WS/sonuclar/leaderboard-val.csv", f"satir {i + 2}, sutun egitim_sure_sa", seed=seed, run_id=r["run_id"])
    add(f"val.latency_ms.{arm}.s{seed}", "latency_toplam", arm, "VOC val (300 gor., b1 FP16)",
        float(r["latency_toplam_ms"]), "ms", "WS/sonuclar/leaderboard-val.csv",
        f"satir {i + 2}, sutun latency_toplam_ms", oturum="val_egitim_sonu_olcumu", seed=seed, run_id=r["run_id"])
    add(f"val.fps.{arm}.s{seed}", "fps", arm, "VOC val (300 gor., b1 FP16)", float(r["fps"]), "FPS",
        "WS/sonuclar/leaderboard-val.csv", f"satir {i + 2}, sutun fps", oturum="val_egitim_sonu_olcumu", seed=seed,
        run_id=r["run_id"])
    add(f"val.en_iyi_epoch.{arm}.s{seed}", "en_iyi_epoch", arm, "VOC val", int(r["en_iyi_epoch"]), "epoch",
        "WS/sonuclar/leaderboard-val.csv", f"satir {i + 2}, sutun en_iyi_epoch", seed=seed, run_id=r["run_id"])

# ---- val gecikme orani (kart / baseline; ayni mimari -> olcum gurultusu; yalniz bilgi)
for arm in ARM_VAL.values():
    if arm == "baseline":
        continue
    sds = [s for s in SEEDS if (arm, s) in VAL_RUN]
    num = float(np.mean([val(f"val.latency_ms.{arm}.s{s}") for s in sds]))
    den = float(np.mean([val(f"val.latency_ms.baseline.s{s}") for s in sds]))
    add(f"val.latency_orani.{arm}", "gecikme_orani", f"{arm}/baseline", "VOC val (egitim sonu olcumu)", num / den, "x", "WS/sonuclar/leaderboard-val.csv",
        "latency_toplam_ms: kart (ayni seed'ler) / baseline (ayni seed'ler); cikarim mimarisi ayni -> fark olcum gurultusu", oturum="val_egitim_sonu_olcumu",
        turetme=f"mean(val.latency_ms.{arm}.s*) / mean(val.latency_ms.baseline.s*)",
        uyari_not=("H08 koşumunda CPU paylasimi (K5 hazirlik kuru-testi) nedeniyle sismis olabilir" if arm == "h08" else "olcum gurultusu"))

# ---- replikasyon (baseline rep_test_*, K3; K5 baseline-test ile bit-bit ayni)
for i, r in lbv[lbv["varyant"] == "baseline"].iterrows():
    seed = int(r["seed"])
    for m, col in (("map50_95", "rep_test_map50_95"), ("map50", "rep_test_map50"), ("ap_s", "rep_test_map_small"),
                   ("ap_m", "rep_test_map_medium"), ("ap_l", "rep_test_map_large")):
        add(f"rep.{m}.baseline.s{seed}", f"rep_test_{m}", "baseline", "VOC07 test (K3 rep_test)",
            float(r[col]) * 100, "puan", "WS/sonuclar/leaderboard-val.csv", f"satir {i + 2}, sutun {col}; x100",
            etiket="replikasyon", seed=seed, run_id=r["run_id"])
for m in ("map50_95", "map50"):
    x = [val(f"rep.{m}.baseline.s{s}") for s in SEEDS]
    add(f"rep.{m}.baseline.ort", f"rep_test_{m}_ort", "baseline", "VOC07 test (K3 rep_test)", float(np.mean(x)), "puan",
        "WS/sonuclar/replikasyon_map50_95.md" if m == "map50_95" else "WS/sonuclar/replikasyon_map50.md",
        "tablo 'Bizim baseline'; yeniden hesap: leaderboard-val.csv rep_test_* 3 seed", etiket="replikasyon", n=3,
        turetme=f"mean(rep.{m}.baseline.s0..s2)")
    add(f"rep.{m}.baseline.sd", f"rep_test_{m}_sd", "baseline", "VOC07 test (K3 rep_test)", sd1(x), "puan",
        "WS/sonuclar/replikasyon_map50_95.md" if m == "map50_95" else "WS/sonuclar/replikasyon_map50.md",
        "tablo 'Bizim baseline' (+-); ddof=1", etiket="replikasyon", n=3, turetme=f"std(ddof=1; rep.{m}.baseline.s0..s2)")
# K3 rep_test == K5 baseline-test (bit-bit)
for s in SEEDS:
    rt = lbt[(lbt["varyant"] == "baseline-test") & (lbt["seed"] == s)].iloc[0]
    assert abs(rt["test_map50_95"] * 100 - val(f"rep.map50_95.baseline.s{s}")) < 1e-3, "rep_test != K5 baseline-test"

# replikasyon literatur araligi (kaynak: literatur) -- WS/notlar/literatur-cipa.md §2.1
LIT = [("zhang_yolo11n", "YOLO11n sifirdan 500 ep", 61.3, 81.5, "zhang_2025_adlyolo", "A1 (Tablo 2+10)"),
       ("lin_yolo11n", "YOLO11-N 600 ep (gen. ayar)", 61.0, 81.2, "lin_2025_yolomaster", "A2 (Tablo 1)"),
       ("zhang_yolov8n", "YOLOv8n sifirdan 500 ep", 60.2, 80.9, "zhang_2025_adlyolo", "A3 (Tablo 10)"),
       ("elhamdouni_yolov8n", "YOLOv8n 200 ep", 63.5, 83.6, "elhamdouni_2025_yolov8nvoc", "B1 (Tablo 3-4, 8-9)"),
       ("lin_yolo11s", "YOLO11s tam ince ayar (best ckpt test uzerinde secilmis)", 64.28, 83.44, "lin_2026_yolopeft",
        "C1 (Tablo 1, Ek H)")]
for k, ad, m5095, m50, atif, satir in LIT:
    for m, v in (("map50_95", m5095), ("map50", m50)):
        add(f"lit.{m}.{k}", f"lit_{m}", ad, "VOC07+12 -> VOC07 test (literatur, farkli protokol)", v, "puan",
            "WS/notlar/literatur-cipa.md", f"§2.1 tablo satiri {satir}", kaynak="literatur", atif=atif,
            not_="AYNI KOSULDA URETILMEDI (baslatma/epoch/secim kumesi farkli); yalniz baglam")
add("rep.kabul.map50_95.alt", "replikasyon_kabul_alt", "baseline", "VOC07 test", 59.8, "puan",
    "WS/sonuclar/replikasyon_map50_95.md", "Tolerans: kabul araligi [0.598, 0.647]; x100", etiket="replikasyon")
add("rep.kabul.map50_95.ust", "replikasyon_kabul_ust", "baseline", "VOC07 test", 64.7, "puan",
    "WS/sonuclar/replikasyon_map50_95.md", "Tolerans: kabul araligi [0.598, 0.647]; x100", etiket="replikasyon")
add("rep.kabul.map50.alt", "replikasyon_kabul_alt", "baseline", "VOC07 test", 80.0, "puan",
    "WS/sonuclar/replikasyon_map50.md", "Tolerans: kabul araligi [0.8, 0.848]; x100", etiket="replikasyon")
add("rep.kabul.map50.ust", "replikasyon_kabul_ust", "baseline", "VOC07 test", 84.8, "puan",
    "WS/sonuclar/replikasyon_map50.md", "Tolerans: kabul araligi [0.8, 0.848]; x100", etiket="replikasyon")
add("rep.karar", "replikasyon_karar", "baseline", "VOC07 test", "YESIL", "", "WS/sonuclar/replikasyon_map50_95.md",
    "baslik 'Replikasyon Kapisi - YESIL'", etiket="replikasyon")
add("rep.lit_alt.map50_95", "lit_aralik_alt", "literatur", "VOC07 test", 61.0, "puan", "WS/sonuclar/replikasyon_map50_95.md",
    "'Literatur degeri/araligi 0.61 - 0.635'; x100", kaynak="literatur", atif="lin_2025_yolomaster")
add("rep.lit_ust.map50_95", "lit_aralik_ust", "literatur", "VOC07 test", 63.5, "puan", "WS/sonuclar/replikasyon_map50_95.md",
    "'Literatur degeri/araligi 0.61 - 0.635'; x100 (0.635)", kaynak="literatur", atif="elhamdouni_2025_yolov8nvoc")
add("rep.lit_alt.map50", "lit_aralik_alt", "literatur", "VOC07 test", 81.2, "puan", "WS/sonuclar/replikasyon_map50.md",
    "'Literatur degeri/araligi 0.812 - 0.836'; x100", kaynak="literatur", atif="lin_2025_yolomaster")
add("rep.lit_ust.map50", "lit_aralik_ust", "literatur", "VOC07 test", 83.6, "puan", "WS/sonuclar/replikasyon_map50.md",
    "'Literatur degeri/araligi 0.812 - 0.836'; x100", kaynak="literatur", atif="elhamdouni_2025_yolov8nvoc")

# ------------------------------------------------------------------------------------------- 2) VAL toplulastirma ve Δ
for arm in ("baseline", "h03"):
    for m in VAL_COLS:
        x = [val(f"val.{m}.{arm}.s{s}") for s in SEEDS]
        add(f"val.{m}.{arm}.ort", f"val_{m}_ort", arm, "VOC val", float(np.mean(x)), "puan",
            "WS/sonuclar/leaderboard-val.csv", "3 seed ortalamasi (turetilmis)", etiket="tam_kosum", n=3,
            turetme=f"mean(val.{m}.{arm}.s0..s2)")
        add(f"val.{m}.{arm}.sd", f"val_{m}_sd", arm, "VOC val", sd1(x), "puan", "WS/sonuclar/leaderboard-val.csv",
            "3 seed std (ddof=1)", etiket="tam_kosum", n=3, turetme=f"std(ddof=1; val.{m}.{arm}.s0..s2)")
for m in ("map50_95", "map50", "ap_s", "ap_m", "ap_l"):
    for arm in ARM_VAL.values():
        if arm == "baseline":
            continue
        ds = []
        for s in SEEDS:
            if (arm, s) not in VAL_RUN:
                continue
            d = val(f"val.{m}.{arm}.s{s}") - val(f"val.{m}.baseline.s{s}")
            ds.append(d)
            add(f"dval.{m}.{arm}.s{s}", f"delta_val_{m}", arm, "VOC val", d, "puan", "WS/sonuclar/leaderboard-val.csv",
                f"satir run_id={VAL_RUN[(arm, s)]} - satir run_id={VAL_RUN[('baseline', s)]}, sutun {VAL_COLS[m]}; x100",
                etiket="tam_kosum" if N_SEED[arm] >= 3 else "on_eleme_n1", seed=s, n_ref="ayni seed baseline",
                turetme=f"val.{m}.{arm}.s{s} - val.{m}.baseline.s{s}")
        if len(ds) >= 3:
            add(f"dval.{m}.{arm}.ort", f"delta_val_{m}_ort", arm, "VOC val", float(np.mean(ds)), "puan",
                "WS/sonuclar/leaderboard-val.csv", "seed-eslesmis Δ ortalamasi", etiket="tam_kosum", n=3,
                turetme=f"mean(dval.{m}.{arm}.s0..s2)")
            add(f"dval.{m}.{arm}.sd", f"delta_val_{m}_sd", arm, "VOC val", sd1(ds), "puan",
                "WS/sonuclar/leaderboard-val.csv", "seed-eslesmis Δ std (ddof=1)", etiket="tam_kosum", n=3,
                turetme=f"std(dval.{m}.{arm}.s0..s2)")
            t, p = stats.ttest_rel([val(f"val.{m}.{arm}.s{s}") for s in SEEDS], [val(f"val.{m}.baseline.s{s}") for s in SEEDS])
            add(f"dval.{m}.{arm}.p_cift", f"delta_val_{m}_p_cift", arm, "VOC val", float(p), "p",
                "WS/sonuclar/leaderboard-val.csv", "seed-esli t (df=2), cift yonlu; yeniden hesap", etiket="tam_kosum",
                n=3, turetme="scipy.stats.ttest_rel(arm, baseline)")
            if float(np.mean(ds)) > 0:
                add(f"dval.{m}.{arm}.p_tek", f"delta_val_{m}_p_tek", arm, "VOC val", float(p) / 2, "p",
                    "WS/sonuclar/leaderboard-val.csv", "tek yonlu (Δ>0) = p_cift/2", etiket="tam_kosum", n=3,
                    turetme=f"dval.{m}.{arm}.p_cift / 2")
# dogrulama: H03 val p == test-analiz.json
assert abs(val("dval.map50_95.h03.p_cift") - tj["val_h03_delta"]["p"]) < 1e-9
assert abs(val("dval.map50_95.h03.ort") - tj["val_h03_delta"]["ort"]) < 1e-6
# ablasyon.csv ile capraz dogrulama (ikincil Δ)
chk = {"H01": ("h01", {"map50": 0.46, "ap_s": -2.78, "ap_m": 2.49, "ap_l": 0.15}),
       "H04": ("h04", {"map50": -0.10, "ap_s": -1.78, "ap_m": 1.04, "ap_l": -0.37}),
       "H05": ("h05", {"map50": -4.29, "ap_s": -2.16, "ap_m": -3.94, "ap_l": -1.96})}
for _, (arm, d) in chk.items():
    for m, v in d.items():
        assert abs(val(f"dval.{m}.{arm}.s0") - v) < 0.006, (arm, m, val(f"dval.{m}.{arm}.s0"), v)

# ---- AP75 val (pycocotools, hata-analizi json; yalniz s0 eslesmesi) + hata turu oracle/sayilari
HATA_VAL = {("baseline", 0): "hata-analizi.md.json",
            ("baseline", 1): "hata-analizi_20260930-101624_baseline_s1.md.json",
            ("baseline", 2): "hata-analizi_20260930-131427_baseline_s2.md.json",
            ("h01", 0): "hata-analizi_20260930-165837_h01_multiscale25_s0.md.json",
            ("h02", 0): "hata-analizi_20260930-201415_h02_innerciou_s0.md.json",
            ("h03", 0): "hata-analizi_20260930-231243_h03_boxcp_s0.md.json",
            ("h03", 1): "hata-analizi_20261001-021011_h03_boxcp_s1.md.json",
            ("h03", 2): "hata-analizi_20261001-050625_h03_boxcp_s2.md.json",
            ("h06", 0): "hata-analizi_20261001-115531_h06_box10_s0.md.json",
            ("h08", 0): "hata-analizi_20261001-182518_h08_boxcp_doz_s0.md.json"}
TURLER = ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]


def hata_kayitlari(prefix, split, j, dosya, tag, arm, seed):
    add(f"{prefix}.map75.{arm}.s{seed}", f"{split}_map75_pycoco", arm, split, j["map75"] * 100, "puan", dosya,
        "anahtar map75 (pycocotools; Ultralytics degeri degil); x100", seed=seed, olcum="pycocotools")
    for t in TURLER:
        add(f"{prefix}.oracle_dap50.{t}.{arm}.s{seed}", f"oracle_dAP50_{t}", arm, split, j["oracle"][t]["d_ap50"] * 100,
            "puan", dosya, f"anahtar oracle.{t}.d_ap50; x100", seed=seed)
        add(f"{prefix}.sayi_op.{t}.{arm}.s{seed}", f"hata_sayisi_op_{t}", arm, split, int(j["hata_sayilari_op"][t]),
            "adet", dosya, f"anahtar hata_sayilari_op.{t} (skor>=0.25)", seed=seed)
        add(f"{prefix}.sayi_tum.{t}.{arm}.s{seed}", f"hata_sayisi_tum_{t}", arm, split, int(j["hata_sayilari"][t]),
            "adet", dosya, f"anahtar hata_sayilari.{t} (conf>=0.001)", seed=seed)


for (arm, s), fn in HATA_VAL.items():
    j = json.load(open(SON / fn, encoding="utf-8"))
    hata_kayitlari("hv", "VOC val", j, f"WS/sonuclar/{fn}", fn, arm, s)
for arm in ("h01", "h02", "h06", "h08"):
    d = val(f"hv.map75.{arm}.s0") - val("hv.map75.baseline.s0")
    add(f"dval.map75.{arm}.s0", "delta_val_map75", arm, "VOC val", d, "puan", "WS/sonuclar/hata-analizi*.md.json",
        "map75 (pycocotools) kart s0 - baseline s0", etiket="on_eleme_n1", seed=0,
        turetme=f"hv.map75.{arm}.s0 - hv.map75.baseline.s0")
assert abs(val("dval.map75.h02.s0") - (-0.62)) < 0.01 and abs(val("dval.map75.h06.s0") - 0.36) < 0.01

# ---- Hata turu: TEST (hafif mod; bu makale icin uretilmis)
HATA_TEST = {("baseline", 0): "hata-analizi-test_20261001-212345_baseline-test_s0.md.json",
             ("baseline", 1): "hata-analizi-test_20261001-212500_baseline-test_s1.md.json",
             ("baseline", 2): "hata-analizi-test_20261001-212610_baseline-test_s2.md.json",
             ("h03", 0): "hata-analizi-test_20261001-212720_h03_boxcp-test_s0.md.json",
             ("h03", 1): "hata-analizi-test_20261001-212828_h03_boxcp-test_s1.md.json",
             ("h03", 2): "hata-analizi-test_20261001-212936_h03_boxcp-test_s2.md.json"}
for (arm, s), fn in HATA_TEST.items():
    j = json.load(open(SON / fn, encoding="utf-8"))
    hata_kayitlari("ht", "VOC07 test", j, f"WS/sonuclar/{fn}", fn, arm, s)
    for sz in ("S", "M", "L"):
        add(f"ht.ap5095_boyut.{sz}.{arm}.s{s}", f"pycoco_ap5095_{sz}", arm, "VOC07 test", j["boyut"][sz]["ap5095"] * 100,
            "puan", f"WS/sonuclar/{fn}", f"anahtar boyut.{sz}.ap5095; x100", seed=s, olcum="pycocotools")
    # pycocotools vs Ultralytics tutarlilik (hata analizi dogrulama)
    add(f"ht.map5095.{arm}.s{s}", "test_map50_95_pycoco", arm, "VOC07 test", j["map5095"] * 100, "puan",
        f"WS/sonuclar/{fn}", "anahtar map5095; x100", seed=s, olcum="pycocotools")

# ---- Kayip paylari (egitim results.csv, son epoch) -- duzeltilmis kazancli loss_items
for (arm, s), rid in VAL_RUN.items():
    if arm not in ("baseline", "h03") and not (arm == "h06"):
        continue
    df = pd.read_csv(WS / "runs" / rid / "train" / "results.csv")
    for ep_lab, row in (("ep1", df.iloc[0]), ("ep100", df.iloc[-1])):
        tot = row["train/box_loss"] + row["train/cls_loss"] + row["train/dfl_loss"]
        for k in ("box", "cls", "dfl"):
            add(f"loss_pay.{k}.{ep_lab}.{arm}.s{s}", f"egitim_kayip_payi_{k}_{ep_lab}", arm, "VOC train", float(row[f"train/{k}_loss"] / tot * 100),
                "%", f"WS/runs/{rid}/train/results.csv", f"satir epoch={int(row['epoch'])}, sutun train/{k}_loss / toplam (loss_items kazancli)",
                seed=s, turetme="train/k_loss / (box+cls+dfl)")
assert abs(val("loss_pay.box.ep100.baseline.s0") - 30.1) < 0.1 and abs(val("loss_pay.cls.ep100.baseline.s0") - 23.5) < 0.1 \
    and abs(val("loss_pay.dfl.ep100.baseline.s0") - 46.3) < 0.1

# ---- mekanizma devrede (kanca kaniti; plasebo-kol degil)
for i, r in lbv.iterrows():
    arm = ARM_VAL[r["varyant"]]
    s = int(r["seed"])
    rid = r["run_id"]
    kk = []
    if arm == "h01":
        kk = [("ms_farkli_boyut_sayisi", "h01_farkli_girdi_boyutu_sayisi", "adet"), ("ms_boyut_min", "h01_boyut_min", "px"),
              ("ms_boyut_max", "h01_boyut_max", "px")]
    elif arm == "h02":
        kk = [("inner_iou_fark_ort", "h02_inner_iou_fark_ort", "IoU")]
    elif arm == "h04":
        kk = [("scale_batch_sayisi", "h04_scale_batch_sayisi", "adet"), ("scale_args", "h04_scale_arg", "")]
    elif arm == "h05":
        kk = [("vfl_bce_grad_orani", "h05_vfl_bce_grad_orani", "oran"), ("vfl_bce_kayip_orani", "h05_vfl_bce_kayip_orani", "oran")]
    elif arm == "h06":
        kk = [("box_gain", "h06_box_gain", "")]
    elif arm in ("h03", "h08"):
        kk = [("boxcp_uygulanan_goruntu", "boxcp_uygulanan_goruntu", "adet"), ("boxcp_cagri", "boxcp_cagri", "adet"),
              ("varyant_etki_sayaci", "boxcp_yapistirilan_nesne", "adet"), ("boxcp_yer_bulunamadi", "boxcp_yer_bulunamadi", "adet")]
    for col, met, bir in kk:
        add(f"mek.{col}.{arm}.s{s}", met, arm, "VOC train (egitim sayaci)", float(r[col]), bir,
            "WS/sonuclar/leaderboard-val.csv", f"satir {i + 2}, sutun {col}", seed=s, run_id=rid,
            amac="mekanizma devrede kaniti (G2): kanca baglanmis ve tetiklenmis")

# ---- Doz (nesne/mosaic) = varyant_etki_sayaci / boxcp_cagri
for arm in ("h03", "h08"):
    for s in SEEDS:
        if (arm, s) not in VAL_RUN:
            continue
        rr = lbv[lbv["run_id"] == VAL_RUN[(arm, s)]].iloc[0]
        add(f"doz.nesne_per_mosaic.{arm}.s{s}", "doz_nesne_per_mosaic", arm, "VOC train", float(rr["varyant_etki_sayaci"] / rr["boxcp_cagri"]),
            "nesne/mosaic", "WS/sonuclar/leaderboard-val.csv", f"run_id={rr['run_id']}: varyant_etki_sayaci / boxcp_cagri",
            seed=s, uyari_not="sayac yaklasiktir (isci surecleri periyodik yazar)", turetme="varyant_etki_sayaci/boxcp_cagri")
add("doz.nesne_per_mosaic.baseline", "doz_nesne_per_mosaic", "baseline", "VOC train", 0.0, "nesne/mosaic",
    "WS/sonuclar/leaderboard-val.csv", "baseline: copy_paste=0 (sayac yok) - tanim geregi 0", turetme="tanim")
h8 = lbv[lbv["varyant"] == "h08_boxcp_doz"].iloc[0]
assert abs(val("doz.nesne_per_mosaic.h08.s0") - h8["boxcp_doz_nesne_per_mosaic"]) < 1e-3
add("doz.h03_ort", "doz_nesne_per_mosaic_ort", "h03", "VOC train", float(np.mean([val(f"doz.nesne_per_mosaic.h03.s{s}") for s in SEEDS])),
    "nesne/mosaic", "WS/sonuclar/leaderboard-val.csv", "3 seed ortalama", n=3, turetme="mean(doz.nesne_per_mosaic.h03.s0..s2)")
add("doz.carpan_h08_h03", "doz_carpan", "h08/h03", "VOC train", val("doz.nesne_per_mosaic.h08.s0") / val("doz.h03_ort"), "x",
    "WS/sonuclar/leaderboard-val.csv", "doz.h08.s0 / doz.h03_ort", turetme="doz.nesne_per_mosaic.h08.s0 / doz.h03_ort")
add("doz.h08_uygulanan_pay", "boxcp_uygulanan_pay", "h08", "VOC train", float(h8["boxcp_uygulanan_pay"]) * 100, "%",
    "WS/sonuclar/leaderboard-val.csv", "sutun boxcp_uygulanan_pay; x100")
add("doz.h08_hedef_alt", "doz_hedef_alt", "h08", "kart", 0.8, "nesne/mosaic", "WS/hipotezler/H08.md", "kart hedef araligi 0.8-1.2 (on-kayit)")
add("doz.h08_hedef_ust", "doz_hedef_ust", "h08", "kart", 1.2, "nesne/mosaic", "WS/hipotezler/H08.md", "kart hedef araligi 0.8-1.2 (on-kayit)")

# ---- Seed gurultusu
sv = [val(f"val.map50_95.baseline.s{s}") for s in SEEDS]
stt = [val(f"rep.map50_95.baseline.s{s}") for s in SEEDS]
add("gurultu.val_sd_baseline", "seed_sd_val_map50_95", "baseline", "VOC val (1000)", sd1(sv), "puan",
    "WS/sonuclar/leaderboard-val.csv", "baseline 3 seed val_map50_95 std (ddof=1)", n=3, turetme="std(val.map50_95.baseline.s0..s2)")
add("gurultu.test_sd_baseline", "seed_sd_test_map50_95", "baseline", "VOC07 test (4952)", sd1(stt), "puan",
    "WS/sonuclar/leaderboard-test.csv", "baseline-test 3 seed test_map50_95 std (ddof=1)", n=3, turetme="std(rep.map50_95.baseline.s0..s2)")
add("gurultu.oran", "seed_sd_oran", "baseline", "val/test", sd1(sv) / sd1(stt), "x", "WS/sonuclar/leaderboard-val.csv",
    "val_sd / test_sd", turetme="gurultu.val_sd_baseline / gurultu.test_sd_baseline")
aps = [val(f"val.ap_s.baseline.s{s}") for s in SEEDS]
add("gurultu.val_ap_s_aralik", "seed_aralik_val_ap_s", "baseline", "VOC val (143 kucuk nesne)", max(aps) - min(aps), "puan",
    "WS/sonuclar/leaderboard-val.csv", "baseline 3 seed val_map_small max-min", n=3, turetme="max-min(val.ap_s.baseline.s0..s2)")
add("gurultu.korelasyon_val_test", "seed_pearson_val_test", "baseline", "val/test", float(np.corrcoef(sv, stt)[0, 1]), "r",
    "WS/sonuclar/leaderboard-val.csv", "baseline 3 seed: val_map50_95 ile test_map50_95 Pearson r", n=3,
    turetme="corrcoef(val.map50_95.baseline.s*, rep.map50_95.baseline.s*)",
    uyari_not="n=3 (df=1): yorumlanamaz; H03 icin r asagida (0.18) -> tesaduf. Seed siralamasi/gurultu iddiasina dayanak yapilmaz")
add("gurultu.val_sd_tespit_esik", "min_tespit_edilebilir_etki", "baseline", "val", 0.5, "puan", "WS/sonuclar/karar.md",
    "§1 'n=3'te esli t-testinin tespit edebilecegi en kucuk etki ~0.5 puan'", etiket="karar_md_beyani")
add("on_kayit.on_eleme_esigi", "on_eleme_esigi", "karar_kurali", "val", 0.30, "puan", "WS/hipotezler/H01.md; WS/konu.md",
    "seed-0 on-eleme Δ >= +0.30 puan val mAP50-95 (on-kayitli)", etiket="on_kayit")
add("on_kayit.tut_esigi", "tut_esigi", "karar_kurali", "val", 0.40, "puan", "WS/hipotezler/H08.md; WS/sonuclar/karar.md §2-A",
    "TUT: 3 seed ort. Δ >= +0.40 ve tek yonlu p<0.10 ve >=2 seed Δ>0", etiket="on_kayit")
add("on_kayit.gercek_zaman_orani", "gecikme_kisiti", "karar_kurali", "latency", 1.10, "x", "WS/varyantlar/h07_post/README.md",
    "KAZANC-A: DOGRULANDI ve gecikme orani <= 1.10", etiket="on_kayit")

# ------------------------------------------------------------------------------------------- 3) TEST per-run
TEST_RUN = {}
for i, r in lbt.iterrows():
    arm = ARM_TEST[r["varyant"]]
    s = int(r["seed"])
    TEST_RUN[(arm, s)] = r["run_id"]
    for m, col in TEST_COLS.items():
        add(f"test.{m}.{arm}.s{s}", f"test_{m}", arm, "VOC07 test (4952)", float(r[col]) * 100, "puan",
            "WS/sonuclar/leaderboard-test.csv", f"satir {i + 2} (run_id={r['run_id']}), sutun {col}; x100",
            etiket="tam_kosum", seed=s, run_id=r["run_id"])
    add(f"test.latency_ms.{arm}.s{s}", "latency_toplam", arm, "VOC07 test oturumu (b1 FP16 uctan uca)", float(r["latency_toplam_ms"]), "ms",
        "WS/sonuclar/leaderboard-test.csv", f"satir {i + 2}, sutun latency_toplam_ms", oturum="test_K5", seed=s, run_id=r["run_id"])
    add(f"test.fps.{arm}.s{s}", "fps", arm, "VOC07 test oturumu", float(r["fps"]), "FPS", "WS/sonuclar/leaderboard-test.csv",
        f"satir {i + 2}, sutun fps", oturum="test_K5", seed=s, run_id=r["run_id"])
for arm in ("baseline", "h03", "tta"):
    for m in TEST_COLS:
        x = [val(f"test.{m}.{arm}.s{s}") for s in SEEDS]
        add(f"test.{m}.{arm}.ort", f"test_{m}_ort", arm, "VOC07 test", float(np.mean(x)), "puan",
            "WS/sonuclar/test-analiz.json", f"test.{arm if arm != 'baseline' else 'baseline'}: tablo.{ {'baseline':'baseline','h03':'h03','tta':'tta'}[arm] }.test_{m} 3 seed; yeniden hesap",
            etiket="tam_kosum", n=3, turetme=f"mean(test.{m}.{arm}.s0..s2)")
        add(f"test.{m}.{arm}.sd", f"test_{m}_sd", arm, "VOC07 test", sd1(x), "puan", "WS/sonuclar/test-analiz.json",
            "3 seed std (ddof=1); yeniden hesap", etiket="tam_kosum", n=3, turetme=f"std(ddof=1; test.{m}.{arm}.s0..s2)")
for arm in ("baseline", "h03", "tta"):
    x = [val(f"test.latency_ms.{arm}.s{s}") for s in SEEDS]
    add(f"test.latency_ms.{arm}.ort", "latency_toplam_ort", arm, "VOC07 test oturumu", float(np.mean(x)), "ms",
        "WS/sonuclar/test-analiz.md", "T-B hiz tablosu; yeniden hesap", oturum="test_K5", n=3, turetme="mean(test.latency_ms.arm.s*)")
    add(f"test.latency_ms.{arm}.sd", "latency_toplam_sd", arm, "VOC07 test oturumu", sd1(x), "ms",
        "WS/sonuclar/test-analiz.md", "T-B hiz tablosu (+-); yeniden hesap", oturum="test_K5", n=3, turetme="std(ddof=1)")
    y = [val(f"test.fps.{arm}.s{s}") for s in SEEDS]
    add(f"test.fps.{arm}.ort", "fps_ort", arm, "VOC07 test oturumu", float(np.mean(y)), "FPS", "WS/sonuclar/test-analiz.md",
        "T-B hiz tablosu; yeniden hesap", oturum="test_K5", n=3, turetme="mean(test.fps.arm.s*)")
add("test.latency_orani.tta", "gecikme_orani", "tta/baseline", "VOC07 test oturumu", val("test.latency_ms.tta.ort") / val("test.latency_ms.baseline.ort"),
    "x", "WS/sonuclar/test-analiz.md", "T-B: 21.64 / 9.43 = x2.29", oturum="test_K5", n=3,
    turetme="test.latency_ms.tta.ort / test.latency_ms.baseline.ort")
add("test.latency_orani.h03", "gecikme_orani", "h03/baseline", "VOC07 test oturumu", val("test.latency_ms.h03.ort") / val("test.latency_ms.baseline.ort"),
    "x", "WS/sonuclar/test-analiz.md", "T-B: 8.60 / 9.43 = 0.91 (ayni mimari; olcum gurultusu; hiz kazanci iddia edilmez)", oturum="test_K5", n=3,
    turetme="test.latency_ms.h03.ort / test.latency_ms.baseline.ort", uyari_not="ayni mimari; fark oturum gurultusu")
# dogrulama: ort/sd == test-analiz.json
for arm, jk in (("baseline", "bort"), ("h03", "ort")):
    pass
assert abs(val("test.map50_95.h03.ort") - tj["test"]["h03"]["test_map50_95"]["ort"]) < 1e-6
assert abs(val("test.map50_95.tta.sd") - tj["test"]["tta"]["test_map50_95"]["sd"]) < 1e-6
assert abs(val("test.latency_orani.tta") - 2.29) < 0.01

# ---- Test Δ (seed-eslesmis) ve istatistikler (test-analiz.json)
for arm in ("h03", "tta"):
    for m in TEST_COLS:
        blk = tj["test"][arm][TEST_COLS[m]]
        for s in SEEDS:
            add(f"dtest.{m}.{arm}.s{s}", f"delta_test_{m}", arm, "VOC07 test", float(blk["dseed"][s]) , "puan",
                "WS/sonuclar/test-analiz.json", f"test.{arm}.{TEST_COLS[m]}.dseed[{s}]", etiket="tam_kosum", seed=s, n_ref="ayni seed baseline")
        add(f"dtest.{m}.{arm}.ort", f"delta_test_{m}_ort", arm, "VOC07 test", float(blk["dort"]), "puan", "WS/sonuclar/test-analiz.json",
            f"test.{arm}.{TEST_COLS[m]}.dort", etiket="tam_kosum", n=3)
        add(f"dtest.{m}.{arm}.sd", f"delta_test_{m}_sd", arm, "VOC07 test", float(blk["dsd"]), "puan", "WS/sonuclar/test-analiz.json",
            f"test.{arm}.{TEST_COLS[m]}.dsd", etiket="tam_kosum", n=3)
        add(f"dtest.{m}.{arm}.p", f"delta_test_{m}_p_cift", arm, "VOC07 test", float(blk["p"]), "p", "WS/sonuclar/test-analiz.json",
            f"test.{arm}.{TEST_COLS[m]}.p (seed-esli t, df=2, cift yonlu; ikincil metriklerde duzeltmesiz)", etiket="tam_kosum", n=3)
for arm in ("h03", "tta"):
    b = tj["bootstrap"][arm]["birlesik"]
    for k, kk in (("ci_lo", "lo"), ("ci_hi", "hi"), ("ort", "ort"), ("alt90", "alt90"), ("p_pozitif", "p_pozitif"), ("p_marj030", "p_marj030")):
        add(f"boot.{arm}.{k}", f"bootstrap_{k}", arm, "VOC07 test", float(b[kk]), "puan" if k in ("ci_lo", "ci_hi", "ort", "alt90") else "olasilik",
            "WS/sonuclar/test-analiz.json", f"bootstrap.{arm}.birlesik.{kk} (B=1000, goruntu-duzeyi esli; %95 yuzdelik)", etiket="tam_kosum", n=3, B=1000)
    for s in SEEDS:
        bs = tj["bootstrap"][arm][f"s{s}"]
        add(f"boot.{arm}.s{s}.ci_lo", "bootstrap_ci_lo", arm, "VOC07 test", float(bs["lo"]), "puan", "WS/sonuclar/test-analiz.json",
            f"bootstrap.{arm}.s{s}.lo", seed=s, B=1000)
        add(f"boot.{arm}.s{s}.ci_hi", "bootstrap_ci_hi", arm, "VOC07 test", float(bs["hi"]), "puan", "WS/sonuclar/test-analiz.json",
            f"bootstrap.{arm}.s{s}.hi", seed=s, B=1000)
    add(f"holm.{arm}", "holm_p", arm, "VOC07 test", float(tj["holm"][arm]), "p", "WS/sonuclar/test-analiz.json",
        f"holm.{arm} (2 karsilastirma, birincil metrik test mAP50-95)", etiket="tam_kosum", n=3)
    add(f"bonf.{arm}", "bonferroni_p", arm, "VOC07 test", float(tj["bonf"][arm]), "p", "WS/sonuclar/test-analiz.json",
        f"bonf.{arm}", etiket="tam_kosum", n=3)
    # per-seed bootstrap SE (test, 4952 goruntu) = (hi-lo)/(2*1.96)
for arm in ("h03",):
    se = [(val(f"boot.{arm}.s{s}.ci_hi") - val(f"boot.{arm}.s{s}.ci_lo")) / (2 * 1.96) for s in SEEDS]
    add("turet.se_test_per_seed_h03", "bootstrap_se_per_seed_test", arm, "VOC07 test (4952)", float(np.mean(se)), "puan",
        "WS/sonuclar/test-analiz.json", "bootstrap.h03.s{0,1,2} %95 CI yari-genisligi/1.96 ortalamasi", turetme="mean((hi-lo)/(2*1.96))")
    add("turet.se_val_per_seed_h03_tahmin", "tahmini_se_per_seed_val", arm, "VOC val (1000)", float(np.mean(se)) * math.sqrt(4952 / 1000), "puan",
        "WS/sonuclar/test-analiz.json", "TAHMIN (iid goruntu varsayimi): se_test x sqrt(4952/1000)", etiket="tahmin_varsayimli",
        turetme="turet.se_test_per_seed_h03 * sqrt(4952/1000)", uyari_not="tahmindir; goruntu-duzeyi bootstrap val'de kosulmadi")

_ps = []
for arm in ("h03", "tta"):
    for m in ("map50_95", "map50", "map75", "ap_s", "ap_m", "ap_l"):
        _ps.append(((arm, m), val(f"dtest.{m}.{arm}.p")))
_order = sorted(range(len(_ps)), key=lambda i: _ps[i][1])
_adj = {}
_run = 0.0
for rank, i in enumerate(_order):
    a = min(1.0, (len(_ps) - rank) * _ps[i][1])
    _run = max(_run, a)
    _adj[_ps[i][0]] = _run
for (arm, m), a in _adj.items():
    add(f"holm12.{arm}.{m}", "holm_p_12_test", arm, "VOC07 test", float(a), "p", "WS/sonuclar/test-analiz.json",
        "12 duzeltmesiz p (H03,TTA x 6 metrik) uzerinde Holm; yeniden hesap (KESIFSEL)", etiket="kesifsel", n=3,
        turetme="Holm(dtest.{metric}.{arm}.p, 12 test)")
add("holm12.min", "holm_p_12_en_kucuk", "tum", "VOC07 test", float(min(_adj.values())), "p", "WS/sonuclar/test-analiz.json",
    "12 testin Holm sonrasi en kucuk p'si", etiket="kesifsel", turetme="min(holm12.*)")

# ---- val -> test tutarlilik
dv = [val(f"dval.map50_95.h03.s{s}") for s in SEEDS]
dt = [val(f"dtest.map50_95.h03.s{s}") for s in SEEDS]
add("tutarlilik.h03.val_test_fark_ort", "h03_val_delta_eksi_test_delta", "h03", "val-test", float(np.mean(dv) - np.mean(dt)), "puan",
    "WS/sonuclar/test-analiz.md", "T-D: 'val Δ - test Δ = +0.37'; yeniden hesap", n=3, turetme="mean(dval.h03) - mean(dtest.h03)")
dd = np.array(dv) - np.array(dt)
tt, pp = stats.ttest_1samp(dd, 0.0)
add("tutarlilik.h03.val_test_fark_p", "h03_val_test_fark_p_cift", "h03", "val-test", float(pp), "p", "WS/sonuclar/test-analiz.json",
    "seed-esli (val Δ_s - test Δ_s) tek-ornek t, df=2 (turetilmis; keşifsel)", n=3, turetme="ttest_1samp(dval.h03.s* - dtest.h03.s*, 0)")
add("tutarlilik.h03.val_test_fark_sd", "h03_val_test_fark_sd", "h03", "val-test", sd1(dd), "puan", "WS/sonuclar/test-analiz.json",
    "seed-esli fark std (ddof=1)", n=3, turetme="std(dval.h03.s* - dtest.h03.s*)")
add("gurultu.korelasyon_val_test_h03", "seed_pearson_val_test", "h03", "val/test",
    float(np.corrcoef([val(f"val.map50_95.h03.s{s}") for s in SEEDS], [val(f"test.map50_95.h03.s{s}") for s in SEEDS])[0, 1]), "r",
    "WS/sonuclar/leaderboard-val.csv", "h03 3 seed: val_map50_95 ile test_map50_95 Pearson r", n=3,
    turetme="corrcoef(val.map50_95.h03.s*, test.map50_95.h03.s*)", uyari_not="n=3 (df=1): yorumlanamaz")
# H03 val mAP50: 3/3 pozitif, p
add("tutarlilik.h03.val_map50_pozitif_seed", "h03_val_map50_pozitif_seed_sayisi", "h03", "VOC val",
    int(sum(1 for s in SEEDS if val(f"dval.map50.h03.s{s}") > 0)), "adet", "WS/sonuclar/leaderboard-val.csv", "dval.map50.h03.s0..s2 > 0 sayisi",
    n=3, turetme="count(dval.map50.h03.s*>0)")

# ------------------------------------------------------------------------------------------- 4) TTA val (H07 T1, 3 best.pt)
t1 = h07[h07["satir"] == "T1"].iloc[0]
t1i = int(h07.index[h07["satir"] == "T1"][0]) + 2
for k, col in (("s0", "delta_full_puan"), ("s1", "delta_s1_puan"), ("s2", "delta_s2_puan")):
    add(f"tta_val.delta.{k}", "delta_val_map50_95", "tta", "VOC val (H07 T1)", float(t1[col]), "puan", "WS/sonuclar/h07_post.csv",
        f"satir {t1i} (T1), sutun {col}", etiket="tam_kosum", seed=int(k[1]),
        not_="seed = baseline best.pt seed; aday secimi seed-0 yarim-A'da yapildi (B-8); s1/s2 onay")
dd3 = [val(f"tta_val.delta.s{k}") for k in range(3)]
add("tta_val.delta.ort", "delta_val_map50_95_ort", "tta", "VOC val (H07 T1)", float(np.mean(dd3)), "puan", "WS/sonuclar/h07_post.csv",
    "T1 satiri s0/s1/s2 ortalamasi (turetilmis)", n=3, turetme="mean(tta_val.delta.s0..s2)")
add("tta_val.delta.sd", "delta_val_map50_95_sd", "tta", "VOC val (H07 T1)", sd1(dd3), "puan", "WS/sonuclar/h07_post.csv",
    "std (ddof=1)", n=3, turetme="std(tta_val.delta.s0..s2)")
t_, p_ = stats.ttest_1samp(dd3, 0.0)
add("tta_val.delta.p_cift", "delta_val_p_cift", "tta", "VOC val (H07 T1)", float(p_), "p", "WS/sonuclar/h07_post.csv",
    "3 baseline best.pt uzerinde Δ tek-ornek t (df=2); turetilmis, keşifsel", n=3, turetme="ttest_1samp(tta_val.delta.s*, 0)")
add("tta_val.bs_alt90", "bootstrap_alt90", "tta", "VOC val (H07 T1)", float(t1["bs_alt90_puan"]), "puan", "WS/sonuclar/h07_post.csv",
    f"satir {t1i}, sutun bs_alt90_puan (B=200, yarim-B)", n=1)
add("tta_val.delta_A", "delta_yarim_A", "tta", "VOC val yarim-A", float(t1["delta_A_puan"]), "puan", "WS/sonuclar/h07_post.csv",
    f"satir {t1i}, sutun delta_A_puan")
add("tta_val.delta_B", "delta_yarim_B", "tta", "VOC val yarim-B", float(t1["delta_B_puan"]), "puan", "WS/sonuclar/h07_post.csv",
    f"satir {t1i}, sutun delta_B_puan")
add("tta_val.latency_orani", "gecikme_orani", "tta/R0", "VOC val oturumu (H07)", float(t1["lat_oran"]), "x", "WS/sonuclar/h07_post.csv",
    f"satir {t1i}, sutun lat_oran", oturum="val_H07", uyari_not="test oturumunda x2.29 (test.latency_orani.tta)")
add("tta.lit_beklenti_alt", "tta_lit_beklenen_alt", "tta", "literatur", 0.3, "puan", "WS/notlar/literatur-cipa.md",
    "TTA beklenen Δ +0.3...+1.0; kirmizi bayrak >+2.0 (YOLOv5x COCO +1.2, karisik etki)", kaynak="literatur", atif="ultralytics_2026_yolov5tta")
add("tta.lit_beklenti_ust", "tta_lit_beklenen_ust", "tta", "literatur", 1.0, "puan", "WS/notlar/literatur-cipa.md",
    "TTA beklenen Δ +0.3...+1.0", kaynak="literatur", atif="ultralytics_2026_yolov5tta")
add("tta.lit_kirmizi_bayrak", "tta_lit_kirmizi_bayrak", "tta", "literatur", 2.0, "puan", "WS/notlar/literatur-cipa.md",
    "kirmizi bayrak esigi > +2.0", kaynak="literatur", atif="ultralytics_2026_yolov5tta")

# ------------------------------------------------------------------------------------------- 5) H07 tum satirlar
for i, r in h07.iterrows():
    sat = r["satir"]
    ln = i + 2
    for m, col, bir, sc in (("map50_95", "map50_95", "puan", 100), ("map50", "map50", "puan", 100), ("ap_s", "ap_s", "puan", 100),
                            ("ap_m", "ap_m", "puan", 100), ("ap_l", "ap_l", "puan", 100), ("delta", "delta_full_puan", "puan", 1),
                            ("delta_A", "delta_A_puan", "puan", 1), ("delta_B", "delta_B_puan", "puan", 1), ("lat_ms", "lat_ms", "ms", 1),
                            ("lat_oran", "lat_oran", "x", 1), ("fps", "fps", "FPS", 1)):
        add(f"h07.{sat}.{m}", f"h07_{m}", sat, "VOC val (H07, baseline s0 best.pt)", float(r[col]) * sc, bir, "WS/sonuclar/h07_post.csv",
            f"satir {ln} ({sat}), sutun {col}" + ("; x100" if sc == 100 else ""), oturum="val_H07", etiket="kesifsel_on_eleme_n1")
    add(f"h07.{sat}.durum", "h07_durum", sat, "VOC val (H07)", str(r["durum"]), "", "WS/sonuclar/h07_post.csv", f"satir {ln}, sutun durum")
    add(f"h07.{sat}.dogrulandi", "h07_dogrulandi", sat, "VOC val (H07)", int(r["dogrulandi"]) if not pd.isna(r["dogrulandi"]) else -1, "0/1",
        "WS/sonuclar/h07_post.csv", f"satir {ln}, sutun dogrulandi (R0: -1 = yok)")
    add(f"h07.{sat}.aile", "h07_aile", sat, "VOC val (H07)", str(r["aile"]), "", "WS/sonuclar/h07_post.csv", f"satir {ln}, sutun aile")
for sat in h07["satir"]:
    for m in ("ap_s", "ap_m", "ap_l"):
        add(f"h07.{sat}.d_{m}", f"h07_delta_{m}", sat, "VOC val (H07)", val(f"h07.{sat}.{m}") - val(f"h07.R0.{m}"), "puan", "WS/sonuclar/h07_post.csv",
            f"satir {sat} - satir R0, sutun {m}; x100", oturum="val_H07", etiket="kesifsel_on_eleme_n1", turetme=f"h07.{sat}.{m} - h07.R0.{m}")
# R0 kaydi baseline s0 ile bit-bit ayni mi (P-kapisi)
assert abs(val("h07.R0.map50_95") - val("val.map50_95.baseline.s0")) < 1e-3

# ------------------------------------------------------------------------------------------- 6) ablasyon.csv metinsel (karar, degisen tek sey)
for i, r in abl.iterrows():
    k = r["kart"]
    for col in ("aile", "degisen_tek_sey", "karar", "n_seed"):
        add(f"abl.{k}.{col}", f"ablasyon_{col}", k, "ablasyon.csv", r[col] if col != "n_seed" else int(r[col]), "",
            "WS/sonuclar/ablasyon.csv", f"satir {i + 2} ({k}), sutun {col}")

# ------------------------------------------------------------------------------------------- 7) kademeli kapi (yalniz anlati; kanit degil)
for key, rid in (("baseline_s0_asama2", "20260930-070132_baseline_s0"), ("h04_asama2", "20261001-080838_h04_scale25_s0"),
                 ("h04_asama3", "20261001-085011_h04_scale25_s0")):
    mj = json.load(open(WS / "runs" / rid / "metrics.json", encoding="utf-8"))
    add(f"kademeli.val_map50_95.{key}", "val_map50_95", key, "VOC val", mj["val_map50_95"] * 100, "puan", f"WS/runs/{rid}/metrics.json",
        "anahtar val_map50_95; x100", etiket="kademeli_kapi", run_id=rid,
        not_="Asama 2 = 5 epoch on-ucus (saglik kapisi); KANIT DEGIL, yalniz 'kademeli kapi' anlatisi (B-14)")
add("kademeli.h04_asama2_delta", "h04_on_ucus_delta", "h04", "VOC val", val("kademeli.val_map50_95.h04_asama2") - val("kademeli.val_map50_95.baseline_s0_asama2"),
    "puan", "WS/runs/*/metrics.json", "H04 asama2 - baseline s0 asama2", etiket="kademeli_kapi",
    turetme="kademeli.val_map50_95.h04_asama2 - kademeli.val_map50_95.baseline_s0_asama2")

# ------------------------------------------------------------------------------------------- 8) butce / tur / koşum
TUR_OF_QUEUE = {None: 0, "k3-baseline": 0, "k3-baseline-v2": 0, "k4-tur1": 1, "k4-h03-seed": 1, "k4-tur2": 2, "k4-tur2b": 2,
                "k4-tur3a": 3, "k5-test": 4}
TUR_AD = {0: "Tur 0 (replikasyon)", 1: "Tur 1", 2: "Tur 2", 3: "Tur 3", 4: "K5 (test, tek acilis)"}
KART_OF_VARYANT = {"baseline": "baseline", "baseline_cm0": "kontrol (baseline_cm0)", "kalibrasyon": "kalibrasyon",
                   "h01_multiscale25": "H01", "h02_innerciou": "H02", "h03_boxcp": "H03", "h04_scale25": "H04", "h05_vfl": "H05",
                   "h06_box10": "H06", "h07_post": "H07", "h08_boxcp_doz": "H08", "baseline-test": "baseline-test",
                   "h03_boxcp-test": "H03-test", "baseline-tta-test": "TTA-test"}
rows = []
EXTRA_ROWS = []
for d in sorted(glob.glob(str(WS / "runs" / "2*"))):
    rid = os.path.basename(d)
    c = json.load(open(d + "/config.json", encoding="utf-8"))
    n = json.load(open(d + "/.done", encoding="utf-8"))
    _row = dict(run_id=rid, kuyruk=c.get("kuyruk"), asama=c.get("asama"), varyant=c.get("varyant"), rc=n.get("rc"), sure_sn=n.get("sure_sn") or 0.0)
    if c.get("kuyruk") == "k5-ek-seed":      # P1: bolum 12'de sondan eklenir (cekirdek id'leri sabit)
        EXTRA_ROWS.append(_row)
        continue
    rows.append(_row)
GECERSIZ = {"20260929-231803_baseline_s0": "yetim (rc=96)", "20261001-114442_h05_vfl_s0": "ariza (rc=95; canli guvenlik kurali yanlis-pozitif)"}
for r in rows:
    r["tur"] = TUR_OF_QUEUE[r["kuyruk"]]
    r["kart"] = KART_OF_VARYANT[r["varyant"]]
    r["kanit"] = bool(r["asama"] == 3 and r["rc"] == 0 and r["varyant"] not in ("kalibrasyon",))
    add(f"butce.kosum.{r['run_id']}", "kosum_suresi", r["kart"], "WS/runs", r["sure_sn"] / 3600, "sa", f"WS/runs/{r['run_id']}/.done",
        "anahtar sure_sn / 3600" + (" (sure_sn yok -> 0)" if r["sure_sn"] == 0.0 else ""), tur=r["tur"], kuyruk=r["kuyruk"], asama=r["asama"], rc=r["rc"],
        kanit_kosumu=r["kanit"], gecersiz=GECERSIZ.get(r["run_id"]))
tot = 0.0
kum = {}
for t in sorted(TUR_AD):
    ts = sum(r["sure_sn"] for r in rows if r["tur"] == t) / 3600
    nk = sum(1 for r in rows if r["tur"] == t)
    tot += ts
    kum[t] = tot
    add(f"butce.tur.{t}.sure_sa", "tur_gpu_sa", TUR_AD[t], "WS/runs", ts, "sa", "WS/runs/*/.done", f"tur={t}: sure_sn toplami/3600", n_kosum=nk,
        turetme=f"sum(butce.kosum.* | tur={t})")
    add(f"butce.tur.{t}.n_kosum", "tur_kosum_sayisi", TUR_AD[t], "WS/runs", nk, "adet", "WS/runs/*/.done", f"tur={t} .done dosya sayisi")
    add(f"butce.kumulatif.{t}", "kumulatif_gpu_sa", TUR_AD[t], "WS/runs", tot, "sa", "WS/runs/*/.done", f"tur<={t} toplam", turetme="cumsum(butce.tur.*)")
add("butce.toplam_sa", "toplam_gpu_sa", "tum", "WS/runs", tot, "sa", "WS/runs/*/.done", "47 .done sure_sn toplami/3600; gunluk sayaci 39.3 sa (CPU analizleri/oturum farki)",
    turetme="sum(butce.tur.*)")
add("butce.toplam_kosum", "toplam_kosum", "tum", "WS/runs", len(rows), "adet", "WS/runs/*/.done", ".done dosya sayisi")
add("butce.kanit_kosum", "kanit_kosum_sayisi", "tum", "WS/runs", sum(1 for r in rows if r["kanit"]), "adet", "WS/runs/*/.done",
    "asama=3 ve rc=0 (12 egitim + 1 H07 + 9 K5 test); H05 ariza ve yetim haric")
add("butce.hedef", "butce_hedef", "konu", "konu.md", 75.0, "sa", "WS/konu.md", "'toplam hesaplama butcesi ~75 GPU saati (ust sinir 90)'", etiket="on_kayit")
add("butce.tavan", "butce_tavan", "konu", "konu.md", 90.0, "sa", "WS/konu.md", "'ust sinir 90 sa'", etiket="on_kayit")
add("butce.tur_ust_sinir", "tur_ust_sinir", "konu", "konu.md", 5, "adet", "WS/konu.md", "'Tur ust siniri 5'", etiket="on_kayit")
add("butce.durma_kurali_tur", "durma_kurali_tetik_turu", "konu", "konu.md; orkestrasyon_durumu.md", 2, "tur", "WS/orkestrasyon_durumu.md",
    "'DURMA KURALI tetiklendi (ardisik 2 turda en iyi Δ<+0.30; tur1 +0.25, tur2 <0)' - tur-2 sonu", etiket="on_kayit")
# kart bazli koşum sayisi/saat (T6)
cards = {}
for r in rows:
    cards.setdefault((r["tur"], r["kart"]), []).append(r)
for (t, k), lst in cards.items():
    add(f"butce.kart.{t}.{k}.n_kosum", "kart_kosum_sayisi", k, TUR_AD[t], len(lst), "adet", "WS/runs/*/.done", f"tur={t}, kart={k}")
    add(f"butce.kart.{t}.{k}.n_asama3", "kart_asama3_sayisi", k, TUR_AD[t], sum(1 for r in lst if r["asama"] == 3 and r["rc"] == 0), "adet",
        "WS/runs/*/.done", f"tur={t}, kart={k}: asama=3 ve rc=0")
    add(f"butce.kart.{t}.{k}.sure_sa", "kart_gpu_sa", k, TUR_AD[t], sum(r["sure_sn"] for r in lst) / 3600, "sa", "WS/runs/*/.done", f"tur={t}, kart={k}")
for (t, k), lst in cards.items():
    add(f"butce.kart.{t}.{k}.n_gecersiz", "kart_gecersiz_kosum", k, TUR_AD[t], sum(1 for r in lst if r["run_id"] in GECERSIZ), "adet", "WS/runs/*/.done",
        f"tur={t}, kart={k}: yetim/ariza (kanit disi)")
add("butce.kart_listesi", "kart_listesi", "tum", "WS/runs", json.dumps([[t, k] for (t, k) in cards], ensure_ascii=False), "json", "WS/runs/*/.done", "tur,kart eslemesi")
assert abs(tot - 38.67) < 0.01
for t, v in zip(sorted(kum), (9.30, 25.15, 35.30, 38.46, 38.67)):
    assert abs(kum[t] - v) < 0.01, (t, kum[t], v)

# ------------------------------------------------------------------------------------------- 9) tur basina en iyi Δ (egitim kartlari)
add("tur.1.en_iyi_delta", "tur_en_iyi_delta", "H03 (3 seed)", "VOC val", val("dval.map50_95.h03.ort"), "puan", "WS/sonuclar/leaderboard-val.csv",
    "tur-1 kartlari H01 +0.07, H02 -0.22, H03 3-seed +0.25 -> en iyi H03", n=3, turetme="max(dval.map50_95.h01.s0, h02.s0, h03.ort)")
add("tur.1.en_iyi_delta_s0", "tur_en_iyi_delta_on_eleme", "H03 s0", "VOC val", val("dval.map50_95.h03.s0"), "puan", "WS/sonuclar/leaderboard-val.csv",
    "tur-1 seed-0 on-eleme en iyi: H03 s0 Δ", n=1, turetme="max(dval.map50_95.{h01,h02,h03}.s0)")
add("tur.2.en_iyi_delta", "tur_en_iyi_delta", "H04", "VOC val", max(val("dval.map50_95.h04.s0"), val("dval.map50_95.h05.s0"), val("dval.map50_95.h06.s0")),
    "puan", "WS/sonuclar/leaderboard-val.csv", "tur-2 egitim kartlari H04/H05/H06 en iyi (H07 egitimsiz haric)", n=1,
    turetme="max(dval.map50_95.{h04,h05,h06}.s0)")
add("tur.3.en_iyi_delta", "tur_en_iyi_delta", "H08", "VOC val", val("dval.map50_95.h08.s0"), "puan", "WS/sonuclar/leaderboard-val.csv",
    "tur-3: H08 tek kart", n=1, turetme="dval.map50_95.h08.s0")
add("tur.2.egitimsiz_en_iyi", "tur_egitimsiz_delta", "H07 T1", "VOC val", val("h07.T1.delta"), "puan", "WS/sonuclar/h07_post.csv",
    "tur-2 egitimsiz H07 T1 (ayri isaret)", n=1)

# ------------------------------------------------------------------------------------------- 10) uyarilar (beklenti ihlali / bayrak)
UY = {
    "dval.map50_95.h05.s0": "BEKLENTI IHLALI (beklenti-ihlali.md G6/D-etki): literatur araligi VFL -0.5...+0.3 (literatur-cipa.md); olculen -2.88. Mekanizma devrede (vfl_bce_grad_orani 0.34) -> uygulama hatasi kaniti yok; "
                           "yorum: ayarlanmamis VFL (alfa/gama/lr) bu kurulumda yakinsamayi yavaslatti; 'VFL kotu' degil 'bu kurulumda -2.88' (B-10).",
    "dval.map50_95.h08.s0": "H08 doz hedefi (0.8-1.2) altinda (0.670); gecikme orani 1.14 olcum artefakti (CPU paylasimi).",
}
for k, u in UY.items():
    BYKEY[k]["uyari"] = u

# ------------------------------------------------------------------------------------------- 10b) Faz 2.5 turetilmis kontroller
# (a) val/test seed s.d. orani F-testi (n=3: df=2,2)
fr = (val("gurultu.val_sd_baseline") / val("gurultu.test_sd_baseline")) ** 2
add("gurultu.f_orani", "varyans_orani_F", "baseline", "val/test", float(fr), "F", "WS/sonuclar/leaderboard-val.csv",
    "F = (sd_val / sd_test)^2, df=(2,2); turetilmis", n=3, turetme="(gurultu.val_sd_baseline/gurultu.test_sd_baseline)^2")
add("gurultu.f_p_tek", "varyans_orani_p_tek_yonlu", "baseline", "val/test", float(stats.f.sf(fr, 2, 2)), "p", "WS/sonuclar/leaderboard-val.csv",
    "P(F(2,2) > F); tek yonlu; sd_val > sd_test iddiasi n=3'te anlamli degil (cift yonlu = 2x)", n=3, turetme="scipy.stats.f.sf(F, 2, 2)",
    uyari_not="val s.d. 0.41 vs test 0.11 farki n=3'te istatistiksel olarak ayirt edilemez (F-testi)")
# (b) H03 mAP50: val Δ - test Δ seed-esli fark (val'de 3/3 pozitif kazanc test'e tasindi mi?)
d50v = [val(f"dval.map50.h03.s{k}") for k in SEEDS]
d50t = [val(f"dtest.map50.h03.s{k}") for k in SEEDS]
t_, p_ = stats.ttest_1samp(np.array(d50v) - np.array(d50t), 0.0)
add("tutarlilik.h03.map50_fark_ort", "h03_map50_val_delta_eksi_test_delta", "h03", "val-test", float(np.mean(d50v) - np.mean(d50t)), "puan",
    "WS/sonuclar/test-analiz.md", "T-D: mAP50 val +0.74 vs test -0.02; yeniden hesap", n=3, turetme="mean(dval.map50.h03) - mean(dtest.map50.h03)")
add("tutarlilik.h03.map50_fark_p", "h03_map50_val_test_fark_p_cift", "h03", "val-test", float(p_), "p", "WS/sonuclar/test-analiz.json",
    "seed-esli (val Δ50_s - test Δ50_s) tek-ornek t, df=2 (turetilmis; KESIFSEL, duzeltmesiz)", n=3, turetme="ttest_1samp(dval.map50.h03.s* - dtest.map50.h03.s*, 0)")
# (c) tek-seed on-eleme kapisinin yanlis-pozitif olasiligi (H0: gercek etki 0) normal yaklasim
for tag, sg in (("gozlenen_sd", val("dval.map50_95.h03.sd")), ("goruntu_se_tahmini", val("turet.se_val_per_seed_h03_tahmin"))):
    add(f"turet.on_eleme_yanlis_pozitif.{tag}", "tek_seed_on_eleme_yanlis_pozitif_olasiligi", "karar_kurali", "val", float(stats.norm.sf(val("on_kayit.on_eleme_esigi") / sg)), "olasilik",
        "WS/sonuclar/leaderboard-val.csv", f"P(Δ_s0 >= +0.30 | gercek etki=0), Δ_s0 ~ N(0, sigma={sg:.3f}); sigma = " + ("H03 val Δ seed s.d." if tag == "gozlenen_sd" else "test bootstrap SE x sqrt(4952/1000)"),
        etiket="tahmin_varsayimli", turetme="1 - Phi(0.30/sigma)", uyari_not="normal yaklasim; n=3'ten gelen sigma belirsiz")
add("turet.min_tespit_edilebilir_etki", "kritik_etki_esli_t", "karar_kurali", "test", float(stats.t.ppf(0.975, 2) * val("dtest.map50_95.h03.sd") / math.sqrt(3)), "puan",
    "WS/sonuclar/test-analiz.json", "t_0.975,df2 x sd_Δ(H03 test mAP50-95) / sqrt(3) (guc %50; karar.md '~0.5')", etiket="tahmin_varsayimli",
    turetme="t.ppf(0.975,2) * dtest.map50_95.h03.sd / sqrt(3)")
# (d) TTA beklenen piksel maliyeti: olcekler (WS/varyantlar/h07_post/README.md, T1) -> (1 + 0.83^2 + 0.67^2)
for i, sc in enumerate((1.0, 0.83, 0.67)):
    add(f"tta.olcek.{i}", "tta_olcek", "T1", "H07 yapilandirma", sc, "x", "WS/varyantlar/h07_post/README.md", "satir T: 'T1 Ultralytics [1,0.83,0.67]+yatay cevirme'", etiket="yapilandirma")
add("tta.beklenen_piksel_orani", "tta_beklenen_maliyet_orani", "T1", "H07", float(sum(val(f"tta.olcek.{i}") ** 2 for i in range(3))), "x", "WS/varyantlar/h07_post/README.md",
    "sum(olcek^2) = 1 + 0.83^2 + 0.67^2: model girdi pikseli orani (NMS/yontem yuku haric)", turetme="sum(tta.olcek.i^2)",
    uyari_not="kaba kestirim: gecikme ~ piksel; olculen test orani x2.29")
# (e) WBF'nin TTA uzerine etkisi (lit beklentisi TTA ile birlikte +0.2..+0.8): W1 - T1 ve AP50
add("h07.W1_eksi_T1.delta", "h07_W1_delta_eksi_T1_delta", "W1-T1", "VOC val (H07)", val("h07.W1.delta") - val("h07.T1.delta"), "puan", "WS/sonuclar/h07_post.csv",
    "W1 ve T1 satirlari", turetme="h07.W1.delta - h07.T1.delta", etiket="kesifsel_on_eleme_n1")
add("h07.W1_eksi_R0.map50", "h07_W1_map50_eksi_R0", "W1-R0", "VOC val (H07)", val("h07.W1.map50") - val("h07.R0.map50"), "puan", "WS/sonuclar/h07_post.csv",
    "W1 ve R0 satirlari, mAP50", turetme="h07.W1.map50 - h07.R0.map50", etiket="kesifsel_on_eleme_n1")
add("h07.T1_eksi_R0.map50", "h07_T1_map50_eksi_R0", "T1-R0", "VOC val (H07)", val("h07.T1.map50") - val("h07.R0.map50"), "puan", "WS/sonuclar/h07_post.csv",
    "T1 ve R0 satirlari, mAP50", turetme="h07.T1.map50 - h07.R0.map50", etiket="kesifsel_on_eleme_n1")
add("lit.wbf_tta_beklenen_alt", "wbf_tta_lit_beklenen_alt", "wbf", "literatur", 0.2, "puan", "WS/notlar/literatur-cipa.md", "WBF 'TTA ile birlikte +0.2...+0.8' (tek model, VOC sayisi bulunamadi)", kaynak="literatur", atif="solovyev_2021_wbf")
add("lit.wbf_tta_beklenen_ust", "wbf_tta_lit_beklenen_ust", "wbf", "literatur", 0.8, "puan", "WS/notlar/literatur-cipa.md", "WBF 'TTA ile birlikte +0.2...+0.8'", kaynak="literatur", atif="solovyev_2021_wbf")
# (f) pycocotools - Ultralytics mAP50-95 uyumu (hata analizi dogrulamasi)
for (arm, sd_) in HATA_TEST:
    add(f"ht.pycoco_fark.{arm}.s{sd_}", "pycocotools_eksi_ultralytics_map50_95", arm, "VOC07 test", val(f"ht.map5095.{arm}.s{sd_}") - val(f"test.map50_95.{arm}.s{sd_}"), "puan",
        "WS/sonuclar/hata-analizi-test_*.md.json; leaderboard-test.csv", "pycocotools map5095 - Ultralytics test_map50_95", seed=sd_, turetme="ht.map5095 - test.map50_95")
# (h) eslestirme duyarliligi: seed-esli t yerine Welch (bagimsiz) t -- sonuc degisiyor mu?
for arm in ("h03", "tta"):
    a = [val(f"test.map50_95.{arm}.s{k}") for k in SEEDS]
    b = [val(f"test.map50_95.baseline.s{k}") for k in SEEDS]
    tt_, pp_ = stats.ttest_ind(a, b, equal_var=False)
    add(f"welch.{arm}.p", "test_map50_95_welch_p_cift", arm, "VOC07 test", float(pp_), "p", "WS/sonuclar/leaderboard-test.csv",
        "Welch t (bagimsiz, esli degil); seed-esli pairing nominal oldugundan duyarlilik kontrolu", n=3, etiket="kesifsel", turetme="ttest_ind(arm, baseline, equal_var=False)")
# (i) P-sinifi mekanik kontroller (dosya varligi / benzersizlik)
_vr = [r for r in lbv["run_id"]]
add("kontrol.val_preds_sayisi", "p_kontrol_val_preds", "tum", "WS/runs", sum(os.path.exists(WS / "runs" / r / "val_preds.json") for r in _vr), "adet", "WS/runs/*/val_preds.json", "12 egitim kosumundan var olanlar")
add("kontrol.best_pt_sayisi", "p_kontrol_best_pt", "tum", "WS/runs", sum(os.path.exists(WS / "runs" / r / "train" / "weights" / "best.pt") for r in _vr), "adet", "WS/runs/*/train/weights/best.pt", "12 egitim kosumundan var olanlar")
_tr = [r for r in lbt["run_id"]]
add("kontrol.test_preds_sayisi", "p_kontrol_test_preds", "tum", "WS/runs", sum(os.path.exists(WS / "runs" / r / "test_preds.json") for r in _tr), "adet", "WS/runs/*/test_preds.json", "9 K5 kosumundan var olanlar")
add("kontrol.test_paket_sayisi", "p_kontrol_test_paket", "tum", "WS/runs", sum(os.path.exists(WS / "runs" / r / "test_paket.npz") for r in _tr), "adet", "WS/runs/*/test_paket.npz", "9 K5 kosumundan var olanlar")
add("kontrol.val_map_benzersiz", "p_kontrol_kollar_farkli", "tum", "VOC val", len(set(round(float(x), 6) for x in lbv["val_map50_95"])), "adet", "WS/sonuclar/leaderboard-val.csv",
    "12 kosumun val_map50_95 degerlerinin (6 hane) benzersiz sayisi: 12 = hicbir kol baska kolla bit-bit ayni degil (plasebo kol yok)")
# (g) val 07/12 kaynak ayrimi (KESIFSEL; evidence/scripts/val_kaynak_analiz.py)
vk = MW / "evidence" / "val_kaynak_analiz.json"
if vk.exists():
    vj = json.load(open(vk, encoding="utf-8"))
    add("valkaynak.n_07", "val_voc07_goruntu", "val", "VOC val", vj["n_07"], "goruntu", "MW/evidence/val_kaynak_analiz.json", "n_07", etiket="kesifsel_post_hoc")
    add("valkaynak.n_12", "val_voc12_goruntu", "val", "VOC val", vj["n_12"], "goruntu", "MW/evidence/val_kaynak_analiz.json", "n_12", etiket="kesifsel_post_hoc")
    for tag in ("07", "12", "tum"):
        for m in ("map50_95", "map50"):
            blk = vj["delta_h03"][f"{tag}.{m}"]
            for kk, x in (("ort", blk["ort"]), ("sd", blk["sd"])):
                add(f"valkaynak.h03.{tag}.{m}.{kk}", f"h03_val_delta_{m}_{kk}_kaynak_{tag}", "h03", f"VOC val ({tag})", float(x), "puan", "MW/evidence/val_kaynak_analiz.json",
                    f"delta_h03.{tag}.{m}.{kk} (pycocotools; H03-baseline seed-esli)", n=3, etiket="kesifsel_post_hoc")
            for k in range(3):
                add(f"valkaynak.h03.{tag}.{m}.s{k}", f"h03_val_delta_{m}_kaynak_{tag}", "h03", f"VOC val ({tag})", float(blk["dseed"][k]), "puan",
                    "MW/evidence/val_kaynak_analiz.json", f"delta_h03.{tag}.{m}.dseed[{k}]", seed=k, etiket="kesifsel_post_hoc")

# ------------------------------------------------------------------------------------------- 11) F6 nitel panel secimi (figures/scripts/f6_nitel.py ciktisi; kesifsel)
f6m = MW / "figures" / "data" / "f6_secim_meta.json"
f6d = MW / "figures" / "data" / "f6_panel.dat"
if f6m.exists() and f6d.exists():
    mj = json.load(open(f6m, encoding="utf-8"))
    F6SRC = "MW/figures/data/f6_secim_meta.json (figures/scripts/f6_nitel.py; test_paket.npz + test_preds.json, seed 0)"
    for k, kk, bir in (("n_havuz", "f6_aday_havuzu", "goruntu"), ("q1_baseline_goruntu_ap", "f6_baseline_goruntu_ap_q1", "AP"),
                       ("q3_baseline_goruntu_ap", "f6_baseline_goruntu_ap_q3", "AP"), ("n_kucuk_iceren_aday", "f6_kucuk_nesneli_aday", "goruntu"),
                       ("ort_delta_tta_tum", "f6_goruntu_ap_delta_tta_ort", "AP"), ("ort_delta_h03_tum", "f6_goruntu_ap_delta_h03_ort", "AP"),
                       ("kazanc_orani_tta_tum", "f6_tta_goruntu_kazanc_orani", "oran"), ("kayip_orani_tta_tum", "f6_tta_goruntu_kayip_orani", "oran"),
                       ("kazanc_orani_h03_tum", "f6_h03_goruntu_kazanc_orani", "oran"), ("kayip_orani_h03_tum", "f6_h03_goruntu_kayip_orani", "oran")):
        add(f"f6.{k}", kk, "nitel panel", "VOC07 test (4952, seed 0)", mj[k], bir, F6SRC, f"anahtar {k}", etiket="kesifsel_post_hoc")
    hdr = None
    for ln in open(f6d, encoding="utf-8"):
        t = ln.rstrip("\n").split(";")
        if hdr is None:
            hdr = t
            continue
        row = dict(zip(hdr, t))
        r = int(row["satir"])
        add(f"f6.satir{r}.goruntu", "f6_goruntu", row["kriter"], "VOC07 test", row["goruntu"], "", "MW/figures/data/f6_panel.dat", f"satir {r}, sutun goruntu", etiket="kesifsel_post_hoc")
        for col in ("n_gt", "n_kucuk", "ap_base", "ap_h03", "ap_tta", "tp_base", "fp_base", "fn_base", "tp_h03", "fp_h03", "fn_h03", "tp_tta", "fp_tta", "fn_tta"):
            add(f"f6.satir{r}.{col}", f"f6_{col}", row["kriter"], "VOC07 test", float(row[col]), "adet" if col.startswith(("n_", "tp", "fp", "fn")) else "AP",
                "MW/figures/data/f6_panel.dat", f"satir {r}, sutun {col}", etiket="kesifsel_post_hoc")

# ------------------------------------------------------------------------------------------- 12) MANUEL KAYITLAR (sabit id'ler) + P1: n=3 genislemesi
# 12a) Faz 5 tur-1 / sef tarafindan elle eklenen kayitlar E1528-E1568: evidence/manuel_kayitlar.json'dan AYNEN, SABIT id'lerle yuklenir
#      (uretici yeniden calistirildiginda silinmesin). Cekirdek E0001-E1527'yi bitirmis olmali.
assert len(R) == 1527, f"cekirdek kayit sayisi {len(R)} != 1527 (id kaymasi)"
for _mr in json.load(open(MW / "evidence" / "manuel_kayitlar.json", encoding="utf-8"))["kayitlar"]:
    assert _mr["id"] == f"E{len(R) + 1:04d}", (_mr["id"], len(R))
    assert _mr["anahtar"] not in BYKEY, _mr["anahtar"]
    R.append(_mr)
    BYKEY[_mr["anahtar"]] = _mr
assert len(R) == 1568 and R[-1]["id"] == "E1568"

# 12b) P1 (hakem B3): H01,H02,H04,H05(r),H06,H08 seed 1-2 (K5 sonrasi val replikasyonu, test YOK) -> E1569+ SONDAN
N3J = json.load(open(SON / "n3-analiz.json", encoding="utf-8"))
NEW_ARMS = ["h01", "h02", "h04", "h05", "h06", "h08"]
ALL7 = ["h01", "h02", "h03", "h04", "h05", "h06", "h08"]
T975 = float(stats.t.ppf(0.975, 2))
assert abs(T975 - val("turet.t_krit_df2")) < 1e-5
N3_METS = {"map50_95": "mAP50-95", "map50": "AP50", "ap_s": "AP_S", "ap_m": "AP_M", "ap_l": "AP_L", "map75": "AP75"}
EK = "K5 sonrasi val replikasyonu (k5-ek-seed kuyrugu; test yok)"
NEWRUNS = lbv_all[~lbv_all.index.isin(lbv.index)]
assert len(NEWRUNS) == 12
MEK_KK = {"h01": [("ms_farkli_boyut_sayisi", "h01_farkli_girdi_boyutu_sayisi", "adet"), ("ms_boyut_min", "h01_boyut_min", "px"), ("ms_boyut_max", "h01_boyut_max", "px")],
          "h02": [("inner_iou_fark_ort", "h02_inner_iou_fark_ort", "IoU")],
          "h04": [("scale_batch_sayisi", "h04_scale_batch_sayisi", "adet"), ("scale_args", "h04_scale_arg", "")],
          "h05": [("vfl_bce_grad_orani", "h05_vfl_bce_grad_orani", "oran"), ("vfl_bce_kayip_orani", "h05_vfl_bce_kayip_orani", "oran")],
          "h06": [("box_gain", "h06_box_gain", "")],
          "h08": [("boxcp_uygulanan_goruntu", "boxcp_uygulanan_goruntu", "adet"), ("boxcp_cagri", "boxcp_cagri", "adet"),
                  ("varyant_etki_sayaci", "boxcp_yapistirilan_nesne", "adet"), ("boxcp_yer_bulunamadi", "boxcp_yer_bulunamadi", "adet")]}
for i, r in NEWRUNS.iterrows():
    arm = ARM_VAL[r["varyant"]]
    seed = int(r["seed"])
    VAL_RUN[(arm, seed)] = r["run_id"]
    for m, col in VAL_COLS.items():
        add(f"val.{m}.{arm}.s{seed}", f"val_{m}", arm, "VOC val (1000 goruntu)", float(r[col]) * 100, "puan", "WS/sonuclar/leaderboard-val.csv",
            f"satir {i + 2} (run_id={r['run_id']}), sutun {col}; x100", etiket="tam_kosum_ek_seed", seed=seed, run_id=r["run_id"], not_=EK)
    add(f"val.egitim_sure_sa.{arm}.s{seed}", "egitim_sure", arm, "VOC train", float(r["egitim_sure_sa"]), "sa", "WS/sonuclar/leaderboard-val.csv",
        f"satir {i + 2}, sutun egitim_sure_sa", seed=seed, run_id=r["run_id"])
    add(f"val.latency_ms.{arm}.s{seed}", "latency_toplam", arm, "VOC val (300 gor., b1 FP16)", float(r["latency_toplam_ms"]), "ms",
        "WS/sonuclar/leaderboard-val.csv", f"satir {i + 2}, sutun latency_toplam_ms", oturum="val_egitim_sonu_olcumu", seed=seed, run_id=r["run_id"])
    add(f"val.fps.{arm}.s{seed}", "fps", arm, "VOC val (300 gor., b1 FP16)", float(r["fps"]), "FPS", "WS/sonuclar/leaderboard-val.csv",
        f"satir {i + 2}, sutun fps", oturum="val_egitim_sonu_olcumu", seed=seed, run_id=r["run_id"])
    add(f"val.en_iyi_epoch.{arm}.s{seed}", "en_iyi_epoch", arm, "VOC val", int(r["en_iyi_epoch"]), "epoch", "WS/sonuclar/leaderboard-val.csv",
        f"satir {i + 2}, sutun en_iyi_epoch", seed=seed, run_id=r["run_id"])
    # mekanizma devrede (G2) -- yeni seed'lerde de
    for col, met, bir in MEK_KK[arm]:
        add(f"mek.{col}.{arm}.s{seed}", met, arm, "VOC train (egitim sayaci)", float(r[col]), bir, "WS/sonuclar/leaderboard-val.csv", f"satir {i + 2}, sutun {col}",
            seed=seed, run_id=r["run_id"], amac="mekanizma devrede kaniti (G2): kanca baglanmis ve tetiklenmis")
    if arm == "h08":
        add(f"doz.nesne_per_mosaic.h08.s{seed}", "doz_nesne_per_mosaic", "h08", "VOC train", float(r["varyant_etki_sayaci"] / r["boxcp_cagri"]), "nesne/mosaic",
            "WS/sonuclar/leaderboard-val.csv", f"run_id={r['run_id']}: varyant_etki_sayaci / boxcp_cagri", seed=seed,
            uyari_not="sayac yaklasiktir (isci surecleri periyodik yazar)", turetme="varyant_etki_sayaci/boxcp_cagri")
        assert abs(val(f"doz.nesne_per_mosaic.h08.s{seed}") - r["boxcp_doz_nesne_per_mosaic"]) < 1e-3
add("doz.h08_ort", "doz_nesne_per_mosaic_ort", "h08", "VOC train", float(np.mean([val(f"doz.nesne_per_mosaic.h08.s{s}") for s in SEEDS])), "nesne/mosaic",
    "WS/sonuclar/leaderboard-val.csv", "3 seed ortalama", n=3, turetme="mean(doz.nesne_per_mosaic.h08.s0..s2)")
add("doz.carpan_h08_h03_n3", "doz_carpan", "h08/h03", "VOC train", val("doz.h08_ort") / val("doz.h03_ort"), "x", "WS/sonuclar/leaderboard-val.csv",
    "doz.h08_ort / doz.h03_ort (3 seed ortalamalari)", n=3, turetme="doz.h08_ort / doz.h03_ort",
    uyari_not="H08 hem yapistirma olasiligini (0.5->1.0) hem aday havuzunu (<=48 -> <=96 px) degistirir: doz ve havuz ayrilmamistir")

# ---- AP75 (pycocotools; hata-analizi.md.json) -- kalan kosumlar
for i, r in lbv_all.iterrows():
    arm = ARM_VAL[r["varyant"]]
    seed = int(r["seed"])
    if f"hv.map75.{arm}.s{seed}" in BYKEY:
        continue
    fn = f"hata-analizi_{r['run_id']}.md.json"
    j = json.load(open(SON / fn, encoding="utf-8"))
    hata_kayitlari("hv", "VOC val", j, f"WS/sonuclar/{fn}", fn, arm, seed)


# ---- seed-eslesmis Δ (s1, s2 ve AP75), ort/sd/p/t araligi (7 kart)
def _agg_keys(arm, m, ds, etiket="tam_kosum"):
    mean, sd_ = float(np.mean(ds)), sd1(ds)
    t_, p_c = stats.ttest_1samp(ds, 0.0)
    p_c = float(p_c)
    p_t = p_c / 2 if mean > 0 else 1 - p_c / 2
    half = T975 * sd_ / math.sqrt(3)
    src = "WS/sonuclar/leaderboard-val.csv" if m != "map75" else "WS/sonuclar/hata-analizi*.md.json"
    tur = f"dval.{m}.{arm}"
    for suf, met, v_, bir, tm in (("ort", "delta_val_{}_ort", mean, "puan", f"mean({tur}.s0..s2)"), ("sd", "delta_val_{}_sd", sd_, "puan", f"std(ddof=1; {tur}.s0..s2)"),
                                  ("p_cift", "delta_val_{}_p_cift", p_c, "p", f"scipy.stats.ttest_1samp({tur}.s0..s2, 0) (= seed-esli t, df=2)"),
                                  ("p_tek", "delta_val_{}_p_tek", p_t, "p", "p_cift/2 (mean>0) ya da 1-p_cift/2 (mean<=0); H1: Δ>0"),
                                  ("t_lo", "delta_val_{}_t_alt", mean - half, "puan", f"{tur}.ort - t(0.975,2)*{tur}.sd/sqrt(3)"),
                                  ("t_hi", "delta_val_{}_t_ust", mean + half, "puan", f"{tur}.ort + t(0.975,2)*{tur}.sd/sqrt(3)")):
        key = f"{tur}.{suf}"
        if key in BYKEY:                       # H03 cekirdek kayitlari zaten var: yalniz TUTARLILIK dogrula
            assert abs(BYKEY[key]["deger"] - v_) < 1e-6, (key, BYKEY[key]["deger"], v_)
            continue
        extra = {}
        if suf in ("t_lo", "t_hi"):
            extra["t_krit_kaynak"] = "turet.t_krit_df2 (E1551)"
        add(key, met.format(m), arm, "VOC val", v_, bir, src, "seed-eslesmis Δ uzerinden (turetilmis)", etiket=etiket, n=3, turetme=tm, **extra)
    return mean, sd_, p_c, p_t, mean - half, mean + half


for arm in ALL7:
    for m in N3_METS:
        ds = []
        for s in SEEDS:
            kd = f"dval.{m}.{arm}.s{s}"
            if kd not in BYKEY:
                if m == "map75":
                    d = val(f"hv.map75.{arm}.s{s}") - val(f"hv.map75.baseline.s{s}")
                    add(kd, "delta_val_map75", arm, "VOC val", d, "puan", "WS/sonuclar/hata-analizi*.md.json", f"map75 (pycocotools) kart s{s} - baseline s{s}",
                        etiket="tam_kosum" if (arm == "h03" or s > 0) else "on_eleme_n1", seed=s, turetme=f"hv.map75.{arm}.s{s} - hv.map75.baseline.s{s}")
                else:
                    d = val(f"val.{m}.{arm}.s{s}") - val(f"val.{m}.baseline.s{s}")
                    add(kd, f"delta_val_{m}", arm, "VOC val", d, "puan", "WS/sonuclar/leaderboard-val.csv",
                        f"satir run_id={VAL_RUN[(arm, s)]} - satir run_id={VAL_RUN[('baseline', s)]}, sutun {VAL_COLS[m]}; x100", etiket="tam_kosum_ek_seed", seed=s,
                        n_ref="ayni seed baseline", turetme=f"val.{m}.{arm}.s{s} - val.{m}.baseline.s{s}")
            ds.append(val(kd))
        if m != "map75" and arm in NEW_ARMS:
            x = [val(f"val.{m}.{arm}.s{s}") for s in SEEDS]
            add(f"val.{m}.{arm}.ort", f"val_{m}_ort", arm, "VOC val", float(np.mean(x)), "puan", "WS/sonuclar/leaderboard-val.csv", "3 seed ortalamasi (turetilmis)",
                etiket="tam_kosum", n=3, turetme=f"mean(val.{m}.{arm}.s0..s2)")
            add(f"val.{m}.{arm}.sd", f"val_{m}_sd", arm, "VOC val", sd1(x), "puan", "WS/sonuclar/leaderboard-val.csv", "3 seed std (ddof=1)",
                etiket="tam_kosum", n=3, turetme=f"std(ddof=1; val.{m}.{arm}.s0..s2)")
        mean, sd_, p_c, p_t, lo, hi = _agg_keys(arm, m, ds)
        # n3-analiz.json (WS cikti dosyasi) ile capraz dogrulama
        blk = N3J["kartlar"][arm.upper()]["metrik"][N3_METS[m]]
        for a, b in ((mean, blk["mean"]), (sd_, blk["sd"]), (p_c, blk["p2"]), (p_t, blk["p1"]), (lo, blk["lo"]), (hi, blk["hi"])):
            assert abs(a - b) < 1e-6, (arm, m, a, b)
        for x1, x2 in zip(ds, blk["seed"]):
            assert abs(x1 - x2) < 1e-6, (arm, m, x1, x2)
for m in N3_METS:
    if m != "map75":
        assert abs(val(f"val.{m}.baseline.ort") - N3J["baseline_ort"][N3_METS[m]]) < 1e-6

# ---- H05 beklenti ihlali uyarisi (n=3'te de suruyor; mekanizma devrede: mek.vfl_bce_grad_orani.h05.s0..s2)
for k in ("dval.map50_95.h05.s1", "dval.map50_95.h05.s2", "dval.map50_95.h05.ort"):
    BYKEY[k]["uyari"] = ("BEKLENTI IHLALI (G6/D-etki) 3 seed'de de suruyor: literatur araligi VFL -0.5...+0.3; olculen -2.5...-3.2 (seed Δ), ort -2.87, 3/3 seed negatif. "
                         "Mekanizma devrede (vfl_bce_grad_orani ~0.34 uc seed'de; E0632 ve mek.*.h05.s1/s2). Ayarsiz VFL (alfa/gama/cls kazanci) 'bu kurulumda' bulgusu; genellenmez.")

# ---- Holm (7 kart, mAP50-95 cift yonlu p; KESIFSEL; tablo renklendirmesi duzeltmesizdir)
_p7 = sorted(((val(f"dval.map50_95.{a}.p_cift"), a) for a in ALL7))
_run = 0.0
for rank, (pv, a) in enumerate(_p7):
    _run = max(_run, min(1.0, (len(_p7) - rank) * pv))
    add(f"dval.map50_95.{a}.holm7", "delta_val_map50_95_holm_p_7_kart", a, "VOC val", float(_run), "p", "WS/sonuclar/n3-analiz.json",
        "7 egitim kartinin cift yonlu seed-esli p'leri uzerinde Holm (turetilmis; KESIFSEL)", etiket="kesifsel", n=3, turetme="Holm(dval.map50_95.{arm}.p_cift, 7 test)")

# ---- on-eleme isabeti (n3-analiz.json)
add("on_kayit.tut_p_tek", "tut_p_esigi", "karar_kurali", "val", 0.10, "p", "WS/sonuclar/n3-analiz.md", "kabul olcutu (on-kayitli): ort Δ >= +0.40 VE tek yonlu p < 0.10", etiket="on_kayit")
for arm in ALL7:
    k = N3J["kartlar"][arm.upper()]
    add(f"n3.on_eleme.{arm}.karar", "on_eleme_karari", arm, "VOC val", k["on_eleme_karar"], "", "WS/sonuclar/n3-analiz.json", f"kartlar.{arm.upper()}.on_eleme_karar", etiket="tam_kosum")
    add(f"n3.on_eleme.{arm}.isabet", "on_eleme_isabeti", arm, "VOC val", k["on_eleme_isabeti"], "", "WS/sonuclar/n3-analiz.json", f"kartlar.{arm.upper()}.on_eleme_isabeti", etiket="tam_kosum")
    add(f"n3.on_eleme.{arm}.gecti", "on_eleme_kapisini_gecti", arm, "VOC val", int(bool(k["on_eleme_gecti"])), "0/1", "WS/sonuclar/n3-analiz.json", f"kartlar.{arm.upper()}.on_eleme_gecti", etiket="tam_kosum")
    add(f"n3.kabul.{arm}", "kabul_olcutu_karsilandi", arm, "VOC val", int(bool(k["kabul"])), "0/1", "WS/sonuclar/n3-analiz.json", f"kartlar.{arm.upper()}.kabul", etiket="tam_kosum")
    assert abs(val(f"dval.map50_95.{arm}.s0") - k["s0_delta"]) < 1e-9
    ok_kabul = val(f"dval.map50_95.{arm}.ort") >= val("on_kayit.tut_esigi") and val(f"dval.map50_95.{arm}.p_tek") < val("on_kayit.tut_p_tek")
    assert ok_kabul == bool(k["kabul"]), arm
_gecen = [a for a in ALL7 if N3J["kartlar"][a.upper()]["on_eleme_gecti"]]
_elenen = [a for a in ALL7 if not N3J["kartlar"][a.upper()]["on_eleme_gecti"]]
_dogru = [a for a in _elenen if str(val(f"n3.on_eleme.{a}.isabet")).startswith("DOGRU")]
add("n3.kabul_sayisi", "kabul_olcutunu_karsilayan_kart", "tum", "VOC val", int(sum(val(f"n3.kabul.{a}") for a in ALL7)), "adet", "WS/sonuclar/n3-analiz.json", "toplam kabul=true",
    etiket="tam_kosum", turetme="sum(n3.kabul.*)")
add("n3.on_eleme.gecen_sayisi", "on_eleme_kapisini_gecen_kart", "tum", "VOC val", len(_gecen), "adet", "WS/sonuclar/n3-analiz.json", "on_eleme_gecti=true sayisi (H03)", etiket="tam_kosum")
add("n3.on_eleme.elenen_sayisi", "on_eleme_ile_elenen_kart", "tum", "VOC val", len(_elenen), "adet", "WS/sonuclar/n3-analiz.json", "on_eleme_gecti=false sayisi", etiket="tam_kosum")
add("n3.on_eleme.dogru_sayisi", "on_eleme_isabetli_eleme", "tum", "VOC val", len(_dogru), "adet", "WS/sonuclar/n3-analiz.json",
    "elenen kartlardan n=3'te de kabul edilmeyenler (kacirilan kazanc yok)", etiket="tam_kosum", turetme="count(n3.on_eleme.*.isabet baslar 'DOGRU')")
assert (len(_gecen), len(_elenen), len(_dogru)) == (1, 6, 6) and val("n3.kabul_sayisi") == 0

# ---- gecikme orani (n=3 ortalama) ve gurultu bandi
for arm in NEW_ARMS:
    num_ = float(np.mean([val(f"val.latency_ms.{arm}.s{s}") for s in SEEDS]))
    den_ = float(np.mean([val(f"val.latency_ms.baseline.s{s}") for s in SEEDS]))
    add(f"val.latency_orani_n3.{arm}", "gecikme_orani", f"{arm}/baseline", "VOC val (egitim sonu olcumu)", num_ / den_, "x", "WS/sonuclar/leaderboard-val.csv",
        "latency_toplam_ms: kart 3 seed ort. / baseline 3 seed ort.; cikarim mimarisi ayni -> fark olcum gurultusu", oturum="val_egitim_sonu_olcumu", n=3,
        turetme=f"mean(val.latency_ms.{arm}.s*) / mean(val.latency_ms.baseline.s*)")
_lr = [val(f"val.latency_orani_n3.{a}") for a in NEW_ARMS] + [val("val.latency_orani.h03")]
add("gurultu.latency_orani_n3_min", "ayni_mimari_gecikme_orani_min", "7 kart", "VOC val", min(_lr), "x", "WS/sonuclar/leaderboard-val.csv",
    "7 kartin 3 seed ort. gecikme oranlarinin en kucugu", n=3, turetme="min(val.latency_orani_n3.*, val.latency_orani.h03)")
add("gurultu.latency_orani_n3_max", "ayni_mimari_gecikme_orani_max", "7 kart", "VOC val", max(_lr), "x", "WS/sonuclar/leaderboard-val.csv",
    "7 kartin 3 seed ort. gecikme oranlarinin en buyugu", n=3, turetme="max(val.latency_orani_n3.*, val.latency_orani.h03)")
_all_lat = [float(x) for x in lbv_all["latency_toplam_ms"]]
add("gurultu.latency_tek_kosum_max_oran", "tek_kosum_gecikme_en_buyuk_oran", "24 kosum", "VOC val", max(_all_lat) / float(np.median(_all_lat)), "x", "WS/sonuclar/leaderboard-val.csv",
    "24 egitim kosumunun latency_toplam_ms en buyugu / medyani (tek-kosum olcum gurultusu ust siniri)", turetme="max(latency)/median(latency)",
    uyari_not="en buyuk deger H08 s0 (CPU paylasimi) kosumudur")

# ---- tur basina en iyi egitim karti (n=3 ortalama)
for t_, arms in ((1, ("h01", "h02", "h03")), (2, ("h04", "h05", "h06")), (3, ("h08",))):
    best = max(arms, key=lambda a: val(f"dval.map50_95.{a}.ort"))
    add(f"tur.{t_}.en_iyi_delta_n3", "tur_en_iyi_delta_n3", best.upper(), "VOC val", val(f"dval.map50_95.{best}.ort"), "puan", "WS/sonuclar/n3-analiz.json",
        f"tur-{t_} egitim kartlarinin 3 seed ort. Δ'sinin en iyisi ({', '.join(a.upper() for a in arms)})", n=3, turetme=f"max(dval.map50_95.{{{','.join(arms)}}}.ort)")
add("tur.durma_kurali_n3_tetik", "durma_kurali_n3_dogrulama", "karar_kurali", "VOC val",
    int(val("tur.1.en_iyi_delta_n3") < 0.30 and val("tur.2.en_iyi_delta_n3") < 0.30), "0/1", "WS/sonuclar/n3-analiz.json",
    "durma kurali (ardisik iki turda en iyi egitim karti Δ < +0.30) n=3 degerleriyle de tetiklenir", turetme="tur.1.en_iyi_delta_n3 < 0.30 and tur.2.en_iyi_delta_n3 < 0.30")

# ---- butce: k5-ek-seed (tur 5), .done sure_sn
TUR_AD[5] = "Tur 5 (ek seed, n=3)"
KART_OF_VARYANT["h07_wbf_tani"] = "H07-WBF tani"
for r in EXTRA_ROWS:
    r["tur"] = 5
    r["kart"] = KART_OF_VARYANT[r["varyant"]]
    r["kanit"] = bool(r["asama"] == 3 and r["rc"] == 0)
    add(f"butce.kosum.{r['run_id']}", "kosum_suresi", r["kart"], "WS/runs", r["sure_sn"] / 3600, "sa", f"WS/runs/{r['run_id']}/.done", "anahtar sure_sn / 3600",
        tur=5, kuyruk=r["kuyruk"], asama=r["asama"], rc=r["rc"], kanit_kosumu=r["kanit"], gecersiz=None)
_ts5 = sum(r["sure_sn"] for r in EXTRA_ROWS) / 3600
add("butce.tur.5.sure_sa", "tur_gpu_sa", TUR_AD[5], "WS/runs", _ts5, "sa", "WS/runs/*/.done", "tur=5 (k5-ek-seed): sure_sn toplami/3600", n_kosum=len(EXTRA_ROWS),
    turetme="sum(butce.kosum.* | tur=5)")
add("butce.tur.5.n_kosum", "tur_kosum_sayisi", TUR_AD[5], "WS/runs", len(EXTRA_ROWS), "adet", "WS/runs/*/.done", "tur=5 .done dosya sayisi")
add("butce.kumulatif.5", "kumulatif_gpu_sa", TUR_AD[5], "WS/runs", tot + _ts5, "sa", "WS/runs/*/.done", "tur<=5 toplam", turetme="butce.kumulatif.4 + butce.tur.5.sure_sa")
add("butce.n3.toplam_sa", "toplam_gpu_sa_n3", "tum", "WS/runs", tot + _ts5, "sa", "WS/runs/*/.done", "60 .done sure_sn toplami/3600 (47 cekirdek + 13 k5-ek-seed)",
    turetme="butce.toplam_sa + butce.tur.5.sure_sa",
    uyari_not="gunluk sayaci 76.1 sa (butce.n3.gunluk_toplam_sa): .done toplamindan ~0.6 sa fazla (onceki gunluk 39.3 vs .done 38.67)")
add("butce.n3.toplam_kosum", "toplam_kosum_n3", "tum", "WS/runs", len(rows) + len(EXTRA_ROWS), "adet", "WS/runs/*/.done", ".done dosya sayisi",
    turetme="butce.toplam_kosum + butce.tur.5.n_kosum")
add("butce.n3.kanit_kosum", "kanit_kosum_sayisi_n3", "tum", "WS/runs", sum(1 for r in rows if r["kanit"]) + sum(1 for r in EXTRA_ROWS if r["kanit"]), "adet", "WS/runs/*/.done",
    "asama=3 ve rc=0: 22 cekirdek + 12 ek-seed egitim + 1 WBF tani", turetme="butce.kanit_kosum + 13")
add("butce.n3.ek_egitim_kosumu", "ek_seed_egitim_kosumu", "k5-ek-seed", "WS/runs", sum(1 for r in EXTRA_ROWS if r["varyant"] != "h07_wbf_tani"), "adet",
    "WS/runs/*/.done", "k5-ek-seed kuyrugu egitim kosumlari")
add("butce.n3.ek_egitim_sa", "ek_seed_egitim_gpu_sa", "k5-ek-seed", "WS/runs", sum(r["sure_sn"] for r in EXTRA_ROWS if r["varyant"] != "h07_wbf_tani") / 3600, "sa",
    "WS/runs/*/.done", "k5-ek-seed egitim kosumlari sure_sn toplami/3600 (WBF tani haric)")
add("butce.n3.gunluk_toplam_sa", "gunluk_toplam_gpu_sa", "tum", "gunluk", 76.1, "sa", "WS/gunluk/deney-gunlugu.md",
    "satir 32: 'Toplam GPU/CPU saati: ~76.1 sa (onceki ~39.3 + k5-ek-seed 132466 sn ~36.8 sa)'", etiket="gunluk_beyani")
_c5 = {}
for r in EXTRA_ROWS:
    _c5.setdefault(r["kart"], []).append(r)
_c5 = dict(sorted(_c5.items(), key=lambda kv: (kv[0].startswith("H07"), kv[0])))   # H01..H08, sonra WBF tani
for k, lst in _c5.items():
    add(f"butce.kart.5.{k}.n_kosum", "kart_kosum_sayisi", k, TUR_AD[5], len(lst), "adet", "WS/runs/*/.done", f"tur=5, kart={k}")
    add(f"butce.kart.5.{k}.n_asama3", "kart_asama3_sayisi", k, TUR_AD[5], sum(1 for r in lst if r["asama"] == 3 and r["rc"] == 0), "adet", "WS/runs/*/.done",
        f"tur=5, kart={k}: asama=3 ve rc=0")
    add(f"butce.kart.5.{k}.sure_sa", "kart_gpu_sa", k, TUR_AD[5], sum(r["sure_sn"] for r in lst) / 3600, "sa", "WS/runs/*/.done", f"tur=5, kart={k}")
    add(f"butce.kart.5.{k}.n_gecersiz", "kart_gecersiz_kosum", k, TUR_AD[5], 0, "adet", "WS/runs/*/.done", f"tur=5, kart={k}: gecersiz yok")
add("butce.kart_listesi_n3", "kart_listesi", "tum", "WS/runs", json.dumps([[5, k] for k in _c5], ensure_ascii=False), "json", "WS/runs/*/.done", "tur=5 kart eslemesi")



# ---- 12c) H05 kirmizi bayrak tanisi: yakinsama egrisi (egitim results.csv; baseline vs H05r, 3 seed) -- beklenti-ihlali.md §4(5)(c) icin DOLAYLI kanit (hipotez-duzeyi)
_YK = (20, 50, 90, 100)
for arm in ("baseline", "h05"):
    for s in SEEDS:
        rid_ = VAL_RUN[(arm, s)]
        df_ = pd.read_csv(WS / "runs" / rid_ / "train" / "results.csv")
        df_.columns = [c.strip() for c in df_.columns]
        for e in _YK:
            row_ = df_[df_["epoch"] == e].iloc[0]
            add(f"yakinsama.val_map50_95.ep{e}.{arm}.s{s}", "val_map50_95_epoch", arm, "VOC val (1000 goruntu)", float(row_["metrics/mAP50-95(B)"]) * 100, "puan",
                f"WS/runs/{rid_}/train/results.csv", f"epoch={e}, sutun metrics/mAP50-95(B); x100", seed=s, run_id=rid_, amac="H05r kirmizi bayrak tanisi (yakinsama hizi)")
        add(f"yakinsama.son10_kazanc.{arm}.s{s}", "son_10_epoch_val_kazanci", arm, "VOC val", val(f"yakinsama.val_map50_95.ep100.{arm}.s{s}") - val(f"yakinsama.val_map50_95.ep90.{arm}.s{s}"),
            "puan", f"WS/runs/{rid_}/train/results.csv", "ep100 - ep90 (val mAP50-95)", seed=s, turetme=f"yakinsama.val_map50_95.ep100.{arm}.s{s} - yakinsama.val_map50_95.ep90.{arm}.s{s}")
for e in _YK:
    gaps = [val(f"yakinsama.val_map50_95.ep{e}.h05.s{s}") - val(f"yakinsama.val_map50_95.ep{e}.baseline.s{s}") for s in SEEDS]
    add(f"yakinsama.h05.fark.ep{e}.ort", "h05_eksi_baseline_val_epoch", "h05", "VOC val", float(np.mean(gaps)), "puan", "WS/runs/*/train/results.csv",
        f"epoch {e}: H05r - baseline (seed-esli), 3 seed ortalamasi", n=3, turetme=f"mean(yakinsama.val_map50_95.ep{e}.h05.s* - yakinsama.val_map50_95.ep{e}.baseline.s*)")
    add(f"yakinsama.h05.fark.ep{e}.max", "h05_eksi_baseline_val_epoch_en_kucuk_seed", "h05", "VOC val", float(max(gaps)), "puan", "WS/runs/*/train/results.csv",
        f"epoch {e}: seed-esli farklarin en az negatifi (3/3 seed isaret kontrolu)", n=3, turetme="max(seed farklari)")
add("yakinsama.h05.en_iyi_epoch_100_sayisi", "h05_en_iyi_epoch_100_olan_seed", "h05", "VOC val", int(sum(1 for s in SEEDS if val(f"val.en_iyi_epoch.h05.s{s}") == 100)), "adet",
    "WS/sonuclar/leaderboard-val.csv", "val.en_iyi_epoch.h05.s0..s2 == 100 sayisi (3: son epoch'ta hala iyilesiyor)", n=3, turetme="count(val.en_iyi_epoch.h05.s*==100)")
add("yakinsama.baseline.en_iyi_epoch_100_sayisi", "baseline_en_iyi_epoch_100_olan_seed", "baseline", "VOC val", int(sum(1 for s in SEEDS if val(f"val.en_iyi_epoch.baseline.s{s}") == 100)), "adet",
    "WS/sonuclar/leaderboard-val.csv", "val.en_iyi_epoch.baseline.s0..s2 == 100 sayisi", n=3, turetme="count(val.en_iyi_epoch.baseline.s*==100)")
for k in ("yakinsama.h05.fark.ep20.ort", "yakinsama.h05.fark.ep50.ort", "yakinsama.h05.fark.ep100.ort"):
    BYKEY[k]["uyari_not"] = "dolayli kanit: H05 gerisi yakinsama hizinda (fark kapaniyor) -- 'cls sinyali zayifladi' hipotezi ile tutarli, ayirt edici deney degil"

# ---- 12d) FAZ 5 TUR-2 (hakem N1/N2): guc/kritik etki, on-eleme sans olasiligi, bagimsiz seed 1-2 ortalamasi -- E2486+ SONDAN (mevcut id'ler degismez)
from scipy.optimize import brentq
_DOSYA2 = "MW/notes/review-tur2.md (N1,N2) + evidence.json (dval.*, dtest.*)"
_T975, _T90, _T80 = float(stats.t.ppf(0.975, 2)), float(stats.t.ppf(0.90, 2)), float(stats.t.ppf(0.80, 2))


def _guc_nct(nc):
    """cift yonlu t-testi (df=2, alfa=0.05) gucu: nc = delta*sqrt(n)/sigma, merkezi olmayan t."""
    return float(stats.nct.sf(_T975, 2, nc) + stats.nct.cdf(-_T975, 2, nc))


_NC50 = float(brentq(lambda x: _guc_nct(x) - 0.5, 0.1, 30))
_NC80 = float(brentq(lambda x: _guc_nct(x) - 0.8, 0.1, 60))
_ZT80 = float(stats.norm.ppf(0.975) + stats.norm.ppf(0.80))
_TT80 = _T975 + _T80
add("guc.nc_gucu50_nct", "merkezi_olmayan_t_nc_guc50", "karar_kurali", "val/test", _NC50, "nc (delta*sqrt(3)/sigma)", _DOSYA2, "brentq: nct(df=2, nc) cift yonlu guc = 0.50; alfa=0.05, n=3",
    etiket="turetilmis_tur2", turetme="solve power_nct(nc; df=2, alpha=0.05 cift yonlu)=0.50")
add("guc.nc_gucu80_nct", "merkezi_olmayan_t_nc_guc80", "karar_kurali", "val/test", _NC80, "nc (delta*sqrt(3)/sigma)", _DOSYA2, "brentq: nct(df=2, nc) cift yonlu guc = 0.80; alfa=0.05, n=3",
    etiket="turetilmis_tur2", turetme="solve power_nct(nc; df=2, alpha=0.05 cift yonlu)=0.80")
add("guc.t_toplam_gucu80_yaklasik", "t_kantil_toplami_guc80", "karar_kurali", "val/test", _TT80, "kat (s/sqrt(3))", _DOSYA2, "t_{0.975,2} + t_{0.80,2} (yaklasik; guc 0.80)",
    etiket="turetilmis_tur2", turetme="t.ppf(0.975,2)+t.ppf(0.80,2)")
add("guc.z_toplam_gucu80_normal", "z_kantil_toplami_guc80", "karar_kurali", "val/test", _ZT80, "kat (s/sqrt(3))", _DOSYA2, "z_{0.975} + z_{0.80} (normal yaklasim; df=2'de iyimser)",
    etiket="turetilmis_tur2", turetme="norm.ppf(0.975)+norm.ppf(0.80)")


def _guc_kayit(pre, ad, s_key, s_val, veri):
    k50 = _T975 * s_val / math.sqrt(3)
    add(f"{pre}.kritik50", "kritik_etki_guc50", ad, veri, k50, "puan", _DOSYA2, f"t_{{0.975,2}}*{s_key}/sqrt(3) (%50 guc, geleneksel = %95 t araligi yari genisligi)",
        etiket="turetilmis_tur2", n=3, turetme=f"turet.t_krit_df2 * {s_key} / sqrt(3)", girdi_sd=s_val)
    add(f"{pre}.kritik50_nct", "kritik_etki_guc50_kesin", ad, veri, _NC50 * s_val / math.sqrt(3), "puan", _DOSYA2, "guc.nc_gucu50_nct * s_delta / sqrt(3) (merkezi olmayan t; s_delta bilinen kabul)",
        etiket="turetilmis_tur2", n=3, turetme=f"guc.nc_gucu50_nct * {s_key} / sqrt(3)")
    add(f"{pre}.kritik80_nct", "kritik_etki_guc80_kesin", ad, veri, _NC80 * s_val / math.sqrt(3), "puan", _DOSYA2, "guc.nc_gucu80_nct * s_delta / sqrt(3) (merkezi olmayan t; s_delta bilinen kabul)",
        etiket="turetilmis_tur2", n=3, turetme=f"guc.nc_gucu80_nct * {s_key} / sqrt(3)")
    add(f"{pre}.kritik80_t_yaklasik", "kritik_etki_guc80_yaklasik", ad, veri, _TT80 * s_val / math.sqrt(3), "puan", _DOSYA2, "(t_{0.975,2}+t_{0.80,2}) * s_delta / sqrt(3) (yaklasik)",
        etiket="turetilmis_tur2", n=3, turetme=f"guc.t_toplam_gucu80_yaklasik * {s_key} / sqrt(3)")


for _a in ALL7:
    _s = val(f"dval.map50_95.{_a}.sd")
    _guc_kayit(f"guc.val.{_a}", _a, f"dval.map50_95.{_a}.sd", _s, "VOC val (1000 goruntu), 3 seed")
    add(f"guc.val.{_a}.kabul_esik_ort", "kabul_icin_gereken_en_kucuk_ort_delta", _a, "VOC val", max(val("on_kayit.tut_esigi"), _T90 * _s / math.sqrt(3)), "puan", _DOSYA2,
        "kabul olcutu (ort>=+0.40 VE tek yonlu p<0.10): max(0.40, t_{0.90,2}*s_delta/sqrt(3))", etiket="turetilmis_tur2", n=3,
        turetme=f"max(on_kayit.tut_esigi, t.ppf(0.90,2) * dval.map50_95.{_a}.sd / sqrt(3))")
_sp = math.sqrt(float(np.mean([val(f"dval.map50_95.{a}.sd") ** 2 for a in ALL7])))
add("guc.val.ortak_sd_delta", "ortak_havuzlanmis_s_delta", "7 kart", "VOC val", _sp, "puan", _DOSYA2, "sqrt(mean(dval.map50_95.*.sd^2)), 7 kart, df=14 (esit varyans kabulu)",
    etiket="turetilmis_tur2", n=3, turetme="sqrt(mean_a(dval.map50_95.{a}.sd^2))",
    uyari_not="kartlarin Δ'lari ayni baseline seed kosumlarini paylasir (bagimsiz degil); s_delta kartlar arasi 0.12-0.79 (esit varyans yaklasiktir)")
_guc_kayit("guc.val.ortak", "7 kart (ortak s_delta)", "guc.val.ortak_sd_delta", _sp, "VOC val (1000 goruntu), 3 seed")
_c_lo = float(math.sqrt(14 / stats.chi2.ppf(0.975, 14)))
_c_hi = float(math.sqrt(14 / stats.chi2.ppf(0.025, 14)))
add("guc.val.ortak.kritik50_alt", "kritik_etki_guc50_ortak_alt_95", "7 kart (ortak s_delta)", "VOC val", val("guc.val.ortak.kritik50") * _c_lo, "puan", _DOSYA2,
    "ortak s_delta'nin %95 GA'sinin alt ucu (chi-kare df=14) ile %50 guclu kritik etki", etiket="turetilmis_tur2", n=3,
    turetme="guc.val.ortak.kritik50 * sqrt(14/chi2.ppf(0.975,14))")
add("guc.val.ortak.kritik50_ust", "kritik_etki_guc50_ortak_ust_95", "7 kart (ortak s_delta)", "VOC val", val("guc.val.ortak.kritik50") * _c_hi, "puan", _DOSYA2,
    "ortak s_delta'nin %95 GA'sinin ust ucu (chi-kare df=14) ile %50 guclu kritik etki", etiket="turetilmis_tur2", n=3,
    turetme="guc.val.ortak.kritik50 * sqrt(14/chi2.ppf(0.025,14))")
add("guc.val.ortak.kritik80_normal", "kritik_etki_guc80_normal", "7 kart (ortak s_delta)", "VOC val", _ZT80 * _sp / math.sqrt(3), "puan", _DOSYA2,
    "(z_{0.975}+z_{0.80}) * s_delta / sqrt(3) (normal yaklasim; df=2 icin iyimser)", etiket="turetilmis_tur2", n=3, turetme="guc.z_toplam_gucu80_normal * guc.val.ortak_sd_delta / sqrt(3)")
for _a, _ad in (("h03", "H03 (test)"), ("tta", "TTA (test)")):
    _guc_kayit(f"guc.test.{_a}", _ad, f"dtest.map50_95.{_a}.sd", val(f"dtest.map50_95.{_a}.sd"), "VOC07 test (4952), 3 seed")
assert abs(val("guc.test.h03.kritik50") - val("turet.min_tespit_edilebilir_etki")) < 1e-9
_ivs = [(val(f"dval.map50_95.{a}.t_lo"), val(f"dval.map50_95.{a}.t_hi")) for a in ALL7]
add("guc.val.aralik_030_kapsayan_kart", "t_araligi_+0.30_kapsayan_kart", "7 kart", "VOC val", sum(1 for lo, hi in _ivs if lo <= 0.30 <= hi), "adet", _DOSYA2,
    "%95 t araligi +0.30'u kapsayan kart sayisi", etiket="turetilmis_tur2", n=3, turetme="count(dval.map50_95.*.t_lo <= 0.30 <= dval.map50_95.*.t_hi)")
add("guc.val.aralik_040_kapsayan_kart", "t_araligi_+0.40_kapsayan_kart", "7 kart", "VOC val", sum(1 for lo, hi in _ivs if lo <= 0.40 <= hi), "adet", _DOSYA2,
    "%95 t araligi +0.40'i kapsayan kart sayisi", etiket="turetilmis_tur2", n=3, turetme="count(dval.map50_95.*.t_lo <= 0.40 <= dval.map50_95.*.t_hi)")
add("guc.val.aralik_marj030_icinde_kart", "t_araligi_pm030_icinde_kart", "7 kart", "VOC val", sum(1 for lo, hi in _ivs if lo >= -0.30 and hi <= 0.30), "adet", _DOSYA2,
    "%95 t araligi [-0.30,+0.30] icinde kalan kart sayisi (esdegerlik, TOST benzeri)", etiket="turetilmis_tur2", n=3, turetme="count(-0.30 <= t_lo and t_hi <= 0.30)")

# ---- N2: on-eleme isabeti sans olasiligi
_elenen2 = [a for a in ALL7 if a != "h03"]
add("onelem.sans_6_6.alt_sinir", "bilgisiz_kapi_6_6_eleme_olasiligi_alt", "karar_kurali", "val", (1 - val("turet.on_eleme_yanlis_pozitif.goruntu_se_tahmini")) ** 6, "olasilik", _DOSYA2,
    "(1 - p_gec)^6, p_gec = 0.275 (goruntu-SE tahmini): kapi hic bilgi tasimasa da 6 null kartin 6'sinin elenmesi", etiket="turetilmis_tur2",
    turetme="(1 - turet.on_eleme_yanlis_pozitif.goruntu_se_tahmini)^6")
add("onelem.sans_6_6.ust_sinir", "bilgisiz_kapi_6_6_eleme_olasiligi_ust", "karar_kurali", "val", (1 - val("turet.on_eleme_yanlis_pozitif.gozlenen_sd")) ** 6, "olasilik", _DOSYA2,
    "(1 - p_gec)^6, p_gec = 0.105 (gozlenen seed s.d.): kapi hic bilgi tasimasa da 6 null kartin 6'sinin elenmesi", etiket="turetilmis_tur2",
    turetme="(1 - turet.on_eleme_yanlis_pozitif.gozlenen_sd)^6")
_pe = 1.0
for _a in ALL7:
    _s = val(f"dval.map50_95.{_a}.sd")
    add(f"onelem.gec_olasilik.{_a}", "null_kartin_seed0_kapisini_gecme_olasiligi", _a, "VOC val", float(stats.norm.sf(val("on_kayit.on_eleme_esigi") / _s)), "olasilik", _DOSYA2,
        "1 - Phi(0.30/s_delta_kart); gercek etki 0 ve seed-0 Δ ~ N(0, s_delta^2) varsayimi", etiket="turetilmis_tur2", n=3, turetme=f"norm.sf(0.30 / dval.map50_95.{_a}.sd)")
    if _a in _elenen2:
        _pe *= 1 - val(f"onelem.gec_olasilik.{_a}")
add("onelem.sans_6_6.kart_sd", "bilgisiz_kapi_6_6_eleme_olasiligi_kart_sd", "karar_kurali", "val", _pe, "olasilik", _DOSYA2,
    "prod_{6 elenen kart} (1 - onelem.gec_olasilik.kart): kart basina s_delta ile", etiket="turetilmis_tur2", n=3, turetme="prod(1 - onelem.gec_olasilik.{h01,h02,h04,h05,h06,h08})")
# isabet tanimina dogrudan: elenen kartin n=3'te de kabul almamasi; seed-0 Δ ile n=3 karari BAGIMSIZ (bilgisiz kapi), gercek etki 0
_rng = np.random.default_rng(20261003)
_NMC = 400000
_pi = 1.0
for _a in _elenen2:
    _s = val(f"dval.map50_95.{_a}.sd")
    _x = _rng.normal(0.0, _s, (_NMC, 3))
    _m, _sdv = _x.mean(1), _x.std(1, ddof=1)
    _acc = float(np.mean((_m >= val("on_kayit.tut_esigi")) & (_m / (_sdv / math.sqrt(3)) > _T90)))
    add(f"onelem.n3_kabul_h0.{_a}", "null_kartin_n3_kabul_olasiligi_mc", _a, "VOC val", _acc, "olasilik", _DOSYA2,
        f"Monte Carlo {_NMC} tekrar, rng seed 20261003: Δ_s ~ N(0, s_delta^2), 3 seed; kabul = ort>=+0.40 ve t>t_{{0.90,2}} (tek yonlu p<0.10)", etiket="turetilmis_tur2", n=3,
        turetme=f"P(mean>=0.40 & t>t.ppf(0.90,2)); sigma = dval.map50_95.{_a}.sd")
    _pi *= 1 - _acc
add("onelem.sans_isabet_6_6.mc", "bilgisiz_kapi_6_6_isabet_olasiligi_mc", "karar_kurali", "val", _pi, "olasilik", _DOSYA2,
    "prod_{6 elenen kart}(1 - P(n=3'te kabul | etki 0)): seed-0 Δ ile n=3 karari bagimsiz (bilgisiz kapi); 'isabet' = n=3'te de kabul almama", etiket="turetilmis_tur2", n=3,
    turetme="prod(1 - onelem.n3_kabul_h0.{h01,h02,h04,h05,h06,h08})")
# bagimsiz seed 1-2 ortalamasi (n=3 ortalamasi seed 0'i, yani eleme kararini belirleyen tohumu, icerir)
_gec_s12 = 0
for _a in ALL7:
    _m12 = float(np.mean([val(f"dval.map50_95.{_a}.s1"), val(f"dval.map50_95.{_a}.s2")]))
    add(f"bagimsiz.val_map50_95.{_a}.ort_s12", "delta_val_map50_95_ort_seed1_2", _a, "VOC val", _m12, "puan", _DOSYA2,
        "yalniz seed 1-2 Δ ortalamasi (seed-0 / eleme tohumu haric; n=2: s.d./p raporlanmaz)", etiket="turetilmis_tur2", n=2,
        turetme=f"mean(dval.map50_95.{_a}.s1, dval.map50_95.{_a}.s2)",
        uyari_not=("H03 seed 0'da >=+0.30 ile secildi (yukari yanli); seed 1-2 bagimsiz" if _a == "h03" else "seed 0 <+0.30 ile elendi (kesilmis); n=3 ortalamasi asagi yanli, seed 1-2 bagimsiz"))
    if _a != "h03" and _m12 >= val("on_kayit.on_eleme_esigi"):
        _gec_s12 += 1
add("bagimsiz.on_eleme_esigi_ustu_sayisi", "bagimsiz_seed12_ort_esik_ustu_elenen_kart", "6 elenen kart", "VOC val", _gec_s12, "adet", _DOSYA2,
    "seed 1-2 ortalamasi >= +0.30 olan elenen kart sayisi", etiket="turetilmis_tur2", n=2, turetme="count(bagimsiz.val_map50_95.{h01,h02,h04,h05,h06,h08}.ort_s12 >= 0.30)")


# ------------------------------------------------------------------------------------------- yaz
meta = {"donanim": "RTX 5050 8 GB (Blackwell sm_120), Windows 10, tek GPU; torch 2.14.0+cu130; ultralytics 8.4.165 @004db34",
        "tarih": "2026-09-29 .. 2026-10-03 (kosumlar; ek seed 2026-10-02..03); evidence uretimi 2026-10-01, n=3 guncellemesi 2026-10-03",
        "tekrar_sayisi": "tum egitim kartlari + baseline + TTA: 3 seed (0,1,2); H01,H02,H04,H05,H06,H08 seed 1-2 = K5 sonrasi val replikasyonu (test yok); test yalniz baseline, H03, TTA",
        "kosul": "COCO on-egitimli YOLO11n -> PASCAL VOC07+12 ince ayar, 640 px, 100 epoch, batch 16; val=1000 goruntu (trainval seed0), test=VOC07 test 4952 (kilitli, tek acilis K5); "
                 "101-nokta AP, difficult nesneler etiketten atildi",
        "birim_notu": "puan = x100 (mAP 0.6349 = 63.49); Δ'lar puan farki",
        "yazilim_ws": "deney_workspace/2-1-enhancing-real-time-object-detection-on-pascal-v",
        "dislanan_kosumlar": {"20260929-231803_baseline_s0": "yetim rc=96", "20261001-114442_h05_vfl_s0": "ariza rc=95 (canli guvenlik kurali yanlis-pozitif)",
                              "kalibrasyon x3": "rc=98", "20260929-230007_baseline_s0": "manuel duman",
                              "asama 1-2 kosumlari": "kapi/kontrol; kanit degil (yalniz kademeli-kapi anlatisi, etiket kademeli_kapi)"},
        "olcek_notu": "Test hata kirilimi: scripts/analiz/hata_analizi.py --preds (K5 sonrasi, kesifsel, post-hoc; karar/secim yok)",
        "kod_koken": "ultralytics 8.4.165 @004db34b12f65fa5a160de2b1090f6943106bd35"}
out = {"deney_meta": meta, "kayitlar": R}
(MW / "evidence").mkdir(exist_ok=True)
json.dump(out, open(MW / "evidence" / "evidence.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"yazildi: evidence.json, {len(R)} kayit")
