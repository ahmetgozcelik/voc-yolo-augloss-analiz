#!/usr/bin/env python
"""Dis degerlendirme (KITTI saglamlik, kilit SONRASI, kesifsel): evidence.json'a `ek5.*` kayitlarini EKLER.

Mevcut kayitlara (ek5.* disindakilere) DOKUNMAZ. Idempotent: onceki calistirmanin `ek5.*` kayitlari silinip ayni baslangic id'sinden yeniden uretilir.
Kaynaklar: WS/sonuclar/ek_deney/saglamlik-analiz.json (WS/scripts/ek_deney/saglamlik_ek.py; MEVCUT test tahminleri, GPU yok, yeni test acilisi yok),
WS ve MW dosya zamanlari (durma kurali, replikasyon cipasi), WS koşum metrics.json (grad orani), MW/evidence.json (MC yeniden uretimi).
Etiket: kilit_sonrasi_kesifsel. Hicbir sayi elle yazilmaz. Birim: puan = x100.
Calistirma (MW kokunden): python evidence/scripts/build_ek_evidence_saglamlik.py
"""
import datetime as dt
import json
import math
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
ET = "kilit_sonrasi_kesifsel"
DOSYA = "WS/sonuclar/ek_deney/saglamlik-analiz.json"
DS = "KITTI test (1504 goruntu, 13 sahne grubu; DontCare=COCO crowd)"
MODELS = ["yolo11s", "yolo26n"]
CARDS = ["H01", "H02", "H03", "H04", "H05", "H06", "H08"]
SUBS = ["s8", "tramsiz7", "ps_misc_siz6"]
SINIF = ["car", "van", "truck", "pedestrian", "person_sitting", "cyclist", "tram", "misc"]


def J(p):
    return json.load(open(p, encoding="utf-8"))


d = J(EVID)
base = [r for r in d["kayitlar"] if not r["anahtar"].startswith("ek5.")]
_eski = [r for r in d["kayitlar"] if r["anahtar"].startswith("ek5.")]
START = (min(int(r["id"][1:]) for r in _eski) if _eski else max(int(r["id"][1:]) for r in base) + 1)
if _eski:
    assert max(int(r["id"][1:]) for r in base) == START - 1, "ek5 kayitlari kayit dizisinin sonunda olmali (idempotent yeniden uretim)"
BASE = {r["anahtar"]: r for r in base}
R, BY = [], {}


def ev(k):
    return BASE[k]["deger"]


def add(anahtar, metrik, yontem, veri_seti, deger, birim, dosya, konum, etiket=ET, kaynak="yazilim", **kw):
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


def zaman(p):
    return dt.datetime.fromtimestamp(Path(p).stat().st_mtime).replace(microsecond=0).isoformat()


S = J(EK / "saglamlik-analiz.json")
SO = S["sonuc"]
YON = S["yontem"]
MAD = {"yolo26n": "YOLO26n", "yolo11s": "YOLO11s"}

# ------------------------------------------------------------------ yontem
add("ek5.sag.yontem.B", "bootstrap_cogaltma_sayisi", "hiyerarsik bootstrap", DS, YON["B"], "adet", DOSYA, "yontem.B")
add("ek5.sag.yontem.grup_sayisi", "test_sahne_grubu_sayisi", "hiyerarsik bootstrap", DS, YON["grup_sayisi"], "adet", DOSYA, "yontem.grup_sayisi")
add("ek5.sag.yontem.yeniden_uretim_max_fark", "tam_orneklem_map_ile_kayitli_test_map_en_buyuk_mutlak_fark", "saglamlik_ek.py", DS, YON["yeniden_uretim_max_fark_map"], "mAP (0-1)", DOSYA,
    "yontem.yeniden_uretim_max_fark_map; 96 koşum (48 best + 48 last)")
add("ek5.sag.yontem.rng_grup", "rng_tohumu_grup_cekimi", "hiyerarsik bootstrap", DS, YON["rng_grup"], "tohum", DOSYA, "yontem.rng_grup")
add("ek5.sag.yontem.rng_tohum", "rng_tohumu_tohum_cekimi", "hiyerarsik bootstrap", DS, YON["rng_tohum"], "tohum", DOSYA, "yontem.rng_tohum")

# ------------------------------------------------------------------ 1) hiyerarsik bootstrap ve alt kume Δ (model x ckpt x kart x alt kume)
for m in MODELS:
    for ck in ("best", "last"):
        for c in CARDS:
            for sub in SUBS:
                e = SO[f"{m}/{ck}/{c}"][sub]
                p = f"ek5.sag.{m}.{ck}.{c}.{sub}"
                yont = f"{c}/{MAD[m]} ({ck}.pt; {sub})"
                kon = f"sonuc[{m}/{ck}/{c}].{sub}."
                add(p + ".delta", "delta_kart_eksi_B_tohum_ort", yont, DS, e["delta_ort"], "puan", DOSYA, kon + "delta_ort", n=3)
                add(p + ".hier_lo", "hiyerarsik_bootstrap_ga95_alt", yont, DS, e["hier_ga95"][0], "puan", DOSYA, kon + "hier_ga95[0]; B=2000; tohum+13 grup", n=3)
                add(p + ".hier_hi", "hiyerarsik_bootstrap_ga95_ust", yont, DS, e["hier_ga95"][1], "puan", DOSYA, kon + "hier_ga95[1]; B=2000; tohum+13 grup", n=3)
                add(p + ".hier_p_le0", "hiyerarsik_bootstrap_P_delta_le_0", yont, DS, e["hier_p_le0"], "olasilik", DOSYA, kon + "hier_p_le0 (cogaltmalarda Delta<=0 orani)", n=3)
                if sub == "s8":
                    add(p + ".grup_lo", "yalniz_grup_bootstrap_ga95_alt", yont, DS, e["grup_ga95"][0], "puan", DOSYA, kon + "grup_ga95[0] (tohumlar sabit; 13 grup)", n=3)
                    add(p + ".grup_hi", "yalniz_grup_bootstrap_ga95_ust", yont, DS, e["grup_ga95"][1], "puan", DOSYA, kon + "grup_ga95[1] (tohumlar sabit; 13 grup)", n=3)
                    add(p + ".grup_p_le0", "yalniz_grup_bootstrap_P_delta_le_0", yont, DS, e["grup_p_le0"], "olasilik", DOSYA, kon + "grup_p_le0", n=3)
                    add(p + ".tohum_lo", "yalniz_tohum_bootstrap_ga95_alt", yont, DS, e["tohum_ga95"][0], "puan", DOSYA, kon + "tohum_ga95[0] (grup sabit; 3 tohum yerine koyarak)", n=3)
                    add(p + ".tohum_hi", "yalniz_tohum_bootstrap_ga95_ust", yont, DS, e["tohum_ga95"][1], "puan", DOSYA, kon + "tohum_ga95[1] (grup sabit; 3 tohum yerine koyarak)", n=3)
                    add(p + ".tohum_p_le0", "yalniz_tohum_bootstrap_P_delta_le_0", yont, DS, e["tohum_p_le0"], "olasilik", DOSYA, kon + "tohum_p_le0", n=3)
                    add(p + ".poz_grup", "sahne_grubu_tohum_ort_delta_pozitif_sayisi", yont, DS, e["pozitif_grup"], "grup /13", DOSYA, kon + "pozitif_grup", n=3)
                    add(p + ".neg_grup", "sahne_grubu_tohum_ort_delta_negatif_sayisi", yont, DS, e["negatif_grup"], "grup /13", DOSYA, kon + "negatif_grup", n=3)
                    add(p + ".poz_grup_katki", "bir_grup_disarida_katki_pozitif_sayisi", yont, DS, e["pozitif_grup_loo_katki"], "grup /13", DOSYA, kon + "pozitif_grup_loo_katki", n=3)

# tutarlilik denetimleri: ek5 tam-orneklem Δ, mevcut ek.test kayitlariyla ayni (best); last Δ ek.test.*.last_delta_ort ile ayni
for m in MODELS:
    for c in CARDS:
        assert abs(BY[f"ek5.sag.{m}.best.{c}.s8.delta"]["deger"] - ev(f"ek.test.{m}.{c}.delta_ort")) < 2e-3, (m, c, "best")
        assert abs(BY[f"ek5.sag.{m}.last.{c}.s8.delta"]["deger"] - ev(f"ek.test.{m}.{c}.last_delta_ort")) < 2e-3, (m, c, "last")

# ------------------------------------------------------------------ 2) sahne grubu basina Δ (best) ve katki
GRUP = list(S["gruplar"].keys())
for m in MODELS:
    for c in CARDS:
        e = SO[f"{m}/best/{c}"]["s8"]
        for g in GRUP:
            gv = e["grup"][g]
            p = f"ek5.sag.{m}.best.{c}.grup.{g}"
            yont = f"{c}/{MAD[m]} (best.pt)"
            if gv["delta_tek_grup"] is not None:
                add(p + ".delta_tek", "grup_ici_map_delta_tohum_ort", yont, DS + f"; grup {g}", gv["delta_tek_grup"], "puan", DOSYA, f"sonuc[{m}/best/{c}].s8.grup[{g}].delta_tek_grup", n=3)
            add(p + ".katki", "toplam_delta_eksi_bu_grup_disarida_delta", yont, DS + f"; grup {g}", gv["katki_puan"], "puan", DOSYA, f"sonuc[{m}/best/{c}].s8.grup[{g}].katki_puan", n=3)
# grup meta
for g in GRUP:
    gm = S["gruplar"][g]
    add(f"ek5.sag.grup.{g}.goruntu", "test_grubu_goruntu_sayisi", "test sahne grubu", DS, gm["goruntu"], "adet", DOSYA, f"gruplar[{g}].goruntu")
    for sn in SINIF:
        add(f"ek5.sag.grup.{g}.n_{sn}", f"test_grubu_{sn}_nesne_sayisi", "test sahne grubu", DS, gm["sinif_sayilari"][sn], "adet", DOSYA, f"gruplar[{g}].sinif_sayilari.{sn}")
tram = {g: S["gruplar"][g]["sinif_sayilari"]["tram"] for g in GRUP}
ntram = sum(tram.values())
add("ek5.sag.tram.toplam", "test_tram_nesne_sayisi", "test sahne grubu", DS, ntram, "adet", DOSYA, "sum_g gruplar[g].sinif_sayilari.tram", kaynak="turetilmis")
add("ek5.sag.tram.grup_sayisi", "tram_iceren_test_grubu_sayisi", "test sahne grubu", DS, sum(1 for x in tram.values() if x > 0), "grup /13", DOSYA, "count(tram>0)", kaynak="turetilmis")
srt = sorted(tram.items(), key=lambda kv: -kv[1])
add("ek5.sag.tram.en_buyuk_grup", "tram_en_cok_olan_grup", "test sahne grubu", DS, srt[0][0], "grup", DOSYA, "argmax_g tram", kaynak="turetilmis")
add("ek5.sag.tram.en_buyuk_grup_sayi", "tram_en_cok_olan_grupta_tram_sayisi", "test sahne grubu", DS, srt[0][1], "adet", DOSYA, "max_g tram", kaynak="turetilmis")
add("ek5.sag.tram.en_buyuk_grup_pay", "tram_en_cok_olan_grubun_payi", "test sahne grubu", DS, srt[0][1] / ntram, "oran", DOSYA, "max_g tram / toplam tram", kaynak="turetilmis")
add("ek5.sag.tram.en_buyuk2_grup_pay", "tram_en_cok_iki_grubun_toplam_payi", "test sahne grubu", DS, (srt[0][1] + srt[1][1]) / ntram, "oran", DOSYA, "(ilk iki grup tram) / toplam", kaynak="turetilmis")

# turetilmis: YOLO11s H01'in toplam Δ'sinda tram grubunun (drive_0056) payi ve pozitif grupların görüntü payı
_g56 = srt[0][0]
_e = SO["yolo11s/best/H01"]["s8"]
add("ek5.sag.yolo11s.best.H01.tram_grubu_katki_pay", "tram_grubunun_toplam_delta_icindeki_payi", "H01/YOLO11s (best.pt)", DS, _e["grup"][_g56]["katki_puan"] / _e["delta_ort"], "oran", DOSYA,
    f"sonuc[yolo11s/best/H01].s8.grup[{_g56}].katki_puan / delta_ort", kaynak="turetilmis", n=3, turetme="katki_puan(tram grubu) / delta_ort")
_poz_img = sum(S["gruplar"][g]["goruntu"] for g in GRUP if (_e["grup"][g]["delta_tek_grup"] or 0) > 0)
add("ek5.sag.yolo11s.best.H01.poz_grup_goruntu_pay", "tek_grup_delta_pozitif_gruplarin_goruntu_payi", "H01/YOLO11s (best.pt)", DS, _poz_img / sum(S["gruplar"][g]["goruntu"] for g in GRUP), "oran", DOSYA,
    "sum(gruplar[g].goruntu | delta_tek_grup>0) / 1504", kaynak="turetilmis", n=3, turetme="goruntu payi")
# ------------------------------------------------------------------ 3) sinif basi Δ (best, tam orneklem)
for m in MODELS:
    for c in CARDS:
        e = SO[f"{m}/best/{c}"]["s8"]
        for sn in SINIF:
            add(f"ek5.sag.{m}.best.{c}.sinif.{sn}.delta", f"sinif_ap5095_delta_{sn}", f"{c}/{MAD[m]} (best.pt)", DS, e["sinif_delta"][sn], "puan", DOSYA,
                f"sonuc[{m}/best/{c}].s8.sinif_delta.{sn}", n=3)

# ------------------------------------------------------------------ 4) on-eleme MC yeniden uretimi (kaynak: evidence.json dval sd, build_evidence.py L1052-1087)
ELENEN = ["h01", "h02", "h04", "h05", "h06", "h08"]
T90 = stats.t.ppf(0.90, 2)
rng = np.random.default_rng(20261003)
pi = 1.0
for a in ELENEN:
    s = ev(f"dval.map50_95.{a}.sd")
    x = rng.normal(0.0, s, (400000, 3))
    mm, sd = x.mean(1), x.std(1, ddof=1)
    acc = float(np.mean((mm >= 0.40) & (mm / (sd / math.sqrt(3)) > T90)))
    pi *= 1 - acc
rec_pi = add("ek5.onele_mc.yeniden_uretim_p_6_6", "bilgisiz_kapi_6_6_isabet_olasiligi_mc_yeniden_uretim", "ön-eleme MC", "VOC val", float(pi), "olasilik",
             "evidence/scripts/onele_mc_duyarlilik.py", "build_evidence.py L1069-1087 ile ayni rng (20261003), 400000 cekim/kart, ayni girdiler", kaynak="turetilmis", n=3,
             turetme="prod_a (1 - P(mean>=0.40 & t>t_{0.90,2})), Delta_s ~ N(0, sd_a^2)")
assert abs(rec_pi["deger"] - ev("onelem.sans_isabet_6_6.mc")) < 1e-12, "MC yeniden uretimi E2560 ile ayni olmali"
add("ek5.onele_mc.nmc", "mc_cekim_sayisi_kart_basina", "ön-eleme MC", "VOC val", 400000, "adet", "MW/evidence/scripts/build_evidence.py", "_NMC (satir 1071)", kaynak="turetilmis")
add("ek5.onele_mc.sigma_alt_a", "ortak_sigma_ust_sinir_kaynagi_h03_gorsel_se", "ön-eleme sans A", "VOC val", ev("turet.se_val_per_seed_h03_tahmin"), "puan", "MW/evidence/scripts/build_evidence.py",
    "satir 641-649 (E0931 = test bootstrap SE x sqrt(4952/1000))", kaynak="turetilmis")
add("ek5.onele_mc.sigma_ust_b", "ortak_sigma_h03_val_seed_sd", "ön-eleme sans A", "VOC val", ev("dval.map50_95.h03.sd"), "puan", "MW/evidence/scripts/build_evidence.py",
    "satir 641-649 (E dval.map50_95.h03.sd)", kaynak="turetilmis")

# ------------------------------------------------------------------ 5) H05 gradyan normu orani: koşum basina, ölçüm tanimi
GR = {}
for s_, rid in ((0, "20261001-145045_h05_vfl_s0"), (1, "20261002-001746_h05_vfl_s1"), (2, "20261002-033917_h05_vfl_s2")):
    mt = J(WS / "runs" / rid / "metrics.json")
    GR[s_] = mt
    add(f"ek5.h05_grad.s{s_}.oran", "vfl_bce_grad_normu_orani", "H05 (VOC, YOLO11n, aşama 3)", "VOC val (egitim sirasinda ornekleme)", mt["vfl_bce_grad_orani"], "oran",
        f"WS/runs/{rid}/metrics.json", "vfl_bce_grad_orani = grad_vfl_toplam / grad_bce_toplam; logit tensorune gore L2 norm; her 200. egitim cagrisi", etiket=ET, n=1, tohum=s_, run_id=rid)
    add(f"ek5.h05_grad.s{s_}.olcum_n", "gradyan_olcum_sayisi", "H05 (VOC, YOLO11n, aşama 3)", "VOC val", mt["vfl_olcum_n"], "adet", f"WS/runs/{rid}/metrics.json", "vfl_olcum_n", tohum=s_, run_id=rid)
    add(f"ek5.h05_grad.s{s_}.cagri", "vfl_egitim_cagri_sayisi", "H05 (VOC, YOLO11n, aşama 3)", "VOC val", mt["varyant_etki_sayaci"], "adet", f"WS/runs/{rid}/metrics.json", "varyant_etki_sayaci", tohum=s_, run_id=rid)
add("ek5.h05_grad.ort", "vfl_bce_grad_orani_3_tohum_ort", "H05 (VOC, YOLO11n, aşama 3)", "VOC val", float(np.mean([GR[i]["vfl_bce_grad_orani"] for i in range(3)])), "oran",
    "WS/runs/*h05_vfl_s{0,1,2}/metrics.json", "mean(ek5.h05_grad.s*.oran)", kaynak="turetilmis", n=3)
add("ek5.h05_grad.aralik", "gradyan_olcum_araligi_adim", "H05 (VOC)", "VOC val", 200, "adim", "WS/varyantlar/h05_vfl/vfl_patch.py", "satir 30 _OLCUM_ARALIGI; olcum kosulu cagri % 200 == 1 (satir 58)")
for rid, ad in (("20261001-080455_h05_vfl_s0", "duman"), ("20261001-081827_h05_vfl_s0", "on_ucus")):
    mt = J(WS / "runs" / rid / "metrics.json")
    add(f"ek5.h05_grad.{ad}.oran", f"vfl_bce_grad_normu_orani_{ad}", "H05 (VOC, YOLO11n)", "VOC val", mt["vfl_bce_grad_orani"], "oran", f"WS/runs/{rid}/metrics.json",
        f"vfl_bce_grad_orani; olcum_n={mt['vfl_olcum_n']}", run_id=rid, uyari_not="kanit degil: duman/on-ucus; oran egitim ilerledikce dusuyor (aşama 3 = 0.34)")

# ------------------------------------------------------------------ 6) durma kurali ve H08: dosya/koşum zamanlari
def tz(anahtar, metrik, p, konum, **kw):
    add(anahtar, metrik, "durma kurali / H08", "VOC deney çalışma alanı", zaman(WS / p) if not str(p).startswith("RUN:") else None, "zaman damgasi", f"WS/{p}", konum,
        etiket=ET, kaynak="yazilim", **kw)


def done_bitis(rid):
    return J(WS / "runs" / rid / ".done")["bitis"]


tz("ek5.durma.siralama_mtime", "SIRALAMA.md_tur3_bolumu_dosya_zamani", "hipotezler/SIRALAMA.md", "dosya mtime (Tur-3 bolumu satir 122-157 'H04-H06 sonuclari bu yazim anında YOKTUR')")
tz("ek5.durma.h08_kod_mtime", "H08_kodu_dosya_zamani", "varyantlar/h08_boxcp_doz/boxcp_doz.py", "dosya mtime")
tz("ek5.durma.h08_beklenti_mtime", "H08_kuyruk_beklentisi_dosya_zamani", "kuyruk/k4-tur3a.beklenti.json", "dosya mtime; _aciklama: 'Kosumdan ONCE yazildi (kart H08.md mtime'indan sonra)'")
tz("ek5.durma.h08_kuyruk_mtime", "H08_kuyrugu_dosya_zamani", "kuyruk/k4-tur3a.jsonl", "dosya mtime")
tz("ek5.durma.h08_kart_mtime", "H08.md_dosya_zamani", "hipotezler/H08.md", "dosya mtime (toplu yeniden yazma; kart yazim zamani icin KULLANILAMAZ)",
   uyari_not="H01,H02,H04,H05,H06,H08 ayni saniyede (2026-10-03 13:14:33): toplu yazma; kartin ön-kayıt zamani bu damgadan çıkarılamaz")
ZH04, ZH06, ZH05 = done_bitis("20261001-085011_h04_scale25_s0"), done_bitis("20261001-115531_h06_box10_s0"), done_bitis("20261001-145045_h05_vfl_s0")
for ad, rid, z in (("h04", "20261001-085011_h04_scale25_s0", ZH04), ("h06", "20261001-115531_h06_box10_s0", ZH06), ("h05r", "20261001-145045_h05_vfl_s0", ZH05)):
    add(f"ek5.durma.tur2_{ad}_bitis", f"tur2_{ad}_kosum_bitisi", "durma kurali", "VOC deney çalışma alanı", z, "zaman damgasi", f"WS/runs/{rid}/.done", "bitis")
h08_run = "20261001-182518_h08_boxcp_doz_s0"
add("ek5.durma.h08_tam_kosum_baslangic", "H08_tam_kosum_config_zamani", "durma kurali", "VOC deney çalışma alanı", zaman(WS / "runs" / h08_run / "config.json"), "zaman damgasi",
    f"WS/runs/{h08_run}/config.json", "dosya mtime")
add("ek5.durma.h08_duman_bitis", "H08_duman_kosum_bitisi", "durma kurali", "VOC deney çalışma alanı", done_bitis("20261001-181324_h08_boxcp_doz_s0"), "zaman damgasi",
    "WS/runs/20261001-181324_h08_boxcp_doz_s0/.done", "bitis")
t_tetik = dt.datetime.fromisoformat(ZH05)
t_h08 = max(dt.datetime.fromisoformat(BY[k]["deger"]) for k in ("ek5.durma.siralama_mtime", "ek5.durma.h08_kod_mtime", "ek5.durma.h08_beklenti_mtime", "ek5.durma.h08_kuyruk_mtime"))
add("ek5.durma.hazirlik_tetik_farki_sa", "H08_hazirligi_ile_durma_kurali_tetiklenmesi_arasi_sure", "durma kurali", "VOC deney çalışma alanı", (t_tetik - t_h08).total_seconds() / 3600, "saat",
    "WS dosya zamanlari", "(tur-2 son kart bitisi) - (SIRALAMA/H08 kodu/kuyruk beklentisi en gec mtime)", kaynak="turetilmis",
    uyari_not="mtime/.done damgalari kriptografik degil; git ilk commit 2026-10-16 (toplu) kullanilamaz")
add("ek5.durma.git_ilk_commit", "WS_dosyalarinin_ilk_git_commit_zamani", "durma kurali", "VOC deney çalışma alanı", "2026-10-16T12:08:58", "zaman damgasi", "git log --follow (64a8f52)",
    "H08.md, SIRALAMA.md, konu.md, deney-gunlugu.md ilk commit'i toplu 'Dalga 2 hazirlik paketi'", uyari_not="deney 2026-09-29..10-03; commit 13 gun sonra: sirayi kanitlamaz")

# ------------------------------------------------------------------ 7) replikasyon cipasi: ön-kayıt zamanı
lc = WS / "notlar" / "literatur-cipa.md"
add("ek5.replikasyon.cipa_mtime", "literatur_cipa_notu_dosya_zamani", "replikasyon cipasi", "VOC07 test", zaman(lc), "zaman damgasi", "WS/notlar/literatur-cipa.md",
    "dosya mtime (cipa tablosu, kapı komutları --lit 0.610 --lit-ust 0.635 --tolerans 0.012)")
for s_, rid in ((0, "20260930-072103_baseline_s0"), (1, "20260930-101624_baseline_s1"), (2, "20260930-131427_baseline_s2")):
    add(f"ek5.replikasyon.baseline_test_s{s_}", "baseline_voc07_test_olcum_zamani", "replikasyon cipasi", "VOC07 test", zaman(WS / "runs" / rid / "metrics.json"), "zaman damgasi",
        f"WS/runs/{rid}/metrics.json", "dosya mtime (rep_test_* yazimi)", tohum=s_, run_id=rid)
add("ek5.replikasyon.kapi_raporu_mtime", "replikasyon_kapisi_raporu_zamani", "replikasyon cipasi", "VOC07 test", zaman(WS / "sonuclar" / "replikasyon_map50_95.md"), "zaman damgasi",
    "WS/sonuclar/replikasyon_map50_95.md", "dosya mtime")
fark = (dt.datetime.fromisoformat(BY["ek5.replikasyon.baseline_test_s0"]["deger"]) - dt.datetime.fromisoformat(BY["ek5.replikasyon.cipa_mtime"]["deger"])).total_seconds() / 3600
add("ek5.replikasyon.cipa_oncelik_sa", "cipa_notu_ile_ilk_baseline_test_olcumu_arasi_sure", "replikasyon cipasi", "VOC07 test", fark, "saat", "WS dosya zamanlari",
    "baseline_test_s0 - cipa_mtime", kaynak="turetilmis", uyari_not="mtime kriptografik degil; git ilk commit 2026-10-16")
n_rep = sum(1 for p in sorted((WS / "runs").glob("*/metrics.json")) if "rep_test_map50_95" in p.read_text(encoding="utf-8"))
add("ek5.replikasyon.rep_test_iceren_kosum", "metrics_json_rep_test_iceren_kosum_sayisi", "replikasyon cipasi", "VOC07 test", n_rep, "adet", "WS/runs/*/metrics.json",
    "rep_test_map50_95 anahtari olan koşum sayisi (tam taramada)", kaynak="turetilmis")
add("ek5.replikasyon.kaynak_63_5", "El_Hamdouni_2025_YOLOv8n_VOC07_test_mAP50_95", "literatur (YOLOv8n, 200 epoch)", "VOC07 test", 63.5, "mAP50-95 [puan]", "WS/notlar/literatur-cipa.md",
    "§2.1 satir B1 (Information 16(10):871, Tablo 3-4, 8-9; mAP50 83.6)", kaynak="literatur", atif="elhamdouni_2025_yolov8nvoc",
    uyari_not="YOLOv8n, belirtilmemis baslatma, 200 epoch; aralik ust siniri; protokol ayni degil; kendi olcumlerimizle sayisal yan yana konmaz")

# ------------------------------------------------------------------ kilit sonrasi listesi
_z = zaman(EK / "saglamlik-analiz.json")
add("ek5.kilit_sonrasi.saglamlik", "kitti_test_acilisindan_sonra_yapilan_is", "KITTI saglamlik analizleri (hiyerarsik bootstrap, grup basina Δ, sinif duyarliligi; 96 degerlendirme best+last)",
    "KITTI (test acilisi 2026-10-22T12:02:15; E3558)", _z, "zaman damgasi", "WS/scripts/ek_deney/saglamlik_ek.py", "WS/sonuclar/ek_deney/saglamlik-analiz.json dosya zamani", etiket="kilit_sonrasi",
    test_e_dokunur="evet (mevcut test_preds*.json yeniden okundu; yeni test cikarimi/acilisi yok; GPU yok)", durum_not="kesifsel; dis degerlendirme istemi; on-kayitli degil")

# ------------------------------------------------------------------ yaz
d["kayitlar"] = base + R
json.dump(d, open(EVID, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{len(R)} kayit eklendi: E{START}..E{START + len(R) - 1}")
