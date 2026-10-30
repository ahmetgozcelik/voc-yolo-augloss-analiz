#!/usr/bin/env python
"""Ek genelleme deneyi (KITTI x YOLO11s/YOLO26n + VOC H05 mekanizma) kanit kayitlari: E2571'den itibaren, evidence.json'a EKLER.

Mevcut kayitlara DOKUNMAZ. Idempotent: onceki calistirmanin urettigi `ek.*` kayitlari (id >= E2571) silinip yeniden uretilir.
NOT: build_evidence.py evidence.json'u SIFIRDAN uretir; o calistirilirsa bu betik yeniden calistirilmalidir (ek.* kayitlari kaybolur).
Hicbir sayi elle yazilmaz; her deger WS/sonuclar/ek_deney/* dosyalarindan okunur ya da ayni dosyalardan formulle turetilir
(turetilenlerde `turetme` alani). Birim: puan = x100.
Calistirma (MW kokunden): python evidence/scripts/build_ek_evidence.py
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
MW = Path(__file__).resolve().parents[2]
ROOT = MW.parents[1]
WS = ROOT / "deney_workspace" / "2-1-enhancing-real-time-object-detection-on-pascal-v"
EK = WS / "sonuclar" / "ek_deney"
EVID = MW / "evidence" / "evidence.json"
START = 2571


def J(p):
    return json.load(open(p, encoding="utf-8"))


d = J(EVID)
# E2571..END uretilen ek.* kayitlaridir (yeniden uretilir). END'den sonraki kayitlar (E3561+: elle/ayri betikle eklenenler: ek.gpu.*, ek.doz.*,
# ek3.* = build_ek_evidence_tur3.py) KORUNUR; bu betik yalniz [START, END] araligini yeniden yazar.
END = 3560
_gen = lambda r: r["anahtar"].startswith("ek.") and START <= int(r["id"][1:]) <= END  # noqa: E731
base = [r for r in d["kayitlar"] if not _gen(r)]
assert max(int(r["id"][1:]) for r in base if int(r["id"][1:]) < START) == START - 1, "beklenen son mevcut kayit E2570"
R = []
BY = {}


def add(anahtar, metrik, yontem, veri_seti, deger, birim, dosya, konum, etiket, **kw):
    assert anahtar not in BY, f"yinelenen anahtar {anahtar}"
    if isinstance(deger, float):
        assert deger == deger and abs(deger) != float("inf")
    rec = {"id": f"E{START + len(R):04d}", "anahtar": anahtar, "metrik": metrik, "yontem": yontem, "veri_seti": veri_seti,
           "deger": deger, "birim": birim, "kaynak": "yazilim", "kaynak_dosya": dosya, "kaynak_konum": konum,
           "atif": None, "durum": "gecerli", "etiket": etiket}
    rec.update(kw)
    R.append(rec)
    BY[anahtar] = rec
    return rec


# ------------------------------------------------------------------------------------------- kaynaklar
TA = J(EK / "test-analiz.json")
VA = J(EK / "val-analiz.json")
H7 = J(EK / "h07_kitti.json")
MEK = J(EK / "mekanizma-analiz.json")
ACILDI = J(EK / "test" / "ACILDI.json")
BOLME = J(WS / "data" / "kitti" / "bolme_raporu.json")
MODELS = ["yolo26n", "yolo11s"]
MAD = {"yolo26n": "YOLO26n", "yolo11s": "YOLO11s"}
CARDS = ["H01", "H02", "H03", "H04", "H05", "H06", "H08"]
FAM = {"H01": "iyilesme", "H03": "iyilesme", "H04": "iyilesme", "H06": "iyilesme", "H08": "iyilesme", "H02": "yonlu_replikasyon", "H05": "yonlu_replikasyon"}
DS_TEST = "KITTI test (1504 goruntu, 13 sahne grubu; DontCare=COCO crowd)"
DS_VAL = "KITTI val (987 goruntu, 10 sahne grubu; DontCare=COCO crowd)"
F_TA = "WS/sonuclar/ek_deney/test-analiz.json"
F_VA = "WS/sonuclar/ek_deney/val-analiz.json"
N = 3


def karar(x):
    if "kabul" in x:
        return "KABUL" if x["kabul"] else "kabul ölçütü sağlanmadı"
    return "replike oldu" if x["replike"] else "replike olmadı"


def y_(m, c):
    return f"{c}/{MAD[m]}"


def delta_block(pre, src, dosya, kpath, ds, m, c, etiket, boot=True, p_not="", **kw):
    """Birincil analiz blogu (test/val): seed Δ'lari, ort, sd, t-GA, p_tek, p_holm, karar (+ kumeli bootstrap GA)."""
    x = src
    y = y_(m, c)
    for i in range(N):
        add(f"{pre}.delta_s{i}", "delta_map50_95_seed", y, ds, x["delta_seed"][i], "puan", dosya, f"{kpath}.delta_seed[{i}] (kart - B, ayni seed; puan)", etiket, seed=i, n=N, **kw)
    add(f"{pre}.delta_ort", "delta_map50_95_ort", y, ds, x["ort"], "puan", dosya, f"{kpath}.ort (3 seed ortalamasi)", etiket, n=N, **kw)
    add(f"{pre}.delta_sd", "delta_map50_95_sd", y, ds, x["sd"], "puan", dosya, f"{kpath}.sd (ornek s.d., ddof=1)", etiket, n=N, **kw)
    add(f"{pre}.ga_lo", "delta_ga95_t_alt", y, ds, x["ga95"][0], "puan", dosya, f"{kpath}.ga95[0] (seed-duzeyi t, df=2)", etiket, n=N, **kw)
    add(f"{pre}.ga_hi", "delta_ga95_t_ust", y, ds, x["ga95"][1], "puan", dosya, f"{kpath}.ga95[1] (seed-duzeyi t, df=2)", etiket, n=N, **kw)
    add(f"{pre}.p_tek", "p_tek_yonlu", y, ds, x["p_tek"], "p", dosya, f"{kpath}.p_tek (tek yonlu, seed-esli t, df=2; {p_not})", etiket, n=N, **kw)
    add(f"{pre}.p_holm", "p_holm", y, ds, x["p_holm"], "p", dosya, f"{kpath}.p_holm (kaynakta 4 ondalik; {p_not.split(';')[0]}; model basina Holm)", etiket, n=N, **kw)
    add(f"{pre}.karar", "karar", y, ds, karar(x), "", dosya, f"{kpath}.kabul|replike -> metin (kabul: Δ>=+0.40 ve p_holm<0.10; replike: Δ<0 ve p_holm<0.10)", etiket, n=N, **kw)
    if boot:
        add(f"{pre}.boot_ga_lo", "delta_ga95_kumeli_bootstrap_alt", y, ds, x["kumeli_boot_ga95"][0], "puan", dosya, f"{kpath}.kumeli_boot_ga95[0] (13 sahne grubu uzerinde kumeli bootstrap)", etiket, n=N, **kw)
        add(f"{pre}.boot_ga_hi", "delta_ga95_kumeli_bootstrap_ust", y, ds, x["kumeli_boot_ga95"][1], "puan", dosya, f"{kpath}.kumeli_boot_ga95[1] (13 sahne grubu uzerinde kumeli bootstrap)", etiket, n=N, **kw)


# ------------------------------------------------------------------------------------------- 1) test (birincil)
for m in MODELS:
    for c in CARDS:
        x = TA[m][c]
        sag_last = TA["_last_pt_ikincil"][m][c]["saglamlik"]
        assert x["saglamlik"] == sag_last
        pn = "Holm-5 iyilesme ailesi" if FAM[c] == "iyilesme" else "Holm-2 yonlu replikasyon ailesi (Δ<0 yonu)"
        delta_block(f"ek.test.{m}.{c}", x, F_TA, f"{m}.{c}", DS_TEST, m, c, "ek_birincil_test", p_not=pn + "; birincil uc nokta best.pt")
        L = TA["_last_pt_ikincil"][m][c]
        y = y_(m, c)
        for i in range(N):
            add(f"ek.test.{m}.{c}.last_delta_s{i}", "delta_map50_95_last_seed", y, DS_TEST, L["delta_seed"][i], "puan", F_TA, f"_last_pt_ikincil.{m}.{c}.delta_seed[{i}] (last.pt; betimsel)", "ek_ikincil_last", seed=i, n=N)
        add(f"ek.test.{m}.{c}.last_delta_ort", "delta_map50_95_last_ort", y, DS_TEST, L["ort"], "puan", F_TA, f"_last_pt_ikincil.{m}.{c}.ort (last.pt; Holm/karar yok)", "ek_ikincil_last", n=N)
        add(f"ek.test.{m}.{c}.last_delta_sd", "delta_map50_95_last_sd", y, DS_TEST, L["sd"], "puan", F_TA, f"_last_pt_ikincil.{m}.{c}.sd", "ek_ikincil_last", n=N)
        add(f"ek.test.{m}.{c}.saglamlik", "saglamlik_etiketi", y, DS_TEST, sag_last, "", F_TA, f"_last_pt_ikincil.{m}.{c}.saglamlik (best.pt ve last.pt Δ ort ayni isaretliyse 'seçimden bağımsız'; ON-KAYIT v5b madde 5)", "ek_ikincil_last", n=N)
    add(f"ek.test.{m}.kritik_etki", "kritik_etki_ort", f"tum kartlar/{MAD[m]}", DS_TEST, TA[m]["_kritik_etki_ort_puan"], "puan", F_TA, f"{m}._kritik_etki_ort_puan (gozlenen sd_Δ ile; %50 guc, Holm-5 ilk adim)", "ek_birincil_test", n=N)

# test ikincil betimsel (3 sinif / 7 sinif)
for met, ad in (("map50_95_3sinif", "sinif3"), ("map50_95_7sinif", "sinif7")):
    for m in MODELS:
        for c in CARDS:
            x = TA["_ikincil_betimsel"][met][m][c]
            y = y_(m, c)
            for i in range(N):
                add(f"ek.test.{m}.{c}.{ad}.delta_s{i}", f"delta_{met}_seed", y, DS_TEST, x["delta_seed"][i], "puan", F_TA, f"_ikincil_betimsel.{met}.{m}.{c}.delta_seed[{i}]", "ek_ikincil_betimsel", seed=i, n=N)
            add(f"ek.test.{m}.{c}.{ad}.delta_ort", f"delta_{met}_ort", y, DS_TEST, x["ort"], "puan", F_TA, f"_ikincil_betimsel.{met}.{m}.{c}.ort (betimsel; Holm/karar yok)", "ek_ikincil_betimsel", n=N)
            add(f"ek.test.{m}.{c}.{ad}.delta_sd", f"delta_{met}_sd", y, DS_TEST, x["sd"], "puan", F_TA, f"_ikincil_betimsel.{met}.{m}.{c}.sd", "ek_ikincil_betimsel", n=N)

# ------------------------------------------------------------------------------------------- 2) val (ikincil)
for m in MODELS:
    for c in CARDS:
        x = VA["dontcare"][m][c]
        delta_block(f"ek.val.{m}.{c}", x, F_VA, f"dontcare.{m}.{c}", DS_VAL, m, c, "ek_ikincil_val", boot=False,
                    p_not=("Holm-5 iyilesme ailesi" if FAM[c] == "iyilesme" else "Holm-2 yonlu replikasyon ailesi (Δ<0 yonu)") + "; val=ikincil/kesifsel")
    add(f"ek.val.{m}.kritik_etki", "kritik_etki_ort", f"tum kartlar/{MAD[m]}", DS_VAL, VA["dontcare"][m]["_kritik_etki_ort_puan"], "puan", F_VA, f"dontcare.{m}._kritik_etki_ort_puan", "ek_ikincil_val", n=N)
for met, ad in (("map50_95_3sinif", "sinif3"), ("map50_95_7sinif", "sinif7"), ("egri_ep11_90", "egri_ep11_90")):
    for m in MODELS:
        for c in CARDS:
            x = VA["ikincil_betimsel"][met][m][c]
            y = y_(m, c)
            for i in range(N):
                add(f"ek.val.{m}.{c}.{ad}.delta_s{i}", f"delta_val_{met}_seed", y, DS_VAL, x["delta_seed"][i], "puan", F_VA, f"ikincil_betimsel.{met}.{m}.{c}.delta_seed[{i}]", "ek_ikincil_betimsel", seed=i, n=N)
            add(f"ek.val.{m}.{c}.{ad}.delta_ort", f"delta_val_{met}_ort", y, DS_VAL, x["ort"], "puan", F_VA, f"ikincil_betimsel.{met}.{m}.{c}.ort (betimsel)", "ek_ikincil_betimsel", n=N)
            add(f"ek.val.{m}.{c}.{ad}.delta_sd", f"delta_val_{met}_sd", y, DS_VAL, x["sd"], "puan", F_VA, f"ikincil_betimsel.{met}.{m}.{c}.sd", "ek_ikincil_betimsel", n=N)

# ------------------------------------------------------------------------------------------- 3) kosum tablosu: best epoch + mutlak test + artefakt
RUN = VA["kosumlar"]  # "model/kart/sN" -> run_id
assert set(RUN.values()) == {p["run_id"] for p in ACILDI["plan"]} and len(RUN) == 48
EP, TB, TL = {}, {}, {}
for key, rid in RUN.items():
    m_, c_, s_ = key.split("/")
    cfg = J(WS / "runs" / rid / "config.json")
    assert cfg["asama"] == 3, rid
    met = J(WS / "runs" / rid / "metrics.json")
    assert met["en_iyi_epoch"] == VA["tani"][key]["en_iyi_epoch"], rid
    EP[key] = met["en_iyi_epoch"]
    t = J(EK / "test" / f"{rid}.json")
    TB[key] = t["dontcare"]["map50_95"] * 100
    TL[key] = t["last"]["dontcare"]["map50_95"] * 100
    add(f"ek.epoch.{m_}.{c_}.s{s_[1:]}", "en_iyi_epoch", y_(m_, c_), "KITTI val (egitim-ici en iyi epoch)", met["en_iyi_epoch"], "epoch", f"WS/runs/{rid}/metrics.json", "en_iyi_epoch (best.pt secildigi epoch; 100 epoch icinde)", "ek_tani", seed=int(s_[1:]), run_id=rid)
    if c_ in ("H05", "H02"):
        k2 = "vfl_bas_sayisi" if c_ == "H05" else "inner_bas_sayisi"
        sayi = met[k2]
        assert sayi == (2 if m_ == "yolo26n" else 1), (rid, sayi)
        add(f"ek.mekbag.{m_}.{c_}.s{s_[1:]}", k2, y_(m_, c_), "KITTI egitim (mekanizma baglama sayaci)", sayi, "adet", f"WS/runs/{rid}/metrics.json", f"{k2} (ON-KAYIT: YOLO26n=2, YOLO11s=1; saglik kapisi kosulu)", "ek_tani", seed=int(s_[1:]), run_id=rid)

for m in MODELS:
    ep_all = [EP[k] for k in EP if k.startswith(m + "/")]
    ep_b = [EP[k] for k in EP if k.startswith(m + "/B/")]
    ep_c = [EP[k] for k in EP if k.startswith(m + "/") and "/B/" not in k]
    for ad, lst, y in (("tum", ep_all, f"tum 8 kol/{MAD[m]}"), ("B", ep_b, f"B/{MAD[m]}"), ("kartlar", ep_c, f"7 kart/{MAD[m]}")):
        tm = f"count/median(ek.epoch.{m}.<kollar>.s*)"
        add(f"ek.epoch.{m}.{ad}.n_kosum", "kosum_sayisi", y, "KITTI val", len(lst), "adet", "WS/runs/*/metrics.json", "en_iyi_epoch listesi uzunlugu", "turetilmis_ek", turetme=tm)
        add(f"ek.epoch.{m}.{ad}.n_ep1", "en_iyi_epoch_eq_1_sayisi", y, "KITTI val", sum(e == 1 for e in lst), "adet", "WS/runs/*/metrics.json", "count(en_iyi_epoch==1)", "turetilmis_ek", turetme=tm)
        add(f"ek.epoch.{m}.{ad}.n_lt20", "en_iyi_epoch_lt_20_sayisi", y, "KITTI val", sum(e < 20 for e in lst), "adet", "WS/runs/*/metrics.json", "count(en_iyi_epoch<20)", "turetilmis_ek", turetme=tm)
        add(f"ek.epoch.{m}.{ad}.medyan", "en_iyi_epoch_medyan", y, "KITTI val", float(st.median(lst)), "epoch", "WS/runs/*/metrics.json", "median(en_iyi_epoch)", "turetilmis_ek", turetme=tm)

ART = "kesifsel_artefakt"
ART_NOT = ("Keşifsel (post-hoc) analiz; karar/Holm yok. Soru: pozitif best.pt Δ'sı baseline'ın best.pt seçim artefaktı mı? Mutlak test mAP50-95 (DontCare kuralı) x100, "
           "WS/sonuclar/ek_deney/test/<run_id>.json dontcare.map50_95 (best) ve last.dontcare.map50_95 (last).")
for m in MODELS:
    Bb = [TB[f"{m}/B/s{i}"] for i in range(N)]
    Bl = [TL[f"{m}/B/s{i}"] for i in range(N)]
    ds = f"KITTI test (1504 goruntu; DontCare kurali), {MAD[m]}"
    for i in range(N):
        rid = RUN[f"{m}/B/s{i}"]
        add(f"ek.artefakt.{m}.B.best_s{i}", "test_map50_95_mutlak_best", f"B/{MAD[m]}", ds, Bb[i], "puan", f"WS/sonuclar/ek_deney/test/{rid}.json", "dontcare.map50_95 x100 (best.pt)", ART, seed=i, run_id=rid, n=N)
        add(f"ek.artefakt.{m}.B.last_s{i}", "test_map50_95_mutlak_last", f"B/{MAD[m]}", ds, Bl[i], "puan", f"WS/sonuclar/ek_deney/test/{rid}.json", "last.dontcare.map50_95 x100 (last.pt)", ART, seed=i, run_id=rid, n=N)
    tm = "mean/stdev(ek.artefakt.<m>.B.best_s0..2 | last_s0..2)"
    add(f"ek.artefakt.{m}.B.best_ort", "test_map50_95_mutlak_best_ort", f"B/{MAD[m]}", ds, st.mean(Bb), "puan", "WS/sonuclar/ek_deney/test/<B run_id>.json", "ort(best_s0..2)", ART, turetme=tm, n=N)
    add(f"ek.artefakt.{m}.B.best_sd", "test_map50_95_mutlak_best_sd", f"B/{MAD[m]}", ds, st.stdev(Bb), "puan", "WS/sonuclar/ek_deney/test/<B run_id>.json", "sd(best_s0..2), ddof=1", ART, turetme=tm, n=N)
    add(f"ek.artefakt.{m}.B.last_ort", "test_map50_95_mutlak_last_ort", f"B/{MAD[m]}", ds, st.mean(Bl), "puan", "WS/sonuclar/ek_deney/test/<B run_id>.json", "ort(last_s0..2)", ART, turetme=tm, n=N)
    add(f"ek.artefakt.{m}.B.last_sd", "test_map50_95_mutlak_last_sd", f"B/{MAD[m]}", ds, st.stdev(Bl), "puan", "WS/sonuclar/ek_deney/test/<B run_id>.json", "sd(last_s0..2), ddof=1", ART, turetme=tm, n=N)
    add(f"ek.artefakt.{m}.B.best_eksi_last", "B_best_eksi_last_ort", f"B/{MAD[m]}", ds, st.mean(Bb) - st.mean(Bl), "puan", "WS/sonuclar/ek_deney/test/<B run_id>.json", "ort(best) - ort(last); negatifse B'nin best.pt'si last.pt'sinden KOTU", ART, turetme=f"ek.artefakt.{m}.B.best_ort - ek.artefakt.{m}.B.last_ort", n=N, not_=ART_NOT)
    cb, cl = [], []
    for c in CARDS:
        b = st.mean(TB[f"{m}/{c}/s{i}"] for i in range(N))
        l = st.mean(TL[f"{m}/{c}/s{i}"] for i in range(N))
        cb.append(b)
        cl.append(l)
        # tutarlilik: kart best ort - B best ort == test-analiz Δ ort (3 ondalik yuvarlama toleransi)
        assert abs((b - st.mean(Bb)) - TA[m][c]["ort"]) < 0.002, (m, c)
        assert abs((l - st.mean(Bl)) - TA["_last_pt_ikincil"][m][c]["ort"]) < 0.002, (m, c, "last")
        add(f"ek.artefakt.{m}.{c}.best_ort", "test_map50_95_mutlak_best_ort", y_(m, c), ds, b, "puan", "WS/sonuclar/ek_deney/test/<kart run_id>.json", "ort(dontcare.map50_95 x100; s0-2; best.pt)", ART, turetme="mean(3 seed)", n=N)
        add(f"ek.artefakt.{m}.{c}.last_ort", "test_map50_95_mutlak_last_ort", y_(m, c), ds, l, "puan", "WS/sonuclar/ek_deney/test/<kart run_id>.json", "ort(last.dontcare.map50_95 x100; s0-2; last.pt)", ART, turetme="mean(3 seed)", n=N)
    kb, kl = st.mean(cb), st.mean(cl)
    yk = f"7 kart ortalamasi/{MAD[m]}"
    add(f"ek.artefakt.{m}.kart_best_ort", "kart_ortalamasi_test_best", yk, ds, kb, "puan", "WS/sonuclar/ek_deney/test/<run_id>.json", "ort(7 kart ortalamasi; best.pt)", ART, turetme=f"mean(ek.artefakt.{m}.<H01..H08>.best_ort)")
    add(f"ek.artefakt.{m}.kart_last_ort", "kart_ortalamasi_test_last", yk, ds, kl, "puan", "WS/sonuclar/ek_deney/test/<run_id>.json", "ort(7 kart ortalamasi; last.pt)", ART, turetme=f"mean(ek.artefakt.{m}.<H01..H08>.last_ort)")
    add(f"ek.artefakt.{m}.kart_best_eksi_last", "kart_best_eksi_last_ort", yk, ds, kb - kl, "puan", "WS/sonuclar/ek_deney/test/<run_id>.json", "kart ort(best) - kart ort(last)", ART, turetme=f"ek.artefakt.{m}.kart_best_ort - ek.artefakt.{m}.kart_last_ort")
    add(f"ek.artefakt.{m}.kayma_best", "ortak_kayma_best", yk, ds, kb - st.mean(Bb), "puan", "WS/sonuclar/ek_deney/test/<run_id>.json", "kart ort - B ort (best.pt) = 7 kartin best Δ ortalamasi", ART, turetme=f"ek.artefakt.{m}.kart_best_ort - ek.artefakt.{m}.B.best_ort", not_=ART_NOT)
    add(f"ek.artefakt.{m}.kayma_last", "ortak_kayma_last", yk, ds, kl - st.mean(Bl), "puan", "WS/sonuclar/ek_deney/test/<run_id>.json", "kart ort - B ort (last.pt) = 7 kartin last Δ ortalamasi", ART, turetme=f"ek.artefakt.{m}.kart_last_ort - ek.artefakt.{m}.B.last_ort", not_=ART_NOT)
    add(f"ek.artefakt.{m}.kayma_farki", "ortak_kayma_best_eksi_last", yk, ds, (kb - st.mean(Bb)) - (kl - st.mean(Bl)), "puan", "WS/sonuclar/ek_deney/test/<run_id>.json",
        "kayma_best - kayma_last = (kart best-last) - (B best-last): ortak pozitif kaymanin best.pt seciminden gelen kismi", ART,
        turetme=f"ek.artefakt.{m}.kayma_best - ek.artefakt.{m}.kayma_last", not_=ART_NOT)

# ------------------------------------------------------------------------------------------- 4) H07 (KITTI val keşifsel; seçilen satır test)
H7_NOT = "Gecikme batch 1 düzeltmesi sonrası (gunluk/deney-gunlugu.md 'EK-H07 gecikme düzeltmesi': ilk koşum tek-batch predict ile CUDA OOM; görüntü başına perf_counter, ısınma 50, ölçüm 300, FP16)."
for m in MODELS:
    h = H7[m]
    ds = f"KITTI val (B best.pt, s0-2 ort.), {MAD[m]}"
    for row, o in h["ozet"].items():
        add(f"ek.h07.{m}.{row}.delta_val", "h07_delta_val_map50_95", f"H07 {row}/{MAD[m]}", ds, o["delta_puan"], "puan", "WS/sonuclar/ek_deney/h07_kitti.json", f"{m}.ozet.{row}.delta_puan (R0'a gore, s0-2 ort.; keşifsel)", "kesifsel_h07", n=N)
    add(f"ek.h07.{m}.secilen_satir", "h07_secilen_test_satiri", f"H07/{MAD[m]}", ds, h["secilen_test_satiri"], "", "WS/sonuclar/ek_deney/h07_kitti.json", f"{m}.secilen_test_satiri (val'de en yüksek ort. mAP50-95; eşitlikte düşük gecikme)", "kesifsel_h07")
    g = h["gecikme_s0"]
    sec = h["secilen_test_satiri"]
    add(f"ek.h07.{m}.gecikme_R0_ms", "gecikme_ms", f"R0/{MAD[m]}", "KITTI val (s0, batch 1)", g["R0_ms"], "ms", "WS/sonuclar/ek_deney/h07_kitti.json", f"{m}.gecikme_s0.R0_ms", "kesifsel_h07", not_=H7_NOT)
    add(f"ek.h07.{m}.gecikme_secilen_ms", "gecikme_ms", f"{sec}/{MAD[m]}", "KITTI val (s0, batch 1)", g[f"{sec}_ms"], "ms", "WS/sonuclar/ek_deney/h07_kitti.json", f"{m}.gecikme_s0.{sec}_ms", "kesifsel_h07", not_=H7_NOT)
    add(f"ek.h07.{m}.gecikme_oran", "gecikme_orani", f"{sec}/R0 {MAD[m]}", "KITTI val (s0, batch 1)", g["oran"], "x", "WS/sonuclar/ek_deney/h07_kitti.json", f"{m}.gecikme_s0.oran", "kesifsel_h07", not_=H7_NOT)
    add(f"ek.h07.{m}.uygulanamaz", "h07_uygulanamaz_aileler", f"H07/{MAD[m]}", "-", ",".join(h["uygulanamaz_aileler"]) or "yok", "", "WS/sonuclar/ek_deney/h07_kitti.json", f"{m}.uygulanamaz_aileler (YOLO26n NMS'siz: N/S/W uygulanamaz; ON-KAYIT H07 bölümü)", "kesifsel_h07")
    # seçilen satırın test değerleri (B best.pt + H07 satırı) ve B'ye göre Δ
    dt = []
    for i in range(N):
        hf = EK / "test" / "h07" / f"{m}_{sec}_s{i}.json"
        hj = J(hf)
        assert hj["run_id"] == RUN[f"{m}/B/s{i}"] and hj["h07_satir"] == sec
        v = hj["dontcare"]["map50_95"] * 100
        dlt = v - TB[f"{m}/B/s{i}"]
        dt.append(dlt)
        add(f"ek.h07.{m}.test_s{i}", "h07_test_map50_95", f"{sec}/{MAD[m]}", DS_TEST, v, "puan", f"WS/sonuclar/ek_deney/test/h07/{m}_{sec}_s{i}.json", "dontcare.map50_95 x100 (B best.pt + seçilen H07 satırı)", "kesifsel_h07", seed=i, run_id=hj["run_id"])
        add(f"ek.h07.{m}.test_delta_s{i}", "h07_test_delta_map50_95", f"{sec}/{MAD[m]}", DS_TEST, dlt, "puan", f"WS/sonuclar/ek_deney/test/h07/{m}_{sec}_s{i}.json", "dontcare.map50_95 x100 - ayni seed B best.pt test (test/<B run_id>.json)", "kesifsel_h07", seed=i, turetme=f"ek.h07.{m}.test_s{i} - ek.artefakt.{m}.B.best_s{i}")
    add(f"ek.h07.{m}.test_delta_ort", "h07_test_delta_ort", f"{sec}/{MAD[m]}", DS_TEST, st.mean(dt), "puan", "WS/sonuclar/ek_deney/test/h07/*.json", "ort(test_delta_s0..2); keşifsel, Holm/karar yok", "kesifsel_h07", turetme="mean(ek.h07.<m>.test_delta_s0..2)", n=N)
    add(f"ek.h07.{m}.test_delta_sd", "h07_test_delta_sd", f"{sec}/{MAD[m]}", DS_TEST, st.stdev(dt), "puan", "WS/sonuclar/ek_deney/test/h07/*.json", "sd(test_delta_s0..2), ddof=1", "kesifsel_h07", turetme="stdev(ek.h07.<m>.test_delta_s0..2)", n=N)

# ------------------------------------------------------------------------------------------- 5) mekanizma kartlari (YOLO11n x VOC val)
ds = "VOC val (1000 goruntu), YOLO11n"
F_M = "WS/sonuclar/ek_deney/mekanizma-analiz.json"
for i in range(N):
    add(f"ek.mek.B.s{i}", "val_map50_95_baseline", "B/YOLO11n", ds, MEK["B"][str(i)] * 100, "puan", F_M, f"B.{i} x100 (ilk deneyin baseline val'i)", "ek_mekanizma", seed=i)
for k, ad in (("H05", "H05 (BCE->VFL, tam)"), ("pos", "H05-pos (pozitif q-agirliklandirma)"), ("neg", "H05-neg (negatif bastirma)")):
    for i in range(N):
        add(f"ek.mek.{k}.delta_s{i}", "delta_val_map50_95_seed", ad, ds, MEK["delta"][k][i], "puan", F_M, f"delta.{k}[{i}] (kart - B, ayni seed)", "ek_mekanizma", seed=i, n=N)
    xs = MEK["delta"][k]
    add(f"ek.mek.{k}.delta_ort", "delta_val_map50_95_ort", ad, ds, st.mean(xs), "puan", F_M, f"ort(delta.{k}[0..2])" + ("; kaynakta .ort" if k != "H05" else ""), "ek_mekanizma", n=N, turetme=f"mean(ek.mek.{k}.delta_s0..2)")
    add(f"ek.mek.{k}.delta_sd", "delta_val_map50_95_sd", ad, ds, st.stdev(xs), "puan", F_M, f"stdev(delta.{k}, ddof=1)", "ek_mekanizma", n=N, turetme=f"stdev(ek.mek.{k}.delta_s0..2)")
for k, ad in (("pos", "H05-pos"), ("neg", "H05-neg")):
    add(f"ek.mek.{k}.ort_kaynak", "delta_val_map50_95_ort_kaynak", ad, ds, MEK[k]["ort"], "puan", F_M, f"{k}.ort", "ek_mekanizma", n=N)
    add(f"ek.mek.{k}.p_tek_neg", "p_tek_yonlu_delta_lt_0", ad, ds, MEK[k]["p_tek_neg"], "p", F_M, f"{k}.p_tek_neg (tek yönlü Δ<0, df=2; bileşen katkısı ölçütü p<0.10)", "ek_mekanizma", n=N)
    add(f"ek.mek.{k}.katki", "bilesen_katkisi", ad, ds, "evet" if MEK[k]["katki"] else "hayir", "", F_M, f"{k}.katki (p<0.10 ise katkı var)", "ek_mekanizma", n=N)
    add(f"ek.mek.{k}.f_pay", "kayip_payi_f", ad, ds, MEK[k]["f_betimsel"], "oran", F_M, f"{k}.f_betimsel = Δ̄_kart / Δ̄_H05 (betimsel)", "ek_mekanizma", n=N)
for i in range(N):
    add(f"ek.mek.I_s{i}", "etkilesim_I_seed", "H05 - (pos+neg)", ds, MEK["etkilesim"]["I_seed"][i], "puan", F_M, f"etkilesim.I_seed[{i}] = Δ_H05,s - (Δ_pos,s + Δ_neg,s)", "ek_mekanizma", seed=i, n=N)
add("ek.mek.I_ort", "etkilesim_I_ort", "H05 - (pos+neg)", ds, MEK["etkilesim"]["I_ort"], "puan", F_M, "etkilesim.I_ort", "ek_mekanizma", n=N)
add("ek.mek.I_se", "etkilesim_I_se", "H05 - (pos+neg)", ds, MEK["etkilesim"]["SE"], "puan", F_M, "etkilesim.SE (SE(Ī))", "ek_mekanizma", n=N)
add("ek.mek.toplamsal", "toplamsallik", "H05 - (pos+neg)", ds, "evet" if MEK["etkilesim"]["toplamsal"] else "hayir", "", F_M, "etkilesim.toplamsal (|Ī| <= 2·SE(Ī) ise toplamsal)", "ek_mekanizma", n=N)
I_ort, SE = MEK["etkilesim"]["I_ort"], MEK["etkilesim"]["SE"]
sinif = ("kayıp esas olarak negatif bastırmadan; toplamsal" if (MEK["neg"]["katki"] and not MEK["pos"]["katki"] and MEK["etkilesim"]["toplamsal"]) else "ON-KAYIT sınıflaması elle gözden geçirilmeli")
add("ek.mek.siniflama", "mekanizma_siniflamasi", "H05-pos / H05-neg", ds, sinif, "", F_M,
    f"ON-KAYIT sınıflama kuralı: neg.katki={MEK['neg']['katki']} (p={MEK['neg']['p_tek_neg']:.4f}), pos.katki={MEK['pos']['katki']} (p={MEK['pos']['p_tek_neg']:.4f}), |Ī|={abs(I_ort):.3f} <= 2·SE={2 * SE:.3f}",
    "turetilmis_ek", n=N, turetme="ek.mek.{pos,neg}.katki, ek.mek.I_ort, ek.mek.I_se -> ON-KAYIT 'Mekanizma kartları' maddeleri 1-3")

# ------------------------------------------------------------------------------------------- 6) KITTI bolme + cakisma + acilis
F_B = "WS/data/kitti/bolme_raporu.json"
for sp in ("train", "val", "test"):
    add(f"ek.bolme.goruntu_{sp}", "goruntu_sayisi", f"KITTI {sp}", "KITTI 2D Object (7481 etiketli goruntu)", BOLME["bolme"]["goruntu"][sp], "goruntu", F_B, f"bolme.goruntu.{sp}", "ek_veri_bolme")
    add(f"ek.bolme.grup_{sp}", "sahne_grubu_sayisi", f"KITTI {sp}", "KITTI 2D Object", BOLME["bolme"]["grup_sayisi"][sp], "adet", F_B, f"bolme.grup_sayisi.{sp}", "ek_veri_bolme")
add("ek.bolme.goruntu_toplam", "goruntu_sayisi", "KITTI", "KITTI 2D Object", BOLME["goruntu"], "goruntu", F_B, "goruntu", "ek_veri_bolme")
add("ek.bolme.drive_sayisi", "drive_sayisi", "KITTI", "KITTI 2D Object", BOLME["sahne_gruplari"]["drive"], "adet", F_B, "sahne_gruplari.drive", "ek_veri_bolme")
add("ek.bolme.grup_sayisi", "sahne_grubu_sayisi_birlestirme_sonrasi", "KITTI", "KITTI 2D Object", BOLME["sahne_gruplari"]["grup"], "adet", F_B, "sahne_gruplari.grup (141 drive -> 59 grup; birlestirme=82)", "ek_veri_bolme")
add("ek.bolme.en_buyuk_grup_goruntu", "en_buyuk_grup_goruntu", "KITTI", "KITTI 2D Object", BOLME["sahne_gruplari"]["en_buyuk_grup_goruntu"], "goruntu", F_B, "sahne_gruplari.en_buyuk_grup_goruntu", "ek_veri_bolme")
for pair, ad in (("test-train", "test_train"), ("val-train", "val_train"), ("test-val", "test_val")):
    add(f"ek.bolme.ncc_max_{ad}", "sizinti_ncc_max", pair, "KITTI 2D Object", BOLME["sizinti_bagimsiz"][pair]["ncc_max"], "r", F_B, f"sizinti_bagimsiz.{pair}.ncc_max (normalize ceprazlama korelasyonu)", "ek_veri_bolme")
    add(f"ek.bolme.ncc_ge090_{ad}", "sizinti_ncc_ge090_sayisi", pair, "KITTI 2D Object", BOLME["sizinti_bagimsiz"][pair]["ncc_ge090"], "adet", F_B, f"sizinti_bagimsiz.{pair}.ncc_ge090", "ek_veri_bolme")
t0 = J(EK / "test" / f"{RUN['yolo26n/B/s0']}.json")["dontcare"]
add("ek.bolme.test_n_gt", "test_gt_nesne", "KITTI test", "KITTI test", t0["n_gt"], "adet", f"WS/sonuclar/ek_deney/test/{RUN['yolo26n/B/s0']}.json", "dontcare.n_gt", "ek_veri_bolme")
add("ek.bolme.test_n_dontcare_bolge", "test_dontcare_bolge", "KITTI test", "KITTI test", t0["n_dontcare_bolge"], "adet", f"WS/sonuclar/ek_deney/test/{RUN['yolo26n/B/s0']}.json", "dontcare.n_dontcare_bolge", "ek_veri_bolme")
add("ek.acilis.kosum", "test_acilis_kosum_sayisi", "KITTI test acilisi", "KITTI test", ACILDI["kosum"], "adet", "WS/sonuclar/ek_deney/test/ACILDI.json", "kosum (best.pt + last.pt birlikte; tek açılış)", "ek_veri_bolme")
add("ek.acilis.zaman", "test_acilis_zamani", "KITTI test acilisi", "KITTI test", ACILDI["acilis"], "zaman damgasi", "WS/sonuclar/ek_deney/test/ACILDI.json", "acilis", "ek_veri_bolme")
add("ek.acilis.on_kayit_sha256", "on_kayit_sha256", "KITTI test acilisi", "KITTI test", ACILDI["on_kayit_sha256"], "", "WS/sonuclar/ek_deney/test/ACILDI.json", "on_kayit_sha256", "ek_veri_bolme")
add("ek.cakisma.voc07_test_coco_train2017", "gorsel_cakisma_sayisi", "VOC07 test ∩ COCO train2017", "VOC07 test (4952 goruntu)", 4, "goruntu", "WS/ek_deney/cakisma-raporu.md",
    "tablo 'VOC2007 ∩ COCO' satırı: VOC test∩COCO train2017 = 4 ('4/4952'; Flickr kimliği eşleşmesi)", "ek_veri_bolme", n_ref=4952)

# ------------------------------------------------------------------------------------------- UYARI (beklenti ihlali: references/beklenti-ihlali.md)
UY = ("BEKLENTI IHLALI ALARMI: ilk deneyde (VOC, YOLO11n) bu kart aşağı yönlüydü (H05 Holm-anlamlı negatif; H02 yalnız düzeltmesiz); KITTI YOLO11s best.pt Δ'sı 7 kartın 7'sinde pozitif, "
      "ama last.pt Δ'sının 7 kart ortalaması ek.artefakt.yolo11s.kayma_last'tır (negatif) ve B'nin best.pt'si last.pt'sinden kötü (ek.artefakt.yolo11s.B.best_eksi_last < 0; ek.artefakt.yolo11s.kayma_farki). "
      "Mekanizmanın bağlı olduğunun kanıtı: ek.mekbag.* (vfl/inner baş sayacı beklenen değerde). Pozitif best.pt Δ'sı baseline best.pt seçim artefaktı olabilir; 'sonuç' değil keşifsel/duyarlılık bulgusu olarak yorumlanmalı.")
for rec in R:
    if rec["anahtar"] in ("ek.test.yolo11s.H02.delta_ort", "ek.test.yolo11s.H05.delta_ort"):
        rec["uyari"] = UY

assert len(R) == END - START + 1, f"uretilen kayit sayisi {len(R)} != {END - START + 1}; END guncelle"
d["kayitlar"] = sorted(base + R, key=lambda r: int(r["id"][1:]))
with open(EVID, "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent=1)
print(f"{len(R)} kayit eklendi: {R[0]['id']} .. {R[-1]['id']}; toplam {len(d['kayitlar'])}")
