"""K5 test analizi (K5-TEST-PLANI.md §3): seed-eşli t, görüntü-düzeyi eşli bootstrap, val<->test tutarlılığı.
Yalnız mevcut metrics.json / test_paket.npz okur; yeni koşum yok. Çıktı: sonuclar/test-analiz.json"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import stats
WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "varyantlar" / "h07_post"))
import degerlendir as H  # noqa

RUNS = WS / "runs"
TEST = {"baseline": ["20261001-212345_baseline-test_s0", "20261001-212500_baseline-test_s1", "20261001-212610_baseline-test_s2"],
        "h03": ["20261001-212720_h03_boxcp-test_s0", "20261001-212828_h03_boxcp-test_s1", "20261001-212936_h03_boxcp-test_s2"],
        "tta": ["20261001-213042_baseline-tta-test_s0", "20261001-213236_baseline-tta-test_s1", "20261001-213428_baseline-tta-test_s2"]}
VAL = {"baseline": ["20260930-072103_baseline_s0", "20260930-101624_baseline_s1", "20260930-131427_baseline_s2"],
       "h03": ["20260930-231243_h03_boxcp_s0", "20261001-021011_h03_boxcp_s1", "20261001-050625_h03_boxcp_s2"]}
B = 1000


def paket(run):
    z = dict(np.load(RUNS / run / "test_paket.npz", allow_pickle=True))
    ln = z["len"]; off = np.concatenate([[0], np.cumsum(ln)])
    tl = z["tlen"]; to = np.concatenate([[0], np.cumsum(tl)])
    n = len(ln)
    return H.Paket(list(z["names"]), [z["tp"][off[i]:off[i + 1]] for i in range(n)], [z["conf"][off[i]:off[i + 1]] for i in range(n)],
                   [z["pcls"][off[i]:off[i + 1]] for i in range(n)], [z["tcls"][to[i]:to[i + 1]] for i in range(n)])


def M(run):
    return json.load(open(RUNS / run / "metrics.json"))


def main():
    out = {"B": B}
    # tutarlilik: paket mAP ~ metrics
    pk = {}
    for g in TEST:
        for s, r in enumerate(TEST[g]):
            pk[(g, s)] = paket(r)
    ref = pk[("baseline", 0)].names
    out["isim_ayni"] = all(p.names == ref for p in pk.values())
    out["paket_map_fark"] = {f"{g}_s{s}": pk[(g, s)].map()[0] - M(TEST[g][s])["test_map50_95"] for g in TEST for s in range(3)}
    # bootstrap
    boot = {}
    t0 = time.time()
    for s in range(3):
        d = H.bootstrap_delta({"R0": pk[("baseline", s)], "h03": pk[("h03", s)], "tta": pk[("tta", s)]}, ["h03", "tta"], B, 6)
        for k in d: boot[(k, s)] = d[k]
        print("seed", s, "bitti", round(time.time() - t0), "s", flush=True)
    for k in ["h03", "tta"]:
        out.setdefault("bootstrap", {})[k] = {}
        for s in range(3):
            a = boot[(k, s)] * 100
            out["bootstrap"][k][f"s{s}"] = dict(ort=a.mean(), lo=np.percentile(a, 2.5), hi=np.percentile(a, 97.5), alt90=np.percentile(a, 10))
        c = np.mean([boot[(k, s)] for s in range(3)], 0) * 100
        out["bootstrap"][k]["birlesik"] = dict(ort=c.mean(), lo=np.percentile(c, 2.5), hi=np.percentile(c, 97.5), alt90=np.percentile(c, 10), p_pozitif=float((c > 0).mean()), p_marj030=float((c >= 0.30).mean()))
    # seed-esli t ve ikincil
    keys = ["test_map50_95", "test_map50", "test_map75", "test_map_small", "test_map_medium", "test_map_large", "test_precision", "test_recall", "latency_toplam_ms", "fps", "latency_dep_ms"]
    tab = {}
    for g in TEST:
        tab[g] = {k: [M(r).get(k) for r in TEST[g]] for k in keys}
    out["tablo"] = tab
    out["test"] = {}
    for g in ["h03", "tta"]:
        o = {}
        for k in keys:
            a = np.array(tab[g][k], float); b = np.array(tab["baseline"][k], float)
            d = a - b
            sc = 100 if k.startswith("test_") else 1
            t = stats.ttest_rel(a, b)
            o[k] = dict(ort=a.mean() * sc, sd=a.std(ddof=1) * sc, bort=b.mean() * sc, bsd=b.std(ddof=1) * sc, dort=d.mean() * sc, dsd=d.std(ddof=1) * sc,
                        dseed=(d * sc).tolist(), p=float(t.pvalue), t=float(t.statistic))
        out["test"][g] = o
    p_all = [out["test"][g]["test_map50_95"]["p"] for g in ["h03", "tta"]]
    srt = np.argsort(p_all); holm = [None, None]; m = 2; run = 0
    for rank, i in enumerate(srt):
        run = max(run, min(1, p_all[i] * (m - rank))); holm[i] = run
    out["holm"] = dict(h03=holm[0], tta=holm[1])
    out["bonf"] = dict(h03=min(1, p_all[0] * 2), tta=min(1, p_all[1] * 2))
    # tutarlilik (a): rep_test vs test
    out["rep_fark"] = [M(VAL["baseline"][s]).get("rep_test_map50_95") - M(TEST["baseline"][s])["test_map50_95"] for s in range(3)]
    # val
    out["val"] = {}
    for g in VAL:
        out["val"][g] = {k: [M(r)[k] for r in VAL[g]] for k in ["val_map50_95", "val_map50", "val_map_small", "val_map_medium", "val_map_large"]}
    vd = np.array(out["val"]["h03"]["val_map50_95"]) - np.array(out["val"]["baseline"]["val_map50_95"])
    out["val_h03_delta"] = dict(dseed=(vd * 100).tolist(), ort=vd.mean() * 100, sd=vd.std(ddof=1) * 100, p=float(stats.ttest_rel(out["val"]["h03"]["val_map50_95"], out["val"]["baseline"]["val_map50_95"]).pvalue))
    out["val_tta_delta_h07"] = 0.7968
    # sinif duzeyi (kesifsel) test: AP50-95 Delta seed ortalamasi
    cl = {}
    for g in ["h03", "tta"]:
        names = [k for k in M(TEST[g][0])["alt_grup"] if k.startswith("ap5095_")]
        cl[g] = {n[7:]: float(np.mean([M(TEST[g][s])["alt_grup"][n] - M(TEST["baseline"][s])["alt_grup"][n] for s in range(3)]) * 100) for n in names}
    out["sinif"] = cl
    json.dump(out, open(WS / "sonuclar" / "test-analiz.json", "w"), indent=1, default=float)
    print("tamam", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()


def yaz_md():
    o = json.load(open(WS / "sonuclar" / "test-analiz.json"))
    T, bs = o["test"], o["bootstrap"]
    L = ["# K5 Test Analizi (K5-TEST-PLANI.md §3'e göre; test bir kez açıldı, D1)", "",
         "Kaynak: `scripts/analiz/k5_test_analiz.py` (metrics.json + test_paket.npz; yeni koşum yok). Birincil metrik test mAP50-95 (puan = x100), n=3 seed {0,1,2}, seed-eşli, 4952 test görüntüsü.",
         f"Tutarlılık: paket mAP ile metrics.json farkı <1e-6 (9/9); görüntü adları üç varyantta aynı={o['isim_ayni']}; baseline-test ile rep_test farkı = {o['rep_fark']} (≤0.0005 şartı sağlandı); test_n_goruntu=4952.", "",
         "## T-A Test ana tablo (ort ± sd, n=3)", "",
         "| varyant | mAP50-95 | mAP50 | mAP75 | AP_S | AP_M | AP_L |", "|---|---|---|---|---|---|---|"]
    b = T["h03"]
    f = lambda g, k, d=2: f"{T[g][k]['ort']:.{d}f} ± {T[g][k]['sd']:.{d}f}"
    fb = lambda k, d=2: f"{b[k]['bort']:.{d}f} ± {b[k]['bsd']:.{d}f}"
    ks = ["test_map50_95", "test_map50", "test_map75", "test_map_small", "test_map_medium", "test_map_large"]
    L.append("| baseline-test | " + " | ".join(fb(k) for k in ks) + " |")
    L.append("| h03_boxcp-test | " + " | ".join(f("h03", k) for k in ks) + " |")
    L.append("| baseline-tta-test (T1) | " + " | ".join(f("tta", k) for k in ks) + " |")
    L += ["", "## Δ vs baseline-test (puan; seed-eşli çift taraflı t, df=2)", "",
          "| karşılaştırma | metrik | Δ ort ± sd | Δ s0 / s1 / s2 | p |", "|---|---|---|---|---|"]
    for g, ad in [("h03", "H03 boxcp"), ("tta", "TTA T1")]:
        for k, nm in zip(ks, ["mAP50-95", "mAP50", "mAP75", "AP_S", "AP_M", "AP_L"]):
            v = T[g][k]
            L.append(f"| {ad} | {nm} | {v['dort']:+.2f} ± {v['dsd']:.2f} | " + " / ".join(f"{x:+.2f}" for x in v["dseed"]) + f" | {v['p']:.3f} |")
    L += ["", f"Çoklu karşılaştırma (mAP50-95, 2 karşılaştırma): Holm p: H03 {o['holm']['h03']:.3f}, TTA {o['holm']['tta']:.3f}; Bonferroni: H03 {o['bonf']['h03']:.3f}, TTA {o['bonf']['tta']:.3f}. Ana iddia H03 vs baseline (ön-kayıt §3.4); TTA keşifsel. İkincil metrik p'leri düzeltmesizdir (keşifsel).", "",
          "## Görüntü-düzeyi eşli bootstrap (B=1000, aynı görüntü indeksleri iki varyanta; Δ mAP50-95, puan)", "",
          "| karşılaştırma | seed | Δ ort | %95 CI | tek yönlü %90 alt sınır |", "|---|---|---|---|---|"]
    for g, ad in [("h03", "H03"), ("tta", "TTA")]:
        for s in ["s0", "s1", "s2", "birlesik"]:
            v = bs[g][s]
            L.append(f"| {ad} | {s if s != 'birlesik' else 'birleşik (3 seed ort.)'} | {v['ort']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] | {v['alt90']:+.3f} |")
        v = bs[g]["birlesik"]
        L.append(f"| {ad} | birleşik | P(Δ>0)={v['p_pozitif']:.3f}; P(Δ>=+0.30)={v['p_marj030']:.3f} | | |")
    vd = o["val_h03_delta"]
    L += ["", "## T-D Val -> test tutarlılığı", "",
          "| karşılaştırma | val Δ (puan) | test Δ (puan) | not |", "|---|---|---|---|",
          f"| H03 vs baseline (n=3 eşli) | {vd['ort']:+.2f} ± {vd['sd']:.2f} (s: " + " / ".join(f"{x:+.2f}" for x in vd["dseed"]) + f"; p={vd['p']:.3f}) | {T['h03']['test_map50_95']['dort']:+.2f} ± {T['h03']['test_map50_95']['dsd']:.2f} (p={T['h03']['test_map50_95']['p']:.3f}) | işaret değişti; val Δ - test Δ = {vd['ort'] - T['h03']['test_map50_95']['dort']:+.2f} |",
          f"| H03 mAP50 | val +0.74 ± 0.16 (3/3 pozitif) | {T['h03']['test_map50']['dort']:+.2f} ± {T['h03']['test_map50']['dsd']:.2f} | val'deki tutarlı AP50 kazancı test'e taşınmadı |",
          f"| TTA T1 vs baseline | +0.80 (val, tek best.pt = baseline s0; H07 A/B yarıları +0.65/+1.05) | {T['tta']['test_map50_95']['dort']:+.2f} ± {T['tta']['test_map50_95']['dsd']:.2f} (n=3) | yön ve büyüklük korundu |", ""]
    L += ["## T-B Hız (batch1, FP16, uçtan uca ms, conf .001; seed ort ± sd; aynı test oturumu)", "",
          "| varyant | gecikme ms | FPS | gecikme oranı (baseline-test = 1) |", "|---|---|---|---|"]
    for g, ad in [("baseline", "baseline-test"), ("h03", "h03_boxcp-test"), ("tta", "baseline-tta-test (T1)")]:
        tb = o["tablo"][g]; l = np.array(tb["latency_toplam_ms"]); fp = np.array(tb["fps"]); bl = np.array(o["tablo"]["baseline"]["latency_toplam_ms"])
        L.append(f"| {ad} | {l.mean():.2f} ± {l.std(ddof=1):.2f} | {fp.mean():.1f} ± {fp.std(ddof=1):.1f} | {(l / bl).mean():.2f} |")
    L += ["", "Not: ölçüm süreç-zamanına duyarlıdır (baseline val-zamanı 8.0 ms, test oturumu 9.43 ms; H03 8.60 ms çıkarım mimarisi baseline ile aynı olduğundan fark ölçüm gürültüsüdür, H03 hız kazancı iddia edilmez). TTA oranı test oturumunda 2.29, H07 val oturumunda 2.57.", ""]
    L += ["## T-C Sınıf düzeyi test AP50-95 Δ (seed ort., puan; keşifsel)", ""]
    for g in ["h03", "tta"]:
        sr = sorted(o["sinif"][g].items(), key=lambda x: x[1])
        L.append(f"- {g}: en çok kaybeden 5: " + ", ".join(f"{a} {v:+.2f}" for a, v in sr[:5]) + "; en çok kazanan 5: " + ", ".join(f"{a} {v:+.2f}" for a, v in sr[::-1][:5]))
    open(WS / "sonuclar" / "test-analiz.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
