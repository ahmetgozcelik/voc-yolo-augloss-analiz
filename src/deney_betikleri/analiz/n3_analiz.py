#!/usr/bin/env python
"""n=3 val analizi (K5 sonrasi, hakem B3). YALNIZ VAL; test'e dokunmaz; GPU yok.
Kullanim (WS kokunden): env/.venv/Scripts/python.exe scripts/analiz/n3_analiz.py
Girdi : runs/*/config.json(.asama==3, -test/h07 haric) + .done + metrics.json; AP75 icin
        sonuclar/hata-analizi*_<run_id>.md.json (hata_analizi.py ciktisi; baseline s0 = hata-analizi.md.json).
Cikti : sonuclar/n3-analiz.md, n3-analiz.json.
Kabul olcutu (on-kayitli): seed-esli ort Delta >= +0.40 puan VE tek yonlu esli t p < 0.10.
On-eleme kurali: seed0 Delta >= +0.30 -> seed 1,2'ye cik.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WS = Path(__file__).resolve().parents[2]; R = WS / "runs"; S = WS / "sonuclar"
KARTLAR = {"H01": "h01_multiscale25", "H02": "h02_innerciou", "H03": "h03_boxcp", "H04": "h04_scale25",
           "H05": "h05_vfl", "H06": "h06_box10", "H08": "h08_boxcp_doz"}
MET = [("mAP50-95", "val_map50_95"), ("AP50", "val_map50"), ("AP_S", "val_map_small"),
       ("AP_M", "val_map_medium"), ("AP_L", "val_map_large")]
ESIK_ON, ESIK_KABUL, P_KABUL = 0.30, 0.40, 0.10

def kosumlar():
    out = {}
    for d in sorted(R.glob("*/")):
        if not ((d / ".done").exists() and (d / "config.json").exists()): continue
        j = json.load(open(d / ".done", encoding="utf-8")); c = json.load(open(d / "config.json", encoding="utf-8"))
        if j.get("rc") != 0 or not j.get("metrics_var") or c.get("asama") != 3: continue
        out.setdefault(c["varyant"], {})[int(c["seed"])] = (c["run_id"], json.load(open(d / "metrics.json", encoding="utf-8")))
    return out

def ap75(rid):
    for p in list(S.glob(f"hata-analizi*_{rid}.md.json")) + [S / "hata-analizi.md.json"]:
        if p.exists():
            try:
                if p.name == "hata-analizi.md.json" and "072103" not in rid: continue
                return json.load(open(p, encoding="utf-8"))["map75"]
            except Exception: pass
    return None

def stat(dm):
    dm = np.array(dm); n = len(dm); m = dm.mean(); sd = dm.std(ddof=1)
    t, p2 = stats.ttest_1samp(dm, 0.0); p1 = p2 / 2 if m > 0 else 1 - p2 / 2
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return dict(mean=m, sd=sd, p1=float(p1), p2=float(p2), lo=m - h, hi=m + h, seed=[float(x) for x in dm])

K = kosumlar(); B = K["baseline"]; sonuc = {}
for kart, v in KARTLAR.items():
    V = K[v]; seeds = sorted(set(V) & set(B)); assert seeds == [0, 1, 2], (kart, seeds)
    r = {"runs": {s: V[s][0] for s in seeds}, "metrik": {}}
    for ad, k in MET:
        r["metrik"][ad] = stat([(V[s][1][k] - B[s][1][k]) * 100 for s in seeds])
    a = [ap75(V[s][0]) for s in seeds]; b = [ap75(B[s][0]) for s in seeds]
    r["metrik"]["AP75"] = stat([(x - y) * 100 for x, y in zip(a, b)]) if None not in a + b else None
    pm = r["metrik"]["mAP50-95"]
    r["s0_delta"] = pm["seed"][0]
    r["on_eleme_gecti"] = pm["seed"][0] >= ESIK_ON * 1.0 - 0.005   # kartlar yuvarlanmis degerle (+0.30) karar vermisti
    r["kabul"] = bool(pm["mean"] >= ESIK_KABUL and pm["p1"] < P_KABUL)
    r["on_eleme_karar"] = "ELE (s0 < +0.30)" if not r["on_eleme_gecti"] else "seed 1,2'ye cik"
    r["on_eleme_isabeti"] = ("DOGRU (n=3'te de kabul yok)" if not r["kabul"] else "YANLIS (n=3 kabul ediyor)") if not r["on_eleme_gecti"] \
        else ("DOGRU (cikis hakki; n=3 kabul etti)" if r["kabul"] else "cikis yanlis alarm (n=3 kabul yok)")
    sonuc[kart] = r
bm = {ad: float(np.mean([B[s][1][k] for s in (0, 1, 2)]) * 100) for ad, k in MET}
json.dump({"baseline_runs": {s: B[s][0] for s in B}, "baseline_ort": bm, "kartlar": sonuc}, open(S / "n3-analiz.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

L = ["# n=3 val analizi (K5 sonrası, hakem B3)", "",
     "*Üretici: `scripts/analiz/n3_analiz.py` (yalnız val, GPU yok). Puan = x100. Δ = seed-eşli (s0↔s0, s1↔s1, s2↔s2) kart − baseline; ort ± sd (ddof=1); p = tek yönlü eşli t (H1: Δ>0), çift yönlü parantezde; aralık = seed-düzeyi %95 t aralığı (df=2, t=4.303). Kabul ölçütü (ön-kayıtlı): ort Δ ≥ +0.40 VE tek yönlü p < 0.10. Ön-eleme: s0 Δ ≥ +0.30. H05 = H05r koşumları (arıza koşumu 20261001-114442 hariç). AP75: `hata_analizi.py` (pycocotools).*", "",
     f"Baseline (n=3, val): " + "; ".join(f"{a} {bm[a]:.2f}" for a, _ in MET), "",
     "## 1. Ana tablo (mAP50-95)", "",
     "| Kart | Δ ort ± sd | seed Δ (s0/s1/s2) | p tek (çift) | %95 t aralığı | Kabul | Ön-eleme (s0 Δ) | Ön-eleme isabeti |", "|---|---|---|---|---|---|---|---|"]
for k, r in sonuc.items():
    m = r["metrik"]["mAP50-95"]
    L.append(f"| {k} | {m['mean']:+.2f} ± {m['sd']:.2f} | {' / '.join(f'{x:+.2f}' for x in m['seed'])} | {m['p1']:.3f} ({m['p2']:.3f}) | [{m['lo']:+.2f}, {m['hi']:+.2f}] | {'EVET' if r['kabul'] else 'HAYIR'} | {r['s0_delta']:+.2f} → {r['on_eleme_karar']} | {r['on_eleme_isabeti']} |")
L += ["", "## 2. İkincil metrikler (Δ ort ± sd, puan)", "", "| Kart | AP50 | AP75 | AP_S | AP_M | AP_L |", "|---|---|---|---|---|---|"]
for k, r in sonuc.items():
    c = lambda a: (f"{r['metrik'][a]['mean']:+.2f} ± {r['metrik'][a]['sd']:.2f}" if r["metrik"][a] else "NA")
    L.append(f"| {k} | {c('AP50')} | {c('AP75')} | {c('AP_S')} | {c('AP_M')} | {c('AP_L')} |")
L += ["", "## 3. Koşumlar", ""] + [f"- {k}: " + ", ".join(f"s{s}={i}" for s, i in r["runs"].items()) for k, r in sonuc.items()] + ["- baseline: " + ", ".join(f"s{s}={i[0]}" for s, i in sorted(B.items()))]
n_kabul = sum(r["kabul"] for r in sonuc.values())
L += ["", "## 4. Okuma", "",
      f"- Kabul ölçütünü karşılayan kart: {n_kabul}/7. n=3'te seed-düzeyi t-testinin tespit edebileceği en küçük etki ≈0.5 puan (df=2); tablo \"etki yok\" değil \"kabul ölçütü sağlanmadı\" der.",
      "- Anlamlı (p<0.05) iki kart de baseline'ın ALTINDA: " + ", ".join(f"{k} ({r['metrik']['mAP50-95']['mean']:+.2f}, çift yönlü p={r['metrik']['mAP50-95']['p2']:.3f})" for k, r in sonuc.items() if r['metrik']['mAP50-95']['p2'] < 0.05) + " (negatif yönlü etkiler; çoklu karşılaştırma düzeltmesiz).",
      "- Ön-eleme isabeti: s0 kapısında elenen " + str(sum(not r["on_eleme_gecti"] for r in sonuc.values())) + " karttan " + str(sum((not r["on_eleme_gecti"]) and (not r["kabul"]) for r in sonuc.values())) + "'ü n=3'te de kabul edilmedi (hiçbir eleme tersine dönmedi: kaçırılan kazanç yok). s0 kapısını geçen kart(lar): " + ", ".join(k for k, r in sonuc.items() if r["on_eleme_gecti"]) + " (n=3'te kabul yok; kapı gürültü tabanında olduğundan çıkış hakkı yanlış alarm çıktı)."]
(S / "n3-analiz.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))
