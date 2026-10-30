import json, csv, numpy as np
from pathlib import Path
WS = Path(__file__).resolve().parents[2]; R = WS / "runs"
M = lambda r: json.load(open(R / r / "metrics.json"))
B = ["20260930-072103_baseline_s0", "20260930-101624_baseline_s1", "20260930-131427_baseline_s2"]
H3 = ["20260930-231243_h03_boxcp_s0", "20261001-021011_h03_boxcp_s1", "20261001-050625_h03_boxcp_s2"]
# n=3 val degerleri scripts/analiz/n3_analiz.py -> sonuclar/n3-analiz.json (once o calistirilir)
N3 = json.load(open(WS / "sonuclar" / "n3-analiz.json", encoding="utf-8"))["kartlar"]
info = {"H01": ("augmentation", "multi_scale 0 -> 0.25 (eğitim girişi 480-800 px)", "ÖLDÜR (n=3: ölçüt karşılanmadı)"),
        "H02": ("loss", "CIoU -> Inner-CIoU (r=0.75)", "ÖLDÜR (n=3: ölçüt karşılanmadı)"),
        "H03": ("augmentation", "kutu-tabanlı küçük-nesne copy-paste (p=0.5, <=48 px; ~0.14 nesne/mosaic)", "ÖLDÜR (3 seed ort Δ +0.25 < +0.30; p=0.107)"),
        "H04": ("augmentation", "mosaic scale 0.5 -> 0.25", "ÖLDÜR (n=3: ölçüt karşılanmadı)"),
        "H05": ("loss", "cls kaybı BCE -> Varifocal (a=0.75, g=2)", "ÖLDÜR (n=3: ölçüt karşılanmadı)"),
        "H06": ("loss", "box kazancı 7.5 -> 10", "ÖLDÜR (n=3: ölçüt karşılanmadı)"),
        "H08": ("augmentation", "H03 yüksek doz (p=1.0, <=96 px; ~0.67 nesne/mosaic)", "ÖLDÜR (n=3: ölçüt karşılanmadı)")}
def d(a, b, k): return (a[k] - b[k]) * 100
rows = []
for k in ["H01", "H02", "H03", "H04", "H05", "H06", "H08"]:
    r = N3[k]; m = r["metrik"]; pm = m["mAP50-95"]
    runs = [r["runs"][str(i)] for i in range(3)]
    ikinci = "; ".join(f"{n} {m[n]['mean']:+.2f}" for n in ["AP50", "AP75", "AP_S", "AP_M", "AP_L"] if m.get(n))
    lat = np.mean([M(runs[i])["latency_toplam_ms"] / M(B[i])["latency_toplam_ms"] for i in range(3)])
    test = ""
    if k == "H03":
        T = ["20261001-212720_h03_boxcp-test_s0", "20261001-212828_h03_boxcp-test_s1", "20261001-212936_h03_boxcp-test_s2"]
        TB = ["20261001-212345_baseline-test_s0", "20261001-212500_baseline-test_s1", "20261001-212610_baseline-test_s2"]
        t = [d(M(a), M(b), "test_map50_95") for a, b in zip(T, TB)]
        test = f"{np.mean(t):+.2f} ± {np.std(t, ddof=1):.2f}"
    rows.append([k, info[k][0], info[k][1], 3, f"{pm['mean']:+.2f} ± {pm['sd']:.2f} (p_tek={pm['p1']:.3f}; %95 [{pm['lo']:+.2f}, {pm['hi']:+.2f}])", test, ikinci + " (val, n=3 ort Δ)", f"{lat:.2f} (çıkarım mimarisi aynı; ölçüm gürültüsü)", info[k][2]])
# H07 post (egitimsiz): h07_post.csv'den hesaplanir
hc = {r["satir"]: r for r in csv.DictReader(open(WS / "sonuclar" / "h07_post.csv", encoding="utf-8"))}
R0 = hc["R0"]
def hd(sat):
    r = hc[sat]
    f = lambda k: (float(r[k]) - float(R0[k])) * 100
    return f"{f('map50_95'):+.2f}", f"AP50 {f('map50'):+.2f}; AP_S {f('ap_s'):+.2f}; AP_M {f('ap_m'):+.2f}; AP_L {f('ap_l'):+.2f}", f"{float(r['lat_oran']):.2f}"
for sat, ad in [("N80", "NMS IoU 0.7 -> 0.8"), ("S03", "Gaussian Soft-NMS sigma 0.3"), ("U704", "test imgsz 640 -> 704"), ("U768", "test imgsz 640 -> 768")]:
    v, ik_, lo = hd(sat)
    rows.append(["H07-" + sat, "post", ad + " (eğitimsiz)", 1, v, "", ik_, lo, "aday değil"])
T = ["20261001-213042_baseline-tta-test_s0", "20261001-213236_baseline-tta-test_s1", "20261001-213428_baseline-tta-test_s2"]
TB = ["20261001-212345_baseline-test_s0", "20261001-212500_baseline-test_s1", "20261001-212610_baseline-test_s2"]
t = [d(M(a), M(b), "test_map50_95") for a, b in zip(T, TB)]
ik = "; ".join(f"{n} {np.mean([d(M(a), M(b), k) for a, b in zip(T, TB)]):+.2f}" for k, n in [("test_map50", "test AP50"), ("test_map75", "test AP75"), ("test_map_small", "test AP_S"), ("test_map_medium", "test AP_M"), ("test_map_large", "test AP_L")])
lr = np.mean([M(a)["latency_toplam_ms"] / M(b)["latency_toplam_ms"] for a, b in zip(T, TB)])
rows.append(["H07-T1", "post", "TTA: ölçekler [1,.83,.67] + yatay çevirme (eğitimsiz)", 3, f"{hd('T1')[0]} (val, eğitimsiz; tek best.pt = baseline s0; bootstrap alt %90 +0.25)", f"{np.mean(t):+.2f} ± {np.std(t, ddof=1):.2f} (n=3)", ik, f"{lr:.2f} (test oturumu) / {hd('T1')[2]} (H07 val oturumu)", "KAZANÇ-B (doğruluk-gecikme ödünleşimi; gerçek-zaman dışı)"])
with open(WS / "sonuclar" / "ablasyon.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["kart", "aile", "degisen_tek_sey", "n_seed", "val_delta_puan_ort_sd", "test_delta_puan_ort_sd", "ikincil_delta_puan", "gecikme_orani", "karar"])
    w.writerows(rows)
print(open(WS / "sonuclar" / "ablasyon.csv", encoding="utf-8").read())
