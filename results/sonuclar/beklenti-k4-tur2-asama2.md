# Beklenti Kapısı — kuyruk `k4-tur2` aşama 2 → **GEÇTİ**
*2026-10-01 08:38* · metrik `val_map50_95` (max) · beklenti: `k4-tur2.beklenti.json` · 3 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h04_scale25 | 1 | 0.5923 | 0.5923 | 587.2 |
| h05_vfl | 1 | 0.5672 | 0.5672 | 610.7 |
| h06_box10 | 1 | 0.5936 | 0.5936 | 609.7 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | aralik | h04_scale25: ort 0.5923 ∈ [0.0, 1.0] (metrik tanimi geregi mAP orani [0,1]) |
| 🟢 OK | aralik | h05_vfl: ort 0.5672 ∈ [0.0, 1.0] (metrik tanimi geregi mAP orani [0,1]) |
| 🟢 OK | aralik | h06_box10: ort 0.5936 ∈ [0.0, 1.0] (metrik tanimi geregi mAP orani [0,1]) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | min_kosum | 3 başarılı koşum (en az 3) |
| 🟢 OK | ikincil_aralik | h04_scale25/s0 varyant_etki_sayaci=4860 ∈ [1,1000000000000] |
| 🟢 OK | ikincil_aralik | h05_vfl/s0 varyant_etki_sayaci=4860 ∈ [1,1000000000000] |
| 🟢 OK | ikincil_aralik | h06_box10/s0 varyant_etki_sayaci=4860 ∈ [1,1000000000000] |
| 🟢 OK | ikincil_aralik | h04_scale25/s0 scale_args=0.25 ∈ [0.2499,0.2501] |
| 🟢 OK | ikincil_aralik | h06_box10/s0 box_gain=10 ∈ [9.999,10.001] |
| 🟢 OK | ikincil_aralik | h05_vfl/s0 vfl_bce_grad_orani=0.5004 ∈ [0.05,0.98] |
| 🟢 OK | aralik | h04_scale25: ort 0.5923 ∈ [0.5717, 0.5967] (baseline on-ucus seed0 0.5867 (20260930-070132_baseline_s0), seed1 0.5908: seed gurultusu ~0.4 puan. Alt = -1.5 puan (~3.7x gurultu): altina dusus = uygulama hatasi suphesi (H04 beklentisi +0.0..+0.5 puan nihai; 5 ep'te scale etkisi olculmez, yalniz NaN/cokme denetimi: kart 'Uygulama notu'). Ust = H04 karti 'goruldugunden iyi' esigi +1.0 puan.) |
| 🟢 OK | aralik | h06_box10: ort 0.5936 ∈ [0.5717, 0.5967] (baseline on-ucus seed0 0.5867 / seed1 0.5908 (seed gurultusu ~0.4 puan); alt -1.5 puan (H06 karti erken-uyari: kararsizlik). Ust = 0.5867+1.0 puan (H06 karti kirmizi bayrak +0.8 nihai; on-ucusta 1.0 puan cok comert ust).) |
| 🟢 OK | aralik | h05_vfl: ort 0.5672 ∈ [0.52, 0.5967] (baseline on-ucus seed0 0.5867 / seed1 0.5908; VFL etkin cls gradyanini kucultur (kart 'Kayip payi notu'; CPU testi grad orani 0.69-0.76) -> 5 ep kisa takvimde geriden baslama toleransi: alt baseline-6.7 puan (yalniz uygulama hatasi/cokme; nihai beklenti -0.5..+0.5 puan). Ust +1.0 puan (kart kirmizi bayrak esigi).) |
| 🟢 OK | ikincil_aralik | h04_scale25/s0 egitim_sure_sn=556.6 ∈ [1,900] |
| 🟢 OK | ikincil_aralik | h05_vfl/s0 egitim_sure_sn=573.7 ∈ [1,900] |
| 🟢 OK | ikincil_aralik | h06_box10/s0 egitim_sure_sn=580.6 ∈ [1,900] |
