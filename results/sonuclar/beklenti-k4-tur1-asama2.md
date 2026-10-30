# Beklenti Kapısı — kuyruk `k4-tur1` aşama 2 → **GEÇTİ**
*2026-09-30 16:58* · metrik `val_map50_95` (max) · beklenti: `k4-tur1.beklenti.json` · 4 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h01_multiscale25 | 1 | 0.5746 | 0.5746 | 651.4 |
| h02_innerciou | 1 | 0.5764 | 0.5764 | 589.2 |
| baseline_cm0 | 1 | 0.6038 | 0.6038 | 583.5 |
| h03_boxcp | 1 | 0.6045 | 0.6045 | 585.9 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | aralik | h01_multiscale25: ort 0.5746 ∈ [0.0, 1.0] (metrik tanimi geregi mAP orani [0,1]) |
| 🟢 OK | aralik | h02_innerciou: ort 0.5764 ∈ [0.0, 1.0] (metrik tanimi geregi mAP orani [0,1]) |
| 🟢 OK | aralik | baseline_cm0: ort 0.6038 ∈ [0.0, 1.0] (metrik tanimi geregi mAP orani [0,1]) |
| 🟢 OK | aralik | h03_boxcp: ort 0.6045 ∈ [0.0, 1.0] (metrik tanimi geregi mAP orani [0,1]) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | min_kosum | 4 başarılı koşum (en az 4) |
| 🟢 OK | ikincil_aralik | h01_multiscale25/s0 varyant_etki_sayaci=4416 ∈ [1,1000000000000] |
| 🟢 OK | ikincil_aralik | h02_innerciou/s0 varyant_etki_sayaci=4860 ∈ [1,1000000000000] |
| 🟢 OK | ikincil_aralik | h03_boxcp/s0 varyant_etki_sayaci=1.111e+04 ∈ [1,1000000000000] |
| 🟢 OK | aralik | h01_multiscale25: ort 0.5746 ∈ [0.5717, 0.5967] (baseline on-ucus seed0 0.5867 (20260930-070132_baseline_s0), seed1 0.5908: seed gurultusu ~0.4 puan. Alt = -1.5 puan (~3.7x gurultu): altina dusus = uygulama hatasi suphesi (H01 beklentisi +0.0..+0.5 puan nihai, 5 ep'te negatif olabilir ama <-1.5 degil). Ust = H01 karti 'goruldugunden iyi' esigi +1.0 puan.) |
| 🟢 OK | aralik | h02_innerciou: ort 0.5764 ∈ [0.5717, 0.5967] (baseline on-ucus seed0 0.5867; alt -1.5 puan (H02 karti (c): epoch-5 baseline-0.03 esigi tam kosum icindir; 5 ep on-ucusta daha siki -1.5). Ust = H02 karti 'goruldugunden iyi' esigi +1.0 puan (yayinlanmis en yuksek Inner-IoU kazanci +0.74).) |
| 🟢 OK | aralik | baseline_cm0: ort 0.6038 ∈ [0.3, 0.75] (KONTROL satiri (mosaic acik, 5 ep). Mosaic-kapali baseline on-ucus 0.5867; mosaic acikken erken epoch'ta dusuk beklenir, dogrudan cipa yok -> genis alt 0.30. Ust 0.75 = baseline on-ucus kuyrugunun comert ust tavani (k3-baseline-v2).) |
| 🟢 OK | yon_vs | h03_boxcp vs baseline_cm0: yönlü fark +0.00067, beklenen + (en az -0.015) |
| 🟢 OK | tavan | h03_boxcp vs baseline_cm0: kazanç +0.00067 ≤ tavan 0.008  |
| 🟢 OK | ikincil_aralik | h01_multiscale25/s0 egitim_sure_sn=622.2 ∈ [1,900] |
| 🟢 OK | ikincil_aralik | h02_innerciou/s0 egitim_sure_sn=560.8 ∈ [1,900] |
| 🟢 OK | ikincil_aralik | baseline_cm0/s0 egitim_sure_sn=553.6 ∈ [1,900] |
| 🟢 OK | ikincil_aralik | h03_boxcp/s0 egitim_sure_sn=555.8 ∈ [1,900] |
