# Beklenti Kapısı — kuyruk `k4-tur1` aşama 1 → **GEÇTİ**
*2026-09-30 16:18* · metrik `val_map50_95` (max) · beklenti: `k4-tur1.beklenti.json` · 4 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h01_multiscale25 | 1 | 0.3919 | 0.3919 | 123.4 |
| h02_innerciou | 1 | 0.4165 | 0.4165 | 109.7 |
| baseline_cm0 | 1 | 0.4221 | 0.4221 | 108.5 |
| h03_boxcp | 1 | 0.4221 | 0.4221 | 108.2 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | aralik | h01_multiscale25: ort 0.3919 ∈ [0.0, 1.0] (metrik tanimi geregi mAP orani [0,1]) |
| 🟢 OK | aralik | h02_innerciou: ort 0.4165 ∈ [0.0, 1.0] (metrik tanimi geregi mAP orani [0,1]) |
| 🟢 OK | aralik | baseline_cm0: ort 0.4221 ∈ [0.0, 1.0] (metrik tanimi geregi mAP orani [0,1]) |
| 🟢 OK | aralik | h03_boxcp: ort 0.4221 ∈ [0.0, 1.0] (metrik tanimi geregi mAP orani [0,1]) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | min_kosum | 4 başarılı koşum (en az 4) |
| 🟢 OK | ikincil_aralik | h01_multiscale25/s0 varyant_etki_sayaci=269 ∈ [1,1000000000000] |
| 🟢 OK | ikincil_aralik | h02_innerciou/s0 varyant_etki_sayaci=294 ∈ [1,1000000000000] |
| 🟢 OK | ikincil_aralik | h03_boxcp/s0 varyant_etki_sayaci=771 ∈ [1,1000000000000] |
| 🟢 OK | aralik | h01_multiscale25: ort 0.3919 ∈ [0.3804, 0.4604] (baseline duman seed0 0.4104 (20260930-065935_baseline_s0; duman ayari ayni). Alt -3 puan: duman (3 ep, %10 veri) tek-seed gurultusu olculmedi, 5 ep'te ~0.4 puan; amac cokme/uygulama hatasi yakalamak, kazanc olcmek degil. Ust +5 puan: aciklanamayan sicrama (sizinti) alarmi.) |
| 🟢 OK | aralik | h02_innerciou: ort 0.4165 ∈ [0.3804, 0.4604] (baseline duman seed0 0.4104 (ayni ayar); H02 karti (c) erken-uyari mantigi: -3 puan alti = uygulama hatasi suphesi. Ust +5 puan: aciklanamayan sicrama alarmi.) |
| 🟢 OK | aralik | baseline_cm0: ort 0.4221 ∈ [0.1, 0.5] (KONTROL satiri (mosaic acik). Dogrudan cipa yok: mosaic-kapali baseline duman 0.4104; 2ep/%5 veri duman 0.127 (WS/runs/20260929-230007). Cok genis: yalniz cokme/sifir-ogrenme yakalar.) |
| 🟢 OK | yon_vs | h03_boxcp vs baseline_cm0: yönlü fark +6.6e-05, beklenen + (en az -0.03) |
| 🟢 OK | tavan | h03_boxcp vs baseline_cm0: kazanç +6.6e-05 ≤ tavan 0.05  |
| 🟢 OK | ikincil_aralik | h01_multiscale25/s0 egitim_sure_sn=91.43 ∈ [1,600] |
| 🟢 OK | ikincil_aralik | h02_innerciou/s0 egitim_sure_sn=78.31 ∈ [1,600] |
| 🟢 OK | ikincil_aralik | baseline_cm0/s0 egitim_sure_sn=77.37 ∈ [1,600] |
| 🟢 OK | ikincil_aralik | h03_boxcp/s0 egitim_sure_sn=77.08 ∈ [1,600] |
