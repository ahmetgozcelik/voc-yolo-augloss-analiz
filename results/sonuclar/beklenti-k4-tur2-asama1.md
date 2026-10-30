# Beklenti Kapısı — kuyruk `k4-tur2` aşama 1 → **GEÇTİ**
*2026-10-01 08:08* · metrik `val_map50_95` (max) · beklenti: `k4-tur2.beklenti.json` · 3 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h04_scale25 | 1 | 0.4273 | 0.4273 | 111.0 |
| h05_vfl | 1 | 0.3561 | 0.3561 | 111.1 |
| h06_box10 | 1 | 0.3994 | 0.3994 | 107.7 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | aralik | h04_scale25: ort 0.4273 ∈ [0.0, 1.0] (metrik tanimi geregi mAP orani [0,1]) |
| 🟢 OK | aralik | h05_vfl: ort 0.3561 ∈ [0.0, 1.0] (metrik tanimi geregi mAP orani [0,1]) |
| 🟢 OK | aralik | h06_box10: ort 0.3994 ∈ [0.0, 1.0] (metrik tanimi geregi mAP orani [0,1]) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | min_kosum | 3 başarılı koşum (en az 3) |
| 🟢 OK | ikincil_aralik | h04_scale25/s0 varyant_etki_sayaci=294 ∈ [1,1000000000000] |
| 🟢 OK | ikincil_aralik | h05_vfl/s0 varyant_etki_sayaci=294 ∈ [1,1000000000000] |
| 🟢 OK | ikincil_aralik | h06_box10/s0 varyant_etki_sayaci=294 ∈ [1,1000000000000] |
| 🟢 OK | ikincil_aralik | h04_scale25/s0 scale_args=0.25 ∈ [0.2499,0.2501] |
| 🟢 OK | ikincil_aralik | h06_box10/s0 box_gain=10 ∈ [9.999,10.001] |
| 🟢 OK | ikincil_aralik | h05_vfl/s0 vfl_alpha=0.75 ∈ [0.7499,0.7501] |
| 🟢 OK | ikincil_aralik | h05_vfl/s0 vfl_gamma=2 ∈ [1.9999,2.0001] |
| 🟢 OK | ikincil_aralik | h05_vfl/s0 vfl_bce_grad_orani=0.7136 ∈ [0.05,0.98] |
| 🟢 OK | aralik | h04_scale25: ort 0.4273 ∈ [0.3804, 0.4604] (baseline duman seed0 0.4104 (20260930-065935_baseline_s0; duman ayari ayni, scale mosaic kapaliyken de devrede). Alt -3 puan: duman (3 ep, %10 veri) tek-seed gurultusu olculmedi, 5 ep'te ~0.4 puan; amac cokme/uygulama hatasi yakalamak. Ust +5 puan: aciklanamayan sicrama (sizinti) alarmi.) |
| 🟢 OK | aralik | h06_box10: ort 0.3994 ∈ [0.3804, 0.4604] (baseline duman seed0 0.4104 (ayni ayar); box kazanci x1.33 ile -3 puan alti = kararsizlik/uygulama hatasi suphesi (H06 karti erken-uyari mantigi). Ust +5 puan: sicrama alarmi.) |
| 🟢 OK | aralik | h05_vfl: ort 0.3561 ∈ [0.3, 0.4604] (baseline duman seed0 0.4104; VFL cls gradyanini ~0.4-0.8x azaltir (CPU testi: grad orani 0.69-0.76, kayip orani 0.43-0.55), 3 epoch/%10 veri kisa takvimde ogrenmeyi yavaslatabilir -> alt sinir baseline-11 puan (yalniz cokme/sifir-ogrenme yakalar). Ust +5 puan sicrama alarmi (kart kirmizi bayragi tam kosumda +1.0).) |
| 🟢 OK | ikincil_aralik | h04_scale25/s0 egitim_sure_sn=79.5 ∈ [1,600] |
| 🟢 OK | ikincil_aralik | h05_vfl/s0 egitim_sure_sn=79.28 ∈ [1,600] |
| 🟢 OK | ikincil_aralik | h06_box10/s0 egitim_sure_sn=76.77 ∈ [1,600] |
