# Beklenti Kapısı — kuyruk `k4-tur3a` aşama 1 → **GEÇTİ**
*2026-10-01 18:15* · metrik `val_map50_95` (max) · beklenti: `k4-tur3a.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h08_boxcp_doz | 1 | 0.4058 | 0.4058 | 112.8 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | aralik | h08_boxcp_doz: ort 0.4058 ∈ [0.0, 1.0] (metrik tanimi geregi mAP orani [0,1]) |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | ikincil_aralik | h08_boxcp_doz/s0 varyant_etki_sayaci=3708 ∈ [1,1000000000000.0] |
| 🟢 OK | ikincil_aralik | h08_boxcp_doz/s0 boxcp_p=1 ∈ [0.9999,1.0001] |
| 🟢 OK | ikincil_aralik | h08_boxcp_doz/s0 boxcp_max_sqrt_alan=96 ∈ [95.99,96.01] |
| 🟢 OK | ikincil_aralik | h08_boxcp_doz/s0 boxcp_doz_nesne_per_mosaic=0.7725 ∈ [0.6,1.5] |
| 🟢 OK | ikincil_aralik | h08_boxcp_doz/s0 boxcp_yer_bulunamadi_pay=0.03006 ∈ [0.0,0.25] |
| 🟢 OK | aralik | h08_boxcp_doz: ort 0.4058 ∈ [0.392078, 0.472078] (eslesmis kontrol baseline_cm0 duman 0.422078 (20260930-161437_baseline_cm0_s0; ayni ayar: seed0, fraction 0.1, 3 ep, close_mosaic=0). Alt -3 puan (H03 duman kalibi: cokme/uygulama hatasi; duman tek-seed gurultusu genis), ust +5 puan (aciklanamayan sicrama alarmi; kart kirmizi bayragi +1.0 tam kosum icindir). AP_S/mAP50 kirmizi bayraklari duman icin YOK: H03 dumaninda AP_S +6.3 puan oynadi (3 ep, %10 veri gurultusu), kriter tam-kosum icindir.) |
| 🟢 OK | ikincil_aralik | h08_boxcp_doz/s0 egitim_sure_sn=80.85 ∈ [1,600] |
