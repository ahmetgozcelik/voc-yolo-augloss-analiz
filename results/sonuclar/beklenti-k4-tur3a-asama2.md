# Beklenti Kapısı — kuyruk `k4-tur3a` aşama 2 → **GEÇTİ**
*2026-10-01 18:25* · metrik `val_map50_95` (max) · beklenti: `k4-tur3a.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h08_boxcp_doz | 1 | 0.6024 | 0.6024 | 596.6 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | aralik | h08_boxcp_doz: ort 0.6024 ∈ [0.0, 1.0] (metrik tanimi geregi mAP orani [0,1]) |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | ikincil_aralik | h08_boxcp_doz/s0 varyant_etki_sayaci=5.252e+04 ∈ [1,1000000000000.0] |
| 🟢 OK | ikincil_aralik | h08_boxcp_doz/s0 boxcp_p=1 ∈ [0.9999,1.0001] |
| 🟢 OK | ikincil_aralik | h08_boxcp_doz/s0 boxcp_max_sqrt_alan=96 ∈ [95.99,96.01] |
| 🟢 OK | ikincil_aralik | h08_boxcp_doz/s0 boxcp_doz_nesne_per_mosaic=0.6738 ∈ [0.6,1.5] |
| 🟢 OK | ikincil_aralik | h08_boxcp_doz/s0 boxcp_yer_bulunamadi_pay=0.03431 ∈ [0.0,0.25] |
| 🟢 OK | aralik | h08_boxcp_doz: ort 0.6024 ∈ [0.588845, 0.613845] (eslesmis kontrol baseline_cm0 on-ucus 0.603845 (20260930-163903_baseline_cm0_s0; tam veri, 5 ep, close_mosaic=0); H03 on-ucus 0.604515 (seed gurultusu ~0.4 puan). Alt = kontrol -1.5 puan (~3.7x gurultu; altina dusus = uygulama hatasi suphesi). Ust = H08 karti kirmizi bayragi: Delta map50_95 > +1.0 puan.) |
| 🟢 OK | ikincil_aralik | h08_boxcp_doz/s0 val_map50=0.8015 ∈ [0.0,0.826047] |
| 🟢 OK | ikincil_aralik | h08_boxcp_doz/s0 val_map_small=0.1812 ∈ [0.0,0.252691] |
| 🟢 OK | ikincil_aralik | h08_boxcp_doz/s0 egitim_sure_sn=566.4 ∈ [1,900] |
