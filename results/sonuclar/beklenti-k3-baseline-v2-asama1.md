# Beklenti Kapısı — kuyruk `k3-baseline-v2` aşama 1 → **GEÇTİ**
*2026-09-30 07:01* · metrik `val_map50_95` (max) · beklenti: `k3-baseline-v2.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| baseline | 1 | 0.4104 | 0.4104 | 115.3 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | aralik | baseline: ort 0.4104 ∈ [0.0, 1.0] (metrik tanımı gereği mAP oranı [0,1] (Ultralytics metrics/mAP50-95(B))) |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
