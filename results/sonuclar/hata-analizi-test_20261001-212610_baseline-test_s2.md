# Hata analizi (hafif, test) — `20261001-212610_baseline-test_s2`

*Üretici: `scripts/analiz/hata_analizi.py --preds` · 4952 görüntü, 12032 nesne, 238921 tahmin (conf≥0.001) · eşleştirme pycocotools COCOeval · işletim noktası skor≥0.25. KEŞİFSEL, post-hoc: K5 test kilidi açıldıktan sonra mevcut tahminlerden CPU'da üretildi; hiçbir seçim/karar bu çıktıya dayanmaz.*

pycocotools mAP50-95=63.32, mAP50=83.07, mAP75=69.43.

## Hata türü sayıları

| tür | sayı (tüm conf) | sayı skor≥0.25 |
|---|---|---|
| TP | 11789 | 10124 |
| Dupe | 13067 | 600 |
| Cls | 17148 | 484 |
| Loc | 43317 | 959 |
| Both | 28384 | 412 |
| Bkg | 125216 | 1832 |
| Miss | 243 | 1908 |

## Oracle ΔAP (puan)

| düzeltilen | ΔAP50 | ΔAP50-95 |
|---|---|---|
| Cls | 2.62 | 2.00 |
| Loc | 2.66 | 5.25 |
| Both | 0.98 | 0.66 |
| Dupe | 0.52 | -0.94 |
| Bkg | 3.84 | 2.49 |
| Miss | 1.90 | 1.38 |
| FP_hepsi | 14.21 | 6.31 |
| Kutu_kusursuz | 0.00 | 19.75 |
| TP_IoU_sirali | 1.33 | 6.23 |

Taban: AP50=83.07, AP50-95=63.32. Not: Dupe/Loc ΔAP50-95 IoU=0.5 tanımına göre.
