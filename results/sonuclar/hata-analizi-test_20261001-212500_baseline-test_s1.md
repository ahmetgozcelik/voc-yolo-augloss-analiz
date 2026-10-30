# Hata analizi (hafif, test) — `20261001-212500_baseline-test_s1`

*Üretici: `scripts/analiz/hata_analizi.py --preds` · 4952 görüntü, 12032 nesne, 210486 tahmin (conf≥0.001) · eşleştirme pycocotools COCOeval · işletim noktası skor≥0.25. KEŞİFSEL, post-hoc: K5 test kilidi açıldıktan sonra mevcut tahminlerden CPU'da üretildi; hiçbir seçim/karar bu çıktıya dayanmaz.*

pycocotools mAP50-95=63.52, mAP50=83.36, mAP75=69.65.

## Hata türü sayıları

| tür | sayı (tüm conf) | sayı skor≥0.25 |
|---|---|---|
| TP | 11756 | 10110 |
| Dupe | 12335 | 530 |
| Cls | 15116 | 466 |
| Loc | 38823 | 923 |
| Both | 25116 | 401 |
| Bkg | 107340 | 1738 |
| Miss | 276 | 1922 |

## Oracle ΔAP (puan)

| düzeltilen | ΔAP50 | ΔAP50-95 |
|---|---|---|
| Cls | 2.54 | 1.88 |
| Loc | 2.70 | 5.36 |
| Both | 0.98 | 0.67 |
| Dupe | 0.47 | -0.87 |
| Bkg | 3.78 | 2.44 |
| Miss | 2.15 | 1.57 |
| FP_hepsi | 13.72 | 6.14 |
| Kutu_kusursuz | 0.00 | 19.85 |
| TP_IoU_sirali | 1.19 | 6.10 |

Taban: AP50=83.36, AP50-95=63.52. Not: Dupe/Loc ΔAP50-95 IoU=0.5 tanımına göre.
