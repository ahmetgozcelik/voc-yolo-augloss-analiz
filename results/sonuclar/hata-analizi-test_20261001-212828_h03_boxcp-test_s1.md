# Hata analizi (hafif, test) — `20261001-212828_h03_boxcp-test_s1`

*Üretici: `scripts/analiz/hata_analizi.py --preds` · 4952 görüntü, 12032 nesne, 227183 tahmin (conf≥0.001) · eşleştirme pycocotools COCOeval · işletim noktası skor≥0.25. KEŞİFSEL, post-hoc: K5 test kilidi açıldıktan sonra mevcut tahminlerden CPU'da üretildi; hiçbir seçim/karar bu çıktıya dayanmaz.*

pycocotools mAP50-95=63.39, mAP50=83.27, mAP75=69.40.

## Hata türü sayıları

| tür | sayı (tüm conf) | sayı skor≥0.25 |
|---|---|---|
| TP | 11777 | 10102 |
| Dupe | 12691 | 569 |
| Cls | 15658 | 458 |
| Loc | 40707 | 950 |
| Both | 26769 | 373 |
| Bkg | 119581 | 1783 |
| Miss | 255 | 1930 |

## Oracle ΔAP (puan)

| düzeltilen | ΔAP50 | ΔAP50-95 |
|---|---|---|
| Cls | 2.59 | 1.93 |
| Loc | 2.69 | 5.37 |
| Both | 0.88 | 0.59 |
| Dupe | 0.48 | -0.93 |
| Bkg | 3.90 | 2.53 |
| Miss | 2.01 | 1.45 |
| FP_hepsi | 14.01 | 6.19 |
| Kutu_kusursuz | 0.00 | 19.88 |
| TP_IoU_sirali | 1.35 | 6.32 |

Taban: AP50=83.27, AP50-95=63.39. Not: Dupe/Loc ΔAP50-95 IoU=0.5 tanımına göre.
