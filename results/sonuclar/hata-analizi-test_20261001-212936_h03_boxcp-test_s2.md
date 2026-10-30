# Hata analizi (hafif, test) — `20261001-212936_h03_boxcp-test_s2`

*Üretici: `scripts/analiz/hata_analizi.py --preds` · 4952 görüntü, 12032 nesne, 226477 tahmin (conf≥0.001) · eşleştirme pycocotools COCOeval · işletim noktası skor≥0.25. KEŞİFSEL, post-hoc: K5 test kilidi açıldıktan sonra mevcut tahminlerden CPU'da üretildi; hiçbir seçim/karar bu çıktıya dayanmaz.*

pycocotools mAP50-95=63.42, mAP50=83.28, mAP75=69.44.

## Hata türü sayıları

| tür | sayı (tüm conf) | sayı skor≥0.25 |
|---|---|---|
| TP | 11778 | 10111 |
| Dupe | 12723 | 574 |
| Cls | 15725 | 479 |
| Loc | 40927 | 929 |
| Both | 26137 | 413 |
| Bkg | 119187 | 1753 |
| Miss | 254 | 1921 |

## Oracle ΔAP (puan)

| düzeltilen | ΔAP50 | ΔAP50-95 |
|---|---|---|
| Cls | 2.64 | 1.99 |
| Loc | 2.72 | 5.35 |
| Both | 0.92 | 0.63 |
| Dupe | 0.47 | -0.91 |
| Bkg | 3.77 | 2.45 |
| Miss | 2.07 | 1.49 |
| FP_hepsi | 13.80 | 6.19 |
| Kutu_kusursuz | 0.00 | 19.86 |
| TP_IoU_sirali | 1.27 | 6.14 |

Taban: AP50=83.28, AP50-95=63.42. Not: Dupe/Loc ΔAP50-95 IoU=0.5 tanımına göre.
