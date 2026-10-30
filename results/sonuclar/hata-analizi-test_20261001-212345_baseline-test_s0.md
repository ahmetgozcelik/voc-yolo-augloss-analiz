# Hata analizi (hafif, test) — `20261001-212345_baseline-test_s0`

*Üretici: `scripts/analiz/hata_analizi.py --preds` · 4952 görüntü, 12032 nesne, 222968 tahmin (conf≥0.001) · eşleştirme pycocotools COCOeval · işletim noktası skor≥0.25. KEŞİFSEL, post-hoc: K5 test kilidi açıldıktan sonra mevcut tahminlerden CPU'da üretildi; hiçbir seçim/karar bu çıktıya dayanmaz.*

pycocotools mAP50-95=63.36, mAP50=83.24, mAP75=69.28.

## Hata türü sayıları

| tür | sayı (tüm conf) | sayı skor≥0.25 |
|---|---|---|
| TP | 11766 | 10109 |
| Dupe | 12566 | 530 |
| Cls | 16311 | 470 |
| Loc | 40259 | 951 |
| Both | 27010 | 404 |
| Bkg | 115056 | 1837 |
| Miss | 266 | 1923 |

## Oracle ΔAP (puan)

| düzeltilen | ΔAP50 | ΔAP50-95 |
|---|---|---|
| Cls | 2.74 | 1.98 |
| Loc | 2.73 | 5.31 |
| Both | 0.95 | 0.64 |
| Dupe | 0.45 | -0.91 |
| Bkg | 3.76 | 2.45 |
| Miss | 2.04 | 1.47 |
| FP_hepsi | 13.94 | 6.27 |
| Kutu_kusursuz | 0.00 | 19.88 |
| TP_IoU_sirali | 1.28 | 6.31 |

Taban: AP50=83.24, AP50-95=63.36. Not: Dupe/Loc ΔAP50-95 IoU=0.5 tanımına göre.
