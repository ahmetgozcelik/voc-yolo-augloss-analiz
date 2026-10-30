# Hata analizi (hafif, test) — `20261001-212720_h03_boxcp-test_s0`

*Üretici: `scripts/analiz/hata_analizi.py --preds` · 4952 görüntü, 12032 nesne, 213494 tahmin (conf≥0.001) · eşleştirme pycocotools COCOeval · işletim noktası skor≥0.25. KEŞİFSEL, post-hoc: K5 test kilidi açıldıktan sonra mevcut tahminlerden CPU'da üretildi; hiçbir seçim/karar bu çıktıya dayanmaz.*

pycocotools mAP50-95=63.07, mAP50=83.17, mAP75=69.19.

## Hata türü sayıları

| tür | sayı (tüm conf) | sayı skor≥0.25 |
|---|---|---|
| TP | 11753 | 10152 |
| Dupe | 12080 | 527 |
| Cls | 14712 | 473 |
| Loc | 39404 | 908 |
| Both | 24784 | 375 |
| Bkg | 110761 | 1744 |
| Miss | 279 | 1880 |

## Oracle ΔAP (puan)

| düzeltilen | ΔAP50 | ΔAP50-95 |
|---|---|---|
| Cls | 2.79 | 2.11 |
| Loc | 2.72 | 5.35 |
| Both | 0.93 | 0.63 |
| Dupe | 0.45 | -0.93 |
| Bkg | 3.88 | 2.53 |
| Miss | 2.11 | 1.55 |
| FP_hepsi | 13.96 | 6.24 |
| Kutu_kusursuz | 0.00 | 20.09 |
| TP_IoU_sirali | 1.25 | 6.30 |

Taban: AP50=83.17, AP50-95=63.07. Not: Dupe/Loc ΔAP50-95 IoU=0.5 tanımına göre.
