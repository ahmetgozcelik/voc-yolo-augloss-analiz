# K5 Test Analizi (K5-TEST-PLANI.md §3'e göre; test bir kez açıldı, D1)

Kaynak: `scripts/analiz/k5_test_analiz.py` (metrics.json + test_paket.npz; yeni koşum yok). Birincil metrik test mAP50-95 (puan = x100), n=3 seed {0,1,2}, seed-eşli, 4952 test görüntüsü.
Tutarlılık: paket mAP ile metrics.json farkı <1e-6 (9/9); görüntü adları üç varyantta aynı=True; baseline-test ile rep_test farkı = [0.0, 0.0, 0.0] (≤0.0005 şartı sağlandı); test_n_goruntu=4952.

## T-A Test ana tablo (ort ± sd, n=3)

| varyant | mAP50-95 | mAP50 | mAP75 | AP_S | AP_M | AP_L |
|---|---|---|---|---|---|---|
| baseline-test | 63.49 ± 0.11 | 83.51 ± 0.17 | 69.62 ± 0.19 | 23.11 ± 1.26 | 45.19 ± 0.46 | 69.85 ± 0.10 |
| h03_boxcp-test | 63.37 ± 0.20 | 83.49 ± 0.08 | 69.51 ± 0.13 | 23.91 ± 1.95 | 45.66 ± 0.57 | 69.63 ± 0.12 |
| baseline-tta-test (T1) | 64.32 ± 0.11 | 83.61 ± 0.10 | 70.74 ± 0.19 | 22.63 ± 2.07 | 44.12 ± 0.47 | 71.36 ± 0.26 |

## Δ vs baseline-test (puan; seed-eşli çift taraflı t, df=2)

| karşılaştırma | metrik | Δ ort ± sd | Δ s0 / s1 / s2 | p |
|---|---|---|---|---|
| H03 boxcp | mAP50-95 | -0.12 ± 0.21 | -0.31 / -0.16 / +0.10 | 0.416 |
| H03 boxcp | mAP50 | -0.02 ± 0.21 | -0.13 / -0.15 / +0.22 | 0.883 |
| H03 boxcp | mAP75 | -0.11 ± 0.13 | -0.10 / -0.25 / +0.01 | 0.268 |
| H03 boxcp | AP_S | +0.81 ± 2.55 | -1.18 / -0.08 / +3.68 | 0.639 |
| H03 boxcp | AP_M | +0.48 ± 0.14 | +0.44 / +0.63 / +0.36 | 0.027 |
| H03 boxcp | AP_L | -0.23 ± 0.09 | -0.32 / -0.23 / -0.14 | 0.048 |
| TTA T1 | mAP50-95 | +0.83 ± 0.22 | +0.88 / +0.59 / +1.02 | 0.022 |
| TTA T1 | mAP50 | +0.10 ± 0.27 | +0.04 / -0.13 / +0.40 | 0.579 |
| TTA T1 | mAP75 | +1.12 ± 0.30 | +1.19 / +0.79 / +1.38 | 0.023 |
| TTA T1 | AP_S | -0.48 ± 0.82 | -0.74 / +0.44 / -1.13 | 0.416 |
| TTA T1 | AP_M | -1.07 ± 0.18 | -0.92 / -1.02 / -1.27 | 0.009 |
| TTA T1 | AP_L | +1.50 ± 0.36 | +1.59 / +1.10 / +1.81 | 0.019 |

Çoklu karşılaştırma (mAP50-95, 2 karşılaştırma): Holm p: H03 0.416, TTA 0.044; Bonferroni: H03 0.832, TTA 0.044. Ana iddia H03 vs baseline (ön-kayıt §3.4); TTA keşifsel. İkincil metrik p'leri düzeltmesizdir (keşifsel).

## Görüntü-düzeyi eşli bootstrap (B=1000, aynı görüntü indeksleri iki varyanta; Δ mAP50-95, puan)

| karşılaştırma | seed | Δ ort | %95 CI | tek yönlü %90 alt sınır |
|---|---|---|---|---|
| H03 | s0 | -0.300 | [-0.744, +0.154] | -0.576 |
| H03 | s1 | -0.164 | [-0.625, +0.253] | -0.470 |
| H03 | s2 | +0.113 | [-0.338, +0.540] | -0.185 |
| H03 | birleşik (3 seed ort.) | -0.117 | [-0.395, +0.155] | -0.286 |
| H03 | birleşik | P(Δ>0)=0.195; P(Δ>=+0.30)=0.000 | | |
| TTA | s0 | +0.855 | [+0.464, +1.252] | +0.579 |
| TTA | s1 | +0.573 | [+0.184, +0.993] | +0.302 |
| TTA | s2 | +1.005 | [+0.611, +1.406] | +0.734 |
| TTA | birleşik (3 seed ort.) | +0.811 | [+0.503, +1.140] | +0.604 |
| TTA | birleşik | P(Δ>0)=1.000; P(Δ>=+0.30)=0.999 | | |

## T-D Val -> test tutarlılığı

| karşılaştırma | val Δ (puan) | test Δ (puan) | not |
|---|---|---|---|
| H03 vs baseline (n=3 eşli) | +0.25 ± 0.24 (s: +0.37 / -0.03 / +0.41; p=0.214) | -0.12 ± 0.21 (p=0.416) | işaret değişti; val Δ - test Δ = +0.37 |
| H03 mAP50 | val +0.74 ± 0.16 (3/3 pozitif) | -0.02 ± 0.21 | val'deki tutarlı AP50 kazancı test'e taşınmadı |
| TTA T1 vs baseline | +0.80 (val, tek best.pt = baseline s0; H07 A/B yarıları +0.65/+1.05) | +0.83 ± 0.22 (n=3) | yön ve büyüklük korundu |

## T-B Hız (batch1, FP16, uçtan uca ms, conf .001; seed ort ± sd; aynı test oturumu)

| varyant | gecikme ms | FPS | gecikme oranı (baseline-test = 1) |
|---|---|---|---|
| baseline-test | 9.43 ± 0.25 | 106.1 ± 2.7 | 1.00 |
| h03_boxcp-test | 8.60 ± 0.40 | 116.5 ± 5.4 | 0.91 |
| baseline-tta-test (T1) | 21.64 ± 0.30 | 46.2 ± 0.6 | 2.29 |

Not: ölçüm süreç-zamanına duyarlıdır (baseline val-zamanı 8.0 ms, test oturumu 9.43 ms; H03 8.60 ms çıkarım mimarisi baseline ile aynı olduğundan fark ölçüm gürültüsüdür, H03 hız kazancı iddia edilmez). TTA oranı test oturumunda 2.29, H07 val oturumunda 2.57.

## T-C Sınıf düzeyi test AP50-95 Δ (seed ort., puan; keşifsel)

- h03: en çok kaybeden 5: horse -1.47, sofa -1.22, pottedplant -0.87, bus -0.80, dog -0.77; en çok kazanan 5: aeroplane +1.70, train +1.00, tvmonitor +0.54, boat +0.52, diningtable +0.50
- tta: en çok kaybeden 5: horse -0.26, bottle -0.23, dog -0.22, bicycle +0.03, motorbike +0.09; en çok kazanan 5: train +2.84, cat +2.31, boat +2.28, aeroplane +2.28, bird +1.13
