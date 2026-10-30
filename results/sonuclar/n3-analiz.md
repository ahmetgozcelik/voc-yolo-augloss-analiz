# n=3 val analizi (K5 sonrası, hakem B3)

*Üretici: `scripts/analiz/n3_analiz.py` (yalnız val, GPU yok). Puan = x100. Δ = seed-eşli (s0↔s0, s1↔s1, s2↔s2) kart − baseline; ort ± sd (ddof=1); p = tek yönlü eşli t (H1: Δ>0), çift yönlü parantezde; aralık = seed-düzeyi %95 t aralığı (df=2, t=4.303). Kabul ölçütü (ön-kayıtlı): ort Δ ≥ +0.40 VE tek yönlü p < 0.10. Ön-eleme: s0 Δ ≥ +0.30. H05 = H05r koşumları (arıza koşumu 20261001-114442 hariç). AP75: `hata_analizi.py` (pycocotools).*

Baseline (n=3, val): mAP50-95 63.07; AP50 81.21; AP_S 22.11; AP_M 44.28; AP_L 70.46

## 1. Ana tablo (mAP50-95)

| Kart | Δ ort ± sd | seed Δ (s0/s1/s2) | p tek (çift) | %95 t aralığı | Kabul | Ön-eleme (s0 Δ) | Ön-eleme isabeti |
|---|---|---|---|---|---|---|---|
| H01 | -0.21 ± 0.61 | +0.07 / -0.91 / +0.21 | 0.693 (0.614) | [-1.72, +1.30] | HAYIR | +0.07 → ELE (s0 < +0.30) | DOGRU (n=3'te de kabul yok) |
| H02 | -0.35 ± 0.12 | -0.22 / -0.47 / -0.35 | 0.980 (0.041) | [-0.66, -0.04] | HAYIR | -0.22 → ELE (s0 < +0.30) | DOGRU (n=3'te de kabul yok) |
| H03 | +0.25 ± 0.24 | +0.37 / -0.03 / +0.41 | 0.107 (0.214) | [-0.35, +0.85] | HAYIR | +0.37 → seed 1,2'ye cik | cikis yanlis alarm (n=3 kabul yok) |
| H04 | -0.85 ± 0.79 | -0.30 / -1.76 / -0.49 | 0.898 (0.204) | [-2.82, +1.12] | HAYIR | -0.30 → ELE (s0 < +0.30) | DOGRU (n=3'te de kabul yok) |
| H05 | -2.87 ± 0.33 | -2.88 / -3.19 / -2.53 | 0.998 (0.004) | [-3.68, -2.05] | HAYIR | -2.88 → ELE (s0 < +0.30) | DOGRU (n=3'te de kabul yok) |
| H06 | +0.04 ± 0.56 | -0.42 / -0.12 / +0.66 | 0.456 (0.912) | [-1.35, +1.43] | HAYIR | -0.42 → ELE (s0 < +0.30) | DOGRU (n=3'te de kabul yok) |
| H08 | -0.08 ± 0.40 | +0.22 / -0.53 / +0.06 | 0.623 (0.754) | [-1.07, +0.90] | HAYIR | +0.22 → ELE (s0 < +0.30) | DOGRU (n=3'te de kabul yok) |

## 2. İkincil metrikler (Δ ort ± sd, puan)

| Kart | AP50 | AP75 | AP_S | AP_M | AP_L |
|---|---|---|---|---|---|
| H01 | +0.23 ± 0.38 | -0.52 ± 1.08 | -0.30 ± 2.24 | +0.84 ± 1.60 | -0.18 ± 0.88 |
| H02 | -0.02 ± 0.16 | -0.42 ± 0.27 | +1.39 ± 1.09 | -0.27 ± 0.15 | -0.32 ± 0.38 |
| H03 | +0.74 ± 0.16 | -0.32 ± 0.18 | +1.34 ± 1.63 | +1.55 ± 2.85 | +0.34 ± 0.24 |
| H04 | -0.50 ± 0.71 | -0.95 ± 1.22 | -1.06 ± 0.68 | -0.75 ± 1.70 | -0.54 ± 0.78 |
| H05 | -4.44 ± 0.15 | -3.14 ± 0.22 | -2.39 ± 0.44 | -4.06 ± 0.69 | -2.10 ± 0.54 |
| H06 | +0.30 ± 0.55 | +0.07 ± 0.30 | +1.23 ± 2.93 | +0.19 ± 0.51 | -0.01 ± 0.55 |
| H08 | +0.25 ± 0.14 | -0.23 ± 0.28 | +0.31 ± 2.84 | -0.35 ± 1.52 | +0.14 ± 0.69 |

## 3. Koşumlar

- H01: s0=20260930-165837_h01_multiscale25_s0, s1=20261003-062922_h01_multiscale25_s1, s2=20261003-094536_h01_multiscale25_s2
- H02: s0=20260930-201415_h02_innerciou_s0, s1=20261002-184450_h02_innerciou_s1, s2=20261002-214253_h02_innerciou_s2
- H03: s0=20260930-231243_h03_boxcp_s0, s1=20261001-021011_h03_boxcp_s1, s2=20261001-050625_h03_boxcp_s2
- H04: s0=20261001-085011_h04_scale25_s0, s1=20261003-004041_h04_scale25_s1, s2=20261003-033504_h04_scale25_s2
- H05: s0=20261001-145045_h05_vfl_s0, s1=20261002-001746_h05_vfl_s1, s2=20261002-033917_h05_vfl_s2
- H06: s0=20261001-115531_h06_box10_s0, s1=20261002-125405_h06_box10_s1, s2=20261002-154859_h06_box10_s2
- H08: s0=20261001-182518_h08_boxcp_doz_s0, s1=20261002-070042_h08_boxcp_doz_s1, s2=20261002-095722_h08_boxcp_doz_s2
- baseline: s0=20260930-072103_baseline_s0, s1=20260930-101624_baseline_s1, s2=20260930-131427_baseline_s2

## 4. Okuma

- Kabul ölçütünü karşılayan kart: 0/7. n=3'te seed-düzeyi t-testinin tespit edebileceği en küçük etki ≈0.5 puan (df=2); tablo "etki yok" değil "kabul ölçütü sağlanmadı" der.
- Anlamlı (p<0.05) iki kart de baseline'ın ALTINDA: H02 (-0.35, çift yönlü p=0.041), H05 (-2.87, çift yönlü p=0.004) (negatif yönlü etkiler; çoklu karşılaştırma düzeltmesiz).
- Ön-eleme isabeti: s0 kapısında elenen 6 karttan 6'ü n=3'te de kabul edilmedi (hiçbir eleme tersine dönmedi: kaçırılan kazanç yok). s0 kapısını geçen kart(lar): H03 (n=3'te kabul yok; kapı gürültü tabanında olduğundan çıkış hakkı yanlış alarm çıktı).
