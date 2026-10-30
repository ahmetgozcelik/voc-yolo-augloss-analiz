# Replikasyon Kapısı — YEŞİL
*2026-09-30 16:10* · metrik `rep_test_map50` (yüksek iyi)

| | değer |
|---|---|
| Literatür değeri/aralığı | 0.812 – 0.836 |
| Kaynak | notlar/literatur-cipa.md §2.3: YOLO11n sıfırdan 0.812-0.815 alt / YOLOv8n 200ep 0.836 üst; tol=2xSD varsayımı |
| Tolerans (mutlak) | ±0.012 → kabul aralığı [0.8, 0.848] |
| Bizim baseline (baseline) | **0.8351 ± 0.00169** (n=3 seed: 0, 1, 2) |
| Seed değerleri | 0.8354, 0.8367, 0.8333 |

**Karar:** YEŞİL — Baseline literatür aralığını tolerans içinde tekrarlıyor. Manipülasyon fazı açılabilir.

Bu tablo makalenin Deneysel Kurulum bölümüne "baseline replikasyonu" olarak girer
(literatür değeri `kaynak: literatur` etiketiyle evidence.json'a alınır).
