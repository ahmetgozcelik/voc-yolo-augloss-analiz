# Replikasyon Kapısı — YEŞİL
*2026-09-30 16:10* · metrik `rep_test_map50_95` (yüksek iyi)

| | değer |
|---|---|
| Literatür değeri/aralığı | 0.61 – 0.635 |
| Kaynak | notlar/literatur-cipa.md §2.3: YOLO11n sıfırdan 500-600ep 0.610-0.613 (Zhang 2025; YOLO-Master) alt / YOLOv8n 200ep 0.635 (El Hamdouni 2025) üst; tol=2xSD (YOLO-PEFT SD≤0.006) |
| Tolerans (mutlak) | ±0.012 → kabul aralığı [0.598, 0.647] |
| Bizim baseline (baseline) | **0.6349 ± 0.00111** (n=3 seed: 0, 1, 2) |
| Seed değerleri | 0.6345, 0.6361, 0.6341 |

**Karar:** YEŞİL — Baseline literatür aralığını tolerans içinde tekrarlıyor. Manipülasyon fazı açılabilir.

Bu tablo makalenin Deneysel Kurulum bölümüne "baseline replikasyonu" olarak girer
(literatür değeri `kaynak: literatur` etiketiyle evidence.json'a alınır).
