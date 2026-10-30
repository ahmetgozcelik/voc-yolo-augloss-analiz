# Beklenti Kapısı — kuyruk `k3-baseline-v2` aşama 2 → **GEÇTİ**
*2026-09-30 07:21* · metrik `val_map50_95` (max) · beklenti: `k3-baseline-v2.beklenti.json` · 2 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| baseline | 2 | 0.5888 | 0.5867, 0.5908 | 582.7 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | farkli_seedler_farkli | baseline: seed'ler farklı |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | aralik | baseline: ort 0.5888 ∈ [0.35, 0.75] (Alt sınır (0.35), kısaltılmış eğitim için GEVŞETİLDİ: COCO ön-eğitimli YOLO11n, VOC'un 20 sınıfının tamamını COCO'dan zaten tanıyor (literatur-cipa.md §3 madde4: aeroplane→airplane, sofa→couch vb. örtüşme); K3 duman kanıtı (bu WS, 2026-09-29, seed0/fraction=0.05/2 epoch) val_map50_95=0.1267'ye 2 epoch/%5 veride ulaştı, 5 epoch/%100 veride bunun belirgin üstü beklenir. Üst sınır (0.75) CÖMERT ÜST TAVAN: 100-epoch tam koşumun beklenen üst ucu (~0.70, K1.5 çıpası) 5 epoch'ta aşılması/eşitlenmesi sızıntı veya val/train karışması şüphesi doğurur (görülmemiş-iyi eşiği).) |
| 🟢 OK | ikincil_aralik | baseline/s0 egitim_sure_sn=556.7 ∈ [1,3600] |
| 🟢 OK | ikincil_aralik | baseline/s1 egitim_sure_sn=551.1 ∈ [1,3600] |
