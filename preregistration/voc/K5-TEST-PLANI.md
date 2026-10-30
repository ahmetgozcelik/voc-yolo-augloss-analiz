# K5 Test Planı (ön-kayıt) — YAZILDI: test kilidi AÇILMADAN önce (2026-10-01)

Durum: kod + kuyruk hazır, **hiçbir test koşumu yapılmadı**. Kilit şefin K5 kararıyla bir kez açılır (D1).
Bu belge test sonucundan önce yazıldığı için ön-kayıt niteliğindedir; sonuç ne olursa raporlanır (D3), test'e bakıp geri dönüp arama yapılmaz.

## 1. Araç ve eşdeğerlik kanıtı
- `WS/scripts/k5_test_degerlendir.py` — eğitimsiz değerlendirme; `rakip_kos.degerlendir` ile aynı `DetectionValidator` ayarları (split=val[yaml], imgsz 640, batch 16, conf 0.001 / iou 0.7 varsayılan, save_json). `--tta-t1` = H07 T1 yapılandırması (Ultralytics [1, .83, .67] + lr-flip; h07 `tta_yamasi_uygula`).
- Anahtar öneki `--onek test|val`. Korumalar: `onek=test` yalnız `data/voc_test.yaml` + '-test' son ekli varyant adı; `onek=val` iken `voc_test.yaml` reddedilir; metrics.json'da karşı önek anahtarı yoksa assert.
- Çıktılar (`{RUN_DIR}`): `metrics.json` (`test_map50_95/map50/map75/precision/recall/map_small/map_medium/map_large`, `alt_grup` sınıf başı `ap50_/ap5095_`, `latency_toplam_ms`/`fps` = h07 uçtan uca batch1 FP16 conf .001 (3 tur medyan), `latency_dep_ms` = conf .25 tek-etiket, `egitim_yok=1`, `param_sayisi`, `cihaz`), `test_preds.json`, `test_paket.npz` (görüntü-düzeyi tp/conf/sınıf istatistiği, bootstrap girdisi). Gecikme ölçüm görüntüleri `data/splits/val.txt`'in ilk 350'sidir (test görüntüsü kullanılmaz).
- **Kuru test (VAL, CPU, CUDA_VISIBLE_DEVICES=-1; test'e dokunulmadı):**
  - Tam VAL (1000 görüntü), baseline_s0 best.pt: `val_map50_95 = 0.629006` vs baseline metrics.json `0.62893` → fark `+0.000076` (< ±0.0005) ✔. Kendi `ap_per_class` hesabı validator ile birebir (fark 0).
  - 20 görüntülük VAL alt kümesi: `--tta-t1` yolu çalışıyor ve tek ölçekten farklı sonuç veriyor (0.7017 vs 0.6470; yalnız yol doğrulaması, anlamlı değil); gecikme + npz + preds yazımı çalışıyor (CPU gecikmesi anlamsız, GPU'da ölçülecek).
  - Kalan risk: GPU FP16 vs CPU FP32 küçük sayısal fark (VAL'de GPU'da h07 R0 0.0008 içinde yeniden üretmişti); test satırlarında tutarlılık kontrolü §4'te.

## 2. Kuyruk `WS/kuyruk/k5-test.jsonl` (9 satır; BAŞLATILMADI; H08 satırları şef H08 sonucuna göre ekler)
Her satır `--gpu-zorunlu`, `--asama 3`, 30 dk sınırı, ebeveyn = best.pt'nin eğitim run'ı.
| Varyant | Seed | best.pt kaynağı | Neden |
|---|---|---|---|
| baseline-test | 0,1,2 | `20260930-072103_baseline_s0`, `20260930-101624_baseline_s1`, `20260930-131427_baseline_s2` | Referans; aynı kodla tam metrik + gecikme tek tabloda; `rep_test_*` ile tutarlılık kontrolü |
| h03_boxcp-test | 0,1,2 | `20260930-231243_h03_boxcp_s0`, `20261001-021011_h03_boxcp_s1`, `20261001-050625_h03_boxcp_s2` | En güçlü (eşik-altı) aday; makalede analiz için (kazanan ilan edilmez, K4 kararı sabit) |
| baseline-tta-test | 0,1,2 | baseline best.pt'leri (yukarıdaki) + `--tta-t1` | H07 T1 = KAZANC-B satırı; gecikme maliyetiyle birlikte test'te doğrulama |

Seed kümesi {0,1,2} her üç varyantta aynıdır (M5). H08 eklenirse aynı seed'ler ve aynı kod.

## 3. Eşleştirilmiş test planı (kilit açıldıktan sonra, `WS/scripts/analiz/` altında; tek geçişli, analiz kararı verilerden bağımsız önceden sabit)
Birincil metrik: `test_map50_95` (K4 seçim metriğiyle aynı). Karşılaştırmalar (hepsi baseline-test'e karşı, seed-eşli):
1. **Seed-eşli t-testi** (`scipy.stats.ttest_rel`, n=3, çift taraflı; Δ_s = varyant_s − baseline_s). n=3'te güç düşüktür; p tek başına karar vermez — Δ ort ± std, her seed'in Δ işareti ve §3.2 birlikte raporlanır.
2. **Görüntü-düzeyi eşleştirilmiş bootstrap CI** (`test_paket.npz`'den `h07.Paket` ile): her seed için B=1000 örneklemde aynı görüntü indeksleri iki varyanta uygulanır, Δ mAP50-95 hesaplanır, %95 yüzdelik CI (ve tek yönlü %90 alt sınır, h07 kuralıyla tutarlı). Seed'ler arası birleşik CI için: her bootstrap turunda aynı görüntü indeksiyle üç seed'in Δ ortalaması alınır (görüntü-yeniden-örnekleme belirsizliği; eğitim-seed varyansı t-testinde ayrıca raporlanır).
3. **"Anlamlı" kelimesi** yalnız p<0.05 **ve** pratik marj (≥ +0.30 puan mAP50-95, H07/H03 kartlarıyla tutarlı) sağlanırsa kullanılır; aksi halde "fark saptanamadı / yön tutarlı ama anlamsız".
4. Çoklu karşılaştırma: ana iddia = tek (kazanan veya H03 vs baseline); diğerleri (TTA, H08) keşifsel olarak etiketlenir; Bonferroni/Holm düzeltilmiş p ayrıca verilir.
5. **Tutarlılık kontrolleri (hata olursa raporlanır, sonuç düzeltilmez):** (a) baseline-test `test_map50_95` ↔ metrics.json `rep_test_map50_95` her seed için |fark| ≤ 0.0005; (b) T1 gecikme oranı h07 VAL değeriyle (≈ lat_oran) kıyas; (c) `test_n_goruntu = 4952`.
6. Sınıf düzeyi (20 sınıf, `alt_grup`) ve boyut (AP_S/M/L) Δ'ları: seed ortalaması + sınıf başı bootstrap; **keşifseldir**, birincil iddia değildir.

## 4. Raporlanacak tablolar (makale/`karar.md`)
- T-A: Test ana tablo — varyant × {mAP50-95, mAP50, mAP75, AP_S/M/L} ort ± std (n=3), Δ vs baseline, p (eşli t), bootstrap CI.
- T-B: Hız/maliyet — gecikme (ms, batch1 FP16, uçtan uca), FPS, param, TTA oranı; "≥30 FPS" bayrağı.
- T-C: Sınıf başı AP50-95 baseline vs aday (Δ) + en çok kazanan/kaybeden 5 sınıf.
- T-D: Val → test tutarlılığı (K4 val Δ vs test Δ, genelleme boşluğu = val ortalaması − test ortalaması).
- T-E: Replikasyon tutarlılığı (rep_test vs k5 baseline-test).
- Beyan: "Test seti K5'te bir kez koşuldu; bu karardan sonra arama yapılmadı" (D1); koşumlar hangi run_id'lerle yapıldı, kod `k5_test_degerlendir.py` sürümü, ultralytics 8.4.165.

## 5. Kurallar
- Test kilidi açılışı ve koşum şefin kararıdır; kuyruk şef tarafından başlatılır. Yürütücü başlatmaz.
- Sonuç ne olursa yazılır; test sonucu baktıktan sonra yeni hipotez/eğitim/hiperparametre araması yapılmaz. Yapılırsa makalede açıkça beyan edilir.
- Çalışmayan/yarım satır varsa (rc≠0) aynı kod, aynı best.pt ile yeniden koşulabilir (eğitimsiz, deterministik değerlendirme; bu "ikinci test açılışı" sayılmaz ama günlüğe yazılır).
