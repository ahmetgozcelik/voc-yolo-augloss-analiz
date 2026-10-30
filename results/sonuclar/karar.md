# K5 Karar Taslağı (deney-tahlilci, mod karar) — konu 2.1 PASCAL VOC / YOLO11n

*Karar VERİLMEZ; taslak. Kullanıcı yetkisiyle (otonom-orkestra tercihi) şef "(Önerilen)" çıkışı uygular. Test kilidi 2026-10-01 21:23'te bir kez açıldı (D1); sonrasında arama yapılmadı. Sayıların kaynağı: `test-analiz.md/json` (script `scripts/analiz/k5_test_analiz.py`), `leaderboard-val.md`, `leaderboard-test.md`, `ablasyon.csv`, `basarisiz-denemeler.md`, `h07_post.csv`. Puan = x100.*

## 1. Ölçülen özet

| Ölçüm | baseline | H03 (en iyi eğitim kartı) | TTA T1 (eğitimsiz, baseline best.pt) |
|---|---|---|---|
| Val mAP50-95 (n=3; TTA: H07, 3 best.pt — s0 tam val, s1/s2 onay) | 63.07 ± 0.41 | 63.32 ± 0.17 (Δ +0.25 ± 0.24, p=0.214 çift / 0.107 tek yönlü) | Δ +0.80 (s0) / +0.40 (s1) / +1.22 (s2) (H07) |
| **Test mAP50-95 (n=3, 4952 görüntü)** | **63.49 ± 0.11** | **63.37 ± 0.20** | **64.32 ± 0.11** |
| Test Δ (seed-eşli) | – | **−0.12 ± 0.21**, seed Δ: −0.31 / −0.16 / +0.10, **p=0.416** (Holm 0.416) | **+0.83 ± 0.22**, seed Δ: +0.88 / +0.59 / +1.02, **p=0.022** (Holm 0.044) |
| Eşli bootstrap %95 CI (B=1000, 3 seed birleşik) | – | [−0.40, +0.16]; P(Δ≥+0.30)=0.000 | [+0.50, +1.14]; P(Δ≥+0.30)=0.999; tek yönlü %90 alt sınır +0.60 |
| Pratik marj (+0.30) | – | CI üst sınırı (+0.16) marjın altında → dışlandı | CI alt sınırı (+0.50) marjın üstünde |
| Test mAP50 / mAP75 Δ | – | −0.02 / −0.11 | +0.10 / +1.12 (p=0.023) |
| Test AP_S / AP_M / AP_L Δ | – | +0.81 (p=0.64) / +0.48 (p=0.027) / −0.23 (p=0.048) | −0.48 / −1.07 (p=0.009) / +1.50 (p=0.019) |
| Gecikme (b1 FP16 uçtan uca, aynı oturum) | 9.43 ± 0.25 ms (106 FPS) | 8.60 ms (aynı mimari; fark ölçüm gürültüsü) | 21.64 ± 0.30 ms (46 FPS) = **×2.29** (H07 val oturumu ×2.57) |

Val↔test tutarlılığı: **H03 val +0.25 → test −0.12** (işaret değişti; val Δ − test Δ = 0.37; val'deki 3/3 seed'de pozitif mAP50 +0.74 test'te −0.02). **TTA val +0.80 → test +0.83**: yön ve büyüklük korundu. Replikasyon tutarlılığı: baseline-test ile K3 `rep_test` farkı 0.0 (3/3 seed; şart ≤0.0005); paket mAP ile metrics farkı <1e-6; test_n_goruntu=4952. Test seed std (0.11) val std'sinden (0.41) ~3.7 kat küçük: 1000 görüntülük val seçim için gürültülü bir ölçü. n=3'te eşli t-testinin tespit edebileceği en küçük etki yaklaşık 0.5 puan (sd_Δ≈0.21, t_0.975,df2=4.30).

Not: H03 hız farkı (−0.84 ms) ve baseline'ın val-zamanı (8.0 ms) ile test-oturumu (9.43 ms) arasındaki fark ölçüm oturumu etkisidir; hız kazancı iddia edilmez. Yeni gecikme ölçümü yapılmadı.

## 2. Üç çıkış için kanıt değerlendirmesi

### A — GEÇTİ: ELENDİ (kısmi pozitif bulgu var)
Koşullar (anlamlı + pratik + bayraksız + ablasyon tutarlı): eğitim-zamanı augmentation/loss kartlarının hiçbiri ön-kayıtlı kapıyı (seed0 Δ≥+0.30; kabul: 3 seed ort. Δ≥+0.40, tek yönlü p<0.10) geçmedi: 0/7 TUT (K5 sonrası n=3 seed-eşli val Δ: H01 -0.21, H02 -0.35, H03 +0.25, H04 -0.85, H05 -2.87, H06 +0.04, H08 -0.08; bkz. §7). Test'te en iyi eğitim kartı H03 baseline'ın altında (−0.12, CI üst +0.16). Tek anlamlı test kazancı TTA: +0.83 (p=0.022, Holm 0.044, CI [+0.50, +1.14]); ancak (i) eğitimsiz, standart bir çıkarım tekniği ve konunun "augmentation ve loss iyileştirmesi" kapsamı dışında, (ii) gecikme ×2.29–2.57 (ön-kayıtlı KAZANÇ-A ölçütü ≤1.10× karşılanmadı; 46 FPS ≥30 FPS olsa da), (iii) konunun hedefi +1–3 puanın alt sınırının altında (+0.83), (iv) keşifsel karşılaştırma, ana iddia (H03) değil. Bu yüzden "tez geçti" demek abartı olur.

### B — GEÇEMEDİ-ANLAŞILDI: **ÖNERİLEN**
Mekanizma kanıtı çok yönlü (bkz. §3). Replikasyon YEŞİL (baseline literatür aralığında) olduğundan negatif sonuç bozuk kurulumdan değil, ölçülen doygunluk ve gürültüden kaynaklanıyor. D3'e göre meşru çıkış.
**Tez cümlesi (taslak "GÜNCEL DURUM" ve makale özet/sonuç diliyle birebir; tur-2):** "COCO ön-eğitimli YOLO11n'in PASCAL VOC ince ayarında, ayarsız ve tek-noktalı yedi eğitim-zamanı müdahalesinin hiçbiri 3 tohumlu doğrulamada ön-kayıtlı kabul ölçütünü karşılamadı (Holm-7 sonrası yalnız H05 −2.87 ayrışır; H02 −0.35 yalnız düzeltmesiz p=0.041); teste giren tek eğitim kartı H03 testte de geçmedi (−0.12); eğitimsiz TTA testte +0.83 puan verdi (keşifsel, Holm-12 p=0.222) ve gecikmeyi ×2.29 artırdı. Bu rejimde Δ≥+0.30 gösterilemedi; 3 tohumlu doğrulamanın %50 güçle ayırabildiği etki ≈1.2 puandır, daha küçük etkiler dışlanamaz. 'Yerel optimum' denmez; nedenler hipotez düzeyindedir."

### C — TEKRARLANAMADI: ELENDİ
Replikasyon kapısı YEŞİL: baseline test mAP50-95 0.6349 ± 0.0011 (aralık 0.610–0.635, tol ±0.012 → [0.598, 0.647]), mAP50 0.8351 ± 0.0017 (aralık 0.812–0.836). Literatürde protokolün birebir karşılığı yok (çıpa güveni düşük-orta; farklı epoch/başlatma), dolayısıyla C'yi destekleyen bir protokol-farkı ölçümü de yok.

## 3. B için bulgular (sayılarla; nedenler hipotez düzeyinde)

1. **Varsayılan augmentation/kayıp dengesi bu rejimde aşılamadı (optimum iddiası yok).** Sekiz kart: ölçek ekseni iki yönde (H01 çok-ölçekli -0.21; H04 scale daraltma -0.85), kutu-kaybı iki müdahale (H02 Inner-CIoU -0.35, AP75 -0.42; H06 box 7.5→10 +0.04, AP75 +0.07), cls kaybı (H05 VFL -2.87, AP50 -4.44), copy-paste doz ekseni (H03 doz 0.14: +0.25; H08 doz 0.67: -0.08). Doz–yanıt düz/ters; hiçbir kart baseline'ı eşik üstünde geçmedi.
2. **Seed gürültüsü etkileri aşıyor; val'de eşik-altı etki test'e taşınmadı.** Baseline val std 0.41 (n=3; AP_S aralığı 2.9 puan), 1000 görüntülük val; H03 val +0.25 → test −0.12 (CI [−0.40, +0.16]); H03'ün val'de 3/3 seed mAP50 +0.74 kazancı test'te −0.02. Test seed std 0.11: seçim ölçüsü olarak val gürültülü; tek-seed ön-eleme (+0.30) gürültü tabanındadır.
3. **Küçük-nesne açığı çözünürlük kaynaklı, eğitim-zamanı müdahaleyle kapanmıyor.** Eğitimsiz test-çözünürlüğü U768: AP_S +4.55 (AP_L −3.41, toplam −2.06; tek ölçüm, deterministik çıkarım). Eğitim müdahaleleri AP_S (n=3 val Δ ort ± sd): H01 -0.30 ± 2.24, H03 +1.34 ± 1.63, H04 -1.06 ± 0.68, H08 +0.31 ± 2.84 (hepsi seed bandı içinde); test'te H03 AP_S +0.81 (p=0.64).
4. **Gerçek-zaman kısıtında post-processing kazancı yok; TTA doğruluk–gecikme ödünleşimi.** ≤1.10× gecikme bandında NMS-IoU (−0.54…−0.10), Soft-NMS (−0.12/−0.59), test çözünürlüğü (−0.31…−5.19) hiçbiri baseline'ı geçmedi (H07 KAZANÇ-A yok). TTA T1: test +0.83 ± 0.22 (CI [+0.50, +1.14]) ama ×2.29 gecikme (21.6 ms, 46 FPS); kazanç yüksek-IoU ve büyük nesnede (mAP75 +1.12, AP_L +1.50), orta nesne kaybıyla (AP_M −1.07, p=0.009), mAP50 değişmedi (+0.10).
5. (Yardımcı) **Kısa ön-uçuş uzun eğitimi öngörmüyor:** H04 5-epoch +0.5 avantaj 100 epoch'ta −0.30'a döndü; ön-uçuş yalnız sağlık kapısı.

## 4. Katkı listesi taslağı (B çıkışı)
1. Ön-kayıtlı, tek-değişkenli, 3 seed ölçülmüş protokolle yedi eğitim-zamanı müdahalesinin VOC/YOLO11n ince ayarında etkisiz/ters olduğunun belgelenmesi (negatif sonuç tablosu).
2. Val-seçici ön-eleme gürültüsünün nicel gösterimi (val std 0.41 vs test 0.11; H03 val→test işaret değişimi).
3. Kutu-tabanlı küçük-nesne copy-paste (H03/H08) doz–yanıt ölçümü: eşik-altı, doygun.
4. Eğitimsiz çıkarım ailelerinin gecikme-bedelli doğruluk haritası; TTA'nın test'te doğrulanmış +0.83 / ×2.29 ödünleşimi.
5. Açık protokol, kod ve ön-kayıt (K5 test planı test'ten önce yazıldı).

## 5. D5 deney kanıt paketi kontrol listesi

| # | Madde | Durum | Dosya |
|---|---|---|---|
| 1 | Deney günlüğü | ✓ | `gunluk/deney-gunlugu.md` (K5 satırı eklendi; bütçe sayacını şef K5 sonrası güncellemeli) |
| 2 | Baseline replikasyon tablosu | ✓ | `sonuclar/replikasyon_map50_95.md`, `replikasyon_map50.md` (YEŞİL) |
| 3 | Ablasyon tablosu | ✓ (tek-değişken ablasyon: TUT zinciri yok) | `sonuclar/ablasyon.csv` |
| 4 | ≥3 seed istatistik | ✓ | `sonuclar/leaderboard-val.md`, `leaderboard-test.md`, `test-analiz.md` (baseline/H03/TTA n=3, val ve test) |
| 5 | Başarısız denemeler tablosu | ✓ | `sonuclar/basarisiz-denemeler.md` |

Notlar: K5 sonrası k5-ek-seed kuyruğu (13 koşum) H01, H02, H04, H05r, H06, H08 için seed 1,2'yi tamamladı; yedi kartın hepsi val'de n=3 (`n3-analiz.md`). Test yalnız baseline, H03 ve TTA için (kilit tek kez açıldı). Eski `leaderboard.md` duman/ön-uçuş satırlarını içeriyordu (ör. h04 0.4273 = 3 epoch duman); aşama-3 filtreli val sürümüyle yenilendi. `leaderboard.py` artık `--asama`, `--kume`, `--haric` seçeneklerini destekler (varsayılan davranış değişmedi); `leaderboard.csv` tüm koşumları içermeye devam eder.

## 6. Sınırlılıklar taslağı
- Tek model ölçeği (YOLO11n, 2.6 M param.) ve tek veri seti (VOC07+12); büyük modellerde ve COCO ön-eğitimsiz rejimde sonuç farklı olabilir.
- Tek GPU (RTX 5050 8 GB); gecikme ölçümü oturuma duyarlı (baseline 8.0 ms val-zamanı vs 9.43 ms test-oturumu); yalnız batch1 FP16 uçtan uca.
- Altı kart K5 sonrası n=3'e tamamlandı (kabul ölçütü karşılanmadı); eşik-altı etkiler (H03) n=3 ile bile zor ayrışır (tespit edilebilir en küçük etki ≈0.5 puan); VFL hiperparametreleri (alpha, gamma, lr) ayarlanmadı; H08 doz hedefinin (0.8–1.2) altında (0.67).
- `difficult` nesneler atıldı (standart VOC işlemesinden farklı olabilir); 101-nokta COCO tipi AP (VOC07 11-nokta mAP50 ile birebir karşılaştırılamaz).
- Val 1000 görüntü (2407 nesne; 143 küçük nesne): AP_S tek başına karar için yetersiz.
- Test seti K3'te yalnız baseline replikasyonu için (`rep_test`), K5'te bir kez açıldı; K5 sonrası arama yapılmadı (D1). TTA yalnız baseline best.pt'leri üzerinde ölçüldü (H03+TTA denenmedi); H03 gecikmesi için baseline ile aynı mimari varsayımı geçerli.
- Çoklu karşılaştırma: ana iddia H03; TTA keşifsel (Holm 0.044); sınıf ve boyut düzeyi p'leri düzeltmesiz keşifsel.
- Literatür çıpası düşük-orta güven (birebir protokol raporu yok); Dwibedi 2017 künyesi makaleye girmeden `atif-dogrulayici` ile doğrulanmalı.


## 6. SEÇİLEN ÇIKIŞ (2026-10-01 22:40)
**B — GEÇEMEDİ, ANLAŞILDI** — *kullanıcı yetkisiyle şef kararı* (kullanıcı 2026-09-29/10-01: "soru sorma, önerilen adımlardan ilerle, onay alma, durma"; önerilen çıkış uygulandı). Makale türü: analiz / negatif-sonuç makalesi. Tez: §2-B tez cümlesi. TTA satırı gerçek-zaman kısıtı DIŞINDA, keşifsel doğruluk–gecikme ödünleşimi olarak raporlanır (iyileştirme iddiası değil).

## 7. n=3 doğrulaması (K5 sonrası, hakem B3) — karar B değişmez


| Kart | Δ ort ± sd | seed Δ (s0/s1/s2) | p tek (çift) | %95 t aralığı | Kabul | Ön-eleme (s0 Δ) | Ön-eleme isabeti |
|---|---|---|---|---|---|---|---|
| H01 | -0.21 ± 0.61 | +0.07 / -0.91 / +0.21 | 0.693 (0.614) | [-1.72, +1.30] | HAYIR | +0.07 → ELE (s0 < +0.30) | DOGRU (n=3'te de kabul yok) |
| H02 | -0.35 ± 0.12 | -0.22 / -0.47 / -0.35 | 0.980 (0.041) | [-0.66, -0.04] | HAYIR | -0.22 → ELE (s0 < +0.30) | DOGRU (n=3'te de kabul yok) |
| H03 | +0.25 ± 0.24 | +0.37 / -0.03 / +0.41 | 0.107 (0.214) | [-0.35, +0.85] | HAYIR | +0.37 → seed 1,2'ye cik | cikis yanlis alarm (n=3 kabul yok) |
| H04 | -0.85 ± 0.79 | -0.30 / -1.76 / -0.49 | 0.898 (0.204) | [-2.82, +1.12] | HAYIR | -0.30 → ELE (s0 < +0.30) | DOGRU (n=3'te de kabul yok) |
| H05 | -2.87 ± 0.33 | -2.88 / -3.19 / -2.53 | 0.998 (0.004) | [-3.68, -2.05] | HAYIR | -2.88 → ELE (s0 < +0.30) | DOGRU (n=3'te de kabul yok) |
| H06 | +0.04 ± 0.56 | -0.42 / -0.12 / +0.66 | 0.456 (0.912) | [-1.35, +1.43] | HAYIR | -0.42 → ELE (s0 < +0.30) | DOGRU (n=3'te de kabul yok) |
| H08 | -0.08 ± 0.40 | +0.22 / -0.53 / +0.06 | 0.623 (0.754) | [-1.07, +0.90] | HAYIR | +0.22 → ELE (s0 < +0.30) | DOGRU (n=3'te de kabul yok) |


Kaynak: `n3-analiz.md/json`. Kabul ölçütü karşılayan eğitim kartı 0/7. Seçilen çıkış B ve tez cümlesi (§2-B) bu tabloyla uyumludur: yedi eğitim-zamanı müdahalesinin hiçbiri val'de 3 seed'lik ön-kayıtlı kabul ölçütünü karşılamadı; teste giren tek eğitim kartı H03 testte de geçmedi. Ön-eleme isabeti: s0 kapısında elenen 6 kartın 6'sı n=3'te de kabul edilmedi (tersine dönen yok); kapıyı geçen tek kart H03 yanlış alarmdı. H02 (−0.35, p çift=0.041) ve H05 (−2.87, p çift=0.004) baseline'ın anlamlı biçimde altında (negatif etki, düzeltmesiz). n=3 t-testi ≈0.5 puandan küçük etkiyi ayıramaz (sınırlılık).
