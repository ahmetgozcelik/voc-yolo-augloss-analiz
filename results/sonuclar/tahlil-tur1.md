# Tur-1 tahlili (K4c) — H01, H02, H03

*Mod: tur tahlili (deney-tahlilci). Yalnız VAL (1000 görüntü, 2407 nesne). `rep_test_*`, `test_kilitli.json`, `rep_test_preds.json`, `rep_test/` okunmadı; `metrics.json`'daki `rep_*` anahtarları yüklenirken atıldı.*
*Kaynaklar: `sonuclar/leaderboard.md/csv` (leaderboard.py, bu oturumda yeniden üretildi), `sonuclar/hata-analizi*.md/.json` (`scripts/analiz/hata_analizi.py`; bu oturumda H01 ve H03 s0/s1/s2 için koşturuldu, baseline s0/s1/s2 ve H02 hazırdı), `hipotezler/H01-H03.md`, `varyantlar/h03_boxcp/` (README, `runs/*h03*/boxcp_sayac/*.json`).*
*Not: `leaderboard.md` tur-2 aşama-1/2 (duman/ön-uçuş, n=1, 3-5 epoch) satırlarını da içerir (h04/h05/h06, baseline_cm0); bunlar tur-1 kararına girmez ve aşağıdaki tabloda yoktur. Hata-türü sayıları pycocotools eşleştirmesidir; birincil metrik Ultralytics `val_map50_95`'tir (ikisi arasındaki fark ≤0.1 puan).*

## 1. Leaderboard (val, puan = x100; ort ± std, n = seed sayısı)

| varyant | n | mAP50-95 | mAP50 | AP_S | AP_M | AP_L | FPS (b1 FP16) | eğitim (sa) | Δ mAP50-95 vs baseline (eşleştirilmiş) | durum |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 63.07 ± 0.41 | 81.21 ± 0.13 | 22.11 ± 1.44 | 44.28 ± 0.90 | 70.46 ± 0.58 | 123.7 ± 1.1 | 2.9 | – | ebeveyn |
| H03 h03_boxcp | 3 | 63.32 ± 0.17 | 81.95 ± 0.08 | 23.45 ± 1.13 | 45.84 ± 1.97 | 70.80 ± 0.40 | 125.1 ± 0.3 | 2.9 | **+0.25 ± 0.24** (çift taraflı p=0.214; tek yönlü 0.107) | n=3, ÖLDÜR (eşik +0.30) |
| H01 h01_multiscale25 | 1 | 62.96 | 81.59 | 20.81 | 45.88 | 70.31 | 115.8 | 3.3 | +0.07 (s0'a karşı) / -0.11 (3 seed ort.'a karşı) | YETERSİZ SEED (n<3); ön-kayıtlı öldürme kuralı (a) ile ÖLDÜR |
| H02 h02_innerciou | 1 | 62.67 | 81.17 | 25.81 | 43.30 | 70.23 | 124.4 | 3.0 | -0.22 (s0'a karşı) / -0.40 (3 seed ort.'a karşı) | YETERSİZ SEED (n<3); kural (a) ile ÖLDÜR |

Gözlemler: (i) Üç baseline seed'inin AP_S aralığı 20.7-23.6 (2.9 puan), AP_M 43.4-45.2 (1.8 puan); n=1 satırlardaki tüm S/M farkları (H01 AP_S -2.8, H02 AP_S +2.2 s0'a karşı) bu bant içindedir. 143 S nesneli val'de AP_S tek başına karar ölçütü olamaz. (ii) H01'in FPS'i 115.8 (baseline -6.4%): çıkarım grafiği değişmediğinden ölçüm gürültüsü/GPU paylaşımı; kartın %10 alarm eşiğinin altında, ama eğitim süresi +12% (11735 sn vs 10445 sn). (iii) Karar H01/H02 için baseline'ın s0 mı yoksa 3-seed ortalaması mı kullanıldığına duyarsızdır: her iki ebeveynle Δ < +0.30.

## 2. Kart başına S / D / I

### H01 — çok-ölçekli eğitim (multi_scale=0.25)

**S (sonuç).** Beklenti: mAP50-95 +0.0…+0.5 (merkez +0.2), AP_S +0…+1.5, AP_M +0…+1.0. Ölçülen (s0): mAP50-95 +0.07 (62.96 vs 62.89 Ultralytics), mAP50 +0.46, AP_S -2.78, AP_M +2.48, AP_L +0.15. Kural (a) tetiklendi (+0.07 < +0.30) → seed 1,2 koşulmadı.

**D (doğrulama).** (1) Etki gerçekten devrede: kartın uygulama notu `varyant_etki_sayaci` ve 480-800 px batch boyutlarını kaydetti. (2) Ölçek kırılımı (işletim noktası skor>=0.25, hata_analizi): S bulunan GT 54/143 (%39.2; baseline s0 %38.5, s1 %35.7, s2 %40.6), yani S recall'u baseline bandında; S'deki Miss 87 vs 88. AP_S'in düşmesi bu yüzden bulunamayan nesnelerden değil sıralama/kutu kalitesinden gelmiş olmalı (AP50_S 43.4 → 36.6; yalnız n=1, 143 nesne). M'de TP 489 vs 487, Miss 212 vs 209 (aynı), Bkg 122 vs 132. AP_M +2.5 (3-seed ortalamaya göre +1.6, baseline std 0.9) ≈ 1.8 sd: tek koşumda ne doğrulanmış ne çürütülmüş. AP_L +0.15, değişmedi. (3) Hata türü kırılımı (op noktası, s0 baseline → H01): TP 1909→1914, Dupe 88→97, Cls 97→90, Loc 160→163, Both 72→77, Bkg 270→274, Miss 498→493; hepsi üç-seed baseline aralığı içinde (ör. Bkg 270-305, Miss 471-505). Loc oracle 5.69 → 5.32, Kutu_kusursuz tavanı 18.0 → 18.5: lokalizasyon yapısı değişmedi. (4) Kırmızı bayrak yok: Δ eşiğin (+1.0) çok altında, ebeveynle bit-bit aynı değil, çıkarım FPS farkı <%10.

**I (yorum).** Mekanizma beklentisi (S/M ölçek-değişmezliği) S için gerçekleşmedi; kazanç orta boya kaydı ama toplam mAP'e yansımadı. Batch-düzeyi ±%25 çözünürlük çeşitlemesi, zaten mosaic+`scale=0.5` ile görüntü-içi ölçek jitter'ı alan ön-eğitimli bir ince ayarda ikinci bir ölçek ekseni olarak marjinal. Sonuç beklentiye (merkez +0.2) göre hafif düşük; "beklentinin tersi" bayrağı (ters yönde, açıklanamayan) yok çünkü Δ ≈ 0 ve gürültü bandı içinde. Karar: **ÖLDÜR (katılıyorum)**; tek-seed olduğu için makalede "n=1, ön-eleme" olarak raporlanmalı, kesin "etkisiz" iddiası yapılmamalı.

### H02 — Inner-CIoU (r=0.75)

**S.** Beklenti: mAP50-95 +0.0…+0.5, AP75 +0.3…+1.0, AP_S ±0.5. Ölçülen (s0): mAP50-95 -0.22 (62.67 vs 62.89), mAP50 +0.03, AP75 (pycocotools) 67.55 vs 68.17 (-0.62; 3-seed baseline ort. 68.37 ± 0.38 → -0.82), AP_S +2.22, AP_M -0.10, AP_L +0.07.

**D.** (1) AP75 hedefi tuttu mu: hayır; AP75 baseline'ın üç seed'inin (68.17 / 68.82 / 68.13) da altında. (2) IoU dağılımı (TP, tüm güven): IoU>=0.85 payı baseline s0 %61.0, H02 %61.3; IoU<0.75 payı %22.6 vs %22.5; IoU 0.95-1.00 payı %24.6 vs %24.2; işletim noktasında IoU<0.75 payı %15.7 vs %15.6; TP ort. IoU 0.8460 vs 0.8467 (baseline seed'leri 0.8453-0.8460); skor-IoU Spearman 0.622 vs 0.633 (baseline 0.616-0.640). Yani yüksek-IoU bölgesinde hiç kayma yok: Inner-CIoU'nun hedeflediği "son rötuş" iyileşmesi ölçülmedi. (3) Hata türü (op noktası): TP 1909→1889, Miss 498→518, Cls 97→100, Loc 160→141, Bkg 270→265, Dupe 88→80; Loc sayısı baseline aralığının (146-162) altında ama bunu kazanç yapan hiçbir AP75 artışı yok; asıl fark TP'nin 20 düşmesi (recall_op %79.3 → %78.5; baseline 79.0-80.4). Loc oracle ΔAP50-95 5.69 → 5.08 (baseline 5.18-5.69; sınırda). (4) Sınıf kırılımı gürültülü: diningtable -5.9 (49 val nesnesi), tvmonitor -3.8, sofa +3.2; sınıf başı baseline std'si 0.2-4.7 puan olduğundan yorumlanamaz. (5) Birim testleri (r=1.0'da özdeşlik, kutu-gradyan normu x1.165) yamanın doğru çalıştığını gösteriyor; arıza değil, etkisizlik. Kırmızı bayrak yok.

**I.** Mekanizma beklentisi (IoU->1 civarında daha dik gradyan) hata analizinde ölçülebilir iz bırakmadı. Kutu-kaybı geometrisi lokalizasyon açığını (Loc-oracle +5.7 puan, AP50-AP75 farkı 13 puan) kapatmıyor; açık atama/hizalama ya da DFL tarafında olabilir (hipotez, ölçülmemiş). Literatürdeki +0.6-0.7 (YOLOv7-tiny, taban 37.6) bu rejimde (taban 63) tekrarlanmadı. Sonuç beklenti aralığının (+0.0…+0.5) hafif altı; yön tersine dönmesi gürültü bandında. Karar: **ÖLDÜR (katılıyorum)**; n=1.

### H03 — kutu-tabanlı küçük-nesne copy-paste (p=0.5)

**S.** Beklenti: mAP50-95 +0.0…+0.4 (merkez +0.15), AP_S +0…+1.7; kapı +0.30. Ölçülen, 3 seed eşleştirilmiş (Ultralytics): mAP50-95 +0.37 / -0.03 / +0.41 → **+0.25 ± 0.24**; mAP50 +0.74 ± 0.16 (3/3 pozitif, tek yönlü p=0.007, ikincil/çoklu karşılaştırmasız); AP_S +1.34 ± 1.63; AP_M +1.55 ± 2.85; AP_L +0.34 ± 0.24 (p=0.067). Ön-kayıtlı kural (3 seed Δ >= +0.30) ve tur-2'de donan sıkı kural (Δ >= +0.40, p<0.10, >=2 seed Δ>0) ikisinde de geçmedi → ÖLDÜR.

**D.** Önemli not: ölçülen Δ (+0.25) kartın beklenti aralığının içinde ve merkezinin (+0.15) üzerinde. Yani beklenti yanlış çıkmadı; ön-kayıtlı öldürme eşiği (+0.30), kartın kendi beklenen merkezinin iki katıydı ve gürültü tabanıyla (seed sd 0.4) aynı büyüklükte. Tedavi gerçekten uygulandı (boxcp sayaçları: seed başına ~199 bin yapıştırma, ~155 bin etkilenen örnek).

*(a) Hata türü kırılımı (op noktası, skor>=0.25; baseline 3 seed ort. → H03 3 seed ort.).* TP 1915.7 → 1918.3; Miss 491.3 → 488.7; Bkg 285.3 → 269.3; Loc 156.0 → 151.3; Cls 95.7 → 89.0; Both 74.0 → 69.3; Dupe 94.7 → 87.0. Hepsi baseline seed aralıkları içinde; yön hepsinde (TP hariç hata azalması) hafifçe olumlu. Kartın endişesi (yapıştırma sahte pozitifi artırır) gerçekleşmedi: tüm-güven Bkg 21633 → 21436, op noktası Bkg 285 → 269. Boyuta göre (baseline ort. → H03 ort.): S: TP 52.7 → 52.7, Miss 88.3 → 88.0, Bkg 55.0 → 51.3; M: TP 494.3 → 486.7, Miss 204.3 → 205.7, Bkg 140.7 → 124.7; L: TP 1368.7 → 1379.0, Miss 198.7 → 195.0, Loc 84.3 → 80.0, Cls 76.3 → 69.3, Bkg 89.7 → 93.3. Yani küçük nesne bulma (S Miss) hiç değişmedi; değişen (zayıf ama tutarlı) şey L'de TP artışı ve M'de FP azalması.

*(b) AP75 ve lokalizasyon.* AP75 (pycocotools) eşleştirilmiş Δ = -0.39 / -0.45 / -0.12 (ort. -0.32; 3/3 negatif). AP50 +0.73 iken AP75 -0.32: kazanç gevşek IoU eşiklerinden geliyor, sıkı eşiklerde yok. TP ort. IoU 0.8461 / 0.8463 / 0.8435 (baseline 0.8453-0.8460), IoU<0.75 payı değişmedi. Bu keşif niteliğinde bir bulgu (n=3, çoklu karşılaştırma); doz tartışmasında risk olarak kullanılır (bkz. §3).

*(c) Boyut başına eşleştirilmiş Δ (pycocotools AP50-95, puan; s0 / s1 / s2 → ort.):*

| boyut | Δ s0 / s1 / s2 | ort. ± sd | tek yönlü p | ΔAP50 ort. | tutarlılık |
|---|---|---|---|---|---|
| S | -0.47 / +1.80 / +2.68 | +1.34 ± 1.62 | 0.145 | +1.53 | 2/3 pozitif; s0'da AP50 -2.98 |
| M | +4.60 / -1.06 / +1.13 | +1.56 ± 2.85 | 0.222 | +2.46 | 2/3; s0'ın +4.6'sı tekrarlanmadı |
| L | +0.58 / +0.09 / +0.34 | +0.33 ± 0.24 | 0.070 | +0.86 | **3/3 pozitif, en dar dağılım** |
| tümü | +0.35 / -0.01 / +0.40 | +0.24 ± 0.23 | 0.101 | +0.73 | 2/3 |

Tek tutarlı (3/3) boyut grubu L; hedeflenen S'de ortalama +1.3 ama sd 1.6, ve işletim noktası S recall Δ = -0.7 / +4.2 / -2.8 puan (ort. +0.2). Mekanizma (küçük nesne pozitif örneği artar) S için kanıtlanamadı; tutarlı gözlem küçük-nesneye özgü değil, L ve mAP50 üzerinde genel bir küçük kazanç.

*(d) Sınıf başına eşleştirilmiş Δ (AP50-95, puan; hata_analizi §1 tablosundan, 3 ondalık çözünürlük, 20 sınıf). Sıra: ort. Δ'ya göre. "Poz" = pozitif seed sayısı; "z" = ort. Δ / baseline'ın 3 seed'lik sınıf std'si; "aday %" = train'de yapıştırma adayı (mosaic ölçeğinde sqrt(alan)<=48 px) nesneler içindeki sınıf payı (§3'te hesabı).*

| sınıf | Δ s0 / s1 / s2 | ort. Δ | ΔAP50 ort. | poz | z | aday % |
|---|---|---|---|---|---|---|
| sofa | -0.1 / +2.8 / +3.8 | +2.17 | +3.33 | 2/3 | 1.0 | 0 |
| pottedplant | +4.8 / +0.7 / +0.2 | +1.90 | +2.13 | **3/3** | 0.95 | 6.8 |
| dog | +1.4 / +1.6 / +1.0 | +1.33 | +0.23 | **3/3** | 1.0 | 0.5 |
| bottle | +1.6 / -0.5 / +2.5 | +1.20 | +1.63 | 2/3 | 0.9 | 13.8 |
| sheep | +4.6 / -5.7 / +4.7 | +1.20 | +0.27 | 2/3 | 0.3 | 4.6 |
| chair | -0.7 / +2.3 / +1.0 | +0.87 | +2.03 | 2/3 | 0.7 | 3.7 |
| boat | +0.3 / +5.6 / -3.6 | +0.77 | +2.30 | 2/3 | 0.2 | 4.2 |
| bicycle | +1.6 / +1.2 / -0.5 | +0.77 | +1.47 | 2/3 | 0.5 | 1.1 |
| cow, diningtable, motorbike | ort. +0.57 / +0.50 / +0.50 | | | 2/3, 1/3, 2/3 | <1 | 1.9 / 0.3 / 0.7 |
| person | -0.5 / +0.5 / +0.4 | +0.13 | -0.30 | 2/3 | 0.6 | **36.4** |
| car | +0.5 / -0.7 / +0.1 | -0.03 | -0.23 | 2/3 | 0 | 16.8 |
| aeroplane, bus, train | -0.07 / -0.27 / -0.43 | | | 1/3, 2/3, 1/3 | | 2.7 / 0.7 / 0.4 |
| tvmonitor | -2.4 / -1.8 / +1.4 | -0.93 | +1.20 | 1/3 | 0.9 | 1.2 |
| horse | -2.6 / -2.5 / +1.3 | -1.27 | -1.70 | 1/3 | 0.9 | 0.4 |
| bird | -0.4 / -3.5 / -1.5 | **-1.80** | -0.90 | **0/3** | 1.7 | 3.7 |
| cat | -4.2 / -0.2 / -2.3 | **-2.23** | -1.03 | **0/3** | 7.4 | 0.1 |

Okuma: 3 seed'de de pozitif yalnız pottedplant (+1.9, esas olarak s0'da +4.8; s1/s2'de +0.7/+0.2) ve dog (+1.3, en tutarlı ama aday payı %0.5); 3 seed'de de negatif bird (-1.8) ve cat (-2.2; baseline sd'sine göre z=-7.4, ama cat'in aday payı %0.1 olduğundan nedensel açıklaması yok). Yapıştırılan sınıf dağılımı ile Δ arasında anlamlı ilişki yok: person (%36) +0.13, car (%17) -0.03, bottle (%14) +1.2; dog/sofa (aday ~%0) +1.3/+2.2. Spearman(aday payı, ort. Δ) = +0.21 (p=0.36, n=20 sınıf). Bu yüzden zayıf sınıflara özgü kazanç iddia edilemez (kartın "zayıf sınıflara özgü bir kazanç iddia edilmez" ön şartıyla uyumlu). Sınıf başı değerler val'de 49-177 nesne üzerinden hesaplandığından tek tek sınıf bulgusu keşif düzeyindedir; cat/bird negatifi çoklu karşılaştırma (20 sınıf) nedeniyle gürültü olabilir ama 3/3 işaret tutarlılığı tur-3 yüksek doz koşusunda yeniden izlenmeli (artarsa yapıştırma-kaynaklı bir yan etki göstergesi).

**I (yorum).** (1) Sonuç beklenti aralığındadır (+0.25 vs +0.0…+0.4); "görülmemiş kadar iyi" bayrağı yok. (2) En sağlam bulgu mAP50'de +0.74 ± 0.16 (3/3), ikincisi AP_L'de +0.34 ± 0.24; ikisi de hedeflenen küçük nesne mekanizmasının değil, genel bir "örnek sayısı/bağlam çeşitliliği" etkisinin imzası. (3) AP75 3/3 negatif: kazanç gevşek-IoU. (4) Seed0'daki AP_M +4.6 parlak görünümü bir kez daha seed0'ın seçim yanlılığını gösterdi (diğer seed'lerde -1.06 / +1.12). Karar: **ÖLDÜR (birincil kural), "eşik-altı ama tutarlı mAP50 etkisi" notuyla; katılıyorum.** Kural sonradan esnetilmez. Kırmızı bayrak: 0; ödünleşim notu: AP75 -0.32.

## 3. H03 "doz" analizi

**Ölçülen tedavi yoğunluğu (varyantlar/h03_boxcp sayaçları ve train etiketleri).**
- Seed başına 1.3997 M mosaic çağrısı (90 epoch x 15551; son 10 epoch kapalı), 198 949 / 198 775 / 199 442 yapıştırılan nesne (seed'ler arası fark <%0.4: tedavi dozu seed'ler arasında değişmiyor).
- 0.142 yapıştırılan nesne / mosaic görüntüsü; yapıştırma uygulanan görüntü oranı %11.1 (155 039 / 1 399 700); p=0.5 olduğundan uygunluk oranı ≈ %22 (görüntülerin yarısında denenir, yarısında uygun aday yok/yer yok); uygulanan görüntü başına 1.28 nesne. (README'deki "~0.12" 20 batchlik CPU testinden; tam koşumda 0.142.)
- Train'de aday havuzu: 37 651 nesnenin yalnız 3 312'si (%8.8) mosaic ölçeğinde (uzun kenar 640) sqrt(alan) <= 48 px; bu nesnelerin %36.4'ü person, %16.8 car, %13.8 bottle, %6.8 pottedplant (IoA süzgeci uygulanmamış üst sınır; sınıf dağılımı için kendi hesabım, `runs/*/boxcp_sayac` sınıf sayacı tutmuyor).
- Bağıl doz: mosaic tuvalinde ort. ~9.7 nesne (4 görüntü x 2.42) varken +0.142 nesne ≈ **+%1.5 toplam nesne**; küçük nesne havuzuna göre (tuvalde ~0.85 aday) ≈ **+%17 küçük-nesne maruziyeti**. Yani "küçük nesneyi artırma" amacına göre bile çok hafif bir tedavi.

**Doz ile etki büyüklüğü.** Tek bir doz ölçüldü (0 ve 0.142); seed'ler arasında doz farkı olmadığından doz-yanıt eğrisi çıkarılamaz. Etki (+0.25 mAP50-95, +0.74 mAP50) tek noktadan "doz düşüklüğüyle tutarlı" denebilir ama "dozu artırınca artar" ancak yeni doz koşulursa sınanır; şimdiki veri bunu desteklemiyor da çürütmüyor. Zayıflatan iki gözlem: (i) hedeflenen boyutta (S) etki en gürültülü, (ii) tutarlı etki L ve mAP50'de, yani mekanizma beklenenden farklı olabilir. Güç hesabı: eşleştirilmiş Δ sd'si 0.24 iken +0.25'i tek yönlü p<0.10 ile yakalamak ~6 seed ister; n=3 ile yakalanabilecek etki ≈ +0.5.

**Yüksek doz sürümünün gerekçesi.** Etki yönü pozitif, kayıp birincil eşiğin 0.05 puan altında kaldı ve tedavi %1.5 nesne dozundaydı; doz ~5-6 kat artırılırsa (kendi kaba tahminim, ölçülmeli): `p=1.0` (yapıştırma denemesi 2x), aday eşiği `sqrt(alan)<=96` (S+M; havuz 3 312 → 10 196 nesne, aday içeren görüntü payı tek görüntüde %11 → %28, tuvalde ~%38 → ~%72 tahmini), 1-3 kopya (aynı). Bu aynı zamanda 3 noktalı doz-yanıt (0, 0.14, ~0.8 nesne/tuval) verir ve "mekanizma küçük-nesne mi, genel çeşitlilik mi?" sorusunu ayırır: küçük-nesne mekanizması ise AP_S artmalı; genel çeşitlilik ise mAP50/AP_L artmalı.
**Riskler:** (1) Maskesiz (kutu-içi) yama arka plan parçası taşır; Ghiasi ve ark. 2021 (arXiv:2012.07177, literatur-cipa §4) örnek maskeleriyle yapıştırır ve kazancı büyük model/uzun eğitim/LSJ ile alır; VOC'ta kutu-yaması farklı bir tedavi (kartta zaten not). Yapıştırma dozu arttıkça bu sınır-artefaktının öğrenilmesi AP75 düşüşünü (şimdiden -0.32, 3/3) büyütebilir; val'de yapıştırma olmadığından dağılım kayması riski. (2) Dwibedi ve ark. 2017 (Cut, Paste and Learn, arXiv:1708.01642; **literatur-cipa.md'de yok, bu oturumda doğrulanmadı; makaleye girmeden künye ve sayfa doğrulaması gerekir**) kesip yapıştırmada sınır/harmanlama artefaktlarının dedektörü yanıltabildiğini ve artefaktlara karşı dayanıklılık için çeşitli harmanlama gerektiğini bildirir; bizim sabit 3 px yumuşatmamız tek harmanlama türü. (3) Sınıf yanlılığı: aday havuzu person/car/bottle ağırlıklı; doz artınca bu sınıfların dengesi bozulur, `cat`/`bird` negatifi büyüyebilir. (4) Eşik <=96 px nesneleri (orta boy) kopyalamak mosaic+scale=0.5 jitter'ıyla 48-144 px'e çıkarır; aynı görüntüde çakışmasız yer bulma zorlaşır (`yer_bulunamadi` şimdi %1.6/aday, artar). (5) Maliyet düşük: eğitim süresi +%1.4 (10587 vs 10445 sn).
**Not:** Yüksek doz yeni bir ön-kayıtlı kart olmalı (H03b veya H08), eşiği önceden sabit; AP75'te Δ < -0.5 İNCELE tetikleyicisi eklenmeli.

## 4. "Başarısız denemelerin analizi" için makale paragrafı (taslak)

> İlk turda üç tek-değişkenli müdahale, ön-kayıtlı kurallarla (seed 0'da Δ >= +0.30 mAP50-95 puan şartı) değerlendirildi; hiçbiri karar eşiğini geçmedi. Batch düzeyinde çok-ölçekli eğitim (multi_scale=0.25) seed 0'da mAP50-95'i 62.89'dan 62.96'ya (+0.07) taşıdı; AP_S 23.6'dan 20.8'e düştü, ancak üç baseline seed'inin AP_S aralığı 20.7-23.6 olduğundan bu fark gürültü bandı içindedir ve orta nesne AP'sindeki +2.5 puanlık artış toplam ölçüte yansımadı. CIoU yerine Inner-CIoU (r=0.75) kaybı mAP50-95'i 0.22 puan düşürdü (62.67) ve hedeflenen AP75'i de yükseltmedi (68.17 -> 67.55); doğru-pozitif IoU dağılımı da değişmedi (IoU >= 0.85 payı %61.0 ve %61.3). Bu iki müdahale tek seed'de elendi (n=1), dolayısıyla "etkisizdir" yerine "bu rejimde Δ >= +0.30 göstermedi" ifadesi kullanılmıştır. Kutu-tabanlı küçük nesne yapıştırma (copy-paste) üç seed'de (eşleştirilmiş) mAP50-95'i ortalama +0.25 ± 0.24 puan (tek yönlü p=0.107) artırdı ve ön-kayıtlı +0.30 eşiğinin altında kaldığı için elendi; mAP50 +0.74 ± 0.16 (üç seed'de de pozitif) ve AP_L +0.34 ± 0.24 artarken AP75 0.32 puan azaldı, küçük nesnelerde (AP_S +1.34 ± 1.63) tutarlı bir kazanç gözlenmedi. Yapıştırma dozu düşüktü (görüntü başına ortalama 0.14 nesne, nesne sayısında yaklaşık %1.5 artış); etkinin eşik altında kalması bu dozla uyumludur, fakat tek doz ölçüldüğü için doz-yanıt ilişkisi sınanmamıştır. Sonuçlar, doygun (mAP50-95 ≈ 63) ve ön-eğitimli bir YOLO11n tabanında 0.3-0.5 puanlık seed gürültüsünün altında kalan bu tür artırımların ancak daha çok seed ile ayırt edilebileceğini göstermektedir.

*(Yazım notu: AP_S/AP_M/AP_L'deki ve sınıf başına kırılımdaki tüm farklar keşif düzeyindedir; çoklu karşılaştırma düzeltmesi yapılmamıştır. Taslakta Dwibedi veya Ghiasi atfı yok; atıflar Bulgular dışına, Tartışma'ya konacaksa künye doğrulanmalı.)*

## 5. Tur-3 önerisi

**Karar özeti:** TUT yok (H01 ÖLDÜR, H02 ÖLDÜR, H03 ÖLDÜR); ebeveyn `baseline` kalır (3 seed: 63.07 ± 0.41). Kırmızı bayrak: 0. Ödünleşim notu: H03'te AP75 -0.32. Hata kütlesi hâlâ: Loc oracle +5.7 (en büyük tek tür), S recall_op %38.5, FP tarafında Bkg (op noktasında 270-305 tahmin) ve Cls.

1. **H03b — yüksek doz kutu copy-paste (öncelikli).** Tek değişken: yapıştırma dozu (`p=1.0`, aday eşiği sqrt(alan)<=96, 1-3 kopya; diğer her şey H03 ile aynı). Gerekçe §3. İzlenecek: AP75 (İNCELE tetikleyicisi Δ < -0.5), AP_S ve AP_L ayrı (mekanizma ayrımı), sınıf Δ'ları (cat, bird), Bkg. Ön-kayıt: seed0 kapısı +0.30, TUT = 3 seed ort. Δ >= +0.40, p<0.10, >=2 seed Δ>0 (tur-2'de dondurulan kural). Bütçe ≈ 3 koşum x 3 sa. Yan seçenek (bütçe izin verirse): doz A (`p=1.0`, <=48 px) ayrı kol → doz-yanıt/eşik ayrımı.
2. **H03c — sınıf-dengeli yapıştırma** (yalnız H03b TUT alırsa veya doz kolu belirsiz kalırsa): aday seçimi sınıf frekansının tersiyle ağırlıklı (person %36 baskınlığını kır; zayıf pottedplant/bottle/chair/boat'a pay); mekanizma: zayıf sınıfların pozitif örnek sayısı. H03'te zayıf sınıf eğilimi var (pottedplant 3/3, bottle 2/3, chair 2/3) ama sınıf payı-Δ ilişkisi yok (Spearman +0.21), bu yüzden bu bir keşif kartı, düşük öncelik.
3. **Lokalizasyon/AP75 yönü, tur-2 sonuçlarına göre.** H02 ve H03 (AP75 -0.32) kutu-kaybı geometrisi ve yapıştırmanın AP75'i artırmadığını gösterdi; Loc-oracle +5.7 açık duruyor. H06 (box 7.5->10) ve H05 (VFL, cls-IoU hizası) tur-2'de koşuyor; bunların ön-eleme sonuçları gelmeden yeni bir lokalizasyon kartı açmayın. TUT alanı çıkarsa onunla H03b birleşimi yalnız her ikisi ayrı TUT alırsa, birleşim kartı olarak (H02 ablasyon planındaki ilke). Hiçbiri TUT almazsa kalan bütçe H03/H07'nin ek seed'lerine (n=6) gitsin: eşleştirilmiş sd 0.24'te +0.25 etkisini yakalamak ~6 seed ister.

**Bütçe görüşü:** Tur-1'de 3 kartın toplamı ≈ 3.25 + 2.97 + 3 x 2.95 ≈ 15 sa GPU; tur-2 (~11 sa) sürüyor. Durma kuralına yaklaşılmadı. Kalan bütçenin en verimli kullanımı: tek kart (H03b) için tam 3 seed ve baseline'a 3 ek seed eklenerek (s3-s5) paired-t gücünü artırmak (baseline std 0.41 şimdiki en büyük belirsizlik kaynağı).
