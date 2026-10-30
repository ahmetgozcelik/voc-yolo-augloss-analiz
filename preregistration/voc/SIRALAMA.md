# Hipotez sıralaması — TUR 1 (K4a)

*Yazım: 2026-09-30 · yalnız val bilgisi (hata-analizi.md, literatur-cipa.md); test bilgisine bakılmadı. Ebeveyn:
`baseline` `20260930-072103_baseline_s0` (val map50_95 0.6289). Protokol: aday seed 0 ile elenir; Δ>=+0.3 puan
ise seed 1,2. Gürültü tabanı ≈0.3–0.5 puan → kartlardaki Δ>=+0.3 kapısı gürültü tabanındadır; kapıyı
geçmek "kanıt" değil "seed'e çıkma hakkıdır".*

## Tur 1 kartları

| Sıra | Kart | Aile | Değişken (tek) | Hedeflenen ölçülmüş zayıflık | Beklenen Δ (val map50_95) | P(Δ>=+0.3, seed0) öznel | Maliyet | Skor = kazanç x P / maliyet |
|---|---|---|---|---|---|---|---|---|
| 1 | **H01** | A3 (aug, ölçek) | `multi_scale=0.25` | Z1: AP_S 0.236, AP_M 0.434; S <e-posta> %38.5 | +0.0…+0.5 (merkez +0.2) | %30 | 3.0–3.4 sa, kod yok | 0.2 x 0.30 / 3.3 ≈ 0.018 |
| 2 | **H02** | A5 (loss) | Inner-CIoU r=0.75 | Z2: AP75 0.682, Loc oracle +5.69, AP50→AP75 12.7 puan | +0.0…+0.5 (merkez +0.2) | %30 | 3.0 sa + ≈30 satır kod (≈0.3 sa-eşdeğeri) | 0.2 x 0.30 / 3.3 ≈ 0.018 |
| 3 | **H03** | A3 (aug, kutu copy-paste) | kutu-tabanlı küçük-nesne yapıştırma p=0.5 | Z1: S nesnelerin %61.5'i işletim noktasında kaçıyor (88/143); Z3 S+M ağırlıklı zayıf sınıflar | +0.0…+0.4 (merkez +0.15); AP_S +0…+1.7 puan | %20 | 3.0–3.4 sa + ≈80 satır kod + test (≈3 sa-eşdeğeri) | 0.15 x 0.20 / 6.3 ≈ 0.005 |

(Olasılıklar öznel tahmindir ve ön-kayıt olarak burada donar; sonradan "haklı çıktı/çıkmadı" diye
güncellenmez. Kazanç birimi: puan.)

### Neden bu sıra
1. **H01 birinci:** sıfır kod, sıfır uygulama riski (tek `--override`); en büyük ölçülmüş boşluğa (küçük/orta nesne) doğrudan
   dokunan en ucuz kol. İlk sonucu en erken o verir; başarısızsa Z1 ölçek yönü için yön bilgisi (yedek B1).
2. **H02 ikinci:** skoru H01 ile eşit; H01'in tek avantajı "kodsuz" olması. H02, hata analizinin
   en büyük AP50-95 oracle'ına (Loc +5.69) ve loss ailesinin tek VOC-ölçekli yayınlanmış kaynağına (Inner-IoU Tablo I, +0.61…+0.74,
   düşük tabanlı) dayanır; makale katkısı (loss) buradan gelebilir. Kod küçük ve GPU'suz birim testli.
3. **H03 üçüncü:** beklenen kazanç ve olasılık en düşük, maliyet en yüksek (yeni dönüşüm + test), ama **mekanik olarak
   tek "yeni kod" makale katkısı adayı**: Ultralytics `copy_paste` VOC'ta etkisizdir (literatur-cipa §4 tuzağı),
   katalog stratejisi #2'nin gerçek karşılığı bu. Kodu H01/H02'nin GPU koşumları (≈6.5 sa) sürerken yazılır, yani
   GPU beklemez; sıra yalnız hangi koşumun önce başlayacağını belirler, yazma gecikmesi yaratmaz.

### Hiçbir kart birleşik değildir (M2). Kapanmış kart yok (henüz) → "H<NN> öldü çünkü" referansı yok.
Hiçbir kart protokol/veri/metrik değiştirmez → **KULLANICI ONAYI GEREKEN KART YOK** (üçü de otomatik koşabilir).

## Bütçe (kalan ≈65 GPU sa; 1 tam koşum ≈3.0 sa; H01/H03 ≈3.3 sa)

| Kalem | Sa |
|---|---|
| Tur 1, yalnız seed0 (H01+H02+H03) | ≈ 9.7 |
| Tur 1, seed1,2'ye çıkan kart başına (beklenen ≈1 kart) | ≈ +6.3 |
| Tur 1 beklenen toplam | ≈ 16 |
| Tur 1 en kötü durum (3 kart de çıkarsa) | ≈ 29 |
| K5 payı (kazanan ek seed + ablasyon + test kilidi, 6–8 koşum) | ≈ 18–24 (rezerv: 20) |
| Turlar 2–5 için kalan havuz (beklenen tur-1 sonrası) | ≈ 65 − 16 − 20 = 29 sa ≈ 9 koşum |

Öneri (şefin kararı, protokolü değiştirmez): tur 1'de seed1,2'ye en çok 2 kart çıkarılsın (en yüksek Δ'lar);
üçüncüsü kapıyı geçse bile sonraki tura ertelensin. Aksi halde en kötü durumda tur 2–5 havuzu ≈16 sa'e (5 koşum) düşer.

## Sonraki turlar için yedek adaylar (kısa; henüz kart değil, tur 2'de sonuca göre seçilir)

- **B1 — `scale` yönü (A3, Z1/Z6):** H01 sonucuna göre: H01 TUT ve AP_S↑ ise `scale=0.9` (geniş jitter); H01 ÖLDÜR/AP_S↓ ise `scale=0.25` (YOLOX-Nano "daralt"; literatür +0…+0.5). Ucuz, kod yok. **[TUR 2'DE KULLANILDI → H04]**
- **B2 — Kayıp kazancı `box=7.5→10` (A5, Z2):** box payı %73.5'e rağmen AP75 boşluğu (**payın %73.5 olduğu önermesi yanlıştır, bkz. Tur 2 düzeltme notu**); ±0.3 beklenti (izole literatür yok); H02 ÖLDÜR ise konum tarafı için en ucuz ikinci deneme. Kod yok (`--override box=10`). **[TUR 2'DE KULLANILDI → H06]**
- **B3 — `close_mosaic` 10→20 (A3/A11, Z6):** mosaic kapanınca cls loss -%21.8 ani düşüş ve val/train cls oranı 1.44 → genelleme açığı; 0…+0.3. Kod yok. *[kullanılmadı; yedek]*
- **B4 — WIoU v3 veya NWD-karışımı (A5, Z1/Z2):** H02 AP_S'te olumlu işaret verirse (NWD: AI-TOD +6.7 AP ama VOC −0.3…+0.3) küçük-nesne konum kaybı; kod ≈40 satır. *[kullanılmadı; H02 AP_S +2.2 gürültü içi olduğundan ve AP75 düştüğünden tetiklenmedi]*
- **B5 — Çıkarım-zamanı satırı (A8, post):** Soft-NMS/DIoU-NMS ve TTA (ölçek yukarı) mevcut baseline `best.pt` üzerinde, eğitim gerektirmez (dakikalar); ayrı satırlarda ve gecikme bedeliyle (TTA ≈3.6x çıkarım → ≈42 FPS tahmin) raporlanır; beklenti 0…+0.4 (Soft-NMS), +0.3…+1.0 (TTA). "Son itme"; kazanan eğitim varyantına uygulanır, makale katkısı değildir. **[TUR 2'DE KULLANILDI → H07]**
- **B6 — `mixup=0.1` (A3, Z3/Z6) — negatif kontrol:** nano'da −0.5…+0.3 (YOLOX-Nano −1.3); yalnız "mixup nano ince ayarda zararlı" bulgusu için (B tipi katkı); düşük öncelik, yalnız bütçe artarsa. *[tur 2'de kullanılmadı, aşağıda gerekçe]*
- **Izgaraya girmemesi gerekenler (tuzak):** `erasing` ve `auto_augment` Ultralytics'te yalnız sınıflandırma görevinde uygulanır (`v8_transforms`'te yoktur; `default.yaml` yorumları "classification"), detect eğitiminde etkisizdir — seed gürültüsü etki gibi görünür; yerleşik `copy_paste` VOC'ta etkisizdir (H03 bunun kutu-tabanlı alternatifidir). Uzun eğitim (150–200 ep) doygunluk nedeniyle (son 20 ep eğimi +0.015 puan/10 ep) değersiz görülür; yalnız transfer/omurga bağlam satırı (YOLO11s, sıfırdan-vs-COCO) ayrı bulgu olarak tur 4–5'te ve kullanıcı onayı gerekirse işaretlenerek.

---

# TUR 2 (K4a) — H04, H05, H06, H07

*Yazım: 2026-09-30 · yalnız val bilgisi; test bilgisine (rep_test_*, test_kilitli.json, rep_test_preds.json) bakılmadı. Ebeveyn: `baseline` `20260930-072103_baseline_s0` (val 0.6289). H03 (tam koşum) sürerken yazıldı, **sonucu bilinmiyor**; tur-2 kartları H03'e koşullu değildir (H03 TUT çıkarsa birleşim ancak ikisi de 3-seed TUT alırsa ayrı birleşim kartıdır).*

## Kapanmış kartlar ve referanslar
- **H01 öldü** (multi_scale=0.25, seed0 Δ +0.07 < +0.30; AP_S −2.8 gürültü içi, AP_M +2.5): batch-düzeyi ölçek çeşitliliği küçük nesneyi iyileştirmedi. → H04 aynı eksende **farklı düğme ve ters yön** (mosaic `scale` daralt), B1'in H01 öncesi yazılmış kuralıyla.
- **H02 öldü** (Inner-CIoU r=0.75, Δ −0.22, AP75 −0.62): kutu-IoU geometrisi AP75 açığını kapatmadı. → H06 aynı ailede **biçim değil ağırlık**; H05 açığın atama/kalite-hedefi tarafını sınar.
- **H03:** koşuyor (≈02:15 biter), sonuç bilinmiyor; tur-2 ebeveyni baseline_s0 kalır.

## Düzeltme notu (M1/M6, karar verileri için kritik)
`hata-analizi.md` §8 ve "Hipotez hammaddesi" Z2 satırındaki **"ağırlıklı kayıp payı box %73.5 / cls %3.8 / dfl %22.6" geçersizdir**: `results.csv` kayıpları (`train/box_loss` vb.) `loss.py:469-475` gereği kazançlar (box 7.5, cls 0.5, dfl 1.5) uygulandıktan sonra `loss.detach()` ile yazılır; analiz kazançları ikinci kez uygulamıştır. Gerçek ep100 payları: box 0.689 / cls 0.538 / dfl 1.060 → **box %30, cls %24, dfl %46** (ep1: %22 / %50 / %28). Etki: (a) "box kaybın %73.5'i, bu yüzden en yüksek kaldıraç" gerekçesi çöker; H06 bu gerekçeye dayanmaz (beklentisi sıfır-merkezli, +%4 TUT olasılığı); (b) dfl en büyük paydır → `dfl` gain yedek B7 olur; (c) H05'te cls payı %24 olduğundan VFL'nin cls gradyanını küçültmesi regresyon/cls dengesini de kaydırır: kartta karışıklık ve ablasyon (iii) olarak yazıldı; (d) H02 kartı (kapanmış, dokunulmadı) aynı yanlış payı alıntılamıştı — kararı (ÖLDÜR) etkilemez. Kayıtların oranları (val/train, ani düşüşler) kazançtan bağımsızdır ve geçerlidir. **Şefe öneri:** hata-analizi.md §8 ve Z2 satırı düzeltilsin (bu ajan dosyayı değiştirmedi).

## Tur 2 ortak karar kuralı (koşumdan ÖNCE donar; H04–H06 kartlarına da aynen yazıldı)
Gerekçe: baseline 3 seed val 0.6289 / 0.6353 / 0.6278 (ort 0.6307, std 0.41 puan). Seed'ler manipülasyonla RNG akışını değiştirdiğinden eşleştirme varyansı az düşürür; eşleştirilmiş Δ'nın seed başına std'si ≈0.58 (=√2·0.41), 3-seed ortalamasının standart hatası ≈0.33 puan. Bu yüzden:
1. **Ön-eleme (yalnız "seed'e çıkma hakkı", kanıt değil):** `v_s0 − 0.6289 >= +0.0030`. Not: baseline_s0 3-seed ortalamanın 0.18 puan altında; eşik 3-seed ortalamaya göre yalnız +0.12'dir ve bilinçli olarak geniş tutuldu (yanlış-eleme maliyeti yüksek). Σ: gerçek etki 0 iken geçme ≈%39, gerçek +0.4 iken ≈%75 (σ=0.41). Altında kalan kart "n=1'de saptanabilir iyileşme yok" diye ÖLDÜR; "etki yok" kanıtı olarak yazılmaz.
2. **Tur başına en çok 1 kart seed1,2'ye çıkar:** ön-elemeyi geçenlerden en yüksek `v_s0` olan; diğerleri "ön-elemeyi geçti, seed'e çıkmadı" diye ertelenir (K5 payı/sonraki tur).
3. **TUT (3 seed, eşleştirilmiş b0=0.6289, b1=0.6353, b2=0.6278):** ortalama Δ >= +0.40 puan **ve** tek yönlü eşleştirilmiş t (df=2, `scipy.stats.ttest_rel(v,b,alternative='greater')`) p<0.10 **ve** >=2 seed'de Δ>0. **ÖLDÜR:** ortalama Δ < +0.15 veya >=2 seed'de Δ<=0. Aradaki: İNCELE (tur içinde ek seed yok; K5 payından n=5 için seed3,4 hem varyant hem baseline). Güç: gerçek +0.6 iken TUT olasılığı ≈%35–40, gerçek 0 iken yanlış-TUT ≈%5; kural bilerek muhafazakârdır.
4. **Çoklu karşılaştırma:** tur 1+2'de 6 eğitim kartı var → hepsi boş iken en az bir yanlış-TUT olasılığı ≈%25. Bu yüzden TUT = "K5 adayı"; nihai iddia K5'te 3 seed + kilitli test (bir kez) ile doğrulanır. Makalede TUT'lar "p<0.10, düzeltmesiz" diye yazılır.
5. Kırmızı bayrak eşiği (karttaki "görülmemiş iyi") aşılırsa TUT verilmez, İNCELE.
6. H07 (eğitimsiz) bu kuralın dışındadır: tek kontrol noktası, val'in yarım-bölünmesi + bootstrap + iki ek ağırlık onayı (kart içinde).

## Tur 2 kartları

| Yürütme sırası | Kart | Aile | Değişken (tek) | Hedeflenen ölçülmüş zayıflık | Beklenen Δ (val map50_95) | P(3-seed TUT) öznel | Maliyet (sa) | Skor = kazanç(TUT ise) x P / maliyet |
|---|---|---|---|---|---|---|---|---|
| 1 | **H07** | A8 (post, eğitimsiz) | NMS-IoU / Soft-NMS / test-imgsz / TTA / ölçek-yukarı TTA / WBF (12 satır, tek tek) | Z1 S <e-posta> %72 (tüm conf) vs L %98; kalabalıkta <e-posta> %67; Loc oracle +5.69 | N 0…+0.2, S 0…+0.4, T +0.3…+1.0, W +0.2…+0.8; U beklenti yok | P(>=1 satır doğrulanır) %45; kısıt-içi %12 | ≈0.8 GPU + ≈150–200 satır kod (GPU'suz) | ≈0.6 x 0.45 / 0.8 ≈ 0.34 (ölçeği farklı: makale katkısı değil, "son itme"; ayrı değerlendirilir) |
| 2 | **H04** | A3 (aug, ölçek) | `scale 0.5→0.25` | Z6 cls val/train 1.44; Z1 S nesneler 16²–32² px, `scale=0.5` onları 8–16 px'e küçültebilir | +0.0…+0.5 (merkez +0.15) | %12 | ≈3.15 (kod yok) | 0.5 x 0.12 / 3.15 ≈ 0.019 |
| 3 | **H05** | A5 (loss, kalite-hedefi) | cls kaybı BCE → VFL (α=0.75, γ=2) | Z4: skor–IoU Spearman 0.62; TP-IoU-sıralı oracle +5.63; Bkg FP 270 | −0.5…+0.5 (merkez 0.0) | %8 | ≈3.5 (3.3 + ≈20 satır kod/test) | 0.6 x 0.08 / 3.5 ≈ 0.014 |
| 4 | **H06** | A5 (loss, ağırlık) | `box 7.5→10.0` | Z2: AP75 0.682 vs AP50 0.811 (12.7 puan); TP'lerin %15.7'si IoU<0.75 | −0.3…+0.3 (merkez 0.0) | %4 | ≈3.15 (kod yok) | 0.5 x 0.04 / 3.15 ≈ 0.006 |

(Olasılıklar öznel tahmindir ve ön-kayıt olarak burada donar.)

### Neden bu sıra
1. **H07 önce:** ≈0.8 GPU sa, eğitim yok; gecikme ölçümü GPU'nun boş olmasını ister, bu da H03 bitimi ile tur-2 kuyruğu başlangıcı arasındaki boşluğa denk gelir. Sonucu eğitim kartlarını etkilemez (ebeveyn zincirine girmez), ama hız–doğruluk tablosunu ve K5 "son itme" satırını erkenden çıkarır. Makale katkısı değil; bu nedenle skoru ayrı okunur.
2. **H04 ikinci:** en yüksek TUT olasılığı (çıpa YOLOX-Nano "daralt" +0…+0.5), sıfır kod, B1'in H01 öncesi yazılmış ön-kayıtlı kuralı; H01'in ters yönü olduğu için ölçek ekseninde iki yönlü bilgi tamamlanır.
3. **H05 üçüncü:** loss ailesinde makale katkısı adayı (H02'nin "AP75 açığı atama/kalite hedefi tarafında olabilir" öğrenimi; Z4 oracle +5.63 en büyük kalite-hedefi payı). Olasılık düşük ama üst-kuyruk büyük; kodu H04'ün koşumu (≈3 sa) sürerken yazılıp CPU'da test edilir, GPU beklemez. Sırada H06'dan önce, çünkü H06'nın dayandığı kayıp-payı gerekçesi yanlış çıktı.
4. **H06 son:** beklenti sıfır-merkezli, TUT olasılığı en düşük (%4); ancak sıfır kod ve H02 ile birlikte "kutu kaybı biçimi ve ağırlığı" tek temiz başarısız-deneme bulgusunu tamamlar. Bütçe kısılırsa ilk düşen budur.

### Seçilmeyenler (gerekçeli)
- **Mixup (B6):** YOLOX-Nano'da −1.3 AP (Ge 2021 Tab.5), çıpa −0.5…+0.3; "mixup nano ince ayarda zararlı" bulgusu yalnız bütçe artarsa negatif kontrol olarak (tur 4–5).
- **close_mosaic 10→20 (B3):** ep81–90 vs 91–100 farkı +0.03 puan (Z6) — kapanmış mosaic'in payı gürültü altında; en düşük kaldıraç, yedekte.
- **WIoU/NWD (B4):** H02 (AP_S +2.2 gürültü içi, AP75 −0.62) tetiklemedi; konum-kaybı ailesine daha fazla GPU harcamadan önce H06 sonucu beklenir.
- **Yeni yedek B7 — `dfl` gain 1.5→3.0 (A5, Z2):** düzeltilmiş payda dfl en büyük terim (%46) ve ince kenar yerelleştirmeyi hedefler; H06 ÖLDÜR ise konum tarafında kalan tek tek-düğme. Kod yok, ±0.3 (izole kaynak yok).

### Hiçbir kart birleşik değildir (M2). KULLANICI ONAYI GEREKEN KART YOK (hiçbiri split/metrik/veri seti değiştirmez; H07 yalnız val'de eğitimsiz çıkarım satırlarıdır, kazanan satırın test ölçümü K5'in kilitli-test adımındadır).

## Tur 2 bütçesi (kullanılan ≈20 GPU sa (H03 dahil) → kalan ≈55 sa)

| Kalem | Sa |
|---|---|
| H04 + H05 + H06 seed0 (3 x 3.0–3.3) + duman/ön-uçuş ≈0.45 | ≈ 9.7 |
| H07 (eğitimsiz, GPU) | ≈ 0.8 (tavan 1.0) |
| **Tur 2 temel toplam** | **≈ 10.5** |
| Seed1,2'ye çıkan tek kart (kural 2: en çok 1) | ≈ +6.1–6.3 |
| P(>=1 kart ön-elemeyi geçer) ≈ 1−0.65·0.70·0.70 | ≈ %68 → beklenen +4.2 |
| **Tur 2 beklenen toplam / en kötü** | **≈ 14.7 / ≈ 16.8** |
| K5 payı (rezerv) | ≈ 20 |
| Tur 3–5 havuzu (beklenen tur-2 sonrası; H03 seed1,2'si hariç) | ≈ 55 − 14.7 − 20 ≈ 20 sa ≈ 6 koşum |

Uyarı: H03 ön-elemeyi geçerse onun seed1,2'si (+6.5 sa) ayrıca havuzdan düşer → tur 3–5 havuzu ≈13–14 sa (≈4 koşum). Bu durumda tur 3 en çok 2 eğitim kartı alır; şef tur-3'ü H03/H04–H06 sonuçlarına göre daraltır.

---

# TUR 3 (K4a, kısmi) — H08 + bekleyen yer tutucular

*Yazım: 2026-10-01 · yalnız val bilgisi; test bilgisine bakılmadı. Girdi: `sonuclar/tahlil-tur1.md` (H01–H03 tahlili, §3 doz analizi, §5 tur-3 önerisi). H04–H06 sonuçları bu yazım anında YOKTUR (kartlarda "Sonuç" boş); bu bölüm onlara koşullu yer tutucular içerir. Ebeveyn: baseline (s0–s2 eşleştirilmiş; 3 seed 63.07 ± 0.41). Karar kuralı: **Tur 2'de donmuş kural** aynen (ön-eleme seed0 Δ>=+0.30; TUT 3 seed ort. Δ>=+0.40, p<0.10, >=2 seed Δ>0; ÖLDÜR ort. Δ<+0.15 veya >=2 seed Δ<=0; tur başına en çok 1 kart seed1,2'ye).*

## Kapanmış kart referansı
- **H03 öldü** (kutu-tabanlı copy-paste p=0.5; 3 seed eşleştirilmiş Δ +0.25 ± 0.24 < +0.30, tek yönlü p=0.107): birincil eşiğin 0.05 puan altı, ancak mAP50 +0.74 ± 0.16 (3/3) ve AP_L +0.34 (3/3) tutarlı, AP75 −0.32 (3/3 negatif); tedavi dozu yalnız 0.142 nesne/mosaic (≈%1.5 nesne). → **H08, H03'ün düzeltmesi değil, yalnızca doz-yanıt noktasıdır; ebeveyni H03 değil baseline'dır** (H03'ün üstüne kurulmaz, kapanmış kural sonradan esnetilmedi).
- H01, H02 öldü (Tur 2 referansları); H04–H07 sonuçları bekleniyor.

## Tur 3 kartları

| Yürütme sırası | Kart | Aile | Değişken (tek) | Hedeflenen ölçülmüş zayıflık | Beklenen Δ (val map50_95) | P(seed0 ön-eleme) / P(3-seed TUT) öznel | Maliyet (sa) | Skor = kazanç(TUT ise) x P(TUT) / maliyet |
|---|---|---|---|---|---|---|---|---|
| 1 | **H08** (H03b) | A3 (aug, kutu copy-paste, doz) | yapıştırma dozu: `p 0.5→1.0` + `max_sqrt_alan 48→96` (0.142 → ≥0.8 nesne/mosaic) | tahlil §3: doz-yanıt eğrisi tek noktalı; H03 izi: mAP50 +0.74 (3/3), AP_L +0.34 (3/3), AP75 −0.32 (3/3); Z1 S/M kütlesi | +0.0…+0.8 (merkez +0.30); mAP50 +0.8…+2.0; AP75 −0.3…−1.0 (merkez −0.6) | %40 / %12–15 | seed0 ≈3.1; beklenen toplam ≈5.9; tavan ≈10 | ≈0.5 x 0.135 / 3.1 ≈ 0.022 (tur-2 H04 0.019 ile aynı büyüklükte; doz-yanıt bulgusu TUT olmasa da makalede kullanılır → bilgi değeri skordan yüksek) |
| 2 (yer tutucu) | **Y1** — lokalizasyon / kayıp yönü (kart değil, id rezerve edilmedi) | A5 | H06 ve H05 ön-eleme sonuçlarına göre seçilir: (a) H06 ÖLDÜR ve H05 ÖLDÜR → B7 `dfl` gain 1.5→3.0 (dfl en büyük kayıp payı %46; Loc oracle +5.7); (b) H05 ön-elemeyi geçerse → VFL α/γ veya cls-kazancı ablasyonu (H05'in kendi ablasyon planı, yeni kart); (c) H06 ön-elemeyi geçerse → `box` 10→12.5 doz-yanıt kartı (aynı H08 mantığı: tek düğme, iki doz) | Z2 Loc oracle +5.7, AP50→AP75 12.7 puan; AP75 H02 (−0.62), H03 (−0.32) ve beklenen H08 ile birlikte düşük | belirlenecek (izole kaynak yok: ±0.3) | belirlenecek (%5–10) | ≈3.15 | belirlenecek |
| 3 (yer tutucu) | **Y2** — ölçek/mosaic yönü (kart değil) | A3 | H04 sonucuna göre: H04 ön-elemeyi geçerse `scale` 0.25→0.1 veya `scale` doz kartı; H04 ÖLDÜR ise B3 `close_mosaic` 10→20 veya `mixup` negatif kontrolü (yalnız bütçe artarsa) | Z6 cls val/train 1.44; S nesneler | belirlenecek | belirlenecek | ≈3.15 | belirlenecek |

(Olasılıklar öznel tahmindir ve ön-kayıt olarak burada donar. Y1/Y2 kart olarak yazılana kadar skor/öncelik verilmez; "H08 + H04–H06 sonuçlarına bağlı" tur-3'ün ikinci yarısıdır. Tur başına ≤3 kart sınırı: H08 + en çok 2 yer tutucu.)

### Neden bu sıra
1. **H08 birinci (tek hazır kart):** tur-1 tahlilinin açık önerisi (§5 madde 1) ve tek **doz-yanıt noktası**; H04–H06 sonuçlarından bağımsız, kod maliyeti çok küçük (H03 modülü yeniden kullanılır; alt-sınıf + kos.py), H03'ün tutarlı izini (mAP50, AP_L) *büyütüp büyütmediğini* sınar. TUT çıkmasa bile 3 noktalı doz-yanıt satırı (0 / 0.142 / ≥0.8 nesne/mosaic) makalenin başarısız-denemeler bölümüne girer ve "H03 izi gürültü mü, doz-yanıtlı mı" sorusunu kapatır. Risk: AP75 düşüşü büyüyebilir (İNCELE tetikleyicisi, karta yazıldı); bu bir ödünleşim bulgusudur, hata değil.
2. **Y1/Y2 sonra:** H04–H06 ön-eleme sonuçları yokken lokalizasyon/loss yönünde yeni kart açılmaz (tahlil §5 madde 3: "ön-eleme sonuçları gelmeden yeni lokalizasyon kartı açmayın"). Hiçbiri TUT almazsa kalan bütçe H08 ve/veya en iyi İNCELE kartının ek seed'lerine gider (aşağıdaki n=5/6 notu).
3. **Birleşim yok:** H08 ile bir loss/ölçek kartı birleşik tek kart olmaz (M2); ancak ikisi de ayrı 3-seed TUT alırsa açıkça "birleşim" kartı olarak yazılır.

### Hiçbir kart birleşik değildir (M2). H08 için KULLANICI ONAYI GEREKMEZ (split/metrik/veri seti değişmez; yalnız train augmentasyon dozu). Y1/Y2 yazıldığında aynı denetim yeniden yapılır.

## Bütçe notu — Tur 3 (kalan toplam şefin kayıt defterine göre; aşağıdakiler SIRALAMA'daki Tur-2 tablosundan türetilmiş tahminlerdir)

| Kalem | Sa |
|---|---|
| H08 seed0 (3.1) + duman/CPU testi (0.3) | ≈ 3.4 |
| H08 seed1,2 (yalnız ön-eleme geçerse, P≈%40) | +6.2 (beklenen +2.5) |
| **H08 beklenen toplam / tavan** | **≈ 5.9 / ≈ 10** |
| Y1 + Y2 seed0 (yer tutucular; yalnız kart yazılırsa) | ≈ 6.3 |
| K5 payı (rezerv, dokunulmaz) | ≈ 20 |

Tur-2 tablosundaki "tur 3–5 havuzu ≈20 sa" ve H03 seed1,2'nin gerçekleşen +6.2 sa'i düşüldüğünde havuz ≈13–14 sa'e iner (Tur-2 uyarısı gerçekleşti); H08 beklenen ≈5.9 sa bunun ≈%43'ü, tavan ≈10 sa ≈%72'sidir. Bu yüzden: H08'in seed1,2'si "tur başına en çok 1 kart" kuralına tabidir ve Y1/Y2 aynı turda seed1,2'ye çıkamaz.

## Not: "baseline'a s3–s5 ekle" önerisi (tahlil-tur1 §5 son paragraf) — maliyet/fayda (KARAR ŞEFİN; K5 payından)

- **Maliyet:** 3 baseline koşumu ≈ 3 x 2.93 ≈ **8.8 sa** (baseline: 2.90 / 2.95 / 2.91 sa). Varyant s3–s5 de eklenirse (eşleştirilmiş analiz için gerekli) varyant başına +9.3 sa daha.
- **Fayda (hesap, öznel varsayım: baseline sd 0.41 puan, varyant sd ≈ aynı, bağımsız):** Δ̄'nın standart hatası, baseline n_b ve varyant n_v ile ≈ √(0.41²/n_v + 0.41²/n_b) (eşleştirme varyansı az düşürdüğü için eşleştirmesiz yaklaşıklık; Tur-2 notu: SE≈0.33 @ n=3/3). **Yalnız baseline'a s3–s5** (varyant n=3): √(0.41²/3 + 0.41²/6) ≈ **0.29** (−%13 vs 0.335). Varyant da n=6 olursa: √(2·0.41²/6) ≈ 0.24 (−%29). Yani baseline-tek-başına ek seed ≈8.8 sa'e %13 SE iyileşmesi; asıl güç kazancı varyant tarafındadır.
- **Kural uyumu:** Donmuş TUT kuralı eşleştirilmiş (b0,b1,b2) tanımlıdır; baseline s3–s5 bu kuralı DEĞİŞTİRMEZ ve geriye dönük TUT/ÖLDÜR kararlarını etkilemez (kural sonradan esnetilmez). Faydası yalnız makalede (baseline'ın referans belirsizliği: 63.07 ± 0.41 n=3 → n=6) ve K5'te kullanılabilir.
- **Öneri (şefe, bağlayıcı değil):** (i) baseline s3–s5'i **tek başına şimdi koşma**; (ii) bir kart İNCELE'de kalırsa (Tur-2 kural 3) K5 payından **eşli seed3,4** koş (baseline 2 + varyant 2 ≈ 12 sa, n=5; kural zaten bunu öngörüyor) — baseline ek seed'leri böylece "bedava" gelir; (iii) hiçbir kart TUT/İNCELE almazsa ve kalan bütçe ≥ 9 sa ise baseline s3–s5, "baseline belirsizliği n=6" satırı olarak son tura bırakılır (bütçe kalanı için en düşük öncelikli kullanım, makalenin ana iddiasını değiştirmez).
- **Yan etki:** baseline_s0 3-seed ortalamanın 0.18 puan altında olduğundan (Tur-2 notu) n=6 ortalaması ön-eleme eşiğinin (0.6289) neden geniş tutulduğunu da somutlaştırır; bu yalnız raporlama bilgisidir.
