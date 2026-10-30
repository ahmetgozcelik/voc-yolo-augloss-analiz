# Hata analizi (K3.5) — baseline `20261001-021011_h03_boxcp_s1`

*Üretici: `scripts/analiz/hata_analizi.py` · yalnız VAL (1000 görüntü, 2407 nesne; difficult nesneler etiketlerde yok) · tahmin sayısı 43421 (conf=0.001, max_det=300, NMS IoU=0.7) · eşleştirme: pycocotools COCOeval · işletim noktası skor≥0.25.*

**Sızıntı notu:** Bu analizde test kümesi (VOC07 test), `rep_test*` dosyaları ve `metrics.json` içindeki `rep_*` anahtarları hiç kullanılmadı; betik bu anahtarları yükler yüklemez atar. Buradan çıkan hipotezler yalnız val bilgisine dayanır.

## 0. Genel ölçüler ve doğrulama

| ölçü | pycocotools (bu betik) | Ultralytics metrics.json (best.pt val) | fark |
|---|---|---|---|
| mAP50-95 | 0.6346 | 0.6351 | -0.0004 |
| mAP50 | 0.8169 | 0.8195 | -0.0026 |
| mAP75 (yalnız pycocotools) | 0.6836 | – | – |

AP50 ile AP50-95 arasındaki fark 18.2 puan; AP50→AP75 farkı 13.3 puan. pycocotools ile Ultralytics AP'si arasındaki küçük fark interpolasyon/eşleştirme (Ultralytics kendi IoU-matrisi eşleşmesini kullanır) kaynaklıdır.

## 1. Sınıf başına AP ve sınıf frekansı

| sınıf (AP50-95 artan) | AP50-95 | AP50 | AP75 | AP50−AP75 | train nesne | val nesne | val GT medyan kenar px |
|---|---|---|---|---|---|---|---|
| pottedplant | 0.312 | 0.562 | 0.256 | 0.305 | 1378 | 109 | 71 |
| chair | 0.475 | 0.654 | 0.535 | 0.120 | 2975 | 177 | 99 |
| bottle | 0.529 | 0.704 | 0.581 | 0.123 | 1669 | 95 | 70 |
| boat | 0.563 | 0.791 | 0.638 | 0.153 | 1076 | 64 | 112 |
| bird | 0.576 | 0.776 | 0.596 | 0.180 | 1514 | 91 | 115 |
| diningtable | 0.595 | 0.719 | 0.681 | 0.038 | 775 | 49 | 211 |
| horse | 0.615 | 0.795 | 0.641 | 0.154 | 1006 | 66 | 209 |
| tvmonitor | 0.624 | 0.843 | 0.708 | 0.135 | 1036 | 72 | 108 |
| motorbike | 0.638 | 0.834 | 0.666 | 0.167 | 985 | 67 | 239 |
| sofa | 0.640 | 0.771 | 0.707 | 0.064 | 755 | 59 | 237 |
| person | 0.641 | 0.890 | 0.689 | 0.201 | 12471 | 785 | 141 |
| car | 0.652 | 0.849 | 0.707 | 0.142 | 3045 | 222 | 92 |
| cow | 0.672 | 0.841 | 0.787 | 0.054 | 791 | 56 | 145 |
| sheep | 0.705 | 0.870 | 0.741 | 0.129 | 1013 | 57 | 119 |
| aeroplane | 0.706 | 0.900 | 0.709 | 0.191 | 1091 | 80 | 184 |
| bicycle | 0.708 | 0.920 | 0.743 | 0.177 | 1002 | 62 | 171 |
| bus | 0.733 | 0.846 | 0.772 | 0.075 | 767 | 55 | 192 |
| train | 0.735 | 0.896 | 0.803 | 0.093 | 868 | 57 | 225 |
| cat | 0.784 | 0.931 | 0.837 | 0.094 | 1505 | 88 | 278 |
| dog | 0.789 | 0.946 | 0.875 | 0.071 | 1929 | 96 | 211 |

En zayıf 5 sınıf (AP50-95): pottedplant 0.312, chair 0.475, bottle 0.529, boat 0.563, bird 0.576. Sınıf frekansı (train nesne) ile AP50-95 Spearman ρ=-0.18 (permütasyon p=0.443), AP50 ile ρ=0.09; sınıf başına train küçük-nesne oranı ile ρ=-0.48; val GT medyan kenar uzunluğu ile ρ=0.59 (n=20 sınıf, gürültülü). Frekans AP'nin zayıf belirleyicisi (|ρ|<0.3): zayıf sınıflarda sorun veri azlığından çok nesne boyutu/görünümü/bağlamıdır.

## 2. Nesne boyutu kırılımı (orijinal çözünürlük; COCO S<32²≤M<96²≤L)

| boyut | GT sayısı | AP50-95 | AP50 | <e-posta> (tüm conf, sınıf ort.) | <e-posta>:0.95 (sınıf ort.) | <e-posta> skor≥0.25 |
|---|---|---|---|---|---|---|
| all | 2407 | 0.635 | 0.817 | 0.961 | 0.767 | 0.797 |
| S | 143 | 0.225 | 0.382 | 0.752 | 0.472 | 0.399 |
| M | 704 | 0.442 | 0.649 | 0.927 | 0.662 | 0.699 |
| L | 1560 | 0.712 | 0.880 | 0.978 | 0.816 | 0.878 |

İnce alan aralıkları (GT düzeyi, eşleşme bayrağı):

| alan (px²) | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) |
|---|---|---|---|---|
| <16² | 9 | 33.3% | 0.0% | 0.0% |
| 16²–32² | 134 | 87.3% | 42.5% | 54.5% |
| 32²–64² | 377 | 95.8% | 66.8% | 74.8% |
| 64²–96² | 327 | 95.4% | 73.4% | 78.3% |
| 96²–192² | 698 | 98.0% | 81.8% | 83.4% |
| ≥192² | 862 | 99.0% | 92.6% | 92.6% |

Sınıf × boyut AP50-95 (n<5 ise '–'):

| sınıf | n_S | AP_S | n_M | AP_M | n_L | AP_L |
|---|---|---|---|---|---|---|
| aeroplane | 0 | – | 21 | 0.55 | 59 | 0.76 |
| bicycle | 0 | – | 12 | 0.39 | 50 | 0.79 |
| bird | 3 | – | 36 | 0.28 | 52 | 0.77 |
| boat | 4 | – | 25 | 0.54 | 35 | 0.65 |
| bottle | 13 | 0.18 | 52 | 0.47 | 30 | 0.74 |
| bus | 3 | – | 13 | 0.42 | 39 | 0.88 |
| car | 38 | 0.35 | 80 | 0.64 | 104 | 0.76 |
| cat | 0 | – | 2 | – | 86 | 0.80 |
| chair | 3 | – | 79 | 0.46 | 95 | 0.51 |
| cow | 0 | – | 25 | 0.67 | 31 | 0.72 |
| diningtable | 0 | – | 1 | – | 48 | 0.61 |
| dog | 0 | – | 10 | 0.74 | 86 | 0.80 |
| horse | 3 | – | 15 | 0.17 | 48 | 0.76 |
| motorbike | 3 | – | 10 | 0.56 | 54 | 0.68 |
| person | 51 | 0.22 | 215 | 0.51 | 519 | 0.73 |
| pottedplant | 12 | 0.27 | 57 | 0.31 | 40 | 0.37 |
| sheep | 8 | 0.36 | 16 | 0.62 | 33 | 0.84 |
| sofa | 0 | – | 1 | – | 58 | 0.65 |
| train | 0 | – | 4 | – | 53 | 0.76 |
| tvmonitor | 2 | – | 30 | 0.58 | 40 | 0.67 |

Küçük nesneler (n=143, val GT'nin %5.9'i): AP50-95 0.225 vs büyük 0.712 (fark 48.7 puan); <e-posta> 0.752 vs 0.978. Not: ölçüm orijinal piksel; eğitim/çıkarım 640 px letterbox olduğundan (~1.3× büyütme) S nesneler modelde ≈<41 px görünür. Val'de S örneği az olduğundan S-AP güven aralığı geniştir.

## 3. Hata türü ayrıştırması (TIDE benzeri)

Sayılar: tüm tahminler (conf≥0.001) ve işletim noktası (skor≥0.25). `Miss` tüm conf'ta hiçbir kutuyla IoU≥0.5 eşleşmeyen GT; işletim noktası sütununda ise o eşikte bulunamayan GT (eşleşmeyen + düşük skorla eşleşen).

| tür | sayı (tüm conf) | oran | sayı skor≥0.25 |
|---|---|---|---|
| TP | 2330 | 5.4% | 1918 |
| Dupe | 2430 | 5.6% | 81 |
| Cls | 3427 | 7.9% | 91 |
| Loc | 7808 | 18.0% | 157 |
| Both | 5372 | 12.4% | 66 |
| Bkg | 22054 | 50.8% | 265 |
| Miss | 77 | 3.2% (GT'nin) | 489 |

Oracle ΔAP (puan): ilgili hata türü düzeltilirse mAP ne kadar artardı (AP50 ve AP50-95, sınıf ortalaması). Taban: AP50=81.69, AP50-95=63.46.

| düzeltilen | ΔAP50 (puan) | ΔAP50-95 (puan) |
|---|---|---|
| Cls | 2.40 | 1.79 |
| Loc | 2.77 | 5.50 |
| Both | 1.15 | 0.81 |
| Dupe | 0.56 | -0.86 |
| Bkg | 3.60 | 2.32 |
| Miss | 3.31 | 2.48 |
| FP_hepsi | 13.86 | 6.11 |
| Kutu_kusursuz | 0.00 | 18.23 |
| TP_IoU_sirali | 1.38 | 5.31 |

Protokol etkisi: val'de etiketlerden çıkarılmış difficult nesne sayısı 493 (XML). Bunlardan biriyle aynı sınıf IoU≥0.5 örtüşen ve eşleşmemiş (FP sayılan) tahmin: 910 (skor≥0.25: 131; işletim noktası FP'lerinin 19.8%'i). Bu tahminler devkit gibi yok sayılsaydı ΔAP50 = +1.45, ΔAP50-95 = +1.02 puan (Ultralytics protokolünün val üzerinde ölçülmüş cezası; makalede belirtilmeli, Bkg/Loc/Dupe sayılarının bir kısmı bu yüzden yapaydır).


Not: `Dupe` ve `Loc` ΔAP50-95 değerleri IoU=0.5 tanımına göredir; bu kutular daha yüksek IoU eşiklerinde TP olabildiği için silme oracle'ı (Dupe) AP50-95'i düşürebilir; Loc oracle'ı kutuyu GT'ye yapıştırdığı için AP50-95 kazancı AP50'den büyüktür.


Hata türü × boyut (işletim noktası; tam uzun biçim `hata-kirilim.csv`):

| tür | S | M | L |
|---|---|---|---|
| TP (skor≥0.25, tahmin kutusu alanı) | 55 | 477 | 1386 |
| Dupe (skor≥0.25, tahmin kutusu alanı) | 1 | 25 | 55 |
| Cls (skor≥0.25, tahmin kutusu alanı) | 2 | 19 | 70 |
| Loc (skor≥0.25, tahmin kutusu alanı) | 8 | 69 | 80 |
| Both (skor≥0.25, tahmin kutusu alanı) | 1 | 13 | 52 |
| Bkg (skor≥0.25, tahmin kutusu alanı) | 51 | 123 | 91 |
| Miss (skor≥0.25 bulunamayan GT, GT alanı) | 86 | 212 | 191 |
| GT sayısı | 143 | 704 | 1560 |

AP50'ye en çok mal olan hata türü: Bkg (+3.60), sonra Miss (+3.31) ve Loc (+2.77). Oracle'lar birbirinden bağımsız ölçüldü, toplamları ΔAP'ye eşit değildir (etkileşim). `Kutu_kusursuz` = tüm TP kutuları GT'ye yapıştırılırsa AP50-95'in erişeceği üst sınır; `TP_IoU_sirali` = yalnız TP'lerin skorları IoU sırasına göre dağıtılırsa (FP skorları aynı) kazanç, yani IoU-farkında skorlamanın (VFL/QFL/IoU-aware) teorik payı.


Sınıf başına ΔAP50 (puan, oracle):

| sınıf | Cls | Loc | Both | Dupe | Bkg | Miss |
|---|---|---|---|---|---|---|
| aeroplane | 0.9 | 2.7 | 0.3 | 0.4 | 1.1 | 2.5 |
| bicycle | 0.3 | 1.2 | 0.4 | 0.5 | 0.4 | 3.4 |
| bird | 2.0 | 5.9 | 0.2 | 1.1 | 3.1 | 3.5 |
| boat | 0.8 | 4.0 | 0.7 | 0.5 | 4.5 | 4.0 |
| bottle | 0.3 | 2.4 | 0.1 | 0.4 | 10.8 | 5.5 |
| bus | 1.8 | 0.1 | 0.2 | 0.7 | 2.2 | 6.9 |
| car | 0.8 | 1.8 | 0.8 | 0.7 | 3.6 | 2.8 |
| cat | 2.2 | 2.6 | 0.2 | 0.3 | 0.2 | 1.3 |
| chair | 1.6 | 5.0 | 1.9 | 0.6 | 6.6 | 0.9 |
| cow | 7.1 | 0.9 | 0.4 | 0.9 | 3.4 | 3.2 |
| diningtable | 1.8 | 2.3 | 10.7 | 0.4 | 1.4 | 6.6 |
| dog | 1.9 | 0.9 | 0.1 | 0.1 | 0.5 | 1.3 |
| horse | 11.5 | 1.7 | 0.3 | 0.9 | 0.7 | 9.5 |
| motorbike | 2.3 | 3.9 | 0.8 | 0.4 | 2.4 | 2.4 |
| person | 0.1 | 3.0 | 0.1 | 0.9 | 2.3 | 1.6 |
| pottedplant | 0.5 | 10.4 | 1.4 | 0.7 | 9.2 | 6.9 |
| sheep | 4.8 | 0.4 | 0.6 | 0.2 | 4.3 | 0.0 |
| sofa | 6.0 | 0.6 | 2.4 | 1.5 | 5.3 | 1.3 |
| train | 0.3 | 4.1 | 0.3 | 0.1 | 2.5 | 1.7 |
| tvmonitor | 0.8 | 1.4 | 1.2 | 0.0 | 7.5 | 1.3 |

Kaçırılan GT (77) alt kırılımı: yalnız başka sınıfla örtüşen kutu var (sınıf karışıklığı) 22; aynı sınıf kutu var ama IoU∈[0.1,0.5) (lokalizasyon) 29; aynı sınıf IoU≥0.5 kutu var ama başka GT'ye atanmış (atama çakışması/örtüşme) 0; hiç kutu yok (gerçek kaçırma) 26 (kalan sınıflama kesin değil: bu üç grup kesişebilir).

## 4. Lokalizasyon kalitesi

| IoU aralığı | TP (tüm conf) | oran | TP skor≥0.25 | oran |
|---|---|---|---|---|
| 0.50–0.55 | 76 | 3.3% | 32 | 1.7% |
| 0.55–0.60 | 76 | 3.3% | 40 | 2.1% |
| 0.60–0.65 | 87 | 3.7% | 39 | 2.0% |
| 0.65–0.70 | 127 | 5.5% | 71 | 3.7% |
| 0.70–0.75 | 161 | 6.9% | 115 | 6.0% |
| 0.75–0.80 | 165 | 7.1% | 124 | 6.5% |
| 0.80–0.85 | 231 | 9.9% | 191 | 10.0% |
| 0.85–0.90 | 318 | 13.6% | 281 | 14.7% |
| 0.90–0.95 | 521 | 22.4% | 473 | 24.7% |
| 0.95–1.00 | 568 | 24.4% | 552 | 28.8% |

TP IoU ortalaması 0.846 (skor≥0.25: 0.871); işletim noktası TP'lerinin %15.5'i IoU<0.75. TP'lerde skor–IoU Spearman ρ=0.63 (skorun konum kalitesini ne kadar taşıdığı).

Sınıf başına AP50 − AP75 (büyükten küçüğe) ve TP ort. IoU:

| sınıf | AP50 | AP75 | AP50−AP75 (puan) | TP ort. IoU |
|---|---|---|---|---|
| pottedplant | 0.562 | 0.256 | 30.5 | 0.766 |
| person | 0.890 | 0.689 | 20.1 | 0.835 |
| aeroplane | 0.900 | 0.709 | 19.1 | 0.858 |
| bird | 0.776 | 0.596 | 18.0 | 0.832 |
| bicycle | 0.920 | 0.743 | 17.7 | 0.857 |
| motorbike | 0.834 | 0.666 | 16.7 | 0.842 |
| horse | 0.795 | 0.641 | 15.4 | 0.850 |
| boat | 0.791 | 0.638 | 15.3 | 0.836 |
| car | 0.849 | 0.707 | 14.2 | 0.847 |
| tvmonitor | 0.843 | 0.708 | 13.5 | 0.859 |
| sheep | 0.870 | 0.741 | 12.9 | 0.881 |
| bottle | 0.704 | 0.581 | 12.3 | 0.842 |
| chair | 0.654 | 0.535 | 12.0 | 0.820 |
| cat | 0.931 | 0.837 | 9.4 | 0.902 |
| train | 0.896 | 0.803 | 9.3 | 0.885 |
| bus | 0.846 | 0.772 | 7.5 | 0.894 |
| dog | 0.946 | 0.875 | 7.1 | 0.896 |
| sofa | 0.771 | 0.707 | 6.4 | 0.873 |
| cow | 0.841 | 0.787 | 5.4 | 0.876 |
| diningtable | 0.719 | 0.681 | 3.8 | 0.885 |

AP50→AP75 düşüşü sınıf ortalaması 13.3 puan; en büyük düşüş pottedplant (31), person (20), aeroplane (19). Kutu-regresyon (IoU/DFL/box kazancı) manipülasyonlarının hedef alanı budur.

## 5. Sınıf karışıklığı

İşletim noktası (skor≥0.25, IoU≥0.5): her GT için eşleşen en yüksek skorlu kutu (doğru sınıf varsa o). Matris dosyası: `hata-analizi_20261001-021011_h03_boxcp_s1_karisiklik.csv`.

| GT sınıfı → tahmin | sayı | GT'nin payı |
|---|---|---|
| chair → person | 6 | 3.4% |
| horse → sheep | 6 | 9.1% |
| cow → horse | 4 | 7.1% |
| cow → sheep | 4 | 7.1% |
| horse → cow | 3 | 4.5% |
| sofa → chair | 3 | 5.1% |
| sofa → person | 3 | 5.1% |
| motorbike → person | 2 | 3.0% |
| sheep → dog | 2 | 3.5% |
| boat → car | 1 | 1.6% |
| bottle → person | 1 | 1.1% |
| car → boat | 1 | 0.5% |

Çift bazında (yönsüz) toplam: cow/horse 7; chair/person 6; horse/sheep 6; cow/sheep 4; chair/sofa 3; person/sofa 3; motorbike/person 3; dog/sheep 2.


Tüm-conf alternatif görünüm: `Cls` türü tahminler (IoU≥0.5 başka sınıf GT, skor≥0.05), GT sınıfı → tahmin:

| GT → tahmin | tahmin sayısı |
|---|---|
| sofa → chair | 26 |
| person → chair | 24 |
| person → sofa | 18 |
| cow → horse | 16 |
| dog → cat | 14 |
| chair → sofa | 10 |
| motorbike → person | 9 |
| cow → sheep | 8 |
| horse → cow | 8 |
| horse → sheep | 7 |

Kanonik çiftler (yönsüz; `Cls` tahmin sayısı skor≥0.05 | işletim noktası GT sayısı): cat/dog 20 | 1; cow/sheep 14 | 4; chair/diningtable 8 | 2; chair/sofa 36 | 3; horse/cow 24 | 7; bus/train 5 | 0; motorbike/bicycle 10 | 0.

İşletim noktasında GT'lerin 2.1%'i başka sınıf olarak etiketlendi (50/2407); 17.6%'i hiç bulunamadı. Sınıf karışıklığı oracle ΔAP50 = 2.40 puan (§3; Cls, altı hata türü içinde 4. sırada); işletim noktasında karışan GT oranı yalnız 2.1%, yani sorun sınıf ayrımından çok kutu/skor düzeyindedir.

## 6. Kalabalık ve örtüşme

| nesne/görüntü | görüntü | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) | FP/görüntü (skor≥0.25) |
|---|---|---|---|---|---|---|
| 1 | 440 | 440 | 99.1% | 93.2% | 91.4% | 0.39 |
| 2 | 252 | 504 | 97.2% | 85.1% | 87.5% | 0.43 |
| 3–4 | 195 | 654 | 96.6% | 77.4% | 83.0% | 0.96 |
| 5–8 | 88 | 511 | 95.5% | 70.1% | 76.7% | 1.33 |
| ≥9 | 25 | 298 | 95.3% | 72.1% | 71.5% | 3.04 |

Aynı sınıftan başka GT ile en yüksek IoU (örtüşme) gruplarına göre recall:

| komşu GT IoU | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 |
|---|---|---|---|
| <0.1 (komşu yok) | 1908 | 96.8% | 81.1% |
| 0.1–0.3 | 357 | 95.8% | 74.8% |
| 0.3–0.5 | 105 | 99.0% | 75.2% |
| 0.5–0.7 | 33 | 100.0% | 63.6% |
| ≥0.7 | 4 | 100.0% | 75.0% |

Aynı sınıf GT çiftleri (IoU≥0.3): 75 çift; ikisi de bulundu 74, yalnız biri 1, hiçbiri 0. IoU≥0.7 (NMS eşiği üstü, doğrudan NMS etkilenir): 2 çift, ikisi de bulundu 2, yalnız biri 0, hiçbiri 0. Kaçırılan GT'lerin 0/77'i aynı sınıf IoU≥0.5 bir kutuya sahip ama o kutu başka GT'ye atanmış (olası NMS/atama kaybı).

Kalabalıkla recall düşüşü: tek nesneli görüntüde 93.2% iken ≥9 nesnelide 72.1% (skor≥0.25); tüm-conf recall 99.1% → 95.3%, <e-posta> 91.4% → 71.5%. Çiftlerde tek-bulunma ve atama çakışması sayıları küçükse kayıp NMS'ten değil düşük skor/zayıf konumdan gelir.

## 7. Kalibrasyon (skor vs gerçek precision)

Tahmin 'doğru' = IoU≥0.5 ile eşleşen (<e-posta>); ayrıca IoU≥0.75 için precision. Not: YOLO sınıf skoru olasılık değil, TAL hizalama hedefli (IoU'ya bağlı) bir skordur; düşük skor bölgesindeki düşük precision 'aday havuzu' doğasından gelir.

| skor aralığı | tahmin | ort. skor | <e-posta> | <e-posta> | TP ort. IoU | skor−precision |
|---|---|---|---|---|---|---|
| 0.0–0.1 | 39794 | 0.008 | 0.007 | 0.006 | 0.715 | 0.001 |
| 0.1–0.2 | 793 | 0.143 | 0.113 | 0.079 | 0.763 | 0.030 |
| 0.2–0.3 | 434 | 0.245 | 0.235 | 0.173 | 0.769 | 0.010 |
| 0.3–0.4 | 265 | 0.347 | 0.343 | 0.226 | 0.779 | 0.004 |
| 0.4–0.5 | 208 | 0.446 | 0.466 | 0.327 | 0.791 | -0.020 |
| 0.5–0.6 | 218 | 0.548 | 0.537 | 0.376 | 0.781 | 0.011 |
| 0.6–0.7 | 200 | 0.652 | 0.670 | 0.495 | 0.829 | -0.018 |
| 0.7–0.8 | 296 | 0.752 | 0.834 | 0.632 | 0.819 | -0.082 |
| 0.8–0.9 | 681 | 0.858 | 0.959 | 0.875 | 0.890 | -0.101 |
| 0.9–1.0 | 532 | 0.927 | 0.991 | 0.974 | 0.944 | -0.064 |

ECE (tüm tahminler, 10 eşit genişlikli kutu) = 0.005; ECE (skor≥0.1) = 0.046. `skor−precision` pozitifse aşırı güvenli, negatifse eksik güvenli. 
Skor≥0.5 tahminlerin <e-posta> = 87.1%, <e-posta> = 76.9% (fark 10.2 puan). Skor 0.5 üstü kutularda (skor−precision) ortalaması -0.051: model yüksek skorlarda eksik güvenli (skor < gerçek <e-posta>), sıralama iyi; asıl boşluk <e-posta> ile @0.75 arasındaki konum kalitesidir. TP'lerde skor–IoU Spearman ρ=0.63: skor konum kalitesini kısmen taşıyor. TP_IoU_sirali oracle'ı (§3) IoU-farkında skorlamanın teorik payını sayısallaştırıyor.

## 8. Eğitim dinamiği (results.csv)

| gösterge | değer |
|---|---|
| epoch sayısı | 100 |
| en iyi epoch (mAP50-95) | 97 (0.6341) |
| son epoch mAP50-95 | 0.6339 |
| epoch 50 / 80 / 90 / 100 | 0.6208 / 0.6327 / 0.6341 / 0.6339 |
| son 20 epoch eğimi (puan/10 epoch) | 0.065 |
| önceki 20 epoch (ep 71-90) eğimi | 0.162 |
| son 20 epoch std (puan) | 0.051 |
| ep 81-90 ort. vs ep 91-100 ort. (close_mosaic=10) | 0.6331 vs 0.6336 (Δ +0.05 puan) |
| mAP50 son epoch | 0.8226 |
| results.csv en iyi ep mAP50-95 vs metrics.json (best.pt yeniden val) | 0.6341 vs 0.6351 (fark -0.10 puan; ölçüm-protokolü gürültü tabanı) |

Loss tablosu (train loss mosaic açıkken ep1/ep90, kapalıyken ep100; val/train oranı):

| loss | train ep1 | train ep90 | train ep100 | val ep100 | val/train ep100 | val/train ep90 | train loss ep90→91 değişim |
|---|---|---|---|---|---|---|---|
| box | 1.008 | 0.803 | 0.687 | 0.871 | 1.27 | 1.09 | -7.4% |
| cls | 2.312 | 0.816 | 0.539 | 0.773 | 1.44 | 0.95 | -22.5% |
| dfl | 1.279 | 1.139 | 1.057 | 1.161 | 1.10 | 1.02 | -3.5% |

Kayıp payı (results.csv loss_items kazançlı; loss.py:469-471 — önceki sürüm kazançları iki kez uyguluyordu, 2026-10-01 düzeltildi) ep100: box 30.1%, cls 23.6%, dfl 46.3% (ep1: box 21.9%, cls 50.3%, dfl 27.8%). Son 20 epoch mAP eğimi +0.065 puan/10 ep (std 0.05) → doygun (eğim seed gürültüsü altında). close_mosaic sonrası (ep 91-100) ortalama +0.05 puan. Train loss mosaic kapanınca ani düşer (cls -22.5%); bu rejim değişimi, train/val farkının bir kısmının augmentation kaynaklı olduğunu gösterir. val cls loss son 20 epoch 0.781→0.773 (yükselmiyor): val loss bozulmadığı için klasik aşırı öğrenme yok, ama mosaic kapalıyken val/train oranı cls 1.44, box 1.27: train kaybı val kaybından ayrışıyor (genelleme açığı); val mAP doygun.

## 9. Seed varyansı

| koşum | seed | epoch | mAP50-95 | mAP50 | mAP_S | mAP_M | mAP_L |
|---|---|---|---|---|---|---|---|
| 20260929-230007_baseline_s0 | 0 | 2 | 0.1267 | 0.1807 | 0.023 | 0.096 | 0.146 |
| 20260930-065935_baseline_s0 | 0 | 3 | 0.4104 | 0.5813 | 0.098 | 0.289 | 0.456 |
| 20260930-070132_baseline_s0 | 0 | 5 | 0.5867 | 0.7807 | 0.178 | 0.421 | 0.664 |
| 20260930-071120_baseline_s1 | 1 | 5 | 0.5908 | 0.7849 | 0.169 | 0.388 | 0.672 |
| 20260930-072103_baseline_s0 | 0 | 100 | 0.6289 | 0.8113 | 0.236 | 0.434 | 0.702 |
| 20260930-101624_baseline_s1 | 1 | 100 | 0.6353 | 0.8136 | 0.207 | 0.452 | 0.711 |
| 20260930-131427_baseline_s2 | 2 | 100 | 0.6278 | 0.8113 | 0.220 | 0.443 | 0.701 |

Epoch grubuna göre (aynı epoch sayısı, seed'ler arası; n≥2 iken std):

| epoch | n koşum | ort. mAP50-95 | std |
|---|---|---|---|
| 2 | 1 | 0.1267 | n<2 |
| 3 | 1 | 0.4104 | n<2 |
| 5 | 2 | 0.5888 | 0.0029 |
| 100 | 3 | 0.6307 | 0.0041 |

*Tam (100 ep) koşumlarda seed1/seed2 bittiğinde şef bu betiği yeniden koşturunca tablo otomatik dolar (`--seed-glob`). Ön-uçuş (5 ep) iki seed yalnız erken işarettir; 5 epoch'ta mAP henüz yakınsamadığından tam koşum seed std'si için temsilî değildir.*

## 10. Hız profili (metrics.json)

| gösterge | değer |
|---|---|
| parametre | 2593740 |
| GFLOPs (640) | 6.52 |
| ön-işleme ms | 1.2769 |
| çıkarım ms | 6.0274 |
| NMS/son-işlem ms | 0.6927 |
| toplam ms | 7.997 |
| FPS | 125.047 |
| batch | 1 |
| FP16 (1=evet) | 1.0 |
| ölçüm görüntüsü | 300 |
| ısınma görüntüsü | 50 |
| eğitim süresi (saat) | 2.9284 |

Varyant hız bütçesi (yalnız tahmin; gerçek ölçüm varyant koşumunda yapılır):

| varyant sınıfı | çarpan | tahmini toplam ms | tahmini FPS | not |
|---|---|---|---|---|
| eğitim-zamanı (aug/loss) varyantları | aynı | 8.00 | 125 | çıkarım grafiği değişmez; hız bütçesi ≈ 0 |
| Soft-NMS/DIoU-NMS (son-işlem ×1–5 varsayımı) | NMS ×1–5 | 8.00–10.77 | 93–125 | tahmin (ölçülmedi) |
| TTA (Ultralytics augment, ~3 ölçek+flip; literatür çarpanı 3.6× çıkarım) | çıkarım ×3.6 (literatür) | 23.67 | 42 | tahmin; literatur-cipa.md §4 (Ultralytics TTA dokümanı) |
| YOLO11s'e geçiş (bağlam satırı) | ≈3.4× GFLOPs | – | – | ölçülmedi |

Baseline batch=1 FP16 toplam 8.00 ms (125 FPS; çıkarım 6.03 ms, NMS 0.69 ms = toplamın %9'i). 'Gerçek zamanlı' için ≥30 FPS kabul edilirse (33 ms) ~4.2× gecikme payı vardır; TTA bile (~42 FPS tahmini) eşiğin üstünde kalabilir, ancak hız-doğruluk satırı ayrı raporlanmalıdır. Not: hız ölçümü GPU'nun paylaşıldığı bir anda alınmış olabilir (eğitim kuyruğu); karşılaştırmalı hız ölçümü aynı koşulda yapılmalıdır.
