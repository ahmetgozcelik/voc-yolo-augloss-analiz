# Hata analizi (K3.5) — baseline `20261002-184450_h02_innerciou_s1`

*Üretici: `scripts/analiz/hata_analizi.py` · yalnız VAL (1000 görüntü, 2407 nesne; difficult nesneler etiketlerde yok) · tahmin sayısı 43861 (conf=0.001, max_det=300, NMS IoU=0.7) · eşleştirme: pycocotools COCOeval · işletim noktası skor≥0.25.*

**Sızıntı notu:** Bu analizde test kümesi (VOC07 test), `rep_test*` dosyaları ve `metrics.json` içindeki `rep_*` anahtarları hiç kullanılmadı; betik bu anahtarları yükler yüklemez atar. Buradan çıkan hipotezler yalnız val bilgisine dayanır.

## 0. Genel ölçüler ve doğrulama

| ölçü | pycocotools (bu betik) | Ultralytics metrics.json (best.pt val) | fark |
|---|---|---|---|
| mAP50-95 | 0.6300 | 0.6306 | -0.0007 |
| mAP50 | 0.8127 | 0.8148 | -0.0021 |
| mAP75 (yalnız pycocotools) | 0.6871 | – | – |

AP50 ile AP50-95 arasındaki fark 18.3 puan; AP50→AP75 farkı 12.6 puan. pycocotools ile Ultralytics AP'si arasındaki küçük fark interpolasyon/eşleştirme (Ultralytics kendi IoU-matrisi eşleşmesini kullanır) kaynaklıdır.

## 1. Sınıf başına AP ve sınıf frekansı

| sınıf (AP50-95 artan) | AP50-95 | AP50 | AP75 | AP50−AP75 | train nesne | val nesne | val GT medyan kenar px |
|---|---|---|---|---|---|---|---|
| pottedplant | 0.306 | 0.557 | 0.243 | 0.314 | 1378 | 109 | 71 |
| chair | 0.452 | 0.624 | 0.507 | 0.117 | 2975 | 177 | 99 |
| bottle | 0.531 | 0.686 | 0.622 | 0.065 | 1669 | 95 | 70 |
| boat | 0.551 | 0.766 | 0.650 | 0.116 | 1076 | 64 | 112 |
| diningtable | 0.574 | 0.706 | 0.656 | 0.050 | 775 | 49 | 211 |
| bird | 0.595 | 0.792 | 0.628 | 0.165 | 1514 | 91 | 115 |
| tvmonitor | 0.606 | 0.827 | 0.732 | 0.095 | 1036 | 72 | 108 |
| sofa | 0.613 | 0.744 | 0.631 | 0.113 | 755 | 59 | 237 |
| motorbike | 0.625 | 0.829 | 0.665 | 0.165 | 985 | 67 | 239 |
| horse | 0.629 | 0.839 | 0.667 | 0.172 | 1006 | 66 | 209 |
| person | 0.637 | 0.891 | 0.687 | 0.205 | 12471 | 785 | 141 |
| car | 0.648 | 0.844 | 0.722 | 0.122 | 3045 | 222 | 92 |
| cow | 0.682 | 0.874 | 0.776 | 0.098 | 791 | 56 | 145 |
| bicycle | 0.687 | 0.886 | 0.767 | 0.119 | 1002 | 62 | 171 |
| aeroplane | 0.709 | 0.903 | 0.717 | 0.185 | 1091 | 80 | 184 |
| bus | 0.735 | 0.851 | 0.785 | 0.066 | 767 | 55 | 192 |
| sheep | 0.736 | 0.893 | 0.783 | 0.110 | 1013 | 57 | 119 |
| train | 0.739 | 0.888 | 0.808 | 0.080 | 868 | 57 | 225 |
| dog | 0.769 | 0.936 | 0.864 | 0.071 | 1929 | 96 | 211 |
| cat | 0.774 | 0.918 | 0.834 | 0.083 | 1505 | 88 | 278 |

En zayıf 5 sınıf (AP50-95): pottedplant 0.306, chair 0.452, bottle 0.531, boat 0.551, diningtable 0.574. Sınıf frekansı (train nesne) ile AP50-95 Spearman ρ=-0.14 (permütasyon p=0.567), AP50 ile ρ=0.07; sınıf başına train küçük-nesne oranı ile ρ=-0.41; val GT medyan kenar uzunluğu ile ρ=0.57 (n=20 sınıf, gürültülü). Frekans AP'nin zayıf belirleyicisi (|ρ|<0.3): zayıf sınıflarda sorun veri azlığından çok nesne boyutu/görünümü/bağlamıdır.

## 2. Nesne boyutu kırılımı (orijinal çözünürlük; COCO S<32²≤M<96²≤L)

| boyut | GT sayısı | AP50-95 | AP50 | <e-posta> (tüm conf, sınıf ort.) | <e-posta>:0.95 (sınıf ort.) | <e-posta> skor≥0.25 |
|---|---|---|---|---|---|---|
| all | 2407 | 0.630 | 0.813 | 0.960 | 0.769 | 0.797 |
| S | 143 | 0.225 | 0.384 | 0.728 | 0.441 | 0.406 |
| M | 704 | 0.448 | 0.645 | 0.945 | 0.692 | 0.719 |
| L | 1560 | 0.704 | 0.870 | 0.977 | 0.816 | 0.868 |

İnce alan aralıkları (GT düzeyi, eşleşme bayrağı):

| alan (px²) | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) |
|---|---|---|---|---|
| <16² | 9 | 55.6% | 0.0% | 33.3% |
| 16²–32² | 134 | 85.8% | 43.3% | 53.0% |
| 32²–64² | 377 | 94.2% | 70.0% | 75.1% |
| 64²–96² | 327 | 96.0% | 74.0% | 79.2% |
| 96²–192² | 698 | 97.6% | 80.7% | 84.5% |
| ≥192² | 862 | 99.0% | 91.8% | 91.3% |

Sınıf × boyut AP50-95 (n<5 ise '–'):

| sınıf | n_S | AP_S | n_M | AP_M | n_L | AP_L |
|---|---|---|---|---|---|---|
| aeroplane | 0 | – | 21 | 0.53 | 59 | 0.77 |
| bicycle | 0 | – | 12 | 0.38 | 50 | 0.76 |
| bird | 3 | – | 36 | 0.35 | 52 | 0.77 |
| boat | 4 | – | 25 | 0.51 | 35 | 0.64 |
| bottle | 13 | 0.13 | 52 | 0.51 | 30 | 0.73 |
| bus | 3 | – | 13 | 0.46 | 39 | 0.88 |
| car | 38 | 0.31 | 80 | 0.65 | 104 | 0.76 |
| cat | 0 | – | 2 | – | 86 | 0.79 |
| chair | 3 | – | 79 | 0.44 | 95 | 0.49 |
| cow | 0 | – | 25 | 0.66 | 31 | 0.74 |
| diningtable | 0 | – | 1 | – | 48 | 0.58 |
| dog | 0 | – | 10 | 0.72 | 86 | 0.78 |
| horse | 3 | – | 15 | 0.23 | 48 | 0.76 |
| motorbike | 3 | – | 10 | 0.48 | 54 | 0.69 |
| person | 51 | 0.21 | 215 | 0.51 | 519 | 0.72 |
| pottedplant | 12 | 0.16 | 57 | 0.31 | 40 | 0.37 |
| sheep | 8 | 0.39 | 16 | 0.61 | 33 | 0.86 |
| sofa | 0 | – | 1 | – | 58 | 0.62 |
| train | 0 | – | 4 | – | 53 | 0.76 |
| tvmonitor | 2 | – | 30 | 0.58 | 40 | 0.64 |

Küçük nesneler (n=143, val GT'nin %5.9'i): AP50-95 0.225 vs büyük 0.704 (fark 47.9 puan); <e-posta> 0.728 vs 0.977. Not: ölçüm orijinal piksel; eğitim/çıkarım 640 px letterbox olduğundan (~1.3× büyütme) S nesneler modelde ≈<41 px görünür. Val'de S örneği az olduğundan S-AP güven aralığı geniştir.

## 3. Hata türü ayrıştırması (TIDE benzeri)

Sayılar: tüm tahminler (conf≥0.001) ve işletim noktası (skor≥0.25). `Miss` tüm conf'ta hiçbir kutuyla IoU≥0.5 eşleşmeyen GT; işletim noktası sütununda ise o eşikte bulunamayan GT (eşleşmeyen + düşük skorla eşleşen).

| tür | sayı (tüm conf) | oran | sayı skor≥0.25 |
|---|---|---|---|
| TP | 2323 | 5.3% | 1918 |
| Dupe | 2567 | 5.9% | 93 |
| Cls | 3499 | 8.0% | 91 |
| Loc | 7844 | 17.9% | 171 |
| Both | 5547 | 12.6% | 75 |
| Bkg | 22081 | 50.3% | 265 |
| Miss | 84 | 3.5% (GT'nin) | 489 |

Oracle ΔAP (puan): ilgili hata türü düzeltilirse mAP ne kadar artardı (AP50 ve AP50-95, sınıf ortalaması). Taban: AP50=81.27, AP50-95=63.00.

| düzeltilen | ΔAP50 (puan) | ΔAP50-95 (puan) |
|---|---|---|
| Cls | 2.70 | 1.98 |
| Loc | 2.71 | 5.45 |
| Both | 1.30 | 0.93 |
| Dupe | 0.60 | -0.70 |
| Bkg | 3.49 | 2.28 |
| Miss | 3.28 | 2.42 |
| FP_hepsi | 14.33 | 6.70 |
| Kutu_kusursuz | 0.00 | 18.27 |
| TP_IoU_sirali | 1.38 | 5.47 |

Protokol etkisi: val'de etiketlerden çıkarılmış difficult nesne sayısı 493 (XML). Bunlardan biriyle aynı sınıf IoU≥0.5 örtüşen ve eşleşmemiş (FP sayılan) tahmin: 873 (skor≥0.25: 131; işletim noktası FP'lerinin 18.8%'i). Bu tahminler devkit gibi yok sayılsaydı ΔAP50 = +1.49, ΔAP50-95 = +1.04 puan (Ultralytics protokolünün val üzerinde ölçülmüş cezası; makalede belirtilmeli, Bkg/Loc/Dupe sayılarının bir kısmı bu yüzden yapaydır).


Not: `Dupe` ve `Loc` ΔAP50-95 değerleri IoU=0.5 tanımına göredir; bu kutular daha yüksek IoU eşiklerinde TP olabildiği için silme oracle'ı (Dupe) AP50-95'i düşürebilir; Loc oracle'ı kutuyu GT'ye yapıştırdığı için AP50-95 kazancı AP50'den büyüktür.


Hata türü × boyut (işletim noktası; tam uzun biçim `hata-kirilim.csv`):

| tür | S | M | L |
|---|---|---|---|
| TP (skor≥0.25, tahmin kutusu alanı) | 57 | 493 | 1368 |
| Dupe (skor≥0.25, tahmin kutusu alanı) | 0 | 32 | 61 |
| Cls (skor≥0.25, tahmin kutusu alanı) | 1 | 14 | 76 |
| Loc (skor≥0.25, tahmin kutusu alanı) | 6 | 69 | 96 |
| Both (skor≥0.25, tahmin kutusu alanı) | 2 | 18 | 55 |
| Bkg (skor≥0.25, tahmin kutusu alanı) | 55 | 119 | 91 |
| Miss (skor≥0.25 bulunamayan GT, GT alanı) | 85 | 198 | 206 |
| GT sayısı | 143 | 704 | 1560 |

AP50'ye en çok mal olan hata türü: Bkg (+3.49), sonra Miss (+3.28) ve Loc (+2.71). Oracle'lar birbirinden bağımsız ölçüldü, toplamları ΔAP'ye eşit değildir (etkileşim). `Kutu_kusursuz` = tüm TP kutuları GT'ye yapıştırılırsa AP50-95'in erişeceği üst sınır; `TP_IoU_sirali` = yalnız TP'lerin skorları IoU sırasına göre dağıtılırsa (FP skorları aynı) kazanç, yani IoU-farkında skorlamanın (VFL/QFL/IoU-aware) teorik payı.


Sınıf başına ΔAP50 (puan, oracle):

| sınıf | Cls | Loc | Both | Dupe | Bkg | Miss |
|---|---|---|---|---|---|---|
| aeroplane | 0.3 | 2.3 | 0.1 | 0.5 | 1.0 | 3.6 |
| bicycle | 2.6 | 1.4 | 0.8 | 0.4 | 0.5 | 3.1 |
| bird | 2.9 | 4.9 | 0.4 | 1.2 | 1.6 | 7.5 |
| boat | 0.2 | 4.6 | 0.3 | 0.7 | 5.5 | 3.9 |
| bottle | -0.1 | 2.9 | 0.3 | 0.4 | 9.9 | 5.5 |
| bus | 1.6 | 0.5 | 0.3 | 0.6 | 2.6 | 3.4 |
| car | 1.1 | 1.8 | 0.7 | 0.5 | 3.7 | 3.2 |
| cat | 3.0 | 2.8 | 0.3 | 0.3 | 0.3 | 0.0 |
| chair | 3.8 | 3.6 | 2.2 | 0.8 | 7.1 | 2.9 |
| cow | 4.4 | 2.0 | 0.0 | 0.7 | 3.5 | 1.7 |
| diningtable | 1.9 | 1.5 | 12.6 | 0.6 | 2.9 | 3.2 |
| dog | 2.7 | 0.4 | 0.3 | 0.2 | 0.3 | 1.1 |
| horse | 9.9 | 1.1 | 0.2 | 0.8 | 0.7 | 5.5 |
| motorbike | 3.3 | 2.8 | 0.9 | 0.2 | 3.1 | 2.3 |
| person | 0.2 | 3.0 | 0.2 | 1.0 | 2.2 | 1.5 |
| pottedplant | 0.8 | 10.3 | 1.6 | 1.0 | 8.2 | 7.5 |
| sheep | 5.8 | 0.0 | 0.1 | 0.2 | 4.0 | 0.0 |
| sofa | 6.7 | 3.0 | 3.3 | 1.8 | 4.4 | 1.2 |
| train | 2.0 | 4.4 | 0.2 | 0.1 | 2.2 | 5.3 |
| tvmonitor | 0.5 | 1.1 | 1.2 | 0.1 | 6.2 | 3.6 |

Kaçırılan GT (84) alt kırılımı: yalnız başka sınıfla örtüşen kutu var (sınıf karışıklığı) 24; aynı sınıf kutu var ama IoU∈[0.1,0.5) (lokalizasyon) 30; aynı sınıf IoU≥0.5 kutu var ama başka GT'ye atanmış (atama çakışması/örtüşme) 2; hiç kutu yok (gerçek kaçırma) 28 (kalan sınıflama kesin değil: bu üç grup kesişebilir).

## 4. Lokalizasyon kalitesi

| IoU aralığı | TP (tüm conf) | oran | TP skor≥0.25 | oran |
|---|---|---|---|---|
| 0.50–0.55 | 71 | 3.1% | 33 | 1.7% |
| 0.55–0.60 | 96 | 4.1% | 56 | 2.9% |
| 0.60–0.65 | 87 | 3.7% | 42 | 2.2% |
| 0.65–0.70 | 113 | 4.9% | 68 | 3.5% |
| 0.70–0.75 | 132 | 5.7% | 82 | 4.3% |
| 0.75–0.80 | 194 | 8.4% | 150 | 7.8% |
| 0.80–0.85 | 200 | 8.6% | 160 | 8.3% |
| 0.85–0.90 | 343 | 14.8% | 303 | 15.8% |
| 0.90–0.95 | 528 | 22.7% | 485 | 25.3% |
| 0.95–1.00 | 559 | 24.1% | 539 | 28.1% |

TP IoU ortalaması 0.847 (skor≥0.25: 0.870); işletim noktası TP'lerinin %14.7'i IoU<0.75. TP'lerde skor–IoU Spearman ρ=0.63 (skorun konum kalitesini ne kadar taşıdığı).

Sınıf başına AP50 − AP75 (büyükten küçüğe) ve TP ort. IoU:

| sınıf | AP50 | AP75 | AP50−AP75 (puan) | TP ort. IoU |
|---|---|---|---|---|
| pottedplant | 0.557 | 0.243 | 31.4 | 0.768 |
| person | 0.891 | 0.687 | 20.5 | 0.834 |
| aeroplane | 0.903 | 0.717 | 18.5 | 0.862 |
| horse | 0.839 | 0.667 | 17.2 | 0.846 |
| motorbike | 0.829 | 0.665 | 16.5 | 0.837 |
| bird | 0.792 | 0.628 | 16.5 | 0.843 |
| car | 0.844 | 0.722 | 12.2 | 0.847 |
| bicycle | 0.886 | 0.767 | 11.9 | 0.865 |
| chair | 0.624 | 0.507 | 11.7 | 0.823 |
| boat | 0.766 | 0.650 | 11.6 | 0.829 |
| sofa | 0.744 | 0.631 | 11.3 | 0.886 |
| sheep | 0.893 | 0.783 | 11.0 | 0.881 |
| cow | 0.874 | 0.776 | 9.8 | 0.873 |
| tvmonitor | 0.827 | 0.732 | 9.5 | 0.846 |
| cat | 0.918 | 0.834 | 8.3 | 0.900 |
| train | 0.888 | 0.808 | 8.0 | 0.891 |
| dog | 0.936 | 0.864 | 7.1 | 0.899 |
| bus | 0.851 | 0.785 | 6.6 | 0.892 |
| bottle | 0.686 | 0.622 | 6.5 | 0.842 |
| diningtable | 0.706 | 0.656 | 5.0 | 0.889 |

AP50→AP75 düşüşü sınıf ortalaması 12.6 puan; en büyük düşüş pottedplant (31), person (20), aeroplane (19). Kutu-regresyon (IoU/DFL/box kazancı) manipülasyonlarının hedef alanı budur.

## 5. Sınıf karışıklığı

İşletim noktası (skor≥0.25, IoU≥0.5): her GT için eşleşen en yüksek skorlu kutu (doğru sınıf varsa o). Matris dosyası: `hata-analizi_20261002-184450_h02_innerciou_s1_karisiklik.csv`.

| GT sınıfı → tahmin | sayı | GT'nin payı |
|---|---|---|
| chair → person | 7 | 4.0% |
| horse → sheep | 5 | 7.6% |
| cow → horse | 4 | 7.1% |
| cow → sheep | 3 | 5.4% |
| dog → cat | 3 | 3.1% |
| sofa → chair | 3 | 5.1% |
| sofa → person | 3 | 5.1% |
| car → train | 2 | 0.9% |
| motorbike → bicycle | 2 | 3.0% |
| motorbike → person | 2 | 3.0% |
| bicycle → motorbike | 1 | 1.6% |
| bicycle → person | 1 | 1.6% |

Çift bazında (yönsüz) toplam: chair/person 7; horse/sheep 5; cow/horse 4; cat/dog 4; chair/sofa 4; cow/sheep 3; person/sofa 3; bicycle/motorbike 3.


Tüm-conf alternatif görünüm: `Cls` türü tahminler (IoU≥0.5 başka sınıf GT, skor≥0.05), GT sınıfı → tahmin:

| GT → tahmin | tahmin sayısı |
|---|---|
| person → chair | 27 |
| sofa → chair | 26 |
| person → sofa | 23 |
| dog → cat | 14 |
| motorbike → bicycle | 12 |
| chair → sofa | 10 |
| cow → sheep | 9 |
| chair → diningtable | 9 |
| horse → sheep | 9 |
| sheep → dog | 9 |

Kanonik çiftler (yönsüz; `Cls` tahmin sayısı skor≥0.05 | işletim noktası GT sayısı): cat/dog 22 | 4; cow/sheep 17 | 3; chair/diningtable 11 | 1; chair/sofa 36 | 4; horse/cow 14 | 4; bus/train 4 | 2; motorbike/bicycle 17 | 3.

İşletim noktasında GT'lerin 2.3%'i başka sınıf olarak etiketlendi (56/2407); 17.5%'i hiç bulunamadı. Sınıf karışıklığı oracle ΔAP50 = 2.70 puan (§3; Cls, altı hata türü içinde 4. sırada); işletim noktasında karışan GT oranı yalnız 2.3%, yani sorun sınıf ayrımından çok kutu/skor düzeyindedir.

## 6. Kalabalık ve örtüşme

| nesne/görüntü | görüntü | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) | FP/görüntü (skor≥0.25) |
|---|---|---|---|---|---|---|
| 1 | 440 | 440 | 99.3% | 92.0% | 92.7% | 0.37 |
| 2 | 252 | 504 | 96.8% | 84.1% | 86.1% | 0.48 |
| 3–4 | 195 | 654 | 96.2% | 77.7% | 83.5% | 1.01 |
| 5–8 | 88 | 511 | 95.1% | 71.2% | 76.9% | 1.47 |
| ≥9 | 25 | 298 | 95.0% | 72.8% | 71.1% | 3.36 |

Aynı sınıftan başka GT ile en yüksek IoU (örtüşme) gruplarına göre recall:

| komşu GT IoU | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 |
|---|---|---|---|
| <0.1 (komşu yok) | 1908 | 96.5% | 80.5% |
| 0.1–0.3 | 357 | 96.1% | 77.6% |
| 0.3–0.5 | 105 | 99.0% | 77.1% |
| 0.5–0.7 | 33 | 90.9% | 66.7% |
| ≥0.7 | 4 | 100.0% | 75.0% |

Aynı sınıf GT çiftleri (IoU≥0.3): 75 çift; ikisi de bulundu 72, yalnız biri 1, hiçbiri 2. IoU≥0.7 (NMS eşiği üstü, doğrudan NMS etkilenir): 2 çift, ikisi de bulundu 2, yalnız biri 0, hiçbiri 0. Kaçırılan GT'lerin 2/84'i aynı sınıf IoU≥0.5 bir kutuya sahip ama o kutu başka GT'ye atanmış (olası NMS/atama kaybı).

Kalabalıkla recall düşüşü: tek nesneli görüntüde 92.0% iken ≥9 nesnelide 72.8% (skor≥0.25); tüm-conf recall 99.3% → 95.0%, <e-posta> 92.7% → 71.1%. Çiftlerde tek-bulunma ve atama çakışması sayıları küçükse kayıp NMS'ten değil düşük skor/zayıf konumdan gelir.

## 7. Kalibrasyon (skor vs gerçek precision)

Tahmin 'doğru' = IoU≥0.5 ile eşleşen (<e-posta>); ayrıca IoU≥0.75 için precision. Not: YOLO sınıf skoru olasılık değil, TAL hizalama hedefli (IoU'ya bağlı) bir skordur; düşük skor bölgesindeki düşük precision 'aday havuzu' doğasından gelir.

| skor aralığı | tahmin | ort. skor | <e-posta> | <e-posta> | TP ort. IoU | skor−precision |
|---|---|---|---|---|---|---|
| 0.0–0.1 | 40191 | 0.008 | 0.007 | 0.006 | 0.724 | 0.001 |
| 0.1–0.2 | 836 | 0.142 | 0.127 | 0.093 | 0.756 | 0.015 |
| 0.2–0.3 | 381 | 0.245 | 0.163 | 0.121 | 0.762 | 0.083 |
| 0.3–0.4 | 251 | 0.348 | 0.291 | 0.215 | 0.779 | 0.057 |
| 0.4–0.5 | 219 | 0.447 | 0.443 | 0.306 | 0.778 | 0.004 |
| 0.5–0.6 | 226 | 0.549 | 0.562 | 0.434 | 0.811 | -0.013 |
| 0.6–0.7 | 242 | 0.648 | 0.665 | 0.504 | 0.809 | -0.017 |
| 0.7–0.8 | 340 | 0.752 | 0.862 | 0.674 | 0.836 | -0.110 |
| 0.8–0.9 | 620 | 0.857 | 0.947 | 0.856 | 0.884 | -0.090 |
| 0.9–1.0 | 555 | 0.926 | 0.991 | 0.968 | 0.941 | -0.065 |

ECE (tüm tahminler, 10 eşit genişlikli kutu) = 0.006; ECE (skor≥0.1) = 0.053. `skor−precision` pozitifse aşırı güvenli, negatifse eksik güvenli. 
Skor≥0.5 tahminlerin <e-posta> = 86.6%, <e-posta> = 76.5% (fark 10.1 puan). Skor 0.5 üstü kutularda (skor−precision) ortalaması -0.059: model yüksek skorlarda eksik güvenli (skor < gerçek <e-posta>), sıralama iyi; asıl boşluk <e-posta> ile @0.75 arasındaki konum kalitesidir. TP'lerde skor–IoU Spearman ρ=0.63: skor konum kalitesini kısmen taşıyor. TP_IoU_sirali oracle'ı (§3) IoU-farkında skorlamanın teorik payını sayısallaştırıyor.

## 8. Eğitim dinamiği (results.csv)

| gösterge | değer |
|---|---|
| epoch sayısı | 100 |
| en iyi epoch (mAP50-95) | 96 (0.6342) |
| son epoch mAP50-95 | 0.6333 |
| epoch 50 / 80 / 90 / 100 | 0.6099 / 0.6309 / 0.6334 / 0.6333 |
| son 20 epoch eğimi (puan/10 epoch) | 0.077 |
| önceki 20 epoch (ep 71-90) eğimi | 0.303 |
| son 20 epoch std (puan) | 0.061 |
| ep 81-90 ort. vs ep 91-100 ort. (close_mosaic=10) | 0.6327 vs 0.6333 (Δ +0.06 puan) |
| mAP50 son epoch | 0.8172 |
| results.csv en iyi ep mAP50-95 vs metrics.json (best.pt yeniden val) | 0.6342 vs 0.6306 (fark 0.35 puan; ölçüm-protokolü gürültü tabanı) |

Loss tablosu (train loss mosaic açıkken ep1/ep90, kapalıyken ep100; val/train oranı):

| loss | train ep1 | train ep90 | train ep100 | val ep100 | val/train ep100 | val/train ep90 | train loss ep90→91 değişim |
|---|---|---|---|---|---|---|---|
| box | 1.186 | 0.947 | 0.806 | 1.008 | 1.25 | 1.06 | -8.4% |
| cls | 2.336 | 0.822 | 0.553 | 0.780 | 1.41 | 0.95 | -22.2% |
| dfl | 1.290 | 1.155 | 1.067 | 1.164 | 1.09 | 1.01 | -4.0% |

Kayıp payı (results.csv loss_items kazançlı; loss.py:469-471 — önceki sürüm kazançları iki kez uyguluyordu, 2026-10-01 düzeltildi) ep100: box 33.2%, cls 22.8%, dfl 44.0% (ep1: box 24.7%, cls 48.5%, dfl 26.8%). Son 20 epoch mAP eğimi +0.077 puan/10 ep (std 0.06) → doygun (eğim seed gürültüsü altında). close_mosaic sonrası (ep 91-100) ortalama +0.06 puan. Train loss mosaic kapanınca ani düşer (cls -22.2%); bu rejim değişimi, train/val farkının bir kısmının augmentation kaynaklı olduğunu gösterir. val cls loss son 20 epoch 0.789→0.780 (yükselmiyor): val loss bozulmadığı için klasik aşırı öğrenme yok, ama mosaic kapalıyken val/train oranı cls 1.41, box 1.25: train kaybı val kaybından ayrışıyor (genelleme açığı); val mAP doygun.

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
| ön-işleme ms | 1.2629 |
| çıkarım ms | 6.6803 |
| NMS/son-işlem ms | 0.6892 |
| toplam ms | 8.6325 |
| FPS | 115.842 |
| batch | 1 |
| FP16 (1=evet) | 1.0 |
| ölçüm görüntüsü | 300 |
| ısınma görüntüsü | 50 |
| eğitim süresi (saat) | 2.959 |

Varyant hız bütçesi (yalnız tahmin; gerçek ölçüm varyant koşumunda yapılır):

| varyant sınıfı | çarpan | tahmini toplam ms | tahmini FPS | not |
|---|---|---|---|---|
| eğitim-zamanı (aug/loss) varyantları | aynı | 8.63 | 116 | çıkarım grafiği değişmez; hız bütçesi ≈ 0 |
| Soft-NMS/DIoU-NMS (son-işlem ×1–5 varsayımı) | NMS ×1–5 | 8.63–11.39 | 88–116 | tahmin (ölçülmedi) |
| TTA (Ultralytics augment, ~3 ölçek+flip; literatür çarpanı 3.6× çıkarım) | çıkarım ×3.6 (literatür) | 26.00 | 38 | tahmin; literatur-cipa.md §4 (Ultralytics TTA dokümanı) |
| YOLO11s'e geçiş (bağlam satırı) | ≈3.4× GFLOPs | – | – | ölçülmedi |

Baseline batch=1 FP16 toplam 8.63 ms (116 FPS; çıkarım 6.68 ms, NMS 0.69 ms = toplamın %8'i). 'Gerçek zamanlı' için ≥30 FPS kabul edilirse (33 ms) ~3.9× gecikme payı vardır; TTA bile (~38 FPS tahmini) eşiğin üstünde kalabilir, ancak hız-doğruluk satırı ayrı raporlanmalıdır. Not: hız ölçümü GPU'nun paylaşıldığı bir anda alınmış olabilir (eğitim kuyruğu); karşılaştırmalı hız ölçümü aynı koşulda yapılmalıdır.
