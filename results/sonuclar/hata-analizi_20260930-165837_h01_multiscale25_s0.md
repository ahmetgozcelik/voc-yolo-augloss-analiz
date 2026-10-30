# Hata analizi (K3.5) — baseline `20260930-165837_h01_multiscale25_s0`

*Üretici: `scripts/analiz/hata_analizi.py` · yalnız VAL (1000 görüntü, 2407 nesne; difficult nesneler etiketlerde yok) · tahmin sayısı 42049 (conf=0.001, max_det=300, NMS IoU=0.7) · eşleştirme: pycocotools COCOeval · işletim noktası skor≥0.25.*

**Sızıntı notu:** Bu analizde test kümesi (VOC07 test), `rep_test*` dosyaları ve `metrics.json` içindeki `rep_*` anahtarları hiç kullanılmadı; betik bu anahtarları yükler yüklemez atar. Buradan çıkan hipotezler yalnız val bilgisine dayanır.

## 0. Genel ölçüler ve doğrulama

| ölçü | pycocotools (bu betik) | Ultralytics metrics.json (best.pt val) | fark |
|---|---|---|---|
| mAP50-95 | 0.6292 | 0.6296 | -0.0005 |
| mAP50 | 0.8138 | 0.8159 | -0.0021 |
| mAP75 (yalnız pycocotools) | 0.6845 | – | – |

AP50 ile AP50-95 arasındaki fark 18.5 puan; AP50→AP75 farkı 12.9 puan. pycocotools ile Ultralytics AP'si arasındaki küçük fark interpolasyon/eşleştirme (Ultralytics kendi IoU-matrisi eşleşmesini kullanır) kaynaklıdır.

## 1. Sınıf başına AP ve sınıf frekansı

| sınıf (AP50-95 artan) | AP50-95 | AP50 | AP75 | AP50−AP75 | train nesne | val nesne | val GT medyan kenar px |
|---|---|---|---|---|---|---|---|
| pottedplant | 0.288 | 0.544 | 0.252 | 0.292 | 1378 | 109 | 71 |
| chair | 0.473 | 0.659 | 0.528 | 0.131 | 2975 | 177 | 99 |
| boat | 0.536 | 0.785 | 0.613 | 0.172 | 1076 | 64 | 112 |
| bottle | 0.539 | 0.709 | 0.602 | 0.107 | 1669 | 95 | 70 |
| diningtable | 0.576 | 0.718 | 0.624 | 0.094 | 775 | 49 | 211 |
| tvmonitor | 0.589 | 0.786 | 0.713 | 0.074 | 1036 | 72 | 108 |
| sofa | 0.595 | 0.729 | 0.625 | 0.104 | 755 | 59 | 237 |
| bird | 0.596 | 0.787 | 0.667 | 0.120 | 1514 | 91 | 115 |
| motorbike | 0.624 | 0.831 | 0.666 | 0.165 | 985 | 67 | 239 |
| horse | 0.625 | 0.841 | 0.665 | 0.176 | 1006 | 66 | 209 |
| person | 0.637 | 0.888 | 0.681 | 0.207 | 12471 | 785 | 141 |
| car | 0.647 | 0.849 | 0.706 | 0.143 | 3045 | 222 | 92 |
| cow | 0.675 | 0.845 | 0.790 | 0.055 | 791 | 56 | 145 |
| bicycle | 0.701 | 0.901 | 0.749 | 0.152 | 1002 | 62 | 171 |
| aeroplane | 0.701 | 0.897 | 0.715 | 0.182 | 1091 | 80 | 184 |
| bus | 0.722 | 0.837 | 0.773 | 0.064 | 767 | 55 | 192 |
| sheep | 0.751 | 0.908 | 0.812 | 0.097 | 1013 | 57 | 119 |
| cat | 0.764 | 0.915 | 0.813 | 0.102 | 1505 | 88 | 278 |
| train | 0.768 | 0.912 | 0.847 | 0.065 | 868 | 57 | 225 |
| dog | 0.775 | 0.935 | 0.850 | 0.085 | 1929 | 96 | 211 |

En zayıf 5 sınıf (AP50-95): pottedplant 0.288, chair 0.473, boat 0.536, bottle 0.539, diningtable 0.576. Sınıf frekansı (train nesne) ile AP50-95 Spearman ρ=-0.11 (permütasyon p=0.648), AP50 ile ρ=0.06; sınıf başına train küçük-nesne oranı ile ρ=-0.38; val GT medyan kenar uzunluğu ile ρ=0.54 (n=20 sınıf, gürültülü). Frekans AP'nin zayıf belirleyicisi (|ρ|<0.3): zayıf sınıflarda sorun veri azlığından çok nesne boyutu/görünümü/bağlamıdır.

## 2. Nesne boyutu kırılımı (orijinal çözünürlük; COCO S<32²≤M<96²≤L)

| boyut | GT sayısı | AP50-95 | AP50 | <e-posta> (tüm conf, sınıf ort.) | <e-posta>:0.95 (sınıf ort.) | <e-posta> skor≥0.25 |
|---|---|---|---|---|---|---|
| all | 2407 | 0.629 | 0.814 | 0.965 | 0.770 | 0.795 |
| S | 143 | 0.208 | 0.366 | 0.763 | 0.475 | 0.392 |
| M | 704 | 0.459 | 0.663 | 0.941 | 0.681 | 0.699 |
| L | 1560 | 0.703 | 0.871 | 0.978 | 0.815 | 0.876 |

İnce alan aralıkları (GT düzeyi, eşleşme bayrağı):

| alan (px²) | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) |
|---|---|---|---|---|
| <16² | 9 | 44.4% | 0.0% | 22.2% |
| 16²–32² | 134 | 87.3% | 41.8% | 56.0% |
| 32²–64² | 377 | 96.0% | 66.3% | 74.5% |
| 64²–96² | 327 | 96.0% | 74.0% | 81.7% |
| 96²–192² | 698 | 97.9% | 81.2% | 83.7% |
| ≥192² | 862 | 98.6% | 92.7% | 91.1% |

Sınıf × boyut AP50-95 (n<5 ise '–'):

| sınıf | n_S | AP_S | n_M | AP_M | n_L | AP_L |
|---|---|---|---|---|---|---|
| aeroplane | 0 | – | 21 | 0.56 | 59 | 0.74 |
| bicycle | 0 | – | 12 | 0.41 | 50 | 0.78 |
| bird | 3 | – | 36 | 0.34 | 52 | 0.77 |
| boat | 4 | – | 25 | 0.56 | 35 | 0.59 |
| bottle | 13 | 0.17 | 52 | 0.51 | 30 | 0.74 |
| bus | 3 | – | 13 | 0.47 | 39 | 0.86 |
| car | 38 | 0.32 | 80 | 0.63 | 104 | 0.77 |
| cat | 0 | – | 2 | – | 86 | 0.78 |
| chair | 3 | – | 79 | 0.45 | 95 | 0.51 |
| cow | 0 | – | 25 | 0.66 | 31 | 0.72 |
| diningtable | 0 | – | 1 | – | 48 | 0.58 |
| dog | 0 | – | 10 | 0.68 | 86 | 0.79 |
| horse | 3 | – | 15 | 0.25 | 48 | 0.76 |
| motorbike | 3 | – | 10 | 0.49 | 54 | 0.68 |
| person | 51 | 0.21 | 215 | 0.50 | 519 | 0.73 |
| pottedplant | 12 | 0.20 | 57 | 0.29 | 40 | 0.34 |
| sheep | 8 | 0.34 | 16 | 0.68 | 33 | 0.87 |
| sofa | 0 | – | 1 | – | 58 | 0.60 |
| train | 0 | – | 4 | – | 53 | 0.79 |
| tvmonitor | 2 | – | 30 | 0.52 | 40 | 0.66 |

Küçük nesneler (n=143, val GT'nin %5.9'i): AP50-95 0.208 vs büyük 0.703 (fark 49.5 puan); <e-posta> 0.763 vs 0.978. Not: ölçüm orijinal piksel; eğitim/çıkarım 640 px letterbox olduğundan (~1.3× büyütme) S nesneler modelde ≈<41 px görünür. Val'de S örneği az olduğundan S-AP güven aralığı geniştir.

## 3. Hata türü ayrıştırması (TIDE benzeri)

Sayılar: tüm tahminler (conf≥0.001) ve işletim noktası (skor≥0.25). `Miss` tüm conf'ta hiçbir kutuyla IoU≥0.5 eşleşmeyen GT; işletim noktası sütununda ise o eşikte bulunamayan GT (eşleşmeyen + düşük skorla eşleşen).

| tür | sayı (tüm conf) | oran | sayı skor≥0.25 |
|---|---|---|---|
| TP | 2330 | 5.5% | 1914 |
| Dupe | 2446 | 5.8% | 97 |
| Cls | 3391 | 8.1% | 90 |
| Loc | 7708 | 18.3% | 163 |
| Both | 5261 | 12.5% | 77 |
| Bkg | 20913 | 49.7% | 274 |
| Miss | 77 | 3.2% (GT'nin) | 493 |

Oracle ΔAP (puan): ilgili hata türü düzeltilirse mAP ne kadar artardı (AP50 ve AP50-95, sınıf ortalaması). Taban: AP50=81.38, AP50-95=62.92.

| düzeltilen | ΔAP50 (puan) | ΔAP50-95 (puan) |
|---|---|---|
| Cls | 2.47 | 1.93 |
| Loc | 2.52 | 5.32 |
| Both | 1.16 | 0.79 |
| Dupe | 0.64 | -0.72 |
| Bkg | 3.54 | 2.24 |
| Miss | 2.82 | 2.07 |
| FP_hepsi | 14.71 | 6.78 |
| Kutu_kusursuz | 0.00 | 18.46 |
| TP_IoU_sirali | 1.55 | 5.87 |

Protokol etkisi: val'de etiketlerden çıkarılmış difficult nesne sayısı 493 (XML). Bunlardan biriyle aynı sınıf IoU≥0.5 örtüşen ve eşleşmemiş (FP sayılan) tahmin: 868 (skor≥0.25: 119; işletim noktası FP'lerinin 17.0%'i). Bu tahminler devkit gibi yok sayılsaydı ΔAP50 = +1.28, ΔAP50-95 = +0.86 puan (Ultralytics protokolünün val üzerinde ölçülmüş cezası; makalede belirtilmeli, Bkg/Loc/Dupe sayılarının bir kısmı bu yüzden yapaydır).


Not: `Dupe` ve `Loc` ΔAP50-95 değerleri IoU=0.5 tanımına göredir; bu kutular daha yüksek IoU eşiklerinde TP olabildiği için silme oracle'ı (Dupe) AP50-95'i düşürebilir; Loc oracle'ı kutuyu GT'ye yapıştırdığı için AP50-95 kazancı AP50'den büyüktür.


Hata türü × boyut (işletim noktası; tam uzun biçim `hata-kirilim.csv`):

| tür | S | M | L |
|---|---|---|---|
| TP (skor≥0.25, tahmin kutusu alanı) | 54 | 489 | 1371 |
| Dupe (skor≥0.25, tahmin kutusu alanı) | 0 | 27 | 70 |
| Cls (skor≥0.25, tahmin kutusu alanı) | 1 | 22 | 67 |
| Loc (skor≥0.25, tahmin kutusu alanı) | 8 | 65 | 90 |
| Both (skor≥0.25, tahmin kutusu alanı) | 3 | 15 | 59 |
| Bkg (skor≥0.25, tahmin kutusu alanı) | 57 | 122 | 95 |
| Miss (skor≥0.25 bulunamayan GT, GT alanı) | 87 | 212 | 194 |
| GT sayısı | 143 | 704 | 1560 |

AP50'ye en çok mal olan hata türü: Bkg (+3.54), sonra Miss (+2.82) ve Loc (+2.52). Oracle'lar birbirinden bağımsız ölçüldü, toplamları ΔAP'ye eşit değildir (etkileşim). `Kutu_kusursuz` = tüm TP kutuları GT'ye yapıştırılırsa AP50-95'in erişeceği üst sınır; `TP_IoU_sirali` = yalnız TP'lerin skorları IoU sırasına göre dağıtılırsa (FP skorları aynı) kazanç, yani IoU-farkında skorlamanın (VFL/QFL/IoU-aware) teorik payı.


Sınıf başına ΔAP50 (puan, oracle):

| sınıf | Cls | Loc | Both | Dupe | Bkg | Miss |
|---|---|---|---|---|---|---|
| aeroplane | 0.7 | 1.6 | 0.8 | 0.3 | 1.0 | 2.4 |
| bicycle | 2.0 | 0.3 | 0.9 | 0.3 | 0.5 | 3.1 |
| bird | 2.1 | 3.7 | 0.4 | 1.4 | 3.2 | 4.5 |
| boat | 0.2 | 3.8 | 0.2 | 0.9 | 5.9 | 4.0 |
| bottle | 1.1 | 1.6 | 0.2 | 0.4 | 11.3 | 1.6 |
| bus | 4.2 | 0.7 | 0.2 | 0.5 | 1.8 | 4.8 |
| car | 1.5 | 1.6 | 0.4 | 0.8 | 3.4 | 2.8 |
| cat | 3.2 | 2.4 | 0.1 | 0.2 | 0.1 | 3.2 |
| chair | 2.6 | 3.9 | 1.8 | 0.6 | 7.3 | 1.9 |
| cow | 3.9 | 3.1 | 0.1 | 0.1 | 3.9 | 1.6 |
| diningtable | 1.1 | 1.9 | 9.3 | 0.8 | 2.1 | 4.8 |
| dog | 1.2 | 0.9 | 0.4 | 0.3 | 0.9 | 0.0 |
| horse | 8.8 | 0.7 | 0.7 | 0.7 | 1.3 | 2.7 |
| motorbike | 2.5 | 3.9 | 0.9 | 0.2 | 2.1 | 3.7 |
| person | 0.3 | 2.5 | 0.2 | 0.9 | 2.1 | 2.1 |
| pottedplant | 0.4 | 11.1 | 2.1 | 1.3 | 7.2 | 7.9 |
| sheep | 4.9 | 0.0 | 0.4 | 0.0 | 2.8 | 0.0 |
| sofa | 7.8 | 0.7 | 2.8 | 2.8 | 4.3 | 1.3 |
| train | 0.3 | 3.9 | 0.4 | 0.2 | 1.6 | 1.9 |
| tvmonitor | 0.6 | 2.5 | 1.2 | 0.1 | 8.0 | 2.2 |

Kaçırılan GT (77) alt kırılımı: yalnız başka sınıfla örtüşen kutu var (sınıf karışıklığı) 16; aynı sınıf kutu var ama IoU∈[0.1,0.5) (lokalizasyon) 33; aynı sınıf IoU≥0.5 kutu var ama başka GT'ye atanmış (atama çakışması/örtüşme) 3; hiç kutu yok (gerçek kaçırma) 25 (kalan sınıflama kesin değil: bu üç grup kesişebilir).

## 4. Lokalizasyon kalitesi

| IoU aralığı | TP (tüm conf) | oran | TP skor≥0.25 | oran |
|---|---|---|---|---|
| 0.50–0.55 | 76 | 3.3% | 37 | 1.9% |
| 0.55–0.60 | 90 | 3.9% | 45 | 2.4% |
| 0.60–0.65 | 101 | 4.3% | 53 | 2.8% |
| 0.65–0.70 | 113 | 4.8% | 76 | 4.0% |
| 0.70–0.75 | 127 | 5.5% | 82 | 4.3% |
| 0.75–0.80 | 176 | 7.6% | 126 | 6.6% |
| 0.80–0.85 | 247 | 10.6% | 192 | 10.0% |
| 0.85–0.90 | 301 | 12.9% | 260 | 13.6% |
| 0.90–0.95 | 544 | 23.3% | 505 | 26.4% |
| 0.95–1.00 | 555 | 23.8% | 538 | 28.1% |

TP IoU ortalaması 0.845 (skor≥0.25: 0.869); işletim noktası TP'lerinin %15.3'i IoU<0.75. TP'lerde skor–IoU Spearman ρ=0.62 (skorun konum kalitesini ne kadar taşıdığı).

Sınıf başına AP50 − AP75 (büyükten küçüğe) ve TP ort. IoU:

| sınıf | AP50 | AP75 | AP50−AP75 (puan) | TP ort. IoU |
|---|---|---|---|---|
| pottedplant | 0.544 | 0.252 | 29.2 | 0.768 |
| person | 0.888 | 0.681 | 20.7 | 0.832 |
| aeroplane | 0.897 | 0.715 | 18.2 | 0.861 |
| horse | 0.841 | 0.665 | 17.6 | 0.846 |
| boat | 0.785 | 0.613 | 17.2 | 0.816 |
| motorbike | 0.831 | 0.666 | 16.5 | 0.844 |
| bicycle | 0.901 | 0.749 | 15.2 | 0.862 |
| car | 0.849 | 0.706 | 14.3 | 0.841 |
| chair | 0.659 | 0.528 | 13.1 | 0.821 |
| bird | 0.787 | 0.667 | 12.0 | 0.842 |
| bottle | 0.709 | 0.602 | 10.7 | 0.828 |
| sofa | 0.729 | 0.625 | 10.4 | 0.869 |
| cat | 0.915 | 0.813 | 10.2 | 0.908 |
| sheep | 0.908 | 0.812 | 9.7 | 0.892 |
| diningtable | 0.718 | 0.624 | 9.4 | 0.870 |
| dog | 0.935 | 0.850 | 8.5 | 0.897 |
| tvmonitor | 0.786 | 0.713 | 7.4 | 0.861 |
| train | 0.912 | 0.847 | 6.5 | 0.904 |
| bus | 0.837 | 0.773 | 6.4 | 0.902 |
| cow | 0.845 | 0.790 | 5.5 | 0.873 |

AP50→AP75 düşüşü sınıf ortalaması 12.9 puan; en büyük düşüş pottedplant (29), person (21), aeroplane (18). Kutu-regresyon (IoU/DFL/box kazancı) manipülasyonlarının hedef alanı budur.

## 5. Sınıf karışıklığı

İşletim noktası (skor≥0.25, IoU≥0.5): her GT için eşleşen en yüksek skorlu kutu (doğru sınıf varsa o). Matris dosyası: `hata-analizi_20260930-165837_h01_multiscale25_s0_karisiklik.csv`.

| GT sınıfı → tahmin | sayı | GT'nin payı |
|---|---|---|
| chair → person | 6 | 3.4% |
| cow → sheep | 5 | 8.9% |
| horse → sheep | 4 | 6.1% |
| dog → cat | 3 | 3.1% |
| sofa → chair | 3 | 5.1% |
| sofa → person | 3 | 5.1% |
| bus → car | 2 | 3.6% |
| car → train | 2 | 0.9% |
| sheep → dog | 2 | 3.5% |
| aeroplane → bicycle | 1 | 1.2% |
| bicycle → motorbike | 1 | 1.6% |
| bicycle → person | 1 | 1.6% |

Çift bazında (yönsüz) toplam: chair/person 7; cow/sheep 6; horse/sheep 4; cat/dog 4; chair/sofa 3; person/sofa 3; bus/car 2; car/train 2.


Tüm-conf alternatif görünüm: `Cls` türü tahminler (IoU≥0.5 başka sınıf GT, skor≥0.05), GT sınıfı → tahmin:

| GT → tahmin | tahmin sayısı |
|---|---|
| sofa → chair | 30 |
| person → chair | 26 |
| person → sofa | 20 |
| cow → horse | 16 |
| chair → sofa | 12 |
| horse → cow | 12 |
| sheep → cow | 11 |
| dog → cat | 11 |
| horse → sheep | 10 |
| cow → sheep | 8 |

Kanonik çiftler (yönsüz; `Cls` tahmin sayısı skor≥0.05 | işletim noktası GT sayısı): cat/dog 18 | 4; cow/sheep 19 | 6; chair/diningtable 7 | 2; chair/sofa 42 | 3; horse/cow 28 | 2; bus/train 5 | 0; motorbike/bicycle 12 | 2.

İşletim noktasında GT'lerin 2.2%'i başka sınıf olarak etiketlendi (53/2407); 17.7%'i hiç bulunamadı. Sınıf karışıklığı oracle ΔAP50 = 2.47 puan (§3; Cls, altı hata türü içinde 4. sırada); işletim noktasında karışan GT oranı yalnız 2.2%, yani sorun sınıf ayrımından çok kutu/skor düzeyindedir.

## 6. Kalabalık ve örtüşme

| nesne/görüntü | görüntü | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) | FP/görüntü (skor≥0.25) |
|---|---|---|---|---|---|---|
| 1 | 440 | 440 | 98.6% | 93.2% | 91.4% | 0.40 |
| 2 | 252 | 504 | 97.4% | 83.3% | 89.1% | 0.50 |
| 3–4 | 195 | 654 | 96.2% | 77.1% | 82.4% | 0.93 |
| 5–8 | 88 | 511 | 95.9% | 70.5% | 78.1% | 1.50 |
| ≥9 | 25 | 298 | 96.0% | 73.8% | 68.8% | 3.40 |

Aynı sınıftan başka GT ile en yüksek IoU (örtüşme) gruplarına göre recall:

| komşu GT IoU | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 |
|---|---|---|---|
| <0.1 (komşu yok) | 1908 | 96.9% | 80.4% |
| 0.1–0.3 | 357 | 95.8% | 77.3% |
| 0.3–0.5 | 105 | 99.0% | 75.2% |
| 0.5–0.7 | 33 | 97.0% | 66.7% |
| ≥0.7 | 4 | 75.0% | 75.0% |

Aynı sınıf GT çiftleri (IoU≥0.3): 75 çift; ikisi de bulundu 72, yalnız biri 3, hiçbiri 0. IoU≥0.7 (NMS eşiği üstü, doğrudan NMS etkilenir): 2 çift, ikisi de bulundu 1, yalnız biri 1, hiçbiri 0. Kaçırılan GT'lerin 3/77'i aynı sınıf IoU≥0.5 bir kutuya sahip ama o kutu başka GT'ye atanmış (olası NMS/atama kaybı).

Kalabalıkla recall düşüşü: tek nesneli görüntüde 93.2% iken ≥9 nesnelide 73.8% (skor≥0.25); tüm-conf recall 98.6% → 96.0%, <e-posta> 91.4% → 68.8%. Çiftlerde tek-bulunma ve atama çakışması sayıları küçükse kayıp NMS'ten değil düşük skor/zayıf konumdan gelir.

## 7. Kalibrasyon (skor vs gerçek precision)

Tahmin 'doğru' = IoU≥0.5 ile eşleşen (<e-posta>); ayrıca IoU≥0.75 için precision. Not: YOLO sınıf skoru olasılık değil, TAL hizalama hedefli (IoU'ya bağlı) bir skordur; düşük skor bölgesindeki düşük precision 'aday havuzu' doğasından gelir.

| skor aralığı | tahmin | ort. skor | <e-posta> | <e-posta> | TP ort. IoU | skor−precision |
|---|---|---|---|---|---|---|
| 0.0–0.1 | 38375 | 0.008 | 0.007 | 0.006 | 0.730 | 0.001 |
| 0.1–0.2 | 809 | 0.139 | 0.120 | 0.068 | 0.730 | 0.019 |
| 0.2–0.3 | 462 | 0.247 | 0.234 | 0.156 | 0.760 | 0.013 |
| 0.3–0.4 | 296 | 0.350 | 0.304 | 0.226 | 0.769 | 0.046 |
| 0.4–0.5 | 190 | 0.449 | 0.389 | 0.289 | 0.795 | 0.060 |
| 0.5–0.6 | 183 | 0.549 | 0.585 | 0.470 | 0.820 | -0.036 |
| 0.6–0.7 | 230 | 0.653 | 0.743 | 0.517 | 0.802 | -0.091 |
| 0.7–0.8 | 321 | 0.754 | 0.835 | 0.642 | 0.830 | -0.080 |
| 0.8–0.9 | 637 | 0.858 | 0.953 | 0.873 | 0.887 | -0.095 |
| 0.9–1.0 | 546 | 0.927 | 0.993 | 0.969 | 0.940 | -0.066 |

ECE (tüm tahminler, 10 eşit genişlikli kutu) = 0.006; ECE (skor≥0.1) = 0.053. `skor−precision` pozitifse aşırı güvenli, negatifse eksik güvenli. 
Skor≥0.5 tahminlerin <e-posta> = 88.4%, <e-posta> = 78.0% (fark 10.4 puan). Skor 0.5 üstü kutularda (skor−precision) ortalaması -0.074: model yüksek skorlarda eksik güvenli (skor < gerçek <e-posta>), sıralama iyi; asıl boşluk <e-posta> ile @0.75 arasındaki konum kalitesidir. TP'lerde skor–IoU Spearman ρ=0.62: skor konum kalitesini kısmen taşıyor. TP_IoU_sirali oracle'ı (§3) IoU-farkında skorlamanın teorik payını sayısallaştırıyor.

## 8. Eğitim dinamiği (results.csv)

| gösterge | değer |
|---|---|
| epoch sayısı | 100 |
| en iyi epoch (mAP50-95) | 100 (0.6293) |
| son epoch mAP50-95 | 0.6293 |
| epoch 50 / 80 / 90 / 100 | 0.6196 / 0.6278 / 0.6286 / 0.6293 |
| son 20 epoch eğimi (puan/10 epoch) | 0.074 |
| önceki 20 epoch (ep 71-90) eğimi | 0.128 |
| son 20 epoch std (puan) | 0.061 |
| ep 81-90 ort. vs ep 91-100 ort. (close_mosaic=10) | 0.6281 vs 0.6289 (Δ +0.08 puan) |
| mAP50 son epoch | 0.8148 |
| results.csv en iyi ep mAP50-95 vs metrics.json (best.pt yeniden val) | 0.6293 vs 0.6296 (fark -0.03 puan; ölçüm-protokolü gürültü tabanı) |

Loss tablosu (train loss mosaic açıkken ep1/ep90, kapalıyken ep100; val/train oranı):

| loss | train ep1 | train ep90 | train ep100 | val ep100 | val/train ep100 | val/train ep90 | train loss ep90→91 değişim |
|---|---|---|---|---|---|---|---|
| box | 1.021 | 0.814 | 0.711 | 0.869 | 1.22 | 1.06 | -6.3% |
| cls | 2.353 | 0.837 | 0.575 | 0.788 | 1.37 | 0.94 | -21.3% |
| dfl | 1.300 | 1.151 | 1.077 | 1.154 | 1.07 | 1.00 | -3.2% |

Kayıp payı (results.csv loss_items kazançlı; loss.py:469-471 — önceki sürüm kazançları iki kez uyguluyordu, 2026-10-01 düzeltildi) ep100: box 30.1%, cls 24.3%, dfl 45.6% (ep1: box 21.8%, cls 50.3%, dfl 27.8%). Son 20 epoch mAP eğimi +0.074 puan/10 ep (std 0.06) → doygun (eğim seed gürültüsü altında). close_mosaic sonrası (ep 91-100) ortalama +0.08 puan. Train loss mosaic kapanınca ani düşer (cls -21.3%); bu rejim değişimi, train/val farkının bir kısmının augmentation kaynaklı olduğunu gösterir. val cls loss son 20 epoch 0.795→0.788 (yükselmiyor): val loss bozulmadığı için klasik aşırı öğrenme yok, ama mosaic kapalıyken val/train oranı cls 1.37, box 1.22: train kaybı val kaybından ayrışıyor (genelleme açığı); val mAP doygun.

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
| ön-işleme ms | 1.3243 |
| çıkarım ms | 6.6138 |
| NMS/son-işlem ms | 0.6986 |
| toplam ms | 8.6367 |
| FPS | 115.784 |
| batch | 1 |
| FP16 (1=evet) | 1.0 |
| ölçüm görüntüsü | 300 |
| ısınma görüntüsü | 50 |
| eğitim süresi (saat) | 3.2521 |

Varyant hız bütçesi (yalnız tahmin; gerçek ölçüm varyant koşumunda yapılır):

| varyant sınıfı | çarpan | tahmini toplam ms | tahmini FPS | not |
|---|---|---|---|---|
| eğitim-zamanı (aug/loss) varyantları | aynı | 8.64 | 116 | çıkarım grafiği değişmez; hız bütçesi ≈ 0 |
| Soft-NMS/DIoU-NMS (son-işlem ×1–5 varsayımı) | NMS ×1–5 | 8.64–11.43 | 87–116 | tahmin (ölçülmedi) |
| TTA (Ultralytics augment, ~3 ölçek+flip; literatür çarpanı 3.6× çıkarım) | çıkarım ×3.6 (literatür) | 25.83 | 39 | tahmin; literatur-cipa.md §4 (Ultralytics TTA dokümanı) |
| YOLO11s'e geçiş (bağlam satırı) | ≈3.4× GFLOPs | – | – | ölçülmedi |

Baseline batch=1 FP16 toplam 8.64 ms (116 FPS; çıkarım 6.61 ms, NMS 0.70 ms = toplamın %8'i). 'Gerçek zamanlı' için ≥30 FPS kabul edilirse (33 ms) ~3.9× gecikme payı vardır; TTA bile (~39 FPS tahmini) eşiğin üstünde kalabilir, ancak hız-doğruluk satırı ayrı raporlanmalıdır. Not: hız ölçümü GPU'nun paylaşıldığı bir anda alınmış olabilir (eğitim kuyruğu); karşılaştırmalı hız ölçümü aynı koşulda yapılmalıdır.
