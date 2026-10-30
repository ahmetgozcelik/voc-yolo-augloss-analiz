# Hata analizi (K3.5) — baseline `20260930-072103_baseline_s0`

*Üretici: `scripts/analiz/hata_analizi.py` · yalnız VAL (1000 görüntü, 2407 nesne; difficult nesneler etiketlerde yok) · tahmin sayısı 42519 (conf=0.001, max_det=300, NMS IoU=0.7) · eşleştirme: pycocotools COCOeval · işletim noktası skor≥0.25.*

**Sızıntı notu:** Bu analizde test kümesi (VOC07 test), `rep_test*` dosyaları ve `metrics.json` içindeki `rep_*` anahtarları hiç kullanılmadı; betik bu anahtarları yükler yüklemez atar. Buradan çıkan hipotezler yalnız val bilgisine dayanır.

## 0. Genel ölçüler ve doğrulama

| ölçü | pycocotools (bu betik) | Ultralytics metrics.json (best.pt val) | fark |
|---|---|---|---|
| mAP50-95 | 0.6285 | 0.6289 | -0.0005 |
| mAP50 | 0.8086 | 0.8113 | -0.0027 |
| mAP75 (yalnız pycocotools) | 0.6817 | – | – |

AP50 ile AP50-95 arasındaki fark 18.0 puan; AP50→AP75 farkı 12.7 puan. pycocotools ile Ultralytics AP'si arasındaki küçük fark interpolasyon/eşleştirme (Ultralytics kendi IoU-matrisi eşleşmesini kullanır) kaynaklıdır.

## 1. Sınıf başına AP ve sınıf frekansı

| sınıf (AP50-95 artan) | AP50-95 | AP50 | AP75 | AP50−AP75 | train nesne | val nesne | val GT medyan kenar px |
|---|---|---|---|---|---|---|---|
| pottedplant | 0.267 | 0.508 | 0.202 | 0.306 | 1378 | 109 | 71 |
| chair | 0.477 | 0.646 | 0.547 | 0.099 | 2975 | 177 | 99 |
| bottle | 0.513 | 0.668 | 0.590 | 0.078 | 1669 | 95 | 70 |
| boat | 0.548 | 0.768 | 0.611 | 0.156 | 1076 | 64 | 112 |
| sofa | 0.588 | 0.708 | 0.648 | 0.060 | 755 | 59 | 237 |
| diningtable | 0.590 | 0.722 | 0.662 | 0.060 | 775 | 49 | 211 |
| bird | 0.593 | 0.781 | 0.636 | 0.145 | 1514 | 91 | 115 |
| motorbike | 0.631 | 0.824 | 0.664 | 0.159 | 985 | 67 | 239 |
| tvmonitor | 0.636 | 0.842 | 0.737 | 0.105 | 1036 | 72 | 108 |
| person | 0.637 | 0.892 | 0.674 | 0.217 | 12471 | 785 | 141 |
| car | 0.643 | 0.840 | 0.701 | 0.139 | 3045 | 222 | 92 |
| horse | 0.652 | 0.849 | 0.682 | 0.167 | 1006 | 66 | 209 |
| cow | 0.665 | 0.834 | 0.769 | 0.065 | 791 | 56 | 145 |
| bicycle | 0.683 | 0.886 | 0.750 | 0.136 | 1002 | 62 | 171 |
| sheep | 0.690 | 0.853 | 0.724 | 0.129 | 1013 | 57 | 119 |
| aeroplane | 0.695 | 0.909 | 0.724 | 0.185 | 1091 | 80 | 184 |
| bus | 0.738 | 0.854 | 0.775 | 0.079 | 767 | 55 | 192 |
| train | 0.750 | 0.906 | 0.811 | 0.096 | 868 | 57 | 225 |
| dog | 0.781 | 0.941 | 0.873 | 0.068 | 1929 | 96 | 211 |
| cat | 0.792 | 0.941 | 0.853 | 0.088 | 1505 | 88 | 278 |

En zayıf 5 sınıf (AP50-95): pottedplant 0.267, chair 0.477, bottle 0.513, boat 0.548, sofa 0.588. Sınıf frekansı (train nesne) ile AP50-95 Spearman ρ=-0.13 (permütasyon p=0.574), AP50 ile ρ=0.05; sınıf başına train küçük-nesne oranı ile ρ=-0.42; val GT medyan kenar uzunluğu ile ρ=0.55 (n=20 sınıf, gürültülü). Frekans AP'nin zayıf belirleyicisi (|ρ|<0.3): zayıf sınıflarda sorun veri azlığından çok nesne boyutu/görünümü/bağlamıdır.

## 2. Nesne boyutu kırılımı (orijinal çözünürlük; COCO S<32²≤M<96²≤L)

| boyut | GT sayısı | AP50-95 | AP50 | <e-posta> (tüm conf, sınıf ort.) | <e-posta>:0.95 (sınıf ort.) | <e-posta> skor≥0.25 |
|---|---|---|---|---|---|---|
| all | 2407 | 0.628 | 0.809 | 0.961 | 0.768 | 0.793 |
| S | 143 | 0.236 | 0.434 | 0.722 | 0.439 | 0.385 |
| M | 704 | 0.434 | 0.629 | 0.954 | 0.681 | 0.703 |
| L | 1560 | 0.702 | 0.865 | 0.979 | 0.818 | 0.871 |

İnce alan aralıkları (GT düzeyi, eşleşme bayrağı):

| alan (px²) | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) |
|---|---|---|---|---|
| <16² | 9 | 55.6% | 0.0% | 11.1% |
| 16²–32² | 134 | 84.3% | 41.0% | 56.0% |
| 32²–64² | 377 | 95.5% | 67.9% | 74.0% |
| 64²–96² | 327 | 95.7% | 73.1% | 80.4% |
| 96²–192² | 698 | 97.4% | 81.4% | 83.0% |
| ≥192² | 862 | 99.0% | 91.8% | 91.4% |

Sınıf × boyut AP50-95 (n<5 ise '–'):

| sınıf | n_S | AP_S | n_M | AP_M | n_L | AP_L |
|---|---|---|---|---|---|---|
| aeroplane | 0 | – | 21 | 0.54 | 59 | 0.75 |
| bicycle | 0 | – | 12 | 0.42 | 50 | 0.75 |
| bird | 3 | – | 36 | 0.38 | 52 | 0.76 |
| boat | 4 | – | 25 | 0.54 | 35 | 0.61 |
| bottle | 13 | 0.15 | 52 | 0.49 | 30 | 0.69 |
| bus | 3 | – | 13 | 0.46 | 39 | 0.88 |
| car | 38 | 0.30 | 80 | 0.64 | 104 | 0.76 |
| cat | 0 | – | 2 | – | 86 | 0.81 |
| chair | 3 | – | 79 | 0.47 | 95 | 0.50 |
| cow | 0 | – | 25 | 0.65 | 31 | 0.72 |
| diningtable | 0 | – | 1 | – | 48 | 0.60 |
| dog | 0 | – | 10 | 0.73 | 86 | 0.79 |
| horse | 3 | – | 15 | 0.27 | 48 | 0.79 |
| motorbike | 3 | – | 10 | 0.57 | 54 | 0.66 |
| person | 51 | 0.21 | 215 | 0.51 | 519 | 0.72 |
| pottedplant | 12 | 0.15 | 57 | 0.28 | 40 | 0.33 |
| sheep | 8 | 0.43 | 16 | 0.57 | 33 | 0.81 |
| sofa | 0 | – | 1 | – | 58 | 0.60 |
| train | 0 | – | 4 | – | 53 | 0.78 |
| tvmonitor | 2 | – | 30 | 0.56 | 40 | 0.71 |

Küçük nesneler (n=143, val GT'nin %5.9'i): AP50-95 0.236 vs büyük 0.702 (fark 46.6 puan); <e-posta> 0.722 vs 0.979. Not: ölçüm orijinal piksel; eğitim/çıkarım 640 px letterbox olduğundan (~1.3× büyütme) S nesneler modelde ≈<41 px görünür. Val'de S örneği az olduğundan S-AP güven aralığı geniştir.

## 3. Hata türü ayrıştırması (TIDE benzeri)

Sayılar: tüm tahminler (conf≥0.001) ve işletim noktası (skor≥0.25). `Miss` tüm conf'ta hiçbir kutuyla IoU≥0.5 eşleşmeyen GT; işletim noktası sütununda ise o eşikte bulunamayan GT (eşleşmeyen + düşük skorla eşleşen).

| tür | sayı (tüm conf) | oran | sayı skor≥0.25 |
|---|---|---|---|
| TP | 2324 | 5.5% | 1909 |
| Dupe | 2498 | 5.9% | 88 |
| Cls | 3468 | 8.2% | 97 |
| Loc | 7641 | 18.0% | 160 |
| Both | 5324 | 12.5% | 72 |
| Bkg | 21264 | 50.0% | 270 |
| Miss | 83 | 3.4% (GT'nin) | 498 |

Oracle ΔAP (puan): ilgili hata türü düzeltilirse mAP ne kadar artardı (AP50 ve AP50-95, sınıf ortalaması). Taban: AP50=80.86, AP50-95=62.85.

| düzeltilen | ΔAP50 (puan) | ΔAP50-95 (puan) |
|---|---|---|
| Cls | 2.94 | 2.10 |
| Loc | 3.02 | 5.69 |
| Both | 1.30 | 0.91 |
| Dupe | 0.47 | -0.92 |
| Bkg | 3.52 | 2.29 |
| Miss | 3.27 | 2.44 |
| FP_hepsi | 14.78 | 6.66 |
| Kutu_kusursuz | 0.00 | 18.01 |
| TP_IoU_sirali | 1.67 | 5.63 |

Protokol etkisi: val'de etiketlerden çıkarılmış difficult nesne sayısı 493 (XML). Bunlardan biriyle aynı sınıf IoU≥0.5 örtüşen ve eşleşmemiş (FP sayılan) tahmin: 891 (skor≥0.25: 129; işletim noktası FP'lerinin 18.8%'i). Bu tahminler devkit gibi yok sayılsaydı ΔAP50 = +1.31, ΔAP50-95 = +0.91 puan (Ultralytics protokolünün val üzerinde ölçülmüş cezası; makalede belirtilmeli, Bkg/Loc/Dupe sayılarının bir kısmı bu yüzden yapaydır).


Not: `Dupe` ve `Loc` ΔAP50-95 değerleri IoU=0.5 tanımına göredir; bu kutular daha yüksek IoU eşiklerinde TP olabildiği için silme oracle'ı (Dupe) AP50-95'i düşürebilir; Loc oracle'ı kutuyu GT'ye yapıştırdığı için AP50-95 kazancı AP50'den büyüktür.


Hata türü × boyut (işletim noktası; tam uzun biçim `hata-kirilim.csv`):

| tür | S | M | L |
|---|---|---|---|
| TP (skor≥0.25, tahmin kutusu alanı) | 55 | 487 | 1367 |
| Dupe (skor≥0.25, tahmin kutusu alanı) | 1 | 21 | 66 |
| Cls (skor≥0.25, tahmin kutusu alanı) | 2 | 20 | 75 |
| Loc (skor≥0.25, tahmin kutusu alanı) | 6 | 64 | 90 |
| Both (skor≥0.25, tahmin kutusu alanı) | 0 | 17 | 55 |
| Bkg (skor≥0.25, tahmin kutusu alanı) | 52 | 132 | 86 |
| Miss (skor≥0.25 bulunamayan GT, GT alanı) | 88 | 209 | 201 |
| GT sayısı | 143 | 704 | 1560 |

AP50'ye en çok mal olan hata türü: Bkg (+3.52), sonra Miss (+3.27) ve Loc (+3.02). Oracle'lar birbirinden bağımsız ölçüldü, toplamları ΔAP'ye eşit değildir (etkileşim). `Kutu_kusursuz` = tüm TP kutuları GT'ye yapıştırılırsa AP50-95'in erişeceği üst sınır; `TP_IoU_sirali` = yalnız TP'lerin skorları IoU sırasına göre dağıtılırsa (FP skorları aynı) kazanç, yani IoU-farkında skorlamanın (VFL/QFL/IoU-aware) teorik payı.


Sınıf başına ΔAP50 (puan, oracle):

| sınıf | Cls | Loc | Both | Dupe | Bkg | Miss |
|---|---|---|---|---|---|---|
| aeroplane | 0.5 | 1.8 | 0.3 | 0.3 | 1.0 | 2.3 |
| bicycle | 2.4 | 2.0 | 0.5 | 0.3 | 1.0 | 1.3 |
| bird | 2.9 | 6.6 | 0.2 | 0.7 | 2.9 | 3.6 |
| boat | 1.1 | 5.3 | 0.2 | 0.4 | 3.8 | 6.4 |
| bottle | 0.9 | 2.6 | 0.4 | 0.3 | 11.1 | 4.5 |
| bus | 2.5 | 0.2 | 0.1 | 0.3 | 2.4 | 7.1 |
| car | 1.2 | 1.9 | 0.8 | 0.6 | 3.7 | 2.8 |
| cat | 1.1 | 2.5 | 0.1 | 0.1 | 0.1 | 2.1 |
| chair | 2.8 | 4.9 | 2.1 | 0.9 | 6.6 | 3.1 |
| cow | 6.2 | 3.4 | 0.0 | 0.6 | 3.5 | 1.5 |
| diningtable | 1.5 | 1.9 | 11.3 | 0.1 | 2.4 | 1.6 |
| dog | 2.1 | 0.5 | 0.2 | 0.1 | 0.7 | 1.3 |
| horse | 10.2 | 1.2 | 0.1 | 0.7 | 0.5 | 10.4 |
| motorbike | 3.3 | 3.7 | 0.6 | 0.4 | 2.4 | 6.5 |
| person | 0.2 | 2.7 | 0.1 | 0.8 | 2.2 | 2.1 |
| pottedplant | 0.5 | 12.0 | 3.0 | 1.0 | 7.9 | 5.0 |
| sheep | 10.0 | 0.2 | 0.5 | 0.2 | 4.1 | 0.0 |
| sofa | 8.4 | 2.7 | 4.1 | 1.1 | 4.6 | 1.2 |
| train | 0.5 | 2.8 | 0.7 | 0.2 | 2.3 | 0.0 |
| tvmonitor | 0.6 | 1.4 | 0.6 | 0.2 | 7.4 | 2.5 |

Kaçırılan GT (83) alt kırılımı: yalnız başka sınıfla örtüşen kutu var (sınıf karışıklığı) 23; aynı sınıf kutu var ama IoU∈[0.1,0.5) (lokalizasyon) 35; aynı sınıf IoU≥0.5 kutu var ama başka GT'ye atanmış (atama çakışması/örtüşme) 2; hiç kutu yok (gerçek kaçırma) 23 (kalan sınıflama kesin değil: bu üç grup kesişebilir).

## 4. Lokalizasyon kalitesi

| IoU aralığı | TP (tüm conf) | oran | TP skor≥0.25 | oran |
|---|---|---|---|---|
| 0.50–0.55 | 74 | 3.2% | 39 | 2.0% |
| 0.55–0.60 | 87 | 3.7% | 37 | 1.9% |
| 0.60–0.65 | 86 | 3.7% | 40 | 2.1% |
| 0.65–0.70 | 131 | 5.6% | 84 | 4.4% |
| 0.70–0.75 | 149 | 6.4% | 99 | 5.2% |
| 0.75–0.80 | 174 | 7.5% | 125 | 6.5% |
| 0.80–0.85 | 205 | 8.8% | 170 | 8.9% |
| 0.85–0.90 | 334 | 14.4% | 289 | 15.1% |
| 0.90–0.95 | 512 | 22.0% | 476 | 24.9% |
| 0.95–1.00 | 572 | 24.6% | 550 | 28.8% |

TP IoU ortalaması 0.846 (skor≥0.25: 0.871); işletim noktası TP'lerinin %15.7'i IoU<0.75. TP'lerde skor–IoU Spearman ρ=0.62 (skorun konum kalitesini ne kadar taşıdığı).

Sınıf başına AP50 − AP75 (büyükten küçüğe) ve TP ort. IoU:

| sınıf | AP50 | AP75 | AP50−AP75 (puan) | TP ort. IoU |
|---|---|---|---|---|
| pottedplant | 0.508 | 0.202 | 30.6 | 0.758 |
| person | 0.892 | 0.674 | 21.7 | 0.834 |
| aeroplane | 0.909 | 0.724 | 18.5 | 0.851 |
| horse | 0.849 | 0.682 | 16.7 | 0.864 |
| motorbike | 0.824 | 0.664 | 15.9 | 0.845 |
| boat | 0.768 | 0.611 | 15.6 | 0.836 |
| bird | 0.781 | 0.636 | 14.5 | 0.833 |
| car | 0.840 | 0.701 | 13.9 | 0.840 |
| bicycle | 0.886 | 0.750 | 13.6 | 0.859 |
| sheep | 0.853 | 0.724 | 12.9 | 0.876 |
| tvmonitor | 0.842 | 0.737 | 10.5 | 0.862 |
| chair | 0.646 | 0.547 | 9.9 | 0.830 |
| train | 0.906 | 0.811 | 9.6 | 0.894 |
| cat | 0.941 | 0.853 | 8.8 | 0.898 |
| bus | 0.854 | 0.775 | 7.9 | 0.905 |
| bottle | 0.668 | 0.590 | 7.8 | 0.837 |
| dog | 0.941 | 0.873 | 6.8 | 0.897 |
| cow | 0.834 | 0.769 | 6.5 | 0.864 |
| diningtable | 0.722 | 0.662 | 6.0 | 0.871 |
| sofa | 0.708 | 0.648 | 6.0 | 0.885 |

AP50→AP75 düşüşü sınıf ortalaması 12.7 puan; en büyük düşüş pottedplant (31), person (22), aeroplane (19). Kutu-regresyon (IoU/DFL/box kazancı) manipülasyonlarının hedef alanı budur.

## 5. Sınıf karışıklığı

İşletim noktası (skor≥0.25, IoU≥0.5): her GT için eşleşen en yüksek skorlu kutu (doğru sınıf varsa o). Matris dosyası: `hata-analizi_karisiklik.csv`.

| GT sınıfı → tahmin | sayı | GT'nin payı |
|---|---|---|
| chair → person | 5 | 2.8% |
| horse → sheep | 5 | 7.6% |
| bus → car | 3 | 5.5% |
| cow → horse | 3 | 5.4% |
| motorbike → person | 3 | 4.5% |
| sheep → dog | 3 | 5.3% |
| sofa → chair | 3 | 5.1% |
| sofa → person | 3 | 5.1% |
| bicycle → motorbike | 2 | 3.2% |
| cow → sheep | 2 | 3.6% |
| bird → bottle | 1 | 1.1% |
| boat → car | 1 | 1.6% |

Çift bazında (yönsüz) toplam: chair/person 5; horse/sheep 5; chair/sofa 4; bus/car 3; cow/horse 3; motorbike/person 3; dog/sheep 3; person/sofa 3.


Tüm-conf alternatif görünüm: `Cls` türü tahminler (IoU≥0.5 başka sınıf GT, skor≥0.05), GT sınıfı → tahmin:

| GT → tahmin | tahmin sayısı |
|---|---|
| sofa → chair | 23 |
| person → chair | 21 |
| person → sofa | 15 |
| dog → cat | 15 |
| horse → sheep | 11 |
| cow → horse | 10 |
| sheep → cow | 10 |
| motorbike → bicycle | 10 |
| cow → sheep | 9 |
| person → dog | 9 |

Kanonik çiftler (yönsüz; `Cls` tahmin sayısı skor≥0.05 | işletim noktası GT sayısı): cat/dog 23 | 1; cow/sheep 19 | 3; chair/diningtable 7 | 2; chair/sofa 31 | 4; horse/cow 14 | 3; bus/train 5 | 0; motorbike/bicycle 15 | 2.

İşletim noktasında GT'lerin 2.2%'i başka sınıf olarak etiketlendi (53/2407); 17.9%'i hiç bulunamadı. Sınıf karışıklığı oracle ΔAP50 = 2.94 puan (§3; Cls, altı hata türü içinde 4. sırada); işletim noktasında karışan GT oranı yalnız 2.2%, yani sorun sınıf ayrımından çok kutu/skor düzeyindedir.

## 6. Kalabalık ve örtüşme

| nesne/görüntü | görüntü | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) | FP/görüntü (skor≥0.25) |
|---|---|---|---|---|---|---|
| 1 | 440 | 440 | 99.3% | 93.2% | 91.8% | 0.38 |
| 2 | 252 | 504 | 97.4% | 83.7% | 89.1% | 0.47 |
| 3–4 | 195 | 654 | 96.8% | 76.0% | 82.7% | 0.99 |
| 5–8 | 88 | 511 | 94.3% | 70.8% | 76.5% | 1.31 |
| ≥9 | 25 | 298 | 94.3% | 73.2% | 67.1% | 3.76 |

Aynı sınıftan başka GT ile en yüksek IoU (örtüşme) gruplarına göre recall:

| komşu GT IoU | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 |
|---|---|---|---|
| <0.1 (komşu yok) | 1908 | 96.7% | 80.2% |
| 0.1–0.3 | 357 | 95.5% | 77.3% |
| 0.3–0.5 | 105 | 98.1% | 73.3% |
| 0.5–0.7 | 33 | 93.9% | 66.7% |
| ≥0.7 | 4 | 100.0% | 75.0% |

Aynı sınıf GT çiftleri (IoU≥0.3): 75 çift; ikisi de bulundu 71, yalnız biri 4, hiçbiri 0. IoU≥0.7 (NMS eşiği üstü, doğrudan NMS etkilenir): 2 çift, ikisi de bulundu 2, yalnız biri 0, hiçbiri 0. Kaçırılan GT'lerin 2/83'i aynı sınıf IoU≥0.5 bir kutuya sahip ama o kutu başka GT'ye atanmış (olası NMS/atama kaybı).

Kalabalıkla recall düşüşü: tek nesneli görüntüde 93.2% iken ≥9 nesnelide 73.2% (skor≥0.25); tüm-conf recall 99.3% → 94.3%, <e-posta> 91.8% → 67.1%. Çiftlerde tek-bulunma ve atama çakışması sayıları küçükse kayıp NMS'ten değil düşük skor/zayıf konumdan gelir.

## 7. Kalibrasyon (skor vs gerçek precision)

Tahmin 'doğru' = IoU≥0.5 ile eşleşen (<e-posta>); ayrıca IoU≥0.75 için precision. Not: YOLO sınıf skoru olasılık değil, TAL hizalama hedefli (IoU'ya bağlı) bir skordur; düşük skor bölgesindeki düşük precision 'aday havuzu' doğasından gelir.

| skor aralığı | tahmin | ort. skor | <e-posta> | <e-posta> | TP ort. IoU | skor−precision |
|---|---|---|---|---|---|---|
| 0.0–0.1 | 38853 | 0.008 | 0.007 | 0.006 | 0.726 | 0.001 |
| 0.1–0.2 | 846 | 0.143 | 0.130 | 0.095 | 0.761 | 0.013 |
| 0.2–0.3 | 417 | 0.248 | 0.204 | 0.129 | 0.738 | 0.044 |
| 0.3–0.4 | 251 | 0.346 | 0.295 | 0.199 | 0.771 | 0.052 |
| 0.4–0.5 | 226 | 0.447 | 0.509 | 0.350 | 0.774 | -0.062 |
| 0.5–0.6 | 193 | 0.553 | 0.549 | 0.368 | 0.786 | 0.003 |
| 0.6–0.7 | 238 | 0.652 | 0.685 | 0.521 | 0.820 | -0.033 |
| 0.7–0.8 | 299 | 0.750 | 0.839 | 0.666 | 0.839 | -0.090 |
| 0.8–0.9 | 641 | 0.857 | 0.956 | 0.855 | 0.882 | -0.099 |
| 0.9–1.0 | 555 | 0.927 | 0.989 | 0.971 | 0.943 | -0.062 |

ECE (tüm tahminler, 10 eşit genişlikli kutu) = 0.006; ECE (skor≥0.1) = 0.052. `skor−precision` pozitifse aşırı güvenli, negatifse eksik güvenli. 
Skor≥0.5 tahminlerin <e-posta> = 87.3%, <e-posta> = 76.9% (fark 10.4 puan). Skor 0.5 üstü kutularda (skor−precision) ortalaması -0.056: model yüksek skorlarda eksik güvenli (skor < gerçek <e-posta>), sıralama iyi; asıl boşluk <e-posta> ile @0.75 arasındaki konum kalitesidir. TP'lerde skor–IoU Spearman ρ=0.62: skor konum kalitesini kısmen taşıyor. TP_IoU_sirali oracle'ı (§3) IoU-farkında skorlamanın teorik payını sayısallaştırıyor.

## 8. Eğitim dinamiği (results.csv)

| gösterge | değer |
|---|---|
| epoch sayısı | 100 |
| en iyi epoch (mAP50-95) | 93 (0.6318) |
| son epoch mAP50-95 | 0.6313 |
| epoch 50 / 80 / 90 / 100 | 0.6185 / 0.6302 / 0.6309 / 0.6313 |
| son 20 epoch eğimi (puan/10 epoch) | 0.015 |
| önceki 20 epoch (ep 71-90) eğimi | 0.158 |
| son 20 epoch std (puan) | 0.035 |
| ep 81-90 ort. vs ep 91-100 ort. (close_mosaic=10) | 0.6310 vs 0.6313 (Δ +0.03 puan) |
| mAP50 son epoch | 0.8134 |
| results.csv en iyi ep mAP50-95 vs metrics.json (best.pt yeniden val) | 0.6318 vs 0.6289 (fark 0.29 puan; ölçüm-protokolü gürültü tabanı) |

Loss tablosu (train loss mosaic açıkken ep1/ep90, kapalıyken ep100; val/train oranı):

| loss | train ep1 | train ep90 | train ep100 | val ep100 | val/train ep100 | val/train ep90 | train loss ep90→91 değişim |
|---|---|---|---|---|---|---|---|
| box | 0.996 | 0.796 | 0.689 | 0.863 | 1.25 | 1.09 | -6.7% |
| cls | 2.275 | 0.803 | 0.538 | 0.776 | 1.44 | 0.97 | -21.8% |
| dfl | 1.284 | 1.141 | 1.060 | 1.151 | 1.09 | 1.01 | -3.7% |

Kayıp payı (results.csv loss_items kazançlı; loss.py:469-471 — önceki sürüm kazançları iki kez uyguluyordu, 2026-10-01 düzeltildi) ep100: box 30.1%, cls 23.5%, dfl 46.3% (ep1: box 21.9%, cls 50.0%, dfl 28.2%). Son 20 epoch mAP eğimi +0.015 puan/10 ep (std 0.03) → doygun (eğim seed gürültüsü altında). close_mosaic sonrası (ep 91-100) ortalama +0.03 puan. Train loss mosaic kapanınca ani düşer (cls -21.8%); bu rejim değişimi, train/val farkının bir kısmının augmentation kaynaklı olduğunu gösterir. val cls loss son 20 epoch 0.783→0.776 (yükselmiyor): val loss bozulmadığı için klasik aşırı öğrenme yok, ama mosaic kapalıyken val/train oranı cls 1.44, box 1.25: train kaybı val kaybından ayrışıyor (genelleme açığı); val mAP doygun.

## 9. Seed varyansı

| koşum | seed | epoch | mAP50-95 | mAP50 | mAP_S | mAP_M | mAP_L |
|---|---|---|---|---|---|---|---|
| 20260930-065935_baseline_s0 | 0 | 3 | 0.4104 | 0.5813 | 0.098 | 0.289 | 0.456 |
| 20260930-070132_baseline_s0 | 0 | 5 | 0.5867 | 0.7807 | 0.178 | 0.421 | 0.664 |
| 20260930-071120_baseline_s1 | 1 | 5 | 0.5908 | 0.7849 | 0.169 | 0.388 | 0.672 |
| 20260930-072103_baseline_s0 | 0 | 100 | 0.6289 | 0.8113 | 0.236 | 0.434 | 0.702 |
| 20260930-101624_baseline_s1 | 1 | 100 | 0.6353 | 0.8136 | 0.207 | 0.452 | 0.711 |
| 20260930-131427_baseline_s2 | 2 | 100 | 0.6278 | 0.8113 | 0.220 | 0.443 | 0.701 |

Epoch grubuna göre (aynı epoch sayısı, seed'ler arası; n≥2 iken std):

| epoch | n koşum | ort. mAP50-95 | std |
|---|---|---|---|
| 3 | 1 | 0.4104 | n<2 |
| 5 | 2 | 0.5888 | 0.0029 |
| 100 | 3 | 0.6307 | 0.0041 |

*Tam (100 ep) koşumlarda seed1/seed2 bittiğinde şef bu betiği yeniden koşturunca tablo otomatik dolar (`--seed-glob`). Ön-uçuş (5 ep) iki seed yalnız erken işarettir; 5 epoch'ta mAP henüz yakınsamadığından tam koşum seed std'si için temsilî değildir.*

## 10. Hız profili (metrics.json)

| gösterge | değer |
|---|---|
| parametre | 2593740 |
| GFLOPs (640) | 6.52 |
| ön-işleme ms | 1.2869 |
| çıkarım ms | 6.037 |
| NMS/son-işlem ms | 0.6793 |
| toplam ms | 8.0032 |
| FPS | 124.951 |
| batch | 1 |
| FP16 (1=evet) | 1.0 |
| ölçüm görüntüsü | 300 |
| ısınma görüntüsü | 50 |
| eğitim süresi (saat) | 2.9013 |

Varyant hız bütçesi (yalnız tahmin; gerçek ölçüm varyant koşumunda yapılır):

| varyant sınıfı | çarpan | tahmini toplam ms | tahmini FPS | not |
|---|---|---|---|---|
| eğitim-zamanı (aug/loss) varyantları | aynı | 8.00 | 125 | çıkarım grafiği değişmez; hız bütçesi ≈ 0 |
| Soft-NMS/DIoU-NMS (son-işlem ×1–5 varsayımı) | NMS ×1–5 | 8.00–10.72 | 93–125 | tahmin (ölçülmedi) |
| TTA (Ultralytics augment, ~3 ölçek+flip; literatür çarpanı 3.6× çıkarım) | çıkarım ×3.6 (literatür) | 23.70 | 42 | tahmin; literatur-cipa.md §4 (Ultralytics TTA dokümanı) |
| YOLO11s'e geçiş (bağlam satırı) | ≈3.4× GFLOPs | – | – | ölçülmedi |

Baseline batch=1 FP16 toplam 8.00 ms (125 FPS; çıkarım 6.04 ms, NMS 0.68 ms = toplamın %8'i). 'Gerçek zamanlı' için ≥30 FPS kabul edilirse (33 ms) ~4.2× gecikme payı vardır; TTA bile (~42 FPS tahmini) eşiğin üstünde kalabilir, ancak hız-doğruluk satırı ayrı raporlanmalıdır. Not: hız ölçümü GPU'nun paylaşıldığı bir anda alınmış olabilir (eğitim kuyruğu); karşılaştırmalı hız ölçümü aynı koşulda yapılmalıdır.


## Hipotez hammaddesi

*Baseline seed0 tam koşum (100 ep), yalnız val. Sayılar yukarıdaki bölümlerden; beklenen Δ değerleri `notlar/literatur-cipa.md` §4 tablosundan (mAP50-95 puan, YOLO11n ön-eğitimli, 100 ep). Burada kart yazılmadı: hipotez-tasarımcısına girdi.*

**Sızıntı uyarısı:** Test kümesi (VOC07 test), `rep_test*` dosyaları ve `metrics.json`'daki `rep_*` anahtarları bu analizde hiç kullanılmadı; aşağıdaki hiçbir sayı veya yön seçimi test bilgisine dayanmaz. Şeffaflık notu: analiz oturumunun başında `metrics.json`'un ham başı terminale yazdırılırken `rep_ap50_*` anahtarlarının ilk satırları (filtre öncesi) ekranda göründü; bu değerler hiçbir hesaba, tabloya veya hipotez kararına girmedi, betik `rep_*` anahtarlarını yükler yüklemez atar. Şef isterse bu ifşa karar günlüğüne kaydedilmeli.

**Gürültü tabanı (kart eşikleri için):** 5-epoch ön-uçuşta seed0/seed1 farkı 0.41 puan (0.5867 vs 0.5908); aynı koşumda results.csv en iyi epoch (0.6318) ile best.pt yeniden val (0.6289) farkı 0.29 puan; son 20 epoch std 0.035 puan. Tam koşum seed1/2 gelene kadar ≈0.3–0.5 puanın altındaki Δ'lar ayırt edilemez sayılmalı; S/M alt-AP'lerinde gürültü daha büyük (5 ep: AP_M 0.421 vs 0.388; val'de yalnız 143 S nesne).

### Ölçülmüş zayıflıklar → hedefleyen manipülasyon aileleri

| # | Ölçülmüş zayıflık (val, baseline s0) | Hedefleyen manipülasyon ailesi (somut kollar) | Literatürden beklenen Δ | Hız bedeli |
|---|---|---|---|---|
| Z1 | **Küçük/orta nesne**: AP_S 0.236, AP_M 0.434 vs AP_L 0.702; <e-posta> (tüm conf) S 0.72 / M 0.95 / L 0.98; işletim noktasında (skor≥0.25) S 38.5%, M 70.3%, L 87.1% bulunuyor; bulunamayan 498 GT'nin 88'i S (S'nin %61.5'i), 209'u M (%29.7), 201'i L (%12.9). S+M val GT'nin %35.2'si. 16²–32² aralığında <e-posta> %56 (büyük kutularda %91). | **Aug/ölçek**: `scale` iki yönlü tarama (0.5→0.25 ve →0.9; YOLOX-Nano "daralt" bulgusu ile zoom-in savunusu çelişiyor, yön val'de belirlenir), `multi_scale`, mosaic ölçek aralığı. **Aug/bbox-copy-paste**: VOC'ta Ultralytics `copy_paste` segment yokluğundan etkisiz (literatur-cipa §4 tuzağı); küçük-nesne ağırlıklı kutu-tabanlı özel yapıştırma `rakip/` dışında yazılmalı. **Post**: TTA (ölçek yukarı) ayrı satır. | scale/mosaic ölçeği ±0.5 (daralt +0…+0.5); copy-paste maskeli 0…+0.3; TTA +0.3…+1.0 | aug: 0; TTA ≈ çıkarım ×3.6 → ~42 FPS tahmin |
| Z2 | **Konum hassasiyeti**: AP50 0.809 → AP75 0.682 → AP50-95 0.628 (AP50−AP50-95 = 18.0 puan); "tüm TP kutuları kusursuz" oracle'ı +18.0 puan (üst sınır, erişilemez); işletim noktası TP'lerinin %15.7'si IoU<0.75, TP ort. IoU 0.846; Loc-türü FP oracle'ı ΔAP50 +3.02. En büyük AP50−AP75 düşüşü pottedplant 30.6, person 21.7, aeroplane 18.5 puan. | **Loss**: `box` kazancı (7.5'ten yukarı/aşağı), `dfl` kazancı, `reg_max`; IoU varyantları (CIoU zaten taban; SIoU/EIoU/WIoU/Inner-IoU ayrı kol, aynı ailede 1–2 tanesi yeter); kayıp payı (DÜZELTİLDİ 2026-10-01: önceki %73.5 box değeri kazançların çift uygulanmasından kaynaklıydı) — doğru değer §8 tablosunda; box-odaklı ayarın özel kaldıraç üstünlüğü yok. | Kayıp kazançları ±0.3; yeni IoU kayıpları CIoU'ya göre −0.3…+0.5 (WIoU AP75 +1.47 bildirimi farklı dedektörde; Tong 2023) | 0 (yalnız eğitim) |
| Z3 | **Zayıf sınıf grubu**: pottedplant AP 0.267 (AP50 0.508), chair 0.477, bottle 0.513, boat 0.548, sofa 0.588. Frekans açıklamıyor: train nesne sayısı ile AP Spearman ρ=−0.13 (p=0.57); chair 2975 train nesne ile 2. en zayıf, bus 767 nesneyle AP 0.738, sofa/diningtable 755–775 nesneyle 0.59. Oracle ΔAP50 (sınıf içi): pottedplant Loc +12.0 / Bkg +7.9 / Miss +5.0; bottle Bkg +11.1; chair Bkg +6.6 / Loc +4.9; boat Miss +6.4 / Loc +5.3. Bu sınıflar küçük-orta ve belirsiz sınırlı (pottedplant S+M = 69/109). | Aug (bağlam/örtüşme): mosaic olasılığı/close_mosaic, bbox copy-paste (hedef sınıf ağırlıklı), mixup. Loss: sınıf-ağırlıklı cls kaybı düşük öncelikli (frekans belirleyici değil). | mixup −0.5…+0.3 (nano); close_mosaic 0…+0.3 | 0 |
| Z4 | **Skor/IoU uyumu ve kalibrasyon**: TP'lerde skor–IoU Spearman ρ=0.62; skor≥0.5 tahminlerde <e-posta> 87.3% ama @0.75 76.9% (10.4 puan konum boşluğu); ECE (skor≥0.1) 0.052, model yüksek skorlarda eksik güvenli (skor−precision −0.06…−0.10, 0.7–1.0 aralığı). "TP'leri IoU sırasına diz" oracle'ı (FP skorları aynı) AP50-95 +5.63, AP50 +1.67 (üst sınır). Kalibrasyon bozuk değil; sıralama kalitesi (IoU-farkında) kaldıraç. | **Loss**: Varifocal/QFL/IoU-aware BCE hedefi (Ultralytics TAL hedefi zaten hizalama-metriği/IoU ağırlıklı: kazanç sınırlı beklenir), Focal/BCE cls kazancı (`cls` 0.5→1.0). | Focal/VFL/QFL −0.5…+0.3 (Ultralytics BCE+TAL tabanında kaynak yok; Redmon 2018 focal −2 bildirimi) | 0 |
| Z5 | **Arka plan FP / düşük-skor kaçırma (hata kütlesi dağınık)**: oracle ΔAP50: Bkg +3.52, Miss +3.27, Loc +3.02, Cls +2.94, Both +1.30, Dupe +0.47 (tek baskın tür yok); tüm FP'ler silinirse +14.78. İşletim noktasında FP: Bkg 270, Loc 160, Cls 97, Dupe 88, Both 72; GT'lerin yalnız %2.2'si başka sınıfa karışıyor (cat/dog 1, cow/sheep 3, chair/diningtable 2 GT). Bkg/Loc/Dupe'nin bir kısmı protokol artefaktı: difficult nesneleri bulan 891 tahmin FP sayılıyor (ΔAP50 +1.31, AP50-95 +0.91 puan; skor≥0.25'te 129 FP = %18.8). | **Post**: Soft-NMS/DIoU-NMS/WBF ayrı satırlar (Dupe+atama çakışması küçük: 75 örtüşen aynı-sınıf GT çiftinden 71'inde ikisi de bulundu, yalnız 2/83 kaçırılan GT atama çakışmalı → NMS kaynaklı kayıp az). **Eğitim**: difficult'lı nesneleri görmezden gelen özel etiket kullanmak protokolü değiştirir; ayrı bulgu olarak raporlanır, manipülasyon değil. | Soft-NMS 0…+0.4; DIoU-NMS 0…+0.2; WBF (TTA ile) +0.2…+0.8 | NMS ×1–5 → ~93–125 FPS tahmin |
| Z6 | **Doygunluk ve genelleme açığı**: val mAP50-95 ep80'den beri 0.6302→0.6313 (son 20 ep eğim +0.015 puan/10 ep, std 0.035); close_mosaic (ep91-100) ortalama Δ +0.03 puan (gürültü içinde); mosaic kapalıyken val/train loss oranı cls 1.44, box 1.25 (ep90'da 0.97/1.09); val loss yükselmiyor. Uzun eğitim kolu değersiz; asıl soru regularizasyon gücü. | Aug gücü ızgarası: `mixup` 0→0.05–0.15, `mosaic` olasılığı, `close_mosaic` 10→(5,20), `erasing`/`auto_augment` (varsayılan randaugment açık). | mixup −0.5…+0.3; close_mosaic 0…+0.3; mosaic ölçeği ±0.5 | 0 |

### Önerilen öncelik sıralaması (ölçülmüş kaldıraç × literatür beklentisi)
1. **Z1 ölçek/küçük-nesne augmentation** (scale iki yön, multi_scale, S-ağırlıklı bbox-copy-paste): en büyük ölçülü boşluk (S/M recall işletim noktasında %38/%70) ve augmentation hedefiyle doğrudan uyumlu; yön belirsiz olduğu için iki kollu ön-kayıt.
2. **Z2 box/dfl kazanç ve IoU-varyant kayıpları**: 18 puanlık AP50→AP50-95 boşluğu (kayıp payı gerekçesi 2026-10-01 düzeltildi, bkz. §8); beklenen Δ küçük (≤+0.5) ama loss ailesinin ana adayı. Kazanç ızgarası önce (ucuz, tek parametre), IoU varyantı sonra.
3. **Z4 IoU-farkında sınıflama kaybı (VFL/QFL)**: oracle üst sınırı büyük (+5.6) fakat TAL hedefi zaten IoU-farkında; beklenen Δ −0.5…+0.3, riskli/az olası.
4. **Z6/Z3 mixup, close_mosaic, mosaic p**: genelleme açığı var ama nano modelde literatür nötr/negatif; ucuz tek-parametreli kollar, ön-uçuşta elenebilir.
5. **Post ailesi (TTA, Soft-NMS, DIoU-NMS, WBF)**: eğitim-zamanı kazancından ayrı raporlanır; NMS kaynaklı kayıp ölçüldü ve küçük (≈0…+0.4 beklenti), TTA küçük nesneye yarayabilir (+0.3…+1.0) ama ~3.6× çıkarım.

**Uyarılar:** (i) Ultralytics `copy_paste` VOC kutu etiketlerinde etkisizdir; bu ayar ızgaraya girerse seed gürültüsü etki gibi görünür. (ii) Val'de S nesne yalnız 143, AP_S tek başına karar ölçütü yapılmamalı; hem AP_S/AP_M hem toplam mAP raporlanmalı. (iii) Sınıf karışıklığı (cat/dog vb.) ölçüldü ve küçük; sınıf-odaklı hipotezler düşük öncelik. (iv) pycocotools ile Ultralytics mAP50-95 farkı 0.05 puan (0.6285 vs 0.6289), ΔAP karşılaştırmaları aynı ölçekte yapılabilir.
