# Hata analizi (K3.5) — baseline `20260930-101624_baseline_s1`

*Üretici: `scripts/analiz/hata_analizi.py` · yalnız VAL (1000 görüntü, 2407 nesne; difficult nesneler etiketlerde yok) · tahmin sayısı 40960 (conf=0.001, max_det=300, NMS IoU=0.7) · eşleştirme: pycocotools COCOeval · işletim noktası skor≥0.25.*

**Sızıntı notu:** Bu analizde test kümesi (VOC07 test), `rep_test*` dosyaları ve `metrics.json` içindeki `rep_*` anahtarları hiç kullanılmadı; betik bu anahtarları yükler yüklemez atar. Buradan çıkan hipotezler yalnız val bilgisine dayanır.

## 0. Genel ölçüler ve doğrulama

| ölçü | pycocotools (bu betik) | Ultralytics metrics.json (best.pt val) | fark |
|---|---|---|---|
| mAP50-95 | 0.6348 | 0.6353 | -0.0005 |
| mAP50 | 0.8115 | 0.8136 | -0.0021 |
| mAP75 (yalnız pycocotools) | 0.6882 | – | – |

AP50 ile AP50-95 arasındaki fark 17.7 puan; AP50→AP75 farkı 12.3 puan. pycocotools ile Ultralytics AP'si arasındaki küçük fark interpolasyon/eşleştirme (Ultralytics kendi IoU-matrisi eşleşmesini kullanır) kaynaklıdır.

## 1. Sınıf başına AP ve sınıf frekansı

| sınıf (AP50-95 artan) | AP50-95 | AP50 | AP75 | AP50−AP75 | train nesne | val nesne | val GT medyan kenar px |
|---|---|---|---|---|---|---|---|
| pottedplant | 0.305 | 0.561 | 0.260 | 0.300 | 1378 | 109 | 71 |
| chair | 0.452 | 0.629 | 0.515 | 0.113 | 2975 | 177 | 99 |
| boat | 0.507 | 0.734 | 0.541 | 0.193 | 1076 | 64 | 112 |
| bottle | 0.534 | 0.726 | 0.607 | 0.119 | 1669 | 95 | 70 |
| diningtable | 0.563 | 0.681 | 0.648 | 0.034 | 775 | 49 | 211 |
| bird | 0.611 | 0.786 | 0.672 | 0.115 | 1514 | 91 | 115 |
| sofa | 0.612 | 0.738 | 0.627 | 0.111 | 755 | 59 | 237 |
| person | 0.636 | 0.892 | 0.676 | 0.216 | 12471 | 785 | 141 |
| horse | 0.640 | 0.826 | 0.673 | 0.154 | 1006 | 66 | 209 |
| tvmonitor | 0.642 | 0.840 | 0.745 | 0.096 | 1036 | 72 | 108 |
| motorbike | 0.643 | 0.835 | 0.698 | 0.137 | 985 | 67 | 239 |
| car | 0.659 | 0.848 | 0.723 | 0.125 | 3045 | 222 | 92 |
| cow | 0.681 | 0.819 | 0.760 | 0.058 | 791 | 56 | 145 |
| bicycle | 0.696 | 0.910 | 0.755 | 0.155 | 1002 | 62 | 171 |
| aeroplane | 0.708 | 0.891 | 0.745 | 0.146 | 1091 | 80 | 184 |
| bus | 0.728 | 0.839 | 0.766 | 0.073 | 767 | 55 | 192 |
| train | 0.757 | 0.919 | 0.849 | 0.070 | 868 | 57 | 225 |
| sheep | 0.762 | 0.892 | 0.796 | 0.096 | 1013 | 57 | 119 |
| dog | 0.773 | 0.938 | 0.850 | 0.088 | 1929 | 96 | 211 |
| cat | 0.786 | 0.926 | 0.858 | 0.068 | 1505 | 88 | 278 |

En zayıf 5 sınıf (AP50-95): pottedplant 0.305, chair 0.452, boat 0.507, bottle 0.534, diningtable 0.563. Sınıf frekansı (train nesne) ile AP50-95 Spearman ρ=-0.16 (permütasyon p=0.493), AP50 ile ρ=0.11; sınıf başına train küçük-nesne oranı ile ρ=-0.40; val GT medyan kenar uzunluğu ile ρ=0.54 (n=20 sınıf, gürültülü). Frekans AP'nin zayıf belirleyicisi (|ρ|<0.3): zayıf sınıflarda sorun veri azlığından çok nesne boyutu/görünümü/bağlamıdır.

## 2. Nesne boyutu kırılımı (orijinal çözünürlük; COCO S<32²≤M<96²≤L)

| boyut | GT sayısı | AP50-95 | AP50 | <e-posta> (tüm conf, sınıf ort.) | <e-posta>:0.95 (sınıf ort.) | <e-posta> skor≥0.25 |
|---|---|---|---|---|---|---|
| all | 2407 | 0.635 | 0.812 | 0.964 | 0.771 | 0.790 |
| S | 143 | 0.207 | 0.351 | 0.781 | 0.473 | 0.357 |
| M | 704 | 0.452 | 0.646 | 0.912 | 0.663 | 0.700 |
| L | 1560 | 0.711 | 0.873 | 0.980 | 0.818 | 0.871 |

İnce alan aralıkları (GT düzeyi, eşleşme bayrağı):

| alan (px²) | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) |
|---|---|---|---|---|
| <16² | 9 | 55.6% | 0.0% | 22.2% |
| 16²–32² | 134 | 88.1% | 38.1% | 56.7% |
| 32²–64² | 377 | 96.0% | 66.6% | 75.3% |
| 64²–96² | 327 | 95.7% | 74.0% | 78.6% |
| 96²–192² | 698 | 97.6% | 81.5% | 84.1% |
| ≥192² | 862 | 99.1% | 91.5% | 91.1% |

Sınıf × boyut AP50-95 (n<5 ise '–'):

| sınıf | n_S | AP_S | n_M | AP_M | n_L | AP_L |
|---|---|---|---|---|---|---|
| aeroplane | 0 | – | 21 | 0.53 | 59 | 0.77 |
| bicycle | 0 | – | 12 | 0.41 | 50 | 0.77 |
| bird | 3 | – | 36 | 0.36 | 52 | 0.79 |
| boat | 4 | – | 25 | 0.50 | 35 | 0.57 |
| bottle | 13 | 0.25 | 52 | 0.50 | 30 | 0.71 |
| bus | 3 | – | 13 | 0.42 | 39 | 0.88 |
| car | 38 | 0.31 | 80 | 0.64 | 104 | 0.78 |
| cat | 0 | – | 2 | – | 86 | 0.81 |
| chair | 3 | – | 79 | 0.45 | 95 | 0.48 |
| cow | 0 | – | 25 | 0.66 | 31 | 0.75 |
| diningtable | 0 | – | 1 | – | 48 | 0.58 |
| dog | 0 | – | 10 | 0.67 | 86 | 0.79 |
| horse | 3 | – | 15 | 0.24 | 48 | 0.77 |
| motorbike | 3 | – | 10 | 0.56 | 54 | 0.69 |
| person | 51 | 0.21 | 215 | 0.50 | 519 | 0.72 |
| pottedplant | 12 | 0.14 | 57 | 0.31 | 40 | 0.37 |
| sheep | 8 | 0.31 | 16 | 0.71 | 33 | 0.89 |
| sofa | 0 | – | 1 | – | 58 | 0.62 |
| train | 0 | – | 4 | – | 53 | 0.77 |
| tvmonitor | 2 | – | 30 | 0.58 | 40 | 0.71 |

Küçük nesneler (n=143, val GT'nin %5.9'i): AP50-95 0.207 vs büyük 0.711 (fark 50.4 puan); <e-posta> 0.781 vs 0.980. Not: ölçüm orijinal piksel; eğitim/çıkarım 640 px letterbox olduğundan (~1.3× büyütme) S nesneler modelde ≈<41 px görünür. Val'de S örneği az olduğundan S-AP güven aralığı geniştir.

## 3. Hata türü ayrıştırması (TIDE benzeri)

Sayılar: tüm tahminler (conf≥0.001) ve işletim noktası (skor≥0.25). `Miss` tüm conf'ta hiçbir kutuyla IoU≥0.5 eşleşmeyen GT; işletim noktası sütununda ise o eşikte bulunamayan GT (eşleşmeyen + düşük skorla eşleşen).

| tür | sayı (tüm conf) | oran | sayı skor≥0.25 |
|---|---|---|---|
| TP | 2333 | 5.7% | 1902 |
| Dupe | 2432 | 5.9% | 91 |
| Cls | 3207 | 7.8% | 96 |
| Loc | 7657 | 18.7% | 146 |
| Both | 5052 | 12.3% | 75 |
| Bkg | 20279 | 49.5% | 281 |
| Miss | 74 | 3.1% (GT'nin) | 505 |

Oracle ΔAP (puan): ilgili hata türü düzeltilirse mAP ne kadar artardı (AP50 ve AP50-95, sınıf ortalaması). Taban: AP50=81.15, AP50-95=63.48.

| düzeltilen | ΔAP50 (puan) | ΔAP50-95 (puan) |
|---|---|---|
| Cls | 2.51 | 1.80 |
| Loc | 2.51 | 5.18 |
| Both | 1.21 | 0.82 |
| Dupe | 0.64 | -0.89 |
| Bkg | 3.57 | 2.30 |
| Miss | 3.00 | 2.25 |
| FP_hepsi | 14.84 | 6.39 |
| Kutu_kusursuz | 0.00 | 17.68 |
| TP_IoU_sirali | 1.66 | 5.39 |

Protokol etkisi: val'de etiketlerden çıkarılmış difficult nesne sayısı 493 (XML). Bunlardan biriyle aynı sınıf IoU≥0.5 örtüşen ve eşleşmemiş (FP sayılan) tahmin: 894 (skor≥0.25: 120; işletim noktası FP'lerinin 17.4%'i). Bu tahminler devkit gibi yok sayılsaydı ΔAP50 = +1.24, ΔAP50-95 = +0.83 puan (Ultralytics protokolünün val üzerinde ölçülmüş cezası; makalede belirtilmeli, Bkg/Loc/Dupe sayılarının bir kısmı bu yüzden yapaydır).


Not: `Dupe` ve `Loc` ΔAP50-95 değerleri IoU=0.5 tanımına göredir; bu kutular daha yüksek IoU eşiklerinde TP olabildiği için silme oracle'ı (Dupe) AP50-95'i düşürebilir; Loc oracle'ı kutuyu GT'ye yapıştırdığı için AP50-95 kazancı AP50'den büyüktür.


Hata türü × boyut (işletim noktası; tam uzun biçim `hata-kirilim.csv`):

| tür | S | M | L |
|---|---|---|---|
| TP (skor≥0.25, tahmin kutusu alanı) | 46 | 486 | 1370 |
| Dupe (skor≥0.25, tahmin kutusu alanı) | 2 | 28 | 61 |
| Cls (skor≥0.25, tahmin kutusu alanı) | 4 | 13 | 79 |
| Loc (skor≥0.25, tahmin kutusu alanı) | 6 | 59 | 81 |
| Both (skor≥0.25, tahmin kutusu alanı) | 1 | 17 | 57 |
| Bkg (skor≥0.25, tahmin kutusu alanı) | 58 | 135 | 88 |
| Miss (skor≥0.25 bulunamayan GT, GT alanı) | 92 | 211 | 202 |
| GT sayısı | 143 | 704 | 1560 |

AP50'ye en çok mal olan hata türü: Bkg (+3.57), sonra Miss (+3.00) ve Loc (+2.51). Oracle'lar birbirinden bağımsız ölçüldü, toplamları ΔAP'ye eşit değildir (etkileşim). `Kutu_kusursuz` = tüm TP kutuları GT'ye yapıştırılırsa AP50-95'in erişeceği üst sınır; `TP_IoU_sirali` = yalnız TP'lerin skorları IoU sırasına göre dağıtılırsa (FP skorları aynı) kazanç, yani IoU-farkında skorlamanın (VFL/QFL/IoU-aware) teorik payı.


Sınıf başına ΔAP50 (puan, oracle):

| sınıf | Cls | Loc | Both | Dupe | Bkg | Miss |
|---|---|---|---|---|---|---|
| aeroplane | 0.6 | 2.6 | 0.2 | 0.6 | 1.2 | 2.4 |
| bicycle | 1.3 | 1.1 | 0.9 | 0.2 | 0.7 | 1.6 |
| bird | 2.4 | 4.5 | 0.2 | 1.1 | 2.8 | 5.5 |
| boat | 3.1 | 3.8 | 0.2 | 1.5 | 6.3 | 5.1 |
| bottle | 0.6 | 1.2 | 0.3 | 0.3 | 10.5 | 2.5 |
| bus | 1.4 | 0.5 | 0.2 | 0.2 | 2.8 | 6.8 |
| car | 1.0 | 2.3 | 0.4 | 0.7 | 4.0 | 2.3 |
| cat | 2.4 | 2.0 | 0.1 | 0.1 | 0.1 | 3.5 |
| chair | 2.6 | 2.7 | 2.0 | 0.8 | 8.0 | 2.6 |
| cow | 8.8 | 0.9 | 0.1 | 1.0 | 5.2 | 1.4 |
| diningtable | 2.7 | 1.5 | 10.9 | 0.3 | 3.3 | 3.1 |
| dog | 1.1 | 0.9 | 0.3 | 0.2 | 0.8 | 0.0 |
| horse | 9.0 | 2.9 | 0.2 | 0.7 | 0.9 | 4.1 |
| motorbike | 1.9 | 3.8 | 0.9 | 0.4 | 1.3 | 5.3 |
| person | 0.2 | 2.9 | 0.1 | 0.9 | 2.1 | 1.5 |
| pottedplant | 1.0 | 9.8 | 2.0 | 0.9 | 6.9 | 5.6 |
| sheep | 4.0 | 0.0 | 0.8 | 0.5 | 3.3 | 0.0 |
| sofa | 4.8 | 3.1 | 2.8 | 1.8 | 3.2 | 2.6 |
| train | 0.4 | 2.7 | 0.3 | 0.1 | 1.8 | 1.8 |
| tvmonitor | 0.8 | 1.0 | 1.5 | 0.2 | 6.1 | 2.5 |

Kaçırılan GT (74) alt kırılımı: yalnız başka sınıfla örtüşen kutu var (sınıf karışıklığı) 20; aynı sınıf kutu var ama IoU∈[0.1,0.5) (lokalizasyon) 28; aynı sınıf IoU≥0.5 kutu var ama başka GT'ye atanmış (atama çakışması/örtüşme) 2; hiç kutu yok (gerçek kaçırma) 24 (kalan sınıflama kesin değil: bu üç grup kesişebilir).

## 4. Lokalizasyon kalitesi

| IoU aralığı | TP (tüm conf) | oran | TP skor≥0.25 | oran |
|---|---|---|---|---|
| 0.50–0.55 | 89 | 3.8% | 35 | 1.8% |
| 0.55–0.60 | 84 | 3.6% | 48 | 2.5% |
| 0.60–0.65 | 79 | 3.4% | 43 | 2.3% |
| 0.65–0.70 | 121 | 5.2% | 61 | 3.2% |
| 0.70–0.75 | 152 | 6.5% | 98 | 5.2% |
| 0.75–0.80 | 163 | 7.0% | 118 | 6.2% |
| 0.80–0.85 | 229 | 9.8% | 184 | 9.7% |
| 0.85–0.90 | 326 | 14.0% | 283 | 14.9% |
| 0.90–0.95 | 532 | 22.8% | 494 | 26.0% |
| 0.95–1.00 | 558 | 23.9% | 538 | 28.3% |

TP IoU ortalaması 0.845 (skor≥0.25: 0.871); işletim noktası TP'lerinin %15.0'i IoU<0.75. TP'lerde skor–IoU Spearman ρ=0.64 (skorun konum kalitesini ne kadar taşıdığı).

Sınıf başına AP50 − AP75 (büyükten küçüğe) ve TP ort. IoU:

| sınıf | AP50 | AP75 | AP50−AP75 (puan) | TP ort. IoU |
|---|---|---|---|---|
| pottedplant | 0.561 | 0.260 | 30.0 | 0.755 |
| person | 0.892 | 0.676 | 21.6 | 0.833 |
| boat | 0.734 | 0.541 | 19.3 | 0.817 |
| bicycle | 0.910 | 0.755 | 15.5 | 0.849 |
| horse | 0.826 | 0.673 | 15.4 | 0.857 |
| aeroplane | 0.891 | 0.745 | 14.6 | 0.856 |
| motorbike | 0.835 | 0.698 | 13.7 | 0.842 |
| car | 0.848 | 0.723 | 12.5 | 0.846 |
| bottle | 0.726 | 0.607 | 11.9 | 0.828 |
| bird | 0.786 | 0.672 | 11.5 | 0.848 |
| chair | 0.629 | 0.515 | 11.3 | 0.812 |
| sofa | 0.738 | 0.627 | 11.1 | 0.877 |
| sheep | 0.892 | 0.796 | 9.6 | 0.896 |
| tvmonitor | 0.840 | 0.745 | 9.6 | 0.870 |
| dog | 0.938 | 0.850 | 8.8 | 0.891 |
| bus | 0.839 | 0.766 | 7.3 | 0.894 |
| train | 0.919 | 0.849 | 7.0 | 0.891 |
| cat | 0.926 | 0.858 | 6.8 | 0.911 |
| cow | 0.819 | 0.760 | 5.8 | 0.888 |
| diningtable | 0.681 | 0.648 | 3.4 | 0.875 |

AP50→AP75 düşüşü sınıf ortalaması 12.3 puan; en büyük düşüş pottedplant (30), person (22), boat (19). Kutu-regresyon (IoU/DFL/box kazancı) manipülasyonlarının hedef alanı budur.

## 5. Sınıf karışıklığı

İşletim noktası (skor≥0.25, IoU≥0.5): her GT için eşleşen en yüksek skorlu kutu (doğru sınıf varsa o). Matris dosyası: `hata-analizi_20260930-101624_baseline_s1_karisiklik.csv`.

| GT sınıfı → tahmin | sayı | GT'nin payı |
|---|---|---|
| chair → person | 6 | 3.4% |
| cow → horse | 5 | 8.9% |
| cow → sheep | 5 | 8.9% |
| horse → sheep | 4 | 6.1% |
| sofa → person | 4 | 6.8% |
| sheep → cow | 3 | 5.3% |
| motorbike → person | 2 | 3.0% |
| person → bird | 2 | 0.3% |
| sofa → chair | 2 | 3.4% |
| bicycle → person | 1 | 1.6% |
| bird → bottle | 1 | 1.1% |
| boat → bus | 1 | 1.6% |

Çift bazında (yönsüz) toplam: cow/sheep 8; chair/person 6; cow/horse 5; horse/sheep 4; person/sofa 4; motorbike/person 3; chair/sofa 3; bird/person 2.


Tüm-conf alternatif görünüm: `Cls` türü tahminler (IoU≥0.5 başka sınıf GT, skor≥0.05), GT sınıfı → tahmin:

| GT → tahmin | tahmin sayısı |
|---|---|
| sofa → chair | 31 |
| person → sofa | 24 |
| person → chair | 22 |
| cow → horse | 14 |
| chair → sofa | 10 |
| dog → cat | 10 |
| motorbike → bicycle | 9 |
| sheep → cow | 9 |
| cow → sheep | 7 |
| cat → dog | 7 |

Kanonik çiftler (yönsüz; `Cls` tahmin sayısı skor≥0.05 | işletim noktası GT sayısı): cat/dog 17 | 1; cow/sheep 16 | 8; chair/diningtable 7 | 2; chair/sofa 41 | 3; horse/cow 16 | 5; bus/train 3 | 0; motorbike/bicycle 14 | 0.

İşletim noktasında GT'lerin 2.2%'i başka sınıf olarak etiketlendi (54/2407); 18.3%'i hiç bulunamadı. Sınıf karışıklığı oracle ΔAP50 = 2.51 puan (§3; Cls, altı hata türü içinde 4. sırada); işletim noktasında karışan GT oranı yalnız 2.2%, yani sorun sınıf ayrımından çok kutu/skor düzeyindedir.

## 6. Kalabalık ve örtüşme

| nesne/görüntü | görüntü | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) | FP/görüntü (skor≥0.25) |
|---|---|---|---|---|---|---|
| 1 | 440 | 440 | 98.9% | 92.5% | 92.5% | 0.37 |
| 2 | 252 | 504 | 97.8% | 82.1% | 87.5% | 0.45 |
| 3–4 | 195 | 654 | 96.6% | 76.0% | 82.4% | 0.94 |
| 5–8 | 88 | 511 | 95.9% | 71.4% | 75.7% | 1.55 |
| ≥9 | 25 | 298 | 95.0% | 73.5% | 72.8% | 3.76 |

Aynı sınıftan başka GT ile en yüksek IoU (örtüşme) gruplarına göre recall:

| komşu GT IoU | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 |
|---|---|---|---|
| <0.1 (komşu yok) | 1908 | 96.8% | 80.1% |
| 0.1–0.3 | 357 | 97.5% | 75.4% |
| 0.3–0.5 | 105 | 100.0% | 74.3% |
| 0.5–0.7 | 33 | 90.9% | 69.7% |
| ≥0.7 | 4 | 100.0% | 75.0% |

Aynı sınıf GT çiftleri (IoU≥0.3): 75 çift; ikisi de bulundu 72, yalnız biri 2, hiçbiri 1. IoU≥0.7 (NMS eşiği üstü, doğrudan NMS etkilenir): 2 çift, ikisi de bulundu 2, yalnız biri 0, hiçbiri 0. Kaçırılan GT'lerin 2/74'i aynı sınıf IoU≥0.5 bir kutuya sahip ama o kutu başka GT'ye atanmış (olası NMS/atama kaybı).

Kalabalıkla recall düşüşü: tek nesneli görüntüde 92.5% iken ≥9 nesnelide 73.5% (skor≥0.25); tüm-conf recall 98.9% → 95.0%, <e-posta> 92.5% → 72.8%. Çiftlerde tek-bulunma ve atama çakışması sayıları küçükse kayıp NMS'ten değil düşük skor/zayıf konumdan gelir.

## 7. Kalibrasyon (skor vs gerçek precision)

Tahmin 'doğru' = IoU≥0.5 ile eşleşen (<e-posta>); ayrıca IoU≥0.75 için precision. Not: YOLO sınıf skoru olasılık değil, TAL hizalama hedefli (IoU'ya bağlı) bir skordur; düşük skor bölgesindeki düşük precision 'aday havuzu' doğasından gelir.

| skor aralığı | tahmin | ort. skor | <e-posta> | <e-posta> | TP ort. IoU | skor−precision |
|---|---|---|---|---|---|---|
| 0.0–0.1 | 37326 | 0.008 | 0.008 | 0.007 | 0.721 | 0.000 |
| 0.1–0.2 | 818 | 0.142 | 0.109 | 0.073 | 0.743 | 0.033 |
| 0.2–0.3 | 395 | 0.245 | 0.200 | 0.147 | 0.772 | 0.045 |
| 0.3–0.4 | 271 | 0.347 | 0.325 | 0.218 | 0.770 | 0.022 |
| 0.4–0.5 | 205 | 0.443 | 0.454 | 0.307 | 0.795 | -0.010 |
| 0.5–0.6 | 209 | 0.553 | 0.531 | 0.388 | 0.780 | 0.021 |
| 0.6–0.7 | 233 | 0.651 | 0.661 | 0.472 | 0.809 | -0.010 |
| 0.7–0.8 | 307 | 0.757 | 0.853 | 0.668 | 0.835 | -0.097 |
| 0.8–0.9 | 657 | 0.857 | 0.954 | 0.868 | 0.886 | -0.097 |
| 0.9–1.0 | 539 | 0.926 | 0.996 | 0.980 | 0.943 | -0.070 |

ECE (tüm tahminler, 10 eşit genişlikli kutu) = 0.005; ECE (skor≥0.1) = 0.053. `skor−precision` pozitifse aşırı güvenli, negatifse eksik güvenli. 
Skor≥0.5 tahminlerin <e-posta> = 86.9%, <e-posta> = 76.8% (fark 10.1 puan). Skor 0.5 üstü kutularda (skor−precision) ortalaması -0.051: model yüksek skorlarda eksik güvenli (skor < gerçek <e-posta>), sıralama iyi; asıl boşluk <e-posta> ile @0.75 arasındaki konum kalitesidir. TP'lerde skor–IoU Spearman ρ=0.64: skor konum kalitesini kısmen taşıyor. TP_IoU_sirali oracle'ı (§3) IoU-farkında skorlamanın teorik payını sayısallaştırıyor.

## 8. Eğitim dinamiği (results.csv)

| gösterge | değer |
|---|---|
| epoch sayısı | 100 |
| en iyi epoch (mAP50-95) | 100 (0.6354) |
| son epoch mAP50-95 | 0.6354 |
| epoch 50 / 80 / 90 / 100 | 0.6211 / 0.6331 / 0.6338 / 0.6354 |
| son 20 epoch eğimi (puan/10 epoch) | 0.079 |
| önceki 20 epoch (ep 71-90) eğimi | 0.111 |
| son 20 epoch std (puan) | 0.073 |
| ep 81-90 ort. vs ep 91-100 ort. (close_mosaic=10) | 0.6337 vs 0.6343 (Δ +0.07 puan) |
| mAP50 son epoch | 0.8172 |
| results.csv en iyi ep mAP50-95 vs metrics.json (best.pt yeniden val) | 0.6354 vs 0.6353 (fark 0.01 puan; ölçüm-protokolü gürültü tabanı) |

Loss tablosu (train loss mosaic açıkken ep1/ep90, kapalıyken ep100; val/train oranı):

| loss | train ep1 | train ep90 | train ep100 | val ep100 | val/train ep100 | val/train ep90 | train loss ep90→91 değişim |
|---|---|---|---|---|---|---|---|
| box | 1.002 | 0.797 | 0.689 | 0.864 | 1.25 | 1.09 | -6.7% |
| cls | 2.301 | 0.809 | 0.540 | 0.783 | 1.45 | 0.97 | -22.9% |
| dfl | 1.284 | 1.143 | 1.059 | 1.152 | 1.09 | 1.01 | -3.6% |

Kazançlarla (box 7.5, cls 0.5, dfl 1.5) ağırlıklı kayıp payı ep100: box 73.5%, cls 3.8%, dfl 22.6% (ep1: box 71.0%, cls 10.9%, dfl 18.2%). Son 20 epoch mAP eğimi +0.079 puan/10 ep (std 0.07) → doygun (eğim seed gürültüsü altında). close_mosaic sonrası (ep 91-100) ortalama +0.07 puan. Train loss mosaic kapanınca ani düşer (cls -22.9%); bu rejim değişimi, train/val farkının bir kısmının augmentation kaynaklı olduğunu gösterir. val cls loss son 20 epoch 0.795→0.783 (yükselmiyor): val loss bozulmadığı için klasik aşırı öğrenme yok, ama mosaic kapalıyken val/train oranı cls 1.45, box 1.25: train kaybı val kaybından ayrışıyor (genelleme açığı); val mAP doygun.

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
| ön-işleme ms | 1.2664 |
| çıkarım ms | 6.1815 |
| NMS/son-işlem ms | 0.6897 |
| toplam ms | 8.1377 |
| FPS | 122.885 |
| batch | 1 |
| FP16 (1=evet) | 1.0 |
| ölçüm görüntüsü | 300 |
| ısınma görüntüsü | 50 |
| eğitim süresi (saat) | 2.9454 |

Varyant hız bütçesi (yalnız tahmin; gerçek ölçüm varyant koşumunda yapılır):

| varyant sınıfı | çarpan | tahmini toplam ms | tahmini FPS | not |
|---|---|---|---|---|
| eğitim-zamanı (aug/loss) varyantları | aynı | 8.14 | 123 | çıkarım grafiği değişmez; hız bütçesi ≈ 0 |
| Soft-NMS/DIoU-NMS (son-işlem ×1–5 varsayımı) | NMS ×1–5 | 8.14–10.90 | 92–123 | tahmin (ölçülmedi) |
| TTA (Ultralytics augment, ~3 ölçek+flip; literatür çarpanı 3.6× çıkarım) | çıkarım ×3.6 (literatür) | 24.21 | 41 | tahmin; literatur-cipa.md §4 (Ultralytics TTA dokümanı) |
| YOLO11s'e geçiş (bağlam satırı) | ≈3.4× GFLOPs | – | – | ölçülmedi |

Baseline batch=1 FP16 toplam 8.14 ms (123 FPS; çıkarım 6.18 ms, NMS 0.69 ms = toplamın %8'i). 'Gerçek zamanlı' için ≥30 FPS kabul edilirse (33 ms) ~4.1× gecikme payı vardır; TTA bile (~41 FPS tahmini) eşiğin üstünde kalabilir, ancak hız-doğruluk satırı ayrı raporlanmalıdır. Not: hız ölçümü GPU'nun paylaşıldığı bir anda alınmış olabilir (eğitim kuyruğu); karşılaştırmalı hız ölçümü aynı koşulda yapılmalıdır.
