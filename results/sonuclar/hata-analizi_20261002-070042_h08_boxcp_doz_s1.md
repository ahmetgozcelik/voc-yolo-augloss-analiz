# Hata analizi (K3.5) — baseline `20261002-070042_h08_boxcp_doz_s1`

*Üretici: `scripts/analiz/hata_analizi.py` · yalnız VAL (1000 görüntü, 2407 nesne; difficult nesneler etiketlerde yok) · tahmin sayısı 44008 (conf=0.001, max_det=300, NMS IoU=0.7) · eşleştirme: pycocotools COCOeval · işletim noktası skor≥0.25.*

**Sızıntı notu:** Bu analizde test kümesi (VOC07 test), `rep_test*` dosyaları ve `metrics.json` içindeki `rep_*` anahtarları hiç kullanılmadı; betik bu anahtarları yükler yüklemez atar. Buradan çıkan hipotezler yalnız val bilgisine dayanır.

## 0. Genel ölçüler ve doğrulama

| ölçü | pycocotools (bu betik) | Ultralytics metrics.json (best.pt val) | fark |
|---|---|---|---|
| mAP50-95 | 0.6293 | 0.6300 | -0.0008 |
| mAP50 | 0.8126 | 0.8146 | -0.0020 |
| mAP75 (yalnız pycocotools) | 0.6834 | – | – |

AP50 ile AP50-95 arasındaki fark 18.3 puan; AP50→AP75 farkı 12.9 puan. pycocotools ile Ultralytics AP'si arasındaki küçük fark interpolasyon/eşleştirme (Ultralytics kendi IoU-matrisi eşleşmesini kullanır) kaynaklıdır.

## 1. Sınıf başına AP ve sınıf frekansı

| sınıf (AP50-95 artan) | AP50-95 | AP50 | AP75 | AP50−AP75 | train nesne | val nesne | val GT medyan kenar px |
|---|---|---|---|---|---|---|---|
| pottedplant | 0.303 | 0.585 | 0.261 | 0.324 | 1378 | 109 | 71 |
| chair | 0.479 | 0.655 | 0.543 | 0.113 | 2975 | 177 | 99 |
| bottle | 0.528 | 0.692 | 0.610 | 0.082 | 1669 | 95 | 70 |
| boat | 0.560 | 0.782 | 0.632 | 0.150 | 1076 | 64 | 112 |
| diningtable | 0.573 | 0.701 | 0.646 | 0.055 | 775 | 49 | 211 |
| bird | 0.583 | 0.768 | 0.634 | 0.133 | 1514 | 91 | 115 |
| sofa | 0.602 | 0.759 | 0.614 | 0.145 | 755 | 59 | 237 |
| tvmonitor | 0.606 | 0.813 | 0.715 | 0.098 | 1036 | 72 | 108 |
| horse | 0.613 | 0.801 | 0.663 | 0.138 | 1006 | 66 | 209 |
| motorbike | 0.623 | 0.807 | 0.673 | 0.135 | 985 | 67 | 239 |
| person | 0.631 | 0.883 | 0.670 | 0.213 | 12471 | 785 | 141 |
| car | 0.652 | 0.845 | 0.700 | 0.144 | 3045 | 222 | 92 |
| cow | 0.679 | 0.850 | 0.808 | 0.041 | 791 | 56 | 145 |
| train | 0.700 | 0.856 | 0.759 | 0.096 | 868 | 57 | 225 |
| aeroplane | 0.713 | 0.915 | 0.728 | 0.187 | 1091 | 80 | 184 |
| bicycle | 0.720 | 0.927 | 0.786 | 0.141 | 1002 | 62 | 171 |
| bus | 0.733 | 0.835 | 0.775 | 0.060 | 767 | 55 | 192 |
| sheep | 0.741 | 0.923 | 0.789 | 0.134 | 1013 | 57 | 119 |
| cat | 0.763 | 0.930 | 0.819 | 0.111 | 1505 | 88 | 278 |
| dog | 0.781 | 0.927 | 0.844 | 0.083 | 1929 | 96 | 211 |

En zayıf 5 sınıf (AP50-95): pottedplant 0.303, chair 0.479, bottle 0.528, boat 0.560, diningtable 0.573. Sınıf frekansı (train nesne) ile AP50-95 Spearman ρ=-0.11 (permütasyon p=0.634), AP50 ile ρ=0.05; sınıf başına train küçük-nesne oranı ile ρ=-0.36; val GT medyan kenar uzunluğu ile ρ=0.51 (n=20 sınıf, gürültülü). Frekans AP'nin zayıf belirleyicisi (|ρ|<0.3): zayıf sınıflarda sorun veri azlığından çok nesne boyutu/görünümü/bağlamıdır.

## 2. Nesne boyutu kırılımı (orijinal çözünürlük; COCO S<32²≤M<96²≤L)

| boyut | GT sayısı | AP50-95 | AP50 | <e-posta> (tüm conf, sınıf ort.) | <e-posta>:0.95 (sınıf ort.) | <e-posta> skor≥0.25 |
|---|---|---|---|---|---|---|
| all | 2407 | 0.629 | 0.813 | 0.963 | 0.770 | 0.791 |
| S | 143 | 0.242 | 0.392 | 0.784 | 0.502 | 0.350 |
| M | 704 | 0.433 | 0.650 | 0.951 | 0.682 | 0.706 |
| L | 1560 | 0.705 | 0.870 | 0.979 | 0.820 | 0.871 |

İnce alan aralıkları (GT düzeyi, eşleşme bayrağı):

| alan (px²) | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) |
|---|---|---|---|---|
| <16² | 9 | 55.6% | 0.0% | 22.2% |
| 16²–32² | 134 | 87.3% | 37.3% | 58.2% |
| 32²–64² | 377 | 95.5% | 67.4% | 75.6% |
| 64²–96² | 327 | 94.8% | 74.3% | 77.4% |
| 96²–192² | 698 | 97.6% | 80.5% | 84.1% |
| ≥192² | 862 | 98.8% | 92.3% | 91.0% |

Sınıf × boyut AP50-95 (n<5 ise '–'):

| sınıf | n_S | AP_S | n_M | AP_M | n_L | AP_L |
|---|---|---|---|---|---|---|
| aeroplane | 0 | – | 21 | 0.54 | 59 | 0.78 |
| bicycle | 0 | – | 12 | 0.42 | 50 | 0.79 |
| bird | 3 | – | 36 | 0.31 | 52 | 0.78 |
| boat | 4 | – | 25 | 0.53 | 35 | 0.65 |
| bottle | 13 | 0.13 | 52 | 0.47 | 30 | 0.74 |
| bus | 3 | – | 13 | 0.47 | 39 | 0.88 |
| car | 38 | 0.34 | 80 | 0.65 | 104 | 0.76 |
| cat | 0 | – | 2 | – | 86 | 0.78 |
| chair | 3 | – | 79 | 0.48 | 95 | 0.51 |
| cow | 0 | – | 25 | 0.70 | 31 | 0.70 |
| diningtable | 0 | – | 1 | – | 48 | 0.58 |
| dog | 0 | – | 10 | 0.61 | 86 | 0.81 |
| horse | 3 | – | 15 | 0.23 | 48 | 0.75 |
| motorbike | 3 | – | 10 | 0.51 | 54 | 0.67 |
| person | 51 | 0.20 | 215 | 0.48 | 519 | 0.72 |
| pottedplant | 12 | 0.23 | 57 | 0.29 | 40 | 0.37 |
| sheep | 8 | 0.42 | 16 | 0.65 | 33 | 0.86 |
| sofa | 0 | – | 1 | – | 58 | 0.61 |
| train | 0 | – | 4 | – | 53 | 0.72 |
| tvmonitor | 2 | – | 30 | 0.55 | 40 | 0.65 |

Küçük nesneler (n=143, val GT'nin %5.9'i): AP50-95 0.242 vs büyük 0.705 (fark 46.4 puan); <e-posta> 0.784 vs 0.979. Not: ölçüm orijinal piksel; eğitim/çıkarım 640 px letterbox olduğundan (~1.3× büyütme) S nesneler modelde ≈<41 px görünür. Val'de S örneği az olduğundan S-AP güven aralığı geniştir.

## 3. Hata türü ayrıştırması (TIDE benzeri)

Sayılar: tüm tahminler (conf≥0.001) ve işletim noktası (skor≥0.25). `Miss` tüm conf'ta hiçbir kutuyla IoU≥0.5 eşleşmeyen GT; işletim noktası sütununda ise o eşikte bulunamayan GT (eşleşmeyen + düşük skorla eşleşen).

| tür | sayı (tüm conf) | oran | sayı skor≥0.25 |
|---|---|---|---|
| TP | 2325 | 5.3% | 1905 |
| Dupe | 2518 | 5.7% | 82 |
| Cls | 3506 | 8.0% | 96 |
| Loc | 7997 | 18.2% | 153 |
| Both | 5558 | 12.6% | 76 |
| Bkg | 22104 | 50.2% | 266 |
| Miss | 82 | 3.4% (GT'nin) | 502 |

Oracle ΔAP (puan): ilgili hata türü düzeltilirse mAP ne kadar artardı (AP50 ve AP50-95, sınıf ortalaması). Taban: AP50=81.26, AP50-95=62.93.

| düzeltilen | ΔAP50 (puan) | ΔAP50-95 (puan) |
|---|---|---|
| Cls | 2.66 | 1.96 |
| Loc | 2.76 | 5.54 |
| Both | 1.14 | 0.77 |
| Dupe | 0.56 | -0.85 |
| Bkg | 3.54 | 2.28 |
| Miss | 3.10 | 2.29 |
| FP_hepsi | 14.63 | 6.56 |
| Kutu_kusursuz | 0.00 | 18.34 |
| TP_IoU_sirali | 1.57 | 5.88 |

Protokol etkisi: val'de etiketlerden çıkarılmış difficult nesne sayısı 493 (XML). Bunlardan biriyle aynı sınıf IoU≥0.5 örtüşen ve eşleşmemiş (FP sayılan) tahmin: 895 (skor≥0.25: 131; işletim noktası FP'lerinin 19.5%'i). Bu tahminler devkit gibi yok sayılsaydı ΔAP50 = +1.34, ΔAP50-95 = +0.91 puan (Ultralytics protokolünün val üzerinde ölçülmüş cezası; makalede belirtilmeli, Bkg/Loc/Dupe sayılarının bir kısmı bu yüzden yapaydır).


Not: `Dupe` ve `Loc` ΔAP50-95 değerleri IoU=0.5 tanımına göredir; bu kutular daha yüksek IoU eşiklerinde TP olabildiği için silme oracle'ı (Dupe) AP50-95'i düşürebilir; Loc oracle'ı kutuyu GT'ye yapıştırdığı için AP50-95 kazancı AP50'den büyüktür.


Hata türü × boyut (işletim noktası; tam uzun biçim `hata-kirilim.csv`):

| tür | S | M | L |
|---|---|---|---|
| TP (skor≥0.25, tahmin kutusu alanı) | 48 | 489 | 1368 |
| Dupe (skor≥0.25, tahmin kutusu alanı) | 0 | 18 | 64 |
| Cls (skor≥0.25, tahmin kutusu alanı) | 3 | 12 | 81 |
| Loc (skor≥0.25, tahmin kutusu alanı) | 5 | 63 | 85 |
| Both (skor≥0.25, tahmin kutusu alanı) | 1 | 15 | 60 |
| Bkg (skor≥0.25, tahmin kutusu alanı) | 49 | 123 | 94 |
| Miss (skor≥0.25 bulunamayan GT, GT alanı) | 93 | 207 | 202 |
| GT sayısı | 143 | 704 | 1560 |

AP50'ye en çok mal olan hata türü: Bkg (+3.54), sonra Miss (+3.10) ve Loc (+2.76). Oracle'lar birbirinden bağımsız ölçüldü, toplamları ΔAP'ye eşit değildir (etkileşim). `Kutu_kusursuz` = tüm TP kutuları GT'ye yapıştırılırsa AP50-95'in erişeceği üst sınır; `TP_IoU_sirali` = yalnız TP'lerin skorları IoU sırasına göre dağıtılırsa (FP skorları aynı) kazanç, yani IoU-farkında skorlamanın (VFL/QFL/IoU-aware) teorik payı.


Sınıf başına ΔAP50 (puan, oracle):

| sınıf | Cls | Loc | Both | Dupe | Bkg | Miss |
|---|---|---|---|---|---|---|
| aeroplane | 0.4 | 0.7 | 0.2 | 0.4 | 1.6 | 0.0 |
| bicycle | 0.4 | 1.4 | 0.4 | 0.2 | 0.5 | 3.4 |
| bird | 2.1 | 6.4 | 0.2 | 0.6 | 3.1 | 5.4 |
| boat | 0.4 | 4.6 | 0.3 | 0.7 | 4.6 | 3.7 |
| bottle | 0.6 | 2.7 | 0.3 | 0.5 | 11.4 | 3.0 |
| bus | 4.1 | 0.3 | 0.3 | 0.7 | 1.7 | 6.8 |
| car | 1.0 | 1.8 | 0.7 | 0.4 | 3.3 | 3.2 |
| cat | 2.9 | 1.7 | 0.2 | 0.3 | 0.3 | 0.0 |
| chair | 3.3 | 3.3 | 1.8 | 0.7 | 6.8 | 1.2 |
| cow | 6.9 | 3.1 | 0.2 | 0.4 | 3.7 | 1.7 |
| diningtable | 1.7 | 1.3 | 9.6 | 0.6 | 3.5 | 4.8 |
| dog | 2.2 | 1.3 | 0.4 | 0.3 | 0.5 | 1.1 |
| horse | 11.3 | 2.7 | 0.4 | 0.5 | 0.6 | 8.2 |
| motorbike | 2.8 | 4.4 | 0.9 | 0.2 | 2.4 | 3.7 |
| person | 0.0 | 3.3 | 0.2 | 0.8 | 2.1 | 2.5 |
| pottedplant | 0.8 | 7.5 | 2.1 | 1.4 | 7.3 | 7.8 |
| sheep | 3.7 | 0.0 | 0.3 | 0.2 | 3.1 | 0.0 |
| sofa | 7.7 | 0.7 | 2.7 | 1.6 | 4.3 | 0.0 |
| train | 0.2 | 7.2 | 0.2 | 0.2 | 2.7 | 3.3 |
| tvmonitor | 0.6 | 0.7 | 1.3 | 0.4 | 7.1 | 2.2 |

Kaçırılan GT (82) alt kırılımı: yalnız başka sınıfla örtüşen kutu var (sınıf karışıklığı) 20; aynı sınıf kutu var ama IoU∈[0.1,0.5) (lokalizasyon) 33; aynı sınıf IoU≥0.5 kutu var ama başka GT'ye atanmış (atama çakışması/örtüşme) 1; hiç kutu yok (gerçek kaçırma) 28 (kalan sınıflama kesin değil: bu üç grup kesişebilir).

## 4. Lokalizasyon kalitesi

| IoU aralığı | TP (tüm conf) | oran | TP skor≥0.25 | oran |
|---|---|---|---|---|
| 0.50–0.55 | 84 | 3.6% | 36 | 1.9% |
| 0.55–0.60 | 77 | 3.3% | 37 | 1.9% |
| 0.60–0.65 | 99 | 4.3% | 51 | 2.7% |
| 0.65–0.70 | 108 | 4.6% | 68 | 3.6% |
| 0.70–0.75 | 148 | 6.4% | 97 | 5.1% |
| 0.75–0.80 | 164 | 7.1% | 117 | 6.1% |
| 0.80–0.85 | 226 | 9.7% | 182 | 9.6% |
| 0.85–0.90 | 311 | 13.4% | 272 | 14.3% |
| 0.90–0.95 | 534 | 23.0% | 488 | 25.6% |
| 0.95–1.00 | 574 | 24.7% | 557 | 29.2% |

TP IoU ortalaması 0.847 (skor≥0.25: 0.872); işletim noktası TP'lerinin %15.2'i IoU<0.75. TP'lerde skor–IoU Spearman ρ=0.61 (skorun konum kalitesini ne kadar taşıdığı).

Sınıf başına AP50 − AP75 (büyükten küçüğe) ve TP ort. IoU:

| sınıf | AP50 | AP75 | AP50−AP75 (puan) | TP ort. IoU |
|---|---|---|---|---|
| pottedplant | 0.585 | 0.261 | 32.4 | 0.757 |
| person | 0.883 | 0.670 | 21.3 | 0.837 |
| aeroplane | 0.915 | 0.728 | 18.7 | 0.843 |
| boat | 0.782 | 0.632 | 15.0 | 0.833 |
| sofa | 0.759 | 0.614 | 14.5 | 0.870 |
| car | 0.845 | 0.700 | 14.4 | 0.848 |
| bicycle | 0.927 | 0.786 | 14.1 | 0.864 |
| horse | 0.801 | 0.663 | 13.8 | 0.856 |
| motorbike | 0.807 | 0.673 | 13.5 | 0.841 |
| sheep | 0.923 | 0.789 | 13.4 | 0.885 |
| bird | 0.768 | 0.634 | 13.3 | 0.837 |
| chair | 0.655 | 0.543 | 11.3 | 0.821 |
| cat | 0.930 | 0.819 | 11.1 | 0.899 |
| tvmonitor | 0.813 | 0.715 | 9.8 | 0.852 |
| train | 0.856 | 0.759 | 9.6 | 0.883 |
| dog | 0.927 | 0.844 | 8.3 | 0.899 |
| bottle | 0.692 | 0.610 | 8.2 | 0.839 |
| bus | 0.835 | 0.775 | 6.0 | 0.906 |
| diningtable | 0.701 | 0.646 | 5.5 | 0.874 |
| cow | 0.850 | 0.808 | 4.1 | 0.884 |

AP50→AP75 düşüşü sınıf ortalaması 12.9 puan; en büyük düşüş pottedplant (32), person (21), aeroplane (19). Kutu-regresyon (IoU/DFL/box kazancı) manipülasyonlarının hedef alanı budur.

## 5. Sınıf karışıklığı

İşletim noktası (skor≥0.25, IoU≥0.5): her GT için eşleşen en yüksek skorlu kutu (doğru sınıf varsa o). Matris dosyası: `hata-analizi_20261002-070042_h08_boxcp_doz_s1_karisiklik.csv`.

| GT sınıfı → tahmin | sayı | GT'nin payı |
|---|---|---|
| sofa → chair | 5 | 8.5% |
| chair → person | 4 | 2.3% |
| cow → horse | 4 | 7.1% |
| horse → sheep | 4 | 6.1% |
| dog → cat | 3 | 3.1% |
| bus → car | 2 | 3.6% |
| chair → sofa | 2 | 1.1% |
| cow → sheep | 2 | 3.6% |
| diningtable → chair | 2 | 4.1% |
| motorbike → person | 2 | 3.0% |
| sofa → person | 2 | 3.4% |
| bird → cat | 1 | 1.1% |

Çift bazında (yönsüz) toplam: chair/sofa 7; chair/person 4; cow/horse 4; horse/sheep 4; cat/dog 4; bus/car 3; chair/diningtable 3; person/sofa 3.


Tüm-conf alternatif görünüm: `Cls` türü tahminler (IoU≥0.5 başka sınıf GT, skor≥0.05), GT sınıfı → tahmin:

| GT → tahmin | tahmin sayısı |
|---|---|
| person → chair | 30 |
| sofa → chair | 26 |
| person → sofa | 18 |
| cow → horse | 12 |
| chair → sofa | 12 |
| cat → dog | 11 |
| dog → cat | 11 |
| horse → sheep | 8 |
| cow → sheep | 7 |
| chair → diningtable | 7 |

Kanonik çiftler (yönsüz; `Cls` tahmin sayısı skor≥0.05 | işletim noktası GT sayısı): cat/dog 22 | 4; cow/sheep 13 | 2; chair/diningtable 12 | 3; chair/sofa 38 | 7; horse/cow 17 | 4; bus/train 2 | 0; motorbike/bicycle 8 | 1.

İşletim noktasında GT'lerin 2.0%'i başka sınıf olarak etiketlendi (47/2407); 18.3%'i hiç bulunamadı. Sınıf karışıklığı oracle ΔAP50 = 2.66 puan (§3; Cls, altı hata türü içinde 4. sırada); işletim noktasında karışan GT oranı yalnız 2.0%, yani sorun sınıf ayrımından çok kutu/skor düzeyindedir.

## 6. Kalabalık ve örtüşme

| nesne/görüntü | görüntü | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) | FP/görüntü (skor≥0.25) |
|---|---|---|---|---|---|---|
| 1 | 440 | 440 | 98.6% | 92.5% | 91.4% | 0.35 |
| 2 | 252 | 504 | 97.6% | 83.3% | 87.9% | 0.58 |
| 3–4 | 195 | 654 | 96.3% | 76.0% | 81.5% | 0.91 |
| 5–8 | 88 | 511 | 94.9% | 70.5% | 78.1% | 1.33 |
| ≥9 | 25 | 298 | 95.3% | 74.2% | 71.1% | 3.16 |

Aynı sınıftan başka GT ile en yüksek IoU (örtüşme) gruplarına göre recall:

| komşu GT IoU | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 |
|---|---|---|---|
| <0.1 (komşu yok) | 1908 | 96.7% | 80.6% |
| 0.1–0.3 | 357 | 95.2% | 73.9% |
| 0.3–0.5 | 105 | 98.1% | 73.3% |
| 0.5–0.7 | 33 | 100.0% | 69.7% |
| ≥0.7 | 4 | 100.0% | 75.0% |

Aynı sınıf GT çiftleri (IoU≥0.3): 75 çift; ikisi de bulundu 73, yalnız biri 2, hiçbiri 0. IoU≥0.7 (NMS eşiği üstü, doğrudan NMS etkilenir): 2 çift, ikisi de bulundu 2, yalnız biri 0, hiçbiri 0. Kaçırılan GT'lerin 1/82'i aynı sınıf IoU≥0.5 bir kutuya sahip ama o kutu başka GT'ye atanmış (olası NMS/atama kaybı).

Kalabalıkla recall düşüşü: tek nesneli görüntüde 92.5% iken ≥9 nesnelide 74.2% (skor≥0.25); tüm-conf recall 98.6% → 95.3%, <e-posta> 91.4% → 71.1%. Çiftlerde tek-bulunma ve atama çakışması sayıları küçükse kayıp NMS'ten değil düşük skor/zayıf konumdan gelir.

## 7. Kalibrasyon (skor vs gerçek precision)

Tahmin 'doğru' = IoU≥0.5 ile eşleşen (<e-posta>); ayrıca IoU≥0.75 için precision. Not: YOLO sınıf skoru olasılık değil, TAL hizalama hedefli (IoU'ya bağlı) bir skordur; düşük skor bölgesindeki düşük precision 'aday havuzu' doğasından gelir.

| skor aralığı | tahmin | ort. skor | <e-posta> | <e-posta> | TP ort. IoU | skor−precision |
|---|---|---|---|---|---|---|
| 0.0–0.1 | 40339 | 0.008 | 0.007 | 0.006 | 0.726 | 0.001 |
| 0.1–0.2 | 860 | 0.141 | 0.129 | 0.090 | 0.739 | 0.011 |
| 0.2–0.3 | 391 | 0.245 | 0.197 | 0.141 | 0.772 | 0.048 |
| 0.3–0.4 | 259 | 0.349 | 0.274 | 0.205 | 0.777 | 0.075 |
| 0.4–0.5 | 210 | 0.450 | 0.414 | 0.295 | 0.790 | 0.035 |
| 0.5–0.6 | 202 | 0.549 | 0.584 | 0.396 | 0.798 | -0.035 |
| 0.6–0.7 | 244 | 0.650 | 0.693 | 0.508 | 0.811 | -0.043 |
| 0.7–0.8 | 315 | 0.756 | 0.854 | 0.679 | 0.840 | -0.098 |
| 0.8–0.9 | 656 | 0.857 | 0.957 | 0.864 | 0.886 | -0.101 |
| 0.9–1.0 | 532 | 0.924 | 0.989 | 0.962 | 0.940 | -0.065 |

ECE (tüm tahminler, 10 eşit genişlikli kutu) = 0.006; ECE (skor≥0.1) = 0.056. `skor−precision` pozitifse aşırı güvenli, negatifse eksik güvenli. 
Skor≥0.5 tahminlerin <e-posta> = 87.7%, <e-posta> = 76.8% (fark 10.9 puan). Skor 0.5 üstü kutularda (skor−precision) ortalaması -0.068: model yüksek skorlarda eksik güvenli (skor < gerçek <e-posta>), sıralama iyi; asıl boşluk <e-posta> ile @0.75 arasındaki konum kalitesidir. TP'lerde skor–IoU Spearman ρ=0.61: skor konum kalitesini kısmen taşıyor. TP_IoU_sirali oracle'ı (§3) IoU-farkında skorlamanın teorik payını sayısallaştırıyor.

## 8. Eğitim dinamiği (results.csv)

| gösterge | değer |
|---|---|
| epoch sayısı | 100 |
| en iyi epoch (mAP50-95) | 94 (0.6288) |
| son epoch mAP50-95 | 0.6281 |
| epoch 50 / 80 / 90 / 100 | 0.6167 / 0.6280 / 0.6284 / 0.6281 |
| son 20 epoch eğimi (puan/10 epoch) | 0.014 |
| önceki 20 epoch (ep 71-90) eğimi | 0.119 |
| son 20 epoch std (puan) | 0.032 |
| ep 81-90 ort. vs ep 91-100 ort. (close_mosaic=10) | 0.6281 vs 0.6284 (Δ +0.03 puan) |
| mAP50 son epoch | 0.8164 |
| results.csv en iyi ep mAP50-95 vs metrics.json (best.pt yeniden val) | 0.6288 vs 0.6300 (fark -0.12 puan; ölçüm-protokolü gürültü tabanı) |

Loss tablosu (train loss mosaic açıkken ep1/ep90, kapalıyken ep100; val/train oranı):

| loss | train ep1 | train ep90 | train ep100 | val ep100 | val/train ep100 | val/train ep90 | train loss ep90→91 değişim |
|---|---|---|---|---|---|---|---|
| box | 0.967 | 0.765 | 0.695 | 0.876 | 1.26 | 1.15 | -1.9% |
| cls | 2.289 | 0.789 | 0.548 | 0.791 | 1.44 | 1.01 | -19.2% |
| dfl | 1.244 | 1.110 | 1.060 | 1.165 | 1.10 | 1.05 | -0.5% |

Kayıp payı (results.csv loss_items kazançlı; loss.py:469-471 — önceki sürüm kazançları iki kez uyguluyordu, 2026-10-01 düzeltildi) ep100: box 30.2%, cls 23.8%, dfl 46.0% (ep1: box 21.5%, cls 50.9%, dfl 27.6%). Son 20 epoch mAP eğimi +0.014 puan/10 ep (std 0.03) → doygun (eğim seed gürültüsü altında). close_mosaic sonrası (ep 91-100) ortalama +0.03 puan. Train loss mosaic kapanınca ani düşer (cls -19.2%); bu rejim değişimi, train/val farkının bir kısmının augmentation kaynaklı olduğunu gösterir. val cls loss son 20 epoch 0.794→0.791 (yükselmiyor): val loss bozulmadığı için klasik aşırı öğrenme yok, ama mosaic kapalıyken val/train oranı cls 1.44, box 1.26: train kaybı val kaybından ayrışıyor (genelleme açığı); val mAP doygun.

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
| ön-işleme ms | 1.2536 |
| çıkarım ms | 6.0762 |
| NMS/son-işlem ms | 0.6907 |
| toplam ms | 8.0205 |
| FPS | 124.68 |
| batch | 1 |
| FP16 (1=evet) | 1.0 |
| ölçüm görüntüsü | 300 |
| ısınma görüntüsü | 50 |
| eğitim süresi (saat) | 2.936 |

Varyant hız bütçesi (yalnız tahmin; gerçek ölçüm varyant koşumunda yapılır):

| varyant sınıfı | çarpan | tahmini toplam ms | tahmini FPS | not |
|---|---|---|---|---|
| eğitim-zamanı (aug/loss) varyantları | aynı | 8.02 | 125 | çıkarım grafiği değişmez; hız bütçesi ≈ 0 |
| Soft-NMS/DIoU-NMS (son-işlem ×1–5 varsayımı) | NMS ×1–5 | 8.02–10.78 | 93–125 | tahmin (ölçülmedi) |
| TTA (Ultralytics augment, ~3 ölçek+flip; literatür çarpanı 3.6× çıkarım) | çıkarım ×3.6 (literatür) | 23.82 | 42 | tahmin; literatur-cipa.md §4 (Ultralytics TTA dokümanı) |
| YOLO11s'e geçiş (bağlam satırı) | ≈3.4× GFLOPs | – | – | ölçülmedi |

Baseline batch=1 FP16 toplam 8.02 ms (125 FPS; çıkarım 6.08 ms, NMS 0.69 ms = toplamın %9'i). 'Gerçek zamanlı' için ≥30 FPS kabul edilirse (33 ms) ~4.2× gecikme payı vardır; TTA bile (~42 FPS tahmini) eşiğin üstünde kalabilir, ancak hız-doğruluk satırı ayrı raporlanmalıdır. Not: hız ölçümü GPU'nun paylaşıldığı bir anda alınmış olabilir (eğitim kuyruğu); karşılaştırmalı hız ölçümü aynı koşulda yapılmalıdır.
