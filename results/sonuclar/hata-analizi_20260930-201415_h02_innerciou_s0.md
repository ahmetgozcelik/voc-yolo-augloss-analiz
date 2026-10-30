# Hata analizi (K3.5) — baseline `20260930-201415_h02_innerciou_s0`

*Üretici: `scripts/analiz/hata_analizi.py` · yalnız VAL (1000 görüntü, 2407 nesne; difficult nesneler etiketlerde yok) · tahmin sayısı 41520 (conf=0.001, max_det=300, NMS IoU=0.7) · eşleştirme: pycocotools COCOeval · işletim noktası skor≥0.25.*

**Sızıntı notu:** Bu analizde test kümesi (VOC07 test), `rep_test*` dosyaları ve `metrics.json` içindeki `rep_*` anahtarları hiç kullanılmadı; betik bu anahtarları yükler yüklemez atar. Buradan çıkan hipotezler yalnız val bilgisine dayanır.

## 0. Genel ölçüler ve doğrulama

| ölçü | pycocotools (bu betik) | Ultralytics metrics.json (best.pt val) | fark |
|---|---|---|---|
| mAP50-95 | 0.6262 | 0.6267 | -0.0005 |
| mAP50 | 0.8094 | 0.8117 | -0.0022 |
| mAP75 (yalnız pycocotools) | 0.6755 | – | – |

AP50 ile AP50-95 arasındaki fark 18.3 puan; AP50→AP75 farkı 13.4 puan. pycocotools ile Ultralytics AP'si arasındaki küçük fark interpolasyon/eşleştirme (Ultralytics kendi IoU-matrisi eşleşmesini kullanır) kaynaklıdır.

## 1. Sınıf başına AP ve sınıf frekansı

| sınıf (AP50-95 artan) | AP50-95 | AP50 | AP75 | AP50−AP75 | train nesne | val nesne | val GT medyan kenar px |
|---|---|---|---|---|---|---|---|
| pottedplant | 0.283 | 0.546 | 0.222 | 0.324 | 1378 | 109 | 71 |
| chair | 0.450 | 0.618 | 0.506 | 0.112 | 2975 | 177 | 99 |
| bottle | 0.524 | 0.690 | 0.588 | 0.102 | 1669 | 95 | 70 |
| diningtable | 0.531 | 0.674 | 0.602 | 0.072 | 775 | 49 | 211 |
| boat | 0.570 | 0.791 | 0.654 | 0.138 | 1076 | 64 | 112 |
| tvmonitor | 0.598 | 0.832 | 0.644 | 0.189 | 1036 | 72 | 108 |
| bird | 0.607 | 0.780 | 0.650 | 0.130 | 1514 | 91 | 115 |
| sofa | 0.620 | 0.750 | 0.656 | 0.093 | 755 | 59 | 237 |
| motorbike | 0.621 | 0.822 | 0.641 | 0.182 | 985 | 67 | 239 |
| horse | 0.625 | 0.820 | 0.672 | 0.148 | 1006 | 66 | 209 |
| person | 0.642 | 0.891 | 0.689 | 0.202 | 12471 | 785 | 141 |
| car | 0.651 | 0.851 | 0.714 | 0.138 | 3045 | 222 | 92 |
| cow | 0.677 | 0.857 | 0.796 | 0.061 | 791 | 56 | 145 |
| bicycle | 0.698 | 0.909 | 0.761 | 0.148 | 1002 | 62 | 171 |
| sheep | 0.708 | 0.866 | 0.752 | 0.114 | 1013 | 57 | 119 |
| aeroplane | 0.709 | 0.889 | 0.708 | 0.181 | 1091 | 80 | 184 |
| bus | 0.715 | 0.836 | 0.763 | 0.074 | 767 | 55 | 192 |
| train | 0.747 | 0.903 | 0.805 | 0.097 | 868 | 57 | 225 |
| cat | 0.767 | 0.928 | 0.825 | 0.103 | 1505 | 88 | 278 |
| dog | 0.782 | 0.934 | 0.863 | 0.071 | 1929 | 96 | 211 |

En zayıf 5 sınıf (AP50-95): pottedplant 0.283, chair 0.450, bottle 0.524, diningtable 0.531, boat 0.570. Sınıf frekansı (train nesne) ile AP50-95 Spearman ρ=-0.12 (permütasyon p=0.619), AP50 ile ρ=0.08; sınıf başına train küçük-nesne oranı ile ρ=-0.41; val GT medyan kenar uzunluğu ile ρ=0.56 (n=20 sınıf, gürültülü). Frekans AP'nin zayıf belirleyicisi (|ρ|<0.3): zayıf sınıflarda sorun veri azlığından çok nesne boyutu/görünümü/bağlamıdır.

## 2. Nesne boyutu kırılımı (orijinal çözünürlük; COCO S<32²≤M<96²≤L)

| boyut | GT sayısı | AP50-95 | AP50 | <e-posta> (tüm conf, sınıf ort.) | <e-posta>:0.95 (sınıf ort.) | <e-posta> skor≥0.25 |
|---|---|---|---|---|---|---|
| all | 2407 | 0.626 | 0.809 | 0.957 | 0.764 | 0.785 |
| S | 143 | 0.258 | 0.425 | 0.734 | 0.464 | 0.399 |
| M | 704 | 0.433 | 0.635 | 0.910 | 0.654 | 0.689 |
| L | 1560 | 0.702 | 0.870 | 0.973 | 0.813 | 0.863 |

İnce alan aralıkları (GT düzeyi, eşleşme bayrağı):

| alan (px²) | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) |
|---|---|---|---|---|
| <16² | 9 | 55.6% | 0.0% | 22.2% |
| 16²–32² | 134 | 87.3% | 42.5% | 57.5% |
| 32²–64² | 377 | 95.8% | 66.3% | 70.0% |
| 64²–96² | 327 | 94.8% | 71.9% | 79.8% |
| 96²–192² | 698 | 96.8% | 80.1% | 84.2% |
| ≥192² | 862 | 99.0% | 91.4% | 91.9% |

Sınıf × boyut AP50-95 (n<5 ise '–'):

| sınıf | n_S | AP_S | n_M | AP_M | n_L | AP_L |
|---|---|---|---|---|---|---|
| aeroplane | 0 | – | 21 | 0.56 | 59 | 0.76 |
| bicycle | 0 | – | 12 | 0.40 | 50 | 0.77 |
| bird | 3 | – | 36 | 0.39 | 52 | 0.78 |
| boat | 4 | – | 25 | 0.56 | 35 | 0.65 |
| bottle | 13 | 0.21 | 52 | 0.48 | 30 | 0.71 |
| bus | 3 | – | 13 | 0.41 | 39 | 0.87 |
| car | 38 | 0.33 | 80 | 0.63 | 104 | 0.77 |
| cat | 0 | – | 2 | – | 86 | 0.78 |
| chair | 3 | – | 79 | 0.44 | 95 | 0.48 |
| cow | 0 | – | 25 | 0.63 | 31 | 0.76 |
| diningtable | 0 | – | 1 | – | 48 | 0.54 |
| dog | 0 | – | 10 | 0.69 | 86 | 0.80 |
| horse | 3 | – | 15 | 0.21 | 48 | 0.77 |
| motorbike | 3 | – | 10 | 0.48 | 54 | 0.67 |
| person | 51 | 0.22 | 215 | 0.51 | 519 | 0.73 |
| pottedplant | 12 | 0.20 | 57 | 0.30 | 40 | 0.33 |
| sheep | 8 | 0.40 | 16 | 0.56 | 33 | 0.84 |
| sofa | 0 | – | 1 | – | 58 | 0.63 |
| train | 0 | – | 4 | – | 53 | 0.77 |
| tvmonitor | 2 | – | 30 | 0.56 | 40 | 0.64 |

Küçük nesneler (n=143, val GT'nin %5.9'i): AP50-95 0.258 vs büyük 0.702 (fark 44.4 puan); <e-posta> 0.734 vs 0.973. Not: ölçüm orijinal piksel; eğitim/çıkarım 640 px letterbox olduğundan (~1.3× büyütme) S nesneler modelde ≈<41 px görünür. Val'de S örneği az olduğundan S-AP güven aralığı geniştir.

## 3. Hata türü ayrıştırması (TIDE benzeri)

Sayılar: tüm tahminler (conf≥0.001) ve işletim noktası (skor≥0.25). `Miss` tüm conf'ta hiçbir kutuyla IoU≥0.5 eşleşmeyen GT; işletim noktası sütununda ise o eşikte bulunamayan GT (eşleşmeyen + düşük skorla eşleşen).

| tür | sayı (tüm conf) | oran | sayı skor≥0.25 |
|---|---|---|---|
| TP | 2322 | 5.6% | 1889 |
| Dupe | 2453 | 5.9% | 80 |
| Cls | 3333 | 8.0% | 100 |
| Loc | 7751 | 18.7% | 141 |
| Both | 5144 | 12.4% | 74 |
| Bkg | 20517 | 49.4% | 265 |
| Miss | 85 | 3.5% (GT'nin) | 518 |

Oracle ΔAP (puan): ilgili hata türü düzeltilirse mAP ne kadar artardı (AP50 ve AP50-95, sınıf ortalaması). Taban: AP50=80.94, AP50-95=62.62.

| düzeltilen | ΔAP50 (puan) | ΔAP50-95 (puan) |
|---|---|---|
| Cls | 3.07 | 2.16 |
| Loc | 2.35 | 5.08 |
| Both | 1.38 | 0.96 |
| Dupe | 0.52 | -0.79 |
| Bkg | 3.50 | 2.21 |
| Miss | 3.53 | 2.61 |
| FP_hepsi | 14.35 | 6.50 |
| Kutu_kusursuz | 0.00 | 18.32 |
| TP_IoU_sirali | 1.15 | 5.40 |

Protokol etkisi: val'de etiketlerden çıkarılmış difficult nesne sayısı 493 (XML). Bunlardan biriyle aynı sınıf IoU≥0.5 örtüşen ve eşleşmemiş (FP sayılan) tahmin: 842 (skor≥0.25: 123; işletim noktası FP'lerinin 18.6%'i). Bu tahminler devkit gibi yok sayılsaydı ΔAP50 = +1.35, ΔAP50-95 = +0.90 puan (Ultralytics protokolünün val üzerinde ölçülmüş cezası; makalede belirtilmeli, Bkg/Loc/Dupe sayılarının bir kısmı bu yüzden yapaydır).


Not: `Dupe` ve `Loc` ΔAP50-95 değerleri IoU=0.5 tanımına göredir; bu kutular daha yüksek IoU eşiklerinde TP olabildiği için silme oracle'ı (Dupe) AP50-95'i düşürebilir; Loc oracle'ı kutuyu GT'ye yapıştırdığı için AP50-95 kazancı AP50'den büyüktür.


Hata türü × boyut (işletim noktası; tam uzun biçim `hata-kirilim.csv`):

| tür | S | M | L |
|---|---|---|---|
| TP (skor≥0.25, tahmin kutusu alanı) | 53 | 467 | 1369 |
| Dupe (skor≥0.25, tahmin kutusu alanı) | 1 | 22 | 57 |
| Cls (skor≥0.25, tahmin kutusu alanı) | 2 | 22 | 76 |
| Loc (skor≥0.25, tahmin kutusu alanı) | 7 | 54 | 80 |
| Both (skor≥0.25, tahmin kutusu alanı) | 1 | 18 | 55 |
| Bkg (skor≥0.25, tahmin kutusu alanı) | 47 | 130 | 88 |
| Miss (skor≥0.25 bulunamayan GT, GT alanı) | 86 | 219 | 213 |
| GT sayısı | 143 | 704 | 1560 |

AP50'ye en çok mal olan hata türü: Miss (+3.53), sonra Bkg (+3.50) ve Cls (+3.07). Oracle'lar birbirinden bağımsız ölçüldü, toplamları ΔAP'ye eşit değildir (etkileşim). `Kutu_kusursuz` = tüm TP kutuları GT'ye yapıştırılırsa AP50-95'in erişeceği üst sınır; `TP_IoU_sirali` = yalnız TP'lerin skorları IoU sırasına göre dağıtılırsa (FP skorları aynı) kazanç, yani IoU-farkında skorlamanın (VFL/QFL/IoU-aware) teorik payı.


Sınıf başına ΔAP50 (puan, oracle):

| sınıf | Cls | Loc | Both | Dupe | Bkg | Miss |
|---|---|---|---|---|---|---|
| aeroplane | 0.6 | 2.8 | 0.0 | 0.9 | 1.1 | 3.6 |
| bicycle | 0.4 | 1.2 | 0.6 | 0.2 | 0.5 | 1.4 |
| bird | 4.2 | 5.0 | 0.2 | 1.0 | 2.6 | 6.3 |
| boat | 1.5 | 3.0 | 0.3 | 0.3 | 5.9 | 4.0 |
| bottle | 2.0 | 1.8 | 0.9 | 0.3 | 10.0 | 3.0 |
| bus | 5.0 | 0.5 | 0.2 | 0.5 | 2.2 | 6.8 |
| car | 1.2 | 1.8 | 0.7 | 0.5 | 3.6 | 2.8 |
| cat | 3.0 | 2.0 | 0.1 | 0.1 | 0.1 | 3.1 |
| chair | 4.6 | 3.8 | 2.2 | 0.9 | 6.8 | 2.5 |
| cow | 6.2 | 1.9 | 0.0 | 0.4 | 4.2 | 1.7 |
| diningtable | 1.0 | 1.0 | 13.9 | 0.2 | 2.4 | 6.1 |
| dog | 0.9 | 1.1 | 0.1 | 0.3 | 0.8 | 1.1 |
| horse | 11.9 | 0.7 | 0.5 | 0.8 | 1.0 | 4.0 |
| motorbike | 2.0 | 4.4 | 0.4 | 0.2 | 2.3 | 6.5 |
| person | 0.2 | 3.0 | 0.1 | 1.0 | 2.1 | 1.5 |
| pottedplant | 0.7 | 7.0 | 3.1 | 1.9 | 8.0 | 7.4 |
| sheep | 7.7 | 0.0 | 0.5 | 0.3 | 3.7 | 0.0 |
| sofa | 7.3 | 1.4 | 2.8 | 0.5 | 3.4 | 1.3 |
| train | 0.6 | 4.0 | 0.2 | 0.1 | 1.7 | 3.6 |
| tvmonitor | 0.5 | 0.7 | 0.8 | 0.0 | 7.6 | 3.7 |

Kaçırılan GT (85) alt kırılımı: yalnız başka sınıfla örtüşen kutu var (sınıf karışıklığı) 20; aynı sınıf kutu var ama IoU∈[0.1,0.5) (lokalizasyon) 37; aynı sınıf IoU≥0.5 kutu var ama başka GT'ye atanmış (atama çakışması/örtüşme) 3; hiç kutu yok (gerçek kaçırma) 25 (kalan sınıflama kesin değil: bu üç grup kesişebilir).

## 4. Lokalizasyon kalitesi

| IoU aralığı | TP (tüm conf) | oran | TP skor≥0.25 | oran |
|---|---|---|---|---|
| 0.50–0.55 | 74 | 3.2% | 30 | 1.6% |
| 0.55–0.60 | 80 | 3.4% | 41 | 2.2% |
| 0.60–0.65 | 93 | 4.0% | 41 | 2.2% |
| 0.65–0.70 | 125 | 5.4% | 82 | 4.3% |
| 0.70–0.75 | 151 | 6.5% | 101 | 5.3% |
| 0.75–0.80 | 166 | 7.1% | 114 | 6.0% |
| 0.80–0.85 | 210 | 9.0% | 170 | 9.0% |
| 0.85–0.90 | 335 | 14.4% | 289 | 15.3% |
| 0.90–0.95 | 526 | 22.7% | 474 | 25.1% |
| 0.95–1.00 | 562 | 24.2% | 547 | 29.0% |

TP IoU ortalaması 0.847 (skor≥0.25: 0.872); işletim noktası TP'lerinin %15.6'i IoU<0.75. TP'lerde skor–IoU Spearman ρ=0.63 (skorun konum kalitesini ne kadar taşıdığı).

Sınıf başına AP50 − AP75 (büyükten küçüğe) ve TP ort. IoU:

| sınıf | AP50 | AP75 | AP50−AP75 (puan) | TP ort. IoU |
|---|---|---|---|---|
| pottedplant | 0.546 | 0.222 | 32.4 | 0.760 |
| person | 0.891 | 0.689 | 20.2 | 0.834 |
| tvmonitor | 0.832 | 0.644 | 18.9 | 0.854 |
| motorbike | 0.822 | 0.641 | 18.2 | 0.840 |
| aeroplane | 0.889 | 0.708 | 18.1 | 0.848 |
| horse | 0.820 | 0.672 | 14.8 | 0.862 |
| bicycle | 0.909 | 0.761 | 14.8 | 0.850 |
| boat | 0.791 | 0.654 | 13.8 | 0.831 |
| car | 0.851 | 0.714 | 13.8 | 0.852 |
| bird | 0.780 | 0.650 | 13.0 | 0.847 |
| sheep | 0.866 | 0.752 | 11.4 | 0.884 |
| chair | 0.618 | 0.506 | 11.2 | 0.826 |
| cat | 0.928 | 0.825 | 10.3 | 0.903 |
| bottle | 0.690 | 0.588 | 10.2 | 0.836 |
| train | 0.903 | 0.805 | 9.7 | 0.888 |
| sofa | 0.750 | 0.656 | 9.3 | 0.873 |
| bus | 0.836 | 0.763 | 7.4 | 0.896 |
| diningtable | 0.674 | 0.602 | 7.2 | 0.864 |
| dog | 0.934 | 0.863 | 7.1 | 0.897 |
| cow | 0.857 | 0.796 | 6.1 | 0.881 |

AP50→AP75 düşüşü sınıf ortalaması 13.4 puan; en büyük düşüş pottedplant (32), person (20), tvmonitor (19). Kutu-regresyon (IoU/DFL/box kazancı) manipülasyonlarının hedef alanı budur.

## 5. Sınıf karışıklığı

İşletim noktası (skor≥0.25, IoU≥0.5): her GT için eşleşen en yüksek skorlu kutu (doğru sınıf varsa o). Matris dosyası: `hata-analizi_20260930-201415_h02_innerciou_s0_karisiklik.csv`.

| GT sınıfı → tahmin | sayı | GT'nin payı |
|---|---|---|
| chair → person | 7 | 4.0% |
| cow → sheep | 6 | 10.7% |
| horse → sheep | 6 | 9.1% |
| chair → diningtable | 4 | 2.3% |
| sofa → person | 4 | 6.8% |
| cow → horse | 3 | 5.4% |
| sofa → chair | 3 | 5.1% |
| bus → car | 2 | 3.6% |
| car → train | 2 | 0.9% |
| diningtable → chair | 2 | 4.1% |
| motorbike → car | 2 | 3.0% |
| motorbike → person | 2 | 3.0% |

Çift bazında (yönsüz) toplam: chair/person 8; cow/sheep 7; horse/sheep 6; chair/diningtable 6; person/sofa 4; chair/sofa 4; cow/horse 3; bus/car 3.


Tüm-conf alternatif görünüm: `Cls` türü tahminler (IoU≥0.5 başka sınıf GT, skor≥0.05), GT sınıfı → tahmin:

| GT → tahmin | tahmin sayısı |
|---|---|
| person → chair | 33 |
| sofa → chair | 23 |
| person → sofa | 21 |
| cow → horse | 13 |
| dog → cat | 13 |
| chair → sofa | 10 |
| horse → sheep | 10 |
| motorbike → bicycle | 9 |
| cow → sheep | 8 |
| car → bus | 8 |

Kanonik çiftler (yönsüz; `Cls` tahmin sayısı skor≥0.05 | işletim noktası GT sayısı): cat/dog 19 | 1; cow/sheep 13 | 7; chair/diningtable 10 | 6; chair/sofa 33 | 4; horse/cow 21 | 3; bus/train 4 | 1; motorbike/bicycle 12 | 0.

İşletim noktasında GT'lerin 2.7%'i başka sınıf olarak etiketlendi (64/2407); 18.3%'i hiç bulunamadı. Sınıf karışıklığı oracle ΔAP50 = 3.07 puan (§3; Cls, altı hata türü içinde 3. sırada); işletim noktasında karışan GT oranı yalnız 2.7%, yani sorun sınıf ayrımından çok kutu/skor düzeyindedir.

## 6. Kalabalık ve örtüşme

| nesne/görüntü | görüntü | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) | FP/görüntü (skor≥0.25) |
|---|---|---|---|---|---|---|
| 1 | 440 | 440 | 99.1% | 92.7% | 92.7% | 0.34 |
| 2 | 252 | 504 | 97.4% | 80.2% | 86.7% | 0.52 |
| 3–4 | 195 | 654 | 95.7% | 76.8% | 81.5% | 0.95 |
| 5–8 | 88 | 511 | 95.5% | 70.6% | 76.9% | 1.26 |
| ≥9 | 25 | 298 | 94.3% | 71.8% | 71.5% | 3.40 |

Aynı sınıftan başka GT ile en yüksek IoU (örtüşme) gruplarına göre recall:

| komşu GT IoU | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 |
|---|---|---|---|
| <0.1 (komşu yok) | 1908 | 96.5% | 79.1% |
| 0.1–0.3 | 357 | 96.1% | 76.2% |
| 0.3–0.5 | 105 | 99.0% | 77.1% |
| 0.5–0.7 | 33 | 87.9% | 69.7% |
| ≥0.7 | 4 | 100.0% | 75.0% |

Aynı sınıf GT çiftleri (IoU≥0.3): 75 çift; ikisi de bulundu 69, yalnız biri 5, hiçbiri 1. IoU≥0.7 (NMS eşiği üstü, doğrudan NMS etkilenir): 2 çift, ikisi de bulundu 2, yalnız biri 0, hiçbiri 0. Kaçırılan GT'lerin 3/85'i aynı sınıf IoU≥0.5 bir kutuya sahip ama o kutu başka GT'ye atanmış (olası NMS/atama kaybı).

Kalabalıkla recall düşüşü: tek nesneli görüntüde 92.7% iken ≥9 nesnelide 71.8% (skor≥0.25); tüm-conf recall 99.1% → 94.3%, <e-posta> 92.7% → 71.5%. Çiftlerde tek-bulunma ve atama çakışması sayıları küçükse kayıp NMS'ten değil düşük skor/zayıf konumdan gelir.

## 7. Kalibrasyon (skor vs gerçek precision)

Tahmin 'doğru' = IoU≥0.5 ile eşleşen (<e-posta>); ayrıca IoU≥0.75 için precision. Not: YOLO sınıf skoru olasılık değil, TAL hizalama hedefli (IoU'ya bağlı) bir skordur; düşük skor bölgesindeki düşük precision 'aday havuzu' doğasından gelir.

| skor aralığı | tahmin | ort. skor | <e-posta> | <e-posta> | TP ort. IoU | skor−precision |
|---|---|---|---|---|---|---|
| 0.0–0.1 | 37938 | 0.008 | 0.007 | 0.006 | 0.728 | 0.001 |
| 0.1–0.2 | 802 | 0.142 | 0.131 | 0.100 | 0.764 | 0.011 |
| 0.2–0.3 | 390 | 0.244 | 0.251 | 0.169 | 0.749 | -0.007 |
| 0.3–0.4 | 254 | 0.348 | 0.283 | 0.185 | 0.761 | 0.064 |
| 0.4–0.5 | 210 | 0.449 | 0.395 | 0.295 | 0.782 | 0.053 |
| 0.5–0.6 | 199 | 0.552 | 0.588 | 0.407 | 0.803 | -0.036 |
| 0.6–0.7 | 226 | 0.654 | 0.743 | 0.491 | 0.799 | -0.090 |
| 0.7–0.8 | 299 | 0.751 | 0.806 | 0.645 | 0.841 | -0.055 |
| 0.8–0.9 | 645 | 0.858 | 0.955 | 0.864 | 0.885 | -0.097 |
| 0.9–1.0 | 557 | 0.925 | 0.991 | 0.973 | 0.942 | -0.066 |

ECE (tüm tahminler, 10 eşit genişlikli kutu) = 0.005; ECE (skor≥0.1) = 0.051. `skor−precision` pozitifse aşırı güvenli, negatifse eksik güvenli. 
Skor≥0.5 tahminlerin <e-posta> = 88.0%, <e-posta> = 77.1% (fark 10.9 puan). Skor 0.5 üstü kutularda (skor−precision) ortalaması -0.069: model yüksek skorlarda eksik güvenli (skor < gerçek <e-posta>), sıralama iyi; asıl boşluk <e-posta> ile @0.75 arasındaki konum kalitesidir. TP'lerde skor–IoU Spearman ρ=0.63: skor konum kalitesini kısmen taşıyor. TP_IoU_sirali oracle'ı (§3) IoU-farkında skorlamanın teorik payını sayısallaştırıyor.

## 8. Eğitim dinamiği (results.csv)

| gösterge | değer |
|---|---|
| epoch sayısı | 100 |
| en iyi epoch (mAP50-95) | 100 (0.6259) |
| son epoch mAP50-95 | 0.6259 |
| epoch 50 / 80 / 90 / 100 | 0.6172 / 0.6236 / 0.6252 / 0.6259 |
| son 20 epoch eğimi (puan/10 epoch) | 0.048 |
| önceki 20 epoch (ep 71-90) eğimi | 0.142 |
| son 20 epoch std (puan) | 0.045 |
| ep 81-90 ort. vs ep 91-100 ort. (close_mosaic=10) | 0.6249 vs 0.6252 (Δ +0.03 puan) |
| mAP50 son epoch | 0.8091 |
| results.csv en iyi ep mAP50-95 vs metrics.json (best.pt yeniden val) | 0.6259 vs 0.6267 (fark -0.09 puan; ölçüm-protokolü gürültü tabanı) |

Loss tablosu (train loss mosaic açıkken ep1/ep90, kapalıyken ep100; val/train oranı):

| loss | train ep1 | train ep90 | train ep100 | val ep100 | val/train ep100 | val/train ep90 | train loss ep90→91 değişim |
|---|---|---|---|---|---|---|---|
| box | 1.179 | 0.947 | 0.806 | 1.004 | 1.25 | 1.06 | -8.1% |
| cls | 2.304 | 0.819 | 0.550 | 0.797 | 1.45 | 0.98 | -21.8% |
| dfl | 1.289 | 1.150 | 1.067 | 1.154 | 1.08 | 1.00 | -3.6% |

Kazançlarla (box 7.5, cls 0.5, dfl 1.5) ağırlıklı kayıp payı ep100: box 76.3%, cls 3.5%, dfl 20.2% (ep1: box 74.1%, cls 9.7%, dfl 16.2%). Son 20 epoch mAP eğimi +0.048 puan/10 ep (std 0.05) → doygun (eğim seed gürültüsü altında). close_mosaic sonrası (ep 91-100) ortalama +0.03 puan. Train loss mosaic kapanınca ani düşer (cls -21.8%); bu rejim değişimi, train/val farkının bir kısmının augmentation kaynaklı olduğunu gösterir. val cls loss son 20 epoch 0.801→0.797 (yükselmiyor): val loss bozulmadığı için klasik aşırı öğrenme yok, ama mosaic kapalıyken val/train oranı cls 1.45, box 1.25: train kaybı val kaybından ayrışıyor (genelleme açığı); val mAP doygun.

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
| ön-işleme ms | 1.2988 |
| çıkarım ms | 6.0553 |
| NMS/son-işlem ms | 0.6857 |
| toplam ms | 8.0398 |
| FPS | 124.38 |
| batch | 1 |
| FP16 (1=evet) | 1.0 |
| ölçüm görüntüsü | 300 |
| ısınma görüntüsü | 50 |
| eğitim süresi (saat) | 2.966 |

Varyant hız bütçesi (yalnız tahmin; gerçek ölçüm varyant koşumunda yapılır):

| varyant sınıfı | çarpan | tahmini toplam ms | tahmini FPS | not |
|---|---|---|---|---|
| eğitim-zamanı (aug/loss) varyantları | aynı | 8.04 | 124 | çıkarım grafiği değişmez; hız bütçesi ≈ 0 |
| Soft-NMS/DIoU-NMS (son-işlem ×1–5 varsayımı) | NMS ×1–5 | 8.04–10.78 | 93–124 | tahmin (ölçülmedi) |
| TTA (Ultralytics augment, ~3 ölçek+flip; literatür çarpanı 3.6× çıkarım) | çıkarım ×3.6 (literatür) | 23.78 | 42 | tahmin; literatur-cipa.md §4 (Ultralytics TTA dokümanı) |
| YOLO11s'e geçiş (bağlam satırı) | ≈3.4× GFLOPs | – | – | ölçülmedi |

Baseline batch=1 FP16 toplam 8.04 ms (124 FPS; çıkarım 6.06 ms, NMS 0.69 ms = toplamın %9'i). 'Gerçek zamanlı' için ≥30 FPS kabul edilirse (33 ms) ~4.1× gecikme payı vardır; TTA bile (~42 FPS tahmini) eşiğin üstünde kalabilir, ancak hız-doğruluk satırı ayrı raporlanmalıdır. Not: hız ölçümü GPU'nun paylaşıldığı bir anda alınmış olabilir (eğitim kuyruğu); karşılaştırmalı hız ölçümü aynı koşulda yapılmalıdır.
