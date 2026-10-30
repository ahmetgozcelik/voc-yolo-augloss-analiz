# Hata analizi (K3.5) — baseline `20261003-062922_h01_multiscale25_s1`

*Üretici: `scripts/analiz/hata_analizi.py` · yalnız VAL (1000 görüntü, 2407 nesne; difficult nesneler etiketlerde yok) · tahmin sayısı 43109 (conf=0.001, max_det=300, NMS IoU=0.7) · eşleştirme: pycocotools COCOeval · işletim noktası skor≥0.25.*

**Sızıntı notu:** Bu analizde test kümesi (VOC07 test), `rep_test*` dosyaları ve `metrics.json` içindeki `rep_*` anahtarları hiç kullanılmadı; betik bu anahtarları yükler yüklemez atar. Buradan çıkan hipotezler yalnız val bilgisine dayanır.

## 0. Genel ölçüler ve doğrulama

| ölçü | pycocotools (bu betik) | Ultralytics metrics.json (best.pt val) | fark |
|---|---|---|---|
| mAP50-95 | 0.6258 | 0.6263 | -0.0005 |
| mAP50 | 0.8091 | 0.8115 | -0.0024 |
| mAP75 (yalnız pycocotools) | 0.6706 | – | – |

AP50 ile AP50-95 arasındaki fark 18.3 puan; AP50→AP75 farkı 13.8 puan. pycocotools ile Ultralytics AP'si arasındaki küçük fark interpolasyon/eşleştirme (Ultralytics kendi IoU-matrisi eşleşmesini kullanır) kaynaklıdır.

## 1. Sınıf başına AP ve sınıf frekansı

| sınıf (AP50-95 artan) | AP50-95 | AP50 | AP75 | AP50−AP75 | train nesne | val nesne | val GT medyan kenar px |
|---|---|---|---|---|---|---|---|
| pottedplant | 0.292 | 0.561 | 0.247 | 0.314 | 1378 | 109 | 71 |
| chair | 0.444 | 0.628 | 0.479 | 0.149 | 2975 | 177 | 99 |
| bottle | 0.536 | 0.715 | 0.624 | 0.091 | 1669 | 95 | 70 |
| boat | 0.555 | 0.778 | 0.619 | 0.159 | 1076 | 64 | 112 |
| diningtable | 0.560 | 0.705 | 0.596 | 0.109 | 775 | 49 | 211 |
| tvmonitor | 0.581 | 0.808 | 0.633 | 0.175 | 1036 | 72 | 108 |
| horse | 0.592 | 0.771 | 0.634 | 0.138 | 1006 | 66 | 209 |
| bird | 0.602 | 0.798 | 0.645 | 0.153 | 1514 | 91 | 115 |
| sofa | 0.617 | 0.742 | 0.633 | 0.109 | 755 | 59 | 237 |
| person | 0.630 | 0.885 | 0.674 | 0.211 | 12471 | 785 | 141 |
| motorbike | 0.639 | 0.818 | 0.653 | 0.166 | 985 | 67 | 239 |
| cow | 0.647 | 0.828 | 0.733 | 0.095 | 791 | 56 | 145 |
| car | 0.653 | 0.841 | 0.721 | 0.120 | 3045 | 222 | 92 |
| aeroplane | 0.703 | 0.893 | 0.716 | 0.177 | 1091 | 80 | 184 |
| bicycle | 0.703 | 0.896 | 0.767 | 0.129 | 1002 | 62 | 171 |
| sheep | 0.711 | 0.875 | 0.758 | 0.117 | 1013 | 57 | 119 |
| bus | 0.712 | 0.839 | 0.735 | 0.104 | 767 | 55 | 192 |
| train | 0.761 | 0.921 | 0.823 | 0.098 | 868 | 57 | 225 |
| dog | 0.771 | 0.941 | 0.852 | 0.089 | 1929 | 96 | 211 |
| cat | 0.807 | 0.938 | 0.872 | 0.065 | 1505 | 88 | 278 |

En zayıf 5 sınıf (AP50-95): pottedplant 0.292, chair 0.444, bottle 0.536, boat 0.555, diningtable 0.560. Sınıf frekansı (train nesne) ile AP50-95 Spearman ρ=-0.15 (permütasyon p=0.535), AP50 ile ρ=0.10; sınıf başına train küçük-nesne oranı ile ρ=-0.43; val GT medyan kenar uzunluğu ile ρ=0.58 (n=20 sınıf, gürültülü). Frekans AP'nin zayıf belirleyicisi (|ρ|<0.3): zayıf sınıflarda sorun veri azlığından çok nesne boyutu/görünümü/bağlamıdır.

## 2. Nesne boyutu kırılımı (orijinal çözünürlük; COCO S<32²≤M<96²≤L)

| boyut | GT sayısı | AP50-95 | AP50 | <e-posta> (tüm conf, sınıf ort.) | <e-posta>:0.95 (sınıf ort.) | <e-posta> skor≥0.25 |
|---|---|---|---|---|---|---|
| all | 2407 | 0.626 | 0.809 | 0.959 | 0.765 | 0.789 |
| S | 143 | 0.223 | 0.410 | 0.757 | 0.450 | 0.413 |
| M | 704 | 0.445 | 0.644 | 0.924 | 0.664 | 0.699 |
| L | 1560 | 0.700 | 0.867 | 0.978 | 0.815 | 0.864 |

İnce alan aralıkları (GT düzeyi, eşleşme bayrağı):

| alan (px²) | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) |
|---|---|---|---|---|
| <16² | 9 | 55.6% | 0.0% | 11.1% |
| 16²–32² | 134 | 85.8% | 44.0% | 56.0% |
| 32²–64² | 377 | 96.0% | 67.1% | 73.2% |
| 64²–96² | 327 | 94.5% | 73.1% | 79.2% |
| 96²–192² | 698 | 97.3% | 80.5% | 82.2% |
| ≥192² | 862 | 98.8% | 91.2% | 90.1% |

Sınıf × boyut AP50-95 (n<5 ise '–'):

| sınıf | n_S | AP_S | n_M | AP_M | n_L | AP_L |
|---|---|---|---|---|---|---|
| aeroplane | 0 | – | 21 | 0.55 | 59 | 0.76 |
| bicycle | 0 | – | 12 | 0.40 | 50 | 0.78 |
| bird | 3 | – | 36 | 0.36 | 52 | 0.76 |
| boat | 4 | – | 25 | 0.55 | 35 | 0.62 |
| bottle | 13 | 0.19 | 52 | 0.52 | 30 | 0.71 |
| bus | 3 | – | 13 | 0.41 | 39 | 0.86 |
| car | 38 | 0.33 | 80 | 0.64 | 104 | 0.77 |
| cat | 0 | – | 2 | – | 86 | 0.83 |
| chair | 3 | – | 79 | 0.44 | 95 | 0.47 |
| cow | 0 | – | 25 | 0.66 | 31 | 0.70 |
| diningtable | 0 | – | 1 | – | 48 | 0.57 |
| dog | 0 | – | 10 | 0.75 | 86 | 0.78 |
| horse | 3 | – | 15 | 0.20 | 48 | 0.74 |
| motorbike | 3 | – | 10 | 0.53 | 54 | 0.69 |
| person | 51 | 0.21 | 215 | 0.49 | 519 | 0.72 |
| pottedplant | 12 | 0.21 | 57 | 0.28 | 40 | 0.36 |
| sheep | 8 | 0.44 | 16 | 0.55 | 33 | 0.84 |
| sofa | 0 | – | 1 | – | 58 | 0.63 |
| train | 0 | – | 4 | – | 53 | 0.78 |
| tvmonitor | 2 | – | 30 | 0.52 | 40 | 0.65 |

Küçük nesneler (n=143, val GT'nin %5.9'i): AP50-95 0.223 vs büyük 0.700 (fark 47.7 puan); <e-posta> 0.757 vs 0.978. Not: ölçüm orijinal piksel; eğitim/çıkarım 640 px letterbox olduğundan (~1.3× büyütme) S nesneler modelde ≈<41 px görünür. Val'de S örneği az olduğundan S-AP güven aralığı geniştir.

## 3. Hata türü ayrıştırması (TIDE benzeri)

Sayılar: tüm tahminler (conf≥0.001) ve işletim noktası (skor≥0.25). `Miss` tüm conf'ta hiçbir kutuyla IoU≥0.5 eşleşmeyen GT; işletim noktası sütununda ise o eşikte bulunamayan GT (eşleşmeyen + düşük skorla eşleşen).

| tür | sayı (tüm conf) | oran | sayı skor≥0.25 |
|---|---|---|---|
| TP | 2322 | 5.4% | 1899 |
| Dupe | 2351 | 5.5% | 101 |
| Cls | 3333 | 7.7% | 90 |
| Loc | 7578 | 17.6% | 147 |
| Both | 5490 | 12.7% | 68 |
| Bkg | 22035 | 51.1% | 259 |
| Miss | 85 | 3.5% (GT'nin) | 508 |

Oracle ΔAP (puan): ilgili hata türü düzeltilirse mAP ne kadar artardı (AP50 ve AP50-95, sınıf ortalaması). Taban: AP50=80.91, AP50-95=62.58.

| düzeltilen | ΔAP50 (puan) | ΔAP50-95 (puan) |
|---|---|---|
| Cls | 2.68 | 2.03 |
| Loc | 2.87 | 5.43 |
| Both | 1.07 | 0.73 |
| Dupe | 0.67 | -0.87 |
| Bkg | 3.64 | 2.33 |
| Miss | 3.44 | 2.57 |
| FP_hepsi | 14.44 | 6.18 |
| Kutu_kusursuz | 0.00 | 18.33 |
| TP_IoU_sirali | 1.57 | 5.74 |

Protokol etkisi: val'de etiketlerden çıkarılmış difficult nesne sayısı 493 (XML). Bunlardan biriyle aynı sınıf IoU≥0.5 örtüşen ve eşleşmemiş (FP sayılan) tahmin: 870 (skor≥0.25: 123; işletim noktası FP'lerinin 18.5%'i). Bu tahminler devkit gibi yok sayılsaydı ΔAP50 = +1.26, ΔAP50-95 = +0.88 puan (Ultralytics protokolünün val üzerinde ölçülmüş cezası; makalede belirtilmeli, Bkg/Loc/Dupe sayılarının bir kısmı bu yüzden yapaydır).


Not: `Dupe` ve `Loc` ΔAP50-95 değerleri IoU=0.5 tanımına göredir; bu kutular daha yüksek IoU eşiklerinde TP olabildiği için silme oracle'ı (Dupe) AP50-95'i düşürebilir; Loc oracle'ı kutuyu GT'ye yapıştırdığı için AP50-95 kazancı AP50'den büyüktür.


Hata türü × boyut (işletim noktası; tam uzun biçim `hata-kirilim.csv`):

| tür | S | M | L |
|---|---|---|---|
| TP (skor≥0.25, tahmin kutusu alanı) | 59 | 485 | 1355 |
| Dupe (skor≥0.25, tahmin kutusu alanı) | 1 | 30 | 70 |
| Cls (skor≥0.25, tahmin kutusu alanı) | 3 | 16 | 71 |
| Loc (skor≥0.25, tahmin kutusu alanı) | 6 | 58 | 83 |
| Both (skor≥0.25, tahmin kutusu alanı) | 1 | 14 | 53 |
| Bkg (skor≥0.25, tahmin kutusu alanı) | 52 | 112 | 95 |
| Miss (skor≥0.25 bulunamayan GT, GT alanı) | 84 | 212 | 212 |
| GT sayısı | 143 | 704 | 1560 |

AP50'ye en çok mal olan hata türü: Bkg (+3.64), sonra Miss (+3.44) ve Loc (+2.87). Oracle'lar birbirinden bağımsız ölçüldü, toplamları ΔAP'ye eşit değildir (etkileşim). `Kutu_kusursuz` = tüm TP kutuları GT'ye yapıştırılırsa AP50-95'in erişeceği üst sınır; `TP_IoU_sirali` = yalnız TP'lerin skorları IoU sırasına göre dağıtılırsa (FP skorları aynı) kazanç, yani IoU-farkında skorlamanın (VFL/QFL/IoU-aware) teorik payı.


Sınıf başına ΔAP50 (puan, oracle):

| sınıf | Cls | Loc | Both | Dupe | Bkg | Miss |
|---|---|---|---|---|---|---|
| aeroplane | 0.6 | 2.7 | 0.2 | 0.6 | 1.3 | 2.3 |
| bicycle | 0.4 | 2.7 | 0.3 | 0.1 | 0.5 | 3.0 |
| bird | 2.3 | 6.6 | 0.2 | 0.9 | 2.1 | 5.5 |
| boat | 1.7 | 4.5 | 0.3 | 0.6 | 4.4 | 5.2 |
| bottle | 0.3 | 2.0 | 0.3 | 0.4 | 10.3 | 2.4 |
| bus | 2.3 | 0.2 | 0.3 | 0.4 | 2.8 | 6.8 |
| car | 1.1 | 2.1 | 0.5 | 0.7 | 3.6 | 3.1 |
| cat | 2.1 | 2.1 | 0.1 | 0.2 | 0.2 | 1.2 |
| chair | 3.5 | 4.6 | 2.3 | 1.0 | 5.9 | 2.9 |
| cow | 7.2 | 2.1 | 0.1 | 0.7 | 4.7 | 1.4 |
| diningtable | 0.5 | 2.4 | 9.7 | 1.0 | 3.2 | 4.7 |
| dog | 1.5 | 0.7 | 0.3 | 0.2 | 0.5 | 0.0 |
| horse | 12.7 | 4.4 | 0.2 | 1.2 | 0.7 | 9.2 |
| motorbike | 2.9 | 2.7 | 0.6 | 0.4 | 2.4 | 6.4 |
| person | 0.4 | 3.2 | 0.2 | 1.0 | 2.2 | 1.9 |
| pottedplant | 0.4 | 8.8 | 2.3 | 1.0 | 8.9 | 6.2 |
| sheep | 5.7 | 0.1 | 0.6 | 0.3 | 4.1 | 0.0 |
| sofa | 7.2 | 1.0 | 2.0 | 2.4 | 4.0 | 1.3 |
| train | 0.2 | 3.2 | 0.1 | 0.1 | 1.4 | 3.8 |
| tvmonitor | 0.6 | 1.3 | 0.7 | 0.2 | 9.6 | 1.3 |

Kaçırılan GT (85) alt kırılımı: yalnız başka sınıfla örtüşen kutu var (sınıf karışıklığı) 17; aynı sınıf kutu var ama IoU∈[0.1,0.5) (lokalizasyon) 34; aynı sınıf IoU≥0.5 kutu var ama başka GT'ye atanmış (atama çakışması/örtüşme) 4; hiç kutu yok (gerçek kaçırma) 30 (kalan sınıflama kesin değil: bu üç grup kesişebilir).

## 4. Lokalizasyon kalitesi

| IoU aralığı | TP (tüm conf) | oran | TP skor≥0.25 | oran |
|---|---|---|---|---|
| 0.50–0.55 | 89 | 3.8% | 42 | 2.2% |
| 0.55–0.60 | 84 | 3.6% | 39 | 2.1% |
| 0.60–0.65 | 97 | 4.2% | 47 | 2.5% |
| 0.65–0.70 | 117 | 5.0% | 66 | 3.5% |
| 0.70–0.75 | 156 | 6.7% | 104 | 5.5% |
| 0.75–0.80 | 147 | 6.3% | 116 | 6.1% |
| 0.80–0.85 | 217 | 9.3% | 164 | 8.6% |
| 0.85–0.90 | 345 | 14.9% | 300 | 15.8% |
| 0.90–0.95 | 513 | 22.1% | 486 | 25.6% |
| 0.95–1.00 | 557 | 24.0% | 535 | 28.2% |

TP IoU ortalaması 0.844 (skor≥0.25: 0.870); işletim noktası TP'lerinin %15.7'i IoU<0.75. TP'lerde skor–IoU Spearman ρ=0.62 (skorun konum kalitesini ne kadar taşıdığı).

Sınıf başına AP50 − AP75 (büyükten küçüğe) ve TP ort. IoU:

| sınıf | AP50 | AP75 | AP50−AP75 (puan) | TP ort. IoU |
|---|---|---|---|---|
| pottedplant | 0.561 | 0.247 | 31.4 | 0.743 |
| person | 0.885 | 0.674 | 21.1 | 0.832 |
| aeroplane | 0.893 | 0.716 | 17.7 | 0.848 |
| tvmonitor | 0.808 | 0.633 | 17.5 | 0.847 |
| motorbike | 0.818 | 0.653 | 16.6 | 0.851 |
| boat | 0.778 | 0.619 | 15.9 | 0.826 |
| bird | 0.798 | 0.645 | 15.3 | 0.844 |
| chair | 0.628 | 0.479 | 14.9 | 0.811 |
| horse | 0.771 | 0.634 | 13.8 | 0.858 |
| bicycle | 0.896 | 0.767 | 12.9 | 0.868 |
| car | 0.841 | 0.721 | 12.0 | 0.853 |
| sheep | 0.875 | 0.758 | 11.7 | 0.887 |
| diningtable | 0.705 | 0.596 | 10.9 | 0.867 |
| sofa | 0.742 | 0.633 | 10.9 | 0.872 |
| bus | 0.839 | 0.735 | 10.4 | 0.888 |
| train | 0.921 | 0.823 | 9.8 | 0.893 |
| cow | 0.828 | 0.733 | 9.5 | 0.867 |
| bottle | 0.715 | 0.624 | 9.1 | 0.832 |
| dog | 0.941 | 0.852 | 8.9 | 0.890 |
| cat | 0.938 | 0.872 | 6.5 | 0.915 |

AP50→AP75 düşüşü sınıf ortalaması 13.8 puan; en büyük düşüş pottedplant (31), person (21), aeroplane (18). Kutu-regresyon (IoU/DFL/box kazancı) manipülasyonlarının hedef alanı budur.

## 5. Sınıf karışıklığı

İşletim noktası (skor≥0.25, IoU≥0.5): her GT için eşleşen en yüksek skorlu kutu (doğru sınıf varsa o). Matris dosyası: `hata-analizi_20261003-062922_h01_multiscale25_s1_karisiklik.csv`.

| GT sınıfı → tahmin | sayı | GT'nin payı |
|---|---|---|
| chair → person | 7 | 4.0% |
| horse → sheep | 6 | 9.1% |
| chair → sofa | 4 | 2.3% |
| cow → horse | 4 | 7.1% |
| sofa → chair | 4 | 6.8% |
| bus → car | 2 | 3.6% |
| car → boat | 2 | 0.9% |
| chair → diningtable | 2 | 1.1% |
| cow → sheep | 2 | 3.6% |
| motorbike → bicycle | 2 | 3.0% |
| motorbike → person | 2 | 3.0% |
| sofa → person | 2 | 3.4% |

Çift bazında (yönsüz) toplam: chair/sofa 8; chair/person 7; horse/sheep 6; cow/horse 4; bus/car 3; cow/sheep 3; boat/car 2; chair/diningtable 2.


Tüm-conf alternatif görünüm: `Cls` türü tahminler (IoU≥0.5 başka sınıf GT, skor≥0.05), GT sınıfı → tahmin:

| GT → tahmin | tahmin sayısı |
|---|---|
| sofa → chair | 30 |
| person → chair | 23 |
| person → sofa | 20 |
| dog → cat | 15 |
| chair → sofa | 13 |
| cow → sheep | 12 |
| cow → horse | 11 |
| horse → sheep | 10 |
| motorbike → bicycle | 10 |
| sheep → cow | 10 |

Kanonik çiftler (yönsüz; `Cls` tahmin sayısı skor≥0.05 | işletim noktası GT sayısı): cat/dog 22 | 2; cow/sheep 22 | 3; chair/diningtable 7 | 2; chair/sofa 43 | 8; horse/cow 13 | 4; bus/train 3 | 0; motorbike/bicycle 13 | 2.

İşletim noktasında GT'lerin 2.4%'i başka sınıf olarak etiketlendi (58/2407); 18.1%'i hiç bulunamadı. Sınıf karışıklığı oracle ΔAP50 = 2.68 puan (§3; Cls, altı hata türü içinde 4. sırada); işletim noktasında karışan GT oranı yalnız 2.4%, yani sorun sınıf ayrımından çok kutu/skor düzeyindedir.

## 6. Kalabalık ve örtüşme

| nesne/görüntü | görüntü | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) | FP/görüntü (skor≥0.25) |
|---|---|---|---|---|---|---|
| 1 | 440 | 440 | 99.1% | 92.7% | 92.3% | 0.35 |
| 2 | 252 | 504 | 97.2% | 82.7% | 86.7% | 0.48 |
| 3–4 | 195 | 654 | 96.0% | 76.6% | 81.8% | 1.07 |
| 5–8 | 88 | 511 | 94.5% | 69.7% | 74.4% | 1.25 |
| ≥9 | 25 | 298 | 95.6% | 72.8% | 68.5% | 2.92 |

Aynı sınıftan başka GT ile en yüksek IoU (örtüşme) gruplarına göre recall:

| komşu GT IoU | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 |
|---|---|---|---|
| <0.1 (komşu yok) | 1908 | 96.6% | 79.7% |
| 0.1–0.3 | 357 | 96.1% | 77.6% |
| 0.3–0.5 | 105 | 97.1% | 75.2% |
| 0.5–0.7 | 33 | 90.9% | 60.6% |
| ≥0.7 | 4 | 100.0% | 75.0% |

Aynı sınıf GT çiftleri (IoU≥0.3): 75 çift; ikisi de bulundu 69, yalnız biri 5, hiçbiri 1. IoU≥0.7 (NMS eşiği üstü, doğrudan NMS etkilenir): 2 çift, ikisi de bulundu 2, yalnız biri 0, hiçbiri 0. Kaçırılan GT'lerin 4/85'i aynı sınıf IoU≥0.5 bir kutuya sahip ama o kutu başka GT'ye atanmış (olası NMS/atama kaybı).

Kalabalıkla recall düşüşü: tek nesneli görüntüde 92.7% iken ≥9 nesnelide 72.8% (skor≥0.25); tüm-conf recall 99.1% → 95.6%, <e-posta> 92.3% → 68.5%. Çiftlerde tek-bulunma ve atama çakışması sayıları küçükse kayıp NMS'ten değil düşük skor/zayıf konumdan gelir.

## 7. Kalibrasyon (skor vs gerçek precision)

Tahmin 'doğru' = IoU≥0.5 ile eşleşen (<e-posta>); ayrıca IoU≥0.75 için precision. Not: YOLO sınıf skoru olasılık değil, TAL hizalama hedefli (IoU'ya bağlı) bir skordur; düşük skor bölgesindeki düşük precision 'aday havuzu' doğasından gelir.

| skor aralığı | tahmin | ort. skor | <e-posta> | <e-posta> | TP ort. IoU | skor−precision |
|---|---|---|---|---|---|---|
| 0.0–0.1 | 39539 | 0.008 | 0.007 | 0.006 | 0.723 | 0.001 |
| 0.1–0.2 | 804 | 0.144 | 0.141 | 0.083 | 0.727 | 0.004 |
| 0.2–0.3 | 350 | 0.244 | 0.191 | 0.129 | 0.753 | 0.053 |
| 0.3–0.4 | 248 | 0.349 | 0.246 | 0.169 | 0.756 | 0.103 |
| 0.4–0.5 | 206 | 0.448 | 0.461 | 0.282 | 0.776 | -0.013 |
| 0.5–0.6 | 209 | 0.549 | 0.574 | 0.407 | 0.802 | -0.025 |
| 0.6–0.7 | 250 | 0.653 | 0.692 | 0.536 | 0.824 | -0.039 |
| 0.7–0.8 | 331 | 0.755 | 0.843 | 0.659 | 0.837 | -0.088 |
| 0.8–0.9 | 637 | 0.859 | 0.943 | 0.852 | 0.883 | -0.084 |
| 0.9–1.0 | 535 | 0.925 | 0.994 | 0.974 | 0.943 | -0.069 |

ECE (tüm tahminler, 10 eşit genişlikli kutu) = 0.005; ECE (skor≥0.1) = 0.052. `skor−precision` pozitifse aşırı güvenli, negatifse eksik güvenli. 
Skor≥0.5 tahminlerin <e-posta> = 86.9%, <e-posta> = 76.5% (fark 10.4 puan). Skor 0.5 üstü kutularda (skor−precision) ortalaması -0.061: model yüksek skorlarda eksik güvenli (skor < gerçek <e-posta>), sıralama iyi; asıl boşluk <e-posta> ile @0.75 arasındaki konum kalitesidir. TP'lerde skor–IoU Spearman ρ=0.62: skor konum kalitesini kısmen taşıyor. TP_IoU_sirali oracle'ı (§3) IoU-farkında skorlamanın teorik payını sayısallaştırıyor.

## 8. Eğitim dinamiği (results.csv)

| gösterge | değer |
|---|---|
| epoch sayısı | 100 |
| en iyi epoch (mAP50-95) | 100 (0.6245) |
| son epoch mAP50-95 | 0.6245 |
| epoch 50 / 80 / 90 / 100 | 0.6095 / 0.6213 / 0.6226 / 0.6245 |
| son 20 epoch eğimi (puan/10 epoch) | 0.150 |
| önceki 20 epoch (ep 71-90) eğimi | 0.173 |
| son 20 epoch std (puan) | 0.090 |
| ep 81-90 ort. vs ep 91-100 ort. (close_mosaic=10) | 0.6222 vs 0.6237 (Δ +0.15 puan) |
| mAP50 son epoch | 0.8105 |
| results.csv en iyi ep mAP50-95 vs metrics.json (best.pt yeniden val) | 0.6245 vs 0.6263 (fark -0.17 puan; ölçüm-protokolü gürültü tabanı) |

Loss tablosu (train loss mosaic açıkken ep1/ep90, kapalıyken ep100; val/train oranı):

| loss | train ep1 | train ep90 | train ep100 | val ep100 | val/train ep100 | val/train ep90 | train loss ep90→91 değişim |
|---|---|---|---|---|---|---|---|
| box | 1.022 | 0.818 | 0.712 | 0.870 | 1.22 | 1.06 | -6.5% |
| cls | 2.374 | 0.839 | 0.571 | 0.802 | 1.40 | 0.96 | -22.0% |
| dfl | 1.300 | 1.154 | 1.077 | 1.157 | 1.07 | 1.00 | -3.2% |

Kayıp payı (results.csv loss_items kazançlı; loss.py:469-471 — önceki sürüm kazançları iki kez uyguluyordu, 2026-10-01 düzeltildi) ep100: box 30.2%, cls 24.2%, dfl 45.6% (ep1: box 21.8%, cls 50.6%, dfl 27.7%). Son 20 epoch mAP eğimi +0.150 puan/10 ep (std 0.09) → doygun (eğim seed gürültüsü altında). close_mosaic sonrası (ep 91-100) ortalama +0.15 puan. Train loss mosaic kapanınca ani düşer (cls -22.0%); bu rejim değişimi, train/val farkının bir kısmının augmentation kaynaklı olduğunu gösterir. val cls loss son 20 epoch 0.805→0.802 (yükselmiyor): val loss bozulmadığı için klasik aşırı öğrenme yok, ama mosaic kapalıyken val/train oranı cls 1.40, box 1.22: train kaybı val kaybından ayrışıyor (genelleme açığı); val mAP doygun.

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
| ön-işleme ms | 1.3136 |
| çıkarım ms | 6.028 |
| NMS/son-işlem ms | 0.6826 |
| toplam ms | 8.0242 |
| FPS | 124.623 |
| batch | 1 |
| FP16 (1=evet) | 1.0 |
| ölçüm görüntüsü | 300 |
| ısınma görüntüsü | 50 |
| eğitim süresi (saat) | 3.2622 |

Varyant hız bütçesi (yalnız tahmin; gerçek ölçüm varyant koşumunda yapılır):

| varyant sınıfı | çarpan | tahmini toplam ms | tahmini FPS | not |
|---|---|---|---|---|
| eğitim-zamanı (aug/loss) varyantları | aynı | 8.02 | 125 | çıkarım grafiği değişmez; hız bütçesi ≈ 0 |
| Soft-NMS/DIoU-NMS (son-işlem ×1–5 varsayımı) | NMS ×1–5 | 8.02–10.75 | 93–125 | tahmin (ölçülmedi) |
| TTA (Ultralytics augment, ~3 ölçek+flip; literatür çarpanı 3.6× çıkarım) | çıkarım ×3.6 (literatür) | 23.70 | 42 | tahmin; literatur-cipa.md §4 (Ultralytics TTA dokümanı) |
| YOLO11s'e geçiş (bağlam satırı) | ≈3.4× GFLOPs | – | – | ölçülmedi |

Baseline batch=1 FP16 toplam 8.02 ms (125 FPS; çıkarım 6.03 ms, NMS 0.68 ms = toplamın %9'i). 'Gerçek zamanlı' için ≥30 FPS kabul edilirse (33 ms) ~4.1× gecikme payı vardır; TTA bile (~42 FPS tahmini) eşiğin üstünde kalabilir, ancak hız-doğruluk satırı ayrı raporlanmalıdır. Not: hız ölçümü GPU'nun paylaşıldığı bir anda alınmış olabilir (eğitim kuyruğu); karşılaştırmalı hız ölçümü aynı koşulda yapılmalıdır.
