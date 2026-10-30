# Hata analizi (K3.5) — baseline `20261003-094536_h01_multiscale25_s2`

*Üretici: `scripts/analiz/hata_analizi.py` · yalnız VAL (1000 görüntü, 2407 nesne; difficult nesneler etiketlerde yok) · tahmin sayısı 45123 (conf=0.001, max_det=300, NMS IoU=0.7) · eşleştirme: pycocotools COCOeval · işletim noktası skor≥0.25.*

**Sızıntı notu:** Bu analizde test kümesi (VOC07 test), `rep_test*` dosyaları ve `metrics.json` içindeki `rep_*` anahtarları hiç kullanılmadı; betik bu anahtarları yükler yüklemez atar. Buradan çıkan hipotezler yalnız val bilgisine dayanır.

## 0. Genel ölçüler ve doğrulama

| ölçü | pycocotools (bu betik) | Ultralytics metrics.json (best.pt val) | fark |
|---|---|---|---|
| mAP50-95 | 0.6295 | 0.6299 | -0.0004 |
| mAP50 | 0.8131 | 0.8156 | -0.0025 |
| mAP75 (yalnız pycocotools) | 0.6805 | – | – |

AP50 ile AP50-95 arasındaki fark 18.4 puan; AP50→AP75 farkı 13.3 puan. pycocotools ile Ultralytics AP'si arasındaki küçük fark interpolasyon/eşleştirme (Ultralytics kendi IoU-matrisi eşleşmesini kullanır) kaynaklıdır.

## 1. Sınıf başına AP ve sınıf frekansı

| sınıf (AP50-95 artan) | AP50-95 | AP50 | AP75 | AP50−AP75 | train nesne | val nesne | val GT medyan kenar px |
|---|---|---|---|---|---|---|---|
| pottedplant | 0.314 | 0.565 | 0.300 | 0.265 | 1378 | 109 | 71 |
| chair | 0.462 | 0.636 | 0.510 | 0.125 | 2975 | 177 | 99 |
| bottle | 0.494 | 0.660 | 0.560 | 0.101 | 1669 | 95 | 70 |
| boat | 0.562 | 0.783 | 0.630 | 0.153 | 1076 | 64 | 112 |
| sofa | 0.586 | 0.724 | 0.603 | 0.121 | 755 | 59 | 237 |
| diningtable | 0.598 | 0.735 | 0.651 | 0.084 | 775 | 49 | 211 |
| tvmonitor | 0.604 | 0.827 | 0.699 | 0.128 | 1036 | 72 | 108 |
| bird | 0.608 | 0.806 | 0.661 | 0.145 | 1514 | 91 | 115 |
| horse | 0.631 | 0.834 | 0.679 | 0.155 | 1006 | 66 | 209 |
| person | 0.636 | 0.892 | 0.680 | 0.212 | 12471 | 785 | 141 |
| motorbike | 0.645 | 0.827 | 0.671 | 0.156 | 985 | 67 | 239 |
| car | 0.647 | 0.831 | 0.720 | 0.111 | 3045 | 222 | 92 |
| aeroplane | 0.690 | 0.883 | 0.706 | 0.178 | 1091 | 80 | 184 |
| bicycle | 0.695 | 0.892 | 0.748 | 0.144 | 1002 | 62 | 171 |
| cow | 0.719 | 0.893 | 0.819 | 0.074 | 791 | 56 | 145 |
| sheep | 0.721 | 0.877 | 0.766 | 0.111 | 1013 | 57 | 119 |
| train | 0.728 | 0.893 | 0.781 | 0.112 | 868 | 57 | 225 |
| bus | 0.736 | 0.847 | 0.793 | 0.055 | 767 | 55 | 192 |
| cat | 0.749 | 0.922 | 0.789 | 0.132 | 1505 | 88 | 278 |
| dog | 0.766 | 0.936 | 0.844 | 0.092 | 1929 | 96 | 211 |

En zayıf 5 sınıf (AP50-95): pottedplant 0.314, chair 0.462, bottle 0.494, boat 0.562, sofa 0.586. Sınıf frekansı (train nesne) ile AP50-95 Spearman ρ=-0.15 (permütasyon p=0.527), AP50 ile ρ=-0.01; sınıf başına train küçük-nesne oranı ile ρ=-0.39; val GT medyan kenar uzunluğu ile ρ=0.54 (n=20 sınıf, gürültülü). Frekans AP'nin zayıf belirleyicisi (|ρ|<0.3): zayıf sınıflarda sorun veri azlığından çok nesne boyutu/görünümü/bağlamıdır.

## 2. Nesne boyutu kırılımı (orijinal çözünürlük; COCO S<32²≤M<96²≤L)

| boyut | GT sayısı | AP50-95 | AP50 | <e-posta> (tüm conf, sınıf ort.) | <e-posta>:0.95 (sınıf ort.) | <e-posta> skor≥0.25 |
|---|---|---|---|---|---|---|
| all | 2407 | 0.630 | 0.813 | 0.960 | 0.767 | 0.795 |
| S | 143 | 0.224 | 0.386 | 0.765 | 0.481 | 0.392 |
| M | 704 | 0.450 | 0.657 | 0.898 | 0.646 | 0.717 |
| L | 1560 | 0.706 | 0.873 | 0.979 | 0.816 | 0.867 |

İnce alan aralıkları (GT düzeyi, eşleşme bayrağı):

| alan (px²) | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) |
|---|---|---|---|---|
| <16² | 9 | 44.4% | 0.0% | 22.2% |
| 16²–32² | 134 | 88.1% | 41.8% | 60.4% |
| 32²–64² | 377 | 95.2% | 69.0% | 74.8% |
| 64²–96² | 327 | 94.8% | 74.9% | 80.4% |
| 96²–192² | 698 | 97.1% | 82.1% | 83.2% |
| ≥192² | 862 | 99.2% | 90.5% | 91.0% |

Sınıf × boyut AP50-95 (n<5 ise '–'):

| sınıf | n_S | AP_S | n_M | AP_M | n_L | AP_L |
|---|---|---|---|---|---|---|
| aeroplane | 0 | – | 21 | 0.51 | 59 | 0.74 |
| bicycle | 0 | – | 12 | 0.42 | 50 | 0.77 |
| bird | 3 | – | 36 | 0.35 | 52 | 0.79 |
| boat | 4 | – | 25 | 0.53 | 35 | 0.64 |
| bottle | 13 | 0.22 | 52 | 0.45 | 30 | 0.67 |
| bus | 3 | – | 13 | 0.48 | 39 | 0.87 |
| car | 38 | 0.32 | 80 | 0.65 | 104 | 0.75 |
| cat | 0 | – | 2 | – | 86 | 0.77 |
| chair | 3 | – | 79 | 0.44 | 95 | 0.49 |
| cow | 0 | – | 25 | 0.70 | 31 | 0.78 |
| diningtable | 0 | – | 1 | – | 48 | 0.61 |
| dog | 0 | – | 10 | 0.70 | 86 | 0.78 |
| horse | 3 | – | 15 | 0.21 | 48 | 0.78 |
| motorbike | 3 | – | 10 | 0.60 | 54 | 0.69 |
| person | 51 | 0.20 | 215 | 0.50 | 519 | 0.72 |
| pottedplant | 12 | 0.21 | 57 | 0.32 | 40 | 0.38 |
| sheep | 8 | 0.35 | 16 | 0.65 | 33 | 0.84 |
| sofa | 0 | – | 1 | – | 58 | 0.59 |
| train | 0 | – | 4 | – | 53 | 0.74 |
| tvmonitor | 2 | – | 30 | 0.51 | 40 | 0.70 |

Küçük nesneler (n=143, val GT'nin %5.9'i): AP50-95 0.224 vs büyük 0.706 (fark 48.2 puan); <e-posta> 0.765 vs 0.979. Not: ölçüm orijinal piksel; eğitim/çıkarım 640 px letterbox olduğundan (~1.3× büyütme) S nesneler modelde ≈<41 px görünür. Val'de S örneği az olduğundan S-AP güven aralığı geniştir.

## 3. Hata türü ayrıştırması (TIDE benzeri)

Sayılar: tüm tahminler (conf≥0.001) ve işletim noktası (skor≥0.25). `Miss` tüm conf'ta hiçbir kutuyla IoU≥0.5 eşleşmeyen GT; işletim noktası sütununda ise o eşikte bulunamayan GT (eşleşmeyen + düşük skorla eşleşen).

| tür | sayı (tüm conf) | oran | sayı skor≥0.25 |
|---|---|---|---|
| TP | 2324 | 5.2% | 1914 |
| Dupe | 2551 | 5.7% | 101 |
| Cls | 3637 | 8.1% | 110 |
| Loc | 8052 | 17.8% | 153 |
| Both | 5559 | 12.3% | 59 |
| Bkg | 23000 | 51.0% | 272 |
| Miss | 83 | 3.4% (GT'nin) | 493 |

Oracle ΔAP (puan): ilgili hata türü düzeltilirse mAP ne kadar artardı (AP50 ve AP50-95, sınıf ortalaması). Taban: AP50=81.31, AP50-95=62.95.

| düzeltilen | ΔAP50 (puan) | ΔAP50-95 (puan) |
|---|---|---|
| Cls | 2.77 | 2.02 |
| Loc | 2.46 | 5.33 |
| Both | 1.00 | 0.67 |
| Dupe | 0.53 | -0.73 |
| Bkg | 3.61 | 2.32 |
| Miss | 3.45 | 2.56 |
| FP_hepsi | 14.18 | 6.44 |
| Kutu_kusursuz | 0.00 | 18.36 |
| TP_IoU_sirali | 1.32 | 5.54 |

Protokol etkisi: val'de etiketlerden çıkarılmış difficult nesne sayısı 493 (XML). Bunlardan biriyle aynı sınıf IoU≥0.5 örtüşen ve eşleşmemiş (FP sayılan) tahmin: 931 (skor≥0.25: 120; işletim noktası FP'lerinin 17.3%'i). Bu tahminler devkit gibi yok sayılsaydı ΔAP50 = +1.21, ΔAP50-95 = +0.80 puan (Ultralytics protokolünün val üzerinde ölçülmüş cezası; makalede belirtilmeli, Bkg/Loc/Dupe sayılarının bir kısmı bu yüzden yapaydır).


Not: `Dupe` ve `Loc` ΔAP50-95 değerleri IoU=0.5 tanımına göredir; bu kutular daha yüksek IoU eşiklerinde TP olabildiği için silme oracle'ı (Dupe) AP50-95'i düşürebilir; Loc oracle'ı kutuyu GT'ye yapıştırdığı için AP50-95 kazancı AP50'den büyüktür.


Hata türü × boyut (işletim noktası; tam uzun biçim `hata-kirilim.csv`):

| tür | S | M | L |
|---|---|---|---|
| TP (skor≥0.25, tahmin kutusu alanı) | 54 | 486 | 1374 |
| Dupe (skor≥0.25, tahmin kutusu alanı) | 2 | 28 | 71 |
| Cls (skor≥0.25, tahmin kutusu alanı) | 4 | 16 | 90 |
| Loc (skor≥0.25, tahmin kutusu alanı) | 8 | 64 | 81 |
| Both (skor≥0.25, tahmin kutusu alanı) | 1 | 17 | 41 |
| Bkg (skor≥0.25, tahmin kutusu alanı) | 61 | 117 | 94 |
| Miss (skor≥0.25 bulunamayan GT, GT alanı) | 87 | 199 | 207 |
| GT sayısı | 143 | 704 | 1560 |

AP50'ye en çok mal olan hata türü: Bkg (+3.61), sonra Miss (+3.45) ve Cls (+2.77). Oracle'lar birbirinden bağımsız ölçüldü, toplamları ΔAP'ye eşit değildir (etkileşim). `Kutu_kusursuz` = tüm TP kutuları GT'ye yapıştırılırsa AP50-95'in erişeceği üst sınır; `TP_IoU_sirali` = yalnız TP'lerin skorları IoU sırasına göre dağıtılırsa (FP skorları aynı) kazanç, yani IoU-farkında skorlamanın (VFL/QFL/IoU-aware) teorik payı.


Sınıf başına ΔAP50 (puan, oracle):

| sınıf | Cls | Loc | Both | Dupe | Bkg | Miss |
|---|---|---|---|---|---|---|
| aeroplane | 0.3 | 3.0 | 0.3 | 0.4 | 1.4 | 3.5 |
| bicycle | 2.4 | 0.8 | 0.6 | 0.4 | 0.7 | 3.3 |
| bird | 1.9 | 3.4 | 0.4 | 0.8 | 3.0 | 4.8 |
| boat | 0.8 | 4.3 | 0.2 | 0.6 | 4.2 | 6.7 |
| bottle | 1.0 | 3.1 | 0.2 | 0.2 | 10.7 | 2.9 |
| bus | 3.3 | 0.8 | 0.1 | 0.2 | 1.7 | 7.0 |
| car | 1.6 | 2.1 | 0.4 | 0.6 | 4.2 | 3.2 |
| cat | 2.6 | 2.2 | 0.1 | 0.1 | 0.2 | 3.5 |
| chair | 3.9 | 3.4 | 2.2 | 0.7 | 6.9 | 1.5 |
| cow | 2.0 | 1.7 | 0.0 | 0.4 | 4.1 | 1.7 |
| diningtable | 0.7 | 1.0 | 8.0 | 0.7 | 3.0 | 3.3 |
| dog | 2.0 | 0.6 | 0.1 | 0.2 | 0.5 | 1.3 |
| horse | 10.6 | 0.5 | 0.0 | 0.4 | 0.5 | 10.0 |
| motorbike | 3.2 | 2.4 | 0.6 | 0.4 | 2.3 | 2.5 |
| person | 0.2 | 2.9 | 0.1 | 0.9 | 2.0 | 2.0 |
| pottedplant | 0.5 | 9.5 | 2.4 | 1.0 | 7.4 | 6.4 |
| sheep | 6.8 | 0.0 | 0.1 | 0.1 | 4.8 | 0.0 |
| sofa | 10.6 | 0.3 | 3.4 | 2.1 | 3.7 | 1.3 |
| train | 0.5 | 5.8 | 0.2 | 0.1 | 1.5 | 3.2 |
| tvmonitor | 0.7 | 1.2 | 0.5 | 0.3 | 9.3 | 1.1 |

Kaçırılan GT (83) alt kırılımı: yalnız başka sınıfla örtüşen kutu var (sınıf karışıklığı) 21; aynı sınıf kutu var ama IoU∈[0.1,0.5) (lokalizasyon) 35; aynı sınıf IoU≥0.5 kutu var ama başka GT'ye atanmış (atama çakışması/örtüşme) 0; hiç kutu yok (gerçek kaçırma) 27 (kalan sınıflama kesin değil: bu üç grup kesişebilir).

## 4. Lokalizasyon kalitesi

| IoU aralığı | TP (tüm conf) | oran | TP skor≥0.25 | oran |
|---|---|---|---|---|
| 0.50–0.55 | 81 | 3.5% | 34 | 1.8% |
| 0.55–0.60 | 85 | 3.7% | 39 | 2.0% |
| 0.60–0.65 | 88 | 3.8% | 50 | 2.6% |
| 0.65–0.70 | 112 | 4.8% | 68 | 3.6% |
| 0.70–0.75 | 146 | 6.3% | 100 | 5.2% |
| 0.75–0.80 | 177 | 7.6% | 133 | 6.9% |
| 0.80–0.85 | 217 | 9.3% | 171 | 8.9% |
| 0.85–0.90 | 327 | 14.1% | 284 | 14.8% |
| 0.90–0.95 | 526 | 22.6% | 489 | 25.5% |
| 0.95–1.00 | 565 | 24.3% | 546 | 28.5% |

TP IoU ortalaması 0.846 (skor≥0.25: 0.871); işletim noktası TP'lerinin %15.2'i IoU<0.75. TP'lerde skor–IoU Spearman ρ=0.61 (skorun konum kalitesini ne kadar taşıdığı).

Sınıf başına AP50 − AP75 (büyükten küçüğe) ve TP ort. IoU:

| sınıf | AP50 | AP75 | AP50−AP75 (puan) | TP ort. IoU |
|---|---|---|---|---|
| pottedplant | 0.565 | 0.300 | 26.5 | 0.763 |
| person | 0.892 | 0.680 | 21.2 | 0.835 |
| aeroplane | 0.883 | 0.706 | 17.8 | 0.850 |
| motorbike | 0.827 | 0.671 | 15.6 | 0.849 |
| horse | 0.834 | 0.679 | 15.5 | 0.859 |
| boat | 0.783 | 0.630 | 15.3 | 0.830 |
| bird | 0.806 | 0.661 | 14.5 | 0.851 |
| bicycle | 0.892 | 0.748 | 14.4 | 0.862 |
| cat | 0.922 | 0.789 | 13.2 | 0.898 |
| tvmonitor | 0.827 | 0.699 | 12.8 | 0.853 |
| chair | 0.636 | 0.510 | 12.5 | 0.819 |
| sofa | 0.724 | 0.603 | 12.1 | 0.873 |
| train | 0.893 | 0.781 | 11.2 | 0.880 |
| car | 0.831 | 0.720 | 11.1 | 0.849 |
| sheep | 0.877 | 0.766 | 11.1 | 0.886 |
| bottle | 0.660 | 0.560 | 10.1 | 0.825 |
| dog | 0.936 | 0.844 | 9.2 | 0.900 |
| diningtable | 0.735 | 0.651 | 8.4 | 0.881 |
| cow | 0.893 | 0.819 | 7.4 | 0.880 |
| bus | 0.847 | 0.793 | 5.5 | 0.899 |

AP50→AP75 düşüşü sınıf ortalaması 13.3 puan; en büyük düşüş pottedplant (27), person (21), aeroplane (18). Kutu-regresyon (IoU/DFL/box kazancı) manipülasyonlarının hedef alanı budur.

## 5. Sınıf karışıklığı

İşletim noktası (skor≥0.25, IoU≥0.5): her GT için eşleşen en yüksek skorlu kutu (doğru sınıf varsa o). Matris dosyası: `hata-analizi_20261003-094536_h01_multiscale25_s2_karisiklik.csv`.

| GT sınıfı → tahmin | sayı | GT'nin payı |
|---|---|---|
| chair → person | 9 | 5.1% |
| horse → sheep | 6 | 9.1% |
| sofa → chair | 5 | 8.5% |
| sofa → person | 4 | 6.8% |
| car → train | 3 | 1.4% |
| chair → diningtable | 3 | 1.7% |
| chair → sofa | 3 | 1.7% |
| dog → cat | 3 | 3.1% |
| bus → car | 2 | 3.6% |
| cow → horse | 2 | 3.6% |
| cow → sheep | 2 | 3.6% |
| sheep → dog | 2 | 3.5% |

Çift bazında (yönsüz) toplam: chair/person 9; chair/sofa 8; horse/sheep 6; person/sofa 4; chair/diningtable 4; cat/dog 4; car/train 3; cow/sheep 3.


Tüm-conf alternatif görünüm: `Cls` türü tahminler (IoU≥0.5 başka sınıf GT, skor≥0.05), GT sınıfı → tahmin:

| GT → tahmin | tahmin sayısı |
|---|---|
| sofa → chair | 28 |
| person → sofa | 24 |
| person → chair | 22 |
| dog → cat | 17 |
| cow → horse | 13 |
| sheep → dog | 12 |
| cow → sheep | 11 |
| horse → sheep | 10 |
| chair → sofa | 9 |
| sheep → cow | 9 |

Kanonik çiftler (yönsüz; `Cls` tahmin sayısı skor≥0.05 | işletim noktası GT sayısı): cat/dog 25 | 4; cow/sheep 20 | 3; chair/diningtable 8 | 4; chair/sofa 37 | 8; horse/cow 17 | 2; bus/train 4 | 1; motorbike/bicycle 14 | 2.

İşletim noktasında GT'lerin 2.6%'i başka sınıf olarak etiketlendi (63/2407); 17.3%'i hiç bulunamadı. Sınıf karışıklığı oracle ΔAP50 = 2.77 puan (§3; Cls, altı hata türü içinde 3. sırada); işletim noktasında karışan GT oranı yalnız 2.6%, yani sorun sınıf ayrımından çok kutu/skor düzeyindedir.

## 6. Kalabalık ve örtüşme

| nesne/görüntü | görüntü | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) | FP/görüntü (skor≥0.25) |
|---|---|---|---|---|---|---|
| 1 | 440 | 440 | 97.7% | 91.8% | 90.9% | 0.36 |
| 2 | 252 | 504 | 97.8% | 83.1% | 87.5% | 0.52 |
| 3–4 | 195 | 654 | 96.0% | 78.4% | 82.6% | 0.95 |
| 5–8 | 88 | 511 | 95.3% | 71.0% | 77.7% | 1.44 |
| ≥9 | 25 | 298 | 96.0% | 72.1% | 72.1% | 3.68 |

Aynı sınıftan başka GT ile en yüksek IoU (örtüşme) gruplarına göre recall:

| komşu GT IoU | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 |
|---|---|---|---|
| <0.1 (komşu yok) | 1908 | 96.6% | 80.6% |
| 0.1–0.3 | 357 | 96.4% | 76.8% |
| 0.3–0.5 | 105 | 97.1% | 72.4% |
| 0.5–0.7 | 33 | 93.9% | 69.7% |
| ≥0.7 | 4 | 100.0% | 75.0% |

Aynı sınıf GT çiftleri (IoU≥0.3): 75 çift; ikisi de bulundu 71, yalnız biri 2, hiçbiri 2. IoU≥0.7 (NMS eşiği üstü, doğrudan NMS etkilenir): 2 çift, ikisi de bulundu 2, yalnız biri 0, hiçbiri 0. Kaçırılan GT'lerin 0/83'i aynı sınıf IoU≥0.5 bir kutuya sahip ama o kutu başka GT'ye atanmış (olası NMS/atama kaybı).

Kalabalıkla recall düşüşü: tek nesneli görüntüde 91.8% iken ≥9 nesnelide 72.1% (skor≥0.25); tüm-conf recall 97.7% → 96.0%, <e-posta> 90.9% → 72.1%. Çiftlerde tek-bulunma ve atama çakışması sayıları küçükse kayıp NMS'ten değil düşük skor/zayıf konumdan gelir.

## 7. Kalibrasyon (skor vs gerçek precision)

Tahmin 'doğru' = IoU≥0.5 ile eşleşen (<e-posta>); ayrıca IoU≥0.75 için precision. Not: YOLO sınıf skoru olasılık değil, TAL hizalama hedefli (IoU'ya bağlı) bir skordur; düşük skor bölgesindeki düşük precision 'aday havuzu' doğasından gelir.

| skor aralığı | tahmin | ort. skor | <e-posta> | <e-posta> | TP ort. IoU | skor−precision |
|---|---|---|---|---|---|---|
| 0.0–0.1 | 41450 | 0.008 | 0.007 | 0.006 | 0.723 | 0.001 |
| 0.1–0.2 | 860 | 0.139 | 0.114 | 0.083 | 0.761 | 0.025 |
| 0.2–0.3 | 408 | 0.250 | 0.228 | 0.140 | 0.752 | 0.022 |
| 0.3–0.4 | 275 | 0.349 | 0.287 | 0.225 | 0.794 | 0.062 |
| 0.4–0.5 | 202 | 0.450 | 0.421 | 0.292 | 0.787 | 0.029 |
| 0.5–0.6 | 191 | 0.550 | 0.581 | 0.387 | 0.780 | -0.031 |
| 0.6–0.7 | 237 | 0.650 | 0.722 | 0.523 | 0.817 | -0.072 |
| 0.7–0.8 | 329 | 0.752 | 0.845 | 0.687 | 0.846 | -0.093 |
| 0.8–0.9 | 624 | 0.858 | 0.955 | 0.859 | 0.882 | -0.098 |
| 0.9–1.0 | 547 | 0.926 | 0.991 | 0.967 | 0.941 | -0.065 |

ECE (tüm tahminler, 10 eşit genişlikli kutu) = 0.006; ECE (skor≥0.1) = 0.055. `skor−precision` pozitifse aşırı güvenli, negatifse eksik güvenli. 
Skor≥0.5 tahminlerin <e-posta> = 88.1%, <e-posta> = 77.2% (fark 10.8 puan). Skor 0.5 üstü kutularda (skor−precision) ortalaması -0.072: model yüksek skorlarda eksik güvenli (skor < gerçek <e-posta>), sıralama iyi; asıl boşluk <e-posta> ile @0.75 arasındaki konum kalitesidir. TP'lerde skor–IoU Spearman ρ=0.61: skor konum kalitesini kısmen taşıyor. TP_IoU_sirali oracle'ı (§3) IoU-farkında skorlamanın teorik payını sayısallaştırıyor.

## 8. Eğitim dinamiği (results.csv)

| gösterge | değer |
|---|---|
| epoch sayısı | 100 |
| en iyi epoch (mAP50-95) | 91 (0.6323) |
| son epoch mAP50-95 | 0.6309 |
| epoch 50 / 80 / 90 / 100 | 0.6108 / 0.6299 / 0.6320 / 0.6309 |
| son 20 epoch eğimi (puan/10 epoch) | 0.070 |
| önceki 20 epoch (ep 71-90) eğimi | 0.204 |
| son 20 epoch std (puan) | 0.072 |
| ep 81-90 ort. vs ep 91-100 ort. (close_mosaic=10) | 0.6308 vs 0.6314 (Δ +0.07 puan) |
| mAP50 son epoch | 0.8170 |
| results.csv en iyi ep mAP50-95 vs metrics.json (best.pt yeniden val) | 0.6323 vs 0.6299 (fark 0.24 puan; ölçüm-protokolü gürültü tabanı) |

Loss tablosu (train loss mosaic açıkken ep1/ep90, kapalıyken ep100; val/train oranı):

| loss | train ep1 | train ep90 | train ep100 | val ep100 | val/train ep100 | val/train ep90 | train loss ep90→91 değişim |
|---|---|---|---|---|---|---|---|
| box | 1.024 | 0.823 | 0.717 | 0.867 | 1.21 | 1.05 | -6.9% |
| cls | 2.368 | 0.842 | 0.572 | 0.784 | 1.37 | 0.94 | -21.3% |
| dfl | 1.305 | 1.161 | 1.080 | 1.159 | 1.07 | 1.00 | -3.8% |

Kayıp payı (results.csv loss_items kazançlı; loss.py:469-471 — önceki sürüm kazançları iki kez uyguluyordu, 2026-10-01 düzeltildi) ep100: box 30.3%, cls 24.1%, dfl 45.6% (ep1: box 21.8%, cls 50.4%, dfl 27.8%). Son 20 epoch mAP eğimi +0.070 puan/10 ep (std 0.07) → doygun (eğim seed gürültüsü altında). close_mosaic sonrası (ep 91-100) ortalama +0.07 puan. Train loss mosaic kapanınca ani düşer (cls -21.3%); bu rejim değişimi, train/val farkının bir kısmının augmentation kaynaklı olduğunu gösterir. val cls loss son 20 epoch 0.793→0.784 (yükselmiyor): val loss bozulmadığı için klasik aşırı öğrenme yok, ama mosaic kapalıyken val/train oranı cls 1.37, box 1.21: train kaybı val kaybından ayrışıyor (genelleme açığı); val mAP doygun.

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
| ön-işleme ms | 1.3339 |
| çıkarım ms | 6.0172 |
| NMS/son-işlem ms | 0.6845 |
| toplam ms | 8.0356 |
| FPS | 124.446 |
| batch | 1 |
| FP16 (1=evet) | 1.0 |
| ölçüm görüntüsü | 300 |
| ısınma görüntüsü | 50 |
| eğitim süresi (saat) | 3.2788 |

Varyant hız bütçesi (yalnız tahmin; gerçek ölçüm varyant koşumunda yapılır):

| varyant sınıfı | çarpan | tahmini toplam ms | tahmini FPS | not |
|---|---|---|---|---|
| eğitim-zamanı (aug/loss) varyantları | aynı | 8.04 | 124 | çıkarım grafiği değişmez; hız bütçesi ≈ 0 |
| Soft-NMS/DIoU-NMS (son-işlem ×1–5 varsayımı) | NMS ×1–5 | 8.04–10.77 | 93–124 | tahmin (ölçülmedi) |
| TTA (Ultralytics augment, ~3 ölçek+flip; literatür çarpanı 3.6× çıkarım) | çıkarım ×3.6 (literatür) | 23.68 | 42 | tahmin; literatur-cipa.md §4 (Ultralytics TTA dokümanı) |
| YOLO11s'e geçiş (bağlam satırı) | ≈3.4× GFLOPs | – | – | ölçülmedi |

Baseline batch=1 FP16 toplam 8.04 ms (124 FPS; çıkarım 6.02 ms, NMS 0.68 ms = toplamın %9'i). 'Gerçek zamanlı' için ≥30 FPS kabul edilirse (33 ms) ~4.1× gecikme payı vardır; TTA bile (~42 FPS tahmini) eşiğin üstünde kalabilir, ancak hız-doğruluk satırı ayrı raporlanmalıdır. Not: hız ölçümü GPU'nun paylaşıldığı bir anda alınmış olabilir (eğitim kuyruğu); karşılaştırmalı hız ölçümü aynı koşulda yapılmalıdır.
