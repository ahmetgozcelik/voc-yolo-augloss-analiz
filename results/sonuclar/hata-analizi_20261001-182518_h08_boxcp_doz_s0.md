# Hata analizi (K3.5) — baseline `20261001-182518_h08_boxcp_doz_s0`

*Üretici: `scripts/analiz/hata_analizi.py` · yalnız VAL (1000 görüntü, 2407 nesne; difficult nesneler etiketlerde yok) · tahmin sayısı 43291 (conf=0.001, max_det=300, NMS IoU=0.7) · eşleştirme: pycocotools COCOeval · işletim noktası skor≥0.25.*

**Sızıntı notu:** Bu analizde test kümesi (VOC07 test), `rep_test*` dosyaları ve `metrics.json` içindeki `rep_*` anahtarları hiç kullanılmadı; betik bu anahtarları yükler yüklemez atar. Buradan çıkan hipotezler yalnız val bilgisine dayanır.

## 0. Genel ölçüler ve doğrulama

| ölçü | pycocotools (bu betik) | Ultralytics metrics.json (best.pt val) | fark |
|---|---|---|---|
| mAP50-95 | 0.6305 | 0.6312 | -0.0007 |
| mAP50 | 0.8136 | 0.8151 | -0.0015 |
| mAP75 (yalnız pycocotools) | 0.6824 | – | – |

AP50 ile AP50-95 arasındaki fark 18.3 puan; AP50→AP75 farkı 13.1 puan. pycocotools ile Ultralytics AP'si arasındaki küçük fark interpolasyon/eşleştirme (Ultralytics kendi IoU-matrisi eşleşmesini kullanır) kaynaklıdır.

## 1. Sınıf başına AP ve sınıf frekansı

| sınıf (AP50-95 artan) | AP50-95 | AP50 | AP75 | AP50−AP75 | train nesne | val nesne | val GT medyan kenar px |
|---|---|---|---|---|---|---|---|
| pottedplant | 0.311 | 0.559 | 0.297 | 0.261 | 1378 | 109 | 71 |
| chair | 0.483 | 0.658 | 0.539 | 0.119 | 2975 | 177 | 99 |
| bottle | 0.513 | 0.682 | 0.572 | 0.109 | 1669 | 95 | 70 |
| boat | 0.522 | 0.742 | 0.580 | 0.162 | 1076 | 64 | 112 |
| sofa | 0.577 | 0.719 | 0.623 | 0.096 | 755 | 59 | 237 |
| diningtable | 0.583 | 0.728 | 0.643 | 0.085 | 775 | 49 | 211 |
| bird | 0.587 | 0.782 | 0.604 | 0.178 | 1514 | 91 | 115 |
| horse | 0.627 | 0.845 | 0.645 | 0.199 | 1006 | 66 | 209 |
| tvmonitor | 0.628 | 0.841 | 0.716 | 0.125 | 1036 | 72 | 108 |
| person | 0.628 | 0.884 | 0.671 | 0.213 | 12471 | 785 | 141 |
| motorbike | 0.629 | 0.825 | 0.658 | 0.167 | 985 | 67 | 239 |
| car | 0.664 | 0.845 | 0.732 | 0.114 | 3045 | 222 | 92 |
| cow | 0.695 | 0.863 | 0.813 | 0.050 | 791 | 56 | 145 |
| bicycle | 0.702 | 0.893 | 0.763 | 0.131 | 1002 | 62 | 171 |
| aeroplane | 0.709 | 0.891 | 0.722 | 0.169 | 1091 | 80 | 184 |
| train | 0.722 | 0.905 | 0.799 | 0.106 | 868 | 57 | 225 |
| sheep | 0.738 | 0.885 | 0.803 | 0.082 | 1013 | 57 | 119 |
| bus | 0.752 | 0.869 | 0.811 | 0.058 | 767 | 55 | 192 |
| dog | 0.766 | 0.936 | 0.832 | 0.104 | 1929 | 96 | 211 |
| cat | 0.772 | 0.920 | 0.823 | 0.096 | 1505 | 88 | 278 |

En zayıf 5 sınıf (AP50-95): pottedplant 0.311, chair 0.483, bottle 0.513, boat 0.522, sofa 0.577. Sınıf frekansı (train nesne) ile AP50-95 Spearman ρ=-0.13 (permütasyon p=0.589), AP50 ile ρ=0.01; sınıf başına train küçük-nesne oranı ile ρ=-0.37; val GT medyan kenar uzunluğu ile ρ=0.53 (n=20 sınıf, gürültülü). Frekans AP'nin zayıf belirleyicisi (|ρ|<0.3): zayıf sınıflarda sorun veri azlığından çok nesne boyutu/görünümü/bağlamıdır.

## 2. Nesne boyutu kırılımı (orijinal çözünürlük; COCO S<32²≤M<96²≤L)

| boyut | GT sayısı | AP50-95 | AP50 | <e-posta> (tüm conf, sınıf ort.) | <e-posta>:0.95 (sınıf ort.) | <e-posta> skor≥0.25 |
|---|---|---|---|---|---|---|
| all | 2407 | 0.630 | 0.814 | 0.958 | 0.764 | 0.792 |
| S | 143 | 0.215 | 0.350 | 0.701 | 0.423 | 0.343 |
| M | 704 | 0.445 | 0.653 | 0.925 | 0.672 | 0.690 |
| L | 1560 | 0.710 | 0.877 | 0.977 | 0.816 | 0.879 |

İnce alan aralıkları (GT düzeyi, eşleşme bayrağı):

| alan (px²) | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) |
|---|---|---|---|---|
| <16² | 9 | 44.4% | 0.0% | 11.1% |
| 16²–32² | 134 | 85.1% | 36.6% | 55.2% |
| 32²–64² | 377 | 95.0% | 65.8% | 72.7% |
| 64²–96² | 327 | 95.1% | 72.8% | 78.9% |
| 96²–192² | 698 | 97.6% | 82.5% | 84.0% |
| ≥192² | 862 | 98.5% | 92.2% | 91.6% |

Sınıf × boyut AP50-95 (n<5 ise '–'):

| sınıf | n_S | AP_S | n_M | AP_M | n_L | AP_L |
|---|---|---|---|---|---|---|
| aeroplane | 0 | – | 21 | 0.52 | 59 | 0.78 |
| bicycle | 0 | – | 12 | 0.38 | 50 | 0.78 |
| bird | 3 | – | 36 | 0.34 | 52 | 0.77 |
| boat | 4 | – | 25 | 0.50 | 35 | 0.61 |
| bottle | 13 | 0.09 | 52 | 0.49 | 30 | 0.72 |
| bus | 3 | – | 13 | 0.51 | 39 | 0.88 |
| car | 38 | 0.32 | 80 | 0.66 | 104 | 0.77 |
| cat | 0 | – | 2 | – | 86 | 0.79 |
| chair | 3 | – | 79 | 0.46 | 95 | 0.53 |
| cow | 0 | – | 25 | 0.71 | 31 | 0.75 |
| diningtable | 0 | – | 1 | – | 48 | 0.59 |
| dog | 0 | – | 10 | 0.70 | 86 | 0.78 |
| horse | 3 | – | 15 | 0.22 | 48 | 0.77 |
| motorbike | 3 | – | 10 | 0.46 | 54 | 0.68 |
| person | 51 | 0.21 | 215 | 0.49 | 519 | 0.72 |
| pottedplant | 12 | 0.18 | 57 | 0.33 | 40 | 0.37 |
| sheep | 8 | 0.39 | 16 | 0.65 | 33 | 0.86 |
| sofa | 0 | – | 1 | – | 58 | 0.58 |
| train | 0 | – | 4 | – | 53 | 0.74 |
| tvmonitor | 2 | – | 30 | 0.54 | 40 | 0.71 |

Küçük nesneler (n=143, val GT'nin %5.9'i): AP50-95 0.215 vs büyük 0.710 (fark 49.4 puan); <e-posta> 0.701 vs 0.977. Not: ölçüm orijinal piksel; eğitim/çıkarım 640 px letterbox olduğundan (~1.3× büyütme) S nesneler modelde ≈<41 px görünür. Val'de S örneği az olduğundan S-AP güven aralığı geniştir.

## 3. Hata türü ayrıştırması (TIDE benzeri)

Sayılar: tüm tahminler (conf≥0.001) ve işletim noktası (skor≥0.25). `Miss` tüm conf'ta hiçbir kutuyla IoU≥0.5 eşleşmeyen GT; işletim noktası sütununda ise o eşikte bulunamayan GT (eşleşmeyen + düşük skorla eşleşen).

| tür | sayı (tüm conf) | oran | sayı skor≥0.25 |
|---|---|---|---|
| TP | 2317 | 5.4% | 1906 |
| Dupe | 2438 | 5.6% | 84 |
| Cls | 3285 | 7.6% | 102 |
| Loc | 7909 | 18.3% | 146 |
| Both | 5399 | 12.5% | 72 |
| Bkg | 21943 | 50.7% | 254 |
| Miss | 90 | 3.7% (GT'nin) | 501 |

Oracle ΔAP (puan): ilgili hata türü düzeltilirse mAP ne kadar artardı (AP50 ve AP50-95, sınıf ortalaması). Taban: AP50=81.36, AP50-95=63.05.

| düzeltilen | ΔAP50 (puan) | ΔAP50-95 (puan) |
|---|---|---|
| Cls | 2.60 | 1.82 |
| Loc | 2.44 | 5.26 |
| Both | 1.10 | 0.74 |
| Dupe | 0.50 | -0.83 |
| Bkg | 3.47 | 2.22 |
| Miss | 3.51 | 2.62 |
| FP_hepsi | 13.99 | 6.26 |
| Kutu_kusursuz | 0.00 | 18.31 |
| TP_IoU_sirali | 1.44 | 5.50 |

Protokol etkisi: val'de etiketlerden çıkarılmış difficult nesne sayısı 493 (XML). Bunlardan biriyle aynı sınıf IoU≥0.5 örtüşen ve eşleşmemiş (FP sayılan) tahmin: 869 (skor≥0.25: 122; işletim noktası FP'lerinin 18.5%'i). Bu tahminler devkit gibi yok sayılsaydı ΔAP50 = +1.41, ΔAP50-95 = +0.96 puan (Ultralytics protokolünün val üzerinde ölçülmüş cezası; makalede belirtilmeli, Bkg/Loc/Dupe sayılarının bir kısmı bu yüzden yapaydır).


Not: `Dupe` ve `Loc` ΔAP50-95 değerleri IoU=0.5 tanımına göredir; bu kutular daha yüksek IoU eşiklerinde TP olabildiği için silme oracle'ı (Dupe) AP50-95'i düşürebilir; Loc oracle'ı kutuyu GT'ye yapıştırdığı için AP50-95 kazancı AP50'den büyüktür.


Hata türü × boyut (işletim noktası; tam uzun biçim `hata-kirilim.csv`):

| tür | S | M | L |
|---|---|---|---|
| TP (skor≥0.25, tahmin kutusu alanı) | 47 | 469 | 1390 |
| Dupe (skor≥0.25, tahmin kutusu alanı) | 1 | 22 | 61 |
| Cls (skor≥0.25, tahmin kutusu alanı) | 2 | 20 | 80 |
| Loc (skor≥0.25, tahmin kutusu alanı) | 4 | 66 | 76 |
| Both (skor≥0.25, tahmin kutusu alanı) | 1 | 18 | 53 |
| Bkg (skor≥0.25, tahmin kutusu alanı) | 45 | 116 | 93 |
| Miss (skor≥0.25 bulunamayan GT, GT alanı) | 94 | 218 | 189 |
| GT sayısı | 143 | 704 | 1560 |

AP50'ye en çok mal olan hata türü: Miss (+3.51), sonra Bkg (+3.47) ve Cls (+2.60). Oracle'lar birbirinden bağımsız ölçüldü, toplamları ΔAP'ye eşit değildir (etkileşim). `Kutu_kusursuz` = tüm TP kutuları GT'ye yapıştırılırsa AP50-95'in erişeceği üst sınır; `TP_IoU_sirali` = yalnız TP'lerin skorları IoU sırasına göre dağıtılırsa (FP skorları aynı) kazanç, yani IoU-farkında skorlamanın (VFL/QFL/IoU-aware) teorik payı.


Sınıf başına ΔAP50 (puan, oracle):

| sınıf | Cls | Loc | Both | Dupe | Bkg | Miss |
|---|---|---|---|---|---|---|
| aeroplane | 1.0 | 2.2 | 0.2 | 0.5 | 1.4 | 1.2 |
| bicycle | 0.7 | 1.2 | 0.6 | 0.3 | 0.6 | 4.6 |
| bird | 2.7 | 4.5 | 0.4 | 1.0 | 2.7 | 4.5 |
| boat | 2.5 | 4.2 | 0.2 | 0.4 | 7.5 | 5.0 |
| bottle | 0.9 | 1.3 | 0.5 | 0.5 | 10.5 | 1.5 |
| bus | 2.2 | 0.3 | 0.1 | 0.3 | 1.5 | 7.1 |
| car | 1.5 | 2.1 | 0.4 | 0.4 | 3.6 | 4.0 |
| cat | 2.0 | 2.3 | 0.1 | 0.2 | 0.3 | 2.1 |
| chair | 2.9 | 3.1 | 2.1 | 0.7 | 5.9 | 2.6 |
| cow | 5.1 | 2.3 | 0.0 | 0.2 | 4.3 | 1.7 |
| diningtable | 1.4 | 2.7 | 8.3 | 1.2 | 2.5 | 6.6 |
| dog | 1.3 | 0.7 | 0.1 | 0.2 | 0.4 | 2.3 |
| horse | 11.3 | 0.0 | 0.3 | 0.6 | 1.0 | 7.0 |
| motorbike | 3.2 | 3.3 | 0.8 | 0.3 | 2.1 | 4.9 |
| person | 0.3 | 3.3 | 0.1 | 0.9 | 2.2 | 2.3 |
| pottedplant | 0.4 | 8.8 | 2.1 | 1.1 | 6.4 | 6.9 |
| sheep | 6.1 | 0.4 | 0.7 | 0.4 | 3.4 | 0.0 |
| sofa | 6.0 | 1.7 | 3.6 | 0.7 | 4.2 | 1.4 |
| train | 0.2 | 3.5 | 0.4 | 0.0 | 1.7 | 3.6 |
| tvmonitor | 0.2 | 0.7 | 0.9 | 0.2 | 7.4 | 1.2 |

Kaçırılan GT (90) alt kırılımı: yalnız başka sınıfla örtüşen kutu var (sınıf karışıklığı) 28; aynı sınıf kutu var ama IoU∈[0.1,0.5) (lokalizasyon) 31; aynı sınıf IoU≥0.5 kutu var ama başka GT'ye atanmış (atama çakışması/örtüşme) 2; hiç kutu yok (gerçek kaçırma) 29 (kalan sınıflama kesin değil: bu üç grup kesişebilir).

## 4. Lokalizasyon kalitesi

| IoU aralığı | TP (tüm conf) | oran | TP skor≥0.25 | oran |
|---|---|---|---|---|
| 0.50–0.55 | 88 | 3.8% | 35 | 1.8% |
| 0.55–0.60 | 77 | 3.3% | 39 | 2.0% |
| 0.60–0.65 | 95 | 4.1% | 50 | 2.6% |
| 0.65–0.70 | 119 | 5.1% | 75 | 3.9% |
| 0.70–0.75 | 141 | 6.1% | 95 | 5.0% |
| 0.75–0.80 | 169 | 7.3% | 126 | 6.6% |
| 0.80–0.85 | 219 | 9.5% | 174 | 9.1% |
| 0.85–0.90 | 339 | 14.6% | 290 | 15.2% |
| 0.90–0.95 | 503 | 21.7% | 466 | 24.4% |
| 0.95–1.00 | 567 | 24.5% | 556 | 29.2% |

TP IoU ortalaması 0.845 (skor≥0.25: 0.871); işletim noktası TP'lerinin %15.4'i IoU<0.75. TP'lerde skor–IoU Spearman ρ=0.63 (skorun konum kalitesini ne kadar taşıdığı).

Sınıf başına AP50 − AP75 (büyükten küçüğe) ve TP ort. IoU:

| sınıf | AP50 | AP75 | AP50−AP75 (puan) | TP ort. IoU |
|---|---|---|---|---|
| pottedplant | 0.559 | 0.297 | 26.1 | 0.768 |
| person | 0.884 | 0.671 | 21.3 | 0.831 |
| horse | 0.845 | 0.645 | 19.9 | 0.847 |
| bird | 0.782 | 0.604 | 17.8 | 0.844 |
| aeroplane | 0.891 | 0.722 | 16.9 | 0.846 |
| motorbike | 0.825 | 0.658 | 16.7 | 0.837 |
| boat | 0.742 | 0.580 | 16.2 | 0.831 |
| bicycle | 0.893 | 0.763 | 13.1 | 0.880 |
| tvmonitor | 0.841 | 0.716 | 12.5 | 0.851 |
| chair | 0.658 | 0.539 | 11.9 | 0.815 |
| car | 0.845 | 0.732 | 11.4 | 0.854 |
| bottle | 0.682 | 0.572 | 10.9 | 0.828 |
| train | 0.905 | 0.799 | 10.6 | 0.877 |
| dog | 0.936 | 0.832 | 10.4 | 0.897 |
| cat | 0.920 | 0.823 | 9.6 | 0.904 |
| sofa | 0.719 | 0.623 | 9.6 | 0.872 |
| diningtable | 0.728 | 0.643 | 8.5 | 0.864 |
| sheep | 0.885 | 0.803 | 8.2 | 0.889 |
| bus | 0.869 | 0.811 | 5.8 | 0.910 |
| cow | 0.863 | 0.813 | 5.0 | 0.883 |

AP50→AP75 düşüşü sınıf ortalaması 13.1 puan; en büyük düşüş pottedplant (26), person (21), horse (20). Kutu-regresyon (IoU/DFL/box kazancı) manipülasyonlarının hedef alanı budur.

## 5. Sınıf karışıklığı

İşletim noktası (skor≥0.25, IoU≥0.5): her GT için eşleşen en yüksek skorlu kutu (doğru sınıf varsa o). Matris dosyası: `hata-analizi_20261001-182518_h08_boxcp_doz_s0_karisiklik.csv`.

| GT sınıfı → tahmin | sayı | GT'nin payı |
|---|---|---|
| chair → person | 5 | 2.8% |
| horse → sheep | 5 | 7.6% |
| bus → car | 3 | 5.5% |
| cow → horse | 3 | 5.4% |
| cow → sheep | 3 | 5.4% |
| sheep → cow | 3 | 5.3% |
| diningtable → chair | 2 | 4.1% |
| dog → cat | 2 | 2.1% |
| motorbike → person | 2 | 3.0% |
| person → cat | 2 | 0.3% |
| sofa → chair | 2 | 3.4% |
| sofa → person | 2 | 3.4% |

Çift bazında (yönsüz) toplam: cow/sheep 6; chair/person 5; horse/sheep 5; bus/car 3; cow/horse 3; chair/diningtable 3; cat/dog 2; motorbike/person 2.


Tüm-conf alternatif görünüm: `Cls` türü tahminler (IoU≥0.5 başka sınıf GT, skor≥0.05), GT sınıfı → tahmin:

| GT → tahmin | tahmin sayısı |
|---|---|
| person → chair | 25 |
| sofa → chair | 22 |
| person → sofa | 16 |
| dog → cat | 14 |
| cow → horse | 10 |
| cat → dog | 9 |
| cow → sheep | 8 |
| sheep → dog | 8 |
| sheep → cow | 8 |
| horse → sheep | 8 |

Kanonik çiftler (yönsüz; `Cls` tahmin sayısı skor≥0.05 | işletim noktası GT sayısı): cat/dog 23 | 2; cow/sheep 16 | 6; chair/diningtable 8 | 3; chair/sofa 29 | 2; horse/cow 17 | 3; bus/train 3 | 1; motorbike/bicycle 11 | 1.

İşletim noktasında GT'lerin 2.3%'i başka sınıf olarak etiketlendi (56/2407); 17.9%'i hiç bulunamadı. Sınıf karışıklığı oracle ΔAP50 = 2.60 puan (§3; Cls, altı hata türü içinde 3. sırada); işletim noktasında karışan GT oranı yalnız 2.3%, yani sorun sınıf ayrımından çok kutu/skor düzeyindedir.

## 6. Kalabalık ve örtüşme

| nesne/görüntü | görüntü | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) | FP/görüntü (skor≥0.25) |
|---|---|---|---|---|---|---|
| 1 | 440 | 440 | 98.9% | 92.0% | 92.3% | 0.35 |
| 2 | 252 | 504 | 97.6% | 83.9% | 88.1% | 0.46 |
| 3–4 | 195 | 654 | 95.1% | 77.8% | 81.8% | 0.99 |
| 5–8 | 88 | 511 | 94.1% | 71.0% | 77.1% | 1.32 |
| ≥9 | 25 | 298 | 96.3% | 69.1% | 68.5% | 3.20 |

Aynı sınıftan başka GT ile en yüksek IoU (örtüşme) gruplarına göre recall:

| komşu GT IoU | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 |
|---|---|---|---|
| <0.1 (komşu yok) | 1908 | 96.1% | 80.4% |
| 0.1–0.3 | 357 | 96.1% | 75.1% |
| 0.3–0.5 | 105 | 100.0% | 75.2% |
| 0.5–0.7 | 33 | 97.0% | 66.7% |
| ≥0.7 | 4 | 75.0% | 75.0% |

Aynı sınıf GT çiftleri (IoU≥0.3): 75 çift; ikisi de bulundu 72, yalnız biri 3, hiçbiri 0. IoU≥0.7 (NMS eşiği üstü, doğrudan NMS etkilenir): 2 çift, ikisi de bulundu 1, yalnız biri 1, hiçbiri 0. Kaçırılan GT'lerin 2/90'i aynı sınıf IoU≥0.5 bir kutuya sahip ama o kutu başka GT'ye atanmış (olası NMS/atama kaybı).

Kalabalıkla recall düşüşü: tek nesneli görüntüde 92.0% iken ≥9 nesnelide 69.1% (skor≥0.25); tüm-conf recall 98.9% → 96.3%, <e-posta> 92.3% → 68.5%. Çiftlerde tek-bulunma ve atama çakışması sayıları küçükse kayıp NMS'ten değil düşük skor/zayıf konumdan gelir.

## 7. Kalibrasyon (skor vs gerçek precision)

Tahmin 'doğru' = IoU≥0.5 ile eşleşen (<e-posta>); ayrıca IoU≥0.75 için precision. Not: YOLO sınıf skoru olasılık değil, TAL hizalama hedefli (IoU'ya bağlı) bir skordur; düşük skor bölgesindeki düşük precision 'aday havuzu' doğasından gelir.

| skor aralığı | tahmin | ort. skor | <e-posta> | <e-posta> | TP ort. IoU | skor−precision |
|---|---|---|---|---|---|---|
| 0.0–0.1 | 39694 | 0.008 | 0.006 | 0.006 | 0.721 | 0.001 |
| 0.1–0.2 | 813 | 0.142 | 0.135 | 0.089 | 0.740 | 0.007 |
| 0.2–0.3 | 373 | 0.243 | 0.220 | 0.137 | 0.729 | 0.023 |
| 0.3–0.4 | 258 | 0.345 | 0.306 | 0.205 | 0.775 | 0.039 |
| 0.4–0.5 | 207 | 0.448 | 0.396 | 0.285 | 0.782 | 0.052 |
| 0.5–0.6 | 209 | 0.552 | 0.598 | 0.426 | 0.802 | -0.046 |
| 0.6–0.7 | 220 | 0.650 | 0.673 | 0.523 | 0.820 | -0.022 |
| 0.7–0.8 | 338 | 0.755 | 0.867 | 0.669 | 0.832 | -0.112 |
| 0.8–0.9 | 629 | 0.858 | 0.948 | 0.847 | 0.883 | -0.090 |
| 0.9–1.0 | 550 | 0.927 | 0.991 | 0.978 | 0.944 | -0.064 |

ECE (tüm tahminler, 10 eşit genişlikli kutu) = 0.005; ECE (skor≥0.1) = 0.050. `skor−precision` pozitifse aşırı güvenli, negatifse eksik güvenli. 
Skor≥0.5 tahminlerin <e-posta> = 87.7%, <e-posta> = 77.1% (fark 10.6 puan). Skor 0.5 üstü kutularda (skor−precision) ortalaması -0.067: model yüksek skorlarda eksik güvenli (skor < gerçek <e-posta>), sıralama iyi; asıl boşluk <e-posta> ile @0.75 arasındaki konum kalitesidir. TP'lerde skor–IoU Spearman ρ=0.63: skor konum kalitesini kısmen taşıyor. TP_IoU_sirali oracle'ı (§3) IoU-farkında skorlamanın teorik payını sayısallaştırıyor.

## 8. Eğitim dinamiği (results.csv)

| gösterge | değer |
|---|---|
| epoch sayısı | 100 |
| en iyi epoch (mAP50-95) | 97 (0.6300) |
| son epoch mAP50-95 | 0.6296 |
| epoch 50 / 80 / 90 / 100 | 0.6129 / 0.6281 / 0.6285 / 0.6296 |
| son 20 epoch eğimi (puan/10 epoch) | 0.075 |
| önceki 20 epoch (ep 71-90) eğimi | 0.112 |
| son 20 epoch std (puan) | 0.054 |
| ep 81-90 ort. vs ep 91-100 ort. (close_mosaic=10) | 0.6286 vs 0.6294 (Δ +0.07 puan) |
| mAP50 son epoch | 0.8157 |
| results.csv en iyi ep mAP50-95 vs metrics.json (best.pt yeniden val) | 0.6300 vs 0.6312 (fark -0.11 puan; ölçüm-protokolü gürültü tabanı) |

Loss tablosu (train loss mosaic açıkken ep1/ep90, kapalıyken ep100; val/train oranı):

| loss | train ep1 | train ep90 | train ep100 | val ep100 | val/train ep100 | val/train ep90 | train loss ep90→91 değişim |
|---|---|---|---|---|---|---|---|
| box | 0.965 | 0.758 | 0.696 | 0.874 | 1.26 | 1.16 | -0.9% |
| cls | 2.243 | 0.790 | 0.549 | 0.787 | 1.43 | 1.00 | -18.6% |
| dfl | 1.242 | 1.107 | 1.063 | 1.163 | 1.09 | 1.05 | -0.1% |

Kayıp payı (results.csv loss_items kazançlı; loss.py:469-471 — önceki sürüm kazançları iki kez uyguluyordu, 2026-10-01 düzeltildi) ep100: box 30.2%, cls 23.8%, dfl 46.1% (ep1: box 21.7%, cls 50.4%, dfl 27.9%). Son 20 epoch mAP eğimi +0.075 puan/10 ep (std 0.05) → doygun (eğim seed gürültüsü altında). close_mosaic sonrası (ep 91-100) ortalama +0.07 puan. Train loss mosaic kapanınca ani düşer (cls -18.6%); bu rejim değişimi, train/val farkının bir kısmının augmentation kaynaklı olduğunu gösterir. val cls loss son 20 epoch 0.798→0.787 (yükselmiyor): val loss bozulmadığı için klasik aşırı öğrenme yok, ama mosaic kapalıyken val/train oranı cls 1.43, box 1.26: train kaybı val kaybından ayrışıyor (genelleme açığı); val mAP doygun.

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
| ön-işleme ms | 1.3576 |
| çıkarım ms | 6.9726 |
| NMS/son-işlem ms | 0.8086 |
| toplam ms | 9.1388 |
| FPS | 109.423 |
| batch | 1 |
| FP16 (1=evet) | 1.0 |
| ölçüm görüntüsü | 300 |
| ısınma görüntüsü | 50 |
| eğitim süresi (saat) | 2.9587 |

Varyant hız bütçesi (yalnız tahmin; gerçek ölçüm varyant koşumunda yapılır):

| varyant sınıfı | çarpan | tahmini toplam ms | tahmini FPS | not |
|---|---|---|---|---|
| eğitim-zamanı (aug/loss) varyantları | aynı | 9.14 | 109 | çıkarım grafiği değişmez; hız bütçesi ≈ 0 |
| Soft-NMS/DIoU-NMS (son-işlem ×1–5 varsayımı) | NMS ×1–5 | 9.14–12.37 | 81–109 | tahmin (ölçülmedi) |
| TTA (Ultralytics augment, ~3 ölçek+flip; literatür çarpanı 3.6× çıkarım) | çıkarım ×3.6 (literatür) | 27.27 | 37 | tahmin; literatur-cipa.md §4 (Ultralytics TTA dokümanı) |
| YOLO11s'e geçiş (bağlam satırı) | ≈3.4× GFLOPs | – | – | ölçülmedi |

Baseline batch=1 FP16 toplam 9.14 ms (109 FPS; çıkarım 6.97 ms, NMS 0.81 ms = toplamın %9'i). 'Gerçek zamanlı' için ≥30 FPS kabul edilirse (33 ms) ~3.6× gecikme payı vardır; TTA bile (~37 FPS tahmini) eşiğin üstünde kalabilir, ancak hız-doğruluk satırı ayrı raporlanmalıdır. Not: hız ölçümü GPU'nun paylaşıldığı bir anda alınmış olabilir (eğitim kuyruğu); karşılaştırmalı hız ölçümü aynı koşulda yapılmalıdır.
