# Hata analizi (K3.5) — baseline `20261002-095722_h08_boxcp_doz_s2`

*Üretici: `scripts/analiz/hata_analizi.py` · yalnız VAL (1000 görüntü, 2407 nesne; difficult nesneler etiketlerde yok) · tahmin sayısı 42727 (conf=0.001, max_det=300, NMS IoU=0.7) · eşleştirme: pycocotools COCOeval · işletim noktası skor≥0.25.*

**Sızıntı notu:** Bu analizde test kümesi (VOC07 test), `rep_test*` dosyaları ve `metrics.json` içindeki `rep_*` anahtarları hiç kullanılmadı; betik bu anahtarları yükler yüklemez atar. Buradan çıkan hipotezler yalnız val bilgisine dayanır.

## 0. Genel ölçüler ve doğrulama

| ölçü | pycocotools (bu betik) | Ultralytics metrics.json (best.pt val) | fark |
|---|---|---|---|
| mAP50-95 | 0.6276 | 0.6284 | -0.0007 |
| mAP50 | 0.8119 | 0.8141 | -0.0022 |
| mAP75 (yalnız pycocotools) | 0.6785 | – | – |

AP50 ile AP50-95 arasındaki fark 18.4 puan; AP50→AP75 farkı 13.3 puan. pycocotools ile Ultralytics AP'si arasındaki küçük fark interpolasyon/eşleştirme (Ultralytics kendi IoU-matrisi eşleşmesini kullanır) kaynaklıdır.

## 1. Sınıf başına AP ve sınıf frekansı

| sınıf (AP50-95 artan) | AP50-95 | AP50 | AP75 | AP50−AP75 | train nesne | val nesne | val GT medyan kenar px |
|---|---|---|---|---|---|---|---|
| pottedplant | 0.297 | 0.552 | 0.243 | 0.309 | 1378 | 109 | 71 |
| chair | 0.467 | 0.651 | 0.530 | 0.121 | 2975 | 177 | 99 |
| bottle | 0.504 | 0.666 | 0.567 | 0.099 | 1669 | 95 | 70 |
| boat | 0.551 | 0.771 | 0.620 | 0.150 | 1076 | 64 | 112 |
| sofa | 0.574 | 0.727 | 0.600 | 0.127 | 755 | 59 | 237 |
| bird | 0.583 | 0.791 | 0.597 | 0.193 | 1514 | 91 | 115 |
| diningtable | 0.601 | 0.718 | 0.704 | 0.014 | 775 | 49 | 211 |
| tvmonitor | 0.620 | 0.836 | 0.704 | 0.133 | 1036 | 72 | 108 |
| motorbike | 0.623 | 0.838 | 0.672 | 0.167 | 985 | 67 | 239 |
| horse | 0.626 | 0.835 | 0.647 | 0.188 | 1006 | 66 | 209 |
| person | 0.635 | 0.883 | 0.673 | 0.210 | 12471 | 785 | 141 |
| car | 0.663 | 0.847 | 0.732 | 0.115 | 3045 | 222 | 92 |
| cow | 0.665 | 0.841 | 0.752 | 0.090 | 791 | 56 | 145 |
| aeroplane | 0.701 | 0.905 | 0.734 | 0.170 | 1091 | 80 | 184 |
| bicycle | 0.720 | 0.906 | 0.775 | 0.132 | 1002 | 62 | 171 |
| train | 0.724 | 0.869 | 0.798 | 0.071 | 868 | 57 | 225 |
| bus | 0.730 | 0.849 | 0.794 | 0.055 | 767 | 55 | 192 |
| sheep | 0.745 | 0.910 | 0.764 | 0.145 | 1013 | 57 | 119 |
| cat | 0.748 | 0.900 | 0.815 | 0.085 | 1505 | 88 | 278 |
| dog | 0.775 | 0.944 | 0.848 | 0.095 | 1929 | 96 | 211 |

En zayıf 5 sınıf (AP50-95): pottedplant 0.297, chair 0.467, bottle 0.504, boat 0.551, sofa 0.574. Sınıf frekansı (train nesne) ile AP50-95 Spearman ρ=-0.12 (permütasyon p=0.620), AP50 ile ρ=0.04; sınıf başına train küçük-nesne oranı ile ρ=-0.37; val GT medyan kenar uzunluğu ile ρ=0.51 (n=20 sınıf, gürültülü). Frekans AP'nin zayıf belirleyicisi (|ρ|<0.3): zayıf sınıflarda sorun veri azlığından çok nesne boyutu/görünümü/bağlamıdır.

## 2. Nesne boyutu kırılımı (orijinal çözünürlük; COCO S<32²≤M<96²≤L)

| boyut | GT sayısı | AP50-95 | AP50 | <e-posta> (tüm conf, sınıf ort.) | <e-posta>:0.95 (sınıf ort.) | <e-posta> skor≥0.25 |
|---|---|---|---|---|---|---|
| all | 2407 | 0.628 | 0.812 | 0.956 | 0.762 | 0.787 |
| S | 143 | 0.216 | 0.373 | 0.730 | 0.462 | 0.350 |
| M | 704 | 0.440 | 0.652 | 0.922 | 0.649 | 0.695 |
| L | 1560 | 0.703 | 0.871 | 0.976 | 0.814 | 0.869 |

İnce alan aralıkları (GT düzeyi, eşleşme bayrağı):

| alan (px²) | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) |
|---|---|---|---|---|
| <16² | 9 | 44.4% | 0.0% | 22.2% |
| 16²–32² | 134 | 86.6% | 37.3% | 59.0% |
| 32²–64² | 377 | 95.2% | 65.8% | 74.3% |
| 64²–96² | 327 | 93.9% | 73.7% | 78.0% |
| 96²–192² | 698 | 97.6% | 81.5% | 83.5% |
| ≥192² | 862 | 98.6% | 91.2% | 91.0% |

Sınıf × boyut AP50-95 (n<5 ise '–'):

| sınıf | n_S | AP_S | n_M | AP_M | n_L | AP_L |
|---|---|---|---|---|---|---|
| aeroplane | 0 | – | 21 | 0.52 | 59 | 0.76 |
| bicycle | 0 | – | 12 | 0.42 | 50 | 0.80 |
| bird | 3 | – | 36 | 0.31 | 52 | 0.77 |
| boat | 4 | – | 25 | 0.56 | 35 | 0.61 |
| bottle | 13 | 0.11 | 52 | 0.50 | 30 | 0.67 |
| bus | 3 | – | 13 | 0.44 | 39 | 0.89 |
| car | 38 | 0.35 | 80 | 0.66 | 104 | 0.77 |
| cat | 0 | – | 2 | – | 86 | 0.77 |
| chair | 3 | – | 79 | 0.45 | 95 | 0.50 |
| cow | 0 | – | 25 | 0.66 | 31 | 0.72 |
| diningtable | 0 | – | 1 | – | 48 | 0.61 |
| dog | 0 | – | 10 | 0.70 | 86 | 0.79 |
| horse | 3 | – | 15 | 0.25 | 48 | 0.76 |
| motorbike | 3 | – | 10 | 0.56 | 54 | 0.66 |
| person | 51 | 0.21 | 215 | 0.49 | 519 | 0.73 |
| pottedplant | 12 | 0.19 | 57 | 0.33 | 40 | 0.34 |
| sheep | 8 | 0.38 | 16 | 0.60 | 33 | 0.88 |
| sofa | 0 | – | 1 | – | 58 | 0.58 |
| train | 0 | – | 4 | – | 53 | 0.74 |
| tvmonitor | 2 | – | 30 | 0.52 | 40 | 0.70 |

Küçük nesneler (n=143, val GT'nin %5.9'i): AP50-95 0.216 vs büyük 0.703 (fark 48.7 puan); <e-posta> 0.730 vs 0.976. Not: ölçüm orijinal piksel; eğitim/çıkarım 640 px letterbox olduğundan (~1.3× büyütme) S nesneler modelde ≈<41 px görünür. Val'de S örneği az olduğundan S-AP güven aralığı geniştir.

## 3. Hata türü ayrıştırması (TIDE benzeri)

Sayılar: tüm tahminler (conf≥0.001) ve işletim noktası (skor≥0.25). `Miss` tüm conf'ta hiçbir kutuyla IoU≥0.5 eşleşmeyen GT; işletim noktası sütununda ise o eşikte bulunamayan GT (eşleşmeyen + düşük skorla eşleşen).

| tür | sayı (tüm conf) | oran | sayı skor≥0.25 |
|---|---|---|---|
| TP | 2317 | 5.4% | 1894 |
| Dupe | 2407 | 5.6% | 83 |
| Cls | 3281 | 7.7% | 99 |
| Loc | 8081 | 18.9% | 148 |
| Both | 5308 | 12.4% | 82 |
| Bkg | 21333 | 49.9% | 271 |
| Miss | 90 | 3.7% (GT'nin) | 513 |

Oracle ΔAP (puan): ilgili hata türü düzeltilirse mAP ne kadar artardı (AP50 ve AP50-95, sınıf ortalaması). Taban: AP50=81.19, AP50-95=62.76.

| düzeltilen | ΔAP50 (puan) | ΔAP50-95 (puan) |
|---|---|---|
| Cls | 2.79 | 2.13 |
| Loc | 2.82 | 5.58 |
| Both | 1.18 | 0.81 |
| Dupe | 0.50 | -0.81 |
| Bkg | 3.26 | 2.10 |
| Miss | 3.73 | 2.77 |
| FP_hepsi | 14.01 | 6.35 |
| Kutu_kusursuz | 0.00 | 18.43 |
| TP_IoU_sirali | 1.40 | 5.41 |

Protokol etkisi: val'de etiketlerden çıkarılmış difficult nesne sayısı 493 (XML). Bunlardan biriyle aynı sınıf IoU≥0.5 örtüşen ve eşleşmemiş (FP sayılan) tahmin: 846 (skor≥0.25: 120; işletim noktası FP'lerinin 17.6%'i). Bu tahminler devkit gibi yok sayılsaydı ΔAP50 = +1.29, ΔAP50-95 = +0.87 puan (Ultralytics protokolünün val üzerinde ölçülmüş cezası; makalede belirtilmeli, Bkg/Loc/Dupe sayılarının bir kısmı bu yüzden yapaydır).


Not: `Dupe` ve `Loc` ΔAP50-95 değerleri IoU=0.5 tanımına göredir; bu kutular daha yüksek IoU eşiklerinde TP olabildiği için silme oracle'ı (Dupe) AP50-95'i düşürebilir; Loc oracle'ı kutuyu GT'ye yapıştırdığı için AP50-95 kazancı AP50'den büyüktür.


Hata türü × boyut (işletim noktası; tam uzun biçim `hata-kirilim.csv`):

| tür | S | M | L |
|---|---|---|---|
| TP (skor≥0.25, tahmin kutusu alanı) | 46 | 486 | 1362 |
| Dupe (skor≥0.25, tahmin kutusu alanı) | 2 | 25 | 56 |
| Cls (skor≥0.25, tahmin kutusu alanı) | 1 | 20 | 78 |
| Loc (skor≥0.25, tahmin kutusu alanı) | 4 | 68 | 76 |
| Both (skor≥0.25, tahmin kutusu alanı) | 0 | 18 | 64 |
| Bkg (skor≥0.25, tahmin kutusu alanı) | 48 | 115 | 108 |
| Miss (skor≥0.25 bulunamayan GT, GT alanı) | 93 | 215 | 205 |
| GT sayısı | 143 | 704 | 1560 |

AP50'ye en çok mal olan hata türü: Miss (+3.73), sonra Bkg (+3.26) ve Loc (+2.82). Oracle'lar birbirinden bağımsız ölçüldü, toplamları ΔAP'ye eşit değildir (etkileşim). `Kutu_kusursuz` = tüm TP kutuları GT'ye yapıştırılırsa AP50-95'in erişeceği üst sınır; `TP_IoU_sirali` = yalnız TP'lerin skorları IoU sırasına göre dağıtılırsa (FP skorları aynı) kazanç, yani IoU-farkında skorlamanın (VFL/QFL/IoU-aware) teorik payı.


Sınıf başına ΔAP50 (puan, oracle):

| sınıf | Cls | Loc | Both | Dupe | Bkg | Miss |
|---|---|---|---|---|---|---|
| aeroplane | 0.3 | 2.2 | 0.2 | 0.6 | 1.3 | 0.0 |
| bicycle | 1.3 | 1.3 | 0.9 | 0.3 | 0.5 | 3.3 |
| bird | 1.2 | 5.4 | 0.3 | 1.0 | 2.4 | 6.5 |
| boat | 3.9 | 3.6 | 0.3 | 0.8 | 5.0 | 5.2 |
| bottle | 1.0 | 3.9 | 0.4 | 0.5 | 10.1 | 3.5 |
| bus | 0.8 | 1.0 | 0.4 | 0.5 | 1.1 | 8.9 |
| car | 1.2 | 2.2 | 0.8 | 0.5 | 3.4 | 4.4 |
| cat | 4.4 | 3.0 | 0.1 | 0.3 | 0.1 | 3.1 |
| chair | 2.7 | 4.1 | 2.2 | 0.6 | 5.6 | 1.6 |
| cow | 6.4 | 0.6 | 0.0 | 0.6 | 5.0 | 0.0 |
| diningtable | 2.0 | 1.6 | 8.6 | 0.1 | 2.9 | 4.9 |
| dog | 2.4 | 0.1 | 0.3 | 0.2 | 0.4 | 0.0 |
| horse | 12.3 | 0.9 | 0.1 | 0.5 | 0.5 | 8.7 |
| motorbike | 1.4 | 4.0 | 0.5 | 0.4 | 2.0 | 5.2 |
| person | 0.1 | 3.4 | 0.3 | 1.0 | 2.1 | 1.8 |
| pottedplant | 0.6 | 9.2 | 2.0 | 1.4 | 6.8 | 8.7 |
| sheep | 5.0 | 0.0 | 0.2 | 0.1 | 2.9 | 0.0 |
| sofa | 7.3 | 1.5 | 3.7 | 0.6 | 5.5 | 1.2 |
| train | 1.2 | 7.2 | 0.2 | 0.1 | 1.5 | 5.1 |
| tvmonitor | 0.5 | 1.2 | 2.1 | 0.1 | 6.3 | 2.5 |

Kaçırılan GT (90) alt kırılımı: yalnız başka sınıfla örtüşen kutu var (sınıf karışıklığı) 24; aynı sınıf kutu var ama IoU∈[0.1,0.5) (lokalizasyon) 38; aynı sınıf IoU≥0.5 kutu var ama başka GT'ye atanmış (atama çakışması/örtüşme) 1; hiç kutu yok (gerçek kaçırma) 27 (kalan sınıflama kesin değil: bu üç grup kesişebilir).

## 4. Lokalizasyon kalitesi

| IoU aralığı | TP (tüm conf) | oran | TP skor≥0.25 | oran |
|---|---|---|---|---|
| 0.50–0.55 | 76 | 3.3% | 36 | 1.9% |
| 0.55–0.60 | 80 | 3.5% | 35 | 1.8% |
| 0.60–0.65 | 89 | 3.8% | 44 | 2.3% |
| 0.65–0.70 | 121 | 5.2% | 75 | 4.0% |
| 0.70–0.75 | 141 | 6.1% | 104 | 5.5% |
| 0.75–0.80 | 164 | 7.1% | 108 | 5.7% |
| 0.80–0.85 | 236 | 10.2% | 189 | 10.0% |
| 0.85–0.90 | 311 | 13.4% | 261 | 13.8% |
| 0.90–0.95 | 541 | 23.3% | 503 | 26.6% |
| 0.95–1.00 | 558 | 24.1% | 539 | 28.5% |

TP IoU ortalaması 0.847 (skor≥0.25: 0.872); işletim noktası TP'lerinin %15.5'i IoU<0.75. TP'lerde skor–IoU Spearman ρ=0.62 (skorun konum kalitesini ne kadar taşıdığı).

Sınıf başına AP50 − AP75 (büyükten küçüğe) ve TP ort. IoU:

| sınıf | AP50 | AP75 | AP50−AP75 (puan) | TP ort. IoU |
|---|---|---|---|---|
| pottedplant | 0.552 | 0.243 | 30.9 | 0.757 |
| person | 0.883 | 0.673 | 21.0 | 0.837 |
| bird | 0.791 | 0.597 | 19.3 | 0.838 |
| horse | 0.835 | 0.647 | 18.8 | 0.843 |
| aeroplane | 0.905 | 0.734 | 17.0 | 0.852 |
| motorbike | 0.838 | 0.672 | 16.7 | 0.844 |
| boat | 0.771 | 0.620 | 15.0 | 0.820 |
| sheep | 0.910 | 0.764 | 14.5 | 0.892 |
| tvmonitor | 0.836 | 0.704 | 13.3 | 0.860 |
| bicycle | 0.906 | 0.775 | 13.2 | 0.865 |
| sofa | 0.727 | 0.600 | 12.7 | 0.878 |
| chair | 0.651 | 0.530 | 12.1 | 0.818 |
| car | 0.847 | 0.732 | 11.5 | 0.855 |
| bottle | 0.666 | 0.567 | 9.9 | 0.845 |
| dog | 0.944 | 0.848 | 9.5 | 0.897 |
| cow | 0.841 | 0.752 | 9.0 | 0.867 |
| cat | 0.900 | 0.815 | 8.5 | 0.898 |
| train | 0.869 | 0.798 | 7.1 | 0.889 |
| bus | 0.849 | 0.794 | 5.5 | 0.888 |
| diningtable | 0.718 | 0.704 | 1.4 | 0.885 |

AP50→AP75 düşüşü sınıf ortalaması 13.3 puan; en büyük düşüş pottedplant (31), person (21), bird (19). Kutu-regresyon (IoU/DFL/box kazancı) manipülasyonlarının hedef alanı budur.

## 5. Sınıf karışıklığı

İşletim noktası (skor≥0.25, IoU≥0.5): her GT için eşleşen en yüksek skorlu kutu (doğru sınıf varsa o). Matris dosyası: `hata-analizi_20261002-095722_h08_boxcp_doz_s2_karisiklik.csv`.

| GT sınıfı → tahmin | sayı | GT'nin payı |
|---|---|---|
| chair → person | 6 | 3.4% |
| horse → sheep | 6 | 9.1% |
| cow → horse | 5 | 8.9% |
| sofa → chair | 5 | 8.5% |
| cow → sheep | 4 | 7.1% |
| sofa → person | 3 | 5.1% |
| car → boat | 2 | 0.9% |
| chair → diningtable | 2 | 1.1% |
| motorbike → person | 2 | 3.0% |
| bicycle → motorbike | 1 | 1.6% |
| bicycle → person | 1 | 1.6% |
| boat → chair | 1 | 1.6% |

Çift bazında (yönsüz) toplam: chair/person 6; horse/sheep 6; chair/sofa 6; cow/horse 5; cow/sheep 4; person/sofa 3; chair/diningtable 3; motorbike/person 3.


Tüm-conf alternatif görünüm: `Cls` türü tahminler (IoU≥0.5 başka sınıf GT, skor≥0.05), GT sınıfı → tahmin:

| GT → tahmin | tahmin sayısı |
|---|---|
| sofa → chair | 27 |
| person → chair | 19 |
| person → sofa | 17 |
| dog → cat | 14 |
| chair → sofa | 11 |
| motorbike → bicycle | 11 |
| cow → horse | 9 |
| horse → sheep | 9 |
| cat → dog | 8 |
| cow → sheep | 7 |

Kanonik çiftler (yönsüz; `Cls` tahmin sayısı skor≥0.05 | işletim noktası GT sayısı): cat/dog 22 | 2; cow/sheep 11 | 4; chair/diningtable 7 | 3; chair/sofa 38 | 6; horse/cow 10 | 5; bus/train 5 | 0; motorbike/bicycle 17 | 1.

İşletim noktasında GT'lerin 2.4%'i başka sınıf olarak etiketlendi (58/2407); 18.4%'i hiç bulunamadı. Sınıf karışıklığı oracle ΔAP50 = 2.79 puan (§3; Cls, altı hata türü içinde 4. sırada); işletim noktasında karışan GT oranı yalnız 2.4%, yani sorun sınıf ayrımından çok kutu/skor düzeyindedir.

## 6. Kalabalık ve örtüşme

| nesne/görüntü | görüntü | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) | FP/görüntü (skor≥0.25) |
|---|---|---|---|---|---|---|
| 1 | 440 | 440 | 98.0% | 91.1% | 90.7% | 0.40 |
| 2 | 252 | 504 | 96.8% | 83.7% | 87.9% | 0.51 |
| 3–4 | 195 | 654 | 96.2% | 76.3% | 82.7% | 0.90 |
| 5–8 | 88 | 511 | 94.7% | 70.8% | 75.3% | 1.36 |
| ≥9 | 25 | 298 | 95.6% | 70.5% | 72.1% | 3.32 |

Aynı sınıftan başka GT ile en yüksek IoU (örtüşme) gruplarına göre recall:

| komşu GT IoU | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 |
|---|---|---|---|
| <0.1 (komşu yok) | 1908 | 96.3% | 79.5% |
| 0.1–0.3 | 357 | 94.7% | 76.5% |
| 0.3–0.5 | 105 | 100.0% | 75.2% |
| 0.5–0.7 | 33 | 97.0% | 66.7% |
| ≥0.7 | 4 | 100.0% | 75.0% |

Aynı sınıf GT çiftleri (IoU≥0.3): 75 çift; ikisi de bulundu 73, yalnız biri 2, hiçbiri 0. IoU≥0.7 (NMS eşiği üstü, doğrudan NMS etkilenir): 2 çift, ikisi de bulundu 2, yalnız biri 0, hiçbiri 0. Kaçırılan GT'lerin 1/90'i aynı sınıf IoU≥0.5 bir kutuya sahip ama o kutu başka GT'ye atanmış (olası NMS/atama kaybı).

Kalabalıkla recall düşüşü: tek nesneli görüntüde 91.1% iken ≥9 nesnelide 70.5% (skor≥0.25); tüm-conf recall 98.0% → 95.6%, <e-posta> 90.7% → 72.1%. Çiftlerde tek-bulunma ve atama çakışması sayıları küçükse kayıp NMS'ten değil düşük skor/zayıf konumdan gelir.

## 7. Kalibrasyon (skor vs gerçek precision)

Tahmin 'doğru' = IoU≥0.5 ile eşleşen (<e-posta>); ayrıca IoU≥0.75 için precision. Not: YOLO sınıf skoru olasılık değil, TAL hizalama hedefli (IoU'ya bağlı) bir skordur; düşük skor bölgesindeki düşük precision 'aday havuzu' doğasından gelir.

| skor aralığı | tahmin | ort. skor | <e-posta> | <e-posta> | TP ort. IoU | skor−precision |
|---|---|---|---|---|---|---|
| 0.0–0.1 | 39085 | 0.008 | 0.007 | 0.006 | 0.728 | 0.001 |
| 0.1–0.2 | 842 | 0.142 | 0.125 | 0.097 | 0.756 | 0.018 |
| 0.2–0.3 | 365 | 0.245 | 0.195 | 0.145 | 0.771 | 0.051 |
| 0.3–0.4 | 294 | 0.348 | 0.320 | 0.221 | 0.774 | 0.029 |
| 0.4–0.5 | 213 | 0.449 | 0.427 | 0.244 | 0.772 | 0.022 |
| 0.5–0.6 | 200 | 0.552 | 0.545 | 0.390 | 0.801 | 0.007 |
| 0.6–0.7 | 219 | 0.649 | 0.685 | 0.502 | 0.818 | -0.036 |
| 0.7–0.8 | 313 | 0.752 | 0.834 | 0.658 | 0.835 | -0.082 |
| 0.8–0.9 | 612 | 0.856 | 0.961 | 0.861 | 0.884 | -0.104 |
| 0.9–1.0 | 584 | 0.927 | 0.988 | 0.961 | 0.938 | -0.061 |

ECE (tüm tahminler, 10 eşit genişlikli kutu) = 0.005; ECE (skor≥0.1) = 0.050. `skor−precision` pozitifse aşırı güvenli, negatifse eksik güvenli. 
Skor≥0.5 tahminlerin <e-posta> = 87.4%, <e-posta> = 76.9% (fark 10.5 puan). Skor 0.5 üstü kutularda (skor−precision) ortalaması -0.055: model yüksek skorlarda eksik güvenli (skor < gerçek <e-posta>), sıralama iyi; asıl boşluk <e-posta> ile @0.75 arasındaki konum kalitesidir. TP'lerde skor–IoU Spearman ρ=0.62: skor konum kalitesini kısmen taşıyor. TP_IoU_sirali oracle'ı (§3) IoU-farkında skorlamanın teorik payını sayısallaştırıyor.

## 8. Eğitim dinamiği (results.csv)

| gösterge | değer |
|---|---|
| epoch sayısı | 100 |
| en iyi epoch (mAP50-95) | 97 (0.6326) |
| son epoch mAP50-95 | 0.6320 |
| epoch 50 / 80 / 90 / 100 | 0.6162 / 0.6299 / 0.6317 / 0.6320 |
| son 20 epoch eğimi (puan/10 epoch) | 0.143 |
| önceki 20 epoch (ep 71-90) eğimi | 0.272 |
| son 20 epoch std (puan) | 0.093 |
| ep 81-90 ort. vs ep 91-100 ort. (close_mosaic=10) | 0.6304 vs 0.6318 (Δ +0.14 puan) |
| mAP50 son epoch | 0.8197 |
| results.csv en iyi ep mAP50-95 vs metrics.json (best.pt yeniden val) | 0.6326 vs 0.6284 (fark 0.42 puan; ölçüm-protokolü gürültü tabanı) |

Loss tablosu (train loss mosaic açıkken ep1/ep90, kapalıyken ep100; val/train oranı):

| loss | train ep1 | train ep90 | train ep100 | val ep100 | val/train ep100 | val/train ep90 | train loss ep90→91 değişim |
|---|---|---|---|---|---|---|---|
| box | 0.965 | 0.766 | 0.695 | 0.869 | 1.25 | 1.14 | -1.7% |
| cls | 2.275 | 0.791 | 0.546 | 0.786 | 1.44 | 1.00 | -18.7% |
| dfl | 1.248 | 1.109 | 1.063 | 1.158 | 1.09 | 1.05 | -0.5% |

Kayıp payı (results.csv loss_items kazançlı; loss.py:469-471 — önceki sürüm kazançları iki kez uyguluyordu, 2026-10-01 düzeltildi) ep100: box 30.2%, cls 23.7%, dfl 46.1% (ep1: box 21.5%, cls 50.7%, dfl 27.8%). Son 20 epoch mAP eğimi +0.143 puan/10 ep (std 0.09) → doygun (eğim seed gürültüsü altında). close_mosaic sonrası (ep 91-100) ortalama +0.14 puan. Train loss mosaic kapanınca ani düşer (cls -18.7%); bu rejim değişimi, train/val farkının bir kısmının augmentation kaynaklı olduğunu gösterir. val cls loss son 20 epoch 0.798→0.786 (yükselmiyor): val loss bozulmadığı için klasik aşırı öğrenme yok, ama mosaic kapalıyken val/train oranı cls 1.44, box 1.25: train kaybı val kaybından ayrışıyor (genelleme açığı); val mAP doygun.

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
| ön-işleme ms | 1.2752 |
| çıkarım ms | 6.0724 |
| NMS/son-işlem ms | 0.6946 |
| toplam ms | 8.0422 |
| FPS | 124.344 |
| batch | 1 |
| FP16 (1=evet) | 1.0 |
| ölçüm görüntüsü | 300 |
| ısınma görüntüsü | 50 |
| eğitim süresi (saat) | 2.9368 |

Varyant hız bütçesi (yalnız tahmin; gerçek ölçüm varyant koşumunda yapılır):

| varyant sınıfı | çarpan | tahmini toplam ms | tahmini FPS | not |
|---|---|---|---|---|
| eğitim-zamanı (aug/loss) varyantları | aynı | 8.04 | 124 | çıkarım grafiği değişmez; hız bütçesi ≈ 0 |
| Soft-NMS/DIoU-NMS (son-işlem ×1–5 varsayımı) | NMS ×1–5 | 8.04–10.82 | 92–124 | tahmin (ölçülmedi) |
| TTA (Ultralytics augment, ~3 ölçek+flip; literatür çarpanı 3.6× çıkarım) | çıkarım ×3.6 (literatür) | 23.83 | 42 | tahmin; literatur-cipa.md §4 (Ultralytics TTA dokümanı) |
| YOLO11s'e geçiş (bağlam satırı) | ≈3.4× GFLOPs | – | – | ölçülmedi |

Baseline batch=1 FP16 toplam 8.04 ms (124 FPS; çıkarım 6.07 ms, NMS 0.69 ms = toplamın %9'i). 'Gerçek zamanlı' için ≥30 FPS kabul edilirse (33 ms) ~4.1× gecikme payı vardır; TTA bile (~42 FPS tahmini) eşiğin üstünde kalabilir, ancak hız-doğruluk satırı ayrı raporlanmalıdır. Not: hız ölçümü GPU'nun paylaşıldığı bir anda alınmış olabilir (eğitim kuyruğu); karşılaştırmalı hız ölçümü aynı koşulda yapılmalıdır.
