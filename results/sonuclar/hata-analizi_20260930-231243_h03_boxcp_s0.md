# Hata analizi (K3.5) — baseline `20260930-231243_h03_boxcp_s0`

*Üretici: `scripts/analiz/hata_analizi.py` · yalnız VAL (1000 görüntü, 2407 nesne; difficult nesneler etiketlerde yok) · tahmin sayısı 40949 (conf=0.001, max_det=300, NMS IoU=0.7) · eşleştirme: pycocotools COCOeval · işletim noktası skor≥0.25.*

**Sızıntı notu:** Bu analizde test kümesi (VOC07 test), `rep_test*` dosyaları ve `metrics.json` içindeki `rep_*` anahtarları hiç kullanılmadı; betik bu anahtarları yükler yüklemez atar. Buradan çıkan hipotezler yalnız val bilgisine dayanır.

## 0. Genel ölçüler ve doğrulama

| ölçü | pycocotools (bu betik) | Ultralytics metrics.json (best.pt val) | fark |
|---|---|---|---|
| mAP50-95 | 0.6319 | 0.6326 | -0.0007 |
| mAP50 | 0.8177 | 0.8203 | -0.0026 |
| mAP75 (yalnız pycocotools) | 0.6777 | – | – |

AP50 ile AP50-95 arasındaki fark 18.6 puan; AP50→AP75 farkı 14.0 puan. pycocotools ile Ultralytics AP'si arasındaki küçük fark interpolasyon/eşleştirme (Ultralytics kendi IoU-matrisi eşleşmesini kullanır) kaynaklıdır.

## 1. Sınıf başına AP ve sınıf frekansı

| sınıf (AP50-95 artan) | AP50-95 | AP50 | AP75 | AP50−AP75 | train nesne | val nesne | val GT medyan kenar px |
|---|---|---|---|---|---|---|---|
| pottedplant | 0.315 | 0.578 | 0.275 | 0.303 | 1378 | 109 | 71 |
| chair | 0.470 | 0.653 | 0.513 | 0.141 | 2975 | 177 | 99 |
| bottle | 0.529 | 0.700 | 0.600 | 0.099 | 1669 | 95 | 70 |
| boat | 0.551 | 0.794 | 0.620 | 0.175 | 1076 | 64 | 112 |
| diningtable | 0.582 | 0.735 | 0.631 | 0.104 | 775 | 49 | 211 |
| sofa | 0.587 | 0.729 | 0.642 | 0.087 | 755 | 59 | 237 |
| bird | 0.589 | 0.775 | 0.630 | 0.146 | 1514 | 91 | 115 |
| tvmonitor | 0.612 | 0.847 | 0.711 | 0.136 | 1036 | 72 | 108 |
| horse | 0.626 | 0.834 | 0.653 | 0.181 | 1006 | 66 | 209 |
| person | 0.632 | 0.878 | 0.675 | 0.203 | 12471 | 785 | 141 |
| motorbike | 0.640 | 0.826 | 0.656 | 0.170 | 985 | 67 | 239 |
| car | 0.648 | 0.839 | 0.704 | 0.135 | 3045 | 222 | 92 |
| cow | 0.679 | 0.855 | 0.738 | 0.116 | 791 | 56 | 145 |
| aeroplane | 0.692 | 0.896 | 0.718 | 0.177 | 1091 | 80 | 184 |
| bicycle | 0.699 | 0.910 | 0.724 | 0.186 | 1002 | 62 | 171 |
| sheep | 0.736 | 0.885 | 0.747 | 0.139 | 1013 | 57 | 119 |
| bus | 0.744 | 0.847 | 0.795 | 0.052 | 767 | 55 | 192 |
| cat | 0.750 | 0.912 | 0.802 | 0.110 | 1505 | 88 | 278 |
| train | 0.762 | 0.924 | 0.844 | 0.080 | 868 | 57 | 225 |
| dog | 0.795 | 0.937 | 0.877 | 0.060 | 1929 | 96 | 211 |

En zayıf 5 sınıf (AP50-95): pottedplant 0.315, chair 0.470, bottle 0.529, boat 0.551, diningtable 0.582. Sınıf frekansı (train nesne) ile AP50-95 Spearman ρ=-0.15 (permütasyon p=0.524), AP50 ile ρ=-0.02; sınıf başına train küçük-nesne oranı ile ρ=-0.41; val GT medyan kenar uzunluğu ile ρ=0.55 (n=20 sınıf, gürültülü). Frekans AP'nin zayıf belirleyicisi (|ρ|<0.3): zayıf sınıflarda sorun veri azlığından çok nesne boyutu/görünümü/bağlamıdır.

## 2. Nesne boyutu kırılımı (orijinal çözünürlük; COCO S<32²≤M<96²≤L)

| boyut | GT sayısı | AP50-95 | AP50 | <e-posta> (tüm conf, sınıf ort.) | <e-posta>:0.95 (sınıf ort.) | <e-posta> skor≥0.25 |
|---|---|---|---|---|---|---|
| all | 2407 | 0.632 | 0.818 | 0.958 | 0.761 | 0.796 |
| S | 143 | 0.231 | 0.404 | 0.754 | 0.473 | 0.378 |
| M | 704 | 0.480 | 0.693 | 0.900 | 0.649 | 0.705 |
| L | 1560 | 0.707 | 0.875 | 0.977 | 0.812 | 0.875 |

İnce alan aralıkları (GT düzeyi, eşleşme bayrağı):

| alan (px²) | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) |
|---|---|---|---|---|
| <16² | 9 | 44.4% | 0.0% | 11.1% |
| 16²–32² | 134 | 87.3% | 40.3% | 60.4% |
| 32²–64² | 377 | 96.0% | 69.2% | 74.8% |
| 64²–96² | 327 | 94.2% | 71.9% | 78.3% |
| 96²–192² | 698 | 97.6% | 82.1% | 82.5% |
| ≥192² | 862 | 99.0% | 91.9% | 90.6% |

Sınıf × boyut AP50-95 (n<5 ise '–'):

| sınıf | n_S | AP_S | n_M | AP_M | n_L | AP_L |
|---|---|---|---|---|---|---|
| aeroplane | 0 | – | 21 | 0.52 | 59 | 0.75 |
| bicycle | 0 | – | 12 | 0.40 | 50 | 0.78 |
| bird | 3 | – | 36 | 0.35 | 52 | 0.75 |
| boat | 4 | – | 25 | 0.50 | 35 | 0.65 |
| bottle | 13 | 0.19 | 52 | 0.49 | 30 | 0.74 |
| bus | 3 | – | 13 | 0.50 | 39 | 0.88 |
| car | 38 | 0.31 | 80 | 0.62 | 104 | 0.78 |
| cat | 0 | – | 2 | – | 86 | 0.77 |
| chair | 3 | – | 79 | 0.44 | 95 | 0.51 |
| cow | 0 | – | 25 | 0.71 | 31 | 0.72 |
| diningtable | 0 | – | 1 | – | 48 | 0.59 |
| dog | 0 | – | 10 | 0.70 | 86 | 0.81 |
| horse | 3 | – | 15 | 0.22 | 48 | 0.76 |
| motorbike | 3 | – | 10 | 0.47 | 54 | 0.70 |
| person | 51 | 0.21 | 215 | 0.49 | 519 | 0.72 |
| pottedplant | 12 | 0.26 | 57 | 0.35 | 40 | 0.34 |
| sheep | 8 | 0.33 | 16 | 0.63 | 33 | 0.87 |
| sofa | 0 | – | 1 | – | 58 | 0.59 |
| train | 0 | – | 4 | – | 53 | 0.78 |
| tvmonitor | 2 | – | 30 | 0.58 | 40 | 0.66 |

Küçük nesneler (n=143, val GT'nin %5.9'i): AP50-95 0.231 vs büyük 0.707 (fark 47.6 puan); <e-posta> 0.754 vs 0.977. Not: ölçüm orijinal piksel; eğitim/çıkarım 640 px letterbox olduğundan (~1.3× büyütme) S nesneler modelde ≈<41 px görünür. Val'de S örneği az olduğundan S-AP güven aralığı geniştir.

## 3. Hata türü ayrıştırması (TIDE benzeri)

Sayılar: tüm tahminler (conf≥0.001) ve işletim noktası (skor≥0.25). `Miss` tüm conf'ta hiçbir kutuyla IoU≥0.5 eşleşmeyen GT; işletim noktası sütununda ise o eşikte bulunamayan GT (eşleşmeyen + düşük skorla eşleşen).

| tür | sayı (tüm conf) | oran | sayı skor≥0.25 |
|---|---|---|---|
| TP | 2325 | 5.7% | 1915 |
| Dupe | 2366 | 5.8% | 87 |
| Cls | 3128 | 7.6% | 93 |
| Loc | 7765 | 19.0% | 157 |
| Both | 5088 | 12.4% | 63 |
| Bkg | 20277 | 49.5% | 274 |
| Miss | 82 | 3.4% (GT'nin) | 492 |

Oracle ΔAP (puan): ilgili hata türü düzeltilirse mAP ne kadar artardı (AP50 ve AP50-95, sınıf ortalaması). Taban: AP50=81.77, AP50-95=63.19.

| düzeltilen | ΔAP50 (puan) | ΔAP50-95 (puan) |
|---|---|---|
| Cls | 2.65 | 1.97 |
| Loc | 2.71 | 5.50 |
| Both | 1.20 | 0.80 |
| Dupe | 0.50 | -0.85 |
| Bkg | 3.38 | 2.14 |
| Miss | 3.50 | 2.62 |
| FP_hepsi | 13.47 | 5.90 |
| Kutu_kusursuz | 0.00 | 18.58 |
| TP_IoU_sirali | 1.41 | 5.35 |

Protokol etkisi: val'de etiketlerden çıkarılmış difficult nesne sayısı 493 (XML). Bunlardan biriyle aynı sınıf IoU≥0.5 örtüşen ve eşleşmemiş (FP sayılan) tahmin: 836 (skor≥0.25: 123; işletim noktası FP'lerinin 18.2%'i). Bu tahminler devkit gibi yok sayılsaydı ΔAP50 = +1.14, ΔAP50-95 = +0.73 puan (Ultralytics protokolünün val üzerinde ölçülmüş cezası; makalede belirtilmeli, Bkg/Loc/Dupe sayılarının bir kısmı bu yüzden yapaydır).


Not: `Dupe` ve `Loc` ΔAP50-95 değerleri IoU=0.5 tanımına göredir; bu kutular daha yüksek IoU eşiklerinde TP olabildiği için silme oracle'ı (Dupe) AP50-95'i düşürebilir; Loc oracle'ı kutuyu GT'ye yapıştırdığı için AP50-95 kazancı AP50'den büyüktür.


Hata türü × boyut (işletim noktası; tam uzun biçim `hata-kirilim.csv`):

| tür | S | M | L |
|---|---|---|---|
| TP (skor≥0.25, tahmin kutusu alanı) | 52 | 487 | 1376 |
| Dupe (skor≥0.25, tahmin kutusu alanı) | 3 | 26 | 58 |
| Cls (skor≥0.25, tahmin kutusu alanı) | 4 | 14 | 75 |
| Loc (skor≥0.25, tahmin kutusu alanı) | 5 | 68 | 84 |
| Both (skor≥0.25, tahmin kutusu alanı) | 0 | 12 | 51 |
| Bkg (skor≥0.25, tahmin kutusu alanı) | 49 | 131 | 94 |
| Miss (skor≥0.25 bulunamayan GT, GT alanı) | 89 | 208 | 195 |
| GT sayısı | 143 | 704 | 1560 |

AP50'ye en çok mal olan hata türü: Miss (+3.50), sonra Bkg (+3.38) ve Loc (+2.71). Oracle'lar birbirinden bağımsız ölçüldü, toplamları ΔAP'ye eşit değildir (etkileşim). `Kutu_kusursuz` = tüm TP kutuları GT'ye yapıştırılırsa AP50-95'in erişeceği üst sınır; `TP_IoU_sirali` = yalnız TP'lerin skorları IoU sırasına göre dağıtılırsa (FP skorları aynı) kazanç, yani IoU-farkında skorlamanın (VFL/QFL/IoU-aware) teorik payı.


Sınıf başına ΔAP50 (puan, oracle):

| sınıf | Cls | Loc | Both | Dupe | Bkg | Miss |
|---|---|---|---|---|---|---|
| aeroplane | 0.4 | 2.8 | 0.1 | 0.3 | 1.1 | 4.8 |
| bicycle | 0.4 | 1.8 | 0.5 | 0.2 | 0.6 | 3.0 |
| bird | 2.4 | 6.7 | 0.3 | 1.2 | 3.0 | 5.6 |
| boat | -0.1 | 4.3 | 0.2 | 0.6 | 4.7 | 5.1 |
| bottle | 0.4 | 2.7 | 0.3 | 0.3 | 10.1 | 5.5 |
| bus | 3.3 | 0.5 | 0.1 | 0.1 | 2.6 | 6.7 |
| car | 1.2 | 2.0 | 0.6 | 0.6 | 3.9 | 2.3 |
| cat | 2.5 | 2.1 | 0.1 | 0.2 | 0.2 | 3.1 |
| chair | 3.2 | 3.5 | 2.1 | 0.5 | 6.9 | 1.4 |
| cow | 7.1 | 1.8 | 0.0 | 0.9 | 3.1 | 1.6 |
| diningtable | 0.7 | 1.5 | 9.9 | 0.1 | 2.8 | 5.0 |
| dog | 2.3 | 1.1 | 0.3 | 0.2 | 0.4 | 1.1 |
| horse | 10.7 | 1.5 | 0.1 | 0.4 | 0.6 | 8.3 |
| motorbike | 3.6 | 3.4 | 0.7 | 0.3 | 2.3 | 3.8 |
| person | 0.3 | 4.2 | 0.2 | 0.9 | 2.2 | 1.5 |
| pottedplant | 0.6 | 8.4 | 3.6 | 1.1 | 7.1 | 7.2 |
| sheep | 6.1 | 0.1 | 0.1 | 0.2 | 4.1 | 0.0 |
| sofa | 7.4 | 2.7 | 3.5 | 1.5 | 3.1 | 2.6 |
| train | 0.3 | 2.4 | 0.3 | 0.2 | 1.5 | 0.0 |
| tvmonitor | 0.3 | 0.6 | 1.1 | 0.3 | 7.4 | 1.3 |

Kaçırılan GT (82) alt kırılımı: yalnız başka sınıfla örtüşen kutu var (sınıf karışıklığı) 18; aynı sınıf kutu var ama IoU∈[0.1,0.5) (lokalizasyon) 42; aynı sınıf IoU≥0.5 kutu var ama başka GT'ye atanmış (atama çakışması/örtüşme) 1; hiç kutu yok (gerçek kaçırma) 21 (kalan sınıflama kesin değil: bu üç grup kesişebilir).

## 4. Lokalizasyon kalitesi

| IoU aralığı | TP (tüm conf) | oran | TP skor≥0.25 | oran |
|---|---|---|---|---|
| 0.50–0.55 | 62 | 2.7% | 26 | 1.4% |
| 0.55–0.60 | 88 | 3.8% | 47 | 2.5% |
| 0.60–0.65 | 100 | 4.3% | 49 | 2.6% |
| 0.65–0.70 | 128 | 5.5% | 86 | 4.5% |
| 0.70–0.75 | 154 | 6.6% | 108 | 5.6% |
| 0.75–0.80 | 173 | 7.4% | 129 | 6.7% |
| 0.80–0.85 | 218 | 9.4% | 165 | 8.6% |
| 0.85–0.90 | 327 | 14.1% | 284 | 14.8% |
| 0.90–0.95 | 496 | 21.3% | 464 | 24.2% |
| 0.95–1.00 | 579 | 24.9% | 557 | 29.1% |

TP IoU ortalaması 0.846 (skor≥0.25: 0.869); işletim noktası TP'lerinin %16.5'i IoU<0.75. TP'lerde skor–IoU Spearman ρ=0.64 (skorun konum kalitesini ne kadar taşıdığı).

Sınıf başına AP50 − AP75 (büyükten küçüğe) ve TP ort. IoU:

| sınıf | AP50 | AP75 | AP50−AP75 (puan) | TP ort. IoU |
|---|---|---|---|---|
| pottedplant | 0.578 | 0.275 | 30.3 | 0.770 |
| person | 0.878 | 0.675 | 20.3 | 0.835 |
| bicycle | 0.910 | 0.724 | 18.6 | 0.861 |
| horse | 0.834 | 0.653 | 18.1 | 0.858 |
| aeroplane | 0.896 | 0.718 | 17.7 | 0.853 |
| boat | 0.794 | 0.620 | 17.5 | 0.816 |
| motorbike | 0.826 | 0.656 | 17.0 | 0.834 |
| bird | 0.775 | 0.630 | 14.6 | 0.845 |
| chair | 0.653 | 0.513 | 14.1 | 0.819 |
| sheep | 0.885 | 0.747 | 13.9 | 0.888 |
| tvmonitor | 0.847 | 0.711 | 13.6 | 0.848 |
| car | 0.839 | 0.704 | 13.5 | 0.849 |
| cow | 0.855 | 0.738 | 11.6 | 0.874 |
| cat | 0.912 | 0.802 | 11.0 | 0.898 |
| diningtable | 0.735 | 0.631 | 10.4 | 0.861 |
| bottle | 0.700 | 0.600 | 9.9 | 0.840 |
| sofa | 0.729 | 0.642 | 8.7 | 0.863 |
| train | 0.924 | 0.844 | 8.0 | 0.885 |
| dog | 0.937 | 0.877 | 6.0 | 0.904 |
| bus | 0.847 | 0.795 | 5.2 | 0.911 |

AP50→AP75 düşüşü sınıf ortalaması 14.0 puan; en büyük düşüş pottedplant (30), person (20), bicycle (19). Kutu-regresyon (IoU/DFL/box kazancı) manipülasyonlarının hedef alanı budur.

## 5. Sınıf karışıklığı

İşletim noktası (skor≥0.25, IoU≥0.5): her GT için eşleşen en yüksek skorlu kutu (doğru sınıf varsa o). Matris dosyası: `hata-analizi_20260930-231243_h03_boxcp_s0_karisiklik.csv`.

| GT sınıfı → tahmin | sayı | GT'nin payı |
|---|---|---|
| chair → person | 5 | 2.8% |
| sofa → person | 5 | 8.5% |
| horse → sheep | 4 | 6.1% |
| bird → dog | 3 | 3.3% |
| bus → car | 3 | 5.5% |
| cow → horse | 3 | 5.4% |
| motorbike → person | 3 | 4.5% |
| car → train | 2 | 0.9% |
| chair → diningtable | 2 | 1.1% |
| cow → sheep | 2 | 3.6% |
| dog → cat | 2 | 2.1% |
| sheep → cow | 2 | 3.5% |

Çift bazında (yönsüz) toplam: chair/person 5; person/sofa 5; horse/sheep 4; bus/car 4; cow/sheep 4; bird/dog 3; cow/horse 3; motorbike/person 3.


Tüm-conf alternatif görünüm: `Cls` türü tahminler (IoU≥0.5 başka sınıf GT, skor≥0.05), GT sınıfı → tahmin:

| GT → tahmin | tahmin sayısı |
|---|---|
| person → chair | 26 |
| sofa → chair | 21 |
| person → sofa | 20 |
| sheep → cow | 12 |
| cow → horse | 11 |
| horse → sheep | 9 |
| sofa → person | 8 |
| cat → dog | 8 |
| motorbike → bicycle | 8 |
| cow → sheep | 7 |

Kanonik çiftler (yönsüz; `Cls` tahmin sayısı skor≥0.05 | işletim noktası GT sayısı): cat/dog 15 | 3; cow/sheep 19 | 4; chair/diningtable 8 | 3; chair/sofa 26 | 3; horse/cow 17 | 3; bus/train 4 | 1; motorbike/bicycle 12 | 1.

İşletim noktasında GT'lerin 2.2%'i başka sınıf olarak etiketlendi (54/2407); 17.7%'i hiç bulunamadı. Sınıf karışıklığı oracle ΔAP50 = 2.65 puan (§3; Cls, altı hata türü içinde 4. sırada); işletim noktasında karışan GT oranı yalnız 2.2%, yani sorun sınıf ayrımından çok kutu/skor düzeyindedir.

## 6. Kalabalık ve örtüşme

| nesne/görüntü | görüntü | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) | FP/görüntü (skor≥0.25) |
|---|---|---|---|---|---|---|
| 1 | 440 | 440 | 99.3% | 92.0% | 93.0% | 0.38 |
| 2 | 252 | 504 | 97.4% | 85.5% | 87.3% | 0.45 |
| 3–4 | 195 | 654 | 96.0% | 76.5% | 80.9% | 1.01 |
| 5–8 | 88 | 511 | 95.3% | 71.4% | 75.1% | 1.34 |
| ≥9 | 25 | 298 | 94.6% | 71.8% | 72.1% | 3.12 |

Aynı sınıftan başka GT ile en yüksek IoU (örtüşme) gruplarına göre recall:

| komşu GT IoU | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 |
|---|---|---|---|
| <0.1 (komşu yok) | 1908 | 96.8% | 80.5% |
| 0.1–0.3 | 357 | 95.2% | 76.8% |
| 0.3–0.5 | 105 | 99.0% | 75.2% |
| 0.5–0.7 | 33 | 90.9% | 72.7% |
| ≥0.7 | 4 | 100.0% | 75.0% |

Aynı sınıf GT çiftleri (IoU≥0.3): 75 çift; ikisi de bulundu 71, yalnız biri 2, hiçbiri 2. IoU≥0.7 (NMS eşiği üstü, doğrudan NMS etkilenir): 2 çift, ikisi de bulundu 2, yalnız biri 0, hiçbiri 0. Kaçırılan GT'lerin 1/82'i aynı sınıf IoU≥0.5 bir kutuya sahip ama o kutu başka GT'ye atanmış (olası NMS/atama kaybı).

Kalabalıkla recall düşüşü: tek nesneli görüntüde 92.0% iken ≥9 nesnelide 71.8% (skor≥0.25); tüm-conf recall 99.3% → 94.6%, <e-posta> 93.0% → 72.1%. Çiftlerde tek-bulunma ve atama çakışması sayıları küçükse kayıp NMS'ten değil düşük skor/zayıf konumdan gelir.

## 7. Kalibrasyon (skor vs gerçek precision)

Tahmin 'doğru' = IoU≥0.5 ile eşleşen (<e-posta>); ayrıca IoU≥0.75 için precision. Not: YOLO sınıf skoru olasılık değil, TAL hizalama hedefli (IoU'ya bağlı) bir skordur; düşük skor bölgesindeki düşük precision 'aday havuzu' doğasından gelir.

| skor aralığı | tahmin | ort. skor | <e-posta> | <e-posta> | TP ort. IoU | skor−precision |
|---|---|---|---|---|---|---|
| 0.0–0.1 | 37343 | 0.008 | 0.008 | 0.007 | 0.726 | 0.000 |
| 0.1–0.2 | 797 | 0.143 | 0.108 | 0.082 | 0.764 | 0.035 |
| 0.2–0.3 | 370 | 0.243 | 0.181 | 0.141 | 0.758 | 0.062 |
| 0.3–0.4 | 260 | 0.350 | 0.327 | 0.208 | 0.767 | 0.023 |
| 0.4–0.5 | 205 | 0.447 | 0.405 | 0.254 | 0.780 | 0.042 |
| 0.5–0.6 | 208 | 0.551 | 0.548 | 0.361 | 0.767 | 0.003 |
| 0.6–0.7 | 255 | 0.652 | 0.675 | 0.478 | 0.809 | -0.023 |
| 0.7–0.8 | 280 | 0.755 | 0.854 | 0.657 | 0.833 | -0.099 |
| 0.8–0.9 | 659 | 0.855 | 0.962 | 0.865 | 0.883 | -0.107 |
| 0.9–1.0 | 572 | 0.927 | 0.983 | 0.960 | 0.942 | -0.055 |

ECE (tüm tahminler, 10 eşit genişlikli kutu) = 0.005; ECE (skor≥0.1) = 0.056. `skor−precision` pozitifse aşırı güvenli, negatifse eksik güvenli. 
Skor≥0.5 tahminlerin <e-posta> = 87.2%, <e-posta> = 76.0% (fark 11.2 puan). Skor 0.5 üstü kutularda (skor−precision) ortalaması -0.056: model yüksek skorlarda eksik güvenli (skor < gerçek <e-posta>), sıralama iyi; asıl boşluk <e-posta> ile @0.75 arasındaki konum kalitesidir. TP'lerde skor–IoU Spearman ρ=0.64: skor konum kalitesini kısmen taşıyor. TP_IoU_sirali oracle'ı (§3) IoU-farkında skorlamanın teorik payını sayısallaştırıyor.

## 8. Eğitim dinamiği (results.csv)

| gösterge | değer |
|---|---|
| epoch sayısı | 100 |
| en iyi epoch (mAP50-95) | 100 (0.6339) |
| son epoch mAP50-95 | 0.6339 |
| epoch 50 / 80 / 90 / 100 | 0.6177 / 0.6318 / 0.6331 / 0.6339 |
| son 20 epoch eğimi (puan/10 epoch) | 0.142 |
| önceki 20 epoch (ep 71-90) eğimi | 0.220 |
| son 20 epoch std (puan) | 0.090 |
| ep 81-90 ort. vs ep 91-100 ort. (close_mosaic=10) | 0.6320 vs 0.6333 (Δ +0.13 puan) |
| mAP50 son epoch | 0.8188 |
| results.csv en iyi ep mAP50-95 vs metrics.json (best.pt yeniden val) | 0.6339 vs 0.6326 (fark 0.13 puan; ölçüm-protokolü gürültü tabanı) |

Loss tablosu (train loss mosaic açıkken ep1/ep90, kapalıyken ep100; val/train oranı):

| loss | train ep1 | train ep90 | train ep100 | val ep100 | val/train ep100 | val/train ep90 | train loss ep90→91 değişim |
|---|---|---|---|---|---|---|---|
| box | 0.990 | 0.804 | 0.691 | 0.867 | 1.25 | 1.08 | -7.5% |
| cls | 2.266 | 0.816 | 0.541 | 0.783 | 1.45 | 0.96 | -22.3% |
| dfl | 1.270 | 1.138 | 1.060 | 1.154 | 1.09 | 1.01 | -3.4% |

Kayıp payı (results.csv loss_items kazançlı; loss.py:469-471 — önceki sürüm kazançları iki kez uyguluyordu, 2026-10-01 düzeltildi) ep100: box 30.1%, cls 23.6%, dfl 46.2% (ep1: box 21.9%, cls 50.1%, dfl 28.1%). Son 20 epoch mAP eğimi +0.142 puan/10 ep (std 0.09) → doygun (eğim seed gürültüsü altında). close_mosaic sonrası (ep 91-100) ortalama +0.13 puan. Train loss mosaic kapanınca ani düşer (cls -22.3%); bu rejim değişimi, train/val farkının bir kısmının augmentation kaynaklı olduğunu gösterir. val cls loss son 20 epoch 0.790→0.783 (yükselmiyor): val loss bozulmadığı için klasik aşırı öğrenme yok, ama mosaic kapalıyken val/train oranı cls 1.45, box 1.25: train kaybı val kaybından ayrışıyor (genelleme açığı); val mAP doygun.

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
| ön-işleme ms | 1.2619 |
| çıkarım ms | 6.069 |
| NMS/son-işlem ms | 0.68 |
| toplam ms | 8.0109 |
| FPS | 124.83 |
| batch | 1 |
| FP16 (1=evet) | 1.0 |
| ölçüm görüntüsü | 300 |
| ısınma görüntüsü | 50 |
| eğitim süresi (saat) | 2.9408 |

Varyant hız bütçesi (yalnız tahmin; gerçek ölçüm varyant koşumunda yapılır):

| varyant sınıfı | çarpan | tahmini toplam ms | tahmini FPS | not |
|---|---|---|---|---|
| eğitim-zamanı (aug/loss) varyantları | aynı | 8.01 | 125 | çıkarım grafiği değişmez; hız bütçesi ≈ 0 |
| Soft-NMS/DIoU-NMS (son-işlem ×1–5 varsayımı) | NMS ×1–5 | 8.01–10.73 | 93–125 | tahmin (ölçülmedi) |
| TTA (Ultralytics augment, ~3 ölçek+flip; literatür çarpanı 3.6× çıkarım) | çıkarım ×3.6 (literatür) | 23.79 | 42 | tahmin; literatur-cipa.md §4 (Ultralytics TTA dokümanı) |
| YOLO11s'e geçiş (bağlam satırı) | ≈3.4× GFLOPs | – | – | ölçülmedi |

Baseline batch=1 FP16 toplam 8.01 ms (125 FPS; çıkarım 6.07 ms, NMS 0.68 ms = toplamın %8'i). 'Gerçek zamanlı' için ≥30 FPS kabul edilirse (33 ms) ~4.2× gecikme payı vardır; TTA bile (~42 FPS tahmini) eşiğin üstünde kalabilir, ancak hız-doğruluk satırı ayrı raporlanmalıdır. Not: hız ölçümü GPU'nun paylaşıldığı bir anda alınmış olabilir (eğitim kuyruğu); karşılaştırmalı hız ölçümü aynı koşulda yapılmalıdır.
