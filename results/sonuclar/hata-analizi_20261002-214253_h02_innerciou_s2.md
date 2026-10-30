# Hata analizi (K3.5) — baseline `20261002-214253_h02_innerciou_s2`

*Üretici: `scripts/analiz/hata_analizi.py` · yalnız VAL (1000 görüntü, 2407 nesne; difficult nesneler etiketlerde yok) · tahmin sayısı 45527 (conf=0.001, max_det=300, NMS IoU=0.7) · eşleştirme: pycocotools COCOeval · işletim noktası skor≥0.25.*

**Sızıntı notu:** Bu analizde test kümesi (VOC07 test), `rep_test*` dosyaları ve `metrics.json` içindeki `rep_*` anahtarları hiç kullanılmadı; betik bu anahtarları yükler yüklemez atar. Buradan çıkan hipotezler yalnız val bilgisine dayanır.

## 0. Genel ölçüler ve doğrulama

| ölçü | pycocotools (bu betik) | Ultralytics metrics.json (best.pt val) | fark |
|---|---|---|---|
| mAP50-95 | 0.6236 | 0.6243 | -0.0007 |
| mAP50 | 0.8069 | 0.8093 | -0.0024 |
| mAP75 (yalnız pycocotools) | 0.6760 | – | – |

AP50 ile AP50-95 arasındaki fark 18.3 puan; AP50→AP75 farkı 13.1 puan. pycocotools ile Ultralytics AP'si arasındaki küçük fark interpolasyon/eşleştirme (Ultralytics kendi IoU-matrisi eşleşmesini kullanır) kaynaklıdır.

## 1. Sınıf başına AP ve sınıf frekansı

| sınıf (AP50-95 artan) | AP50-95 | AP50 | AP75 | AP50−AP75 | train nesne | val nesne | val GT medyan kenar px |
|---|---|---|---|---|---|---|---|
| pottedplant | 0.291 | 0.541 | 0.270 | 0.271 | 1378 | 109 | 71 |
| chair | 0.483 | 0.661 | 0.553 | 0.108 | 2975 | 177 | 99 |
| boat | 0.519 | 0.720 | 0.623 | 0.097 | 1076 | 64 | 112 |
| bottle | 0.522 | 0.699 | 0.596 | 0.103 | 1669 | 95 | 70 |
| diningtable | 0.565 | 0.706 | 0.590 | 0.116 | 775 | 49 | 211 |
| sofa | 0.577 | 0.728 | 0.629 | 0.099 | 755 | 59 | 237 |
| bird | 0.578 | 0.764 | 0.631 | 0.133 | 1514 | 91 | 115 |
| tvmonitor | 0.600 | 0.836 | 0.703 | 0.133 | 1036 | 72 | 108 |
| motorbike | 0.623 | 0.809 | 0.644 | 0.166 | 985 | 67 | 239 |
| horse | 0.628 | 0.828 | 0.656 | 0.173 | 1006 | 66 | 209 |
| person | 0.630 | 0.888 | 0.671 | 0.217 | 12471 | 785 | 141 |
| car | 0.647 | 0.834 | 0.701 | 0.133 | 3045 | 222 | 92 |
| cow | 0.671 | 0.850 | 0.742 | 0.108 | 791 | 56 | 145 |
| aeroplane | 0.686 | 0.874 | 0.701 | 0.174 | 1091 | 80 | 184 |
| bicycle | 0.695 | 0.903 | 0.762 | 0.141 | 1002 | 62 | 171 |
| sheep | 0.722 | 0.872 | 0.783 | 0.089 | 1013 | 57 | 119 |
| bus | 0.731 | 0.847 | 0.748 | 0.100 | 767 | 55 | 192 |
| train | 0.750 | 0.907 | 0.818 | 0.089 | 868 | 57 | 225 |
| dog | 0.774 | 0.938 | 0.845 | 0.093 | 1929 | 96 | 211 |
| cat | 0.780 | 0.930 | 0.854 | 0.076 | 1505 | 88 | 278 |

En zayıf 5 sınıf (AP50-95): pottedplant 0.291, chair 0.483, boat 0.519, bottle 0.522, diningtable 0.565. Sınıf frekansı (train nesne) ile AP50-95 Spearman ρ=-0.11 (permütasyon p=0.639), AP50 ile ρ=0.01; sınıf başına train küçük-nesne oranı ile ρ=-0.39; val GT medyan kenar uzunluğu ile ρ=0.53 (n=20 sınıf, gürültülü). Frekans AP'nin zayıf belirleyicisi (|ρ|<0.3): zayıf sınıflarda sorun veri azlığından çok nesne boyutu/görünümü/bağlamıdır.

## 2. Nesne boyutu kırılımı (orijinal çözünürlük; COCO S<32²≤M<96²≤L)

| boyut | GT sayısı | AP50-95 | AP50 | <e-posta> (tüm conf, sınıf ort.) | <e-posta>:0.95 (sınıf ort.) | <e-posta> skor≥0.25 |
|---|---|---|---|---|---|---|
| all | 2407 | 0.624 | 0.807 | 0.960 | 0.764 | 0.794 |
| S | 143 | 0.222 | 0.389 | 0.741 | 0.446 | 0.427 |
| M | 704 | 0.439 | 0.639 | 0.929 | 0.675 | 0.702 |
| L | 1560 | 0.698 | 0.865 | 0.980 | 0.812 | 0.870 |

İnce alan aralıkları (GT düzeyi, eşleşme bayrağı):

| alan (px²) | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) |
|---|---|---|---|---|
| <16² | 9 | 55.6% | 0.0% | 22.2% |
| 16²–32² | 134 | 83.6% | 45.5% | 53.7% |
| 32²–64² | 377 | 96.0% | 68.2% | 74.8% |
| 64²–96² | 327 | 94.8% | 72.5% | 79.5% |
| 96²–192² | 698 | 97.7% | 81.2% | 84.8% |
| ≥192² | 862 | 99.3% | 91.6% | 91.0% |

Sınıf × boyut AP50-95 (n<5 ise '–'):

| sınıf | n_S | AP_S | n_M | AP_M | n_L | AP_L |
|---|---|---|---|---|---|---|
| aeroplane | 0 | – | 21 | 0.51 | 59 | 0.75 |
| bicycle | 0 | – | 12 | 0.41 | 50 | 0.76 |
| bird | 3 | – | 36 | 0.35 | 52 | 0.75 |
| boat | 4 | – | 25 | 0.52 | 35 | 0.59 |
| bottle | 13 | 0.22 | 52 | 0.48 | 30 | 0.71 |
| bus | 3 | – | 13 | 0.42 | 39 | 0.89 |
| car | 38 | 0.35 | 80 | 0.63 | 104 | 0.76 |
| cat | 0 | – | 2 | – | 86 | 0.80 |
| chair | 3 | – | 79 | 0.47 | 95 | 0.52 |
| cow | 0 | – | 25 | 0.68 | 31 | 0.71 |
| diningtable | 0 | – | 1 | – | 48 | 0.57 |
| dog | 0 | – | 10 | 0.68 | 86 | 0.79 |
| horse | 3 | – | 15 | 0.22 | 48 | 0.77 |
| motorbike | 3 | – | 10 | 0.46 | 54 | 0.68 |
| person | 51 | 0.22 | 215 | 0.51 | 519 | 0.72 |
| pottedplant | 12 | 0.17 | 57 | 0.30 | 40 | 0.34 |
| sheep | 8 | 0.46 | 16 | 0.67 | 33 | 0.83 |
| sofa | 0 | – | 1 | – | 58 | 0.59 |
| train | 0 | – | 4 | – | 53 | 0.77 |
| tvmonitor | 2 | – | 30 | 0.54 | 40 | 0.66 |

Küçük nesneler (n=143, val GT'nin %5.9'i): AP50-95 0.222 vs büyük 0.698 (fark 47.6 puan); <e-posta> 0.741 vs 0.980. Not: ölçüm orijinal piksel; eğitim/çıkarım 640 px letterbox olduğundan (~1.3× büyütme) S nesneler modelde ≈<41 px görünür. Val'de S örneği az olduğundan S-AP güven aralığı geniştir.

## 3. Hata türü ayrıştırması (TIDE benzeri)

Sayılar: tüm tahminler (conf≥0.001) ve işletim noktası (skor≥0.25). `Miss` tüm conf'ta hiçbir kutuyla IoU≥0.5 eşleşmeyen GT; işletim noktası sütununda ise o eşikte bulunamayan GT (eşleşmeyen + düşük skorla eşleşen).

| tür | sayı (tüm conf) | oran | sayı skor≥0.25 |
|---|---|---|---|
| TP | 2327 | 5.1% | 1912 |
| Dupe | 2536 | 5.6% | 100 |
| Cls | 3588 | 7.9% | 101 |
| Loc | 8332 | 18.3% | 162 |
| Both | 5816 | 12.8% | 73 |
| Bkg | 22928 | 50.4% | 269 |
| Miss | 80 | 3.3% (GT'nin) | 495 |

Oracle ΔAP (puan): ilgili hata türü düzeltilirse mAP ne kadar artardı (AP50 ve AP50-95, sınıf ortalaması). Taban: AP50=80.69, AP50-95=62.36.

| düzeltilen | ΔAP50 (puan) | ΔAP50-95 (puan) |
|---|---|---|
| Cls | 2.99 | 2.20 |
| Loc | 2.77 | 5.45 |
| Both | 1.18 | 0.82 |
| Dupe | 0.56 | -0.70 |
| Bkg | 3.72 | 2.36 |
| Miss | 3.21 | 2.31 |
| FP_hepsi | 14.96 | 6.93 |
| Kutu_kusursuz | 0.00 | 18.33 |
| TP_IoU_sirali | 1.42 | 5.61 |

Protokol etkisi: val'de etiketlerden çıkarılmış difficult nesne sayısı 493 (XML). Bunlardan biriyle aynı sınıf IoU≥0.5 örtüşen ve eşleşmemiş (FP sayılan) tahmin: 896 (skor≥0.25: 124; işletim noktası FP'lerinin 17.6%'i). Bu tahminler devkit gibi yok sayılsaydı ΔAP50 = +1.44, ΔAP50-95 = +1.00 puan (Ultralytics protokolünün val üzerinde ölçülmüş cezası; makalede belirtilmeli, Bkg/Loc/Dupe sayılarının bir kısmı bu yüzden yapaydır).


Not: `Dupe` ve `Loc` ΔAP50-95 değerleri IoU=0.5 tanımına göredir; bu kutular daha yüksek IoU eşiklerinde TP olabildiği için silme oracle'ı (Dupe) AP50-95'i düşürebilir; Loc oracle'ı kutuyu GT'ye yapıştırdığı için AP50-95 kazancı AP50'den büyüktür.


Hata türü × boyut (işletim noktası; tam uzun biçim `hata-kirilim.csv`):

| tür | S | M | L |
|---|---|---|---|
| TP (skor≥0.25, tahmin kutusu alanı) | 57 | 490 | 1365 |
| Dupe (skor≥0.25, tahmin kutusu alanı) | 0 | 29 | 71 |
| Cls (skor≥0.25, tahmin kutusu alanı) | 3 | 17 | 81 |
| Loc (skor≥0.25, tahmin kutusu alanı) | 9 | 63 | 90 |
| Both (skor≥0.25, tahmin kutusu alanı) | 0 | 15 | 58 |
| Bkg (skor≥0.25, tahmin kutusu alanı) | 48 | 129 | 92 |
| Miss (skor≥0.25 bulunamayan GT, GT alanı) | 82 | 210 | 203 |
| GT sayısı | 143 | 704 | 1560 |

AP50'ye en çok mal olan hata türü: Bkg (+3.72), sonra Miss (+3.21) ve Cls (+2.99). Oracle'lar birbirinden bağımsız ölçüldü, toplamları ΔAP'ye eşit değildir (etkileşim). `Kutu_kusursuz` = tüm TP kutuları GT'ye yapıştırılırsa AP50-95'in erişeceği üst sınır; `TP_IoU_sirali` = yalnız TP'lerin skorları IoU sırasına göre dağıtılırsa (FP skorları aynı) kazanç, yani IoU-farkında skorlamanın (VFL/QFL/IoU-aware) teorik payı.


Sınıf başına ΔAP50 (puan, oracle):

| sınıf | Cls | Loc | Both | Dupe | Bkg | Miss |
|---|---|---|---|---|---|---|
| aeroplane | 0.4 | 2.9 | 1.2 | 0.3 | 1.3 | 2.3 |
| bicycle | 0.6 | 0.7 | 0.6 | 0.2 | 0.5 | 3.1 |
| bird | 3.4 | 5.7 | 0.2 | 1.0 | 3.1 | 5.6 |
| boat | 3.9 | 4.6 | 0.2 | 1.0 | 6.9 | 6.0 |
| bottle | 1.4 | 2.6 | 0.2 | 0.4 | 11.1 | 4.8 |
| bus | 2.3 | 0.6 | 0.1 | 0.4 | 2.3 | 4.9 |
| car | 1.3 | 2.1 | 0.5 | 0.9 | 3.8 | 2.9 |
| cat | 1.7 | 2.8 | 0.2 | 0.2 | 0.2 | 1.2 |
| chair | 3.8 | 3.5 | 2.4 | 0.5 | 5.8 | 1.9 |
| cow | 6.2 | 2.0 | 0.0 | 0.5 | 3.8 | 1.6 |
| diningtable | 0.9 | 1.3 | 10.1 | 0.8 | 3.1 | 4.8 |
| dog | 2.3 | 0.4 | 0.3 | 0.2 | 0.7 | 0.0 |
| horse | 10.1 | 1.1 | 0.3 | 0.6 | 0.9 | 5.5 |
| motorbike | 2.7 | 4.5 | 1.0 | 0.3 | 2.0 | 3.7 |
| person | 0.1 | 3.0 | 0.2 | 1.2 | 2.0 | 1.5 |
| pottedplant | 1.0 | 9.4 | 1.1 | 1.2 | 7.7 | 8.6 |
| sheep | 9.2 | 0.2 | 0.3 | 0.3 | 3.7 | 0.0 |
| sofa | 7.5 | 2.6 | 3.5 | 0.6 | 5.8 | 1.4 |
| train | 0.2 | 4.0 | 0.3 | 0.2 | 1.8 | 1.8 |
| tvmonitor | 0.8 | 1.4 | 0.9 | 0.2 | 7.6 | 2.5 |

Kaçırılan GT (80) alt kırılımı: yalnız başka sınıfla örtüşen kutu var (sınıf karışıklığı) 22; aynı sınıf kutu var ama IoU∈[0.1,0.5) (lokalizasyon) 34; aynı sınıf IoU≥0.5 kutu var ama başka GT'ye atanmış (atama çakışması/örtüşme) 0; hiç kutu yok (gerçek kaçırma) 24 (kalan sınıflama kesin değil: bu üç grup kesişebilir).

## 4. Lokalizasyon kalitesi

| IoU aralığı | TP (tüm conf) | oran | TP skor≥0.25 | oran |
|---|---|---|---|---|
| 0.50–0.55 | 77 | 3.3% | 36 | 1.9% |
| 0.55–0.60 | 85 | 3.7% | 42 | 2.2% |
| 0.60–0.65 | 109 | 4.7% | 52 | 2.7% |
| 0.65–0.70 | 116 | 5.0% | 71 | 3.7% |
| 0.70–0.75 | 137 | 5.9% | 100 | 5.2% |
| 0.75–0.80 | 155 | 6.7% | 119 | 6.2% |
| 0.80–0.85 | 248 | 10.7% | 193 | 10.1% |
| 0.85–0.90 | 341 | 14.7% | 290 | 15.2% |
| 0.90–0.95 | 480 | 20.6% | 452 | 23.6% |
| 0.95–1.00 | 579 | 24.9% | 557 | 29.1% |

TP IoU ortalaması 0.845 (skor≥0.25: 0.869); işletim noktası TP'lerinin %15.7'i IoU<0.75. TP'lerde skor–IoU Spearman ρ=0.62 (skorun konum kalitesini ne kadar taşıdığı).

Sınıf başına AP50 − AP75 (büyükten küçüğe) ve TP ort. IoU:

| sınıf | AP50 | AP75 | AP50−AP75 (puan) | TP ort. IoU |
|---|---|---|---|---|
| pottedplant | 0.541 | 0.270 | 27.1 | 0.758 |
| person | 0.888 | 0.671 | 21.7 | 0.830 |
| aeroplane | 0.874 | 0.701 | 17.4 | 0.849 |
| horse | 0.828 | 0.656 | 17.3 | 0.860 |
| motorbike | 0.809 | 0.644 | 16.6 | 0.854 |
| bicycle | 0.903 | 0.762 | 14.1 | 0.857 |
| bird | 0.764 | 0.631 | 13.3 | 0.837 |
| tvmonitor | 0.836 | 0.703 | 13.3 | 0.845 |
| car | 0.834 | 0.701 | 13.3 | 0.849 |
| diningtable | 0.706 | 0.590 | 11.6 | 0.860 |
| chair | 0.661 | 0.553 | 10.8 | 0.824 |
| cow | 0.850 | 0.742 | 10.8 | 0.864 |
| bottle | 0.699 | 0.596 | 10.3 | 0.833 |
| bus | 0.847 | 0.748 | 10.0 | 0.897 |
| sofa | 0.728 | 0.629 | 9.9 | 0.888 |
| boat | 0.720 | 0.623 | 9.7 | 0.832 |
| dog | 0.938 | 0.845 | 9.3 | 0.900 |
| sheep | 0.872 | 0.783 | 8.9 | 0.894 |
| train | 0.907 | 0.818 | 8.9 | 0.882 |
| cat | 0.930 | 0.854 | 7.6 | 0.898 |

AP50→AP75 düşüşü sınıf ortalaması 13.1 puan; en büyük düşüş pottedplant (27), person (22), aeroplane (17). Kutu-regresyon (IoU/DFL/box kazancı) manipülasyonlarının hedef alanı budur.

## 5. Sınıf karışıklığı

İşletim noktası (skor≥0.25, IoU≥0.5): her GT için eşleşen en yüksek skorlu kutu (doğru sınıf varsa o). Matris dosyası: `hata-analizi_20261002-214253_h02_innerciou_s2_karisiklik.csv`.

| GT sınıfı → tahmin | sayı | GT'nin payı |
|---|---|---|
| chair → person | 8 | 4.5% |
| horse → sheep | 6 | 9.1% |
| sofa → chair | 4 | 6.8% |
| sofa → person | 4 | 6.8% |
| chair → diningtable | 3 | 1.7% |
| cow → horse | 3 | 5.4% |
| motorbike → person | 3 | 4.5% |
| sheep → dog | 3 | 5.3% |
| bus → car | 2 | 3.6% |
| dog → cat | 2 | 2.1% |
| sheep → cow | 2 | 3.5% |
| boat → aeroplane | 1 | 1.6% |

Çift bazında (yönsüz) toplam: chair/person 8; horse/sheep 6; chair/sofa 4; person/sofa 4; chair/diningtable 4; cow/horse 3; motorbike/person 3; dog/sheep 3.


Tüm-conf alternatif görünüm: `Cls` türü tahminler (IoU≥0.5 başka sınıf GT, skor≥0.05), GT sınıfı → tahmin:

| GT → tahmin | tahmin sayısı |
|---|---|
| person → chair | 29 |
| person → sofa | 23 |
| sofa → chair | 22 |
| cow → horse | 15 |
| dog → cat | 13 |
| chair → diningtable | 10 |
| chair → sofa | 9 |
| cat → dog | 8 |
| car → bus | 8 |
| sheep → dog | 8 |

Kanonik çiftler (yönsüz; `Cls` tahmin sayısı skor≥0.05 | işletim noktası GT sayısı): cat/dog 21 | 2; cow/sheep 14 | 3; chair/diningtable 13 | 4; chair/sofa 31 | 4; horse/cow 21 | 3; bus/train 5 | 0; motorbike/bicycle 10 | 1.

İşletim noktasında GT'lerin 2.4%'i başka sınıf olarak etiketlendi (58/2407); 17.7%'i hiç bulunamadı. Sınıf karışıklığı oracle ΔAP50 = 2.99 puan (§3; Cls, altı hata türü içinde 3. sırada); işletim noktasında karışan GT oranı yalnız 2.4%, yani sorun sınıf ayrımından çok kutu/skor düzeyindedir.

## 6. Kalabalık ve örtüşme

| nesne/görüntü | görüntü | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) | FP/görüntü (skor≥0.25) |
|---|---|---|---|---|---|---|
| 1 | 440 | 440 | 98.6% | 92.5% | 91.8% | 0.38 |
| 2 | 252 | 504 | 97.8% | 82.9% | 87.9% | 0.56 |
| 3–4 | 195 | 654 | 96.3% | 76.9% | 82.7% | 0.96 |
| 5–8 | 88 | 511 | 95.1% | 70.5% | 76.7% | 1.34 |
| ≥9 | 25 | 298 | 95.3% | 75.2% | 71.1% | 3.64 |

Aynı sınıftan başka GT ile en yüksek IoU (örtüşme) gruplarına göre recall:

| komşu GT IoU | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 |
|---|---|---|---|
| <0.1 (komşu yok) | 1908 | 96.6% | 80.5% |
| 0.1–0.3 | 357 | 96.4% | 77.0% |
| 0.3–0.5 | 105 | 98.1% | 72.4% |
| 0.5–0.7 | 33 | 100.0% | 69.7% |
| ≥0.7 | 4 | 100.0% | 75.0% |

Aynı sınıf GT çiftleri (IoU≥0.3): 75 çift; ikisi de bulundu 73, yalnız biri 2, hiçbiri 0. IoU≥0.7 (NMS eşiği üstü, doğrudan NMS etkilenir): 2 çift, ikisi de bulundu 2, yalnız biri 0, hiçbiri 0. Kaçırılan GT'lerin 0/80'i aynı sınıf IoU≥0.5 bir kutuya sahip ama o kutu başka GT'ye atanmış (olası NMS/atama kaybı).

Kalabalıkla recall düşüşü: tek nesneli görüntüde 92.5% iken ≥9 nesnelide 75.2% (skor≥0.25); tüm-conf recall 98.6% → 95.3%, <e-posta> 91.8% → 71.1%. Çiftlerde tek-bulunma ve atama çakışması sayıları küçükse kayıp NMS'ten değil düşük skor/zayıf konumdan gelir.

## 7. Kalibrasyon (skor vs gerçek precision)

Tahmin 'doğru' = IoU≥0.5 ile eşleşen (<e-posta>); ayrıca IoU≥0.75 için precision. Not: YOLO sınıf skoru olasılık değil, TAL hizalama hedefli (IoU'ya bağlı) bir skordur; düşük skor bölgesindeki düşük precision 'aday havuzu' doğasından gelir.

| skor aralığı | tahmin | ort. skor | <e-posta> | <e-posta> | TP ort. IoU | skor−precision |
|---|---|---|---|---|---|---|
| 0.0–0.1 | 41823 | 0.008 | 0.007 | 0.006 | 0.722 | 0.001 |
| 0.1–0.2 | 853 | 0.143 | 0.109 | 0.086 | 0.750 | 0.034 |
| 0.2–0.3 | 414 | 0.245 | 0.222 | 0.147 | 0.766 | 0.023 |
| 0.3–0.4 | 274 | 0.349 | 0.299 | 0.208 | 0.780 | 0.049 |
| 0.4–0.5 | 206 | 0.451 | 0.388 | 0.262 | 0.791 | 0.062 |
| 0.5–0.6 | 213 | 0.550 | 0.559 | 0.394 | 0.804 | -0.009 |
| 0.6–0.7 | 232 | 0.650 | 0.698 | 0.478 | 0.795 | -0.048 |
| 0.7–0.8 | 330 | 0.753 | 0.848 | 0.664 | 0.831 | -0.096 |
| 0.8–0.9 | 616 | 0.857 | 0.948 | 0.865 | 0.887 | -0.091 |
| 0.9–1.0 | 566 | 0.927 | 0.982 | 0.961 | 0.939 | -0.056 |

ECE (tüm tahminler, 10 eşit genişlikli kutu) = 0.005; ECE (skor≥0.1) = 0.053. `skor−precision` pozitifse aşırı güvenli, negatifse eksik güvenli. 
Skor≥0.5 tahminlerin <e-posta> = 86.9%, <e-posta> = 76.2% (fark 10.7 puan). Skor 0.5 üstü kutularda (skor−precision) ortalaması -0.060: model yüksek skorlarda eksik güvenli (skor < gerçek <e-posta>), sıralama iyi; asıl boşluk <e-posta> ile @0.75 arasındaki konum kalitesidir. TP'lerde skor–IoU Spearman ρ=0.62: skor konum kalitesini kısmen taşıyor. TP_IoU_sirali oracle'ı (§3) IoU-farkında skorlamanın teorik payını sayısallaştırıyor.

## 8. Eğitim dinamiği (results.csv)

| gösterge | değer |
|---|---|
| epoch sayısı | 100 |
| en iyi epoch (mAP50-95) | 89 (0.6260) |
| son epoch mAP50-95 | 0.6245 |
| epoch 50 / 80 / 90 / 100 | 0.6091 / 0.6244 / 0.6258 / 0.6245 |
| son 20 epoch eğimi (puan/10 epoch) | -0.030 |
| önceki 20 epoch (ep 71-90) eğimi | 0.249 |
| son 20 epoch std (puan) | 0.050 |
| ep 81-90 ort. vs ep 91-100 ort. (close_mosaic=10) | 0.6254 vs 0.6250 (Δ -0.04 puan) |
| mAP50 son epoch | 0.8088 |
| results.csv en iyi ep mAP50-95 vs metrics.json (best.pt yeniden val) | 0.6260 vs 0.6243 (fark 0.17 puan; ölçüm-protokolü gürültü tabanı) |

Loss tablosu (train loss mosaic açıkken ep1/ep90, kapalıyken ep100; val/train oranı):

| loss | train ep1 | train ep90 | train ep100 | val ep100 | val/train ep100 | val/train ep90 | train loss ep90→91 değişim |
|---|---|---|---|---|---|---|---|
| box | 1.184 | 0.949 | 0.806 | 1.016 | 1.26 | 1.07 | -8.4% |
| cls | 2.325 | 0.816 | 0.549 | 0.792 | 1.44 | 0.98 | -21.8% |
| dfl | 1.292 | 1.153 | 1.067 | 1.169 | 1.10 | 1.01 | -4.2% |

Kayıp payı (results.csv loss_items kazançlı; loss.py:469-471 — önceki sürüm kazançları iki kez uyguluyordu, 2026-10-01 düzeltildi) ep100: box 33.3%, cls 22.7%, dfl 44.1% (ep1: box 24.7%, cls 48.4%, dfl 26.9%). Son 20 epoch mAP eğimi -0.030 puan/10 ep (std 0.05) → doygun (eğim seed gürültüsü altında). close_mosaic sonrası (ep 91-100) ortalama -0.04 puan. Train loss mosaic kapanınca ani düşer (cls -21.8%); bu rejim değişimi, train/val farkının bir kısmının augmentation kaynaklı olduğunu gösterir. val cls loss son 20 epoch 0.802→0.792 (yükselmiyor): val loss bozulmadığı için klasik aşırı öğrenme yok, ama mosaic kapalıyken val/train oranı cls 1.44, box 1.26: train kaybı val kaybından ayrışıyor (genelleme açığı); val mAP doygun.

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
| ön-işleme ms | 1.2678 |
| çıkarım ms | 6.0281 |
| NMS/son-işlem ms | 0.6817 |
| toplam ms | 7.9775 |
| FPS | 125.352 |
| batch | 1 |
| FP16 (1=evet) | 1.0 |
| ölçüm görüntüsü | 300 |
| ısınma görüntüsü | 50 |
| eğitim süresi (saat) | 2.9546 |

Varyant hız bütçesi (yalnız tahmin; gerçek ölçüm varyant koşumunda yapılır):

| varyant sınıfı | çarpan | tahmini toplam ms | tahmini FPS | not |
|---|---|---|---|---|
| eğitim-zamanı (aug/loss) varyantları | aynı | 7.98 | 125 | çıkarım grafiği değişmez; hız bütçesi ≈ 0 |
| Soft-NMS/DIoU-NMS (son-işlem ×1–5 varsayımı) | NMS ×1–5 | 7.98–10.70 | 93–125 | tahmin (ölçülmedi) |
| TTA (Ultralytics augment, ~3 ölçek+flip; literatür çarpanı 3.6× çıkarım) | çıkarım ×3.6 (literatür) | 23.65 | 42 | tahmin; literatur-cipa.md §4 (Ultralytics TTA dokümanı) |
| YOLO11s'e geçiş (bağlam satırı) | ≈3.4× GFLOPs | – | – | ölçülmedi |

Baseline batch=1 FP16 toplam 7.98 ms (125 FPS; çıkarım 6.03 ms, NMS 0.68 ms = toplamın %9'i). 'Gerçek zamanlı' için ≥30 FPS kabul edilirse (33 ms) ~4.2× gecikme payı vardır; TTA bile (~42 FPS tahmini) eşiğin üstünde kalabilir, ancak hız-doğruluk satırı ayrı raporlanmalıdır. Not: hız ölçümü GPU'nun paylaşıldığı bir anda alınmış olabilir (eğitim kuyruğu); karşılaştırmalı hız ölçümü aynı koşulda yapılmalıdır.
