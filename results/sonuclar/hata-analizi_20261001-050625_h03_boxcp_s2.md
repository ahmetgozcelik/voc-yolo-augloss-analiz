# Hata analizi (K3.5) — baseline `20261001-050625_h03_boxcp_s2`

*Üretici: `scripts/analiz/hata_analizi.py` · yalnız VAL (1000 görüntü, 2407 nesne; difficult nesneler etiketlerde yok) · tahmin sayısı 43457 (conf=0.001, max_det=300, NMS IoU=0.7) · eşleştirme: pycocotools COCOeval · işletim noktası skor≥0.25.*

**Sızıntı notu:** Bu analizde test kümesi (VOC07 test), `rep_test*` dosyaları ve `metrics.json` içindeki `rep_*` anahtarları hiç kullanılmadı; betik bu anahtarları yükler yüklemez atar. Buradan çıkan hipotezler yalnız val bilgisine dayanır.

## 0. Genel ölçüler ve doğrulama

| ölçü | pycocotools (bu betik) | Ultralytics metrics.json (best.pt val) | fark |
|---|---|---|---|
| mAP50-95 | 0.6309 | 0.6319 | -0.0009 |
| mAP50 | 0.8162 | 0.8187 | -0.0026 |
| mAP75 (yalnız pycocotools) | 0.6802 | – | – |

AP50 ile AP50-95 arasındaki fark 18.5 puan; AP50→AP75 farkı 13.6 puan. pycocotools ile Ultralytics AP'si arasındaki küçük fark interpolasyon/eşleştirme (Ultralytics kendi IoU-matrisi eşleşmesini kullanır) kaynaklıdır.

## 1. Sınıf başına AP ve sınıf frekansı

| sınıf (AP50-95 artan) | AP50-95 | AP50 | AP75 | AP50−AP75 | train nesne | val nesne | val GT medyan kenar px |
|---|---|---|---|---|---|---|---|
| pottedplant | 0.277 | 0.539 | 0.220 | 0.319 | 1378 | 109 | 71 |
| chair | 0.473 | 0.665 | 0.526 | 0.139 | 2975 | 177 | 99 |
| bottle | 0.535 | 0.708 | 0.600 | 0.107 | 1669 | 95 | 70 |
| boat | 0.538 | 0.762 | 0.602 | 0.161 | 1076 | 64 | 112 |
| diningtable | 0.566 | 0.706 | 0.623 | 0.083 | 775 | 49 | 211 |
| bird | 0.596 | 0.795 | 0.641 | 0.154 | 1514 | 91 | 115 |
| sofa | 0.607 | 0.746 | 0.617 | 0.128 | 755 | 59 | 237 |
| tvmonitor | 0.636 | 0.861 | 0.758 | 0.103 | 1036 | 72 | 108 |
| horse | 0.637 | 0.834 | 0.710 | 0.124 | 1006 | 66 | 209 |
| person | 0.637 | 0.893 | 0.683 | 0.210 | 12471 | 785 | 141 |
| motorbike | 0.647 | 0.825 | 0.693 | 0.132 | 985 | 67 | 239 |
| car | 0.657 | 0.839 | 0.708 | 0.131 | 3045 | 222 | 92 |
| cow | 0.688 | 0.870 | 0.752 | 0.118 | 791 | 56 | 145 |
| aeroplane | 0.704 | 0.910 | 0.702 | 0.207 | 1091 | 80 | 184 |
| bicycle | 0.707 | 0.912 | 0.774 | 0.138 | 1002 | 62 | 171 |
| bus | 0.711 | 0.834 | 0.772 | 0.063 | 767 | 55 | 192 |
| sheep | 0.721 | 0.880 | 0.750 | 0.130 | 1013 | 57 | 119 |
| train | 0.751 | 0.887 | 0.805 | 0.082 | 868 | 57 | 225 |
| dog | 0.765 | 0.940 | 0.860 | 0.080 | 1929 | 96 | 211 |
| cat | 0.766 | 0.916 | 0.805 | 0.111 | 1505 | 88 | 278 |

En zayıf 5 sınıf (AP50-95): pottedplant 0.277, chair 0.473, bottle 0.535, boat 0.538, diningtable 0.566. Sınıf frekansı (train nesne) ile AP50-95 Spearman ρ=-0.15 (permütasyon p=0.521), AP50 ile ρ=0.11; sınıf başına train küçük-nesne oranı ile ρ=-0.42; val GT medyan kenar uzunluğu ile ρ=0.57 (n=20 sınıf, gürültülü). Frekans AP'nin zayıf belirleyicisi (|ρ|<0.3): zayıf sınıflarda sorun veri azlığından çok nesne boyutu/görünümü/bağlamıdır.

## 2. Nesne boyutu kırılımı (orijinal çözünürlük; COCO S<32²≤M<96²≤L)

| boyut | GT sayısı | AP50-95 | AP50 | <e-posta> (tüm conf, sınıf ort.) | <e-posta>:0.95 (sınıf ort.) | <e-posta> skor≥0.25 |
|---|---|---|---|---|---|---|
| all | 2407 | 0.631 | 0.816 | 0.961 | 0.765 | 0.799 |
| S | 143 | 0.247 | 0.417 | 0.750 | 0.461 | 0.378 |
| M | 704 | 0.454 | 0.658 | 0.923 | 0.659 | 0.720 |
| L | 1560 | 0.704 | 0.873 | 0.979 | 0.815 | 0.872 |

İnce alan aralıkları (GT düzeyi, eşleşme bayrağı):

| alan (px²) | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) |
|---|---|---|---|---|
| <16² | 9 | 44.4% | 0.0% | 11.1% |
| 16²–32² | 134 | 85.8% | 40.3% | 50.7% |
| 32²–64² | 377 | 97.3% | 69.2% | 74.8% |
| 64²–96² | 327 | 94.5% | 75.2% | 78.6% |
| 96²–192² | 698 | 97.4% | 81.4% | 85.1% |
| ≥192² | 862 | 99.1% | 92.0% | 91.1% |

Sınıf × boyut AP50-95 (n<5 ise '–'):

| sınıf | n_S | AP_S | n_M | AP_M | n_L | AP_L |
|---|---|---|---|---|---|---|
| aeroplane | 0 | – | 21 | 0.52 | 59 | 0.77 |
| bicycle | 0 | – | 12 | 0.43 | 50 | 0.78 |
| bird | 3 | – | 36 | 0.36 | 52 | 0.77 |
| boat | 4 | – | 25 | 0.54 | 35 | 0.60 |
| bottle | 13 | 0.14 | 52 | 0.51 | 30 | 0.73 |
| bus | 3 | – | 13 | 0.39 | 39 | 0.86 |
| car | 38 | 0.34 | 80 | 0.63 | 104 | 0.78 |
| cat | 0 | – | 2 | – | 86 | 0.78 |
| chair | 3 | – | 79 | 0.48 | 95 | 0.49 |
| cow | 0 | – | 25 | 0.72 | 31 | 0.72 |
| diningtable | 0 | – | 1 | – | 48 | 0.58 |
| dog | 0 | – | 10 | 0.74 | 86 | 0.77 |
| horse | 3 | – | 15 | 0.21 | 48 | 0.78 |
| motorbike | 3 | – | 10 | 0.54 | 54 | 0.69 |
| person | 51 | 0.20 | 215 | 0.51 | 519 | 0.72 |
| pottedplant | 12 | 0.20 | 57 | 0.29 | 40 | 0.32 |
| sheep | 8 | 0.41 | 16 | 0.57 | 33 | 0.86 |
| sofa | 0 | – | 1 | – | 58 | 0.61 |
| train | 0 | – | 4 | – | 53 | 0.77 |
| tvmonitor | 2 | – | 30 | 0.56 | 40 | 0.70 |

Küçük nesneler (n=143, val GT'nin %5.9'i): AP50-95 0.247 vs büyük 0.704 (fark 45.7 puan); <e-posta> 0.750 vs 0.979. Not: ölçüm orijinal piksel; eğitim/çıkarım 640 px letterbox olduğundan (~1.3× büyütme) S nesneler modelde ≈<41 px görünür. Val'de S örneği az olduğundan S-AP güven aralığı geniştir.

## 3. Hata türü ayrıştırması (TIDE benzeri)

Sayılar: tüm tahminler (conf≥0.001) ve işletim noktası (skor≥0.25). `Miss` tüm conf'ta hiçbir kutuyla IoU≥0.5 eşleşmeyen GT; işletim noktası sütununda ise o eşikte bulunamayan GT (eşleşmeyen + düşük skorla eşleşen).

| tür | sayı (tüm conf) | oran | sayı skor≥0.25 |
|---|---|---|---|
| TP | 2329 | 5.4% | 1922 |
| Dupe | 2538 | 5.8% | 93 |
| Cls | 3420 | 7.9% | 83 |
| Loc | 7974 | 18.3% | 140 |
| Both | 5219 | 12.0% | 79 |
| Bkg | 21977 | 50.6% | 269 |
| Miss | 78 | 3.2% (GT'nin) | 485 |

Oracle ΔAP (puan): ilgili hata türü düzeltilirse mAP ne kadar artardı (AP50 ve AP50-95, sınıf ortalaması). Taban: AP50=81.62, AP50-95=63.09.

| düzeltilen | ΔAP50 (puan) | ΔAP50-95 (puan) |
|---|---|---|
| Cls | 2.56 | 1.84 |
| Loc | 2.70 | 5.56 |
| Both | 1.19 | 0.80 |
| Dupe | 0.58 | -0.87 |
| Bkg | 3.48 | 2.18 |
| Miss | 3.29 | 2.44 |
| FP_hepsi | 13.98 | 6.15 |
| Kutu_kusursuz | 0.00 | 18.52 |
| TP_IoU_sirali | 1.46 | 5.42 |

Protokol etkisi: val'de etiketlerden çıkarılmış difficult nesne sayısı 493 (XML). Bunlardan biriyle aynı sınıf IoU≥0.5 örtüşen ve eşleşmemiş (FP sayılan) tahmin: 868 (skor≥0.25: 124; işletim noktası FP'lerinin 18.7%'i). Bu tahminler devkit gibi yok sayılsaydı ΔAP50 = +1.28, ΔAP50-95 = +0.86 puan (Ultralytics protokolünün val üzerinde ölçülmüş cezası; makalede belirtilmeli, Bkg/Loc/Dupe sayılarının bir kısmı bu yüzden yapaydır).


Not: `Dupe` ve `Loc` ΔAP50-95 değerleri IoU=0.5 tanımına göredir; bu kutular daha yüksek IoU eşiklerinde TP olabildiği için silme oracle'ı (Dupe) AP50-95'i düşürebilir; Loc oracle'ı kutuyu GT'ye yapıştırdığı için AP50-95 kazancı AP50'den büyüktür.


Hata türü × boyut (işletim noktası; tam uzun biçim `hata-kirilim.csv`):

| tür | S | M | L |
|---|---|---|---|
| TP (skor≥0.25, tahmin kutusu alanı) | 51 | 496 | 1375 |
| Dupe (skor≥0.25, tahmin kutusu alanı) | 0 | 33 | 60 |
| Cls (skor≥0.25, tahmin kutusu alanı) | 1 | 19 | 63 |
| Loc (skor≥0.25, tahmin kutusu alanı) | 4 | 60 | 76 |
| Both (skor≥0.25, tahmin kutusu alanı) | 2 | 15 | 62 |
| Bkg (skor≥0.25, tahmin kutusu alanı) | 54 | 120 | 95 |
| Miss (skor≥0.25 bulunamayan GT, GT alanı) | 89 | 197 | 199 |
| GT sayısı | 143 | 704 | 1560 |

AP50'ye en çok mal olan hata türü: Bkg (+3.48), sonra Miss (+3.29) ve Loc (+2.70). Oracle'lar birbirinden bağımsız ölçüldü, toplamları ΔAP'ye eşit değildir (etkileşim). `Kutu_kusursuz` = tüm TP kutuları GT'ye yapıştırılırsa AP50-95'in erişeceği üst sınır; `TP_IoU_sirali` = yalnız TP'lerin skorları IoU sırasına göre dağıtılırsa (FP skorları aynı) kazanç, yani IoU-farkında skorlamanın (VFL/QFL/IoU-aware) teorik payı.


Sınıf başına ΔAP50 (puan, oracle):

| sınıf | Cls | Loc | Both | Dupe | Bkg | Miss |
|---|---|---|---|---|---|---|
| aeroplane | 0.3 | 3.0 | 0.1 | 0.4 | 1.2 | 2.3 |
| bicycle | 1.0 | 0.5 | 0.6 | 0.9 | 0.5 | 3.3 |
| bird | 2.9 | 5.9 | 0.1 | 0.9 | 3.7 | 2.8 |
| boat | 1.4 | 4.0 | 0.1 | 1.0 | 4.0 | 5.0 |
| bottle | 1.5 | 1.8 | 0.2 | 0.5 | 10.2 | 2.3 |
| bus | 3.7 | 0.6 | 0.2 | 0.4 | 1.7 | 8.4 |
| car | 1.5 | 2.5 | 0.6 | 0.7 | 4.0 | 3.2 |
| cat | 3.6 | 2.7 | 0.1 | 0.3 | 0.1 | 2.5 |
| chair | 2.5 | 2.6 | 3.0 | 0.8 | 7.3 | 1.6 |
| cow | 3.8 | 2.5 | 0.1 | 0.6 | 4.1 | 1.5 |
| diningtable | 0.3 | 1.5 | 11.5 | 0.3 | 2.8 | 4.8 |
| dog | 2.0 | 0.1 | 0.1 | 0.3 | 0.6 | 1.1 |
| horse | 11.5 | 0.4 | 0.1 | 0.5 | 0.7 | 8.7 |
| motorbike | 0.8 | 3.0 | 1.0 | 0.3 | 2.5 | 3.8 |
| person | 0.3 | 2.9 | 0.2 | 1.0 | 2.1 | 1.8 |
| pottedplant | 0.3 | 11.0 | 2.1 | 1.0 | 8.3 | 6.7 |
| sheep | 5.0 | 0.6 | 0.5 | 0.1 | 4.1 | 0.0 |
| sofa | 7.4 | 2.0 | 2.4 | 1.5 | 4.4 | 0.0 |
| train | 0.7 | 5.2 | 0.5 | 0.1 | 1.9 | 3.5 |
| tvmonitor | 0.7 | 1.4 | 0.5 | 0.1 | 5.4 | 2.5 |

Kaçırılan GT (78) alt kırılımı: yalnız başka sınıfla örtüşen kutu var (sınıf karışıklığı) 22; aynı sınıf kutu var ama IoU∈[0.1,0.5) (lokalizasyon) 33; aynı sınıf IoU≥0.5 kutu var ama başka GT'ye atanmış (atama çakışması/örtüşme) 1; hiç kutu yok (gerçek kaçırma) 22 (kalan sınıflama kesin değil: bu üç grup kesişebilir).

## 4. Lokalizasyon kalitesi

| IoU aralığı | TP (tüm conf) | oran | TP skor≥0.25 | oran |
|---|---|---|---|---|
| 0.50–0.55 | 85 | 3.6% | 39 | 2.0% |
| 0.55–0.60 | 76 | 3.3% | 35 | 1.8% |
| 0.60–0.65 | 107 | 4.6% | 52 | 2.7% |
| 0.65–0.70 | 142 | 6.1% | 98 | 5.1% |
| 0.70–0.75 | 133 | 5.7% | 84 | 4.4% |
| 0.75–0.80 | 171 | 7.3% | 138 | 7.2% |
| 0.80–0.85 | 224 | 9.6% | 179 | 9.3% |
| 0.85–0.90 | 302 | 13.0% | 259 | 13.5% |
| 0.90–0.95 | 520 | 22.3% | 488 | 25.4% |
| 0.95–1.00 | 569 | 24.4% | 550 | 28.6% |

TP IoU ortalaması 0.844 (skor≥0.25: 0.868); işletim noktası TP'lerinin %16.0'i IoU<0.75. TP'lerde skor–IoU Spearman ρ=0.64 (skorun konum kalitesini ne kadar taşıdığı).

Sınıf başına AP50 − AP75 (büyükten küçüğe) ve TP ort. IoU:

| sınıf | AP50 | AP75 | AP50−AP75 (puan) | TP ort. IoU |
|---|---|---|---|---|
| pottedplant | 0.539 | 0.220 | 31.9 | 0.740 |
| person | 0.893 | 0.683 | 21.0 | 0.829 |
| aeroplane | 0.910 | 0.702 | 20.7 | 0.851 |
| boat | 0.762 | 0.602 | 16.1 | 0.824 |
| bird | 0.795 | 0.641 | 15.4 | 0.847 |
| chair | 0.665 | 0.526 | 13.9 | 0.819 |
| bicycle | 0.912 | 0.774 | 13.8 | 0.861 |
| motorbike | 0.825 | 0.693 | 13.2 | 0.853 |
| car | 0.839 | 0.708 | 13.1 | 0.852 |
| sheep | 0.880 | 0.750 | 13.0 | 0.880 |
| sofa | 0.746 | 0.617 | 12.8 | 0.876 |
| horse | 0.834 | 0.710 | 12.4 | 0.853 |
| cow | 0.870 | 0.752 | 11.8 | 0.869 |
| cat | 0.916 | 0.805 | 11.1 | 0.900 |
| bottle | 0.708 | 0.600 | 10.7 | 0.836 |
| tvmonitor | 0.861 | 0.758 | 10.3 | 0.858 |
| diningtable | 0.706 | 0.623 | 8.3 | 0.870 |
| train | 0.887 | 0.805 | 8.2 | 0.896 |
| dog | 0.940 | 0.860 | 8.0 | 0.894 |
| bus | 0.834 | 0.772 | 6.3 | 0.893 |

AP50→AP75 düşüşü sınıf ortalaması 13.6 puan; en büyük düşüş pottedplant (32), person (21), aeroplane (21). Kutu-regresyon (IoU/DFL/box kazancı) manipülasyonlarının hedef alanı budur.

## 5. Sınıf karışıklığı

İşletim noktası (skor≥0.25, IoU≥0.5): her GT için eşleşen en yüksek skorlu kutu (doğru sınıf varsa o). Matris dosyası: `hata-analizi_20261001-050625_h03_boxcp_s2_karisiklik.csv`.

| GT sınıfı → tahmin | sayı | GT'nin payı |
|---|---|---|
| chair → person | 6 | 3.4% |
| horse → sheep | 5 | 7.6% |
| dog → cat | 4 | 4.2% |
| sofa → chair | 4 | 6.8% |
| chair → diningtable | 3 | 1.7% |
| cow → horse | 3 | 5.4% |
| cow → sheep | 2 | 3.6% |
| sofa → person | 2 | 3.4% |
| bicycle → person | 1 | 1.6% |
| bird → bottle | 1 | 1.1% |
| bottle → bird | 1 | 1.1% |
| bus → car | 1 | 1.8% |

Çift bazında (yönsüz) toplam: chair/person 6; horse/sheep 5; chair/sofa 5; cat/dog 4; chair/diningtable 4; cow/horse 3; cow/sheep 2; person/sofa 2.


Tüm-conf alternatif görünüm: `Cls` türü tahminler (IoU≥0.5 başka sınıf GT, skor≥0.05), GT sınıfı → tahmin:

| GT → tahmin | tahmin sayısı |
|---|---|
| sofa → chair | 22 |
| person → sofa | 21 |
| person → chair | 20 |
| dog → cat | 14 |
| chair → sofa | 13 |
| cow → horse | 10 |
| motorbike → bicycle | 10 |
| sheep → dog | 8 |
| sheep → cow | 8 |
| horse → sheep | 8 |

Kanonik çiftler (yönsüz; `Cls` tahmin sayısı skor≥0.05 | işletim noktası GT sayısı): cat/dog 19 | 4; cow/sheep 15 | 2; chair/diningtable 10 | 4; chair/sofa 35 | 5; horse/cow 12 | 3; bus/train 7 | 0; motorbike/bicycle 13 | 0.

İşletim noktasında GT'lerin 2.1%'i başka sınıf olarak etiketlendi (50/2407); 17.6%'i hiç bulunamadı. Sınıf karışıklığı oracle ΔAP50 = 2.56 puan (§3; Cls, altı hata türü içinde 4. sırada); işletim noktasında karışan GT oranı yalnız 2.1%, yani sorun sınıf ayrımından çok kutu/skor düzeyindedir.

## 6. Kalabalık ve örtüşme

| nesne/görüntü | görüntü | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) | FP/görüntü (skor≥0.25) |
|---|---|---|---|---|---|---|
| 1 | 440 | 440 | 98.9% | 92.5% | 93.0% | 0.36 |
| 2 | 252 | 504 | 97.2% | 83.5% | 87.3% | 0.48 |
| 3–4 | 195 | 654 | 96.3% | 78.3% | 82.1% | 0.95 |
| 5–8 | 88 | 511 | 94.7% | 71.2% | 75.7% | 1.40 |
| ≥9 | 25 | 298 | 97.3% | 73.2% | 71.8% | 3.08 |

Aynı sınıftan başka GT ile en yüksek IoU (örtüşme) gruplarına göre recall:

| komşu GT IoU | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 |
|---|---|---|---|
| <0.1 (komşu yok) | 1908 | 96.6% | 80.8% |
| 0.1–0.3 | 357 | 96.9% | 76.8% |
| 0.3–0.5 | 105 | 99.0% | 78.1% |
| 0.5–0.7 | 33 | 97.0% | 66.7% |
| ≥0.7 | 4 | 100.0% | 75.0% |

Aynı sınıf GT çiftleri (IoU≥0.3): 75 çift; ikisi de bulundu 73, yalnız biri 2, hiçbiri 0. IoU≥0.7 (NMS eşiği üstü, doğrudan NMS etkilenir): 2 çift, ikisi de bulundu 2, yalnız biri 0, hiçbiri 0. Kaçırılan GT'lerin 1/78'i aynı sınıf IoU≥0.5 bir kutuya sahip ama o kutu başka GT'ye atanmış (olası NMS/atama kaybı).

Kalabalıkla recall düşüşü: tek nesneli görüntüde 92.5% iken ≥9 nesnelide 73.2% (skor≥0.25); tüm-conf recall 98.9% → 97.3%, <e-posta> 93.0% → 71.8%. Çiftlerde tek-bulunma ve atama çakışması sayıları küçükse kayıp NMS'ten değil düşük skor/zayıf konumdan gelir.

## 7. Kalibrasyon (skor vs gerçek precision)

Tahmin 'doğru' = IoU≥0.5 ile eşleşen (<e-posta>); ayrıca IoU≥0.75 için precision. Not: YOLO sınıf skoru olasılık değil, TAL hizalama hedefli (IoU'ya bağlı) bir skordur; düşük skor bölgesindeki düşük precision 'aday havuzu' doğasından gelir.

| skor aralığı | tahmin | ort. skor | <e-posta> | <e-posta> | TP ort. IoU | skor−precision |
|---|---|---|---|---|---|---|
| 0.0–0.1 | 39854 | 0.008 | 0.007 | 0.006 | 0.714 | 0.001 |
| 0.1–0.2 | 797 | 0.141 | 0.129 | 0.088 | 0.745 | 0.011 |
| 0.2–0.3 | 373 | 0.243 | 0.188 | 0.145 | 0.779 | 0.056 |
| 0.3–0.4 | 270 | 0.349 | 0.319 | 0.222 | 0.766 | 0.031 |
| 0.4–0.5 | 194 | 0.449 | 0.433 | 0.294 | 0.774 | 0.016 |
| 0.5–0.6 | 211 | 0.552 | 0.573 | 0.441 | 0.804 | -0.021 |
| 0.6–0.7 | 241 | 0.653 | 0.722 | 0.473 | 0.792 | -0.069 |
| 0.7–0.8 | 325 | 0.752 | 0.831 | 0.631 | 0.822 | -0.079 |
| 0.8–0.9 | 650 | 0.857 | 0.960 | 0.869 | 0.888 | -0.103 |
| 0.9–1.0 | 542 | 0.925 | 0.985 | 0.970 | 0.943 | -0.060 |

ECE (tüm tahminler, 10 eşit genişlikli kutu) = 0.005; ECE (skor≥0.1) = 0.052. `skor−precision` pozitifse aşırı güvenli, negatifse eksik güvenli. 
Skor≥0.5 tahminlerin <e-posta> = 87.5%, <e-posta> = 76.3% (fark 11.2 puan). Skor 0.5 üstü kutularda (skor−precision) ortalaması -0.066: model yüksek skorlarda eksik güvenli (skor < gerçek <e-posta>), sıralama iyi; asıl boşluk <e-posta> ile @0.75 arasındaki konum kalitesidir. TP'lerde skor–IoU Spearman ρ=0.64: skor konum kalitesini kısmen taşıyor. TP_IoU_sirali oracle'ı (§3) IoU-farkında skorlamanın teorik payını sayısallaştırıyor.

## 8. Eğitim dinamiği (results.csv)

| gösterge | değer |
|---|---|
| epoch sayısı | 100 |
| en iyi epoch (mAP50-95) | 96 (0.6342) |
| son epoch mAP50-95 | 0.6337 |
| epoch 50 / 80 / 90 / 100 | 0.6140 / 0.6317 / 0.6329 / 0.6337 |
| son 20 epoch eğimi (puan/10 epoch) | 0.063 |
| önceki 20 epoch (ep 71-90) eğimi | 0.225 |
| son 20 epoch std (puan) | 0.049 |
| ep 81-90 ort. vs ep 91-100 ort. (close_mosaic=10) | 0.6330 vs 0.6336 (Δ +0.07 puan) |
| mAP50 son epoch | 0.8197 |
| results.csv en iyi ep mAP50-95 vs metrics.json (best.pt yeniden val) | 0.6342 vs 0.6319 (fark 0.23 puan; ölçüm-protokolü gürültü tabanı) |

Loss tablosu (train loss mosaic açıkken ep1/ep90, kapalıyken ep100; val/train oranı):

| loss | train ep1 | train ep90 | train ep100 | val ep100 | val/train ep100 | val/train ep90 | train loss ep90→91 değişim |
|---|---|---|---|---|---|---|---|
| box | 1.002 | 0.801 | 0.691 | 0.872 | 1.26 | 1.09 | -7.0% |
| cls | 2.295 | 0.813 | 0.538 | 0.771 | 1.43 | 0.96 | -23.0% |
| dfl | 1.280 | 1.138 | 1.061 | 1.160 | 1.09 | 1.02 | -3.5% |

Kayıp payı (results.csv loss_items kazançlı; loss.py:469-471 — önceki sürüm kazançları iki kez uyguluyordu, 2026-10-01 düzeltildi) ep100: box 30.2%, cls 23.5%, dfl 46.3% (ep1: box 21.9%, cls 50.1%, dfl 28.0%). Son 20 epoch mAP eğimi +0.063 puan/10 ep (std 0.05) → doygun (eğim seed gürültüsü altında). close_mosaic sonrası (ep 91-100) ortalama +0.07 puan. Train loss mosaic kapanınca ani düşer (cls -23.0%); bu rejim değişimi, train/val farkının bir kısmının augmentation kaynaklı olduğunu gösterir. val cls loss son 20 epoch 0.782→0.771 (yükselmiyor): val loss bozulmadığı için klasik aşırı öğrenme yok, ama mosaic kapalıyken val/train oranı cls 1.43, box 1.26: train kaybı val kaybından ayrışıyor (genelleme açığı); val mAP doygun.

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
| ön-işleme ms | 1.2634 |
| çıkarım ms | 6.0195 |
| NMS/son-işlem ms | 0.689 |
| toplam ms | 7.9719 |
| FPS | 125.441 |
| batch | 1 |
| FP16 (1=evet) | 1.0 |
| ölçüm görüntüsü | 300 |
| ısınma görüntüsü | 50 |
| eğitim süresi (saat) | 2.9297 |

Varyant hız bütçesi (yalnız tahmin; gerçek ölçüm varyant koşumunda yapılır):

| varyant sınıfı | çarpan | tahmini toplam ms | tahmini FPS | not |
|---|---|---|---|---|
| eğitim-zamanı (aug/loss) varyantları | aynı | 7.97 | 125 | çıkarım grafiği değişmez; hız bütçesi ≈ 0 |
| Soft-NMS/DIoU-NMS (son-işlem ×1–5 varsayımı) | NMS ×1–5 | 7.97–10.73 | 93–125 | tahmin (ölçülmedi) |
| TTA (Ultralytics augment, ~3 ölçek+flip; literatür çarpanı 3.6× çıkarım) | çıkarım ×3.6 (literatür) | 23.62 | 42 | tahmin; literatur-cipa.md §4 (Ultralytics TTA dokümanı) |
| YOLO11s'e geçiş (bağlam satırı) | ≈3.4× GFLOPs | – | – | ölçülmedi |

Baseline batch=1 FP16 toplam 7.97 ms (125 FPS; çıkarım 6.02 ms, NMS 0.69 ms = toplamın %9'i). 'Gerçek zamanlı' için ≥30 FPS kabul edilirse (33 ms) ~4.2× gecikme payı vardır; TTA bile (~42 FPS tahmini) eşiğin üstünde kalabilir, ancak hız-doğruluk satırı ayrı raporlanmalıdır. Not: hız ölçümü GPU'nun paylaşıldığı bir anda alınmış olabilir (eğitim kuyruğu); karşılaştırmalı hız ölçümü aynı koşulda yapılmalıdır.
