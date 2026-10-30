# Hata analizi (K3.5) — baseline `20260930-131427_baseline_s2`

*Üretici: `scripts/analiz/hata_analizi.py` · yalnız VAL (1000 görüntü, 2407 nesne; difficult nesneler etiketlerde yok) · tahmin sayısı 45699 (conf=0.001, max_det=300, NMS IoU=0.7) · eşleştirme: pycocotools COCOeval · işletim noktası skor≥0.25.*

**Sızıntı notu:** Bu analizde test kümesi (VOC07 test), `rep_test*` dosyaları ve `metrics.json` içindeki `rep_*` anahtarları hiç kullanılmadı; betik bu anahtarları yükler yüklemez atar. Buradan çıkan hipotezler yalnız val bilgisine dayanır.

## 0. Genel ölçüler ve doğrulama

| ölçü | pycocotools (bu betik) | Ultralytics metrics.json (best.pt val) | fark |
|---|---|---|---|
| mAP50-95 | 0.6269 | 0.6278 | -0.0008 |
| mAP50 | 0.8086 | 0.8113 | -0.0027 |
| mAP75 (yalnız pycocotools) | 0.6813 | – | – |

AP50 ile AP50-95 arasındaki fark 18.2 puan; AP50→AP75 farkı 12.7 puan. pycocotools ile Ultralytics AP'si arasındaki küçük fark interpolasyon/eşleştirme (Ultralytics kendi IoU-matrisi eşleşmesini kullanır) kaynaklıdır.

## 1. Sınıf başına AP ve sınıf frekansı

| sınıf (AP50-95 artan) | AP50-95 | AP50 | AP75 | AP50−AP75 | train nesne | val nesne | val GT medyan kenar px |
|---|---|---|---|---|---|---|---|
| pottedplant | 0.275 | 0.546 | 0.225 | 0.321 | 1378 | 109 | 71 |
| chair | 0.463 | 0.636 | 0.507 | 0.129 | 2975 | 177 | 99 |
| bottle | 0.510 | 0.669 | 0.594 | 0.075 | 1669 | 95 | 70 |
| sofa | 0.569 | 0.700 | 0.623 | 0.077 | 755 | 59 | 237 |
| boat | 0.574 | 0.776 | 0.655 | 0.122 | 1076 | 64 | 112 |
| diningtable | 0.575 | 0.700 | 0.677 | 0.023 | 775 | 49 | 211 |
| bird | 0.611 | 0.806 | 0.675 | 0.131 | 1514 | 91 | 115 |
| tvmonitor | 0.622 | 0.833 | 0.725 | 0.108 | 1036 | 72 | 108 |
| horse | 0.624 | 0.839 | 0.641 | 0.198 | 1006 | 66 | 209 |
| person | 0.633 | 0.886 | 0.679 | 0.208 | 12471 | 785 | 141 |
| motorbike | 0.636 | 0.822 | 0.666 | 0.155 | 985 | 67 | 239 |
| car | 0.656 | 0.846 | 0.704 | 0.142 | 3045 | 222 | 92 |
| sheep | 0.674 | 0.882 | 0.712 | 0.170 | 1013 | 57 | 119 |
| cow | 0.676 | 0.841 | 0.757 | 0.085 | 791 | 56 | 145 |
| aeroplane | 0.701 | 0.888 | 0.726 | 0.162 | 1091 | 80 | 184 |
| bicycle | 0.712 | 0.902 | 0.774 | 0.128 | 1002 | 62 | 171 |
| bus | 0.730 | 0.834 | 0.771 | 0.063 | 767 | 55 | 192 |
| train | 0.754 | 0.905 | 0.854 | 0.050 | 868 | 57 | 225 |
| dog | 0.755 | 0.937 | 0.815 | 0.123 | 1929 | 96 | 211 |
| cat | 0.789 | 0.923 | 0.848 | 0.075 | 1505 | 88 | 278 |

En zayıf 5 sınıf (AP50-95): pottedplant 0.275, chair 0.463, bottle 0.510, sofa 0.569, boat 0.574. Sınıf frekansı (train nesne) ile AP50-95 Spearman ρ=-0.13 (permütasyon p=0.578), AP50 ile ρ=0.08; sınıf başına train küçük-nesne oranı ile ρ=-0.41; val GT medyan kenar uzunluğu ile ρ=0.55 (n=20 sınıf, gürültülü). Frekans AP'nin zayıf belirleyicisi (|ρ|<0.3): zayıf sınıflarda sorun veri azlığından çok nesne boyutu/görünümü/bağlamıdır.

## 2. Nesne boyutu kırılımı (orijinal çözünürlük; COCO S<32²≤M<96²≤L)

| boyut | GT sayısı | AP50-95 | AP50 | <e-posta> (tüm conf, sınıf ort.) | <e-posta>:0.95 (sınıf ort.) | <e-posta> skor≥0.25 |
|---|---|---|---|---|---|---|
| all | 2407 | 0.627 | 0.809 | 0.956 | 0.763 | 0.804 |
| S | 143 | 0.220 | 0.373 | 0.744 | 0.466 | 0.406 |
| M | 704 | 0.443 | 0.651 | 0.926 | 0.659 | 0.726 |
| L | 1560 | 0.701 | 0.864 | 0.975 | 0.815 | 0.876 |

İnce alan aralıkları (GT düzeyi, eşleşme bayrağı):

| alan (px²) | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) |
|---|---|---|---|---|
| <16² | 9 | 44.4% | 0.0% | 11.1% |
| 16²–32² | 134 | 85.1% | 43.3% | 56.7% |
| 32²–64² | 377 | 95.8% | 70.8% | 74.3% |
| 64²–96² | 327 | 93.9% | 74.6% | 77.4% |
| 96²–192² | 698 | 97.1% | 81.8% | 84.4% |
| ≥192² | 862 | 99.1% | 92.3% | 91.6% |

Sınıf × boyut AP50-95 (n<5 ise '–'):

| sınıf | n_S | AP_S | n_M | AP_M | n_L | AP_L |
|---|---|---|---|---|---|---|
| aeroplane | 0 | – | 21 | 0.53 | 59 | 0.76 |
| bicycle | 0 | – | 12 | 0.38 | 50 | 0.80 |
| bird | 3 | – | 36 | 0.39 | 52 | 0.78 |
| boat | 4 | – | 25 | 0.57 | 35 | 0.64 |
| bottle | 13 | 0.18 | 52 | 0.45 | 30 | 0.73 |
| bus | 3 | – | 13 | 0.47 | 39 | 0.87 |
| car | 38 | 0.31 | 80 | 0.63 | 104 | 0.78 |
| cat | 0 | – | 2 | – | 86 | 0.80 |
| chair | 3 | – | 79 | 0.44 | 95 | 0.50 |
| cow | 0 | – | 25 | 0.64 | 31 | 0.74 |
| diningtable | 0 | – | 1 | – | 48 | 0.58 |
| dog | 0 | – | 10 | 0.73 | 86 | 0.76 |
| horse | 3 | – | 15 | 0.25 | 48 | 0.75 |
| motorbike | 3 | – | 10 | 0.55 | 54 | 0.68 |
| person | 51 | 0.20 | 215 | 0.51 | 519 | 0.72 |
| pottedplant | 12 | 0.17 | 57 | 0.31 | 40 | 0.30 |
| sheep | 8 | 0.38 | 16 | 0.65 | 33 | 0.78 |
| sofa | 0 | – | 1 | – | 58 | 0.58 |
| train | 0 | – | 4 | – | 53 | 0.77 |
| tvmonitor | 2 | – | 30 | 0.56 | 40 | 0.68 |

Küçük nesneler (n=143, val GT'nin %5.9'i): AP50-95 0.220 vs büyük 0.701 (fark 48.1 puan); <e-posta> 0.744 vs 0.975. Not: ölçüm orijinal piksel; eğitim/çıkarım 640 px letterbox olduğundan (~1.3× büyütme) S nesneler modelde ≈<41 px görünür. Val'de S örneği az olduğundan S-AP güven aralığı geniştir.

## 3. Hata türü ayrıştırması (TIDE benzeri)

Sayılar: tüm tahminler (conf≥0.001) ve işletim noktası (skor≥0.25). `Miss` tüm conf'ta hiçbir kutuyla IoU≥0.5 eşleşmeyen GT; işletim noktası sütununda ise o eşikte bulunamayan GT (eşleşmeyen + düşük skorla eşleşen).

| tür | sayı (tüm conf) | oran | sayı skor≥0.25 |
|---|---|---|---|
| TP | 2318 | 5.1% | 1936 |
| Dupe | 2529 | 5.5% | 105 |
| Cls | 3606 | 7.9% | 94 |
| Loc | 8278 | 18.1% | 162 |
| Both | 5613 | 12.3% | 75 |
| Bkg | 23355 | 51.1% | 305 |
| Miss | 89 | 3.7% (GT'nin) | 471 |

Oracle ΔAP (puan): ilgili hata türü düzeltilirse mAP ne kadar artardı (AP50 ve AP50-95, sınıf ortalaması). Taban: AP50=80.86, AP50-95=62.69.

| düzeltilen | ΔAP50 (puan) | ΔAP50-95 (puan) |
|---|---|---|
| Cls | 3.08 | 2.24 |
| Loc | 2.58 | 5.25 |
| Both | 1.19 | 0.83 |
| Dupe | 0.58 | -0.98 |
| Bkg | 3.35 | 2.17 |
| Miss | 3.68 | 2.74 |
| FP_hepsi | 14.19 | 6.21 |
| Kutu_kusursuz | 0.00 | 18.17 |
| TP_IoU_sirali | 1.48 | 5.57 |

Protokol etkisi: val'de etiketlerden çıkarılmış difficult nesne sayısı 493 (XML). Bunlardan biriyle aynı sınıf IoU≥0.5 örtüşen ve eşleşmemiş (FP sayılan) tahmin: 894 (skor≥0.25: 139; işletim noktası FP'lerinin 18.8%'i). Bu tahminler devkit gibi yok sayılsaydı ΔAP50 = +1.31, ΔAP50-95 = +0.91 puan (Ultralytics protokolünün val üzerinde ölçülmüş cezası; makalede belirtilmeli, Bkg/Loc/Dupe sayılarının bir kısmı bu yüzden yapaydır).


Not: `Dupe` ve `Loc` ΔAP50-95 değerleri IoU=0.5 tanımına göredir; bu kutular daha yüksek IoU eşiklerinde TP olabildiği için silme oracle'ı (Dupe) AP50-95'i düşürebilir; Loc oracle'ı kutuyu GT'ye yapıştırdığı için AP50-95 kazancı AP50'den büyüktür.


Hata türü × boyut (işletim noktası; tam uzun biçim `hata-kirilim.csv`):

| tür | S | M | L |
|---|---|---|---|
| TP (skor≥0.25, tahmin kutusu alanı) | 57 | 510 | 1369 |
| Dupe (skor≥0.25, tahmin kutusu alanı) | 5 | 30 | 70 |
| Cls (skor≥0.25, tahmin kutusu alanı) | 4 | 15 | 75 |
| Loc (skor≥0.25, tahmin kutusu alanı) | 7 | 73 | 82 |
| Both (skor≥0.25, tahmin kutusu alanı) | 1 | 17 | 57 |
| Bkg (skor≥0.25, tahmin kutusu alanı) | 55 | 155 | 95 |
| Miss (skor≥0.25 bulunamayan GT, GT alanı) | 85 | 193 | 193 |
| GT sayısı | 143 | 704 | 1560 |

AP50'ye en çok mal olan hata türü: Miss (+3.68), sonra Bkg (+3.35) ve Cls (+3.08). Oracle'lar birbirinden bağımsız ölçüldü, toplamları ΔAP'ye eşit değildir (etkileşim). `Kutu_kusursuz` = tüm TP kutuları GT'ye yapıştırılırsa AP50-95'in erişeceği üst sınır; `TP_IoU_sirali` = yalnız TP'lerin skorları IoU sırasına göre dağıtılırsa (FP skorları aynı) kazanç, yani IoU-farkında skorlamanın (VFL/QFL/IoU-aware) teorik payı.


Sınıf başına ΔAP50 (puan, oracle):

| sınıf | Cls | Loc | Both | Dupe | Bkg | Miss |
|---|---|---|---|---|---|---|
| aeroplane | 0.6 | 3.1 | 0.3 | 0.5 | 1.2 | 1.1 |
| bicycle | 0.9 | 2.1 | 0.6 | 0.3 | 0.7 | 3.2 |
| bird | 3.9 | 3.8 | 0.1 | 1.1 | 2.8 | 4.7 |
| boat | 2.3 | 3.4 | 0.5 | 0.8 | 4.3 | 5.3 |
| bottle | 1.7 | 1.9 | 0.3 | 0.4 | 9.0 | 7.8 |
| bus | 2.3 | 1.4 | 0.2 | 0.4 | 1.9 | 6.7 |
| car | 1.0 | 1.1 | 0.4 | 0.6 | 4.2 | 2.4 |
| cat | 2.5 | 2.6 | 0.1 | 0.2 | 0.1 | 2.2 |
| chair | 3.3 | 3.4 | 2.1 | 0.8 | 6.0 | 2.7 |
| cow | 6.6 | 2.0 | 0.1 | 0.9 | 3.8 | 3.3 |
| diningtable | 1.7 | 1.9 | 10.8 | 0.1 | 3.3 | 3.2 |
| dog | 2.5 | 0.1 | 0.2 | 0.2 | 0.5 | 1.2 |
| horse | 11.0 | 0.7 | 0.5 | 0.5 | 0.6 | 5.6 |
| motorbike | 4.0 | 3.3 | 0.5 | 0.4 | 2.0 | 5.1 |
| person | 0.4 | 3.2 | 0.2 | 1.0 | 2.1 | 1.9 |
| pottedplant | 1.0 | 9.3 | 2.2 | 1.1 | 7.8 | 6.1 |
| sheep | 5.5 | 0.2 | 0.2 | 0.5 | 3.9 | 0.0 |
| sofa | 8.7 | 2.0 | 3.3 | 1.6 | 4.3 | 2.6 |
| train | 1.3 | 3.6 | 0.3 | 0.1 | 1.8 | 3.5 |
| tvmonitor | 0.4 | 2.4 | 0.8 | 0.1 | 6.8 | 5.0 |

Kaçırılan GT (89) alt kırılımı: yalnız başka sınıfla örtüşen kutu var (sınıf karışıklığı) 26; aynı sınıf kutu var ama IoU∈[0.1,0.5) (lokalizasyon) 36; aynı sınıf IoU≥0.5 kutu var ama başka GT'ye atanmış (atama çakışması/örtüşme) 1; hiç kutu yok (gerçek kaçırma) 26 (kalan sınıflama kesin değil: bu üç grup kesişebilir).

## 4. Lokalizasyon kalitesi

| IoU aralığı | TP (tüm conf) | oran | TP skor≥0.25 | oran |
|---|---|---|---|---|
| 0.50–0.55 | 73 | 3.1% | 42 | 2.2% |
| 0.55–0.60 | 92 | 4.0% | 43 | 2.2% |
| 0.60–0.65 | 99 | 4.3% | 54 | 2.8% |
| 0.65–0.70 | 111 | 4.8% | 69 | 3.6% |
| 0.70–0.75 | 154 | 6.6% | 111 | 5.7% |
| 0.75–0.80 | 162 | 7.0% | 117 | 6.0% |
| 0.80–0.85 | 209 | 9.0% | 171 | 8.8% |
| 0.85–0.90 | 354 | 15.3% | 318 | 16.4% |
| 0.90–0.95 | 496 | 21.4% | 469 | 24.2% |
| 0.95–1.00 | 568 | 24.5% | 542 | 28.0% |

TP IoU ortalaması 0.846 (skor≥0.25: 0.868); işletim noktası TP'lerinin %16.5'i IoU<0.75. TP'lerde skor–IoU Spearman ρ=0.62 (skorun konum kalitesini ne kadar taşıdığı).

Sınıf başına AP50 − AP75 (büyükten küçüğe) ve TP ort. IoU:

| sınıf | AP50 | AP75 | AP50−AP75 (puan) | TP ort. IoU |
|---|---|---|---|---|
| pottedplant | 0.546 | 0.225 | 32.1 | 0.752 |
| person | 0.886 | 0.679 | 20.8 | 0.835 |
| horse | 0.839 | 0.641 | 19.8 | 0.854 |
| sheep | 0.882 | 0.712 | 17.0 | 0.865 |
| aeroplane | 0.888 | 0.726 | 16.2 | 0.851 |
| motorbike | 0.822 | 0.666 | 15.5 | 0.844 |
| car | 0.846 | 0.704 | 14.2 | 0.847 |
| bird | 0.806 | 0.675 | 13.1 | 0.844 |
| chair | 0.636 | 0.507 | 12.9 | 0.817 |
| bicycle | 0.902 | 0.774 | 12.8 | 0.865 |
| dog | 0.937 | 0.815 | 12.3 | 0.886 |
| boat | 0.776 | 0.655 | 12.2 | 0.846 |
| tvmonitor | 0.833 | 0.725 | 10.8 | 0.861 |
| cow | 0.841 | 0.757 | 8.5 | 0.866 |
| sofa | 0.700 | 0.623 | 7.7 | 0.875 |
| cat | 0.923 | 0.848 | 7.5 | 0.902 |
| bottle | 0.669 | 0.594 | 7.5 | 0.839 |
| bus | 0.834 | 0.771 | 6.3 | 0.904 |
| train | 0.905 | 0.854 | 5.0 | 0.897 |
| diningtable | 0.700 | 0.677 | 2.3 | 0.876 |

AP50→AP75 düşüşü sınıf ortalaması 12.7 puan; en büyük düşüş pottedplant (32), person (21), horse (20). Kutu-regresyon (IoU/DFL/box kazancı) manipülasyonlarının hedef alanı budur.

## 5. Sınıf karışıklığı

İşletim noktası (skor≥0.25, IoU≥0.5): her GT için eşleşen en yüksek skorlu kutu (doğru sınıf varsa o). Matris dosyası: `hata-analizi_20260930-131427_baseline_s2_karisiklik.csv`.

| GT sınıfı → tahmin | sayı | GT'nin payı |
|---|---|---|
| chair → person | 8 | 4.5% |
| horse → sheep | 7 | 10.6% |
| sofa → person | 5 | 8.5% |
| cow → sheep | 4 | 7.1% |
| motorbike → person | 3 | 4.5% |
| cat → dog | 2 | 2.3% |
| cow → horse | 2 | 3.6% |
| dog → cat | 2 | 2.1% |
| sheep → dog | 2 | 3.5% |
| sofa → chair | 2 | 3.4% |
| bicycle → motorbike | 1 | 1.6% |
| bird → bottle | 1 | 1.1% |

Çift bazında (yönsüz) toplam: chair/person 8; horse/sheep 7; person/sofa 5; cow/sheep 5; cat/dog 4; motorbike/person 3; cow/horse 2; dog/sheep 2.


Tüm-conf alternatif görünüm: `Cls` türü tahminler (IoU≥0.5 başka sınıf GT, skor≥0.05), GT sınıfı → tahmin:

| GT → tahmin | tahmin sayısı |
|---|---|
| person → chair | 25 |
| sofa → chair | 24 |
| person → sofa | 23 |
| dog → cat | 13 |
| car → bus | 12 |
| cow → sheep | 11 |
| chair → sofa | 10 |
| cow → horse | 10 |
| motorbike → bicycle | 8 |
| horse → sheep | 8 |

Kanonik çiftler (yönsüz; `Cls` tahmin sayısı skor≥0.05 | işletim noktası GT sayısı): cat/dog 19 | 4; cow/sheep 18 | 5; chair/diningtable 10 | 2; chair/sofa 34 | 2; horse/cow 17 | 2; bus/train 5 | 0; motorbike/bicycle 13 | 1.

İşletim noktasında GT'lerin 2.4%'i başka sınıf olarak etiketlendi (57/2407); 16.8%'i hiç bulunamadı. Sınıf karışıklığı oracle ΔAP50 = 3.08 puan (§3; Cls, altı hata türü içinde 3. sırada); işletim noktasında karışan GT oranı yalnız 2.4%, yani sorun sınıf ayrımından çok kutu/skor düzeyindedir.

## 6. Kalabalık ve örtüşme

| nesne/görüntü | görüntü | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 | <e-posta> (tüm conf) | FP/görüntü (skor≥0.25) |
|---|---|---|---|---|---|---|
| 1 | 440 | 440 | 98.4% | 92.7% | 92.7% | 0.42 |
| 2 | 252 | 504 | 97.6% | 84.7% | 88.1% | 0.52 |
| 3–4 | 195 | 654 | 95.7% | 77.8% | 82.1% | 1.07 |
| 5–8 | 88 | 511 | 95.1% | 72.2% | 76.7% | 1.42 |
| ≥9 | 25 | 298 | 94.3% | 74.8% | 69.8% | 3.72 |

Aynı sınıftan başka GT ile en yüksek IoU (örtüşme) gruplarına göre recall:

| komşu GT IoU | GT | <e-posta> (tüm conf) | <e-posta> skor≥0.25 |
|---|---|---|---|
| <0.1 (komşu yok) | 1908 | 96.0% | 81.1% |
| 0.1–0.3 | 357 | 96.6% | 77.9% |
| 0.3–0.5 | 105 | 99.0% | 81.0% |
| 0.5–0.7 | 33 | 100.0% | 69.7% |
| ≥0.7 | 4 | 100.0% | 75.0% |

Aynı sınıf GT çiftleri (IoU≥0.3): 75 çift; ikisi de bulundu 74, yalnız biri 1, hiçbiri 0. IoU≥0.7 (NMS eşiği üstü, doğrudan NMS etkilenir): 2 çift, ikisi de bulundu 2, yalnız biri 0, hiçbiri 0. Kaçırılan GT'lerin 1/89'i aynı sınıf IoU≥0.5 bir kutuya sahip ama o kutu başka GT'ye atanmış (olası NMS/atama kaybı).

Kalabalıkla recall düşüşü: tek nesneli görüntüde 92.7% iken ≥9 nesnelide 74.8% (skor≥0.25); tüm-conf recall 98.4% → 94.3%, <e-posta> 92.7% → 69.8%. Çiftlerde tek-bulunma ve atama çakışması sayıları küçükse kayıp NMS'ten değil düşük skor/zayıf konumdan gelir.

## 7. Kalibrasyon (skor vs gerçek precision)

Tahmin 'doğru' = IoU≥0.5 ile eşleşen (<e-posta>); ayrıca IoU≥0.75 için precision. Not: YOLO sınıf skoru olasılık değil, TAL hizalama hedefli (IoU'ya bağlı) bir skordur; düşük skor bölgesindeki düşük precision 'aday havuzu' doğasından gelir.

| skor aralığı | tahmin | ort. skor | <e-posta> | <e-posta> | TP ort. IoU | skor−precision |
|---|---|---|---|---|---|---|
| 0.0–0.1 | 41940 | 0.008 | 0.006 | 0.006 | 0.721 | 0.002 |
| 0.1–0.2 | 856 | 0.140 | 0.111 | 0.098 | 0.764 | 0.029 |
| 0.2–0.3 | 436 | 0.247 | 0.209 | 0.138 | 0.756 | 0.039 |
| 0.3–0.4 | 270 | 0.347 | 0.293 | 0.211 | 0.781 | 0.055 |
| 0.4–0.5 | 201 | 0.445 | 0.388 | 0.274 | 0.779 | 0.057 |
| 0.5–0.6 | 222 | 0.551 | 0.559 | 0.392 | 0.794 | -0.008 |
| 0.6–0.7 | 265 | 0.647 | 0.713 | 0.472 | 0.796 | -0.066 |
| 0.7–0.8 | 318 | 0.755 | 0.808 | 0.651 | 0.841 | -0.053 |
| 0.8–0.9 | 638 | 0.857 | 0.948 | 0.851 | 0.881 | -0.092 |
| 0.9–1.0 | 553 | 0.928 | 0.991 | 0.973 | 0.942 | -0.063 |

ECE (tüm tahminler, 10 eşit genişlikli kutu) = 0.006; ECE (skor≥0.1) = 0.052. `skor−precision` pozitifse aşırı güvenli, negatifse eksik güvenli. 
Skor≥0.5 tahminlerin <e-posta> = 86.3%, <e-posta> = 75.2% (fark 11.2 puan). Skor 0.5 üstü kutularda (skor−precision) ortalaması -0.056: model yüksek skorlarda eksik güvenli (skor < gerçek <e-posta>), sıralama iyi; asıl boşluk <e-posta> ile @0.75 arasındaki konum kalitesidir. TP'lerde skor–IoU Spearman ρ=0.62: skor konum kalitesini kısmen taşıyor. TP_IoU_sirali oracle'ı (§3) IoU-farkında skorlamanın teorik payını sayısallaştırıyor.

## 8. Eğitim dinamiği (results.csv)

| gösterge | değer |
|---|---|
| epoch sayısı | 100 |
| en iyi epoch (mAP50-95) | 84 (0.6325) |
| son epoch mAP50-95 | 0.6317 |
| epoch 50 / 80 / 90 / 100 | 0.6137 / 0.6307 / 0.6320 / 0.6317 |
| son 20 epoch eğimi (puan/10 epoch) | -0.035 |
| önceki 20 epoch (ep 71-90) eğimi | 0.241 |
| son 20 epoch std (puan) | 0.055 |
| ep 81-90 ort. vs ep 91-100 ort. (close_mosaic=10) | 0.6318 vs 0.6313 (Δ -0.05 puan) |
| mAP50 son epoch | 0.8182 |
| results.csv en iyi ep mAP50-95 vs metrics.json (best.pt yeniden val) | 0.6325 vs 0.6278 (fark 0.47 puan; ölçüm-protokolü gürültü tabanı) |

Loss tablosu (train loss mosaic açıkken ep1/ep90, kapalıyken ep100; val/train oranı):

| loss | train ep1 | train ep90 | train ep100 | val ep100 | val/train ep100 | val/train ep90 | train loss ep90→91 değişim |
|---|---|---|---|---|---|---|---|
| box | 0.998 | 0.798 | 0.687 | 0.866 | 1.26 | 1.08 | -7.1% |
| cls | 2.298 | 0.801 | 0.536 | 0.788 | 1.47 | 0.99 | -21.8% |
| dfl | 1.287 | 1.143 | 1.058 | 1.156 | 1.09 | 1.01 | -4.1% |

Kazançlarla (box 7.5, cls 0.5, dfl 1.5) ağırlıklı kayıp payı ep100: box 73.5%, cls 3.8%, dfl 22.6% (ep1: box 70.9%, cls 10.9%, dfl 18.3%). Son 20 epoch mAP eğimi -0.035 puan/10 ep (std 0.05) → doygun (eğim seed gürültüsü altında). close_mosaic sonrası (ep 91-100) ortalama -0.05 puan. Train loss mosaic kapanınca ani düşer (cls -21.8%); bu rejim değişimi, train/val farkının bir kısmının augmentation kaynaklı olduğunu gösterir. val cls loss son 20 epoch 0.795→0.788 (yükselmiyor): val loss bozulmadığı için klasik aşırı öğrenme yok, ama mosaic kapalıyken val/train oranı cls 1.47, box 1.26: train kaybı val kaybından ayrışıyor (genelleme açığı); val mAP doygun.

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
| ön-işleme ms | 1.3508 |
| çıkarım ms | 6.0658 |
| NMS/son-işlem ms | 0.6881 |
| toplam ms | 8.1047 |
| FPS | 123.386 |
| batch | 1 |
| FP16 (1=evet) | 1.0 |
| ölçüm görüntüsü | 300 |
| ısınma görüntüsü | 50 |
| eğitim süresi (saat) | 2.9055 |

Varyant hız bütçesi (yalnız tahmin; gerçek ölçüm varyant koşumunda yapılır):

| varyant sınıfı | çarpan | tahmini toplam ms | tahmini FPS | not |
|---|---|---|---|---|
| eğitim-zamanı (aug/loss) varyantları | aynı | 8.10 | 123 | çıkarım grafiği değişmez; hız bütçesi ≈ 0 |
| Soft-NMS/DIoU-NMS (son-işlem ×1–5 varsayımı) | NMS ×1–5 | 8.10–10.86 | 92–123 | tahmin (ölçülmedi) |
| TTA (Ultralytics augment, ~3 ölçek+flip; literatür çarpanı 3.6× çıkarım) | çıkarım ×3.6 (literatür) | 23.88 | 42 | tahmin; literatur-cipa.md §4 (Ultralytics TTA dokümanı) |
| YOLO11s'e geçiş (bağlam satırı) | ≈3.4× GFLOPs | – | – | ölçülmedi |

Baseline batch=1 FP16 toplam 8.10 ms (123 FPS; çıkarım 6.07 ms, NMS 0.69 ms = toplamın %8'i). 'Gerçek zamanlı' için ≥30 FPS kabul edilirse (33 ms) ~4.1× gecikme payı vardır; TTA bile (~42 FPS tahmini) eşiğin üstünde kalabilir, ancak hız-doğruluk satırı ayrı raporlanmalıdır. Not: hız ölçümü GPU'nun paylaşıldığı bir anda alınmış olabilir (eğitim kuyruğu); karşılaştırmalı hız ölçümü aynı koşulda yapılmalıdır.
