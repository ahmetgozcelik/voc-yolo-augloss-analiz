# Veri Manifesti — PASCAL VOC 2007+2012, tespit (Main) (2.1 / cv-1)

İndirme: resmî Oxford aynasından üç tar, `curl` ile (aşağıda) · Tarih: 2026-09-29, 19:06:26Z → 19:32:03Z
(`data/VOC/indirme_baslangic.txt`, `indirme_bitis.txt`) · Hedef makine: A, yerel (<makine>, Windows 10,
Git Bash) · Ortam: `env/.venv` (Python 3.11, ultralytics 8.4.165, torch 2.14.0+cu130).
Makine-okur sayılar: `data/data_manifest.json`. Bütün komutlar WS kökünden koşuldu
(`WS = deney_workspace/2-1-enhancing-real-time-object-detection-on-pascal-v`).

## İndirme ve dönüştürme komutları (V1)
```bash
cd data/VOC/raw
B=https://thor.robots.ox.ac.uk/pascal/VOC
curl -L -C - --retry 5 --retry-delay 5 -sS -o VOCtrainval_06-Nov-2007.tar $B/voc2007/VOCtrainval_06-Nov-2007.tar
curl -L -C - --retry 5 --retry-delay 5 -sS -o VOCtest_06-Nov-2007.tar     $B/voc2007/VOCtest_06-Nov-2007.tar
curl -L -C - --max-time 560 -sS -o VOCtrainval_11-May-2012.tar $B/voc2012/VOCtrainval_11-May-2012.tar  # 2 kez koşuldu (2. koşum --max-time 570, -C - ile devam; 1. parça rc=28 zaman aşımı, 563 871 479 B)
md5sum *.tar; sha256sum *.tar
cd ..; mkdir -p images; for t in VOCtrainval_06-Nov-2007 VOCtest_06-Nov-2007 VOCtrainval_11-May-2012; do tar -xf raw/$t.tar -C images; done
cd ../..   # WS kökü
env/.venv/Scripts/python.exe scripts/veri/voc_donustur.py        # XML -> YOLO (Ultralytics VOC.yaml mantığı)
env/.venv/Scripts/python.exe scripts/veri/voc_split_denetim.py   # split + dağılım + sızıntı (sha1, dHash) + yaml'lar
env/.venv/Scripts/python.exe scripts/veri/voc_yakin_kopya.py     # yakın-kopya NCC taraması
env/.venv/Scripts/python.exe scripts/veri/voc_yukleme_testi.py   # Ultralytics yükleme testi (eğitim yok)
env/.venv/Scripts/python.exe scripts/veri/voc_manifest_json.py   # data_manifest.json + sınıf tablosu
```
**Ultralytics indirmesinden sapma (bilinçli):** paketteki `ultralytics/cfg/datasets/VOC.yaml`
aynı üç arşivin **zip** kopyalarını `github.com/ultralytics/assets` sürümünden indiriyor ve bu zip'ler için
resmî hash yok. Biz resmî **tar**'ları Oxford aynasından indirdik (torchvision `voc.py` ile aynı URL),
böylece MD5'i resmî değerle eşleyebildik. Dönüşüm adımı Ultralytics'in adımıyla aynı:
`scripts/veri/voc_donustur.py`, `convert_label` fonksiyonunu kopyalamıyor, paketteki VOC.yaml'ın `download`
metninden okuyup `exec` ediyor. `# Convert` döngüsü birebir aynı. Tek fark, `yaml["path"]` değerinin
`WS/data/VOC` olması (Ultralytics'in global `datasets_dir`'i kullanılmadı).

## `difficult` nesnelerin ele alınışı (önemli)
Ultralytics dönüştürücüsü `if cls in names and int(obj.find("difficult").text) != 1` koşuluyla
**`difficult==1` nesneleri etiket dosyasına hiç yazmıyor**. Sonuç olarak:
- Eğitimde bu nesneler görülmüyor, yani arka plan sayılıyor. train'de atılan difficult nesne sayısı **6672**
  (train'deki tüm nesnelerin %15,1'i). val'de **493**, test'te **2944**.
- Değerlendirmede (val ve kilitli test) bu nesneler GT'de yok. Onları bulan tespitler Ultralytics'te
  **FP** sayılıyor. Resmî VOC devkit'i ise bunları yok sayar (ne TP ne FP). Bu yüzden Ultralytics mAP'i
  devkit'e göre aşağı çekilmiş olur. Makalede "Ultralytics protokolü: difficult nesneler etiketlerden
  çıkarıldı" diye açıkça yazılmalı (bkz. `notlar/literatur-cipa.md` madde 2).
- Kutu dönüşümü: `x=(xmin+xmax)/2-1`, `y=(ymin+ymax)/2-1`, `w=xmax-xmin`, `h=ymax-ymin`, görüntü
  boyutuna normalize (VOC'un 1 tabanlı pikselleri için −1 kayması). Sınıf listesinde olmayan ad yok.
- Tüm görüntülerde en az bir difficult olmayan nesne var: boş etiket dosyası sayısı train 0 / val 0 / test 0.
- Etiket satırı ↔ XML'deki difficult olmayan nesne sayısı uyuşmazlığı: **0** görüntü.

## Bütünlük (V2)
| Arşiv | bayt | MD5 (ölçülen) | MD5 (resmî, torchvision `voc.py`) | eşleşme | SHA256 (ölçülen) |
|---|---|---|---|---|---|
| VOCtrainval_06-Nov-2007.tar | 460 032 000 | c52e279531787c972589f7e41ab4ae64 | c52e279531787c972589f7e41ab4ae64 | EVET | 7d8cd951…7508 |
| VOCtest_06-Nov-2007.tar | 451 020 800 | b6e924de25625d8de591ea690078ad9f | b6e924de25625d8de591ea690078ad9f | EVET | 6836888e…4892 |
| VOCtrainval_11-May-2012.tar | 1 999 639 040 | 6cd6e144f989b92b3379bac3b3de84fd | 6cd6e144f989b92b3379bac3b3de84fd | EVET | e14f7632…cbb |

(Tam SHA256 değerleri `data_manifest.json` içinde. Toplam arşiv boyutu 2,911 GB.)

| Öğe | Sayı |
|---|---|
| VOC2007 JPEG / XML (trainval+test) | 9963 / 9963 |
| VOC2012 JPEG / XML | 17125 / 17125 (5585'i Main trainval'da değil, `images/VOCdevkit/VOC2012/JPEGImages`'da kaldı, kullanılmıyor) |
| Dönüştürülen görüntü (`images/{train,val}{2007,2012}`, `images/test2007`) | 21503 = 2501 + 2510 + 5717 + 5823 + 4952 |
| YOLO etiket dosyası | 21503 (görüntü sayısıyla birebir) |
| Toplam etiket satırı (difficult olmayan nesne) | 52090 = 37651 (train) + 2407 (val) + 12032 (test) |
| XML'deki nesneler, 07 trainval | 15662 (difficult 3054, difficult olmayan 12608) |
| XML'deki nesneler, 07 test | 14976 (difficult 2944, difficult olmayan 12032) |
| XML'deki nesneler, 12 trainval | 31561 (difficult 4111, difficult olmayan 27450) |
| Disk (`du`) | raw tar 2,911 GB · images (+VOCdevkit) 2,863 GB · labels 0,003 GB · `data/VOC` toplam 5,6 GB |

Not: Tar'lar `data/VOC/raw`'da tekrarlanabilirlik için tutuluyor. Silinirse 2,9 GB yer açılır.
`images/VOCdevkit` (0,648 GB) ise XML'leri barındırıyor; boyut ve hata analizi için gerekli, silinmemeli.

## Split (V3, V4)
| bölüm | görüntü | nesne (difficult olmayan) | VOC2007 / VOC2012 | kaynak |
|---|---|---|---|---|
| train | 15551 | 37651 | 4708 / 10843 | havuz − val |
| val | 1000 | 2407 | 303 / 697 | havuzdan seed=0, 07/12 katmanlı |
| test | 4952 | 12032 | 4952 / 0 | resmî VOC2007 test, **KİLİTLİ** |

- Havuz: VOC2007 trainval (5011) + VOC2012 trainval (11540) = 16551. 07 oranı 0,3028, yani
  val'den 07 payı `round(1000·5011/16551)=303` ve 12 payı 697.
- Yöntem: `random.Random(0)`, önce `rng.sample(sorted(pool07), 303)`, sonra aynı rng ile
  `rng.sample(sorted(pool12), 697)`. Kayıtlar `(year, id, relpath)` üçlüsüne göre sıralı. Val'in resmî
  alt kümelere dağılımı: train2007 138, val2007 165, train2012 343, val2012 354.
- Ayrıklık (liste düzeyinde): train∩val 0, train∩test 0, val∩test 0.
- Dosyalar: `data/splits/{train,val,test}.txt` (mutlak görüntü yolları, `/` ayraçlı; Ultralytics
  etiketi `/images/` → `/labels/` ile buluyor) + `data/splits/split_meta.json`.

| liste | sha256 (dosya, mutlak yollar) | sha256 (kanonik `VOC<yıl>/<id>`, sıralı, makineden bağımsız) |
|---|---|---|
| train.txt | fdcd6dde4b8f9a08e48464d6728e19aa01fd3599743742837a2272b15d6d2fe0 | 640e821a5b3758478462b271b21707f754c3b010ba6d3e4d8858e437691ceb76 |
| val.txt | 558db2f0efb443b62702ccb8e961643227d134429118159b3cdfaf7970de111e | dda7a378f496d2a1bb79e6da003990df516cda9bd27d5c37240c2a80497fbd0e |
| test.txt | a10ef0a3f9f1029768d918f9ef5c4093108032dca041e413e3e789851a8efeb0 | 467ea94646a8610a84101de35bf819924fda90ab88c5131a2ae87a2c25a19f4c |

- Ultralytics yaml'ları: `data/voc_dev.yaml` (train=splits/train.txt, val=splits/val.txt; K4 döngüsü) ve
  `data/voc_test.yaml` (train=splits/train.txt, val=splits/test.txt; **YALNIZ K5 kilit açılışında**).
  `names` sırası Ultralytics VOC.yaml ile aynı (0 aeroplane … 19 tvmonitor), nc=20.
- Protokol notu: literatürdeki 07+12 → 07test protokolü 16551 görüntünün tamamıyla eğitir ve Ultralytics
  varsayılanı best.pt'yi test2007 üzerinden seçer. Biz 1000 görüntüyü val için ayırıyoruz (train %6,0 daha
  küçük) ve testi kilitliyoruz. VOC resmî "Best Practice" metni de bunu istiyor: "all development, e.g. feature
  selection and parameter tuning, must use the 'trainval' … set alone". Bu fark `replikasyon.md`'ye taşınmalı.

### Yükleme testi (Ultralytics, eğitim yok)
`scripts/veri/voc_yukleme_testi.py` çıktısı: `ultralytics 8.4.165 nc 20 names[0,19] aeroplane tvmonitor` ·
`train: images=15551 objects=37651 n_classes_seen=20 empty=0` · `val: images=1000 objects=2407
n_classes_seen=20 empty=0` · `voc_test.yaml parse ok` (test etiketleri bu testte okunmadı).
**Önbellek notu:** Ultralytics görüntü yollarını sıralıyor ve `.cache` dosyasını ilk etiketin klasörüne yazıyor.
train ve val ikisi de `images/train2007/...` ile başladığı için aynı `labels/train2007.cache` dosyasını
paylaşıyorlar. Hash denetimi uyuşmazlığı yakalıyor ve yeniden tarıyor, yani doğruluk bozulmuyor; bedeli
koşum başına yaklaşık 8 sn yeniden tarama (ölçülen: iki veri seti yüklemesi toplam 9,6 sn). Aynı anda iki
koşum başlatılırsa önbellek yazımı çakışabilir; tek GPU'da sıralı koşum önerilir.

## Dağılım
Tam tablo: `data/sinif_dagilimi.csv` (split × sınıf: görüntü, nesne, atılan difficult, small/medium/large).
Boyut kovaları COCO tarzı ve **orijinal piksel kutu alanına** göre: small <32², medium 32²–96², large ≥96².
Kutu alanı kullanıldı (COCO maske alanı kullanır).

| id | sınıf | train img | train obj | val img | val obj | test img | test obj | train difficult | train S/M/L |
|---|---|---|---|---|---|---|---|---|---|
| 0 | aeroplane | 853 | 1091 | 55 | 80 | 204 | 285 | 98 | 57/247/787 |
| 1 | bicycle | 746 | 1002 | 49 | 62 | 239 | 337 | 137 | 21/234/747 |
| 2 | bird | 1038 | 1514 | 57 | 91 | 282 | 459 | 212 | 89/486/939 |
| 3 | boat | 651 | 1076 | 38 | 64 | 172 | 263 | 233 | 81/418/577 |
| 4 | bottle | 890 | 1669 | 60 | 95 | 212 | 469 | 331 | 320/919/430 |
| 5 | bus | 568 | 767 | 39 | 55 | 174 | 213 | 86 | 15/141/611 |
| 6 | car | 1750 | 3045 | 124 | 222 | 721 | 1201 | 676 | 392/1217/1436 |
| 7 | cat | 1343 | 1505 | 74 | 88 | 322 | 358 | 23 | 5/81/1419 |
| 8 | chair | 1478 | 2975 | 86 | 177 | 417 | 756 | 1101 | 70/1119/1786 |
| 9 | cow | 413 | 791 | 31 | 56 | 127 | 244 | 184 | 38/257/496 |
| 10 | diningtable | 691 | 775 | 47 | 49 | 190 | 206 | 218 | 6/67/702 |
| 11 | dog | 1624 | 1929 | 83 | 96 | 418 | 489 | 51 | 9/199/1721 |
| 12 | horse | 723 | 1006 | 46 | 66 | 274 | 348 | 79 | 11/132/863 |
| 13 | motorbike | 722 | 985 | 49 | 67 | 222 | 325 | 83 | 9/203/773 |
| 14 | person | 5711 | 12471 | 384 | 785 | 2007 | 4528 | 2168 | 833/3788/7850 |
| 15 | pottedplant | 717 | 1378 | 55 | 109 | 224 | 480 | 221 | 158/644/576 |
| 16 | sheep | 395 | 1013 | 26 | 57 | 97 | 242 | 274 | 116/414/483 |
| 17 | sofa | 684 | 755 | 52 | 59 | 223 | 239 | 369 | 0/51/704 |
| 18 | train | 758 | 868 | 47 | 57 | 259 | 282 | 56 | 5/85/778 |
| 19 | tvmonitor | 778 | 1036 | 53 | 72 | 229 | 308 | 72 | 23/439/574 |

- Dengesizlik (nesne, en çok / en az): train 16,52 (person 12471 / sofa 755) · val 16,02 (person 785 /
  diningtable 49) · test 21,98 (person 4528 / diningtable 206).
- Görüntü başına nesne ortalaması: train 2,421 · val 2,407 · test 2,430.
- Görüntü boyutu (piksel): en büyük kenar 500; min genişlik/yükseklik train 127/71, val 246/99, test 148/139.

| Nesne boyutu | train | val | test |
|---|---|---|---|
| small (<32², orijinal px) | 2258 (%6,0) | 143 (%5,9) | 566 (%4,7) |
| medium | 11141 (%29,6) | 704 (%29,3) | 3823 (%31,8) |
| large (≥96²) | 24252 (%64,4) | 1560 (%64,8) | 7643 (%63,5) |
| small, imgsz=640 ölçeğinde (alan × (640/max(W,H))²) | 1022 (%2,7) | 69 (%2,9) | 223 (%1,9) |
| medium, 640 ölçeğinde | 9171 (%24,4) | 585 (%24,3) | 3011 (%25,0) |
| large, 640 ölçeğinde | 27458 (%72,9) | 1753 (%72,8) | 8798 (%73,1) |

Ultralytics `mAP` S/M/L ayrımı raporlamaz. Hata analizinde (K3.5) kova tanımı yukarıdaki gibi, orijinal
pikselle yapılmalı ve hangi ölçeğin kullanıldığı yazılmalı. VOC'ta küçük nesne oranı düşük (%5–6), bu yüzden
"küçük nesne iyileştirmesi" iddiası test'te yalnızca 566 nesneye dayanır.

## Sızıntı denetimi (V5)
Kapsam: havuz (train∪val, 16551) × test (4952). Test yalnızca bayt hash'i, küçük resim istatistiği ve etiket
sayımı için okundu. Ayrıntılar `data/sizinti_raporu.json`'da.
- **Dosya adı çakışması (havuz∩test): 0.** 07 test id'leri 07 trainval'dan ayrık, 12 adları `20xx_` önekli.
- **İçerik hash çakışması (görüntü baytları sha1, havuz∩test): 0.** VOC2012 trainval ↔ VOC2007 test
  bayt-kopyası da bu sayıya dahil: 0.
- Test içi sha1 kopyası: 0 grup.
- Havuz içi sha1 kopyası: **3 çift**, hepsi VOC2007 trainval'ın kendi içinde (`val2007/000338 = val2007/007284`,
  `val2007/000949 = val2007/005042`, `train2007/008037 = train2007/009623`). Üçü de bizim train'de. train↔val'e
  bölünmüş kopya çifti: **0**. Resmî veri olduğu için dokunulmadı.
- Yakın-kopya (algısal) taraması, havuz × test:
  - dHash (64 bit) Hamming mesafesi 0: **0** · ≤2: 3 · ≤4: 20 çift. Hepsinde 64×64 gri ortalama mutlak fark
    12–121 arasında; eşleşmeler düşük dokulu görüntülere yığılıyor (tek bir test görüntüsü, 007091, 7 adayda
    geçiyor). Kopya değil.
  - 32×32 NCC (yatay ayna dahil): ≥0,99: **0** · ≥0,98: 1 · ≥0,95: 31. Havuzdaki her görüntünün en yakın
    test komşusu için medyan 0,645, 99. yüzdelik 0,924. Tek ≥0,98 aday (`val2012/2011_000096` ↔
    `test2007/006649`, 0,981, ayna) farklı en-boy oranına sahip (375×500 ile 500×333) ve NCC çözünürlükle düşüyor
    (32: 0,981 → 128: 0,969 → 256: 0,962). Kopya değil.
- **Çıkarılan görüntü: 0** (kesin çakışma yok; train 15551, val 1000 değişmedi).
- Kapsam dışı: COCO (ön-eğitim) ↔ VOC07 test tekrarı **denetlenmedi**, çünkü COCO indirilmedi. Bu bir kısıt
  (literatur-cipa.md madde 4).

## Lisans ve etik (V6)
Kaynak resmî sayfalar (HTTP ile alındı, 2026-09-29): `http://host.robots.ox.ac.uk/pascal/VOC/voc2007/`,
`…/voc2012/`. Alıntılar:
- "The VOC2007 data includes some images provided by 'flickr'. Use of these images must respect the
  corresponding terms of use" · "For the purposes of the challenge, the identity of the images in the database,
  e.g. source and name of owner, has been obscured." (2012 sayfasında da aynı "Database Rights" metni var.)
- Atıf gereği: "If you make use of the VOC2007 data, please cite the following reference…" (`pascal-voc-2007`
  misc ve 2012 için `pascal-voc-2012`). Ana makale Everingham ve ark., IJCV 88(2):303–338, 2010
  (`notlar/references-cekirdek.bib`: `everingham_2010_voc`).
- Lisans adı: VOC için açık bir lisans (CC vb.) **yok**. Görüntüler Flickr kullanım şartlarına tabi, her
  görüntünün telif hakkı sahibinde. **Ticari kısıt:** görüntüler yeniden dağıtılmamalı ve ticari kullanım
  Flickr/sahip şartlarına bağlı. Bu çalışma akademik ve ticari olmayan bir deney, veri yeniden dağıtılmıyor
  (data/ git'te izlenmiyor).
- Anotasyonlar için ayrı bir lisans metni resmî sayfada **bulunamadı**. "Araştırma amaçlı" genel kabulü
  resmî metinle doğrulanamadı (Belirsizlikler).
- Kişisel veri: görüntülerde insanlar var (person sınıfı: train'de 5711 görüntü). Sahip kimlikleri
  organizatörlerce gizlenmiş. Yüz ve kimlik analizi yapılmıyor.
- Ultralytics kodu AGPL-3.0. Bu, veriyi değil araç zincirini etkiler (K3/K6'da not edilmeli).

## Literatürle uyum
| Sayı | Kaynak | Ölçülen | Uyum |
|---|---|---|---|
| VOC2007 9963 görüntü, 24640 nesne | resmî VOC2007 sayfası | 9963 görüntü; difficult olmayan 12608+12032 = **24640** | EVET (resmî sayı difficult olmayanları sayıyor; difficult dahil toplam 30638) |
| VOC2011/12 trainval 27450 nesne | resmî sayfa ("11,530 images containing 27,450 ROI annotated objects") | 27450 difficult olmayan nesne, **11540** görüntü | Nesne EVET · görüntü sayısında **10 fark**: `ImageSets/Main/trainval.txt` 11540 satır. Ultralytics VOC.yaml ve yaygın 16551 = 5011+11540 değeri de 11540'ı kullanıyor. Sayfadaki 11530 muhtemelen yazım/sürüm farkı (doğrulanamadı). |
| 07+12 trainval 16551, 07 test 4952 | konu.md "~16,5k + ~5k", Ultralytics VOC.yaml yorumları, literatur-cipa.md | 16551 / 4952 | EVET |
| Ultralytics VOC.yaml: 2012 trainval zip "17125 images" | VOC.yaml yorumu | JPEGImages 17125, Main trainval 11540 | EVET (17125 tüm JPEG; tespit için 11540) |
| Boyut "~2–3 GB", "2.8 GB" | konu.md, VOC.yaml | tar toplamı 2,911 GB (2,71 GiB) | EVET |
| Eğitim kümesi | literatür 16551 | 15551 (1000 val'e ayrıldı) | Bilinçli protokol farkı (V4), baseline'ı literatürün biraz altına itebilir |

## Belirsizlikler
- Anotasyon lisansı resmî sayfada açıkça yazılı değil. Devkit PDF'i (`devkit_doc`) indirilmedi.
- 11530 ile 11540 görüntü farkının nedeni resmî kaynakta doğrulanamadı.
- val küçük: sınıf başına val nesnesi 49 (diningtable) ile 785 (person) arasında. Sınıf başına val AP'si
  gürültülü olacak; seçim toplu mAP ile yapılmalı. Seed gürültüsü K3'te ölçülmeli.
- Boyut kovaları kutu alanına göre (COCO maske alanı kullanır). 640 ölçeğindeki dağılım letterbox
  yaklaşımıyla hesaplandı (Ultralytics eğitimde mosaic/scale artırımıyla farklı ölçekler görür).
- Yakın-kopya denetimi algısal ve eşik tabanlı (dHash ≤4, NCC ≥0,95). Kırpılmış veya yeniden kodlanmış kopyalar
  tamamen dışlanamaz, ancak ≥0,99 NCC ve bayt eşleşmesi 0.
- `splits/*.txt` mutlak yol içeriyor: WS taşınırsa `voc_split_denetim.py` yeniden koşturulmalı (seed=0 ile aynı
  kanonik sha256'ları vermeli). `voc_dev.yaml` ve `voc_test.yaml` içindeki `path` de mutlak.
