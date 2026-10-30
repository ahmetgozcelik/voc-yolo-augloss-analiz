# Gerçek zamanlı YOLO ince ayarında artırma ve kayıp müdahalelerinin çok tohumlu değerlendirmesi

Bu depo, aşağıdaki makalenin kodunu, ön-kayıt belgelerini, tohum düzeyi sonuçlarını, bölme listelerini, koşum kayıtlarını
ve genişletilmiş ekini içerir.

> Ahmed Gürkan Özçelik, Ufuk Asıl. *Gerçek Zamanlı YOLO Modellerinin İnce Ayarında Artırma ve Kayıp Müdahalelerinin Çok
> Tohumlu Değerlendirmesi: PASCAL VOC ve KITTI Üzerine Bir İnceleme* (İngilizce başlık: *A Multi-Seed Evaluation of
> Augmentation and Loss Interventions in Fine-Tuning Real-Time YOLO Detectors: A Study on PASCAL VOC and KITTI*).
> Ostim Teknik Üniversitesi. DOI: TODO (Zenodo arşivi: TODO).

## Özet

COCO ön-eğitimli YOLO11n'in PASCAL VOC 07+12 ince ayarında yedi eğitim-zamanı müdahalesi (çok ölçekli eğitim, ölçek
genliği, Inner-CIoU, kutu kazancı, Varifocal sınıflandırma kaybı, kutu-tabanlı copy-paste ve dozu) ile eğitimsiz çıkarım
seçenekleri (NMS eşiği, Soft-NMS, test çözünürlüğü, TTA, WBF), aynı kartlar KITTI'de YOLO11s ve YOLO26n ile, önceden
yazılmış karar kurallarına bağlı, üç tohumlu ve kilitli test kümeli bir protokolle değerlendirildi. VOC'ta yedi
müdahalenin hiçbiri doğrulamada kabul eşiğini geçmedi (0/7); KITTI'de on iyileşme sınamasından biri geçti (1/10) ve bu
kabul kontrol noktası seçimine duyarlıydı. Makale, küçük kazanç iddialarının tipik bir üç tohumlu ince ayar bütçesinde
gürültüden ayrılamadığını gösterir ve dört gürültü kaynağını belgeler: tohum varyansı, doğrulama örneklemesi, tek
tohumlu ön-eleme ve kontrol noktası seçimi. Sonuçlar, ayarlanmamış varsayılan reçeteye karşı ayarsız tek noktalı
müdahaleler için geçerlidir. Ayrıntı ve tüm çekinceler makaledir (`paper/main.pdf`).

## Klasörler

| Klasör | İçerik |
|---|---|
| `paper/` | Makale kaynağı (`main.tex`, makale içi ek `ek_ana.tex`), genişletilmiş ek (`ek.tex`, `ek_A.tex`–`ek_D.tex`), `tables/`, `figures/` (+ `figures/data/`), `references.bib`, derlenmiş `main.pdf` ve `ek.pdf` |
| `preregistration/` | Hipotez kartları ve VOC karar kuralları (`voc/`), K5 test planı, makullük çıpası; KITTI ve H05 kural belgeleri (`kitti_ve_h05/`), kuyruk beklenti dosyaları (`beklenti/`), SHA256 kaydı |
| `src/` | Müdahale yamaları ve birim testleri (`varyantlar/`), eğitim/değerlendirme/analiz betikleri (`deney_betikleri/`), tablo üreticileri ve sayı denetimleri (`tablo_ureticileri/`), şekil üreticileri (`sekil_ureticileri/`) |
| `results/` | Koşum başına metrik dosyaları (`runs/<koşum>/metrics.json`), analiz çıktıları (`sonuclar/`), tablo ve şekil girdileri (`evidence.json`) |
| `splits/` | VOC eğitim/doğrulama/test listeleri (`voc/`), KITTI sahne-grubu bölmesi, test grupları, `DontCare` bölgeleri ve SHA256 özeti (`kitti/`) |
| `logs/` | Deney günlüğü (`deney-gunlugu.md`), koşum yapılandırmaları (`runs/<koşum>/config.json`), koşum kuyrukları (`kuyruk/`) |
| `env/` | `requirements.txt`, kurulum ve ortam bilgisi, Ultralytics sürüm sabitlemesi |

## Tablo ve şekil → dosya eşlemesi

Makale (`paper/main.pdf`):

| Öğe | Kaynak dosya | Üretici / girdi |
|---|---|---|
| Tablo 1 (hipotez kartları) | `paper/main.tex` | `preregistration/voc/H01.md`–`H08.md` |
| Tablo 2 (kanıt düzeyi) | `paper/main.tex` | makale metni (iddia denetimi kodları) |
| Tablo 3 (taban model makullüğü) | `paper/tables/T1_replikasyon.tex` | `src/tablo_ureticileri/build_tables.py` ← `results/evidence.json` |
| Tablo 4 (VOC ana tablo) | `paper/tables/T_voc_ana.tex` | `src/tablo_ureticileri/build_tables.py` |
| Tablo 5 (KITTI test) | `paper/tables/T7_kitti_test.tex` | `src/tablo_ureticileri/build_ek_tables.py` |
| Şekil 1 (protokol) | `paper/figures/f0_protokol.tex` | elle yazılmış TikZ |
| Şekil 2 (Δ ve aralıklar) | `paper/figures/f1_nokta_aralik.tex` | `src/sekil_ureticileri/f1_nokta_aralik.py` |
| Şekil 3 (doğruluk–gecikme) | `paper/figures/f3_pareto.tex` | `src/sekil_ureticileri/f3_pareto.py` |

Genişletilmiş ek (`paper/ek.pdf`): Tablo A1 `T6_deney_kaydi.tex`; Şekil A1 `f4_butce.tex`; Tablo B1 `T_ekb_mutlak.tex`;
Tablo B2 `T_ekb_test_ikincil.tex`; Tablo B3 `T4_h07_gecikme.tex`; Şekil B1 `f2_hata_turu.tex`; Şekil B2 `f5_doz.tex`;
Şekil B3 `f6_nitel.tex`; Tablo C1–C5 `T8_kitti_ikincil.tex`, `T9_kitti_epoch.tex`, `T10_kitti_test_ek.tex`,
`T11_kitti_sinif.tex`, `T12_kitti_saglamlik.tex`; Şekil C1 `f7_best_last.tex`.

## Yeniden üretim

1. Ortam: Python 3.11, `pip install -r env/requirements.txt` (Ultralytics 8.4.165, commit `004db34`; ayrıntı
   `env/ultralytics-kurulum.md`). Koşumlar tek NVIDIA RTX 5050 (8 GB) ve Windows 10'da yapıldı.
2. Veri: PASCAL VOC 2007/2012 ve KITTI 2D nesne tespiti resmî sitelerinden indirilmelidir (bu depoda görüntü yoktur).
   Bölmeler `splits/` altındadır; VOC dönüşümü `src/deney_betikleri/veri/`, KITTI hazırlığı
   `src/deney_betikleri/ek_deney/kitti_hazirla.py` iledir.
3. Eğitim ve değerlendirme: her koşumun tam komutu ve parametreleri `logs/runs/<koşum>/config.json` ve
   `logs/kuyruk/*.jsonl` dosyalarındadır; müdahaleler `src/varyantlar/<kart>/kos.py` üzerinden çalışır. Bir VOC koşumu
   ≈3 GPU saati sürer.
4. Tablolar ve şekiller: `src/tablo_ureticileri/build_tables.py` ve `build_ek_tables.py`, `results/evidence.json`
   dosyasından `paper/tables/` altındaki tabloları; `src/sekil_ureticileri/*.py` şekil verilerini üretir. Makale
   `paper/` içinde `latexmk -pdf main.tex; latexmk -pdf ek.tex; latexmk -pdf main.tex` ile derlenir.

Bilinen sınırlar: Betikler özgün çalışma alanı düzenine göre yazılmıştır; yerel mutlak yollar bu depoda `<WS>` (deney
çalışma alanı), `<MW>` (makale çalışma alanı), `<REPO>` ve `<YEREL_YOL>` yer tutucularıyla değiştirilmiştir ve yeniden
çalıştırmadan önce yerel yollarla doldurulmalıdır. Nitel panelin (Şekil B3) görüntü karoları VOC test görüntülerinden
türediği için depoda yoktur; `ek.tex` bu şekil olmadan derlenmez, şekil `paper/ek.pdf` içinde yer alır.

## Depoda olmayanlar

Veri kümesi görüntüleri (VOC, KITTI), model ağırlıkları (`.pt`), ham tahmin dökümleri (`*preds*.json`, `.npz`), büyük ham
koşum klasörleri ve önbellekler. Bunlar makaledeki sayıların yeniden hesaplanması için gerekmez: sayılar
`results/runs/*/metrics.json`, `results/sonuclar/` ve `results/evidence.json` dosyalarından türetilmiştir.

## Ön-kayıt ve zaman damgaları

Hipotez kartları ve karar kuralları koşumdan önce yazılmıştır; tek istisna, SÜRDÜR (özgün kayıtta TUT) kuralının tur 2'de,
H01–H03 sonuçlarından sonra donmasıdır. Bu bağımsız olarak doğrulanamaz: dosya damgaları kriptografik değildir, kartlar
herkese açık bir ön-kayıt deposuna (OSF, Zenodo) yüklenmedi, sürüm denetimi 2026-10-16'da tek toplu kayıtla başladı ve
KITTI kural belgesi yalnız SHA256 ile deney günlüğünde kayıtlıdır (`preregistration/SHA256-KAYDI.md`). Altı VOC kart
dosyası (H01, H02, H04, H05, H06, H08) 2026-10-03'te, ön-eleme koşumlarından ve kilitli test açılışından sonra toplu
olarak yeniden yazıldı; koşum öncesi varlıkları yalnız dolaylı kayıtlarla (kuyruk beklenti ve sıralama dosyaları, deney
günlüğü, kod ve sayaç dosyalarının zamanları) desteklenir. Önceden yazılmış kurallardan dokuz sapmanın tamamı makalenin
Sınırlılıklar bölümünde listelenmiştir.

## Alıntı

`CITATION.cff` dosyasına bakın.

## Lisans

Kod (`src/` ve betikler): GNU AGPL-3.0 (`LICENSE`); müdahale yamaları AGPL-3.0 lisanslı Ultralytics koduna dayanır.
Metin, tablolar, şekiller, hipotez kartları ve diğer belgeler: CC BY 4.0 (`LICENSE-docs`). Veri kümeleri kendi
lisanslarına tabidir (KITTI: CC BY-NC-SA 3.0).

## Yapay zekâ kullanımı

Yapay zekâ kullanımı: Deney tasarımı ve kodu, analiz, literatür taraması ve metin hazırlığında yazarların denetiminde
büyük dil modeli tabanlı araçlar kullanılmış; tüm sonuçlar yazarlarca doğrulanmıştır ve sorumluluk yazarlara aittir.
