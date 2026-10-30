# Literatür Çıpası — Konu 2.1 (PASCAL VOC · YOLO11n · augmentation + loss)

*K1.5 çıpa modu · literatur-arastirmacisi · 2026-09-29 · BibTeX: `notlar/references-cekirdek.bib`*

Sabit protokol (şef kararı): COCO ön-eğitimli `yolo11n.pt` → VOC07+12 trainval'a ince ayar (fine-tuning;
trainval'ın bir kısmı model seçimi için val olarak ayrılır) → VOC2007 test (4952 görüntü, kilitli)
değerlendirme; 640 px, 100 epoch, Ultralytics varsayılan hiperparametreler; `ultralytics==8.4.165`.

> **Özet uyarı:** Literatürde bu protokolün birebir aynısı için bildirilmiş sayı **bulunamadı**. Bulunamayan
> kombinasyon: YOLO11n + COCO ön-eğitimi + 100 epoch + 640 px + VOC07 test. Aşağıdaki çıpa, bu protokolü
> **alttan ve üstten sınırlayan** dört bağımsız kaynaktan kuruldu (§2). Güven düzeyi: **düşük-orta**.

---

## 1. Rakibin kaynağı

| Kalem | Değer | Doğrulama |
|---|---|---|
| Yazılım atfı | Jocher, G. & Qiu, J. (2024). *Ultralytics YOLO11*, v11.0.0, AGPL-3.0 → `jocher_2024_yolo11` | Resmî doküman BibTeX'i; **DOI yok** ("pending"). Zenodo DOI'si bulunamadı. |
| YOLO11n resmî COCO değeri | mAP50-95 **39.5** (COCO val2017, 640), 2.6 M param., 6.5 GFLOPs, T4 TensorRT10 1.5 ms | docs.ultralytics.com/models/yolo11 |
| Resmî teknik rapor | Ultralytics'in YOLO11 için yayımladığı hakemli/arXiv bir teknik rapor **bulunamadı**. | — |
| Üçüncü taraf genel bakış | Khanam & Hussain (2024), "YOLOv11: An Overview of the Key Architectural Enhancements", arXiv:2410.17725 → **var, doğrulandı** (`khanam_2024_yolov11overview`). Ultralytics'in yazdığı bir belge değil; mimari (C3k2, SPPF, C2PSA) için ikincil kaynak olarak kullanılabilir. | arxiv.org/abs/2410.17725 |
| Resmî depo | https://github.com/ultralytics/ultralytics | — |
| Env sürümü → git etiketi | `v8.4.165` etiketi **mevcut**. Sürüm notu: "ultralytics 8.4.165 Stop installing top-level tests package from the wheel (#26408)". Commit `004db34b12f65fa5a160de2b1090f6943106bd35` → `ultralytics_2026_v84165` | github.com/ultralytics/ultralytics/releases/tag/v8.4.165 |
| YOLOv8 atfı (karşılaştırma için) | Jocher, Chaurasia, Qiu (2023), *Ultralytics YOLOv8* v8.0.0 → `jocher_2023_yolov8`; YOLOv8n COCO 37.3 | docs.ultralytics.com/models/yolov8 |

`KURULUM.md` için: `ultralytics==8.4.165` @ `004db34b12f6…` (etiket `v8.4.165`).

---

## 2. REPLİKASYON ÇIPASI (VOC07+12 → VOC07 test)

### 2.1 Bulunan değerler (her satır kaynağında okunan tablodan; hiçbiri bizim protokolle birebir aynı değil)

| # | Kaynak | Model | Başlatma | Epoch | imgsz | Seçim / değerlendirme | <e-posta> | <e-posta>:0.95 |
|---|---|---|---|---|---|---|---|---|
| A1 | Zhang ve ark. 2025, *Processes* 13(5):1556 (ADL-YOLO), Tablo 2 + Tablo 10 | **YOLO11n** | **Sıfırdan** ("no official pretrained weights were utilized") | 500 (patience 100) | 640 | Batch 8, SGD lr0 0.01, close_mosaic 0; VOC07 test "for validation" | **81.5** | **61.3** |
| A2 | Lin ve ark. 2025, arXiv:2512.23273 (YOLO-Master), Tablo 1 | **YOLO11-N** | Belirtilmemiş (makale 600 epoch, 640, SGD, cosine, batch 256 genel ayar veriyor → büyük olasılıkla sıfırdan) | 600 (genel ayar) | 640 | Belirtilmemiş | **81.2** | **61.0** |
| A3 | Zhang ve ark. 2025 (ADL-YOLO), Tablo 10 | YOLOv8n | Sıfırdan | 500 | 640 | A1 ile aynı | 80.9 | 60.2 |
| B1 | El Hamdouni ve ark. 2025, *Information* 16(10):871, Tablo 3–4, 8–9 | **YOLOv8n** | **Belirtilmemiş** ("All other hyperparameters remained at the default YOLOv8n values") | 200 (patience 50) | 640 | Ultralytics 8.1.6, batch 16, SGD lr0 0.01, V100; 07 test "validation or test set" | **83.6** | **63.5** |
| C1 | Lin ve ark. 2026, arXiv:2608.07051 (YOLO-PEFT), Tablo 1, Ek H | **YOLO11s** (n değil) | "Full-SFT" = tam ince ayar (ön-eğitimli başlangıç ima ediliyor, kaynak açıkça "COCO" yazmıyor) | ≤600 (patience 100), best checkpoint | 640 | SGD lr0 0.01, cosine, close_mosaic 10; **best checkpoint VOC07 test üzerinde seçiliyor** (yazarlar açıkça belirtiyor) | **83.44** | **64.28** |
| C2 | Aynı, Tablo 7 (tanısal matris) | **YOLO11n** | Full-SFT | 300 | **320** | Aynı seçim kuralı; tek seed | — | 58.8 |
| C3 | Aynı, Ek H / Tablo 18 | 3 seed (0, 28, 42) tekrarları | — | — | — | — | — | **seed içi örneklem SD ≤ 0.006** (mAP50-95, kesir) |

Ek bağlam (çıpa değil, üst tavan): Ultralytics `hyp.VOC.yaml` başlığı — YOLOv5m6, 512 px, 50 epoch, VOC07 test
üzerinde **hiperparametre evrimi** ile: <e-posta> 0.91286, <e-posta>:0.95 0.72664 (test üzerinde ayar → iyimser,
yalnız "orta-boy model tavanı" olarak anılabilir).

**Bağımsız reprodüksiyon kontrolü:** A1 ve A2 birbirinden bağımsız iki ekipten (Guangdong Petrokimya Üniv. vs.
Tencent Youtu) YOLO11n için 61.3 / 61.0 (mAP50-95) ve 81.5 / 81.2 (mAP50) veriyor. Fark 0.3 puan. Bu,
"uzun eğitim, ön-eğitimsiz" rejim için ±0.3'lük bir reprodüksiyon aralığı demek.

### 2.2 Bizim protokole ölçekleme mantığı (sayı üretmiyor; aralığı gerekçelendiriyor)
- **Alt sınır (≈61.0 / 81.2):** Sıfırdan 500–600 epoch'luk YOLO11n (A1, A2). Aynı modelde COCO ön-eğitimi,
  VOC'un 20 sınıfının hepsini içerdiği için beklentiyi yükseltir. 100 epoch ise beklentiyi düşürür. Sıfırdan
  uzun eğitim ile "ön-eğitimli ama kısa" eğitim literatürde yan yana ölçülmedi. Bu yüzden alt sınır ihtiyatlı
  tutuldu. Sıfırdan uzun eğitimden anlamlı ölçüde kötü bir sonuç kurulum hatası şüphesi doğurur.
- **Üst sınır (≈63.5 / 83.6):** Nano ölçekte 640 px'te bulunan en yüksek değer (B1, YOLOv8n, 200 epoch).
  YOLO11n sıfırdan eğitim rejiminde YOLOv8n'den +0.6 (mAP50) / +1.1 (mAP50-95) önde (A1 ile A3 farkı). Buna
  karşılık bizim 100 epoch'umuz B1'in 200 epoch'unun yarısı. Bu iki etki kabaca birbirini götürüyor.
- **Nano-üstü tavan (64.28 / 83.44):** C1, YOLO11s (≈3.6× parametre). Nano modelin bunu belirgin biçimde
  geçmesi beklenmez. Geçerse bu bir **sızıntı/protokol bayrağı** sayılır.
- **Aşağı yönlü protokol farkları (R3 belgesi için):** (i) Literatür kaynaklarının hepsi Ultralytics
  `VOC.yaml`'ı kullanıyor. Bu dosyada `val` = `images/test2007`, dolayısıyla best.pt **test üzerinde
  seçiliyor** (C1'de açıkça yazıyor, A1/B1 "07 test for validation" diyor) ve bu iyimser bir yanlılık üretir.
  Bizde seçim trainval'dan ayrılan val üzerinde yapılıyor. (ii) Val ayrıldığı için eğitim kümesi
  16 551'den küçük. İkisi de bizim baseline'ı literatürün **biraz altına** iter. Etkinin büyüklüğü
  literatürde ölçülmemiş (**bulunamadı**).

### 2.3 `replikasyon_kapisi.py` argümanları (hazır)

Birim **kesir** (Ultralytics `metrics/mAP50-95(B)` 0–1 aralığında yazar). Koşucu yüzde yazıyorsa tüm sayılar ×100.

```bash
# <e-posta>:0.95 (ana metrik)
python .claude/skills/konudan-makale/scripts/replikasyon_kapisi.py --kok <WS> \
  --metrik map50_95 --yon max --lit 0.610 --lit-ust 0.635 --tolerans 0.012 \
  --kaynak "Alt: YOLO11n sıfırdan VOC07+12->07test 0.610-0.613 (Lin 2025 arXiv:2512.23273 Tab.1; Zhang 2025 Processes 13:1556 Tab.10). Üst: YOLOv8n 200ep 640 0.635 (El Hamdouni 2025 Information 16:871 Tab.4). Tol=2xSD, SD<=0.006 (Lin 2026 arXiv:2608.07051 Tab.18)"

# <e-posta>
python .claude/skills/konudan-makale/scripts/replikasyon_kapisi.py --kok <WS> \
  --metrik map50 --yon max --lit 0.812 --lit-ust 0.836 --tolerans 0.012 \
  --kaynak "Alt: YOLO11n sıfırdan 0.812-0.815 (Lin 2025 arXiv:2512.23273; Zhang 2025 Processes 13:1556). Üst: YOLOv8n 200ep 0.836 (El Hamdouni 2025 Information 16:871). Tol=2xSD (Lin 2026 arXiv:2608.07051 Tab.18; mAP50 için SD bildirilmemiş, mAP50-95 SD'si varsayıldı)"
```

Kabul bölgeleri (betik mantığıyla): map50_95 → YEŞİL [0.598, 0.647], SARI [0.586, 0.659], dışı KIRMIZI.
map50 → YEŞİL [0.800, 0.848], SARI [0.788, 0.860].

**Tolerans gerekçesi:** Bildirilen tek seed-varyans ölçüsü YOLO-PEFT'teki 3-seed örneklem SD'si
(≤ 0.006 mAP50-95). Kural gereği ~2×SD = **0.012** alındı. Bağımsız reprodüksiyon aralığı (A1 ile A2
arasında 0.003) bundan dar olduğu için 2×SD daha ihtiyatlı seçenek. mAP50 için SD bildirilmemiş, aynı
değer kullanıldı (varsayım, açıkça işaretli). SARI/KIRMIZI çıkarsa sonucu geçirmek için tolerans genişletilmez.
Önce §2.2'deki protokol farkları belgelenir.

**Güven düzeyi: düşük-orta.** Gerekçeler: (1) protokol birebir eşleşmiyor (ön-eğitim ve epoch farkı);
(2) B1'de başlatma belirtilmemiş; (3) kaynakların üçü hakemsiz ön-baskı veya MDPI; (4) literatür seçimi test
üzerinde yapıyor. Öneri: K3'te baseline SARI çıkarsa bu, çıpanın belirsizliğiyle uyumlu ve kullanıcı onayı
alınarak ilerlenebilir. KIRMIZI-üst (>0.659) çıkarsa önce val/test ayrımı ve `difficult` dönüşümü denetlenir.

---

## 3. Değerlendirme protokolü ve tuzaklar (alan-protokolleri §02 ile çapraz)

1. **AP tanımı:** Ultralytics `ultralytics/utils/metrics.py::compute_ap` (yerel env 8.4.165'te okundu)
   `method="interp"` kullanıyor, yani **101 noktalı COCO-tarzı interpolasyon** (hassasiyet zarfı +
   `np.trapezoid`). `mAP50-95` 10 IoU eşiğinin (0.50:0.05:0.95) ortalaması. VOC2007'nin resmî ölçüsü ise
   **11 noktalı interpolasyonlu AP** (Everingham 2010). 2010 sonrası VOC tüm-nokta alanına geçti
   (Everingham 2015). Sonuç: eski VOC makalelerindeki "mAP" sayıları (ör. YOLOv3+BoF 83.68, Zhang 2019;
   devkit ölçüsü) Ultralytics mAP50 ile **doğrudan kıyaslanamaz**. Makalede "Ultralytics 101-nokta <e-posta>"
   diye adlandırılmalı. İki ölçü arasındaki farkın büyüklüğü için güvenilir bir kaynak **bulunamadı**.
2. **`difficult` nesneler:** Ultralytics `VOC.yaml` dönüştürücüsü `difficult==1` nesneleri etiket
   dosyasına **hiç yazmıyor** (yerel `VOC.yaml` satır 72; docs: "The automatic converter skips objects
   flagged difficult"). Resmî VOC değerlendirmesi ise bu nesneleri **yok sayar**: onlarla eşleşen tespit ne
   TP ne FP sayılır ("Objects marked as 'difficult' are currently ignored by the evaluation", VOC2007
   devkit §4.4). Ultralytics'te bu nesneler GT'de bulunmadığı için onları doğru bulan tespitler **FP**
   sayılır ve mAP resmî devkit'e göre aşağı çekilir. Ayrıca bu nesneler eğitimde de görülmez. Bu fark
   makalede açıkça yazılmalı. Etkinin büyüklüğü için kaynak **bulunamadı**.
3. **Test sızıntısı:** Varsayılan `VOC.yaml`'da `val: images/test2007`. Değiştirilmeden kullanılırsa
   best.pt ve erken durdurma test üzerinden seçilir. VOC devkit bunu yasaklıyor: "all parameter tuning must
   be conducted using the training and validation data alone" (devkit §4.4). Şefin kararı (trainval'dan val
   ayırma) doğru. Özel bir YAML yazılmalı; `val:` yeni ayrılan val listesini, `test:` test2007'yi göstermeli.
   07 test yalnız `model.val(split='test')` ile ve kilit kaydıyla koşulmalı. YOLO-PEFT test-seçimini açıkça
   kabul ediyor ("we … do not present the absolute scores as once-only blind test estimates").
4. **COCO ön-eğitimi VOC sınıflarını kapsıyor:** COCO'nun 80 kategorisi VOC'un 20 sınıfının tamamını içerir
   (aeroplane→airplane, sofa→couch, tvmonitor→tv, diningtable→dining table, pottedplant→potted plant,
   motorbike→motorcycle; diğerleri aynı adla) (Lin 2014). Dolayısıyla "transfer" aslında sınıf-örtüşmeli
   bir alan uyarlaması. "Yeni sınıflara genelleme" iddiası kurulamaz. Strateji 5'teki (transfer ablasyonu)
   bulgu "COCO'nun VOC'a katkısı" diye çerçevelenmeli. COCO train ile VOC07 test arasında görüntü tekrarı
   olup olmadığı **doğrulanmadı**; iki veri seti de Flickr kaynaklı. Tekrar olduğu iddiası da yok, olmadığı
   iddiası da yok. Kısıt olarak yazılmalı.
5. **Değerlendirme ayarları sabit tutulmalı:** Ultralytics val varsayılanları `conf=0.001`, `iou=0.7`,
   `max_det=300` (yerel `default.yaml`). Bu ayarlar mAP'i ve NMS süresini etkiler. RT-DETR Tablo 1:
   YOLOv8'de conf 0.001 ile 0.05 arasında AP 52.9'dan 51.2'ye, NMS süresi 2.36 ms'den 1.06 ms'ye iniyor
   (T4, TensorRT FP16). Tüm varyantlarda aynı değerler kullanılmalı.
6. **Rapor edilecek sayı:** `results.csv`'nin son satırı değil; **best.pt'nin val değeri**, ardından
   kilitli 07 test üzerinde tek bir değerlendirme (alan-protokolleri §02). TTA ve WBF sonuçları ayrı
   satırda verilmeli.
7. **FPS ölçüm protokolü (önerilen, kaynaklı):**
   - RT-DETR (Zhao 2024, §3.2): T4 GPU, **TensorRT FP16**, **NMS dahil uçtan uca** (EfficientNMS eklentisi),
     I/O ve MemoryCopy hariç, COCO val2017 üzerinde ortalama. Yazarlara göre YOLO makaleleri hızı
     genellikle **NMS hariç** raporluyor.
   - YOLOX (Tablo 2): V100, FP16, **batch=1**, "latency … measured without post-processing".
   - Ultralytics resmî YOLO11 tablosu: T4 TensorRT10 (YOLO11n için 1.5 ms). NMS dahil mi, belgede açık
     değil (**doğrulanamadı**).
   - **Öneri:** RTX 5050 üzerinde batch=1, 640 px, FP16 (`half=True`) ve FP32 ayrı ayrı ölçülmeli.
     Isınma koşuları (ör. ilk 50 iterasyon) dışlanmalı. `preprocess / inference / postprocess(NMS)` ms
     ayrı raporlanmalı, "uçtan uca" toplam ayrıca verilmeli. conf=0.001 değeri yazılmalı. Arka uç (PyTorch
     eager / ONNX / TensorRT) her satırda belirtilmeli. Donanım farklı olduğu için T4 sayılarıyla doğrudan
     kıyas yapılmamalı. `torch.compile` kullanılırsa derleme süresi ayrı yazılmalı.
8. **Seed/determinizm:** `deterministic=True`, `seed` varsayılan 0. Yerel `default.yaml` doğrulandı.
   Konu kartındaki {0, 1, 2} seed'leri açıkça geçirilmeli.

---

## 4. Manipülasyonlar için literatür etki büyüklükleri → hipotez kartları / `beklenti.json` tavanları

Referans: Ultralytics varsayılanlarında (yerel `default.yaml`) `mosaic=1.0`, `mixup=0.0`,
`copy_paste=0.0`, `close_mosaic=10`, `box=7.5`, `cls=0.5`, `dfl=1.5`, `optimizer=auto`, `cos_lr=False`,
`fliplr=0.5`, `scale=0.5` tanımlı. EMA eğitimde zaten açık. Box kaybı `CIoU=True` (yerel
`utils/loss.py:134`). DFL ve TAL (Feng 2021) atamasıyla BCE sınıf kaybı da varsayılan olarak geliyor. Bu
yüzden literatürdeki "IoU→CIoU" veya "DFL ekle" kazançları **baseline'da zaten içerilmiş** durumda.

Δ değerleri mAP50-95 puanı (0–100 ölçeği) üzerinden yazıldı. "Görülmemiş eşiği" kural 3'e göre belirlendi:
beklenen aralığın üst ucunun ~2 katı veya literatürdeki en yüksek kazancın üstü.

| Manipülasyon (aile) | Literatürde bildirilen etki (koşul) | Beklenen Δ (YOLO11n, ön-eğitimli, 100 ep) | Görülmemiş iyi eşiği |
|---|---|---|---|
| **MixUp** (A3) | +1.54 mAP (VOC devkit ölçüsü, 0.5) YOLOv3-416, VOC07 test (Zhang 2019, Tab.3); YOLOX-L COCO **+0.9 AP**; YOLOX-Nano COCO **−1.3 AP** ("better to weaken the augmentation for small models", Ge 2021, Tab.5) | **−0.5 … +0.3** (nano model → nötr/negatif) | > +1.0 |
| **Mosaic olasılığı / ölçek jitter'ı** (A3) | Mosaic+MixUp birlikte +2.4 AP (YOLOX, sıfırdan COCO, Tab.2); YOLOX-Nano'da mosaic ölçeğini [0.1,2]→[0.5,1.5] daraltmak + MixUp'ı kaldırmak 24.0'dan 25.3'e çıkarıyor (+1.3); tüm augmentation kaldırılırsa −15.99 mAP (YOLOv3 VOC, Zhang 2019) | Ön-eğitimli ince ayarda ±0.5; `scale` daraltma +0…+0.5 | > +1.0 |
| **close_mosaic** (A3/A11) | YOLOX son 15 epoch'ta kapatıyor; izole Δ **bildirilmemiş** | 0 … +0.3 | > +0.6 |
| **CopyPaste** (A3) | Ghiasi 2021: COCO'da +1.5 box AP (güçlü büyük modeller, LSJ, uzun eğitim; abstract); YOLOX-L'de CopyPaste 49.4, MixUp 49.5, ekstra aug yok 48.6 (+0.8) | **Varsayılan kurulumda 0 (etkisiz, aşağıdaki tuzağa bakın)**; maskeli uygulamada 0 … +0.3 | > +0.8 |
| **Uzun eğitim (+EMA)** (A11) | VOC'ta nano modelde 100→200/300 epoch'u izole eden bir kaynak **bulunamadı**. Dolaylı: YOLOv8n 200 ep 63.5 (B1). EMA zaten açık | +0.3 … +1.0 (maliyet 2–3×, bütçeye dikkat) | > +2.0 |
| **Omurga büyütme → YOLO11s** (A4) | YOLO11s ince ayar 64.28 / 83.44 (C1, ≤600 ep, test-seçimli) | +2 … +4 (katkı değil, bağlam satırı) | — |
| **Cosine LR** (A11) | +0.46 mAP YOLOv3 VOC (Zhang 2019, Tab.3); Faster R-CNN VOC +1.82 | 0 … +0.3 | > +0.8 |
| **Label smoothing** (A5) | +0.45 mAP YOLOv3 VOC (Zhang 2019) | −0.2 … +0.2 (BCE+TAL yumuşak hedeflerle çakışır) | > +0.6 |
| **Soft-NMS** (A8) | VOC2007'de +1.7 (R-FCN ve Faster R-CNN, <e-posta>), COCO'da +1.3 / +1.1 (Bodla 2017, abstract; iki aşamalı dedektörler) | 0 … +0.4 (conf 0.001 + max_det 300 ile tek aşamalı dedektör) | > +1.0 |
| **DIoU-NMS** (A8) | YOLOv3 VOC07: 49.21'den 49.32'ye (**+0.11** AP@[.5:.95]); SSD +0.19 (Zheng 2020, Tab.1–2) | 0 … +0.2 | > +0.5 |
| **WBF** (A8) | Solovyev 2021: model **ensemble**'ları için (Open Images / COCO yarışmaları). Tek model + TTA için VOC sayısı **bulunamadı** | TTA ile birlikte +0.2 … +0.8 | > +1.5 |
| **TTA** (Ultralytics `augment=True`: ölçekler [1, 0.83, 0.67] + yatay çevirme; yerel `nn/tasks.py:567`) | YOLOv5x COCO 0.504 → 0.516 (+1.2 mAP50-95), **ancak TTA koşusu 832 px, baseline 640 px (karışık etki)**; gecikme ~3.6× (Ultralytics TTA dokümanı) | +0.3 … +1.0 | > +2.0 |
| **Transfer ablasyonu (sıfırdan vs COCO)** (A4/A12) | Sıfırdan 500–600 ep YOLO11n 61.0–61.3 (A1, A2). Sıfırdan 100 epoch için YOLO11n kaynağı **bulunamadı**. YOLO26n 100 ep (başlatma belirtilmemiş) 49.9 (MambaPSA, arXiv:2607.12681, çıpa değil) | Ön-eğitimli − sıfırdan(100 ep) ≈ büyük pozitif (kazanç değil, **bulgu**) | — |
| **Box kaybı: SIoU / EIoU / WIoU / Inner-IoU** (A5) | Taban CIoU. YOLOv3 VOC07 AP@[.5:.95]: IoU 46.57 → GIoU 47.73 (+1.16) → DIoU 48.10 → **CIoU 49.21** (IoU'ya göre +2.64, GIoU'ya göre +1.48) (Zheng 2020, Tab.1); SSD VOC'ta CIoU'nun IoU'ya kazancı +0.43. WIoU: YOLOv7 COCO AP75 53.03 → 54.50 (+1.47, abstract). SIoU, EIoU, Inner-IoU için abstract'ta VOC sayısı yok (**bulunamadı**) | CIoU'ya göre −0.3 … +0.5 | > +1.0 |
| **NWD (küçük nesne)** (A5) | AI-TOD'da standart ince ayara göre +6.7 AP (abstract; çok küçük nesneler). VOC'ta örneklerin %14'ü <50 px (El Hamdouni 2025, Tab.1) | −0.3 … +0.3 (VOC küçük-nesne ağırlıklı değil) | > +0.8 |
| **Sınıflandırma kaybı: Focal / VFL / QFL** (A5) | Focal loss YOLOv3'te "dropped our mAP about 2 points" (Redmon 2018, §4); VFL ve GFL'in kazançları farklı dedektör tabanlarında ölçülmüş, Ultralytics BCE+TAL tabanında kaynak **bulunamadı** | −0.5 … +0.3 | > +0.8 |
| **Kayıp kazançları box/cls/dfl** (A1/A5) | Literatürde izole Δ **bulunamadı**. YOLOv5 VOC evrimi farklı kazançlar buluyor, ama bu evrim test üzerinde yapılmış ve farklı mimaride (`hyp.VOC.yaml`) | ±0.3 | > +0.8 |

**Kritik tuzak: CopyPaste VOC'ta çalışmıyor.** Ultralytics `CopyPaste` sınıfı (yerel `data/augment.py:1981`)
`if len(labels["instances"].segments) == 0 or self.p == 0: return labels` kontrolünü yapıyor. VOC tespit
etiketleri yalnız kutu içerdiği için `copy_paste>0` verilse bile **hiçbir şey yapmaz**. Dokümandaki
`default.yaml` açıklaması da aynısını söylüyor: "segment/semantic/obb copy-paste". Bu ayar ızgaraya
konursa sıfır etkili sahte bir kol çıkar ve seed gürültüsü etki gibi görünebilir. Seçenekler: (a) kolu
çıkarmak; (b) VOC `SegmentationObject` maskeleri olan alt küme için segment etiketi üretmek (yalnız bir
alt kümede var, sayısı bu oturumda doğrulanmadı); (c) kutu tabanlı özel bir yapıştırma yazmak (rakip dışı
kod, R2 kuralı gereği `rakip/` dışında). Kart yazılmadan önce karar verilmeli.

**Hata-kütlesi ipucu (K3.5 için, kaynaklı):** A1 Tablo 11'de YOLO11n'in sınıf başına AP50 değerleri (sıfırdan
eğitim): pottedplant 53.1, chair 65.1, bottle 68.0, boat 74.8. Bunlar en zayıf sınıflar. Bizim baseline'da
da benzer bir dağılım beklenir. "En büyük hata kütlesi" analizi bu sınıflarla karşılaştırılabilir.

---

## 5. İlgili çalışmalar (kısa liste; künyeler `.bib`'de)

- **YOLO ailesi:** YOLOv1, Redmon ve ark. CVPR 2016 (`redmon_2016_yolo`); YOLOv3, arXiv:1804.02767
  (`redmon_2018_yolov3`); YOLOv4, arXiv:2004.10934 (`bochkovskiy_2020_yolov4`); YOLOX, arXiv:2107.08430
  (`ge_2021_yolox`); YOLOv7, CVPR 2023 (`wang_2023_yolov7`); YOLOv8, Ultralytics 2023 (`jocher_2023_yolov8`);
  YOLOv10, arXiv:2405.14458 / NeurIPS 2024 (`wang_2024_yolov10`); YOLO11, Ultralytics 2024
  (`jocher_2024_yolo11`) ve genel bakış (`khanam_2024_yolov11overview`).
- **Gerçek zamanlı DETR:** RT-DETR, Zhao ve ark., CVPR 2024, s. 16965–16974 (`zhao_2024_rtdetr`).
  RT-DETR-R50 COCO'da 53.1 AP ve T4'te 108 FPS elde ediyor (arXiv abstract). Bu çalışmada yalnız ilgili
  çalışma ve hız-protokolü referansı olarak kullanılıyor.
- **Veri setleri ve ölçüt:** VOC, Everingham ve ark. IJCV 2010 (`everingham_2010_voc`); retrospektif IJCV 2015
  (`everingham_2015_vocretro`); VOC2007 devkit (`everingham_2007_devkit`); COCO, Lin ve ark. ECCV 2014
  (`lin_2014_coco`).
- **VOC üzerinde YOLO11n/nano çıpaları:** `zhang_2025_adlyolo`, `lin_2025_yolomaster`,
  `elhamdouni_2025_yolov8nvoc`, `lin_2026_yolopeft`.
- **Augmentation / hileler:** `zhang_2019_bagoffreebies`, `zhang_2018_mixup`, `ghiasi_2021_copypaste`,
  `ultralytics_2026_yolov5tta`, `ultralytics_2026_yolov5hypvoc`.
- **Son-işlem:** `bodla_2017_softnms`, `solovyev_2021_wbf`, DIoU-NMS (`zheng_2020_diou`).
- **Kayıp:** `rezatofighi_2019_giou`, `zheng_2020_diou`, `gevorgyan_2022_siou`, `zhang_2022_eiou`,
  `tong_2023_wiseiou`, `zhang_2023_inneriou`, `wang_2021_nwd`, `lin_2017_focal`,
  `zhang_2021_varifocalnet`, `li_2020_gfl`, `feng_2021_tood` (TAL).

**Konumlandırma önerisi:** Katkı "yeni bir modül" değil, **protokol açısından dürüst bir ince ayar
çalışması** olarak kurulmalı. Dört dayanağı var: (i) literatürdeki VOC sayılarının çoğu test üzerinde
seçim yapıyor; biz val/test ayrımıyla "gerçek" baseline'ı veriyoruz (A12 bulgusu). (ii) Nano modelde güçlü
augmentation'ın (MixUp/Mosaic ölçeği) ters tepebildiği literatürde gösterilmiş (YOLOX-Nano −1.3); VOC
ince ayarında bu etkiyi ölçüyoruz. (iii) CIoU tabanına karşı yeni IoU kayıplarının küçük ve seed gürültüsü
içinde kalan kazançları 3 seed ile sınanıyor. (iv) TTA/WBF/Soft-NMS kazançları gecikme bedeliyle birlikte
ayrı satırlarda raporlanıyor. Gerçekçi hedef (+1–3 puan) literatürle uyumlu. +3'ün üzerindeki bir sonuç
yalnız TTA+ensemble ile açıklanabilir; tek başına eğitim-zamanı değişikliğiyle gelirse sızıntı açısından
denetlenmeli.

### Türkçe terim önerileri
ortalama kesinlik ortalaması (mean Average Precision, mAP) · kesişim/birleşim oranı (Intersection over
Union, IoU) · maksimum olmayanı bastırma (Non-Maximum Suppression, NMS) · yumuşak NMS (Soft-NMS) ·
ağırlıklı kutu füzyonu (Weighted Boxes Fusion, WBF) · test-zamanı artırma (Test-Time Augmentation, TTA) ·
veri artırma (data augmentation) · mozaik artırma (Mosaic) · karıştırma (MixUp) · kopyala-yapıştır
(Copy-Paste) · üstel hareketli ortalama (Exponential Moving Average, EMA) · ince ayar (fine-tuning) ·
ön-eğitim (pre-training) · sınırlayıcı kutu regresyon kaybı (bounding-box regression loss) · dağılım odak
kaybı (Distribution Focal Loss, DFL) · görev-hizalı atama (Task-Aligned Assignment, TAL) · gerçek zamanlı
(real-time) · gecikme (latency) · kare/saniye (frames per second, FPS).

---

## Kaynak listesi (URL · erişim tarihi)

| Kaynak | URL | Erişim |
|---|---|---|
| Ultralytics YOLO11 dokümanı (atıf, COCO değerleri) | https://docs.ultralytics.com/models/yolo11/ | 2026-09-29 |
| Ultralytics YOLOv8 dokümanı (atıf) | https://docs.ultralytics.com/models/yolov8/ | 2026-09-29 |
| ultralytics v8.4.165 sürüm etiketi | https://github.com/ultralytics/ultralytics/releases/tag/v8.4.165 | 2026-09-29 |
| Ultralytics VOC veri seti dokümanı (difficult atlanıyor, val=test2007) | https://docs.ultralytics.com/datasets/detect/voc | 2026-09-29 |
| Yerel kod: `VOC.yaml`, `utils/metrics.py`, `data/augment.py`, `nn/tasks.py`, `utils/loss.py`, `cfg/default.yaml` (8.4.165) | WS/env/.venv/Lib/site-packages/ultralytics/… | 2026-09-29 |
| Khanam & Hussain, arXiv:2410.17725 | https://arxiv.org/abs/2410.17725 | 2026-09-29 |
| YOLO-PEFT, arXiv:2608.07051 | https://arxiv.org/abs/2608.07051 · https://arxiv.org/pdf/2608.07051 | 2026-09-29 |
| YOLO-Master, arXiv:2512.23273 | https://arxiv.org/html/2512.23273 · https://github.com/Tencent/YOLO-Master | 2026-09-29 |
| ADL-YOLO, Processes 13(5):1556 | https://doi.org/10.3390/pr13051556 · https://mdpi-res.com/d_attachment/processes/processes-13-01556/article_deploy/processes-13-01556.pdf | 2026-09-29 |
| El Hamdouni ve ark., Information 16(10):871 | https://doi.org/10.3390/info16100871 · https://mdpi-res.com/d_attachment/information/information-16-00871/article_deploy/information-16-00871.pdf | 2026-09-29 |
| MambaPSA, arXiv:2607.12681 (yalnız bağlam) | https://arxiv.org/html/2607.12681 | 2026-09-29 |
| YOLOv5 `hyp.VOC.yaml` | https://raw.githubusercontent.com/ultralytics/yolov5/master/data/hyps/hyp.VOC.yaml | 2026-09-29 |
| YOLOv5 TTA dokümanı | https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation/ | 2026-09-29 |
| VOC2007 devkit | https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/devkit_doc_07-Jun-2007.pdf | 2026-09-29 |
| RT-DETR, arXiv:2304.08069 + Crossref | https://arxiv.org/abs/2304.08069 · https://api.crossref.org/works?query.bibliographic=DETRs+Beat+YOLOs+on+Real-time+Object+Detection | 2026-09-29 |
| Bag of Freebies, arXiv:1902.04103 | https://arxiv.org/abs/1902.04103 | 2026-09-29 |
| YOLOX, arXiv:2107.08430 | https://arxiv.org/abs/2107.08430 | 2026-09-29 |
| YOLOv3, arXiv:1804.02767 | https://arxiv.org/abs/1804.02767 | 2026-09-29 |
| YOLOv4, arXiv:2004.10934 | https://arxiv.org/abs/2004.10934 | 2026-09-29 |
| YOLOv10, arXiv:2405.14458 | https://arxiv.org/abs/2405.14458 | 2026-09-29 |
| mixup, arXiv:1710.09412 | https://arxiv.org/abs/1710.09412 | 2026-09-29 |
| Copy-Paste, arXiv:2012.07177 + Crossref | https://arxiv.org/abs/2012.07177 · https://api.crossref.org/works/10.1109/CVPR46437.2021.00294 | 2026-09-29 |
| Soft-NMS, arXiv:1704.04503 + Crossref | https://arxiv.org/abs/1704.04503 · https://api.crossref.org/works/10.1109/ICCV.2017.593 | 2026-09-29 |
| WBF, arXiv:1910.13302 + Crossref | https://arxiv.org/abs/1910.13302 · https://api.crossref.org/works/10.1016/j.imavis.2021.104117 | 2026-09-29 |
| DIoU/CIoU, arXiv:1911.08287 + Crossref | https://arxiv.org/pdf/1911.08287 · https://api.crossref.org/works/10.1609/aaai.v34i07.6999 | 2026-09-29 |
| GIoU (Crossref) | https://api.crossref.org/works/10.1109/CVPR.2019.00075 | 2026-09-29 |
| SIoU, arXiv:2205.12740 | https://arxiv.org/abs/2205.12740 | 2026-09-29 |
| EIoU (Crossref) | https://api.crossref.org/works/10.1016/j.neucom.2022.07.042 | 2026-09-29 |
| Wise-IoU, arXiv:2301.10051 | https://arxiv.org/abs/2301.10051 | 2026-09-29 |
| Inner-IoU, arXiv:2311.02877 | https://arxiv.org/abs/2311.02877 | 2026-09-29 |
| NWD, arXiv:2110.13389 | https://arxiv.org/abs/2110.13389 | 2026-09-29 |
| Focal loss (Crossref) | https://api.crossref.org/works/10.1109/ICCV.2017.324 | 2026-09-29 |
| VarifocalNet (Crossref) | https://api.crossref.org/works/10.1109/CVPR46437.2021.00841 | 2026-09-29 |
| GFL, arXiv:2006.04388 | https://arxiv.org/abs/2006.04388 | 2026-09-29 |
| TOOD (Crossref) | https://api.crossref.org/works/10.1109/ICCV48922.2021.00349 | 2026-09-29 |
| YOLOv1, YOLOv7, VOC 2010/2015, COCO (Crossref) | https://api.crossref.org/works/10.1109/CVPR.2016.91 · …/10.1109/CVPR52729.2023.00721 · …/10.1007/s11263-009-0275-4 · …/10.1007/s11263-014-0733-5 · …/10.1007/978-3-319-10602-1_48 | 2026-09-29 |

### Doğrulanamayan / eksik kalemler
- YOLO11 için DOI (Zenodo) — yok, dokümanda "pending".
- Ultralytics'in resmî YOLO11 teknik raporu — bulunamadı.
- Birebir protokol (YOLO11n, COCO ön-eğitimli, 100 ep, 640, VOC07 test) için literatür değeri — bulunamadı.
- YOLO-Master'ın CVPR 2026 bildiri künyesi (sayfa/DOI) — doğrulanmadı; arXiv olarak girildi.
- YOLOv10 (NeurIPS 2024) ve GFL (NeurIPS 2020) bildiri sayfaları — doğrulanmadı; arXiv olarak girildi.
- COCO LNCS cilt numarası — Crossref kaydında yok; alana yazılmadı.
- 11 noktalı ile 101 noktalı AP arasındaki sayısal fark, `difficult` atlamanın mAP'e etkisi, test-seçim
  yanlılığının büyüklüğü — kaynak bulunamadı.
- Inner-IoU, SIoU, EIoU için VOC tablosu — bu oturumda okunamadı.
