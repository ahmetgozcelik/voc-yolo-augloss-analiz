# Rakip Kurulumu (K3) — Ultralytics YOLO11n

*rakip-kurucu · 2026-09-29 · WS = `deney_workspace/2-1-enhancing-real-time-object-detection-on-pascal-v`*

## 1. Kimlik ve pinleme (R1)

| Kalem | Değer |
|---|---|
| Yazılım | Ultralytics YOLO ("ultralytics" paketi) |
| Sürüm | `8.4.165` (pip'te kurulu, `env/requirements.lock`) |
| Etiket / commit | `v8.4.165` / `004db34b12f65fa5a160de2b1090f6943106bd35` (2026-09-28 21:40:02 +0200, "Stop installing top-level tests package from the wheel (#26408)") |
| Sığ klon | `WS/rakip/ultralytics/` (`git clone --depth 1 --branch v8.4.165`), `git rev-parse HEAD` = yukarıdaki commit — **birebir eşleşti** |
| Ağırlık | `WS/rakip/weights/yolo11n.pt` (COCO ön-eğitimli), sha256 `WS/rakip/weights/yolo11n.pt.sha256` = `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` (5 613 764 bayt). Ultralytics `assets` deposunun hem `v8.3.0` hem `v8.4.0` (latest) sürüm etiketlerinden indirilip hash karşılaştırıldı — **birebir aynı bayt**, ağırlık dosyası sürümler arası değişmemiş. |
| Lisans | AGPL-3.0 (yalnız araç zincirini etkiler, veriyi değil; `data_manifest.md` §V6) |

**R2 uyumu:** `WS/rakip/` içine hiçbir değişiklik yapılmadı. Zorunlu yama yok → `YAMALAR.md` yazılmadı.

### Pip paketi = pinli commit doğrulaması (R1)
8 anahtar dosya (`cfg/default.yaml`, `cfg/datasets/VOC.yaml`, `utils/metrics.py`, `utils/loss.py`,
`data/augment.py`, `nn/tasks.py`, `engine/trainer.py`, `engine/validator.py`) `env/.venv/Lib/site-packages/ultralytics/`
(pip kurulumu) ile `rakip/ultralytics/ultralytics/` (git sığ klon) arasında sha256 ile karşılaştırıldı.
Ham hash'ler CRLF/LF satır sonu farkı yüzünden (git `core.autocrlf=true`, Windows checkout) eşleşmedi;
`sed 's/\r$//'` ile satır sonları normalize edildikten sonra **8/8 dosya birebir eşleşti**. Sonuç: pip
paketi = pinli commit, sürüm uyuşmazlığı yok, yama gerekmiyor.

## 2. Ortam

| Kalem | Değer |
|---|---|
| Makine | <makine>, Windows 10, i5-12400F, 15.9 GB RAM, 12 mantıksal çekirdek |
| GPU | NVIDIA GeForce RTX 5050, 8 GB VRAM (Blackwell sm_120), sürücü 610.47 |
| Python | 3.11.0, venv `WS/env/.venv` |
| torch | 2.14.0+cu130 (CUDA derleme 13.0), `torch.cuda.is_available()=True` |
| ultralytics | 8.4.165 |
| Ek kurulum (K3) | `faster-coco-eval==1.8.0` — `pip install "faster-coco-eval>=1.6.7"`. Ultralytics'in kendi `val(save_json=True)` yolu, VOC gibi COCO-olmayan veri kümelerinde küçük/orta/büyük (S/M/L) mAP kırılımını hesaplamak için bunu `check_requirements()` ile **otomatik pip-install** etmeye çalışıyor (rakip kodu — dokunulmadı, R2); ağ bağımlılığını ve koşum-zamanı sürprizini önlemek için K3 başında elle, tek seferde kuruldu ve `env/requirements.lock`'a eklendi. Bu bir kod yaması DEĞİL, standart bir ek pip bağımlılığı kurulumu. |

`env/requirements.lock` güncellendi (`pip freeze`): `torch==2.14.0+cu130`, `torchvision==0.29.0+cu130`,
`ultralytics==8.4.165`, `faster-coco-eval==1.8.0`, `pycocotools==2.0.11`, ...

## 3. Protokol (R3) — sabit ayarlar

Kaynak: `notlar/literatur-cipa.md` (şef kararı) — COCO ön-eğitimli `yolo11n.pt` → VOC07+12
trainval'a ince ayar, 640 px, 100 epoch, Ultralytics **varsayılan** hiperparametreleri (K4'e kadar
hiçbir hiperparametre değiştirilmez — `mosaic=1.0`, `mixup=0.0`, `copy_paste=0.0`, `close_mosaic=10`,
`box=7.5`, `cls=0.5`, `dfl=1.5`, `optimizer=auto`, `cos_lr=False`, `fliplr=0.5`, `scale=0.5`, CIoU box
kaybı, BCE+TAL sınıf kaybı — hepsi `cfg/default.yaml`'dan, dokunulmadı).

`WS/scripts/rakip_kos.py` çağrısı: `model.train(data=voc_dev.yaml, epochs=100, imgsz=640, batch=16,
workers=6, seed={0,1,2}, deterministic=True, cache=False, exist_ok=True, amp=True, val=True,
plots=False, device=0)`.

### Protokol farkları tablosu (literatürdeki 07+12→07test'e göre)

| # | Fark | Yön / etki | Gerekçe / kaynak |
|---|---|---|---|
| 1 | Eğitim kümesi 15 551 (16 551 değil); trainval'dan 1000 görüntü **val** olarak ayrıldı (seed=0, katmanlı 07/12) | Baseline'ı literatürün **altına** iter (biraz daha az veri) | VOC devkit §4.4 "all parameter tuning must be conducted using the training and validation data alone"; literatürdeki kaynakların çoğu bunu ihlal ediyor (`data_manifest.md` §Split notu, `literatur-cipa.md` §2.2/§3.3) |
| 2 | `val:` = kendi ayrılan 1000 görüntü (K4 döngüsü boyunca); **07 test yalnız `--rep-test` ile, best.pt üzerinde, bir kez** | Test sızıntısı YOK (literatürün aksine) | `literatur-cipa.md` §3.3; `voc_dev.yaml` / `voc_test.yaml` ayrı dosyalar |
| 3 | `difficult==1` nesneler Ultralytics dönüştürücüsü tarafından etiketlerden **çıkarılmış** (train %15,1, val %17,0, test %19,7 nesne) | Ultralytics mAP'i resmî VOC devkit'e göre **aşağı** çeker (bu nesneleri bulan tespitler FP sayılır) | `literatur-cipa.md` §3 madde 2; `data_manifest.md` "difficult nesnelerin ele alınışı" |
| 4 | Ultralytics 101-nokta (COCO-tarzı) AP; VOC resmî ölçüsü 11-nokta (2010 öncesi) / tüm-nokta (2010 sonrası) | Sayılar VOC devkit ile **doğrudan kıyaslanamaz** | `literatur-cipa.md` §3 madde 1 (`compute_ap`, `method="interp"`) |
| 5 | `conf=0.001`, `iou=0.7`, `max_det=300` (Ultralytics val varsayılanı) tüm koşumlarda sabit | Nötr (sabit tutulduğu için karşılaştırılabilir) | `literatur-cipa.md` §3 madde 5; RT-DETR Tab.1 conf duyarlılığı örneği |
| 6 | S/M/L (küçük/orta/büyük) mAP kırılımı Ultralytics'in kendi COCO-json değerlendirmesiyle (`faster_coco_eval`, `save_json=True`) hesaplanıyor; VOC'un resmî bir S/M/L tanımı yok | Bilgilendirici ek metrik, literatürle kıyas için değil | Bu belge §5; `data_manifest.md` "Dağılım" tablosu (kutu alanına göre kova tanımı, K3.5 hata analizinde çapraz kontrol edilecek) |
| 7 | Hiçbir epoch/veri kısaltması **tam koşumda YOK**: aşama 3 tam 100 epoch, tam 15 551 görüntü, ≥3 seed. Yalnız aşama 1 (duman, fraction 0.1, 3 epoch) ve aşama 2 (ön-uçuş, tam veri, 5 epoch) kısaltılmış — bunlar birer kalite kapısı, **rapor edilecek sonuç değil** | — | `kosum-izleme.md` K6; kırmızı çizgi: "epoch/veri miktarını hız için kısarak tam koşum diye sunma" — ihlal edilmedi |
| 8 | `batch=16` (Ultralytics varsayılanı ve B1/A1 kaynaklarıyla aynı büyüklük mertebesi); `workers=6` | Literatürle karşılaştırılabilirlik korunuyor | §4 aşağıda: batch16 vs 32 ölçümü |

Bu tablo **hiçbir ayarı gizlice zayıflatmıyor**: augmentation açık (mosaic=1.0 vb.), lr Ultralytics
varsayılanı, epoch tam (100). Yalnız veri bölünmesi (1,2) ve etiketleme kuralı (3,4) — ikisi de
Ultralytics'in kendi VOC.yaml davranışından ve VOC devkit kuralına uyma kararından kaynaklanıyor,
rakibi zayıf kurmak için seçilmiş bir ayar değil.

## 4. Donanım kalibrasyonu (batch / workers seçimi, gerekçeli)

Ölçüm: `WS/scripts/rakip_kos.py`, tam veri (`fraction=1.0`), 1 epoch, `--zaman-siniri-dk 2` ile
kesilmiş (kosum.py zaman aşımı, rc=98 — yalnız kalibrasyon, damga leaderboard'a girmiyor).

| Koşum | run klasörü | batch | workers | GPU_mem (reserved) | ölçülen hız |
|---|---|---|---|---|---|
| 1 | `runs/20260929-230542_kalibrasyon_s0` | 16 | 6 | **2.25 GB** | 10.1–10.2 it/s → **161–163 img/s** |
| 2 | `runs/20260929-230810_kalibrasyon_s1` | 32 | 6 | **4.34 GB** | 5.1 it/s → **163 img/s** |
| 3 | `runs/20260929-231039_kalibrasyon_s2` | 16 | 4 | 2.25 GB | 9.9–10.0 it/s → **160 img/s** |

**Sonuç:** batch 16→32 görüntü/sn'i **değiştirmiyor** (163 img/s, aynı) — bu iş GPU-compute sınırlı,
batch büyütmenin hız faydası yok, yalnız VRAM ikiye katlanıyor (2.25→4.34 GB). workers 4→6 arası da
img/s'i değiştirmiyor (160→163, ölçüm gürültüsü içinde) → **dataloader darboğazı yok** (12 mantıksal
çekirdek, VOC görüntüleri küçük, önbellek `.cache` ile hızlı). **Karar: `batch=16`, `workers=6`.**
batch=16 hem Ultralytics'in kendi varsayılanı hem de literatürdeki B1 (YOLOv8n, batch16) ile aynı
büyüklükte → literatürle karşılaştırılabilirlik açısından da doğru seçim; hız kaybı yok (163 img/s
her iki batch'te de aynı), VRAM'de bolca pay bırakıyor (2.25/8 GB ≈ %28 doluluk, diğer sistem
süreçleriyle çakışma riski düşük).

### Epoch / toplam süre tahmini
- Eğitim: 15 551 görüntü / batch 16 = 972 adım / 10.15 it/s ≈ **95,7 sn/epoch** (yalnız ileri+geri geçiş).
- Doğrulama (her epoch sonu, 1000 görüntü, `fraction` yalnız train'i etkiliyor — Ultralytics
  `get_split_fraction`, `data/utils.py:623`, doğrulandı): tahmini **~10 sn/epoch**.
- Toplam **~106 sn/epoch** (checkpoint I/O dahil) → **100 epoch ≈ 10 600 sn ≈ 2,94 saat**.
- Ek (bir kez, koşum sonu): kendi `val_dev` değerlendirmesi (~15 sn) + hız ölçümü (50 ısınma + 300
  ölçüm, batch1, FP16, ~10–15 sn) + (yalnız `--rep-test`: 07 test 4952 görüntü / batch16 = 310 adım,
  ~40 sn).
- **Koşum başına tahmini toplam: ~3,0–3,1 saat** (rep-test'li baseline'da +~1 dk) — literatür
  çıpasının kendi tahminiyle (`literatur-cipa.md` konu.md: "tahmini 3–4 sa/koşum") uyumlu.
- **Kuyruk `k3-baseline` toplamı:** aşama1 (~2 dk) + aşama2 (2 seed × ~9,5 dk ≈ 19 dk) +
  aşama3 (3 seed × ~3,05 saat ≈ **9,15 saat**) ≈ **~9,5 saat GPU** (tek kart, seri).

## 5. Değerlendirme / metrik köprüsü — tasarım notları

- `model.val()` (Ultralytics `Model.val`) yalnız `DetMetrics` nesnesi döndürüyor; COCO-json
  S/M/L kırılımı (`metrics/mAP_small(B)` vb.) yalnız `BaseValidator.__call__`'ın döndürdüğü ham
  `stats` sözlüğünde bulunuyor (`engine/validator.py:310`, `eval_json`). Bu yüzden
  `rakip_kos.py`, rakip kodunu **değiştirmeden**, `ultralytics.models.yolo.detect.DetectionValidator`
  genel API'sini doğrudan örnekleyip çağırıyor (`validator(model=str(best_pt))`) — `model.val()`'ın
  içeride yaptığının birebir aynısı, yalnız dönüş değeri farklı yakalanıyor.
- `save_json=True` ile üretilen `predictions.json` → `{RUN_DIR}/val_preds.json` (dev val) ve
  (yalnız `--rep-test`) `{RUN_DIR}/rep_test_preds.json` (kilitli 07 test) kopyalanıyor; K3.5 hata
  analizi bunları okuyacak (COCO-json: `image_id`, `file_name`, `category_id` (1-tabanlı),
  `bbox` [x,y,w,h piksel], `score`).
- Sınıf başı AP50 / AP50-95 `metrics.json`'ın `alt_grup` alanına `ap50_<sinif>` / `ap5095_<sinif>`
  (ve `--rep-test` ile `rep_ap50_<sinif>` / `rep_ap5095_<sinif>`) anahtarlarıyla yazılıyor — sözleşme
  gereği tek seviye iç içe (`deney-sozlesmesi.md` "Standart koşum sözleşmesi").
- **Test kilidi istisnası (dokümante, kasıtlı):** `--rep-test` bayrağı yalnız bu K3 baseline
  koşumlarında (aşama 3, 3 seed) kullanılıyor; `rep_test_*` anahtarları hem `metrics.json`'a hem
  ayrı `{RUN_DIR}/test_kilitli.json`'a yazılıyor. Bu, genel "test metriğini K5'e kadar yazma"
  kuralının **orkestrasyon_durumu.md "Şef protokol kararları"**nda önceden kayıtlı, tek seferlik
  istisnasıdır: "Baseline best.pt 07 test'te YALNIZ replikasyon için bir kez ölçülür → metrics
  anahtarı `rep_test_*`". K4 manipülasyon döngüsünde `--rep-test` **hiçbir varyantta kullanılmaz**;
  seçim metriği her zaman `val_map50_95`'tir (K4 M4 kuralı, `replikasyon_kapisi.py` bunu denetler).
- Hız ölçümü: `data/splits/val.txt`'in ilk 50 görüntüsü ısınma, sonraki 300'ü ölçüm; batch=1,
  imgsz=640, `half=True` (FP16), backend PyTorch eager (TensorRT/ONNX değil — RTX 5050 T4 ile
  doğrudan kıyaslanamaz, `literatur-cipa.md` §3 madde 7). `predictor.speed`
  (preprocess/inference/postprocess=NMS dahil) ayrı ms/görüntü olarak kaydediliyor.
- `progress.json` yazıcı `on_fit_epoch_end` callback'i ile (rakip koduna dokunmadan, genel
  callback API'si — R2); `kosum.py --kural` her koşum satırında NaN kesme + 10 epoch sonrası
  `val_map50_95 < 0.05` alt sınırı ile canlı erken durdurma kuruyor.

## 6. Duman doğrulaması (aşama 1 — manuel ön-kontrol)

`WS/runs/20260929-230007_baseline_s0` (kosum.py, `--gpu-zorunlu --asama 1`, seed 0, fraction=0.05,
2 epoch, batch16, workers4): **rc=0**, süre 95,2 sn. `metrics.json`: `val_map50_95=0.1267`,
`val_map50=0.1807`, `val_precision=0.807`, `val_recall=0.109`, `val_map_small/medium/large` dolu,
20 sınıfın hepsi için `ap50_*`/`ap5095_*`, hız `fps=105,1` (`latency_toplam_ms=9,51`), `cihaz="cuda:0
/ NVIDIA GeForce RTX 5050"`. `progress.json` her epoch sonunda güncellendi. `val_preds.json`
300 000 satır (1000 görüntü × max_det 300, conf=0,001 varsayılanıyla tutarlı). `--rep-test` yolu
ayrıca ayrı bir tek-seferlik koşumla (seed 99, silindi) uçtan uca doğrulandı: `rep_test_map50_95`
hem `metrics.json`'da hem `test_kilitli.json`'da doğru yazıldı. Kuyruğun kendi aşama-1 satırı
(seed0, fraction 0.1, 3 epoch) henüz **koşulmadı** — kademeli koşum şef tarafından başlatılacak.

## 7. Toplam süre (bu doküman için harcanan, K3 kurulum)

Klon + hash doğrulama + ağırlık indirme: ~3 dk. `rakip_kos.py` geliştirme + `--help`: ~2 dk.
Duman (2 epoch/%5): 95 sn. `--rep-test` kablo denetimi (1 epoch/%2): 158 sn. 3 kalibrasyon koşumu
(batch16/32/16, 1 epoch/tam veri, 2 dk kesme ×3): 360 sn. **Toplam GPU-koşum süresi bu oturumda:
~11 dk.** Kuyruk henüz başlatılmadı (tahmini ~9,5 saat, şef başlatacak).

> **DÜZELTME (şef, 2026-10-01):** Bu belgedeki `workers=6` ilk kalibrasyon seçimidir. 2026-09-29 23:20'deki düşük-RAM sonlandırmasından sonra tüm geçerli koşumlar (k3-baseline-v2 ve sonrası: baseline s0–s2, tüm varyantlar, K5) **`workers=4`** ile koştu (kalibrasyonda 4≈6 hız). Makale `workers=4` yazar.
