# h07_post (H07) — eğitimsiz son-işlem / test-zamanı artırma (yalnız VAL)
Ebeveynden fark (tek cümle): eğitim YOK; baseline seed0 `best.pt` ağırlıkları sabit, her satır referans çıkarım hattından (`conf=0.001, iou=0.7, max_det=300, imgsz=640`, tek ölçek, tek NMS) yalnız BİR bileşenle farklıdır.

Dosyalar: `degerlendir.py` (tek betik), `test_h07.py` + `test_sonuc.txt` (CPU birim testleri), çıktılar `sonuclar/h07_post.csv`, `sonuclar/h07_yari_A.txt`, `sonuclar/h07_yari_B.txt`, `{RUN_DIR}/h07_detay.json`, `{RUN_DIR}/metrics.json`.
Kilitli veri bölümüne betikte REFERANS YOKTUR (yalnız `data/voc_dev.yaml` val listesi; `test_h07.py` T7 bunu kaynak metninde denetler).

## Satırlar (12 + R0)
| Aile | Satır | Değişen bileşen |
|---|---|---|
| R0 | R0 | referans hattı (baseline `metrics.json` val_map50_95 0.62893'ü yeniden üretir; >0.0008 sapma = kart durur, rc=94) |
| N | N50 N60 N80 | NMS iou 0.5 / 0.6 / 0.8 |
| S | S03 S05 | Gaussian Soft-NMS σ=0.3 / 0.5 (sınıf-bazlı; aday havuzu = on-NMS iou=1.0 ile skor>=0.001 en yüksek 1000 aday; bozunum eşiği 0.001; son max_det 300) |
| U | U704 U768 U832 | test `imgsz` 704 / 768 / 832 (tek ölçek, TTA yok) |
| T | T1, T2 | TTA: T1 Ultralytics `[1,0.83,0.67]`+yatay çevirme; T2 ölçek-yukarı `[1,1.2,1.4]`+yatay çevirme (`DetectionModel._predict_augment` yamasında tek fark `s`; `_clip_augmented` DEĞİŞMEDEN kalır — kart "tek fark s listesi") |
| W | W1, W2 | WBF (`ensemble_boxes`, iou_thr 0.6, skip 0.001, conf_type avg) T1 / T2 dallarına: her dal ayrı NMS (0.001/0.7/300) sonra WBF. Kart kuralı: yalnız yarım-A'da daha iyi olan T satırının W'si ADAY olabilir; diğeri "ek" satır olarak raporlanır (CSV `aday_uygun=0`). |

## Yarım bölme ve karar kuralı (kart H07 "Öldürme kuralı"; betik CSV'yi otomatik türetir)
- Val görüntüleri `(stem, ad)` ile sıralanır; çift indeks = A, tek indeks = B (500/500; liste dosyaları `sonuclar/h07_yari_{A,B}.txt`). Yarım metrikler, tam-val doğrulayıcının görüntü-bazlı istatistiklerinden (`tp, conf, pred_cls, target_cls`) `ap_per_class` ile yeniden hesaplanır (R0 ile birebir aynı metrik).
- Aile başına yarım-A'da Δ>0 olan EN İYİ satır **aday**. **DOĞRULANDI** = (a) yarım-B Δ >= +0.30 puan **ve** (b) tam-val eşleştirilmiş görüntü-bootstrap (B=200, aynı yeniden örnekleme R0 ve satır için) %10 yüzdelik (tek yönlü %90 alt sınır) > 0 **ve** (c) baseline seed1, seed2 `best.pt`'de aynı satır için ort Δ >= +0.20 ve iki Δ > 0.
- **Kart metninden belgelenen sapma:** kart bootstrap'ı "pycocotools yeniden örnekleme" olarak yazmıştı; pycocotools yinelenen görüntü kimliğini desteklemez ve Δ'nın tanımlı olduğu metrik native `val_map50_95`'tir (Ultralytics `ap_per_class`). Bu yüzden bootstrap native metrik üzerinde, eşleştirilmiş görüntü dizinleriyle yapılır. COCO AP (faster-coco-eval: AP_S/M/L) tam-val'de ikincil sütundur.
- KAZANÇ-A: DOĞRULANDI ve `lat(satır)/lat(R0) <= 1.10`; KAZANÇ-B: DOĞRULANDI ve oran > 1.10 (+ `fps_ge30` sütunu: toplam ≤33.3 ms); kırmızı bayrak (tam-val Δ: N/S >+1.0, W >+1.5, T/U >+2.0 puan) → INCELE. Betik yalnız `oneri` yazar; karar `deney-tahlilci`nindir.

## Gecikme
batch=1, FP16, aynı oturumda, satırlar ARAKESİTLİ (her turda R0 + 12 satır + R0_dep), ≥3 tur (R0 tur-CV>%3 ise en çok 6 tura kadar), tur medyanı; `val.txt` ilk 50 görüntü ısınma (dışlanır) + sonraki 300 ölçüm; süre = letterbox + tensör/cihaz kopyası + model (TTA dahil) + NMS/Soft-NMS/WBF + kutu ölçekleme + çıktıyı CPU'ya alma (uçtan uca, `cuda.synchronize`).
Ölçüm koşulu conf=0.001 (değerlendirme işletim noktası; NMS payı en yüksek); `R0_dep` (conf 0.25, tek-etiket) baseline `8.0 ms` ile bağlantı içindir. Ölçüm öncesi GPU kullanımı >%15 ise `h07_gecikme_gpu_mesgul=1` (satırlar güvenilmez; kuyruk kuralı bunu yakalar).
Not: Soft-NMS saf numpy uygulamasıdır (Python döngüsü bedeli ölçülür, optimize kütüphane değildir); WBF `ensemble_boxes` Python uygulamasıdır.

## Koşum
`WS/env/.venv/Scripts/python.exe WS/varyantlar/h07_post/degerlendir.py --run-dir {RUN_DIR} --out {RUN_DIR}/metrics.json` (kuyruk: `WS/kuyruk/k4-tur2.jsonl`, `--gpu-zorunlu`; eğitim yok, seed yok).
Kuru çalışma (CPU, küçük alt küme): `CUDA_VISIBLE_DEVICES=-1 python degerlendir.py --run-dir <dizin> --data <mini.yaml> --sonuc-csv <csv> --yari-dizin <dizin> --bs-B 12 --bs-isci 0 --gecikme-isinma 3 --gecikme-olcum 10 --gecikme-tekrar 1 --workers 0`.
`metrics.json`: `val_map50_95` (= R0 yeniden üretimi), `val_map50`, `val_map_small/medium/large`, `egitim_sure_sn` (= toplam süre; `egitim_yok=1`), `param_sayisi`, `cihaz`, `latency_toplam_ms` (R0), `fps`, `varyant_etki_sayaci` (R0'dan farklı çıkan satır sayısı), `h07_*` özetleri; satır bazlı her şey `alt_grup` içinde (`h07_<satir>_map50_95|delta_full_puan|delta_A_puan|delta_B_puan|bs_alt90_puan|delta_s1_puan|delta_s2_puan|lat_ms|lat_oran|dogrulandi|kirmizi_bayrak`).
Kilitli veri anahtarı (`test_*`, `rep_*`) YAZILMAZ (betik assert eder).

## Testler
`CUDA_VISIBLE_DEVICES=-1 python test_h07.py`: T1 Soft-NMS elle hesap (0.8·exp(−2) vb.), T2 yarım bölme (1000'lik gerçek val listesi: 500/500, ayrık, girdi sırasından bağımsız), T3 TTA yaması s=std iken orijinalle BİREBİR aynı (fark 0) + stash 3 dal + T2 farklı şekil, T4 Paket/bootstrap (dilimleme, `ap_per_class` eşitliği, tekrarlanabilirlik), T5 WBF (iki aynı dal → aynı çıktı; gerçek model dalları), T6 Soft-NMS uçtan uca, T7 kaynak denetimi.
Uçtan uca kuru çalışma (24 val görüntüsü, CPU): 13 doğrulayıcı satırı, yarım A/B, bootstrap, s1/s2 onayı, gecikme (13 yapılandırma + R0_dep), CSV, `metrics.json` yazıldı (~65 sn). Sayılar 24 görüntüde anlamsızdır (yalnız tesisat testi).

## WBF tanısı (hakem B4; `wbf_tani.py`, ayrı kuyruk satırı `k5-ek-seed` #1, varyant adı `h07_wbf_tani`)
Eğitimsiz, yalnız val, baseline s0 `best.pt`. Birim testleri (tek görünüm ≈ girdi, iki/üç özdeş görünüm, normalize/denormalize fark testi + yanlış-boyut negatif kontrolü, H07 `wbf_goruntuler` eşdeğerliği) CPU'da: `CUDA_VISIBLE_DEVICES=-1 python wbf_tani.py --mod birim --n-img 20` → `sonuclar/wbf-tani-birim.json`. Tam mod (GPU; kuyruk): tek val geçişinde R0, T1, T2, W1/W2 (H07 varsayılanı), iou_thr {0.5,0.55,0.6,0.7} × skip {0.001,0.01} × conf_type {avg,max,box_and_model_avg} ızgarası (T1 dalları), dal alt-kümeleri/ağırlık, kırpma ve tek-dal kimlik kontrolleri → `sonuclar/wbf-tani.md` + `wbf-tani.csv`. Karar tahlilcinindir; betik yalnız mekanik (a)/(b)/(c) kanıt listesi yazar.
