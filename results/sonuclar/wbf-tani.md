# WBF tanisi (hakem B4; beklenti-ihlali.md §4) — YALNIZ VAL, baseline s0 best.pt

Uretildi: betik `varyantlar/h07_post/wbf_tani.py`; cihaz cuda:0 / NVIDIA GeForce RTX 5050; toplam 156 sn; val 1000 goruntu (data/voc_dev.yaml); kilitli veriye referans yok.

## 0. Ozet (mekanik; yorum karar-tahlilcinindir)
- Tek gecis R0 yeniden-uretimi: R0 mAP50-95 = 0.62893 (baseline metrics.json 0.62893); T1 = 0.63690 (H07 CSV 0.636898); W1 = 0.62611 (H07 CSV 0.626106); T2 = 0.58776 (H07 0.587765); W2 = 0.57149 (H07 0.571495).
- (a) Mekanizma canli ve dogru mu: **EVET** (birim testleri hepsi gecti=True; val duzeyi kimlik testi: tek dal WBF(iou_thr~1) - tek dal NMS(kirpilmis) = +0.000 puan (<0.02 beklenir); uc ozdes dal - tek dal WBF = +0.000 puan (<0.05 beklenir)). Bilgi: tek dal WBF(.6) - tek dal NMS = -0.602 puan (kirpma + .6<IoU<=.7 birlestirmesi).
- Kirpma kontrolu (WBF kutulari girdi-tensoru sinirina kirpar, NMS kirpmaz): R0_clip - R0 = +0.014 puan; T1_clip - T1 = +0.007 puan.
- (b) Etki suruyor mu: W1 - T1 = -1.08 puan; izgara en iyisi `W1_i0.7_s0.001_max` - T1 = +0.45 puan -> **KAYBOLDU (izgarada T1 ustu)**.
- (c) Neden (mekanik kanit listesi, esik 0.5 puan):
  - conf_type=max, avg'dan +0.98 puan iyi: skorda gorunum-sayisi cezasi (avg: <3 gorunumde gorulen kutu skoru kucuk) siralamayi bozuyor
  - izgara en iyisi (W1_i0.7_s0.001_max) W1'den +1.53 puan iyi: WBF hiperparametresi (iou_thr/skip/conf_type) varsayilanda uygun degil
  - W1: AP50 +0.22, AP75 -1.33, AP90 -2.15 (T1'e gore): ince konumlama kaybi, kaba bulma kazanci
  - alt-kume b01: WBF - (ham birlestirme+NMS) = -2.82 puan
  - alt-kume b02: WBF - (ham birlestirme+NMS) = -2.24 puan
  - alt-kume b12: WBF - (ham birlestirme+NMS) = -0.85 puan

## 1. Kod incelemesi: WBF hangi seviyede?
- dal-basina NMS (conf .001, iou .7, multi_label, max_det 300) SONRASI; T1 ise ham dal birlestirme + tek NMS.
- H07 `wbf_goruntuler`: her dal icin `non_max_suppression(conf=.001, iou=.7, multi_label, max_det=300)`; kutular girdi-tensoru koordinatinda (W,H,W,H) ile [0,1]'e normalize + `clip`; `weighted_boxes_fusion(weights=None, iou_thr=.6, skip_box_thr=.001, conf_type='avg')`; sonuc skora gore siralanir, `[:300]`.
- Dal basina NMS'li kutu sayisi ortalama 36; 300 siniri dolan dal orani 0.0% (<=20 goruntu); tensor-disi (clip'e takilan) kutu orani 28.4%.
- `_clip_augmented` (Ultralytics): olcek-1 dali en kaba (stride-32, buyuk nesne) kafasini, en kucuk-olcek dali en ince (stride-8, kucuk nesne) kafasini kaybeder; yani her dal her boyuttaki nesneyi goremez. `conf_type=avg` fuzyonu <3 gorunumde gorulen kutuyu (3 dal icinde ort. * gorunum/3) cezalandirir: **yapisal olarak gormeyen dal bile 'yok' sayilir** (skor boyut-bagimli olceklenir). T2 icin kafa kirpmasi ters yonde (kart: 'tek fark s listesi').

## 2. Birim testleri (CPU/GPU, <=20 goruntu, gercek model dallari)
- U1_temiz_kutular_ayni: **GECTI**
- U1_iou1_ayni: **GECTI**
- U1_skip_filtre: **GECTI**
- U2_ozdes_gorunumler_tek_ile_ayni: **GECTI**
- U3_roundtrip: **GECTI**
- U3_negatif_kontrol_duyarli: **GECTI**
- U4_H07_esdegerlik: **GECTI**
- hw_tensorle_uyumlu: **GECTI**
- U1 tek gorunum (iou_thr .6): 20 goruntu; ayni-sinif IoU>.6 cifti olmayan (temiz) 4, cifti olan (birlesen) 16; temiz goruntulerde girdinin tensor-sinirina KIRPILMIS haline gore max koordinat farki 0.00e+00 px (kirpilmamis girdiye gore 2.05 px: WBF kutulari [0,1]'e kirpar, NMS kirpmaz), max skor farki 0.00e+00; girdi/cikti kutu sayisi 664/491 (fark = NMS .7 > WBF .6 uyumsuzlugundan: .6<IoU<=.7 ciftleri tek-gorunumde bile birlesir).
- U1 iou_thr~1 (kirpilmis girdiye gore; kirpilmamis girdiye gore max 35.31 px): max fark 2.84e-14 px / 0.00e+00 skor; sayi esit=True; skip=0.01 filtresi dogru=True.
- U2 iki/uc ozdes gorunum (tek-gorunum sonucuyla): max fark iki=8.58e-05/uc=1.14e-04 px, skor 0.00e+00/0.00e+00; sayi esit=True.
- U3 normalize/denormalize (dikdortgen tensorlar; her kutu ayri sinif):
  - 480x640: dogru boyut max fark 5.68e-14 px; (H,W) yer degisince 138.7 px (negatif kontrol)
  - 640x480: dogru boyut max fark 5.68e-14 px; (H,W) yer degisince 67.6 px (negatif kontrol)
  - 384x640: dogru boyut max fark 5.68e-14 px; (H,W) yer degisince 1000000000.0 px (negatif kontrol)
  - 640x640: dogru boyut max fark 5.68e-14 px; (H,W) yer degisince 0.0 px (negatif kontrol)
  - 192x256: dogru boyut max fark 1.42e-14 px; (H,W) yer degisince 29.5 px (negatif kontrol)
- U4 bu betigin `wbf_fuse`'u = H07 `wbf_goruntuler` (uc dal, varsayilan): max fark 3.05e-05 px, skor 1.99e-08; sayi esit=True; hw==tensor.shape=True.

## 3. Tam-val sonuclari (puan = x100; Delta: R0'a ve kendi serisinin ham-birlestirme referansina (T1 / T2) gore)

| ad | grup | dallar | w | iou | skip | conf | mAP50-95 | mAP50 | AP75 | AP90 | d vs R0 | d vs T | det/gor | %300 | on-kesme | loc IoU(conf>=.25, IoU>=.5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | ref |  |  |  |  |  | 62.89 | 81.13 | 68.31 | 39.63 | +0.00 | -0.80 | 43 | 1 |  | 0.8634 |
| T1 | ref | 012 |  |  |  |  | 63.69 | 81.93 | 69.08 | 40.34 | +0.80 | +0.00 | 68 | 3 |  | 0.8468 |
| R0_clip | clip-kontrol |  |  |  |  |  | 62.91 | 81.13 | 68.31 | 39.63 | +0.01 | -0.78 | 43 | 1 |  | 0.8636 |
| T1_clip | clip-kontrol | 012 |  |  |  |  | 63.70 | 81.93 | 69.10 | 40.34 | +0.80 | +0.01 | 68 | 3 |  | 0.8470 |
| T2 | ref | 012 |  |  |  |  | 58.78 | 79.58 | 65.57 | 28.55 | -4.12 | +0.00 | 72 | 3 |  | 0.8272 |
| W1 | varsayilan(H07) | 012 |  | 0.6 | 0.001 | avg | 62.61 | 82.16 | 67.75 | 38.19 | -0.28 | -1.08 | 48 | 1 | 48 | 0.8862 |
| W2 | varsayilan(H07) | 012 |  | 0.6 | 0.001 | avg | 57.15 | 80.97 | 63.91 | 23.62 | -5.74 | -1.63 | 50 | 1 | 50 | 0.8594 |
| B0_nms | tek-dal | 0 |  |  |  |  | 16.32 | 25.36 | 17.82 | 5.48 | -46.57 | -47.37 | 33 | 1 |  | 0.8228 |
| B0_wbf | tek-dal | 0 |  | 0.6 | 0.001 | avg | 15.72 | 25.07 | 16.71 | 5.30 | -47.17 | -47.97 | 23 | 0 | 23 | 0.8413 |
| B0_nms_clip | tek-dal | 0 |  |  |  |  | 16.33 | 25.36 | 17.84 | 5.48 | -46.56 | -47.36 | 33 | 1 |  | 0.8231 |
| B0_wbf_i1 | tek-dal | 0 |  | 0.99999 | 0.001 | avg | 16.33 | 25.36 | 17.84 | 5.48 | -46.56 | -47.36 | 33 | 1 | 33 | 0.8231 |
| B0x3_wbf | ozdes-dal | 000 |  | 0.6 | 0.001 | avg | 15.72 | 25.07 | 16.71 | 5.30 | -47.17 | -47.97 | 23 | 0 | 23 | 0.8413 |
| T1_b01_nms | alt-kume | 01 |  |  |  |  | 63.31 | 81.45 | 68.48 | 40.27 | +0.42 | -0.38 | 59 | 2 |  | 0.8576 |
| W1_b01 | alt-kume | 01 |  | 0.6 | 0.001 | avg | 60.49 | 80.44 | 65.31 | 36.31 | -2.40 | -3.20 | 42 | 0 | 42 | 0.8846 |
| T1_b02_nms | alt-kume | 02 |  |  |  |  | 63.00 | 81.59 | 68.57 | 39.25 | +0.11 | -0.69 | 49 | 1 |  | 0.8594 |
| W1_b02 | alt-kume | 02 |  | 0.6 | 0.001 | avg | 60.76 | 81.20 | 65.45 | 35.57 | -2.14 | -2.93 | 35 | 0 | 35 | 0.8823 |
| T1_b12_nms | alt-kume | 12 |  |  |  |  | 62.74 | 81.04 | 67.79 | 39.90 | -0.16 | -0.95 | 53 | 1 |  | 0.8496 |
| W1_b12 | alt-kume | 12 |  | 0.6 | 0.001 | avg | 61.89 | 80.90 | 66.46 | 38.43 | -1.01 | -1.80 | 38 | 0 | 38 | 0.8815 |
| W1_w211 | agirlik | 012 | [2, 1, 1] | 0.6 | 0.001 | avg | 62.12 | 81.90 | 67.38 | 37.34 | -0.77 | -1.57 | 48 | 1 | 48 | 0.8871 |
| W1_w411 | agirlik | 012 | [4, 1, 1] | 0.6 | 0.001 | avg | 59.83 | 79.63 | 64.90 | 35.16 | -3.07 | -3.86 | 48 | 1 | 48 | 0.8887 |
| W1_i0.5_s0.001_avg | izgara | 012 |  | 0.5 | 0.001 | avg | 61.66 | 81.58 | 66.39 | 36.90 | -1.23 | -2.03 | 37 | 0 | 37 | 0.8916 |
| W1_i0.5_s0.001_max | izgara | 012 |  | 0.5 | 0.001 | max | 63.07 | 83.11 | 67.77 | 38.39 | +0.18 | -0.62 | 37 | 0 | 37 | 0.8639 |
| W1_i0.5_s0.001_box_and_model_avg | izgara | 012 |  | 0.5 | 0.001 | box_and_model_avg | 61.28 | 81.12 | 66.03 | 36.69 | -1.62 | -2.41 | 37 | 0 | 37 | 0.8959 |
| W1_i0.55_s0.001_avg | izgara | 012 |  | 0.55 | 0.001 | avg | 62.08 | 81.95 | 67.06 | 37.10 | -0.82 | -1.61 | 42 | 0 | 42 | 0.8881 |
| W1_i0.55_s0.001_max | izgara | 012 |  | 0.55 | 0.001 | max | 63.29 | 83.13 | 68.13 | 38.70 | +0.39 | -0.40 | 42 | 0 | 42 | 0.8631 |
| W1_i0.55_s0.001_box_and_model_avg | izgara | 012 |  | 0.55 | 0.001 | box_and_model_avg | 61.71 | 81.41 | 66.71 | 36.97 | -1.18 | -1.98 | 42 | 0 | 42 | 0.8917 |
| W1_i0.6_s0.001_max | izgara | 012 |  | 0.6 | 0.001 | max | 63.59 | 83.02 | 68.52 | 39.40 | +0.69 | -0.10 | 48 | 1 | 48 | 0.8609 |
| W1_i0.6_s0.001_box_and_model_avg | izgara | 012 |  | 0.6 | 0.001 | box_and_model_avg | 62.24 | 81.57 | 67.48 | 37.90 | -0.65 | -1.44 | 48 | 1 | 48 | 0.8900 |
| W1_i0.7_s0.001_avg | izgara | 012 |  | 0.7 | 0.001 | avg | 63.18 | 81.91 | 68.95 | 38.54 | +0.29 | -0.51 | 65 | 3 | 68 | 0.8800 |
| W1_i0.7_s0.001_max | izgara | 012 |  | 0.7 | 0.001 | max | 64.14 | 82.24 | 69.63 | 40.68 | +1.25 | +0.45 | 65 | 3 | 68 | 0.8530 |
| W1_i0.7_s0.001_box_and_model_avg | izgara | 012 |  | 0.7 | 0.001 | box_and_model_avg | 63.10 | 81.70 | 68.90 | 38.57 | +0.21 | -0.59 | 65 | 3 | 68 | 0.8817 |
| W1_i0.5_s0.01_avg | izgara | 012 |  | 0.5 | 0.01 | avg | 61.91 | 82.02 | 67.11 | 36.48 | -0.98 | -1.78 | 10 | 0 | 10 | 0.8824 |
| W1_i0.5_s0.01_max | izgara | 012 |  | 0.5 | 0.01 | max | 62.91 | 82.74 | 67.80 | 38.34 | +0.02 | -0.78 | 10 | 0 | 10 | 0.8639 |
| W1_i0.5_s0.01_box_and_model_avg | izgara | 012 |  | 0.5 | 0.01 | box_and_model_avg | 61.65 | 81.49 | 66.80 | 36.57 | -1.24 | -2.04 | 10 | 0 | 10 | 0.8858 |
| W1_i0.55_s0.01_avg | izgara | 012 |  | 0.55 | 0.01 | avg | 62.16 | 82.21 | 67.48 | 36.72 | -0.73 | -1.53 | 11 | 0 | 11 | 0.8804 |
| W1_i0.55_s0.01_max | izgara | 012 |  | 0.55 | 0.01 | max | 63.11 | 82.79 | 68.04 | 38.69 | +0.22 | -0.57 | 11 | 0 | 11 | 0.8631 |
| W1_i0.55_s0.01_box_and_model_avg | izgara | 012 |  | 0.55 | 0.01 | box_and_model_avg | 61.92 | 81.65 | 67.21 | 36.82 | -0.98 | -1.77 | 11 | 0 | 11 | 0.8847 |
| W1_i0.6_s0.01_avg | izgara | 012 |  | 0.6 | 0.01 | avg | 62.42 | 82.22 | 67.74 | 37.21 | -0.47 | -1.27 | 12 | 0 | 12 | 0.8800 |
| W1_i0.6_s0.01_max | izgara | 012 |  | 0.6 | 0.01 | max | 63.38 | 82.72 | 68.36 | 39.19 | +0.48 | -0.31 | 12 | 0 | 12 | 0.8609 |
| W1_i0.6_s0.01_box_and_model_avg | izgara | 012 |  | 0.6 | 0.01 | box_and_model_avg | 62.28 | 81.78 | 67.65 | 37.33 | -0.61 | -1.41 | 12 | 0 | 12 | 0.8843 |
| W1_i0.7_s0.01_avg | izgara | 012 |  | 0.7 | 0.01 | avg | 63.00 | 81.79 | 68.79 | 38.33 | +0.10 | -0.69 | 16 | 0 | 16 | 0.8793 |
| W1_i0.7_s0.01_max | izgara | 012 |  | 0.7 | 0.01 | max | 63.94 | 82.01 | 69.39 | 40.67 | +1.05 | +0.25 | 16 | 0 | 16 | 0.8530 |
| W1_i0.7_s0.01_box_and_model_avg | izgara | 012 |  | 0.7 | 0.01 | box_and_model_avg | 62.91 | 81.59 | 68.71 | 38.35 | +0.02 | -0.78 | 16 | 0 | 16 | 0.8811 |
| W1_i0.6_s0.001_absent | izgara | 012 |  | 0.6 | 0.001 | absent_model_aware_avg | 62.44 | 81.88 | 67.67 | 37.99 | -0.45 | -1.25 | 48 | 1 | 48 | 0.8891 |
| W2_i0.6_s0.001_max | izgara | 012 |  | 0.6 | 0.001 | max | 58.60 | 81.64 | 65.26 | 26.43 | -4.29 | -0.18 | 50 | 1 | 50 | 0.8376 |

## 4. Boyut-bazli eslesen-GT skoru (GT'ye IoU>=.5 ile eslesen en iyi tahminin ort. guveni; kapsam = eslesen GT orani) — olcek-skor ceza hipotezi

| ad | skor S | skor M | skor L | kapsam S | kapsam M | kapsam L |
|---|---|---|---|---|---|---|
| R0 | 0.217 | 0.434 | 0.674 | 0.739 | 0.949 | 0.981 |
| T1 | 0.210 | 0.448 | 0.711 | 0.754 | 0.957 | 0.988 |
| W1 | 0.119 | 0.284 | 0.474 | 0.754 | 0.952 | 0.987 |
| W1_i0.6_s0.001_max | 0.235 | 0.484 | 0.739 | 0.754 | 0.952 | 0.987 |
| W1_i0.6_s0.001_box_and_model_avg | 0.094 | 0.274 | 0.446 | 0.754 | 0.952 | 0.987 |
| W1_i0.6_s0.001_absent | 0.103 | 0.278 | 0.455 | 0.754 | 0.952 | 0.987 |
| B0_nms | 0.217 | 0.431 | 0.265 | 0.739 | 0.950 | 0.382 |
| T2 | 0.265 | 0.498 | 0.664 | 0.812 | 0.968 | 0.988 |
| W2 | 0.155 | 0.358 | 0.447 | 0.812 | 0.969 | 0.987 |
(S: sqrt(alan)<32 px, M: 32-96, L: >=96; girdi-tensoru pikseli.)

## 5. Yontem notlari
- Tek val gecisi: her goruntu icin R0, T1 (3 dal), T2 (3 dal) ileri-geciri bir kez; WBF konfigurasyonlari ayni NMS'li dal ciktilari uzerinde isci surecleriyle; metrik Ultralytics `_process_batch` + `ap_per_class` (H07 ile ayni; mAP50-95 yeniden-uretim kontrolu yukarida).
- Atlanan (zaman butcesi): yok
