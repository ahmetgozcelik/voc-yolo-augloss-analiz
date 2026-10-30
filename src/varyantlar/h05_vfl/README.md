# h05_vfl (H05)
Ebeveynden fark (tek cümle): `v8DetectionLoss` sınıflandırma terimi eleman-bazlı BCE yerine **Varifocal ağırlıklı BCE** olur: `out = BCE(p,q) · w`, `w = 0.75·σ(p)²·(1−y) + q·y`, `y = 1[q>0]`, `q` = TAL `target_scores`; `.sum()/target_scores_sum` normalizasyonu ve `cls=0.5` kazancı, TAL ataması, kutu/DFL kayıpları ve diğer her şey baseline ile aynı.

Uygulama: `vfl_patch.py` — `v8DetectionLoss.__init__` sarılır; orijinal `__init__` sonrası `self.bce` (`nn.BCEWithLogitsLoss(reduction="none")`, çağrı `loss.py:451`) yerine aynı çağrı imzalı `VFLElementwise(alpha=0.75, gamma=2.0)` konur.
Ağırlık detach EDİLMEZ (mmdet/Ultralytics `VarifocalLoss` ile aynı). Girdi float32'ye çevrilir (AMP). Yerleşik `VarifocalLoss` (`.mean(1).sum()`) KULLANILMAZ: normalizasyon kayardı.
`kos.py` yamayı uygular, `scripts/rakip_kos.py`'yi aynı CLI ile çağırır (H02 kalıbı; kayıp ana süreçte hesaplandığından DataLoader işçilerinde yama gerekmez). Ablasyon için `--vfl-alpha A --vfl-gamma G` (varsayılan 0.75/2.0; rakip_kos'a iletilmeden çıkarılır).
`metrics.json`: `varyant_imzasi`, `varyant_etki_sayaci` (VFL modülü EĞİTİM çağrısı), `vfl_val_cagri`, `vfl_alpha/gamma`, `vfl_bce_grad_orani` (her 200. eğitim çağrısında aynı (p,q) için ‖∇VFL‖/‖∇BCE‖ logit-gradyan normu oranı; bilgi amaçlı, kartta beklenen azalma >=%50),
`vfl_grad_norm_ort`, `bce_grad_norm_ort`, `vfl_bce_kayip_orani`, `vfl_olcum_n`.
UYARI: `results.csv` `train/cls_loss` ve `val/cls_loss` VFL değeridir (baseline ile sayısal kıyaslanmaz, ~0.4–0.6x); mAP etkilenmez.

Koşum (kuyruk `WS/kuyruk/k4-tur2.jsonl`):
`WS/env/.venv/Scripts/python.exe WS/varyantlar/h05_vfl/kos.py --seed S --epochs E --fraction F --imgsz 640 --batch 16 --workers 4 --run-dir {RUN_DIR} --out {RUN_DIR}/metrics.json`
(duman: E=3 F=0.1; ön-uçuş: E=5 F=1.0; tam: E=100 F=1.0). Duman/ön-uçuşta mosaic kapalıdır (epochs<=close_mosaic); VFL mosaic'ten bağımsız olduğundan standart baseline çıpaları geçerlidir (baseline_cm0 gerekmez).

Testler (CPU, `CUDA_VISIBLE_DEVICES=-1`): `python WS/varyantlar/h05_vfl/test_vfl.py` (çıktı `test_sonuc.txt`): T0 elle hesap (saf-python math ile |fark|<1e-9), T1 ağırlık türleri + float64 gradcheck + float32 gradyan sonlu,
T2 uç değerler (p=−60 → kayıp 0, NaN yok; fp16 girdi → float32 çıktı), T4 alpha=1,gamma=0,y=0 ⇒ BCE ile birebir (0.0 fark), T5 gradyan-normu ölçümü + eğitim/val çağrı ayrımı,
T3 GERÇEK `v8DetectionLoss` yolu (yolo11n.pt, sentetik batch): sayaç>0, model gradyanları sonlu, `cls_loss == Σ VFL(p,q)/target_scores_sum·cls` (fark 0; normalizasyon korunuyor), `box_loss` ve `dfl_loss` BCE'li yolla birebir aynı (fark 0).
Uçtan uca CPU denemesi (48 görüntü/24 val, 2 ep, `kos.py`): `varyant_etki_sayaci=12`, `vfl_alpha=0.75`, `vfl_gamma=2.0`, `vfl_bce_grad_orani=0.76`, `vfl_bce_kayip_orani=0.55`.
