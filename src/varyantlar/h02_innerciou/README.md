# h02_innerciou (H02)
Ebeveynden fark (tek cümle): `BboxLoss` IoU terimi CIoU'dan Inner-CIoU'ya (ratio=0.75) geçer: `loss_iou = Σ w·[(1-CIoU) + IoU - IoU_inner] / Σ w`; DFL, TAL ataması, kazançlar (`box=7.5/cls=0.5/dfl=1.5`) ve diğer her şey baseline ile aynı.

Uygulama: `innerciou_patch.py` (`BboxLoss.forward` sınıf düzeyinde değiştirilir; orijinal `loss.py:119-156` ile DFL dalı satır satır aynı),
`kos.py` (yamayı uygular, sonra `scripts/rakip_kos.py`'yi aynı argümanlarla çağırır; `--inner-ratio R` ablasyon için, varsayılan 0.75).
Kayıp ana süreçte hesaplandığından dataloader işçilerinde yama gerekmez. `metrics.json`: `varyant_imzasi`, `varyant_etki_sayaci` (eğitimde yamalı forward çağrısı),
`inner_val_cagri` (eğitim-içi doğrulama kaybı çağrıları), `inner_ratio`, `inner_iou_fark_ort` (örneklenmiş ort IoU-IoU_inner).
UYARI: `results.csv` `train/box_loss` ve `val/box_loss` Inner-CIoU değeridir (baseline ile sayısal olarak kıyaslanmaz); mAP etkilenmez. Son mAP değerlendirmesi (`DetectionValidator`) kaybı kullanmaz.

Koşum: `python WS/varyantlar/h02_innerciou/kos.py --seed S --epochs E --fraction F --imgsz 640 --batch 16 --workers 4 --run-dir {RUN_DIR} --out {RUN_DIR}/metrics.json`
Testler (CPU, `CUDA_VISIBLE_DEVICES=-1`): `python WS/varyantlar/h02_innerciou/test_innerciou.py` (çıktı: `test_sonuc.txt`).
