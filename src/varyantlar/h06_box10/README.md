# h06_box10 (H06)
Ebeveynden fark (tek cümle): kutu kaybı kazancı `box=7.5 -> 10.0` (`loss[0] *= hyp.box`, `loss.py:469`); `cls=0.5`, `dfl=1.5`, CIoU biçimi, TAL ataması ve diğer her şey baseline ile aynı.

Uygulama: kod YOK; değişikliği komut satırındaki `--override box=10.0` yapar (`scripts/rakip_kos.py` -> `model.train(box=10.0)`; `box=10` tamsayı yazılmaz).
`kos.py` yalnız gözlem/damga sarmalayıcısıdır (eğitim davranışını değiştirmez, RNG tüketmez): `v8DetectionLoss.get_assigned_targets_and_loss` sarılır (sonuç aynen döner),
eğitimde (grad açık) çağrı başına `self.hyp.box/cls/dfl` okunur. `metrics.json`: `varyant_imzasi`, `varyant_etki_sayaci` (hyp.box==10.0, cls==0.5, dfl==1.5 iken eğitim kaybı çağrısı),
`box_gain`, `cls_gain`, `dfl_gain`, `box_egitim_kayip_cagri`, `box_val_kayip_cagri`. `box_gain != 10.0` ise kart geçersizdir (kapı `varyant_etki_sayaci>=1` ile yakalar).
Komutta `--override box=10.0` yoksa sarmalayıcı hata verip çıkar.

UYARI: `results.csv` `train/box_loss` ve `val/box_loss` kazançlıdır (x1.333 büyük görünür; `box_loss/1.3333` ile kıyaslanır); mAP'i etkilemez.

Koşum (kuyruk `WS/kuyruk/k4-tur2.jsonl`):
`WS/env/.venv/Scripts/python.exe WS/varyantlar/h06_box10/kos.py --seed S --epochs E --fraction F --imgsz 640 --batch 16 --workers 4 --override box=10.0 --run-dir {RUN_DIR} --out {RUN_DIR}/metrics.json`
(duman: E=3 F=0.1; ön-uçuş: E=5 F=1.0; tam: E=100 F=1.0). Duman/ön-uçuşta mosaic kapalıdır (epochs<=close_mosaic); `box` kaybı mosaic'ten bağımsız olduğundan standart baseline çıpaları geçerlidir (baseline_cm0 gerekmez).
Uçtan uca CPU denemesi (48 görüntü/24 val, 2 ep): `varyant_etki_sayaci=12`, `box_gain=10.0`, `cls_gain=0.5`, `dfl_gain=1.5`, val kaybı çağrısı 4.
