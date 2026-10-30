# h04_scale25 (H04)
Ebeveynden fark (tek cümle): `scale=0.5 -> 0.25` (RandomPerspective ölçek çarpanı U[0.5,1.5] -> U[0.75,1.25]); `multi_scale`, `mosaic`, `mixup`, `close_mosaic`, `translate`, `hsv_*`, kayıp kazançları ve diğer her şey baseline ile aynı.

Uygulama: kod YOK; değişikliği komut satırındaki `--override scale=0.25` yapar (`scripts/rakip_kos.py` -> `model.train(scale=0.25)`).
`kos.py` yalnız gözlem/damga sarmalayıcısıdır (eğitim davranışını değiştirmez, RNG tüketmez): `DetectionTrainer.preprocess_batch` sonrasında
ana süreçteki eğitim veri kümesinin transform zincirinden `RandomPerspective.scale` okunur; `metrics.json`'a `varyant_imzasi`,
`varyant_etki_sayaci` (args.scale==0.25 **ve** RandomPerspective.scale==0.25 olan eğitim batch'i sayısı), `scale_batch_sayisi`, `scale_args` yazılır; `config.json`'a `varyant_imzasi` eklenir.
Komutta `--override scale=0.25` yoksa sarmalayıcı hata verip çıkar.

Mosaic/`close_mosaic` notu (duman ve ön-uçuş): `epochs (3,5) <= close_mosaic (10)` olduğundan mosaic ilk epoch'tan kapalıdır (baseline duman/ön-uçuşla AYNI koşul);
ama `scale` `RandomPerspective`'te (`augment.py:2873`, `scale=hyp.scale`) mosaic'ten bağımsız uygulanır ve `close_mosaic()` yalnız mosaic/copy_paste/mixup/cutmix'i sıfırlar
(`dataset.py:385-389`) → H04 değişikliği duman/ön-uçuşta da devrededir; bu yüzden **baseline_cm0 kontrolü GEREKMEZ** (H03'ün aksine): standart baseline çıpaları
(duman 0.4104 `20260930-065935_baseline_s0`; ön-uçuş 0.5867 / 0.5908 `20260930-070132_baseline_s0` / `-071120_baseline_s1`) geçerlidir. Tam koşumda (100 ep) mosaic ilk 90 epoch açıktır.

Koşum (kuyruk `WS/kuyruk/k4-tur2.jsonl`; `{RUN_DIR}` kosum.py tarafından çözülür):
`WS/env/.venv/Scripts/python.exe WS/varyantlar/h04_scale25/kos.py --seed S --epochs E --fraction F --imgsz 640 --batch 16 --workers 4 --override scale=0.25 --run-dir {RUN_DIR} --out {RUN_DIR}/metrics.json`
(duman: E=3 F=0.1; ön-uçuş: E=5 F=1.0; tam: E=100 F=1.0). Doğrudan baseline koşucusuyla eşdeğeri: `python WS/scripts/rakip_kos.py ... --override scale=0.25`.

Testler (CPU, `CUDA_VISIBLE_DEVICES=-1`): `python WS/varyantlar/h04_scale25/test_scale.py` (çıktı `test_sonuc.txt`: örneklenen s aralığı scale=0.25 için [0.75,1.25], baseline 0.5 için [0.5,1.5]).
Uçtan uca CPU denemesi (48 görüntü/24 val, 2 ep, `kos.py`): 12 eğitim batch'inin 12'sinde `RandomPerspective.scale==0.25` → `varyant_etki_sayaci=12`, `scale_args=0.25`.
