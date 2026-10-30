# h01_multiscale25 (H01)
Ebeveynden fark (tek cümle): `multi_scale=0.0 -> 0.25` (her eğitim batch'i 480-800 px, 32'nin katı rastgele boyuta yeniden ölçeklenir); başka her şey baseline ile aynı.

Uygulama: kod YOK; değişikliği komut satırındaki `--override multi_scale=0.25` yapar (`scripts/rakip_kos.py` -> `model.train(multi_scale=0.25)`).
`kos.py` yalnız gözlem/damga sarmalayıcısıdır (eğitim davranışını değiştirmez, RNG tüketmez): batch boyutunu kaydeder, `metrics.json`'a
`varyant_imzasi`, `varyant_etki_sayaci` (640 dışı boyutta işlenen batch sayısı), `ms_boyut_min/max`, `ms_farkli_boyut_sayisi` yazar; `config.json`'a `varyant_imzasi` ekler.
Komutta override yoksa sarmalayıcı hata verip çıkar.

Koşum: `python WS/varyantlar/h01_multiscale25/kos.py --seed S --epochs E --fraction F --imgsz 640 --batch 16 --workers 4 --override multi_scale=0.25 --run-dir {RUN_DIR} --out {RUN_DIR}/metrics.json`
(kuyruk: `WS/kuyruk/k4-tur1.jsonl`). Doğrudan baseline koşucusuyla eşdeğeri: `python WS/scripts/rakip_kos.py ... --override multi_scale=0.25`.
CPU uçtan uca deneme (1 ep, 62 görüntü): 8 batch, boyutlar 480..800 arası 7 farklı değer, `varyant_etki_sayaci=8`.
