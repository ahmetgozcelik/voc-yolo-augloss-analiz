# h03_boxcp (H03)
Ebeveynden fark (tek cümle): Mosaic'ten sonra/RandomPerspective'ten önce, p=0.5 olasılıkla görüntüden 1-3 küçük nesnenin (sqrt(alan)<=48 px, IoA>0.1 örtüşmeyen) kutu-içi yaması aynı görüntüde çakışmasız bir konuma yeniden ölçeklemeden, 3 px kenar yumuşatmayla yapıştırılır (etiket sınıfı aynı); başka her şey baseline ile aynı.

Uygulama: `boxcp.py` (`BoxCopyPaste`, Ultralytics `CopyPaste` ile aynı yapıcı imzası) + `kos.py` (import zamanında `ultralytics.data.augment.CopyPaste = BoxCopyPaste` atar,
sonra `scripts/rakip_kos.py`'yi `--override copy_paste=0.5` ile çağırır). Windows spawn: dataset ana süreçte kurulur, `BoxCopyPaste` örnekleri dataset ile pickle'lanıp
DataLoader işçilerine gider (sınıf `boxcp` modülünde); sayaç işçilerde `BOXCP_SAYAC_DIR/<pid>.json` dosyalarına yazılır, `kos.py` bitimde toplar.
Uygulama kararları (kart sessiz): (1) hedef konum mosaic tuvalinin MERKEZ penceresinde (RandomPerspective yalnız orayı tutar; aksi halde yapıştırmaların ~%75'i kırpılır);
(2) yumuşatma yama içinde (komşu piksele taşmaz); (3) etikete kaynak kutunun gerçek boyutu yazılır; (4) min kenar 2 px.
`metrics.json`: `varyant_imzasi`, `varyant_etki_sayaci` (yapıştırılan nesne, ≈), `boxcp_uygulanan_goruntu`, `boxcp_cagri`, `boxcp_aday_toplam`, `boxcp_yer_bulunamadi`, `boxcp_surec_sayisi`.
DİKKAT: epochs <= close_mosaic (10) ise Ultralytics mosaic+copy_paste'i ilk epoch'tan kapatır; duman(3 ep)/ön-uçuş(5 ep) satırlarında bu yüzden `--override close_mosaic=0` kullanılır (kıyas: `baseline_cm0` kontrolü). 100 epoch'ta varsayılan (son 10 epoch kapalı).

Koşum: `python WS/varyantlar/h03_boxcp/kos.py --seed S --epochs E --fraction F --imgsz 640 --batch 16 --workers 4 --override copy_paste=0.5 [--override close_mosaic=0] --run-dir {RUN_DIR} --out {RUN_DIR}/metrics.json`
Testler (CPU): `python WS/varyantlar/h03_boxcp/test_boxcp.py` (çıktı: `test_sonuc.txt`; görseller `ornek_canvas_*.jpg`, `ornek_batch_*.jpg` yalnız görsel kontrol).
