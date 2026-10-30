# SHA256 kaydı (deney günlüğünden aynen alınmış satırlar)

Kural belgelerinin dondurulma özetleri yalnız deney günlüğünde kayıtlıdır; zaman damgası kriptografik değildir (bkz. README, "Zaman damgaları").

| EK-hazırlık | 2026-10-18 | Ek deney (KITTI × YOLO11s/YOLO26n) hazırlık: duman 16/16 (ilk sürüm, momentum 0.9; yalnız kod yolu), duman-cp 4/4 (close_mosaic=0; H03/H08 sayaç>0), ön-uçuş YOLO26n 1/1 + YOLO11s epoch-1 ölçümü, duman-yeni 6/6 (başlık sayacı YOLO26n=2/YOLO11s=1; H05-pos/neg) | — | runs/20261018-13*, 20261018-15* | sağlık kapısı 6/6 GEÇTİ | — | tasarım hakemi tur 1→3 (tur 3: koşuma hazır); ön-kayıt v4 DONDU sha256 031b5386d194a04daf3230bb00252052a1d12f4f8628cdf915da4cf7449aad01; bölme DONDURMA.sha256 sha256 bdfbeb0d6d4aaf6c7758647d1410d9b932b5bfee97aa98deda5621488293c9e0 | süre: YOLO26n ≈45 sn/epoch (≈1.3 sa/koşum), YOLO11s ≈87 sn/epoch (≈2.4 sa/koşum), VRAM ≤4.0 GB; Ultralytics KITTI bölmesi sızdırıyor (106/141 drive ortak) → sahne-grubu bölmesi |

(3) en iyi epoch ve best–last farkı raporlanacak. v4 sha256 031b5386…aad01 (arşiv kopyası) → v5 sha256 cdcff133d5d3b09f41ba1529a0d2aaaa072d0fb8f3a57a672e74b6a6f6998cc9.

kitti_degerlendir.py (alt küme; birim testi 5/5), kitti_test_ac.py, analiz_ek.py, olay_bekle.sh. v5b sha256 1c0d2a228e833dd127ecc5ea7c261d7b4070cf8db83d16df3f5120e8302b9230.

**KITTI TEST AÇILIŞI** 2026-10-22T12:02:15 — ON-KAYIT sha256 1c0d2a228e833dd127ecc5ea7c261d7b4070cf8db83d16df3f5120e8302b9230; DONDURMA sha256 bdfbeb0d6d4aaf6c7758647d1410d9b932b5bfee97aa98deda5621488293c9e0; 48 koşum; koşulmadı: 0

**EK ön-kayıt H05c (hakem tur 3 Ö14)** 2026-10-22 — `hipotezler/ek_deney/ON-KAYIT-H05c.md` koşumdan ÖNCE donduruldu (sha256 cfc1e0f300b7d04033f8553b092738ccb2a642b220b61e2d7d37ed20982a2e9c). VFL + cls 1.47 (gradyan büyüklüğü telafisi), YOLO11n × VOC val, s0–2; VOC test açılmaz. Karar: geri kazanım R ve tohum-eşli D=Δc−ΔH05 tek yönlü t. Aynı zincirde önce B3(c) tanısı: KITTI val'de last.pt DontCare kurallı/DontCare'siz değerlendirme (`scripts/ek_deney/kitti_val_last.py`; keşifsel, test kilidine dokunmaz).
