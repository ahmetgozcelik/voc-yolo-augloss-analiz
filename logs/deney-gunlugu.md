# Deney Günlüğü — <konu kodu · kısa ad>

Bu günlük makalenin iki bölümünü doğrudan besler: **"Deney kaydı / orkestrasyon günlüğü"**
(kaç varyant, hangi komutlarla) ve **"Başarısız denemelerin analizi"** (yz5 raporu §9 — notlandırma
maddesi 5). Her tur bir satırdır; silinmez, üstü çizilmez.

## Tur tablosu

| Tur | Tarih | Hipotez | Ebeveyn → Varyant | Koşum id'leri | Ana metrik (ort±std, n) | Δ vs ebeveyn | Karar | Öğrenilen |
|---|---|---|---|---|---|---|---|---|
| 0 | 2026-09-30 | — (replikasyon) | — → baseline | 20260930-072103_s0, 101624_s1, 131427_s2 | val 0.6307±0.0041 (n=3); 07test 0.6349±0.0011 | — | YEŞİL | lit 0.610–0.635 içinde; seed gürültüsü val ±0.41 puan |
| 1 | 2026-09-30 | H01 multi_scale=0.25 | baseline_s0 → h01_multiscale25 | 20260930-165837_h01_multiscale25_s0 | val 0.6296 (n=1) | +0.07 puan (AP_S −2.8, AP_M +2.5, fps 125→116, eğitim +12%) | ÖLDÜR (seed0 Δ<+0.30) | ölçek çeşitlemesi küçük nesneyi iyileştirmedi; orta boy lehine kaydı |
| 1 | 2026-09-30 | H02 Inner-CIoU r=0.75 | baseline_s0 → h02_innerciou | 20260930-201415_h02_innerciou_s0 | val 0.6267 (n=1) | −0.22 puan (AP75 −0.62, AP_S +2.2 gürültü içi) | ÖLDÜR (seed0 Δ<+0.30) | kutu kaybı geometrisi AP75 açığını kapatmadı |
| 1 | 2026-10-01 | H03 kutu-tabanlı copy-paste p=0.5 | baseline_s0..s2 → h03_boxcp | 20260930-231243_s0, 20261001-021011_s1, 20261001-050625_s2 | val 0.6332±0.0016 (n=3) | eşleştirilmiş +0.25±0.24 puan, p=0.107; mAP50 +0.74 (p=0.007) | ÖLDÜR (ort. Δ<+0.30) | eşik-altı tutarlı AP50 etkisi; doz düşük (0.12 nesne/görüntü) → tur-3 yüksek doz adayı |
| 2 | 2026-10-01 | H07 eğitimsiz post/TTA (12 satır) | baseline_s0 best.pt → h07_post | 20261001-083853_h07_post_s0 | R0 0.6289; T1 0.6369 | T1 +0.80 (A +0.65, B +1.05, s1 +0.40, s2 +1.22) @2.57× gecikme; kısıt içi en iyi N80 −0.10 | KAZANÇ-A yok; KAZANÇ-B T1 (gerçek-zaman dışı) | çözünürlük↑ AP_S +4.6 ama toplam ↓ |
| 2 | 2026-10-01 | H04 mosaic scale 0.5→0.25 | baseline_s0 → h04_scale25 | 20261001-085011_h04_scale25_s0 | val 0.6260 (n=1) | −0.29 puan (AP_S −1.8, AP_M +1.0) | ÖLDÜR (ön-eleme) | scale=0.5 iki yönde yerel optimum (H01 ile birlikte) |
| 2 | 2026-10-01 | H05 Varifocal cls kaybı | baseline_s0 → h05_vfl | 20261001-114442 (ARIZA, kanıt değil); H05r 20261001-145045_h05_vfl_s0 | val 0.6001 (n=1) | −2.88 puan (recall −5.5, tüm boyutlarda kayıp) | ÖLDÜR (ön-eleme) | VFL ince ayar bütçesinde yakınsamayı yavaşlattı; güvenlik kuralları varyanta kalibre edilmeli |
| 2 | 2026-10-01 | H06 box kazancı 7.5→10 | baseline_s0 → h06_box10 | 20261001-115531_h06_box10_s0 | val 0.6247 (n=1) | −0.42 puan (AP75 +0.36, mAP50 −0.31) | ÖLDÜR (ön-eleme) | box/cls/dfl dengesi yerel optimuma yakın (H02 ile) |
| 3 | 2026-10-01 | H08 yüksek dozlu kutu copy-paste (p 1.0, max_sqrt_alan 96; doz 0.670) | baseline_s0 → h08_boxcp_doz | 20261001-182518_h08_boxcp_doz_s0 | val 0.6312 (n=1) | +0.23 puan (mAP50 +0.38, AP75 +0.07, AP_S −2.1) | ÖLDÜR (ön-eleme) | doz ×4.7 etkiyi büyütmedi → doz–yanıt düz; mekanizma doygun |
| K5 | 2026-10-01 | TEST kilidi (tek sefer, 21:23): baseline / H03 / baseline+TTA(T1) × s0–2 | best.pt (val-seçimli) → *-test | runs/20261001-2123…–213428 (9 koşum, eğitimsiz) | test mAP50-95: baseline 63.49±0.11; H03 63.37±0.20; TTA 64.32±0.11 (n=3) | H03 −0.12±0.21 (p=0.416; CI [−0.40,+0.16]); TTA +0.83±0.22 (p=0.022, Holm 0.044; CI [+0.50,+1.14]) @×2.29 gecikme | H03 test'te de eşik altı; TTA = KAZANÇ-B | val→test: H03 +0.25→−0.12 (taşınmadı), TTA +0.80→+0.83 (taşındı); detay sonuclar/test-analiz.md, karar.md |
| K5-ek | 2026-10-02/03 | k5-ek-seed (hakem B3): H01, H02, H04, H05r, H06, H08 için seed 1,2 (val) + H07 WBF tanı | baseline_s1,s2 → kart_s1,s2 | runs/20261002-* ve 20261003-* (13 koşum; WBF tanı 20261002-001505) | val n=3 Δ ort±sd: H01 -0.21±0.61; H02 -0.35±0.12; H03 +0.25±0.24; H04 -0.85±0.79; H05 -2.87±0.33; H06 +0.04±0.56; H08 -0.08±0.40 | kabul ölçütü (Δ≥+0.40, p_tek<0.10) 0/7; H02 (p çift 0.041) ve H05 (0.004) negatif | karar B değişmez; ÖLDÜR'ler n=3'te de geçerli | tek-seed ön-eleme 6/6 elemeyi doğruladı (H03 yanlış alarm); detay sonuclar/n3-analiz.md |
| EK-hazırlık | 2026-10-18 | Ek deney (KITTI × YOLO11s/YOLO26n) hazırlık: duman 16/16 (ilk sürüm, momentum 0.9; yalnız kod yolu), duman-cp 4/4 (close_mosaic=0; H03/H08 sayaç>0), ön-uçuş YOLO26n 1/1 + YOLO11s epoch-1 ölçümü, duman-yeni 6/6 (başlık sayacı YOLO26n=2/YOLO11s=1; H05-pos/neg) | — | runs/20261018-13*, 20261018-15* | sağlık kapısı 6/6 GEÇTİ | — | tasarım hakemi tur 1→3 (tur 3: koşuma hazır); ön-kayıt v4 DONDU sha256 031b5386d194a04daf3230bb00252052a1d12f4f8628cdf915da4cf7449aad01; bölme DONDURMA.sha256 sha256 bdfbeb0d6d4aaf6c7758647d1410d9b932b5bfee97aa98deda5621488293c9e0 | süre: YOLO26n ≈45 sn/epoch (≈1.3 sa/koşum), YOLO11s ≈87 sn/epoch (≈2.4 sa/koşum), VRAM ≤4.0 GB; Ultralytics KITTI bölmesi sızdırıyor (106/141 drive ortak) → sahne-grubu bölmesi |

## Arıza / beklenti ihlali kayıtları
| Tarih | Belirti | Hipotez | Teşhis yöntemi | Kök neden | Düzeltme | Etkilenen koşumlar (geçersiz sayılan) |
|---|---|---|---|---|---|---|
| 2026-09-29 23:20 | k3-baseline kuyruk süreci öldü (duman aşaması) | — | harness bildirimi | Claude Code düşük RAM'de arka plan kabuğunu sonlandırdı | REAP=1, workers 6→4, k3-baseline-v2 | 20260929-231803_baseline_s0 (rc=96, geçersiz) |
| 2026-09-30 10:40 | K3.5 analiz ajanı metrics.json ham başını yazdırırken rep_test AP anahtarlarının ilk satırlarını ekranda gördü (filtre öncesi) | — | ajan beyanı | kazara görüntüleme; hesap/hipoteze girmedi | betik rep_* anahtarlarını yüklerken atar; hipotez ajanına test verisi verilmez | yok (şeffaflık kaydı) |
| 2026-10-01 11:55 | H05 tam koşumu ep5'te canlı kuralla durdu (rc=95, val 0.179 < 0.2538) | H05 | results.csv eğrisi: VFL ısınmada daha derin çukur (ep3 0.067 vs baseline 0.122–0.163) ama ep4→5 yükseliyor (0.150→0.179), cls loss düşüyor; öğrenme bozuk değil | ep5 güvenlik eşiği H02 (BCE) eğrisinden kopyalanmıştı, VFL'nin bilinen yavaş başlangıcına (kartın ön-uçuş sınırları zaten genişletilmişti) kalibre değildi → yanlış-pozitif "öğrenmiyor" alarmı | kart sonucu/öldürme kuralı DEĞİŞMEDİ; yalnız güvenlik kuralı koşumdan ÖNCE yeniden yazıldı: ep20 val<0.40 (baseline ep20 0.52–0.53) → H05r tek koşum (k4-tur2b) | 20261001-114442_h05_vfl_s0 (geçersiz — sonuç kanıtı değil) |
| 2026-10-01 18:40 | K5 hazırlık ajanı CPU kuru-testini `&` ile arka planda başlattı (kural K4 ihlali) | — | ajan beyanı | süreç disiplini | süreç bitti; yalnız CPU/VAL; GPU ve test etkilenmedi; H08 süre ölçümü CPU paylaşımı nedeniyle hafif uzayabilir (not) | yok |
| 2026-10-18 13:06 | ek-duman 16/16 koşum 36 sn'de rc=1 (FileNotFoundError, WinError 2) | — | kosum.py traceback | kuyruk satırlarında yürütülebilir göreli (`env/.venv/...`); Windows CreateProcess göreli yürütülebiliri `cwd`'ye göre çözmez (ilk deney kuyrukları mutlak yol kullanıyordu) | `kuyruk_uret.py` PY mutlak yol; kuyruklar yeniden üretildi; ek-duman yeniden başlatıldı | runs/20261018-1305*_ek_* (16 duman, rc=96 yetim damgası; kanıt değil) |
| 2026-10-18 13:46 | ek-onucus kuyruk süreci öldü (YOLO11s B ön-uçuşu epoch 1 sonunda) | — | harness bildirimi ("system is running low on memory") | Claude Code düşük RAM'de arka plan kabuğunu sonlandırdı (ilk deneydeki 2026-09-29 olayıyla aynı); boş RAM sonrası 10.2/15.9 GB | yeniden başlatma kullanıcı kararına bırakıldı (CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1 önerisi); ön-uçuş ölçümü epoch 1'den alındı (312 it, 1:12, 3.96 GB VRAM, batch 16) | runs/20261018-134358_ek_yolo11s_baseline_s0 (ön-uçuş; kanıt değil) |

## Bütçe sayacı
- Toplam GPU/CPU saati: **75.45 sa (geçerli otorite: tüm `.done` damgalarının toplamı, 60 koşum; makale bu değeri kullanır)**. Eski sayaç tahmini ≈76.1 sa (önceki ≈39.3 sa + k5-ek-seed 132466 sn ≈ 36.8 sa; 2026-10-03). Hedef 75 sa hafif aşıldı (+1.1 sa), tavan 90 sa içinde (kalan ≈13.9 sa); yeni GPU koşumu planlanmıyor.
- Tur sayısı / üst sınır: 3 + k5-ek-seed (hakem B3 n=3 tamamlama) / 5; durma kuralı tur-2 sonunda tetiklendi; tur-3 = ön-kayıtlı tek kart H08
- Kalan bütçe:

**EK-H05 mekanizma** 2026-10-19 12:50 — ek-h05pos 6/6 (sağlık 6/6). Ön-kayıtlı sınıflama (`sonuclar/ek_deney/mekanizma-analiz.json`):
H05 Δ̄ −2.87; H05-neg Δ̄ −1.81 (tek yönlü p 0.009 → katkı var, f 0.63); H05-pos Δ̄ −0.52 (p 0.164 → katkı ölçütü sağlanmadı, f 0.18);
I = (−1.25, +0.33, −0.69), Ī −0.53, SE 0.46 → toplamsal. Sonuç: "kayıp esas olarak negatif bastırmadan; toplamsal".

**EK ön-kayıt v5 (sapma)** 2026-10-19 14:20 — YOLO26n B s0 (20261019-124541) en iyi epoch 16 → v4 sağlık kapısı "en iyi epoch ≥ 20" sağlıklı baseline'ı dışlıyordu
(eğitim kaybı düzenli düşüyor; val 0.13–0.19 bandında, epoch 2'den beri plato; val cls kaybı epoch ≈16'dan sonra artıyor; LR planı VOC ile aynı).
Hiçbir kart Δ'sı görülmeden: (1) KITTI'de madde uyarıya indirildi, yerine NaN yok / eğitim kaybı düşüyor / val ≥ 0.02; (2) v4'te yazılı olup
test betiğinde eksik olan last.pt duyarlılığı eklendi (aynı tek açılış; kuru deneme val'de, scratchpad'de: best 0.1838 / last 0.1533);
(3) en iyi epoch ve best–last farkı raporlanacak. v4 sha256 031b5386…aad01 (arşiv kopyası) → v5 sha256 cdcff133d5d3b09f41ba1529a0d2aaaa072d0fb8f3a57a672e74b6a6f6998cc9.
Zincir kesilmedi (YOLO26n H01 s0 koşuyor). Sapma tasarım hakemine denetletilecek.

**EK ön-kayıt v5b** 2026-10-19 14:18 — hakem tur 4 "v5 düzeltmeyle kabul" (0 bloklayıcı, 7 önemli). Ö1 maruziyet beyanı + körlük kuralı (izleyici metrik basmaz),
Ö2 geçerlilik ölçütü (100 satır, sonlu, LR=B s0, ep81–90<ep11–20, val≥0.02), Ö3 ikincil metrikler (3/7 sınıf, ep11–90 eğri, tanı tablosu),
Ö4 güç beyanı, Ö5/Ö7 last.pt betimsel + sağlamlık etiketi, Ö6 ACILDI planı + --devam. v5 saati düzeltildi (13:59). Kod: saglik_kapisi.py,
kitti_degerlendir.py (alt küme; birim testi 5/5), kitti_test_ac.py, analiz_ek.py, olay_bekle.sh. v5b sha256 1c0d2a228e833dd127ecc5ea7c261d7b4070cf8db83d16df3f5120e8302b9230.

**EK tanı (hakem tur 4 ucuz denetimler)** 2026-10-19 17:00 — B s0 train1000: best 0.5606 / last 0.7842 (val 0.1789 / ≈0.153) → hat sağlam, genelleme açığı;
GT kutu çizimi hizalı. `sonuclar/ek_deney/tani/`. İlk deneme Windows'ta __main__ koruması olmadığı için asılı kaldı (yalnız tanı süreci öldürüldü; zincir etkilenmedi).

**EK-H07 gecikme düzeltmesi (uygulama hatası)** 2026-10-22 — `h07_kitti.py` ilk koşumu, seçilen satırın gecikme ölçümünde CUDA OOM ile düştü (rc=1; h07_kitti.json yazılmadı, hiçbir gecikme değeri üretilmedi). Neden: yol listesi `predict`e tek kaynak olarak verilmişti ve Ultralytics listeyi tek batch'te işler (`LoadPilAndNumpy`), yani ön-kayıttaki "batch 1" sağlanmıyordu. Düzeltme: YOLO11s için VOC H07 ölçüm motoru (`varyantlar/h07_post/degerlendir.py` Motor/gecikme_olc, görüntü başına perf_counter) birebir; YOLO26n (NMS'siz, son_isle uygulanamaz) için görüntü başına `predict` + `torch.cuda.synchronize` + perf_counter. Isınma 50, ölçüm 300, FP16 değişmedi. Doğruluk satırlarının tanımı ve seçim kuralı değişmedi; tam betik yeniden koşuldu. VOC makale gecikme değerleri etkilenmez (orada zaten Motor kullanıldı).

**KITTI TEST AÇILIŞI** 2026-10-22T12:02:15 — ON-KAYIT sha256 1c0d2a228e833dd127ecc5ea7c261d7b4070cf8db83d16df3f5120e8302b9230; DONDURMA sha256 bdfbeb0d6d4aaf6c7758647d1410d9b932b5bfee97aa98deda5621488293c9e0; 48 koşum; koşulmadı: 0

**EK ön-kayıt H05c (hakem tur 3 Ö14)** 2026-10-22 — `hipotezler/ek_deney/ON-KAYIT-H05c.md` koşumdan ÖNCE donduruldu (sha256 cfc1e0f300b7d04033f8553b092738ccb2a642b220b61e2d7d37ed20982a2e9c). VFL + cls 1.47 (gradyan büyüklüğü telafisi), YOLO11n × VOC val, s0–2; VOC test açılmaz. Karar: geri kazanım R ve tohum-eşli D=Δc−ΔH05 tek yönlü t. Aynı zincirde önce B3(c) tanısı: KITTI val'de last.pt DontCare kurallı/DontCare'siz değerlendirme (`scripts/ek_deney/kitti_val_last.py`; keşifsel, test kilidine dokunmaz).

**EK-H05c sonuç** 2026-10-23 02:0x — 3/3 koşum geçerli (100 epoch, batch 16, cls 1.47, VFL çağrısı 97 200). Ön-kayıtlı analiz (`analiz_ek.py h05c`, kod sonuçtan önce commit 1c7c386): Δc = (+0.02, −1.15, −0.40), ort −0.51 ± 0.59; D = Δc − ΔH05 = (+2.90, +2.04, +2.13), ort +2.36 ± 0.47, tek yönlü p = 0.0065; R = 0.822 → **"büyüklük kaybı açıklar"** (önceden yazılan beklenti "kısmen" idi). Etkin gradyan oranı 1.025/1.021/1.026 (manipülasyon denetimi). En iyi epoch H05c 96/96/95, H05 100/100/100. VOC test açılmadı.
