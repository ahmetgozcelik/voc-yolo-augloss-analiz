# Ön-kayıt — Ek deney (ikinci veri seti + iki ek model + VFL mekanizması)
Tarih: 2026-10-18 · Durum: v4 (KITTI; hakem tur 2–3 bulguları işlendi; tur 3: tasarım koşuma hazır) — tam kuyruk henüz başlamadı

## Karar geçmişi
- 2026-10-16 v1: Open Images V6'dan VOC-20 (OI-VOC20) + YOLO11s; danışman makinesinde koşulacaktı.
  Hakem tur 1: 4 BLOKLAYICI (`hakem-tasarim-ek-deney.md`). v1 belgeleri (sınıf eşlemesi, çakışma denetimi) arşivdir.
- 2026-10-18 v2 (bu belge): danışman makinesi kullanılamıyor → koşumlar öğrenci makinesinde (RTX 5050 8 GB).
  İkinci veri seti **KITTI 2D Object** (farklı alan + farklı sınıflar: bulgunun dış geçerliliğini sınar;
  COCO/VOC/OI ile Flickr kaynaklı görüntü çakışması yok). Modeller **YOLO11s** ve **YOLO26n**. Makale baştan yazılmaz.

## Sorular
- S1 (dış geçerlilik): İlk deneyin bulgusu — COCO ön-eğitimli YOLO ince ayarında ön-kayıtlı eğitim-zamanı
  müdahaleleri varsayılanı geçemez — farklı bir alanda (KITTI) ve iki farklı modelde (YOLO11s: aynı aile, büyük ölçek;
  YOLO26n: NMS'siz, DFL'siz, çift başlıklı farklı mimari) **ortam-içi Δ** olarak sürüyor mu?
- S2 (mekanizma): H05 (Varifocal) negatifliği, VFL'nin negatif örnekleri α·p^γ ile bastırmasından mı, pozitifleri
  q ile ağırlıklandırmasından mı kaynaklanıyor? (Ultralytics'te BCE hedefi zaten TAL'ın IoU-hizalı yumuşak skorudur.)

## Veri: KITTI 2D Object (etiketli 7 481 görüntü)
- Kaynak: Ultralytics `kitti.zip` (YOLO formatı, 8 sınıf: car, van, truck, pedestrian, person_sitting, cyclist, tram, misc).
  DontCare bölgeleri için resmî `data_object_label_2.zip` etiketleri; dizi eşlemesi için resmî `devkit_object.zip`
  (`mapping/train_rand.txt`, `mapping/train_mapping.txt`). Lisans: CC BY-NC-SA 3.0 (akademik kullanım).
- **Bölme — sürüş dizisine göre (sızıntı önleme):** ardışık kareler neredeyse aynıdır; rastgele bölme sızıntı yaratır.
  7 481 görüntü, devkit eşlemesiyle ham sürüş kaydına (drive) bağlanır; bölme drive düzeyinde, seed 0:
  test ≈ %20 (≈1 500, KİLİTLİ), val ≈ %13 (≈1 000), train ≈ %67 (≈5 000). Drive'lar sınıf dağılımı dengelenecek
  şekilde atanır: drive sırası seed 0 alt-seed dizisiyle karıştırılır, test ve val hedef paya (±%2) doldurulur; her sınıfın nesnelerinin ≥ %10'u testte, ≥ %5'i val'de, ≥ %40'ı train'de olmalıdır (train alt sınırı 2026-10-18'de, sonuçlardan önce eklendi: ilk denemede person_sitting train'de %0 kalıyordu); kısıtı sağlayan ilk alt-seed seçilir ve kaydedilir. Eşlemesiz görüntü olursa sayısı kaydedilir ve train'e konur.
  Ultralytics'in hazır train/val ayrımı drive-ayrık değilse KULLANILMAZ (denetim sonucu manifest'e).
  **Ek (2026-10-18, sonuçlardan önce, veri denetiminde bulundu):** 2011_09_28 tarihli sabit-kamera kayıtları aynı sahneyi
  farklı drive adlarıyla içerir (drive-ayrık ilk bölmede bölmeler arası NCC ≈ 1.0 çiftler). Bu nedenle bölme birimi
  **sahne grubu**dur: farklı drive'lardan normalize çapraz korelasyonu ≥ 0.95 (96×29 gri) olan görüntü çifti bulunan
  drive'lar birleştirilir (union-find). Ultralytics hazır bölmesi: 141 drive'ın 106'sı train ve val'de ortak → KULLANILMAZ.
  **Ek 2 (2026-10-18, sonuçlardan önce; hakem tur 2 madde 2):** ilk sahne-grubu bölmesinde en büyük grup (2011_09_28 sabit-kamera
  kampüs avlusu, 80 drive / 577 görüntü) test'e düştü (test'in ≈%35'i tek arka plan). Kısıt: **tek bir sahne grubu test ya da
  val bölmesinin hedef büyüklüğünün en çok %25'i olabilir** (test ≈374, val ≈243 görüntü); aşan gruplar train'e atanır.
  (İlk denenen %15 sınırı tek bir sıradan drive'ı — 250–320 görüntü — bile dışladığı için kısıt kümesini olanaksız kıldı;
  %25, yalnız birleşik/sabit-kamera dev gruplarını — 577 ve 442 görüntü — train'e zorlar.)
- **DontCare (hakem tur 2 düzeltmesi):** eğitimde yok sayılır (Ultralytics davranışı). Değerlendirmede DontCare bölgeleri
  her sınıf için COCO `iscrowd=1` yok-say bölgesi olarak eklenir: tespitler önce gerçek nesnelerle eşleşir; yalnız eşleşmemiş
  tespit, DontCare ile IoA ≥ o anki IoU eşiği ise yok sayılır (KITTI "önce eşleştir, eşleşmeyeni yok say" mantığının COCO eşik
  taramasına uyarlaması; resmî sınıf-başı sabit eşik yerine 0.50:0.95). DontCare'siz mAP ikincil olarak da raporlanır.
  best.pt, Ultralytics'in DontCare'siz val fitness'ıyla seçilir (tüm kartlarda aynı); last.pt ile duyarlılık ikincil.
- Sızıntı denetimi (sonuç `data/kitti/bolme_raporu.json`): bölmeler arası en yüksek NCC test–train 0.855, val–train 0.888,
  test–val 0.833 (≥0.90 çift: 0); dHash ≤ 2 çiftlerinin (34/8/3) NCC'si ≤ 0.81 ve görsel incelemede farklı sahnelerdir (benzer
  yol düzeni). Bölme dondu: `data/kitti/DONDURMA.sha256` (splits, yaml'lar, dontcare.json).
- Değerlendirme: COCO tarzı mAP50-95 (pycocotools, maxDets 300), ek olarak mAP50, AP_S/M/L. Betik: `scripts/ek_deney/kitti_degerlendir.py`
  (birim testli 5/5; DontCare'siz modda VOC val'de Ultralytics ile 0.6176/0.6178 ve 0.6221/0.6228; KITTI duman koşumunda 0.0613/0.0612).

## Modeller ve eğitim (iki model için ortak)
- COCO ön-eğitimli `yolo11s.pt` ve `yolo26n.pt`; Ultralytics 8.4.165 (ilk deneyle aynı commit).
- Hiperparametreler ilk deneyle aynı: 100 epoch, 640 px, batch 16 (nbs 64 birikim), workers 4, close_mosaic 10, AMP.
- **Optimizer açıkça sabit (hakem tur 2 BLOKLAYICI düzeltmesi):** `optimizer=MuSGD lr0=0.01 momentum=0.937 warmup_bias_lr=0.0`.
  Gerekçe: ilk deneyde `auto` (iterasyon > 10 000) MuSGD/lr 0.01 seçti VE `warmup_bias_lr=0.0` atadı (trainer.py:1171–1172);
  `args.momentum` (0.937) değişmedi ve ısınma döngüsü (trainer.py:516–529) tüm param gruplarının momentumunu ilk iterasyondan
  itibaren warmup_momentum→args.momentum ile yeniden yazar (kurulumdaki 0.9 etkisizdir). Bu ayarlarla ilk deneyin fiilî
  optimizer davranışı birebir eşlenir. KITTI'de iterasyon = ⌈4990/64⌉×100 = 7 800 < 10 000 olduğundan `auto` AdamW'ye geçerdi.
- Batch 16 bir modelde sığmazsa o modelin TÜM kartları için 8 (nbs 64 korunur); sapma kaydedilir.
- **Sessiz batch yarılama denetimi:** Ultralytics OOM'da batch'i kendiliğinden yarıya indirir (trainer.py:558–591). Her koşumdan
  sonra `stdout.log`'da "Reducing to batch" aranır; bulunan koşum geçersizdir (arıza tablosu) ve o modelin tüm kartları batch 8'e alınır.

## Kartlar (her model için; seed 0,1,2; eşli karşılaştırma aynı seed baseline'ı ile)
Kart tanımları ilk deneyle birebir aynıdır (`hipotezler/H01–H08.md`):
| Kart | Değişken |
|---|---|
| B | — (baseline) |
| H01 | multi_scale=0.25 |
| H02 | Inner-CIoU (ratio 0.75) |
| H03 | kutu-tabanlı copy-paste p=0.5 |
| H04 | mosaic scale 0.5→0.25 |
| H05 | Varifocal (α=0.75, γ=2) |
| H06 | box kazancı 7.5→10 |
| H08 | yüksek dozlu kutu copy-paste (p=1.0, max_sqrt_alan 96) |
Toplam: 2 model × 8 kart × 3 seed = **48 eğitim koşumu**.

**Doz kaydı (2026-10-18, koşumlardan ÖNCE; CPU, 3 000 mosaic, `varyantlar/h08_boxcp_doz/doz_olc.py --data data/kitti_dev.yaml`,
çıktı `sonuclar/ek_deney/doz_h0{3,8}.txt`):** H03 0.980 ± 0.016 nesne/mosaic (VOC 0.142, ×6.9); H08 2.004 ± 0.016 (VOC 0.670;
ilk deneyin [0.6, 1.5] kapısı aşıldı). Kart parametreleri tek değişken kuralı gereği DEĞİŞTİRİLMEZ; KITTI'nin çok sayıda küçük
nesnesi (max_sqrt_alan 48/96 px altı aday bolluğu) dozu yükseltir. H08/H03 doz oranı 4.7 → 2.05: KITTI'de iki kart doz ekseninde
daha yakındır; doz–yanıt yorumu bu oranla sınırlı yapılır. "Yer bulunamadı" payı 0.

YOLO26n'e özgü (kodla doğrulandı, `utils/loss.py` E2ELoss):
- E2ELoss iki `v8DetectionLoss` örneği kurar (bire-çok top-10, bire-bir top-7→1) → H05 yaması (v8DetectionLoss.__init__ sarmalı) iki başlığa da uygulanır.
- `reg_max=1` → DFL yok, L1 uzaklık kaybı; H02 yaması (BboxLoss.forward) L1 yolunu orijinalle aynı korur. H06 `box` kazancı
  yalnız CIoU terimini ölçekler; YOLO26n'de DFL olmadığından bu, regresyon kaybının IoU bileşeninin tamamıdır (L1 terimi `dfl` kazancıyla).
- Başlık-başına kanıt: H02 `inner_bas_sayisi` ve H05 `vfl_bas_sayisi` (eğitimde çağrılan kayıp örneği sayısı) YOLO26n'de = 2,
  YOLO11s'de = 1 olmalıdır; aksi hâlde koşum geçersiz. (Duman: YOLO26n H02/H05 eğitim çağrısı 32 = 16 batch × 2 başlık.)

## Eğitimsiz çıkarım (H07) — B best.pt, yalnız val
- YOLO11s: ilk deneyin 12 satırı (NMS eşiği, Soft-NMS, test çözünürlüğü, TTA, WBF).
- YOLO26n: NMS'siz → NMS/Soft-NMS/WBF satırları **uygulanamaz** (beyan). Test çözünürlüğü satırları; TTA yalnız
  pinli sürüm end2end modelde destekliyorsa (duman testinde doğrulanır), aksi hâlde uygulanamaz.
- Teste giren H07 satırı (model başına en çok bir): val'de B s0–2 ortalamasında en yüksek mAP50-95'li satır; eşitlikte düşük
  gecikmeli olan. Keşifsel (birincil aileye girmez).

## Mekanizma kartları H05-pos ve H05-neg (S2; hakem tur 2: ikisi de koşulsuz)
YOLO11n × VOC (ilk deney kurulumu), seed 0–2, yalnız val (VOC test kilidi AÇILMAZ), her biri 3 koşum:
- **H05-pos:** `--vfl-alpha 1 --vfl-gamma 0` → negatif terim BCE ile aynı; BCE'den tek fark pozitiflerin q ile ağırlıklanması.
- **H05-neg:** `--vfl-poz 1` (α=0.75, γ=2) → pozitifler BCE gibi ağırlıksız; BCE'den tek fark negatiflerin α·p^γ ile bastırılması.
Birim testi: `varyantlar/h05_vfl/test_sonuc.txt` (ayrıştırma 4/4).
Karar (seed-eşli Δ, ilk deneyin B s0–2'sine göre; Δ_H05 = −2.87): her iki kart için Δ ve seed aralığı raporlanır.
Sınıflama (hakem tur 3 madde 6.2 düzeltmesi; sonuçlar okunmadan, 2026-10-18):
1. **Bileşen katkısı:** bir kartın seed-eşli Δ'sı için tek yönlü t testi (Δ < 0, df=2) p < 0.10 ise o bileşen kayba katkı yapar;
   payı f = Δ̄_kart / Δ̄_H05 betimsel olarak (seed aralığıyla) verilir.
2. **Toplamsallık sınaması:** seed-düzeyi I_s = Δ_H05,s − (Δ_pos,s + Δ_neg,s) (aynı seed'in B'sine göre); |Ī| ≤ 2·SE(Ī) ise
   "iki bileşen toplamsal olarak açıklıyor", aksi hâlde "etkileşim (bileşenlerin toplamı H05'i açıklamıyor)".
3. Raporlanan sonuç bu iki adımın birleşimidir (ör. "kayıp esas olarak negatif bastırmadan; toplamsal"). Hiçbir bileşen
   katkı ölçütünü geçmeyip toplamsallık sağlanıyorsa "bileşenler tek tek ayırt edilemedi, toplamları açıklıyor" yazılır.
Yazım dili: "pozitif gradyan zayıflaması" / "negatif bastırma".

## Uç noktalar ve istatistik (hakem tur 2 madde 4 ile yeniden yazıldı)
- **Birincil:** KITTI test mAP50-95 (DontCare kuralıyla), seed-eşli Δ (kart − B), her model ayrı. Val'e dayalı eleme yok: tüm kartlar teste girer.
- **İyileşme ailesi (birincil):** H01, H03, H04, H06, H08 (5 kart), model başına Holm-5, tek yönlü Δ > 0.
  Kabul: Δ ≥ +0.40 puan VE Holm düzeltmeli p < 0.10. n=3 (df=2) ile bu testin gücü düşüktür: Holm-5'in ilk adımında
  (α/5 = 0.02, tek yönlü) %50 güçte kritik Δ ≈ t₀.₉₈,₂/√3 · sd_Δ ≈ 2.8·sd_Δ (sd_Δ = 0.4 ise ≈ 1.1 puan); bu nedenle her kart için Δ, %95 GA ve kritik etki birlikte raporlanır; kabul edilmeyen kart için
  dil "etkisiz" değil **"kabul ölçütü sağlanmadı"** ve GA'nın dışladığı etki aralığı şeklindedir.
- **Yönlü replikasyon ailesi:** H05 ve H02 için tek yönlü Δ < 0, model başına Holm-2, α = 0.10. "Replike oldu" = Δ̄ < 0 VE
  Holm düzeltmeli p < 0.10. İki model ayrı raporlanır; birleşik karar: iki modelde de replike → "iki modelde tekrarlandı",
  birinde → "bir modelde tekrarlandı", hiçbirinde → "tekrarlanmadı". Not: ilk deneyde H05 Holm sonrası anlamlı negatif;
  H02 yalnız düzeltmesiz p = 0.041 (Holm 0.244) — H02 için "anlamlı negatifti" denmez.
- **İki modelin birleşik özeti (iyileşme ailesi):** her model kendi Holm-5'iyle; makalede "kabul edilen kart sayısı / 5" model başına
  verilir, modeller arası havuzlama yapılmaz.
- **Kümeli bootstrap:** test Δ'nın görüntü-düzeyi GA'sı sahne grubu düzeyinde yeniden örneklenir (13 test grubu; grup içi bağımlılık).
- İkincil (keşifsel etiketiyle): val Δ, DontCare'siz mAP, mAP50, AP_S/M/L, sınıf-başı AP — yön olarak, anlamlılık iddiası yok.
  Sınıf-başı AP'de test nesne sayısı düşük sınıflar (person_sitting ≈ 56, tram ≈ 126, truck ≈ 119) ayrıca işaretlenir.

## Yazılacak sınırlılık cümleleri (test açılışından önce sabit)
- KITTI DontCare kuralı COCO eşik taramasına uyarlamadır; resmî KITTI AP (moderate/hard, sınıf-başı eşik) DEĞİLDİR.
- Bölme sahne grubu düzeyindedir; test 13 grup → GA'lar kümeli bootstrap ile; tek alan (Karlsruhe çevresi sürüş).
- YOLO26n'de H06 regresyonun IoU bileşenini, H05 iki başlığı birlikte değiştirir (başlık-başına ayrı etki ölçülmez).
- Copy-paste dozu KITTI'de VOC'tan ≈7× yüksektir (kart parametreleri sabit tutuldu).
- n=3 ile iyileşme ailesinin gücü düşüktür; "kabul ölçütü sağlanmadı" etkisizlik anlamına gelmez.

## Sağlık kapısı (her tam koşum sonrası, otomatik)
`scripts/ek_deney/saglik_kapisi.py`: metrics.json var; `batch_son` = 16 ve stdout'ta "Reducing to batch" yok; H02/H05 için
başlık sayacı (YOLO26n 2, YOLO11s 1); H03/H08 için `boxcp_cagri` > 0; en iyi epoch ≥ 20. Geçmeyen koşum Δ'ya girmez.

## Güvenlik kuralı
Yalnız NaN / OOM / çökme / ilk 20 epoch'ta val mAP50-95 < 0.02 (öğrenmiyor). VFL'ye özel epoch eşiği YOK.
Durdurulan ya da sessiz batch yarılaması görülen koşum Δ hesabına alınmaz; arıza tablosuna yazılır, aynı seed'le yeniden koşulur.

## Duman ve ön-uçuş (kanıt değildir)
Duman (fraction 0.05, 1 epoch) ve ön-uçuş (B, 2 epoch, tam veri) `close_mosaic=0` ile koşulur: aksi hâlde epochs ≤ close_mosaic
olduğundan mosaic ve copy-paste kapalı kalır (H03/H08 sayacı 0, süre/bellek ölçümü temsilî olmaz). Tam koşumlar close_mosaic=10.

## Test kilidi
KITTI test bölümü tüm planlı eğitimler bitene ya da kısma kararı verilene kadar hiçbir betikte okunmaz; **tek sefer** açılır
(tamamlanmış tüm kart × model × seed best.pt + model başına en çok bir H07 satırı). Kısmi tamamlanmada da tek açılış yapılır;
eksik kartlar "koşulmadı" olarak raporlanır, sonradan test için ikinci açılış YOKTUR. Açılış öncesi bu dosyanın ve
`data/kitti/DONDURMA.sha256`'nın SHA256'sı deney günlüğüne yazılır. VOC test bu ek deneyde açılmaz.

## Bütçe (öğrenci makinesi, tahmini)
YOLO11s 24 × ≈2.3 sa + YOLO26n 24 × ≈1.2 sa + H05-pos/neg 6 × ≈2.8 sa + duman/ön-uçuş/H07/test ≈ 4 sa ≈ **105 GPU sa**
(+%10 yedek ≈ 115). Tavan 130 sa. Ön-uçuş sonrası gerçek süreyle güncellenir.
Sıra: H05-pos/neg → YOLO26n → YOLO11s (kısa olanlar önce; kesinti riski). Kuyruklar seed-dış döngülüdür (her seed'in tam kart seti önce biter).
**Kısma sırası** (tavan aşılacaksa, n=3 korunarak kart düşürülür): önce H08, sonra H04 (ilk deneyde en az bilgi veren iki kart).

## Makaleye yansıma (yeniden yazım YOK)
`YOLO-MAKALE-DEVAM.md` Aşama 6 (ek sonuçlar, artefakt analizi, başlık/özet, atıf azaltma, yazarlar/katkılar, hakem turu).
İddia sınırı: yalnız ortam-içi Δ; "genelleşir", model/veri nedenselliği ve mutlak mAP kıyası yazılmaz.

## Değişiklik v5 — 2026-10-19 14:15 (ön-kayıt sapması; YOLO26n B s0 tek koşumu görüldükten sonra, HİÇBİR kart Δ'sı görülmeden)
v4 metni yukarıda değiştirilmeden durur (v4 sha256 031b5386…aad01, arşiv kopyası `arsiv-ON-KAYIT-ek-deney-v4.md`).
1. **Sağlık kapısı "en iyi epoch ≥ 20" maddesi KITTI tam koşumlarında geçersizlik ölçütü olmaktan çıkar, uyarı olarak raporlanır.**
   Gerekçe: maddenin amacı kırık ya da öğrenmeyen eğitimin best.pt'sini dışlamaktı. YOLO26n B s0'da (20261019-124541) eğitim
   kaybı düzenli düştü (box 1.71→0.79, cls 3.08→0.40), val mAP50-95 epoch 2'de 0.171'e ulaşıp 0.13–0.19 bandında dalgalandı;
   en iyi epoch 16 (0.179), val cls kaybı epoch ≈16'dan sonra artıyor (sahne-ayrık val'de aşırı uyum). Car AP50-95 0.571;
   LR planı VOC baseline'ıyla epoch epoch aynı. Bu, maddenin hedeflediği arıza değildir; madde korunursa baseline'ın kendisi
   dışlanır ve deney tanımsız kalır. Kural tüm kartlara ve iki modele aynı biçimde uygulanır; hiçbir kart sonucu görülmediğinden
   seçim yanlılığı üretmez.
   **Yerine geçen geçerlilik ölçütü (aynı amaç):** NaN yok; son 10 epoch'un ortalama eğitim (box+cls) kaybı ilk 10 epoch'unkinden
   düşük; en iyi val mAP50-95 ≥ 0.02 (güvenlik kuralının alt sınırı). Diğer sağlık maddeleri (batch, başlık sayacı, boxcp) aynen.
   VOC mekanizma kartları (YOLO11n × VOC) için eski madde geçerli kalır (6/6 zaten geçti).
2. **last.pt duyarlılığı uygulanır:** v4 "last.pt ile duyarlılık ikincil" diyordu, ancak test açılış betiği yalnız best.pt'yi
   değerlendiriyordu (eksiklik, test açılmadan düzeltildi). Tek test açılışında her koşumun best.pt ve last.pt'si birlikte
   değerlendirilir; last.pt Δ'sı ikincil/keşifsel, Holm ailelerine girmez. Birincil uç nokta değişmez (best.pt).
3. **Raporlama:** her koşumun en iyi epoch'u ve best–last val farkı tabloda verilir. Ek sınırlılık cümlesi: "KITTI sahne-ayrık
   val'de doğrulama eğrisi erken tepe yapıp dalgalandığından best.pt seçimi gürültülü bir doğrulama ölçümüne dayanır; bu, kartlar
   arası Δ'ya ek varyans katar (last.pt duyarlılığı bu yüzden verilir)."

## Değişiklik v5b — 2026-10-19 (hakem tur 4: "v5 düzeltmeyle kabul", `hakem-tasarim-ek-deney-tur4.md`); ilk YOLO26n kart koşumu (H01 s0) bitmeden
0. **Düzeltmeler (v5 metni):** v5 başlığındaki "14:15" saati tahmindi; dosya zaman damgasına göre v5 **13:59**'da yazıldı (commit 9446bcc 14:02).
   v4 tam sha256: 031b5386d194a04daf3230bb00252052a1d12f4f8628cdf915da4cf7449aad01. B s0 son box kaybı 0.78 (v5'te 0.79 yazıldı).
1. **Maruziyet beyanı ve körlük (Ö1):** v5 yazılırken H01 s0 (13:57'de başladı) ≈2 dakikadır koşuyordu; o koşumun ya da başka hiçbir
   kart koşumunun `progress.json`, `stdout.log`, `results.csv` veya `metrics.json` dosyası okunmadı (yalnız zincir log'unun "başladı" satırı).
   **Körlük kuralı (ek-yolo11s kuyruğu bitene kadar):** kart koşumlarının val değerleri okunmaz; izleme yalnız olay türü + run_id + rc
   (izleyici çıktısı metrikleri kırpar) ve sağlık kapısının GEÇTİ/KALDI satırıyla yapılır. Sağlık kapısı KALDI derse yalnız neden metni okunur.
   Bu noktadan sonra ön-kayıtta değişiklik yalnız değerlendirme/raporlama düzeyinde ve maruziyet beyanıyla yapılabilir.
2. **Geçerlilik ölçütü (Ö2; v5 madde 1'deki ölçütün yerine):** results.csv tam 100 satır; tüm kayıp ve metrik sütunları sonlu; LR sütunları
   aynı modelin B s0'ı ile epoch başına aynı; ep81–90 ortalama eğitim (box+cls) kaybı < ep11–20 ortalaması; en iyi val mAP50-95 ≥ 0.02.
   B'ye göreli bir eşik konmaz (güçlü negatif kartları sansürlerdi). (`saglik_kapisi.py ogrenme_denetle`.)
3. **İkincil metrikler (Ö3; betimsel, Holm/karar yok):** DontCare kurallı mAP50-95'in (a) 3 sınıflık (car, pedestrian, cyclist) ve
   (b) person_sitting hariç 7 sınıflık alt kümeleri; (c) val eğrisinin ep11–90 ortalaması; (d) koşum başına eğitim-içi en iyi val ile
   bağımsız yeniden değerlendirme arasındaki fark ve en iyi epoch. Val ve test analizlerinde tohum-eşli Δ olarak verilir.
4. **Güç (Ö4):** v4'teki sd_Δ = 0.4 örneği VOC'tan taşınmıştır. KITTI'de kritik etki her model için gözlenen sd_Δ ile hesaplanır ve
   her kartın Δ'sı, %95 GA'sı ve kümeli GA'sıyla birlikte verilir. Sahne-ayrık val/test gürültüsü yüksek olduğundan kritik etkinin
   ≈3–4 puana çıkması olasıdır; bu durumda iyileşme ailesi yalnız büyük etkileri ayırabilir ve "kabul ölçütü sağlanmadı" sonucu
   açıkça düşük güç olarak yazılır (etkisizlik değil).
5. **best.pt / last.pt (Ö5, Ö7):** Birincil uç nokta değişmez (best.pt). last.pt test Δ'sı ana tabloda karar sütunu olmadan, betimsel
   olarak (Holm yok) verilir. Her kart bir sağlamlık etiketi alır: best.pt ve last.pt Δ̄'sı aynı işaretliyse "seçimden bağımsız",
   değilse "best.pt seçimine duyarlı". Ek sınırlılık cümlesi: "KITTI'de best.pt erken epoch'lardan seçilebildiğinden (B s0: 16/100;
   VOC'ta ≈93) birincil uç nokta VOC'takinden farklı bir eğitim evresini ölçebilir; last.pt duyarlılığı ve sağlamlık etiketi bu yüzden verilir."
6. **Test açılışı sürekliliği (Ö6):** açılıştan önce her planlı koşumun best.pt ve last.pt'si var olmalıdır; plan (run_id + best/last
   sha256) `ACILDI.json`'a dondurulur. Açılış yarıda kalırsa (süreç ölümü vb.) `--devam` ile **aynı açılış** aynı planla sürer:
   ağırlık hash'leri denetlenir, tamamlanmış koşumlar atlanır, plana koşum eklenemez; devam günlüğe yazılır. Bu ikinci açılış sayılmaz.
