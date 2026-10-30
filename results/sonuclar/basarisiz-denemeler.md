# Başarısız denemeler tablosu (D4 zorunlu)

Kaynak: hipotezler/H01–H08.md Sonuç/Öğrenilen bölümleri, `ablasyon.csv`, `h07_post.csv`, `test-analiz.md`. Δ = val mAP50-95 puanı (x100) baseline'a göre (K5 sonrası n=3 seed-eşli ort; kaynak `n3-analiz.md`). Kartlardaki −0.29/+0.23 gibi değerler metrics.json değerlerinin kesilmiş halidir; bu tablo yuvarlanmış değerleri (−0.30/+0.22) kullanır. Seed gürültüsü (baseline n=3, val): std 0.41 puan; ön-eleme eşiği +0.30.

| Kart | Ne denendi (tek değişken) | Ölçülen Δ (val; test varsa) | Karar | Mekanizma düzeyinde neden başarısız |
|---|---|---|---|---|
| H01 | `multi_scale` 0 -> 0.25 (eğitimde 480-800 px) | -0.21 ± 0.61 (n=3; p_tek=0.69; s0 +0.07); AP_S -0.30, AP_M +0.84 | ÖLDÜR | Ölçek çeşitliliği küçük-nesne AP'sini artırmadı; kazanç orta boya kaydı. Ultralytics varsayılan `scale=0.5` mosaic jitter'ı zaten geniş bir ölçek dağılımı üretiyor, ek çok-ölçekli giriş marjinal. AP_S düşüşü baseline seed aralığı (2.9 puan) içinde. Küçük-nesne açığı eğitim çözünürlüğünü rastgeleleştirmekle kapanmıyor. |
| H02 | CIoU -> Inner-CIoU (r=0.75) | -0.35 ± 0.12 (n=3; p çift=0.041, 3/3 seed negatif); AP75 -0.42; AP_S +1.39 (gürültü içi) | ÖLDÜR | Hedeflenen AP75 yükselmedi. Inner-IoU'nun raporlanan kazancı düşük tabanlı (37.6 AP, sıfırdan eğitim) rejimde; COCO ön-eğitimli, TAL atamalı, doygun YOLO11n ince ayarında taşınmadı. Kutu kaybının geometrisi darboğaz değil; AP75 açığı atama/kalite hedefi veya DFL tarafında olabilir (H06 ile birlikte). |
| H03 | Kutu-tabanlı küçük-nesne copy-paste (p=0.5, sqrt(alan)<=48 px; ~0.14 nesne/mosaic) | Val +0.25 ± 0.24 (n=3 eşli, p=0.214 çift yönlü / 0.107 tek yönlü; eşik +0.30 altı). **Test −0.12 ± 0.21 (n=3, p=0.416; bootstrap %95 CI [−0.40, +0.16])** | ÖLDÜR (birincil ölçüt; test de doğruladı) | Etki gevşek IoU'ya özgüydü: val mAP50 +0.74 (3/3 seed) ama AP75 −0.32; test'te mAP50 −0.02, mAP75 −0.11. Val'deki kazanç 1000 görüntülük val'in seed-gürültüsü/seçim etkisiyle uyumlu, test'e (4952 görüntü) taşınmadı. Doz düşüktü (≈0.14 nesne/mosaic) ama H08 ile doz artırımı da yardım etmedi. Test'te yalnız AP_M +0.48 (p=0.027, keşifsel) ve AP_L −0.23 (p=0.048). |
| H04 | mosaic `scale` 0.5 -> 0.25 | -0.85 ± 0.79 (n=3; p_tek=0.90; s0 −0.30); AP_S -1.06, AP_M -0.75 | ÖLDÜR | Ölçek augmentation'ını daraltmak da genişletmek de (H01) mAP50-95'i artırmadı: varsayılan `scale=0.5` iki yönde de yerel optimuma yakın. 5 epoch ön-uçuştaki +0.5 avantaj 100 epoch'ta tersine döndü (kısa koşum uzun eğitimin yönünü öngörmedi). |
| H05 (r) | cls kaybı BCE -> Varifocal (alpha=0.75, gamma=2) | -2.87 ± 0.33 (n=3, H05r; p çift=0.004; 3/3 seed −2.5…−3.2); AP50 -4.44; AP_S/M/L -2.39/-4.06/-2.10 | ÖLDÜR | Tüm boyutlarda tutarlı kayıp. TAL zaten IoU-farkında hizalama hedefi ürettiğinden VFL'nin kalite ağırlıklandırması gereksiz; negatifleri p^gamma ile bastırmak ve pozitifleri q ile ölçeklemek toplam cls gradyanını küçülttü (CPU testinde grad-norm oranı 0.76) ve 100 epoch'luk ince ayar bütçesinde yakınsamayı yavaşlattı, recall düştü. Hiperparametre (alpha, gamma, lr) ayarlanmadı: yöntem değil kurulum uyumsuzluğu olabilir (sınırlılık). |
| H05 (arıza) | Aynı kart, ilk tam koşum `20261001-114442_h05_vfl_s0` | **GEÇERSİZ — kanıt değil** (rc=95, ep5'te canlı güvenlik kuralı: val 0.179 < 0.2538) | Arıza kaydı | Eşik H02'nin BCE eğrisinden kopyalanmış bir uygulama güvenlik kuralıydı, kartın öldürme kuralı değil; VFL ısınmada daha derin çukur yaptığı için yanlış alarm verdi. Sonuç sayılmadı; kalibre kuralla (ep20 < 0.40) tek yeniden koşum H05r yapıldı. Günlüğün arıza tablosunda. |
| H06 | `box` kazancı 7.5 -> 10 | +0.04 ± 0.56 (n=3; p_tek=0.46; s0 −0.42, s2 +0.66); AP75 +0.07, AP50 +0.30 | ÖLDÜR | Kutu-hassasiyetinde küçük bir kayma (AP75 +0.36, gürültü içinde) mAP50 kaybıyla takas oldu; toplam düştü. H02 ile birlikte kutu-regresyon tarafına yapılan iki farklı müdahale AP75 açığını kapatmadı: varsayılan box/cls/dfl dengesi VOC ince ayarında yerel optimuma yakın. |
| H08 | H03'ün yüksek dozu (p=1.0, sqrt(alan)<=96 px; ölçülen 0.670 nesne/mosaic ≈ H03'ün 4.7 katı; kart hedefi 0.8–1.2'nin altında) | -0.08 ± 0.40 (n=3; p_tek=0.62; s0 +0.22); AP50 +0.25, AP75 -0.23 | ÖLDÜR | Doz–yanıt düz/ters: H03'ün mAP50 kazancı (+0.74) yüksek dozda yarıya indi (+0.38). Etki doz sınırlı değil, doygun. AP_S yine artmadı: küçük-nesne açığı eğitim örnek sayısından çok çözünürlük kaynaklı. Doz kart hedefinin altında kaldı (kod parametresi kart gereği değiştirilmedi); bu bir sınırlılık. |
| H07-N | NMS IoU 0.7 -> 0.5/0.6/0.8 (eğitimsiz, val) | −0.54 / −0.21 / −0.10 | Aday değil | Varsayılan 0.7 zaten yakın optimum; düşük eşik (olası mekanizma: bitişik gerçek nesneleri de bastırır, ölçülmedi) AP50-95'i düşürdü (N50 mAP50 +0.38 ama AP50-95 −0.54). |
| H07-S | Gaussian Soft-NMS sigma 0.3 / 0.5 | −0.12 / −0.59 (gecikme ×1.08) | Aday değil | Tek-ölçek, yüksek-recall (conf 0.001) rejiminde varsayılan NMS'e göre ölçülebilir kazanç yok (mekanizma ayrıca sınanmadı); gecikme arttı. |
| H07-U | Test çözünürlüğü 704 / 768 / 832 (eğitimsiz) | −0.31 / −2.06 / −5.19; U768 AP_S +4.55, AP_L −3.41 | Aday değil | Model 640 px'te eğitildi (ölçek uyumsuzluğu olası neden); test çözünürlüğü artınca küçük nesne belirgin kazanır ama büyük nesne ölçek uyumsuzluğundan kaybeder. Bilgi değeri: küçük-nesne açığı kısmen çözünürlük kaynaklı. |
| H07-T2/W1 | Yukarı ölçek TTA (T2); T1 + WBF (W1) | −4.12 / −0.28 (gecikme ×2.55 / ×2.99) | Aday değil | Yukarı ölçekler (eğitim ölçeğinin üstü) büyük nesneyi bozuyor; WBF kutu-ortalama birleştirmesi T1'in flip+aşağı-ölçek kazancını düşürdü. |

Başarısız sayılmayan, ayrı raporlanan satır: **H07-T1 (TTA)** val +0.80 -> test +0.83 ± 0.22 (n=3), ancak gecikme ×2.29–2.57; bkz. `karar.md`, `test-analiz.md`.

Özet: 7 eğitim-zamanı kartı (H01–H06, H08) ve eğitimsiz post kartlarının hiçbiri +0.30 ön-kayıtlı kapıyı eşit/altı gecikmeyle aşmadı. En iyi eğitim kartı H03'tür (val +0.25, test −0.12). Geçersiz sayılan tek koşum H05 arıza koşumudur (ayrıca k3-baseline ilk koşum rc=96 ve duman/ön-uçuş koşumları karar kanıtı değildir).

## n=3 doğrulaması (K5 sonrası, hakem B3)

Tek-seed (n=1) ön-eleme sayıları yukarıdaki satırlarda n=3 değerleriyle değiştirildi (mekanizma yorumları s0 gözlemine dayanır; n=3 AP_S değerleri gürültü bandı içinde ve işaretleri değişebilir); kart karar sütunları değişmedi. Tablo (mAP50-95, seed-eşli, val):


| Kart | Δ ort ± sd | seed Δ (s0/s1/s2) | p tek (çift) | %95 t aralığı | Kabul | Ön-eleme (s0 Δ) | Ön-eleme isabeti |
|---|---|---|---|---|---|---|---|
| H01 | -0.21 ± 0.61 | +0.07 / -0.91 / +0.21 | 0.693 (0.614) | [-1.72, +1.30] | HAYIR | +0.07 → ELE (s0 < +0.30) | DOGRU (n=3'te de kabul yok) |
| H02 | -0.35 ± 0.12 | -0.22 / -0.47 / -0.35 | 0.980 (0.041) | [-0.66, -0.04] | HAYIR | -0.22 → ELE (s0 < +0.30) | DOGRU (n=3'te de kabul yok) |
| H03 | +0.25 ± 0.24 | +0.37 / -0.03 / +0.41 | 0.107 (0.214) | [-0.35, +0.85] | HAYIR | +0.37 → seed 1,2'ye cik | cikis yanlis alarm (n=3 kabul yok) |
| H04 | -0.85 ± 0.79 | -0.30 / -1.76 / -0.49 | 0.898 (0.204) | [-2.82, +1.12] | HAYIR | -0.30 → ELE (s0 < +0.30) | DOGRU (n=3'te de kabul yok) |
| H05 | -2.87 ± 0.33 | -2.88 / -3.19 / -2.53 | 0.998 (0.004) | [-3.68, -2.05] | HAYIR | -2.88 → ELE (s0 < +0.30) | DOGRU (n=3'te de kabul yok) |
| H06 | +0.04 ± 0.56 | -0.42 / -0.12 / +0.66 | 0.456 (0.912) | [-1.35, +1.43] | HAYIR | -0.42 → ELE (s0 < +0.30) | DOGRU (n=3'te de kabul yok) |
| H08 | -0.08 ± 0.40 | +0.22 / -0.53 / +0.06 | 0.623 (0.754) | [-1.07, +0.90] | HAYIR | +0.22 → ELE (s0 < +0.30) | DOGRU (n=3'te de kabul yok) |


Yedi eğitim-zamanı müdahalesinin hiçbiri val'de 3 seed'lik ön-kayıtlı kabul ölçütünü (ort Δ ≥ +0.40, tek yönlü p<0.10) karşılamadı; yalnız H03 s0 kapısını geçti, n=3'te kabul edilmedi ve testte de geçmedi. H02 (−0.35) ve H05 (−2.87) baseline'ın anlamlı biçimde ALTINDA. Not: "H04 −0.30" tek-seed değerdi; n=3 −0.85.
