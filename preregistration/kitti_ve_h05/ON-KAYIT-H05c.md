# ÖN-KAYIT — H05c: Varifocal + cls kazancı telafisi (YOLO11n × VOC, yalnız val)

Yazıldı: 2026-10-22, koşumdan ÖNCE. Tetikleyen: hakem tur 3 Ö14 (MW/notes/review-tur3.md) ve danışman maddesi 3
("fine-tuning negatif çıktı, sebep"). Bu kart ilk deneyin VOC test kilidi açıldıktan SONRA yazılan keşifsel bir
mekanizma kartıdır; VOC test AÇILMAZ, yalnız doğrulama (VOC val 1000) kullanılır. Kart bir iyileşme iddiası taşımaz.

## Soru
H05'in (Varifocal, α=0.75, γ=2, cls=0.5) VOC val kaybı (Δ = −2.87 ± 0.33; mekanizma kartlarına göre esas olarak
negatif bastırmadan) sınıflandırma gradyanının BÜYÜKLÜĞÜNÜN küçülmesinden mi (VFL/BCE gradyan normu oranı 0.342,
0.341, 0.341), yoksa gradyanın BİÇİMİNDEN (hangi örneklerin ağırlıklandığından) mı geliyor?

## Müdahale (tek değişken: H05'e göre)
H05 ile birebir aynı (varyantlar/h05_vfl/kos.py, α=0.75, γ=2, poz=q), yalnız `--override cls=1.47`.
Gerekçe: 0.5 / 0.3413 = 1.465 → 1.47; ölçülen ortalama gradyan normu oranını ≈1'e getirir. Not: kazanç tüm cls
terimini ölçekler (pozitif ve negatif); oran eğitim boyunca ortalamadır, epoch başına sabit değildir.
Seed 0, 1, 2; 100 epoch; imgsz 640; batch 16; diğer her şey H05 ve baseline ile aynı.

## Ölçüt ve karar (önceden)
Δ_c,s = val mAP50-95(H05c, s) − val mAP50-95(B, s) (puan; B ve H05 = ilk deneyin koşumları, analiz_ek.mekanizma ile aynı).
Geri kazanım R = (Δ̄_c − Δ̄_H05) / (−Δ̄_H05).
Tohum-eşli fark D_s = Δ_c,s − Δ_H05,s; tek yönlü t (D > 0), α = 0.10.
- **BÜYÜKLÜK açıklar:** D̄ > 0 ve p < 0.10 ve R ≥ 0.75.
- **KISMİ:** D̄ > 0 ve p < 0.10 ve 0.25 ≤ R < 0.75 → büyüklük kaybın bir kısmını açıklar; kalan pay biçime bağlanır (hipotez).
- **BÜYÜKLÜK açıklamaz:** p ≥ 0.10 ya da R < 0.25 → kayıp gradyan büyüklüğüyle açıklanmaz; biçim (negatif bastırmanın
  dağılımı) hipotezi güçlenir ama kanıtlanmaz.
Beklenen (önceden yazılan tahmin): KISMİ (R ≈ 0.3–0.7). Δ̄_c > +0.40 çıkarsa bu bir iyileşme iddiası DEĞİLDİR (val, keşifsel,
test yok); yalnız "telafi kaybı tümüyle kapattı" olarak yazılır.
İkincil (betimsel): vfl_bce_grad_orani × cls/0.5 (etkin gradyan oranı), en iyi epoch, ep20/50/90 eğri farkları.

## Geçerlilik / güvenlik
H05 yeniden koşumundaki kalibre kural aynen: 20. epoch'ta val < 0.40 ise dur (VFL eğrisine göre kalibre); NaN kes;
min_adim 20. Koşum geçersizse yeniden koşum değil, sonuç "koşulamadı" olarak raporlanır.
Bütçe: 3 × ≈3.4 GPU sa ≈ 10 GPU sa.

## Körlük
Üç tohum bitene kadar H05c val değerleri okunmaz; yalnız olay türü + rc okunur.
