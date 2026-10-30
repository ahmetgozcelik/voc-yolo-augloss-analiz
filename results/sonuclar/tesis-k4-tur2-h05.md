# Teşhis — k4-tur2 / H05 tam koşum erken durdurma (2026-10-01 11:55)
- Olay: `20261001-114442_h05_vfl_s0` rc=95, ep5 val mAP50-95 0.179 < canlı kural alt sınırı 0.2538.
- Kanıt: results.csv ep1–5 H05: 0.344, 0.178, 0.067, 0.150, 0.179; baseline s0: 0.352, 0.256, 0.122, 0.234, 0.284 (s1/s2 ep5 0.282/0.291). cls loss düşüyor; NaN yok.
- Kök neden: ep5 alt sınırı H02'nin BCE eğrisinden (baseline ep5 − 0.03) kopyalanmış uygulama güvenlik kuralı; VFL'nin bilinen yavaş başlangıcına kalibre değil (kart ön-uçuş sınırlarını zaten genişletmişti) → yanlış-pozitif "öğrenmiyor" alarmı.
- Düzeltme: kartın öldürme kuralı/beklentisi değişmedi; güvenlik kuralı koşumdan ÖNCE ep20 val<0.40 olarak yeniden yazıldı (baseline ep20 0.520–0.526) → `kuyruk/k4-tur2b.jsonl` H05r `20261001-145045_h05_vfl_s0` (rc=0, val 0.6001 → ÖLDÜR).
- Geçersiz koşum: `20261001-114442_h05_vfl_s0` (kanıt değil).
