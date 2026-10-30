#!/usr/bin/env python
"""F4 verisi: tur basina GPU saati (cubuk), kumulatif GPU saati (cizgi), tur basina en iyi egitim-karti Δ; durma kurali tetik turu.
P1 (n=3): (i) tur 5 = K5 sonrasi ek seed (k5-ek-seed kuyrugu; .done sure_sn) eklendi (kumulatif ~75.5 sa; gunluk sayaci 76.1 sa);
(ii) tur basina en iyi egitim karti artik 3-seed ortalamasi (R1 H03, R2 H06, R3 H08); seed-0 on-eleme degerleri bos daire olarak kalir.
Kaynak: evidence.json butce.* (WS/runs/*/.done sure_sn), tur.*, dval.*, dtest.*. Cikti: figures/data/f4_tur.dat, f4_ref.dat
Etiketler (lab_*) Python'da yuvarlanir (tablolarla ayni).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ev as E

OUT = E.MW / "figures" / "data"
OUT.mkdir(parents=True, exist_ok=True)
LAB = {0: "R0", 1: "R1", 2: "R2", 3: "R3", 4: "K5", 5: "+tohum"}
lines = ["x;label;gpu;kum;n_kosum;best;lab_best;s0;lab_s0;egitimsiz;lab_tta_val;test_h03;lab_test_h03;test_tta;lab_test_tta;lab_kum"]
# n=3 en iyi (kart adi evidence'ta 'yontem' alaninda) ve seed-0 on-eleme en iyisi
BEST = {1: ("tur.1.en_iyi_delta_n3", "tur.1.en_iyi_delta_s0", "H03"),
        2: ("tur.2.en_iyi_delta_n3", "tur.2.en_iyi_delta", "H04"),
        3: ("tur.3.en_iyi_delta_n3", "tur.3.en_iyi_delta", "H08")}
for t in range(6):
    b = BEST.get(t)
    bv = E.v(b[0]) if b else None
    bcard = E.load()[b[0]]["yontem"] if b else None
    sv = E.v(b[1]) if b else None
    row = [str(t), LAB[t], E.nan(E.v(f"butce.tur.{t}.sure_sa")), E.nan(E.v(f"butce.kumulatif.{t}")), str(E.v(f"butce.tur.{t}.n_kosum")),
           E.nan(bv) if b else "nan", f"{bcard} {E.lab(bv)}" if b else "-",
           E.nan(sv) if b else "nan", f"{b[2]} {E.lab(sv)}" if b else "-",
           E.nan(E.v("tur.2.egitimsiz_en_iyi")) if t == 2 else "nan", E.lab(E.v("tur.2.egitimsiz_en_iyi")) if t == 2 else "-",
           E.nan(E.v("dtest.map50_95.h03.ort")) if t == 4 else "nan", E.lab(E.v("dtest.map50_95.h03.ort")) if t == 4 else "-",
           E.nan(E.v("dtest.map50_95.tta.ort")) if t == 4 else "nan", E.lab(E.v("dtest.map50_95.tta.ort")) if t == 4 else "-",
           E.lab(E.v(f"butce.kumulatif.{t}"), 1, sign=False)]
    lines.append(";".join(row))
(OUT / "f4_tur.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
(OUT / "f4_ref.dat").write_text("target;ceiling;stop_round;prescreen;total;n3_total\n" + ";".join(
    [E.nan(E.v("butce.hedef")), E.nan(E.v("butce.tavan")), str(E.v("butce.durma_kurali_tur")), E.nan(E.v("on_kayit.on_eleme_esigi")), E.nan(E.v("butce.toplam_sa")),
     E.nan(E.v("butce.n3.toplam_sa"))]) + "\n", encoding="utf-8")
print("F4 verisi yazildi")
