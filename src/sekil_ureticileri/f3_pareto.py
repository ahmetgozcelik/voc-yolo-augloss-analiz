#!/usr/bin/env python
"""F3 verisi: H07 satirlari (val Δ mAP50-95 vs gecikme orani) + TTA test noktasi. Kaynak: evidence.json h07.*, dtest.*, boot.*, test.latency_orani.tta.
Isaretci sinifi (T4 ile ayni on-kayitli kural):
  good : H07 'dogrulandi'==1 (on-kayitli DOGRULANDI: yarim-B Δ>=+0.30, bootstrap alt sinir>0, s1/s2 onayi)
  bad  : Δ <= -0.5 puan ve her iki yarida da negatif
  ns   : digerleri
Etiket konumu (anchor, dx/dy) yalniz yerlesim bilgisidir (veri degil).
Cikti: figures/data/f3_h07.dat, f3_test.dat, f3_ref.dat
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ev as E

OUT = E.MW / "figures" / "data"
OUT.mkdir(parents=True, exist_ok=True)
ROWS = ["R0", "N50", "N60", "N80", "S03", "S05", "U704", "U768", "U832", "T1", "T2", "W1", "W2"]
# overview ve zoom icin etiket yerlesimi (anchor); yalniz gorsel
ANCH_A = {"R0": "south west", "N50": "west", "N60": "west", "N80": "west", "S03": "east", "S05": "east", "U704": "west",
          "U768": "east", "U832": "east", "T1": "south east", "T2": "east", "W1": "south west", "W2": "west"}
ANCH_Z = {"R0": "south west", "N50": "west", "N60": "west", "N80": "west", "S03": "east", "S05": "east", "U704": "west"}


def shift(a):
    """etiketi isaretciden uzaklastiran kaydirma (pt): yalniz yerlesim bilgisi."""
    dx = 3 if "west" in a else (-3 if "east" in a else 0)
    dy = 2 if "south" in a else (-2 if "north" in a else 0)
    return dx, dy
lines = ["row;lat;delta;cls;anchorA;dxA;dyA;anchorZ;dxZ;dyZ;ids"]
for r in ROWS:
    lat = E.v(f"h07.{r}.lat_oran")
    d = E.v(f"h07.{r}.delta")
    dA, dB = E.v(f"h07.{r}.delta_A"), E.v(f"h07.{r}.delta_B")
    if E.v(f"h07.{r}.dogrulandi") == 1:
        c = "good"
    elif d <= -0.5 and dA < 0 and dB < 0:
        c = "bad"
    else:
        c = "ns"
    aA, aZ = ANCH_A.get(r, "north"), ANCH_Z.get(r, "north")
    lines.append(";".join([r, E.nan(lat), E.nan(d), c, aA, str(shift(aA)[0]), str(shift(aA)[1]), aZ, str(shift(aZ)[0]), str(shift(aZ)[1]), E.rid(f"h07.{r}.delta")]))
(OUT / "f3_h07.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
lat = E.v("test.latency_orani.tta")
d = E.v("dtest.map50_95.tta.ort")
lo, hi, bo = E.v("boot.tta.ci_lo"), E.v("boot.tta.ci_hi"), E.v("boot.tta.ort")
(OUT / "f3_test.dat").write_text("row;lat;delta;boot_ort;ci_lo;ci_hi;ids\n" + ";".join(
    ["TTA test (n=3)", E.nan(lat), E.nan(d), E.nan(bo), E.nan(lo), E.nan(hi),
     ",".join([E.rid("test.latency_orani.tta"), E.rid("dtest.map50_95.tta.ort"), E.rid("boot.tta.ci_lo"), E.rid("boot.tta.ci_hi")])]) + "\n", encoding="utf-8")
(OUT / "f3_ref.dat").write_text(f"budget\n{E.v('on_kayit.gercek_zaman_orani'):.6g}\n", encoding="utf-8")
print("F3 verisi yazildi")
