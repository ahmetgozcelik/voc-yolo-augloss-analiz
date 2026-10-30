#!/usr/bin/env python
"""F5 verisi (P1, n=3): doz (nesne/mosaic) -> Δ mAP50-95 ve Δ mAP50 (val). Noktalar: baseline (0,0), H03 (3 seed), H08 (3 seed; doz seed basina).
NOT: H08 hem yapistirma olasiligini (0.5 -> 1.0) hem aday havuzunu (<=48 -> <=96 px) degistirir; doz ile havuz ayrilmamistir (caption'da).
Kaynak: evidence.json doz.*, dval.map50_95.*, dval.map50.*, gurultu.* ; baseline sd bandi = baseline val seed std (mAP50-95 ve mAP50).
Etiketler (lab) Python'da yuvarlanir (tablolarla ayni; H08 s0 +0.225 -> 0.22).
Cikti: figures/data/f5_doz.dat (seed noktalari), f5_ort.dat (ortalamalar), f5_ref.dat
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ev as E

OUT = E.MW / "figures" / "data"
OUT.mkdir(parents=True, exist_ok=True)
pts = ["arm;n;doz;d5095;d50;seed"]
pts.append("baseline;3;0;0;0;ref")
for arm in ("h03", "h08"):
    for s in range(3):
        pts.append(f"{arm};3;{E.nan(E.v(f'doz.nesne_per_mosaic.{arm}.s{s}'))};{E.nan(E.v(f'dval.map50_95.{arm}.s{s}'))};{E.nan(E.v(f'dval.map50.{arm}.s{s}'))};{s}")
(OUT / "f5_doz.dat").write_text("\n".join(pts) + "\n", encoding="utf-8")
ort = ["arm;n;doz;d5095;d50;lab5095;lab50", f"baseline;3;0;0;0;{E.lab(0.0, sign=False)};{E.lab(0.0, sign=False)}"]
for arm, dk in (("h03", "doz.h03_ort"), ("h08", "doz.h08_ort")):
    ort.append(f"{arm};3;{E.nan(E.v(dk))};{E.nan(E.v(f'dval.map50_95.{arm}.ort'))};{E.nan(E.v(f'dval.map50.{arm}.ort'))};"
               f"{E.lab(E.v(f'dval.map50_95.{arm}.ort'))};{E.lab(E.v(f'dval.map50.{arm}.ort'))}")
(OUT / "f5_ort.dat").write_text("\n".join(ort) + "\n", encoding="utf-8")
(OUT / "f5_ref.dat").write_text("sd5095;sd50;doz_lo;doz_hi;doz_h03;doz_h08\n" + ";".join(
    [E.nan(E.v("gurultu.val_sd_baseline")), E.nan(E.v("val.map50.baseline.sd")), E.nan(E.v("doz.h08_hedef_alt")), E.nan(E.v("doz.h08_hedef_ust")),
     E.nan(E.v("doz.h03_ort")), E.nan(E.v("doz.h08_ort"))]) + "\n", encoding="utf-8")
print("F5 verisi yazildi")
