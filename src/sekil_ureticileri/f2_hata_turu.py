#!/usr/bin/env python
"""F2 verisi: oracle ΔAP50 (puan) hata turu bazinda; val (baseline n=3, H03 n=3, H06 n=1) ve test (baseline n=3, H03 n=3).
Kaynak: evidence/evidence.json hv.oracle_dap50.* (WS/sonuclar/hata-analizi*.md.json) ve ht.oracle_dap50.* (hata-analizi-test_*.md.json).
Sutunlar: tur; <arm>_mean, <arm>_lo, <arm>_hi (seed min-max; n=1'de lo=hi=mean), eksik arm 'nan'. Hata cubugu: <arm>_em (mean-lo), <arm>_ep (hi-mean).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ev as E

OUT = E.MW / "figures" / "data"
OUT.mkdir(parents=True, exist_ok=True)
TYPES = ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]


def write(prefix, arms, fname):
    cols = ["tur"]
    for a in arms:
        cols += [f"{a}_mean", f"{a}_em", f"{a}_ep"]
    lines = [";".join(cols)]
    for t in TYPES:
        row = [t]
        for a, seeds in arms.items():
            x = [E.v(f"{prefix}.oracle_dap50.{t}.{a}.s{s}") for s in seeds]
            m = float(np.mean(x))
            row += [E.nan(m), E.nan(m - min(x)), E.nan(max(x) - m)]
        lines.append(";".join(row))
    (OUT / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")


write("hv", {"baseline": (0, 1, 2), "h03": (0, 1, 2), "h06": (0,)}, "f2a_val.dat")
write("ht", {"baseline": (0, 1, 2), "h03": (0, 1, 2)}, "f2b_test.dat")
print("F2 verisi yazildi")
