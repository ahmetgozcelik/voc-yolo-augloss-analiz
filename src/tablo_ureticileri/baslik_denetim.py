#!/usr/bin/env python
"""Baslik ekonomisi denetimi (figure-style E2): figur basligi <= 40 kelime, tablo basligi <= 25 kelime. TUM caption sayilir (F6 'Secim olcutu' istisnasi
kaldirildi: Faz 5 hakem bulgusu -- eski denetim secim cumlesini disarida birakip f6'yi 'OK' gosteriyordu).
Calistirma (MW kokunden): python evidence/scripts/baslik_denetim.py   (cikis kodu 2 = butce asildi)"""
import glob
import re
import sys
from pathlib import Path

MW = Path(__file__).resolve().parents[2]
rc = 0
for f in sorted(glob.glob(str(MW / "figures" / "f*.tex"))) + sorted(glob.glob(str(MW / "evidence" / "tables" / "T*.tex"))):
    s = open(f, encoding="utf-8").read()
    m = re.search(r"\\caption\{(.*)\}\\label", s, re.S)
    cap = m.group(1)
    clean = lambda t: len(re.sub(r"\\[a-zA-Z]+\{?|[{}$]", " ", t).split())
    w = clean(cap)
    limit = 40 if "figures" in f.replace("\\", "/") else 25
    ok = w <= limit
    rc = 0 if (ok and rc == 0) else 2
    print(f"{Path(f).name}: {w} kelime (limit {limit}) {'OK' if ok else 'ASTI'}")
sys.exit(rc)
