"""Ortak yardimci: evidence/evidence.json okuyucu (figur ve tablo ureticileri paylasir). Veri YALNIZ evidence.json'dan gelir."""
import json
from pathlib import Path

MW = Path(__file__).resolve().parents[2]
EVID = MW / "evidence" / "evidence.json"
_K = None


def load():
    global _K
    if _K is None:
        d = json.load(open(EVID, encoding="utf-8"))
        _K = {r["anahtar"]: r for r in d["kayitlar"]}
    return _K


def v(key):
    return load()[key]["deger"]


def has(key):
    return key in load()


def rid(key):
    return load()[key]["id"]


def nan(x):
    return "nan" if x is None else f"{x:.6g}"


def lab(x, d=2, sign=True):
    """Etiket metni (TABLOLARLA AYNI yuvarlayici: Python bicimlemesi, gercek double uzerinden). H08 +0.225 -> 0.22 (E0186 = 0.2249999...).
    Figurlerde sayisal etiketler TeX'in yarim-yukari yuvarlamasina birakilmaz; metin olarak (point meta=explicit symbolic) yazilir."""
    s = f"{abs(x):.{d}f}"
    if x < 0 and float(s) != 0.0:
        return "$-$" + s
    return ("+" + s) if sign else s
