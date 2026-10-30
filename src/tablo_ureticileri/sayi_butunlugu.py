"""Sayı bütünlüğü: revize metindeki her sayının orijinalde de bulunup bulunmadığını denetler (REVIZYON_PLANI.md §6).
MW kökünden: python evidence/scripts/sayi_butunlugu.py [--rapor notes/sayi-butunlugu.md]
Orijinal: orijinal/main.tex + orijinal/tables/*.tex + orijinal/figures/*.tex (+ figures/data/*.dat)
Revize : paper/main.tex + paper/*.tex (ekler) + paper/tables/*.tex + paper/figures/*.tex (+ data/*.dat)
Hariç: yorum satırları, \\label/\\ref/\\eqref/\\cite/\\input argümanları, uzunluk/biçim komutları (pt, cm, mm, em,
\\textwidth katsayıları, renk tanımları), dosya adları. Sayılar işaret ve ondalığıyla karşılaştırılır
($-$, \\textminus, − ve - aynı işaret sayılır; "+" ayrı tutulmaz).
"""
import re
import sys
from collections import Counter
from pathlib import Path

MW = Path(__file__).resolve().parents[2]


def dosyalar(kok, ana):
    d = [kok / ana]
    for alt in ("tables", "figures"):
        d += sorted((kok / alt).glob("*.tex"))
        d += sorted((kok / alt / "data").glob("*.dat")) if (kok / alt / "data").exists() else []
    d += [p for p in sorted(kok.glob("*.tex")) if p.name != ana]
    return d


SIL = [
    r"(?<!\\)%.*",                                                     # yorum
    r"\\(?:label|ref|eqref|cite|citep|citet|input|include|includegraphics|url|href|autoref|cref)\*?(?:\[[^\]]*\])?\{[^}]*\}",
    r"\\(?:definecolor|setlength|addlinespace|vspace|hspace|resizebox|rule|arraystretch|renewcommand)\b[^\n]*",
    r"\\(?:cmidrule|cline|multicolumn|multirow)(?:\([^)]*\))?\{[^}]*\}",   # sütun aralığı / birleştirme sayıları
    r"\\(?:hyphenpenalty|exhyphenpenalty|tolerance|emergencystretch|pretolerance)\s*=?\s*\d+",   # dizgi parametreleri
    r"-?\d+(?:\.\d+)?\s*(?:pt|cm|mm|em|ex|in|bp)\b",
    r"-?\d*\.?\d+\s*\\(?:textwidth|linewidth|columnwidth|tabcolsep)",
    r"\b(?:begin|end)\{[^}]*\}",
    r"[A-Za-z_]+\d+\.(?:tex|dat|png|pdf|py|json)",
    r"\{[a-z]+(?:![0-9]+)+\}|![0-9]+",                                 # renk karışımları black!65
    r"mark size=[\d.]+|line width=[\d.]+|opacity=[\d.]+|xshift=[-\d.]+|yshift=[-\d.]+|scale=[\d.]+",
]
SAYI = re.compile(r"(?<![A-Za-z_\\])(?:\$-\$|\\textminus\s*|−|-)?\d+(?:[.,]\d+)?")


def sayilar(p):
    s = p.read_text(encoding="utf-8", errors="replace")
    for d in SIL:
        s = re.sub(d, " ", s)
    out = []
    for m in SAYI.finditer(s):
        t = m.group(0)
        t = re.sub(r"^(\$-\$|\\textminus\s*|−)", "-", t).replace(",", ".")
        out.append(t)
    return out


def normal(t):
    neg = t.startswith("-")
    g = t.lstrip("-")
    return ("-" if neg else "") + g


def main():
    rapor = None
    if "--rapor" in sys.argv:
        rapor = MW / sys.argv[sys.argv.index("--rapor") + 1]
    org = Counter(normal(x) for p in dosyalar(MW / "orijinal", "main.tex") for x in sayilar(p))
    org_mutlak = {x.lstrip("-") for x in org}
    yeni = {}
    for p in dosyalar(MW / "paper", "main.tex"):
        for x in sayilar(p):
            n = normal(x)
            if n not in org:
                yeni.setdefault(n, set()).add(p.relative_to(MW).as_posix())
    kesin = {k: v for k, v in yeni.items() if k.lstrip("-") not in org_mutlak}
    yalniz_isaret = {k: v for k, v in yeni.items() if k not in kesin}
    satir = ["# Sayı bütünlüğü raporu", "",
             f"Orijinalde farklı sayı: {len(org)}; revizede orijinalde olmayan: {len(kesin)} (yalnız işaret farkı: {len(yalniz_isaret)})", ""]
    satir.append("## Orijinalde hiç bulunmayan sayılar")
    satir += [f"- `{k}` — {', '.join(sorted(v))}" for k, v in sorted(kesin.items(), key=lambda kv: float(kv[0]))] or ["- (yok)"]
    satir += ["", "## Mutlak değeri orijinalde var, işareti farklı"]
    satir += [f"- `{k}` — {', '.join(sorted(v))}" for k, v in sorted(yalniz_isaret.items(), key=lambda kv: float(kv[0]))] or ["- (yok)"]
    metin = "\n".join(satir) + "\n"
    print(metin if len(metin) < 6000 else metin[:6000] + "\n…")
    if rapor:
        rapor.write_text(metin, encoding="utf-8")


if __name__ == "__main__":
    main()
