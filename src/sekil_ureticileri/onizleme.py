#!/usr/bin/env python
"""Tek-figurluk / tablo onizleme belgeleri uretir ve derler (Springer sn-jnl ile; \\textwidth gercek sinifla ayni).
Kullanim (MW kokunden): python figures/scripts/onizleme.py f1_nokta_aralik [f2_hata_turu ...] | tablolar
Cikti: figures/_onizleme/<ad>.pdf, <ad>_p<k>.png (110 dpi), Overfull sayisi.
Not: sn-jnl.cls orijinalinde CR-only satir sonu vardir; MiKTeX bunu okuyamaz -> _onizleme/ icine LF'e cevrilmis kopya konur.
"""
import re
import subprocess
import sys
from pathlib import Path

MW = Path(__file__).resolve().parents[2]
OZ = MW / "figures" / "_onizleme"
OZ.mkdir(exist_ok=True)
CLS = OZ / "sn-jnl.cls"
if not CLS.exists():
    CLS.write_bytes((MW.parents[1] / "sn-article-template" / "sn-jnl.cls").read_bytes().replace(b"\r", b"\n"))

HEAD = "\\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}\n\\newcommand{\\figroot}{../}\n\\input{../_preamble_figs.tex}\n\\begin{document}\n"
TAIL = "\\end{document}\n"


def build(ad):
    if ad == "tablolar":
        body = "".join(f"\\input{{../../evidence/tables/{p.stem}.tex}}\n" for p in sorted((MW / "evidence" / "tables").glob("T*.tex")))
    else:
        body = f"\\input{{../{ad}.tex}}\n"
    (OZ / f"{ad}.tex").write_text(HEAD + body + TAIL, encoding="utf-8")
    for _ in range(2):
        r = subprocess.run(["pdflatex", "--enable-installer", "-interaction=nonstopmode", f"{ad}.tex"], cwd=OZ, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = (OZ / f"{ad}.log").read_text(encoding="utf-8", errors="replace")
    err = re.findall(r"^! .*$", log, flags=re.M)
    ov = len(re.findall(r"Overfull \\[hv]box", log))
    print(f"{ad}: pdf={'var' if (OZ / (ad + '.pdf')).exists() else 'YOK'} hata={len(err)} overfull={ov}")
    for e in err[:5]:
        print("   ", e)
    try:
        import pymupdf
        d = pymupdf.open(str(OZ / f"{ad}.pdf"))
        for k, pg in enumerate(d, 1):
            pg.get_pixmap(dpi=110).save(str(OZ / f"{ad}_p{k}.png"))
        print(f"   sayfa={len(d)}")
    except Exception as ex:
        print("   png yok:", ex)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        build(a)
