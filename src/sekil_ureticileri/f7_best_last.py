#!/usr/bin/env python
"""F7 verisi (ek genelleme deneyi): KITTI test, kart basina best.pt ve last.pt Δ mAP50-95 (3 seed ort. + s.d.), model basina bir panel.
Kaynak: evidence/evidence.json `ek.test.<model>.<kart>.{delta_ort,delta_sd,last_delta_ort,last_delta_sd,karar}` (id'ler dat'ta).
Isaretci sinifi: best.pt'de Holm-anlamli iyilesme (karar == KABUL) -> isgood=1 (ucgen + mavi); diger her sey nokta (renksiz).
Cikti: figures/data/f7a_yolo26n.dat, f7b_yolo11s.dat (ayrac ';'); paper/figures/data'ya da kopyalanir.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ev as E  # noqa: E402

OUT = E.MW / "figures" / "data"
OUT.mkdir(parents=True, exist_ok=True)
PAPER = E.MW / "paper" / "figures" / "data"
PAPER.mkdir(parents=True, exist_ok=True)
CARDS = [("H01", "H01 çok ölçekli 0.25"), ("H02", "H02 Inner-CIoU"), ("H03", "H03 kutu copy-paste"), ("H04", "H04 ölçek 0.25"),
         ("H05", "H05 Varifocal cls"), ("H06", "H06 kutu kazancı 10"), ("H08", "H08 yüksek doz copy-paste")]
for fn, m in (("f7a_yolo26n.dat", "yolo26n"), ("f7b_yolo11s.dat", "yolo11s")):
    lines = ["ypos;label;best_m;best_sd;last_m;last_sd;isgood;lab_best;lab_last;ids"]
    for k, (c, lbl) in enumerate(CARDS):
        y = len(CARDS) - k
        p = f"ek.test.{m}.{c}"
        good = int(E.v(f"{p}.karar") == "KABUL")
        ids = ",".join(E.rid(f"{p}.{s}") for s in ("delta_ort", "delta_sd", "last_delta_ort", "last_delta_sd", "karar"))
        lines.append(";".join([str(y), lbl, E.nan(E.v(f"{p}.delta_ort")), E.nan(E.v(f"{p}.delta_sd")), E.nan(E.v(f"{p}.last_delta_ort")),
                               E.nan(E.v(f"{p}.last_delta_sd")), str(good), E.lab(E.v(f"{p}.delta_ort")), E.lab(E.v(f"{p}.last_delta_ort")), ids]))
    (OUT / fn).write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy(OUT / fn, PAPER / fn)
print("F7 verisi yazildi")
