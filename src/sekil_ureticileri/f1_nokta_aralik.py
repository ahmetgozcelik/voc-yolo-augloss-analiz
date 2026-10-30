#!/usr/bin/env python
"""F1 verisi (P1, n=3): kart basina val Δ mAP50-95 (3 seed noktasi + ortalama) + test paneli (H03, TTA). Kaynak: evidence/evidence.json (anahtar ids dat'ta).
Isaretci/renk sinifi (T2/T3 ile ayni on-kayitli kural):
  val  : onceden tanimli 7'li aile, Holm-7 p<0.05 ise anlamli -> good3 (Δ>0) | bad3 (Δ<0); aksi ns3 (yalniz H05 bad3; H02 ham p=0.041 ama Holm-7 0.244 -> ns3). TTA val 7'li ailenin uyesi degil -> ns3. (hakem tur-2 N6)
  test : 12'li kesifsel Holm p<0.05 VE %95 bootstrap GA sifiri kesmiyor ise anlamli; aksi ns3 (TTA Holm-12 = 0.222 -> ns3; T5 ile tutarli)
Sayisal etiketler (lab) Python'da yuvarlanir (tablolarla ayni; H08 yuvarlama tutarliligi).
Cikti: figures/data/f1a_val.dat, f1b_test.dat, f1_ref.dat  (ayrac ';').
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ev as E

OUT = E.MW / "figures" / "data"
OUT.mkdir(parents=True, exist_ok=True)
sd = E.v("gurultu.val_sd_baseline")
thr = E.v("on_kayit.on_eleme_esigi")
acc = E.v("on_kayit.tut_esigi")
ROWS = [("h01", "H01 çok ölçekli 0.25"), ("h02", "H02 Inner-CIoU"), ("h03", "H03 kutu copy-paste"), ("h04", "H04 ölçek 0.25"),
        ("h05", "H05 Varifocal cls"), ("h06", "H06 kutu kazancı 10"), ("h08", "H08 yüksek doz copy-paste"), ("tta", "TTA (eğitimsiz)")]


def cls_val(arm):
    if arm == "tta":      # TTA val Δ 7'li Holm ailesinin uyesi degil -> renksiz
        return "ns3"
    p = E.v(f"dval.map50_95.{arm}.holm7")
    mean = E.v(f"dval.map50_95.{arm}.ort")
    return ("good3" if mean > 0 else "bad3") if p < 0.05 else "ns3"


lines = ["ypos;label;n;mean;s0;s1;s2;cls;lab;ids"]
for k, (arm, lbl) in enumerate(ROWS):
    y = len(ROWS) - k
    if arm == "tta":
        s = [E.v(f"tta_val.delta.s{i}") for i in range(3)]
        mean = E.v("tta_val.delta.ort")
        ids = ",".join(E.rid(f"tta_val.delta.s{i}") for i in range(3))
    else:
        s = [E.v(f"dval.map50_95.{arm}.s{i}") for i in range(3)]
        mean = E.v(f"dval.map50_95.{arm}.ort")
        ids = ",".join([E.rid(f"dval.map50_95.{arm}.s{i}") for i in range(3)] + [E.rid(f"dval.map50_95.{arm}.ort"), E.rid(f"dval.map50_95.{arm}.p_cift")])
    lines.append(";".join([str(y), lbl, "3", E.nan(mean)] + [E.nan(x) for x in s] + [cls_val(arm), E.lab(mean), ids]))
(OUT / "f1a_val.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")

# test paneli
lines = ["ypos;label;n;mean;s0;s1;s2;cls;isns;isgood;isbad;val_mean;arrow_u;boot_ort;ci_lo;ci_hi;ci_ypos;lab;ids"]
for k, (arm, lbl) in enumerate((("h03", "H03 kutu copy-paste"), ("tta", "TTA (eğitimsiz)"))):
    y = 2 - k
    s = [E.v(f"dtest.map50_95.{arm}.s{i}") for i in range(3)]
    mean = E.v(f"dtest.map50_95.{arm}.ort")
    holm12 = E.v(f"holm12.{arm}.map50_95")
    lo, hi, bo = E.v(f"boot.{arm}.ci_lo"), E.v(f"boot.{arm}.ci_hi"), E.v(f"boot.{arm}.ort")
    cl = ("good3" if mean > 0 else "bad3") if holm12 < 0.05 and (lo > 0 or hi < 0) else "ns3"
    vm = E.v("dval.map50_95.h03.ort") if arm == "h03" else E.v("tta_val.delta.ort")
    ids = ",".join([E.rid(f"dtest.map50_95.{arm}.s{i}") for i in range(3)] + [E.rid(f"boot.{arm}.ci_lo"), E.rid(f"boot.{arm}.ci_hi"), E.rid(f"holm12.{arm}.map50_95")])
    lines.append(";".join([str(y), f"{lbl}", "3", E.nan(mean)] + [E.nan(x) for x in s] + [cl, str(int(cl.startswith("ns"))), str(int(cl.startswith("good"))), str(int(cl.startswith("bad"))),
                                                                                         E.nan(vm), E.nan(mean - vm), E.nan(bo), E.nan(lo), E.nan(hi), E.nan(y - 0.28), E.lab(mean), ids]))
(OUT / "f1b_test.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
(OUT / "f1_ref.dat").write_text(f"noise_sd;threshold;accept\n{sd:.6g};{thr:.6g};{acc:.6g}\n", encoding="utf-8")
print("F1 verisi yazildi:", sd, thr, acc)
