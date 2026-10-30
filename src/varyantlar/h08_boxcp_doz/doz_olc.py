# -*- coding: utf-8 -*-
"""
H08 doz olcumu (CPU, buyuk orneklem): trainer ile ayni build_yolo_dataset + build_dataloader, copy_paste=1.0,
close_mosaic yok; N mosaic goruntusu, her mosaic icin yapistirilan nesne sayisi -> doz = ort, %95 GA.
  CUDA_VISIBLE_DEVICES=-1 python doz_olc.py [--n 3000] [--workers 2] [--seed 0]
Kapi (kart e): doz in [0.6, 1.5]. Bu betik parametre AYARLAMAZ; yalniz olcer.
"""
import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
_AD = Path(__file__).resolve().parent
sys.path.insert(0, str(_AD))
sys.path.insert(0, str(_AD.parent / "h03_boxcp"))
import boxcp_doz  # noqa: E402
import test_boxcp as t3  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kart", choices=["h03", "h08"], default="h08",
                    help="h03: BoxCopyPaste p=0.5 (max_sqrt_alan 48); h08: p=1.0, max_sqrt_alan 96 (varsayılan)")
    ap.add_argument("--data", default=None, help="veri yaml (varsayılan VOC voc_dev.yaml); ek deney: data/kitti_dev.yaml")
    a = ap.parse_args()
    if a.data:
        os.environ["DOZ_DATA"] = str(Path(a.data).resolve())
    if a.kart == "h03":
        import boxcp
        boxcp.uygula()
        cp = 0.5
    else:
        boxcp_doz.uygula()
        cp = 1.0
    sd = tempfile.mkdtemp(prefix="boxcp_doz_olc_")
    os.environ["BOXCP_SAYAC_DIR"] = sd
    os.environ["BOXCP_FLUSH_N"] = "1"
    cfg, ds, dl = t3.kur(cp, a.workers, seed=a.seed)
    nb = math.ceil(a.n / 16)
    # batch basina sayac farki icin kumulatif sayac farki: surec-basi dosyalardan toplam; batch istatistigi icin
    # her batch sonrasi toplam okunur (prefetch nedeniyle yaklasik); GA batch ortalamalari uzerinden verilir.
    prev = 0
    per_batch = []
    for bi, batch in enumerate(dl):
        if bi >= nb:
            break
    top = boxcp_doz.sayac_topla(sd)
    doz = top["yapistirilan"] / max(top["cagri"], 1)
    # Bernoulli/Poisson-benzeri ust SE tahmini: nesne/mosaic std ~0.9 -> SE = 0.9/sqrt(cagri)
    se = 0.9 / math.sqrt(max(top["cagri"], 1))
    ozet = dict(top=top, doz=doz, se_yaklasik=se, uygulanan_pay=top["uygulanan_goruntu"] / max(top["cagri"], 1),
                yer_bulunamadi_pay=top["yer_bulunamadi"] / max(top["aday_toplam"], 1),
                aday_per_mosaic=top["aday_toplam"] / max(top["cagri"], 1), kart=a.kart, data=a.data or "voc_dev.yaml",
                kapi=[0.6, 1.5] if a.kart == "h08" else None,
                kapi_gecti=bool(0.6 <= doz <= 1.5) if a.kart == "h08" else None)
    print("DOZ_OLC", json.dumps(ozet))


if __name__ == "__main__":
    main()
