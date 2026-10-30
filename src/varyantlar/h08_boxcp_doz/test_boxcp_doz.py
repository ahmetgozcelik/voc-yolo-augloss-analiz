# -*- coding: utf-8 -*-
"""
H08 CPU testleri (GPU'suz): CUDA_VISIBLE_DEVICES=-1 python test_boxcp_doz.py [--batches 25] [--workers 2]
H03 test_boxcp.py ile ayni olculer, yeni parametrelerle (p=1.0, max_sqrt_alan=96); h03 dosyalari DEGISTIRILMEZ
(kur/batch_ozet ice aktarilir). Dogrulanan:
  A  Birim (tek surec, gercek mosaic ciktisi, 300 goruntu, p=1.0): sinif=kaynak sinif; kaynak sqrt(alan)<=96 ve IoA<=0.1;
     yeni kutu tuval+merkez pencere icinde; yama dikdortgeni mevcut kutu/yama ile IoU=0; yama pikselleri kaynakla birebir;
     etiket sayisi = onceki + yapistirilan; DOZ = yapistirilan/300
  B  DataLoader ISCILERI (Windows spawn, workers=2), trainer ile ayni build_yolo_dataset+build_dataloader, copy_paste=1.0,
     >=20 batch, close_mosaic yok (mosaic acik): isci sayaclari (BOXCP_FLUSH_N=1 -> tam sayim), DOZ = yapistirilan/cagri,
     DOZ KAPISI [0.6, 1.5]; kontrol (copy_paste=0) ile etiket orani; kutu gecerliligi
  C  Gorsel: ornek_canvas_*.jpg, ornek_batch_*.jpg
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

_AD = Path(__file__).resolve().parent
sys.path.insert(0, str(_AD))
sys.path.insert(0, str(_AD.parent / "h03_boxcp"))
import boxcp_doz  # noqa: E402
import boxcp  # noqa: E402
import test_boxcp as t3  # noqa: E402  (H03; yalniz kur / batch_ozet)

MAXS = boxcp_doz.MAX_SQRT_ALAN
DOZ_KAPI = (0.6, 1.5)


def testA(ds, n_img=300):
    import cv2
    from ultralytics.utils.metrics import bbox_ioa
    mosaic = ds.transforms.transforms[0].transforms[0]
    bc_sira = ds.transforms.transforms[0].transforms[1]
    assert type(bc_sira).__name__ == "BoxCopyPasteDoz" and type(bc_sira).__module__ == "boxcp_doz", type(bc_sira)
    assert bc_sira.max_sqrt_alan == 96.0 and bc_sira.p == 1.0, (bc_sira.max_sqrt_alan, bc_sira.p)
    bc = boxcp_doz.BoxCopyPasteDoz(p=1.0)
    assert bc.max_sqrt_alan == 96.0 and bc.n_max == 3 and bc.ioa_esik == 0.1 and bc.zone == "center"
    rng = random.Random(123)
    random.seed(1234)
    np.random.seed(1234)
    n_uyg = n_yap = n_pix = n_pix_yetersiz = n_cls_hata = n_cakisma = n_sinir = n_zone = n_aday_hata = n_len_hata = 0
    maxs_gorulen = 0.0
    ornekler = []
    for t in range(n_img):
        i = rng.randrange(len(ds))
        mos = mosaic(ds.get_image_and_label(i))
        H, W = mos["img"].shape[:2]
        mos_ref = copy.deepcopy(mos)
        b0 = np.asarray(mos_ref["instances"].bboxes, dtype=np.float32).copy()
        cls0 = mos_ref["cls"].copy()
        img0 = mos_ref["img"].copy()
        out = bc(copy.deepcopy(mos))
        yeni = bc.son
        nb = len(out["instances"].bboxes)
        if nb != len(b0) + len(yeni) or len(out["cls"]) != nb:
            n_len_hata += 1
        if not yeni:
            continue
        n_uyg += 1
        rects = [r for (_, _, r) in yeni]
        for k, (j, kutu, r) in enumerate(yeni):
            n_yap += 1
            tx, ty, tx2, ty2 = r
            w0, h0 = b0[j, 2] - b0[j, 0], b0[j, 3] - b0[j, 1]
            maxs_gorulen = max(maxs_gorulen, math.sqrt(w0 * h0))
            if math.sqrt(w0 * h0) > MAXS + 1e-3:
                n_aday_hata += 1
            ioa = bbox_ioa(b0, b0[j:j + 1])[:, 0]
            ioa[j] = 0
            if ioa.max() > 0.1 + 1e-6:
                n_aday_hata += 1
            if out["cls"][len(b0) + k][0] != cls0[j][0]:
                n_cls_hata += 1
            if not (0 <= kutu[0] < kutu[2] <= W and 0 <= kutu[1] < kutu[3] <= H):
                n_sinir += 1
            if not (tx >= W // 4 and tx2 <= W - W // 4 and ty >= H // 4 and ty2 <= H - H // 4):
                n_zone += 1
            digerleri = [tuple(x) for x in b0.tolist()] + [rr for q, rr in enumerate(rects) if q != k]
            for e in digerleri:
                if min(tx2, e[2]) - max(tx, e[0]) > 0 and min(ty2, e[3]) - max(ty, e[1]) > 0:
                    n_cakisma += 1
                    break
            sx1, sy1 = max(int(np.floor(b0[j, 0])), 0), max(int(np.floor(b0[j, 1])), 0)
            ph, pw = ty2 - ty, tx2 - tx
            m = 4
            if ph > 2 * m and pw > 2 * m:
                a = out["img"][ty + m:ty2 - m, tx + m:tx2 - m]
                s = img0[sy1 + m:sy1 + ph - m, sx1 + m:sx1 + pw - m]
                if a.shape == s.shape and np.array_equal(a, s):
                    n_pix += 1
                else:
                    n_pix_yetersiz += 1
            if abs((kutu[0] - tx) - (b0[j, 0] - sx1)) > 1e-3 or abs((kutu[1] - ty) - (b0[j, 1] - sy1)) > 1e-3:
                n_pix_yetersiz += 1
        if len(ornekler) < 6:
            ornekler.append((out["img"].copy(), b0.copy(), [(y[1], y[2]) for y in yeni], [b0[y[0]] for y in yeni]))
    bc0 = boxcp_doz.BoxCopyPasteDoz(p=0.0)
    lab = mosaic(ds.get_image_and_label(0))
    assert bc0(lab) is lab
    doz = n_yap / n_img
    print(f"A  {n_img} mosaic; uygulanan goruntu={n_uyg} ({n_uyg / n_img:.3f}), yapistirilan nesne={n_yap} "
          f"(DOZ = {doz:.3f} nesne/mosaic; ort {n_yap / max(n_uyg, 1):.2f}/uygulanan goruntu); "
          f"en buyuk kaynak sqrt(alan)={maxs_gorulen:.1f}")
    print(f"A  hatalar: uzunluk={n_len_hata} sinif={n_cls_hata} sinir-disi={n_sinir} merkez-pencere-disi={n_zone} "
          f"cakisma={n_cakisma} aday-kriteri={n_aday_hata} | piksel/konum hizasi: dogru={n_pix} hatali={n_pix_yetersiz}")
    assert n_uyg > 0 and n_yap > 0
    assert (n_len_hata, n_cls_hata, n_sinir, n_zone, n_cakisma, n_aday_hata, n_pix_yetersiz) == (0,) * 7
    assert n_pix > 0
    assert maxs_gorulen > 48.0, "96 px havuzu devrede degil (yapistirilanlarin hicbiri 48 px ustu degil)"
    for q, (im, b0, yeniler, kaynaklar) in enumerate(ornekler):
        g = np.ascontiguousarray(im.copy())
        for bb in b0:
            cv2.rectangle(g, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 200, 0), 1)
        for (kutu, r), src in zip(yeniler, kaynaklar):
            cv2.rectangle(g, (int(kutu[0]), int(kutu[1])), (int(kutu[2]), int(kutu[3])), (0, 0, 255), 1)
            cv2.rectangle(g, (int(src[0]), int(src[1])), (int(src[2]), int(src[3])), (255, 0, 0), 1)
        H, W = g.shape[:2]
        g = g[H // 4:H - H // 4, W // 4:W - W // 4]
        cv2.imwrite(str(_AD / f"ornek_canvas_{q}.jpg"), g)
    return dict(uyg=n_uyg, yap=n_yap, doz=doz, piksel_dogru=n_pix, hata=0)


def testB(n_batch, workers):
    import cv2
    sd = tempfile.mkdtemp(prefix="boxcp_doz_sayac_")
    os.environ["BOXCP_SAYAC_DIR"] = sd
    os.environ["BOXCP_FLUSH_N"] = "1"  # tam sayim (yalniz testte)
    main_pid = os.getpid()
    cfg, ds, dl = t3.kur(1.0, workers)
    assert type(ds.transforms.transforms[0].transforms[1]).__name__ == "BoxCopyPasteDoz"
    t_img, t_lbl, bozuk, batches = t3.batch_ozet(dl, n_batch)
    top = boxcp_doz.sayac_topla(sd)
    pids = [int(Path(f).stem) for f in Path(sd).glob("*.json")]
    doz = top["yapistirilan"] / max(top["cagri"], 1)
    print(f"B  workers={workers} batch={n_batch}: {t_img} goruntu, {t_lbl} etiket (ort {t_lbl / t_img:.3f}/goruntu)")
    print(f"B  isci sayaclari: {top} ; pid'ler={pids} ; ana pid={main_pid}")
    print(f"B  DOZ = {doz:.3f} nesne/mosaic ; uygulanan pay = {top['uygulanan_goruntu'] / max(top['cagri'], 1):.3f} ; "
          f"yer-bulunamadi payi = {top['yer_bulunamadi'] / max(top['aday_toplam'], 1):.3f} ; kapi {DOZ_KAPI}")
    assert main_pid not in pids and len(pids) >= 1
    assert top["yapistirilan"] > 0 and top["cagri"] >= 300, top
    assert bozuk == 0, "batch kutulari gecersiz"
    print(f"B  20-batch (n={top['cagri']}) DOZ KAPISI [0.6,1.5] durumu: "
          f"{'GECTI' if DOZ_KAPI[0] <= doz <= DOZ_KAPI[1] else 'GECMEDI (kucuk orneklem; kesin kapi doz_olc.py buyuk orneklem)'}")
    assert 0.4 <= doz <= DOZ_KAPI[1], f"DOZ makul araligin disinda: {doz}"  # yumusak saglik; kesin kapi doz_olc.py
    del dl
    os.environ.pop("BOXCP_SAYAC_DIR", None)
    cfg0, ds0, dl0 = t3.kur(0.0, workers)
    c_img, c_lbl, c_bozuk, _ = t3.batch_ozet(dl0, n_batch)
    print(f"B  KONTROL (copy_paste=0): {c_img} goruntu, {c_lbl} etiket (ort {c_lbl / c_img:.3f}/goruntu) ; "
          f"H08/kontrol etiket orani = {(t_lbl / t_img) / (c_lbl / c_img):.3f}")
    for q, bt in enumerate(batches):
        im = bt["img"][0].permute(1, 2, 0).numpy().copy()
        im = np.ascontiguousarray(im[:, :, ::-1])
        H, W = im.shape[:2]
        mk = bt["batch_idx"] == 0
        for bb in bt["bboxes"][mk]:
            cx, cy, w, h = [float(x) for x in bb]
            cv2.rectangle(im, (int((cx - w / 2) * W), int((cy - h / 2) * H)),
                          (int((cx + w / 2) * W), int((cy + h / 2) * H)), (0, 200, 0), 1)
        cv2.imwrite(str(_AD / f"ornek_batch_{q}.jpg"), im)
    return dict(top=top, pids=pids, doz=doz, t_lbl=t_lbl / t_img, c_lbl=c_lbl / c_img, cagri=top["cagri"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=25)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    import torch
    print("cihaz: cuda_available =", torch.cuda.is_available())
    assert not torch.cuda.is_available(), "CPU testi: CUDA_VISIBLE_DEVICES=-1 ile kos"
    boxcp_doz.uygula()
    cfg, ds, dl = t3.kur(1.0, 0)
    ra = testA(ds)
    del dl
    rb = testB(a.batches, a.workers)
    print("TUM H08 CPU TESTLERI GECTI", json.dumps({"A": ra, "B": {k: rb[k] for k in ("doz", "cagri", "t_lbl", "c_lbl")},
                                                    "B_top": rb["top"]}))


if __name__ == "__main__":
    main()
