# -*- coding: utf-8 -*-
"""
H03 CPU testleri (GPU'suz): CUDA_VISIBLE_DEVICES=-1 python test_boxcp.py [--batches 20] [--workers 2]

  A  Birim (tek surec, gercek mosaic ciktisi uzerinde, 300 goruntu): her yapistirma icin
       - sinif = kaynak sinif; kaynak sqrt(alan)<=48 ve baskasiyla IoA<=0.1
       - yeni kutu tuval icinde + merkez pencerede; yama dikdortgeni HICBIR mevcut kutu/yama ile kesismez (IoU=0)
       - yama pikselleri (kenar yumusatma haric ic bolge) kaynak pikselleriyle BIREBIR ayni  (etiket-goruntu hizasi)
       - etiket sayisi = onceki + yapistirilan; cls/instances uzunluklari esit; p=0 -> labels ayni nesne
  B  DataLoader ISCILERI (Windows spawn, workers=2): gercek build_yolo_dataset + build_dataloader (trainer ile ayni
       cagri), 20 batch: iscilerde BoxCopyPaste calisti mi (surec-basi sayac dosyalari, pid != ana surec),
       yapistirilan > 0, etiket/goruntu tutarli (kutular [0,1], w,h>0); kontrol (copy_paste=0) ile karsilastirma
  C  Gorsel: varyantlar/h03_boxcp/ornek_canvas_*.jpg (tuval, yesil=mevcut, kirmizi=yapistirilan, mavi=kaynak)
             varyantlar/h03_boxcp/ornek_batch_*.jpg (iscilerden gelen nihai egitim batch'i)
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

import numpy as np

_AD = Path(__file__).resolve().parent
sys.path.insert(0, str(_AD))
import boxcp  # noqa: E402

WS = _AD.parents[1]


def kur(copy_paste, workers, seed=0):
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    cfg = get_cfg(overrides=dict(data=os.environ.get("DOZ_DATA", str(WS / "data" / "voc_dev.yaml")), imgsz=640, batch=16, workers=workers,
                                 copy_paste=copy_paste, seed=seed, device="cpu", cache=False))
    data = check_det_dataset(cfg.data)
    torch.manual_seed(seed)
    ds = build_yolo_dataset(cfg, data["train"], 16, data, mode="train", rect=False, stride=32)
    dl = build_dataloader(ds, batch=16, workers=workers, shuffle=True, rank=-1, device="cpu")
    return cfg, ds, dl


def testA(ds, n_img=300):
    import cv2
    from ultralytics.utils.metrics import bbox_ioa
    mosaic = ds.transforms.transforms[0].transforms[0]
    bc_sira = ds.transforms.transforms[0].transforms[1]
    assert type(bc_sira).__name__ == "BoxCopyPaste" and type(bc_sira).__module__ == "boxcp", type(bc_sira)
    bc = boxcp.BoxCopyPaste(p=1.0)
    rng = random.Random(123)
    random.seed(1234)
    np.random.seed(1234)
    n_uyg = n_yap = n_pix = n_pix_yetersiz = n_cls_hata = n_cakisma = n_sinir = n_zone = n_aday_hata = 0
    n_len_hata = 0
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
        bo = np.asarray(out["instances"].bboxes, dtype=np.float32)
        rects = [r for (_, _, r) in yeni]
        for k, (j, kutu, r) in enumerate(yeni):
            n_yap += 1
            tx, ty, tx2, ty2 = r
            w0, h0 = b0[j, 2] - b0[j, 0], b0[j, 3] - b0[j, 1]
            if math.sqrt(w0 * h0) > 48.0 + 1e-3:
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
            # cakisma: yama dikdortgeni mevcut GT'ler ve diger yamalarla kesismemeli
            digerleri = [tuple(x) for x in b0.tolist()] + [rr for q, rr in enumerate(rects) if q != k]
            for e in digerleri:
                if min(tx2, e[2]) - max(tx, e[0]) > 0 and min(ty2, e[3]) - max(ty, e[1]) > 0:
                    n_cakisma += 1
                    break
            # piksel hizasi: ic bolge birebir kaynak
            sx1, sy1 = int(np.floor(b0[j, 0])), int(np.floor(b0[j, 1]))
            sx1, sy1 = max(sx1, 0), max(sy1, 0)
            ph, pw = ty2 - ty, tx2 - tx
            m = 4
            if ph > 2 * m and pw > 2 * m:
                a = out["img"][ty + m:ty2 - m, tx + m:tx2 - m]
                s = img0[sy1 + m:sy1 + ph - m, sx1 + m:sx1 + pw - m]
                if a.shape == s.shape and np.array_equal(a, s):
                    n_pix += 1
                else:
                    n_pix_yetersiz += 1
            # kutunun yamaya gore konumu ayni mi: kutu - yama kaydirmasi = kaynak kutu - kaynak yama kaydirmasi
            if abs((kutu[0] - tx) - (b0[j, 0] - sx1)) > 1e-3 or abs((kutu[1] - ty) - (b0[j, 1] - sy1)) > 1e-3:
                n_pix_yetersiz += 1
        if len(ornekler) < 6 and len(yeni) >= 1:
            ornekler.append((out["img"].copy(), b0.copy(), [(y[1], y[2]) for y in yeni], [b0[y[0]] for y in yeni]))
    # p=0 -> ayni nesne
    bc0 = boxcp.BoxCopyPaste(p=0.0)
    lab = mosaic(ds.get_image_and_label(0))
    assert bc0(lab) is lab
    print(f"A  {n_img} mosaic goruntusu; yapistirma uygulanan goruntu={n_uyg} (p=1.0), yapistirilan nesne={n_yap} "
          f"(ort {n_yap / max(n_uyg, 1):.2f}/goruntu)")
    print(f"A  hatalar: uzunluk={n_len_hata} sinif={n_cls_hata} sinir-disi={n_sinir} merkez-pencere-disi={n_zone} "
          f"cakisma={n_cakisma} aday-kriteri={n_aday_hata} | piksel/konum hizasi: dogru={n_pix} hatali={n_pix_yetersiz}")
    assert n_uyg > 0 and n_yap > 0
    assert (n_len_hata, n_cls_hata, n_sinir, n_zone, n_cakisma, n_aday_hata, n_pix_yetersiz) == (0,) * 7
    assert n_pix > 0
    # gorsel
    for q, (im, b0, yeniler, kaynaklar) in enumerate(ornekler):
        g = np.ascontiguousarray(im[:, :, ::-1].copy()) if False else np.ascontiguousarray(im.copy())
        for bb in b0:
            cv2.rectangle(g, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 200, 0), 1)
        for (kutu, r), src in zip(yeniler, kaynaklar):
            cv2.rectangle(g, (int(kutu[0]), int(kutu[1])), (int(kutu[2]), int(kutu[3])), (0, 0, 255), 1)
            cv2.rectangle(g, (int(src[0]), int(src[1])), (int(src[2]), int(src[3])), (255, 0, 0), 1)
        H, W = g.shape[:2]
        g = g[H // 4:H - H // 4, W // 4:W - W // 4]  # merkez pencere (egitimde kalan bolge)
        g = cv2.resize(g, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(_AD / f"ornek_canvas_{q}.jpg"), g)
    return dict(uyg=n_uyg, yap=n_yap, piksel_dogru=n_pix)


def batch_ozet(dl, n_batch):
    import torch
    toplam_img = toplam_lbl = 0
    bozuk = 0
    batches = []
    for bi, batch in enumerate(dl):
        if bi >= n_batch:
            break
        B = batch["img"].shape[0]
        bx = batch["bboxes"]
        toplam_img += B
        toplam_lbl += bx.shape[0]
        if bx.numel():
            if not (bool((bx[:, :2] >= 0).all()) and bool((bx[:, :2] <= 1).all()) and bool((bx[:, 2:] > 0).all())
                    and bool((bx[:, 2:] <= 1).all())):
                bozuk += 1
            xyxy = torch.cat((bx[:, :2] - bx[:, 2:] / 2, bx[:, :2] + bx[:, 2:] / 2), 1)
            if bool((xyxy < -1e-3).any()) or bool((xyxy > 1 + 1e-3).any()):
                bozuk += 1
        if len(batches) < 3:
            batches.append({k: (v.clone() if hasattr(v, "clone") else v) for k, v in batch.items()
                            if k in ("img", "bboxes", "cls", "batch_idx")})
    return toplam_img, toplam_lbl, bozuk, batches


def testB(n_batch, workers):
    import cv2
    sd = tempfile.mkdtemp(prefix="boxcp_sayac_")
    os.environ["BOXCP_SAYAC_DIR"] = sd
    main_pid = os.getpid()
    cfg, ds, dl = kur(0.5, workers)
    t_img, t_lbl, bozuk, batches = batch_ozet(dl, n_batch)
    # sayaclar: son <50 cagri yazilmamis olabilir -> kisa bekleme gerekmez; dosyalar periyodik yazildi
    top = boxcp.sayac_topla(sd)
    pids = [int(Path(f).stem) for f in Path(sd).glob("*.json")]
    print(f"B  workers={workers} batch={n_batch}: {t_img} goruntu, {t_lbl} etiket (ort {t_lbl / t_img:.3f}/goruntu)")
    print(f"B  isci sayaclari: {top} ; pid'ler={pids} ; ana pid={main_pid}")
    assert main_pid not in pids and len(pids) >= 1, "sayac dosyalari ana surecten gelmemeli"
    assert top["yapistirilan"] > 0, "iscilerde yapistirma yapilmadi"
    assert bozuk == 0, "batch kutulari gecersiz"
    del dl
    os.environ.pop("BOXCP_SAYAC_DIR", None)
    # kontrol: ayni tohum, copy_paste=0
    cfg0, ds0, dl0 = kur(0.0, workers)
    c_img, c_lbl, c_bozuk, _ = batch_ozet(dl0, n_batch)
    print(f"B  KONTROL (copy_paste=0, yerlesik CopyPaste baseline gibi inert): {c_img} goruntu, {c_lbl} etiket "
          f"(ort {c_lbl / c_img:.3f}/goruntu) ; BoxCP/kontrol etiket orani = {(t_lbl / t_img) / (c_lbl / c_img):.3f}")
    assert type(ds0.transforms.transforms[0].transforms[1]).__name__ == "BoxCopyPaste"  # p=0 -> inert
    # gorsel: iscilerden gelen batch'ler
    for q, bt in enumerate(batches):
        im = bt["img"][0].permute(1, 2, 0).numpy().copy()
        im = np.ascontiguousarray(im[:, :, ::-1])  # RGB -> BGR
        H, W = im.shape[:2]
        m = bt["batch_idx"] == 0
        for bb in bt["bboxes"][m]:
            cx, cy, w, h = [float(x) for x in bb]
            cv2.rectangle(im, (int((cx - w / 2) * W), int((cy - h / 2) * H)), (int((cx + w / 2) * W), int((cy + h / 2) * H)),
                          (0, 200, 0), 1)
        cv2.imwrite(str(_AD / f"ornek_batch_{q}.jpg"), im)
    return dict(top=top, pids=pids, t_lbl=t_lbl / t_img, c_lbl=c_lbl / c_img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    import torch
    print("cihaz: cuda_available =", torch.cuda.is_available())
    boxcp.uygula()  # kos.py'nin yaptigi atama
    cfg, ds, dl = kur(0.5, 0)
    ra = testA(ds)
    del dl
    rb = testB(a.batches, a.workers)
    print("TUM H03 CPU TESTLERI GECTI", json.dumps({"A": ra, "B_top": rb["top"], "B_etiket_ort": [rb["t_lbl"], rb["c_lbl"]]}))


if __name__ == "__main__":
    main()
