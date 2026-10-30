# -*- coding: utf-8 -*-
"""
H05 CPU birim testleri (GPU'suz; CUDA_VISIBLE_DEVICES=-1 ile kos):
  T0  elle hesap: kucuk ornekte (p,t) -> beklenen VFL degerleri (saf-python math ile) ile |fark|<1e-9
  T1  negatif hucrede w=alpha*sigmoid^2, pozitifte w=q; float64 gradcheck; float32 gradyan sonlu
  T2  p->-inf, t=0: kayip->0, NaN/inf yok; p->+inf, t>0: sonlu; fp16 girdi -> float32 cikti
  T4  alpha=1, gamma=0 ve y zorla 0 iken w==1 => cikti nn.BCEWithLogitsLoss(reduction='none') ile birebir (|fark|<1e-6)
  T5  egitim yolunda gradyan-norm olcumu ve egitim/val cagri ayrimi
  T3  GERCEK v8DetectionLoss yolu (yolo11n.pt, sentetik batch): SAYAC>0 (ana surec); model gradyanlari sonlu;
      cls_loss == sum(vfl(p,t))/target_scores_sum*cls (NORMALIZASYON KORUNUYOR); box_loss ve dfl_loss BCE'li yolla BIREBIR AYNI
Kullanim: CUDA_VISIBLE_DEVICES=-1 python test_vfl.py
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

_AD = Path(__file__).resolve().parent
sys.path.insert(0, str(_AD))
import vfl_patch as P  # noqa: E402

torch.manual_seed(0)
A, G = 0.75, 2.0


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def elle(p, t, alpha=A, gamma=G):
    """Saf-python referans: BCE(p,t) * w."""
    s = sig(p)
    bce = -(t * math.log(s) + (1 - t) * math.log(1 - s))
    y = 1.0 if t > 0 else 0.0
    w = alpha * s ** gamma * (1 - y) + t * y
    return bce * w


def t0():
    ornek = [(0.0, 0.0), (0.0, 0.6), (2.0, 0.0), (-3.0, 0.9), (1.5, 0.25), (-1.0, 0.0), (4.0, 1.0)]
    p = torch.tensor([o[0] for o in ornek], dtype=torch.float64)
    t = torch.tensor([o[1] for o in ornek], dtype=torch.float64)
    cikti = P.vfl_elementwise(p, t, A, G)
    beklenen = torch.tensor([elle(*o) for o in ornek], dtype=torch.float64)
    d = (cikti - beklenen).abs().max().item()
    # bilinen sabitler: p=0,t=0 -> ln2*0.75*0.25 ; p=0,t=0.6 -> ln2*0.6
    assert abs(cikti[0].item() - math.log(2) * 0.75 * 0.25) < 1e-9
    assert abs(cikti[1].item() - math.log(2) * 0.6) < 1e-9
    print("T0 elle hesap: " + " ".join(f"({o[0]:+.1f},{o[1]:.2f})->{c:.6f}" for o, c in zip(ornek, cikti.tolist()))
          + f" | max|fark| (saf-python)={d:.2e}")
    assert d < 1e-9
    return d


def t1():
    p = torch.tensor([0.0, 1.0, -2.0, 0.5], dtype=torch.float64)
    t = torch.tensor([0.0, 0.0, 0.7, 0.3], dtype=torch.float64)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(p, t, reduction="none")
    cikti = P.vfl_elementwise(p, t, A, G)
    w = cikti / bce
    w_bekl = torch.tensor([A * sig(0.0) ** 2, A * sig(1.0) ** 2, 0.7, 0.3], dtype=torch.float64)
    dw = (w - w_bekl).abs().max().item()
    print(f"T1 agirlik w={[round(x, 5) for x in w.tolist()]} beklenen={[round(x, 5) for x in w_bekl.tolist()]} |fark|={dw:.2e}")
    assert dw < 1e-9
    pp = torch.randn(40, dtype=torch.float64, requires_grad=True)
    tt = (torch.rand(40, dtype=torch.float64) * (torch.rand(40) > 0.6).double())
    ok = torch.autograd.gradcheck(lambda x: P.vfl_elementwise(x, tt, A, G), (pp,), eps=1e-6, atol=1e-6)
    pf = torch.randn(1000, dtype=torch.float32, requires_grad=True)
    tf = torch.rand(1000) * (torch.rand(1000) > 0.9)
    P.vfl_elementwise(pf, tf, A, G).sum().backward()
    sonlu = torch.isfinite(pf.grad).all().item()
    print(f"T1 gradcheck(float64)={ok} | float32 gradyan sonlu={sonlu} ||g||={pf.grad.norm().item():.4f}")
    assert ok and sonlu
    return dw


def t2():
    p = torch.full((100,), -60.0, requires_grad=True)
    t = torch.zeros(100)
    L = P.vfl_elementwise(p, t, A, G).sum()
    L.backward()
    p2 = torch.full((10,), 60.0, requires_grad=True)
    t2_ = torch.full((10,), 0.8)
    L2 = P.vfl_elementwise(p2, t2_, A, G).sum()
    L2.backward()
    m = P.VFLElementwise(A, G)
    o = m(torch.randn(2, 5, 3).half(), torch.rand(2, 5, 3).half())
    print(f"T2 p=-60,t=0: kayip={L.item():.3e} grad sonlu={torch.isfinite(p.grad).all().item()} | "
          f"p=+60,t=0.8: kayip={L2.item():.3e} sonlu={torch.isfinite(L2).item()} | fp16 girdi -> cikti dtype={o.dtype}")
    assert L.item() < 1e-20 and torch.isfinite(p.grad).all() and torch.isfinite(L2) and torch.isfinite(p2.grad).all()
    assert o.dtype == torch.float32 and torch.isfinite(o).all()


def t4():
    """alpha=1,gamma=0, y zorla 0 => w = 1 => BCE ile birebir."""
    p = torch.randn(3, 50, 20)
    t = torch.rand(3, 50, 20) * (torch.rand(3, 50, 20) > 0.8)

    def vfl_y0(p, t, alpha=1.0, gamma=0.0):
        y = torch.zeros_like(t)  # y zorla 0
        w = alpha * torch.sigmoid(p).pow(gamma) * (1 - y) + t * y
        return torch.nn.functional.binary_cross_entropy_with_logits(p, t, reduction="none") * w

    ref = nn.BCEWithLogitsLoss(reduction="none")(p, t)
    d = (vfl_y0(p, t) - ref).abs().max().item()
    t0_ = torch.zeros_like(t)
    d2 = (P.VFLElementwise(1.0, 0.0)(p, t0_) - nn.BCEWithLogitsLoss(reduction="none")(p, t0_)).abs().max().item()
    print(f"T4 alpha=1,gamma=0,y=0 => BCE: |fark| (formul)={d:.2e}; modul (t=0 girdi) |fark|={d2:.2e}")
    assert d < 1e-6 and d2 < 1e-6
    return d


def gercek_model():
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG
    from ultralytics.utils import IterableSimpleNamespace
    ws = _AD.parents[1]
    model = YOLO(str(ws / "rakip" / "weights" / "yolo11n.pt")).model
    model.args = IterableSimpleNamespace(**{**vars(DEFAULT_CFG)})
    model.train()
    for p_ in model.parameters():
        p_.requires_grad_(True)
    return model


def batch_uret():
    imgs = torch.rand(2, 3, 320, 320)
    return imgs, {
        "img": imgs,
        "batch_idx": torch.tensor([0., 0., 1.]),
        "cls": torch.tensor([[14.], [6.], [3.]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.5], [0.25, 0.3, 0.1, 0.12], [0.7, 0.6, 0.3, 0.3]]),
    }


def t3():
    model = gercek_model()
    imgs, batch = batch_uret()
    # (a) BCE'li referans (yama yok)
    P.geri_al()
    model.criterion = None
    preds = model(imgs)
    loss_b, items_b = model.loss(batch, preds)
    # (b) VFL'li; (p, t) yakalanir
    P.uygula(A, G)
    model.criterion = None  # yeni v8DetectionLoss ornegi -> yamali __init__
    yakala = {}
    orig_fwd = P.VFLElementwise.forward

    def sar(self, pred, target):
        yakala["p"], yakala["t"] = pred.detach().clone(), target.detach().clone()
        return orig_fwd(self, pred, target)

    P.VFLElementwise.forward = sar
    P.SAYAC.update(cagri=0, cagri_val=0, grad_olcum_n=0)
    try:
        preds = model(imgs)
        loss_v, items_v = model.loss(batch, preds)
        loss_v.sum().backward()
    finally:
        P.VFLElementwise.forward = orig_fwd
    gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters() if p.grad is not None)).item()
    sayac = P.SAYAC["cagri"]
    assert isinstance(model.criterion.bce, P.VFLElementwise), "self.bce VFL modulu degil"
    cls_kazanc = float(model.args.cls)
    tss = yakala["t"].sum().clamp(min=1)
    bekl_cls = (P.vfl_elementwise(yakala["p"].float(), yakala["t"].float(), A, G).sum() / tss * cls_kazanc).item()
    bekl_bce = (torch.nn.functional.binary_cross_entropy_with_logits(
        yakala["p"].float(), yakala["t"].float(), reduction="none").sum() / tss * cls_kazanc).item()
    cls_v, cls_b = float(items_v["cls_loss"]), float(items_b["cls_loss"])
    d_norm = abs(cls_v - bekl_cls)
    d_ref = abs(cls_b - bekl_bce)
    d_box = abs(float(items_v["box_loss"]) - float(items_b["box_loss"]))
    d_dfl = abs(float(items_v["dfl_loss"]) - float(items_b["dfl_loss"]))
    sonlu = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(f"T3 v8DetectionLoss: SAYAC(cagri)={sayac} ||grad||={gn:.4f} sonlu_grad={sonlu}")
    print(f"T3 cls_loss: VFL={cls_v:.6f} (elle sum/target_scores_sum*cls={bekl_cls:.6f}, |fark|={d_norm:.2e}) "
          f"BCE={cls_b:.6f} (elle={bekl_bce:.6f}, |fark|={d_ref:.2e}) | VFL/BCE cls orani={cls_v / cls_b:.4f}")
    print(f"T3 box_loss |VFL-BCE|={d_box:.2e}  dfl_loss |VFL-BCE|={d_dfl:.2e}  (kutu/DFL kayiplari degismemeli)")
    print(f"T3 target_scores_sum={float(tss):.4f} cls_kazanc={cls_kazanc}")
    assert sayac >= 1, "VFL modulu cagrilmadi"
    assert torch.isfinite(loss_v).all() and sonlu
    assert d_norm < 1e-5 and d_ref < 1e-5, "cls normalizasyonu (target_scores_sum) bozuldu"
    assert d_box < 1e-6 and d_dfl < 1e-6, "kutu/DFL kaybi degisti (tek-degisken ihlali)"
    return sayac


def t5_sayac_grad_olcumu():
    """Modul egitim yolunda grad olcumu (ilk cagri) uretiyor mu; egitim/val ayrimi."""
    P.SAYAC.update(cagri=0, cagri_val=0, grad_olcum_n=0, grad_vfl_toplam=0.0, grad_bce_toplam=0.0,
                   kayip_vfl_toplam=0.0, kayip_bce_toplam=0.0, olcum_poz_n=0)
    m = P.VFLElementwise(A, G)
    p = torch.randn(2, 100, 20, requires_grad=True)
    t = torch.rand(2, 100, 20) * (torch.rand(2, 100, 20) > 0.97)
    m(p, t).sum().backward()
    s = P.SAYAC
    oran = s["grad_vfl_toplam"] / s["grad_bce_toplam"]
    print(f"T5 grad olcumu: n={s['grad_olcum_n']} VFL/BCE grad norm orani={oran:.4f} kayip orani="
          f"{s['kayip_vfl_toplam'] / s['kayip_bce_toplam']:.4f}")
    assert s["grad_olcum_n"] == 1 and 0 < oran < 1.0
    with torch.no_grad():
        m(p, t)
    assert s["cagri"] == 1 and s["cagri_val"] == 1, "egitim/val cagrisi ayrimi bozuk"


if __name__ == "__main__":
    print("cihaz: cuda_available =", torch.cuda.is_available())
    t0(); t1(); t2(); t4(); t5_sayac_grad_olcumu(); t3()
    print("TUM H05 CPU TESTLERI GECTI")
