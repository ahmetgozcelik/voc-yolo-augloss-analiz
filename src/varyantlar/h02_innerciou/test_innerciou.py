# -*- coding: utf-8 -*-
"""
H02 CPU birim testleri (GPU'suz; CUDA_VISIBLE_DEVICES=-1 ile kos):
  T1  ratio=1.0: yamali BboxLoss.forward == orijinal (loss_iou ve loss_dfl), atol 1e-6
  T2  ratio=0.75: yuksek-IoU orneklerde kayip buyur ve kutu-gradyani daha DIK (beklenen yon); gradyan sonlu
  T3  gercek v8DetectionLoss yolundan (yolo11n.pt, sentetik batch): yamali forward ANA surecte cagriliyor (SAYAC>0),
      kayip ve model gradyanlari sonlu
  T4  inner_iou(ratio=1) == iou
Kullanim: CUDA_VISIBLE_DEVICES=-1 python test_innerciou.py
"""
import sys
from pathlib import Path

import torch

_AD = Path(__file__).resolve().parent
sys.path.insert(0, str(_AD))
import innerciou_patch as P  # noqa: E402
from ultralytics.utils.loss import BboxLoss  # noqa: E402
from ultralytics.utils.metrics import bbox_iou  # noqa: E402

torch.manual_seed(0)


def sentetik(n_img=4, n_anchor=400, reg_max=16, yakin=False):
    """BboxLoss.forward girdisi: grid birimli xyxy; fg_mask ~%10."""
    B, A = n_img, n_anchor
    ap = torch.rand(A, 2) * 40
    tb = torch.zeros(B, A, 4)
    wh = 2 + torch.rand(B, A, 2) * 12
    ctr = ap.unsqueeze(0) + (torch.rand(B, A, 2) - 0.5)
    tb[..., :2] = ctr - wh / 2
    tb[..., 2:] = ctr + wh / 2
    jit = (torch.randn(B, A, 4) * (0.05 if yakin else 0.6))
    pb = (tb + jit).clone().requires_grad_(True)
    pred_dist = torch.randn(B, A, 4 * reg_max, requires_grad=True)
    ts = torch.rand(B, A, 20) * (torch.rand(B, A, 1) > 0.5)
    fg = torch.rand(B, A) < 0.10
    fg[0, 0] = True
    return pred_dist, pb, ap, tb, ts, ts.sum().clamp(min=1), fg


def t1():
    P.uygula(1.0)
    m = BboxLoss(16)
    g = sentetik()
    imgsz = torch.tensor([640., 640.]); stride = torch.ones(1)
    a_iou, a_dfl = P.ORIJINAL_FORWARD(m, *g, imgsz, stride)
    b_iou, b_dfl = BboxLoss.forward(m, *g, imgsz, stride)
    d1, d2 = (a_iou - b_iou).abs().item(), (a_dfl - b_dfl).abs().item()
    print(f"T1 ratio=1.0  loss_iou orijinal={a_iou.item():.8f} yamali={b_iou.item():.8f} |fark|={d1:.2e}; dfl |fark|={d2:.2e}")
    assert d1 < 1e-6 and d2 < 1e-6, "ratio=1.0'da Inner-CIoU != CIoU"
    # gradyan esitligi de (kutu ustunde)
    g2 = sentetik()
    for t in (g2[1],):
        pass
    P.geri_al(); a, _ = BboxLoss.forward(m, *g2, imgsz, stride); ga = torch.autograd.grad(a, g2[1])[0]
    P.uygula(1.0); b, _ = BboxLoss.forward(m, *g2, imgsz, stride); gb = torch.autograd.grad(b, g2[1])[0]
    dg = (ga - gb).abs().max().item()
    print(f"T1 gradyan max|fark| (ratio=1.0) = {dg:.2e}")
    assert dg < 1e-5
    return d1, d2, dg


def t2():
    P.uygula(0.75)
    m = BboxLoss(16)
    imgsz = torch.tensor([640., 640.]); stride = torch.ones(1)
    sonuc = {}
    for ad, yakin in (("yuksek-IoU (jit 0.05)", True), ("orta-IoU (jit 0.6)", False)):
        g = sentetik(yakin=yakin)
        pb = g[1]
        P.geri_al(); a, _ = BboxLoss.forward(m, *g, imgsz, stride); ga = torch.autograd.grad(a, pb)[0]
        P.uygula(0.75); b, _ = BboxLoss.forward(m, *g, imgsz, stride); gb = torch.autograd.grad(b, pb)[0]
        iou = bbox_iou(pb[g[6]], g[3][g[6]], xywh=False).mean().item()
        na, nb = ga.norm().item(), gb.norm().item()
        print(f"T2 {ad}: ort IoU={iou:.3f} | loss CIoU={a.item():.5f} Inner-CIoU={b.item():.5f} | "
              f"||grad|| CIoU={na:.5f} Inner-CIoU={nb:.5f} oran={nb/na:.3f} | sonlu={torch.isfinite(gb).all().item()}")
        assert torch.isfinite(b) and torch.isfinite(gb).all()
        assert b.item() > a.item(), "Inner-CIoU kaybi CIoU'dan buyuk olmali (IoU - IoU_inner >= 0 ortalamada)"
        sonuc[ad] = nb / na
    assert sonuc["yuksek-IoU (jit 0.05)"] > 1.0, "yuksek-IoU'da gradyan CIoU'dan DIK olmali"
    return sonuc


def t4():
    g = sentetik()
    pb, tb = g[1][g[6]], g[3][g[6]]
    d = (P.inner_iou(pb, tb, 1.0) - bbox_iou(pb, tb, xywh=False)).abs().max().item()
    print(f"T4 inner_iou(ratio=1) vs iou: max|fark|={d:.2e}")
    assert d < 1e-6
    return d


def t3():
    """Gercek v8DetectionLoss yolu: model.loss(batch, preds) -> yamali forward ana surecte."""
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG
    from ultralytics.utils import IterableSimpleNamespace
    ws = _AD.parents[1]
    model = YOLO(str(ws / "rakip" / "weights" / "yolo11n.pt")).model
    model.args = IterableSimpleNamespace(**{**vars(DEFAULT_CFG)})
    model.train()
    for p_ in model.parameters():
        p_.requires_grad_(True)
    P.uygula(0.75)
    P.SAYAC.update(cagri=0, fark_toplam=0.0, fark_olcum_n=0)
    imgs = torch.rand(2, 3, 320, 320)
    batch = {
        "img": imgs,
        "batch_idx": torch.tensor([0., 0., 1.]),
        "cls": torch.tensor([[14.], [6.], [3.]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.5], [0.25, 0.3, 0.1, 0.12], [0.7, 0.6, 0.3, 0.3]]),
    }
    preds = model(imgs)
    loss, items = model.loss(batch, preds)
    loss.sum().backward()
    gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters() if p.grad is not None)).item()
    sayac = P.SAYAC["cagri"]
    print(f"T3 v8DetectionLoss: loss={loss.sum().item():.4f} items={ {k: round(float(v), 4) for k, v in items.items()} } "
          f"||grad||={gn:.4f} SAYAC(cagri)={sayac} (ana surec pid uzerinde)")
    assert sayac >= 1, "yamali forward cagrilmadi"
    assert torch.isfinite(loss).all() and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    return sayac


if __name__ == "__main__":
    print("cihaz: cuda_available =", torch.cuda.is_available())
    r1 = t1(); r2 = t2(); r4 = t4(); r3 = t3()
    print("TUM H02 CPU TESTLERI GECTI")
