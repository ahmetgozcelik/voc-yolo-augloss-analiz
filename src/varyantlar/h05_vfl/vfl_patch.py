# -*- coding: utf-8 -*-
"""
H05 yamasi: v8DetectionLoss cls terimi BCE -> Varifocal agirlikli BCE (Zhang ve ark. 2021, VarifocalNet).

  y = 1[q > 0],  q = TAL hizalama/IoU hedefi (`target_scores`, v8DetectionLoss icinde zaten IoU-agirlikli)
  w = alpha * sigmoid(p)^gamma * (1 - y) + q * y          (agirlik DETACH EDILMEZ: mmdet/ultralytics VarifocalLoss ile ayni)
  out = BCEWithLogits(p, q, reduction='none') * w          (eleman-bazli; sekil = (bs, anchors, nc))

v8DetectionLoss (pinli ultralytics 8.4.165, loss.py:451-454) bu ciktiyi ayni yolda `.sum() / target_scores_sum`
ile normalize eder ve `cls` kazanci (0.5) ile carpar: NORMALIZASYON VE KAZANC BASELINE ILE AYNI KALIR.
Yerlesik ultralytics.utils.loss.VarifocalLoss `.mean(1).sum()` yaptigindan KULLANILMAZ (normalizasyon kayardi).

Yama: v8DetectionLoss.__init__ sarilir; orijinal __init__ sonrasi `self.bce` (nn.BCEWithLogitsLoss(reduction='none'))
yerine ayni imzali `VFLElementwise` modulu konur. Kayip ANA surecte hesaplanir (DataLoader iscileri yalniz veri
uretir) -> Windows spawn'da yama iscilerde gerekmez. SAYAC ana surecte artar (kanit).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss

ORIJINAL_INIT = v8DetectionLoss.__init__  # yama oncesi (testlerde karsilastirma icin)
ALPHA = 0.75
GAMMA = 2.0
POZ = "q"  # pozitif ağırlığı: "q" (VFL, varsayılan) | "1" (H05-neg: pozitifler BCE gibi ağırlıksız)
BASLAR = set()  # eğitimde çağrılan VFLElementwise örnekleri (YOLO26 E2ELoss: iki başlık → 2)
SAYAC = {"cagri": 0, "cagri_val": 0, "grad_olcum_n": 0, "grad_vfl_toplam": 0.0, "grad_bce_toplam": 0.0,
         "poz_n": 0, "olcum_poz_n": 0, "kayip_vfl_toplam": 0.0, "kayip_bce_toplam": 0.0, "kayip_olcum_n": 0}
_OLCUM_ARALIGI = 200  # her 200. egitim cagrisinda gradyan normu orani (.item() senkronu seyrek)


def vfl_elementwise(p, t, alpha=ALPHA, gamma=GAMMA, poz="q"):
    """Eleman-bazli Varifocal agirlikli BCE. p: logit, t: hedef (q, [0,1]); ayni sekilde cikti. Dtype korunur.
    poz="1": pozitif ağırlığı q yerine 1 (H05-neg; negatif terim VFL ile aynı)."""
    y = (t > 0).to(p.dtype)
    w = alpha * torch.sigmoid(p).pow(gamma) * (1.0 - y) + (y if poz == "1" else t * y)
    return F.binary_cross_entropy_with_logits(p, t, reduction="none") * w


class VFLElementwise(nn.Module):
    """nn.BCEWithLogitsLoss(reduction='none') ile ayni cagri imzasi: forward(pred_scores, target_scores)."""

    def __init__(self, alpha=None, gamma=None, poz=None):
        super().__init__()
        self.alpha = ALPHA if alpha is None else float(alpha)
        self.gamma = GAMMA if gamma is None else float(gamma)
        self.poz = POZ if poz is None else str(poz)

    def forward(self, pred, target):
        p, t = pred.float(), target.float()  # fp16/AMP'ta sigmoid^gamma ve BCE kararli olsun (yerlesik VFL de float kullanir)
        out = vfl_elementwise(p, t, self.alpha, self.gamma, self.poz)
        if not torch.is_grad_enabled():
            SAYAC["cagri_val"] += 1
            return out
        SAYAC["cagri"] += 1
        BASLAR.add(id(self))
        if SAYAC["cagri"] % _OLCUM_ARALIGI == 1:  # seyrek gozlem: VFL / BCE gradyan normu orani + kayip orani
            with torch.no_grad():
                pd = p.detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    g_v, = torch.autograd.grad(vfl_elementwise(pd, t, self.alpha, self.gamma, self.poz).sum(), pd)
                    g_b, = torch.autograd.grad(F.binary_cross_entropy_with_logits(pd, t, reduction="none").sum(), pd)
                SAYAC["grad_vfl_toplam"] += float(g_v.norm().item())
                SAYAC["grad_bce_toplam"] += float(g_b.norm().item())
                SAYAC["kayip_vfl_toplam"] += float(out.sum().item())
                SAYAC["kayip_bce_toplam"] += float(F.binary_cross_entropy_with_logits(p, t, reduction="none").sum().item())
                SAYAC["olcum_poz_n"] += int((t > 0).sum().item())
                SAYAC["grad_olcum_n"] += 1
        return out


def uygula(alpha=None, gamma=None, poz=None):
    """v8DetectionLoss.__init__ sarilir; self.bce -> VFLElementwise. Ayni oran/gamma tekrar cagrida guncellenir."""
    global ALPHA, GAMMA, POZ
    if poz is not None:
        if str(poz) not in ("q", "1"):
            raise ValueError(f"poz 'q' ya da '1' olmalı: {poz}")
        POZ = str(poz)
    if alpha is not None:
        ALPHA = float(alpha)
    if gamma is not None:
        GAMMA = float(gamma)

    def yeni_init(self, *a, **k):
        ORIJINAL_INIT(self, *a, **k)
        self.bce = VFLElementwise(ALPHA, GAMMA, POZ)

    v8DetectionLoss.__init__ = yeni_init
    return ALPHA, GAMMA


def geri_al():
    v8DetectionLoss.__init__ = ORIJINAL_INIT
