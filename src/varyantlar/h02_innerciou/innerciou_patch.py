# -*- coding: utf-8 -*-
"""
H02 yamasi: ultralytics.utils.loss.BboxLoss.forward -> Inner-CIoU (Zhang ve ark. 2023, arXiv:2311.02877).

  loss_iou = sum( w * [(1 - CIoU) + IoU - IoU_inner(ratio)] ) / target_scores_sum

IoU_inner: hem tahmin hem GT kutusu icin MERKEZ SABIT, w ve h'i `ratio` ile kucultulmus (ratio<1) yardimci
kutular arasindaki IoU. ratio=1.0'da IoU_inner == IoU => kayip baseline CIoU kaybina esit.
DFL dali ve agirlik (w) orijinal forward (pinli ultralytics 8.4.165, loss.py:119-156) ile satir satir aynidir.

Yama SINIF duzeyinde (BboxLoss.forward) uygulanir; v8DetectionLoss ornekleri `self.bbox_loss(...)` ile sinif
ozniteligini cozdugunden ve kayip ANA surecte (egitim dongusu) hesaplandigindan dataloader isci
sureclerinde yama gerekmez. SAYAC ana surecte artar (kanit).
"""
import torch
import torch.nn.functional as F

from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist

ORIJINAL_FORWARD = BboxLoss.forward  # yama oncesi (testlerde karsilastirma icin)
BASLAR = set()  # eğitimde çağrılan BboxLoss örnekleri (YOLO26 E2ELoss: iki başlık → 2)
RATIO = 0.75
SAYAC = {"cagri": 0, "cagri_val": 0, "ornek_n": 0, "fark_toplam": 0.0, "fark_olcum_n": 0}
_ORNEKLEME_ARALIGI = 200  # her 200. cagrida (IoU - IoU_inner) ortalamasi (.item() senkronu seyrek)


def inner_iou(b1, b2, ratio, eps=1e-7):
    """xyxy kutular (..., 4); merkez sabit, w/h x ratio kucultulmus kutular arasi IoU."""
    def kucult(b):
        x1, y1, x2, y2 = b.chunk(4, -1)
        # merkez sabit, w/h x ratio: x1' = x1 + w*(1-r)/2, x2' = x2 - w*(1-r)/2 (ratio=1'de tam ozdeslik)
        dx, dy = (x2 - x1) * (1.0 - ratio) / 2, (y2 - y1) * (1.0 - ratio) / 2
        return torch.cat((x1 + dx, y1 + dy, x2 - dx, y2 - dy), -1)
    return bbox_iou(kucult(b1), kucult(b2), xywh=False, eps=eps)


def inner_ciou_loss_iou(pb, tb, weight, target_scores_sum, ratio):
    """Yama forward'inin loss_iou kismi (testlerin dogrudan cagirabilmesi icin ayri)."""
    ciou = bbox_iou(pb, tb, xywh=False, CIoU=True)
    iou = bbox_iou(pb, tb, xywh=False)
    iou_in = inner_iou(pb, tb, ratio)
    return (((1.0 - ciou) + iou - iou_in) * weight).sum() / target_scores_sum, iou, iou_in


def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum,
            fg_mask, imgsz, stride):
    """Orijinal BboxLoss.forward ile ayni imza/cikti; yalniz loss_iou farkli (RATIO)."""
    fg_mask = fg_mask.nonzero(as_tuple=True)  # index once; long-index gathers and their backward do not sync
    weight = target_scores[fg_mask].sum(-1, keepdim=True)
    pb, tb = pred_bboxes[fg_mask], target_bboxes[fg_mask]
    loss_iou, iou, iou_in = inner_ciou_loss_iou(pb, tb, weight, target_scores_sum, RATIO)

    if not torch.is_grad_enabled():  # dogrulama kaybi (egitim-ici val): ayri say
        SAYAC["cagri_val"] += 1
    else:
        SAYAC["cagri"] += 1  # yalniz EGITIM (grad acik) cagrilari
        BASLAR.add(id(self))
    if torch.is_grad_enabled() and SAYAC["cagri"] % _ORNEKLEME_ARALIGI == 1:
        with torch.no_grad():
            if iou.numel():
                SAYAC["fark_toplam"] += float((iou - iou_in).mean().item())
                SAYAC["fark_olcum_n"] += 1
                SAYAC["ornek_n"] = int(iou.numel())

    # DFL loss (orijinalle ayni)
    if self.dfl_loss:
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
        target_ltrb = bbox2dist(anchor_points, target_bboxes)
        target_ltrb = target_ltrb * stride
        target_ltrb[..., 0::2] /= imgsz[1]
        target_ltrb[..., 1::2] /= imgsz[0]
        pred_dist = pred_dist * stride
        pred_dist[..., 0::2] /= imgsz[1]
        pred_dist[..., 1::2] /= imgsz[0]
        loss_dfl = (
            F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
        )
        loss_dfl = loss_dfl.sum() / target_scores_sum

    return loss_iou, loss_dfl


def uygula(ratio=None):
    global RATIO
    if ratio is not None:
        RATIO = float(ratio)
    BboxLoss.forward = forward
    return RATIO


def geri_al():
    BboxLoss.forward = ORIJINAL_FORWARD
