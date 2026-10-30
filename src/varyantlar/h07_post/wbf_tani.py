# -*- coding: utf-8 -*-
"""
WBF TANISI (beklenti-ihlali.md §4: (a) mekanizma canli mi, (b) etki suruyor mu, (c) neden) — YALNIZ VAL, egitim yok.
Hakem B4: H07'de W1 (WBF, T1 dallari) mAP50-95 -0.28 (T1: +0.80), W2 -5.74 (T2: -4.12). Bu betik "WBF'yi kiran sey fikir mi, kod mu?"
sorusunu olcer. Baseline seed0 best.pt (runs/20260930-072103_baseline_s0), data/voc_dev.yaml (val listesi); kilitli veri bolumune
referans YOKTUR.

Kod incelemesi (degerlendir.py `wbf_goruntuler`): WBF, TTA'nin HER DALINA ayri NMS (conf .001, iou .7, multi_label, max_det 300)
uygulandiktan SONRA (NMS'li ciktilara), girdi-tensoru koordinatinda [0,1]'e (W,H,W,H) normalize edilerek, iou_thr .6, skip .001,
conf_type avg, weights=None ile calisir; sonuc skora gore siralanip 300'e kesilir. T1 ise dallarin HAM ciktilarini birlestirip TEK NMS
(.7) uygular -> W1 ile T1 iki bilesenle farklidir (fuzyon + dal-basina NMS).

Modlar:
  --mod birim : CPU, <=20 goruntu: (1) birim testleri (tek gorunum ~ girdi; iki/uc ozdes gorunum; normalize/denormalize fark testi;
                H07 uygulamasiyla esdegerlik; dal istatistikleri). CUDA_VISIBLE_DEVICES=-1 ile kosulur. Sonuc: sonuclar/wbf-tani-birim.json
  --mod tam   : (varsayilan) birim testleri + tam val tek gecis: R0, T1, W1(H07 varsayilani), izgara (iou_thr x skip x conf_type),
                dal alt-kumeleri/agirlik, T2/W2; cikti sonuclar/wbf-tani.md + .csv ve {RUN_DIR}/metrics.json
Kullanim (kuyruk): python wbf_tani.py --run-dir {RUN_DIR} --out {RUN_DIR}/metrics.json
"""
import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
warnings.filterwarnings("ignore")

_AD = Path(__file__).resolve().parent
WS = _AD.parents[1]
sys.path.insert(0, str(_AD))

CONF, IOU0, MAXDET = 0.001, 0.7, 300
STD_S = [1, 0.83, 0.67]
UP_S = [1, 1.2, 1.4]
W0 = WS / "runs" / "20260930-072103_baseline_s0" / "train" / "weights" / "best.pt"
BASE_VAL = 0.62893   # baseline s0 metrics.json val_map50_95
H07_REF = {"T1": 0.636898, "W1": 0.626106, "T2": 0.587765, "W2": 0.571495}   # sonuclar/h07_post.csv (bu betik bunlari yeniden uretir)


# --------------------------------------------------------------------------------------------------------------
# WBF cekirdegi (isci surecinde de calisir: yalniz numpy + ensemble_boxes)
# --------------------------------------------------------------------------------------------------------------
def wbf_fuse(views, hw, iou_thr, skip, conf_type, weights=None, maxdet=MAXDET, olcek_hw=None):
    """views: [(n,6) np [xyxy(girdi-tensoru px), conf, cls], ...] (dal basina NMS'li). hw: (H,W). Donus: ((m,6) np, n_once_kesme).
    H07 `wbf_goruntuler` ile AYNI adimlar: clip(xyxy/(W,H,W,H),0,1) -> weighted_boxes_fusion -> *(W,H,W,H) -> skora gore sirala -> [:maxdet].
    olcek_hw: yalniz negatif kontrol icin (yanlis boyut)."""
    from ensemble_boxes import weighted_boxes_fusion
    H, W = hw if olcek_hw is None else olcek_hw
    olcek = np.array([W, H, W, H], dtype=np.float64)
    bl, sl, ll, wl = [], [], [], []
    for k, v in enumerate(views):
        if v.shape[0] == 0:
            continue
        bl.append(np.clip(v[:, :4].astype(np.float64) / olcek, 0.0, 1.0))
        sl.append(v[:, 4].astype(np.float64))
        ll.append(v[:, 5].astype(int))
        wl.append(1.0 if weights is None else float(weights[k]))
    if not bl:
        return np.zeros((0, 6)), 0
    b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=wl, iou_thr=iou_thr, skip_box_thr=skip, conf_type=conf_type)
    o = np.concatenate([b * olcek, s[:, None], l[:, None].astype(np.float64)], 1)
    o = o[np.argsort(-o[:, 4], kind="stable")]
    return o[:maxdet], int(o.shape[0])


# --------------------------------------------------------------------------------------------------------------
# Konfigurasyonlar
#   kind: 'nms_r0' (tek olcek NMS) | 'nms_cat' (secili dallarin ham birlesimi + tek NMS) | 'nms_branch' (tek dal NMS) | 'wbf'
#   tier: zaman butcesi gerekirse en yuksek tier'dan baslayarak atilir
# --------------------------------------------------------------------------------------------------------------
def konfigurasyonlar():
    C = []

    def ekle(ad, grup, kind, tier, **k):
        d = dict(ad=ad, grup=grup, kind=kind, tier=tier, dallar=[0, 1, 2], seri="T1", iou_thr=None, skip=None, conf_type=None, w=None)
        d.update(k)
        C.append(d)

    ekle("R0", "ref", "nms_r0", 0, dallar=[])
    ekle("T1", "ref", "nms_cat", 0)
    # kirpma kontrolu: WBF kutulari girdi-tensoru sinirina [0,1] kirpar, NMS ciktisi kirpilmaz -> saf kirpmanin etkisi
    ekle("R0_clip", "clip-kontrol", "clip_of", 0, dallar=[], kaynak="R0")
    ekle("T1_clip", "clip-kontrol", "clip_of", 0, kaynak="T1")
    ekle("T2", "ref", "nms_cat", 0, seri="T2")
    ekle("W1", "varsayilan(H07)", "wbf", 0, iou_thr=0.6, skip=0.001, conf_type="avg")
    ekle("W2", "varsayilan(H07)", "wbf", 0, seri="T2", iou_thr=0.6, skip=0.001, conf_type="avg")
    # tek-dal / ozdes-dal (mekanizma)
    ekle("B0_nms", "tek-dal", "nms_branch", 0, dallar=[0])
    ekle("B0_wbf", "tek-dal", "wbf", 0, dallar=[0], iou_thr=0.6, skip=0.001, conf_type="avg")
    ekle("B0_nms_clip", "tek-dal", "clip_of", 0, dallar=[0], kaynak="B0_nms")
    ekle("B0_wbf_i1", "tek-dal", "wbf", 0, dallar=[0], iou_thr=0.99999, skip=0.001, conf_type="avg")
    ekle("B0x3_wbf", "ozdes-dal", "wbf", 0, dallar=[0, 0, 0], iou_thr=0.6, skip=0.001, conf_type="avg")
    # dal alt-kumeleri: fuzyon vs ham-birlesim+NMS (ayni gorunumler)
    for ad, d in (("b01", [0, 1]), ("b02", [0, 2]), ("b12", [1, 2])):
        ekle(f"T1_{ad}_nms", "alt-kume", "nms_cat", 1, dallar=d)
        ekle(f"W1_{ad}", "alt-kume", "wbf", 1, dallar=d, iou_thr=0.6, skip=0.001, conf_type="avg")
    ekle("W1_w211", "agirlik", "wbf", 1, w=[2, 1, 1], iou_thr=0.6, skip=0.001, conf_type="avg")
    ekle("W1_w411", "agirlik", "wbf", 2, w=[4, 1, 1], iou_thr=0.6, skip=0.001, conf_type="avg")
    # izgara (T1 dallari): iou_thr x skip x conf_type
    for sk, it, ct in itertools.product((0.001, 0.01), (0.5, 0.55, 0.6, 0.7), ("avg", "max", "box_and_model_avg")):
        ad = f"W1_i{it}_s{sk}_{ct}"
        if (it, sk, ct) == (0.6, 0.001, "avg"):
            continue   # = W1
        ekle(ad, "izgara", "wbf", 1 if sk == 0.001 else 2, iou_thr=it, skip=sk, conf_type=ct)
    ekle("W1_i0.6_s0.001_absent", "izgara", "wbf", 3, iou_thr=0.6, skip=0.001, conf_type="absent_model_aware_avg")
    ekle("W2_i0.6_s0.001_max", "izgara", "wbf", 2, seri="T2", iou_thr=0.6, skip=0.001, conf_type="max")
    return C


def _gorev(args):
    """Isci: bir goruntu icin secili WBF konfigurasyonlarini hesaplar. args=(views_by_seri, hw_by_seri, [cfg...])."""
    views, hw, cfgs = args
    sonuc = {}
    for c in cfgs:
        vs = [views[c["seri"]][i] for i in c["dallar"]]
        o, npre = wbf_fuse(vs, hw[c["seri"]], c["iou_thr"], c["skip"], c["conf_type"], weights=c["w"])
        sonuc[c["ad"]] = (o.astype(np.float32), npre)
    return sonuc


# --------------------------------------------------------------------------------------------------------------
# Birim testleri (gercek model dallari, <=N goruntu)
# --------------------------------------------------------------------------------------------------------------
def val_goruntuleri(n):
    yml = WS / "data" / "voc_dev.yaml"
    import yaml
    y = yaml.safe_load(yml.read_text(encoding="utf-8"))
    lst = [ln.strip() for ln in (Path(y["path"]) / y["val"]).read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lst[:n]


def _farkli_kutu(a, b):
    """a,b (n,6): her a kutusu icin ayni sinifli en yakin b kutusuna L1 uzakligi (px) ve skor farki; max'lar."""
    if a.shape[0] == 0 and b.shape[0] == 0:
        return 0.0, 0.0
    if a.shape[0] != b.shape[0]:
        return 1e9, 1e9
    mk, ms = 0.0, 0.0
    for r in a:
        m = b[:, 5] == r[5]
        if not m.any():
            return 1e9, 1e9
        d = np.abs(b[m][:, :4] - r[:4]).max(1)
        j = int(d.argmin())
        mk = max(mk, float(d[j]))
        ms = max(ms, float(abs(b[m][j, 4] - r[4])))
    return mk, ms


def _cift_var(v, esik):
    """Ayni sinifta IoU>esik ciftleri var mi (tek-gorunum WBF'i birlestirir)?"""
    from degerlendir import _iou_matris
    for c in np.unique(v[:, 5]):
        b = v[v[:, 5] == c][:, :4]
        if len(b) > 1:
            m = _iou_matris(b)
            np.fill_diagonal(m, 0)
            if (m > esik).any():
                return True
    return False


def birim_testleri(n_img, cihaz):
    import torch
    import cv2
    from ultralytics.data.augment import LetterBox
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import nms
    import degerlendir as D

    D.tta_yamasi_uygula()
    ab = AutoBackend(model=str(W0), device=torch.device(cihaz), fp16=False)
    ab.eval()
    lb = LetterBox((640, 640), auto=True, stride=int(ab.stride))
    R = {"n_img": 0, "cihaz": str(cihaz)}
    br_say, br_300, hw_l, hw_uyum, tasma = [], [], [], True, []
    t1 = dict(n_img=0, n_temiz=0, n_birlesen=0, max_fark_px_temiz=0.0, max_dskor_temiz=0.0, n_kutu_girdi=0, n_kutu_cikti=0)
    t1b = dict(max_fark_px=0.0, max_dskor=0.0, say_esit=True, max_fark_ham_px=0.0)  # iou_thr ~1 (birlestirme yok)
    t1s = dict(max_dskor_temiz=0.0, kutu_esit=True)                        # skip=0.01 filtresi
    t2 = dict(max_fark_iki=0.0, max_dskor_iki=0.0, max_fark_uc=0.0, max_dskor_uc=0.0, say_esit=True)
    t4 = dict(max_fark_H07=0.0, max_dskor_H07=0.0, say_esit=True)
    with torch.inference_mode():
        for f in val_goruntuleri(n_img):
            im0 = cv2.imread(f)
            im = lb(image=im0)
            x = torch.from_numpy(np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))[None]).to(cihaz).float() / 255
            D.TTA["s"] = list(STD_S)
            DetectionModel._predict_augment(ab.model, x.clone())
            st = D.TTA["stash"]
            hw = st["hw"]
            hw_uyum &= tuple(hw) == tuple(x.shape[-2:])
            hw_l.append(tuple(hw))
            dal = [nms.non_max_suppression(yb, CONF, IOU0, nc=0, multi_label=True, max_det=MAXDET)[0].float().cpu().numpy() for yb in st["y"]]
            for d in dal:
                br_say.append(d.shape[0])
                br_300.append(d.shape[0] >= MAXDET)
                if d.shape[0]:
                    H, W = hw
                    tasma.append(float(((d[:, 0] < 0) | (d[:, 1] < 0) | (d[:, 2] > W) | (d[:, 3] > H)).mean()))
            v0 = dal[0]
            H_, W_ = hw
            v0c = v0.copy(); v0c[:, [0, 2]] = v0c[:, [0, 2]].clip(0, W_); v0c[:, [1, 3]] = v0c[:, [1, 3]].clip(0, H_)
            R["n_img"] += 1
            # ---- U1: tek gorunum ----
            o1, _ = wbf_fuse([v0], hw, 0.6, CONF, "avg")
            t1["n_img"] += 1
            t1["n_kutu_girdi"] += int(v0.shape[0]); t1["n_kutu_cikti"] += int(o1.shape[0])
            if _cift_var(v0, 0.6):
                t1["n_birlesen"] += 1
            else:
                t1["n_temiz"] += 1
                mk, ms = _farkli_kutu(o1, v0c)
                mk_ham, _ = _farkli_kutu(o1, v0)
                t1["max_fark_ham_px_temiz"] = max(t1.get("max_fark_ham_px_temiz", 0.0), mk_ham)
                t1["max_fark_px_temiz"] = max(t1["max_fark_px_temiz"], mk)
                t1["max_dskor_temiz"] = max(t1["max_dskor_temiz"], ms)
                t1["say_esit_temiz"] = t1.get("say_esit_temiz", True) and (o1.shape[0] == v0.shape[0])
            ob, _ = wbf_fuse([v0], hw, 0.99999, CONF, "avg")
            mk, ms = _farkli_kutu(ob, v0c)
            mk_ham, _ = _farkli_kutu(ob, v0)
            t1b["max_fark_ham_px"] = max(t1b.get("max_fark_ham_px", 0.0), mk_ham)
            t1b["max_fark_px"] = max(t1b["max_fark_px"], mk)
            t1b["max_dskor"] = max(t1b["max_dskor"], ms)
            t1b["say_esit"] &= (ob.shape[0] == v0.shape[0])
            osk, _ = wbf_fuse([v0], hw, 0.99999, 0.01, "avg")
            beklenen = v0c[v0c[:, 4] >= 0.01]
            t1s["kutu_esit"] &= (osk.shape[0] == beklenen.shape[0])
            _, ms = _farkli_kutu(osk, beklenen)
            t1s["max_dskor_temiz"] = max(t1s["max_dskor_temiz"], ms)
            # ---- U2: iki / uc ozdes gorunum (tek-gorunum sonucuyla birebir esit olmali) ----
            for ad, k in (("iki", 2), ("uc", 3)):
                ok, _ = wbf_fuse([v0] * k, hw, 0.6, CONF, "avg")
                mk, ms = _farkli_kutu(ok, o1)
                t2[f"max_fark_{ad}"] = max(t2[f"max_fark_{ad}"], mk)
                t2[f"max_dskor_{ad}"] = max(t2[f"max_dskor_{ad}"], ms)
                t2["say_esit"] &= (ok.shape[0] == o1.shape[0])
            # ---- U4: H07 uygulamasiyla esdegerlik (uc dal, varsayilan) ----
            oh = D.wbf_goruntuler(st, nms)[0].float().cpu().numpy()
            om, _ = wbf_fuse(dal, hw, 0.6, CONF, "avg")
            mk, ms = _farkli_kutu(om, oh)
            t4["max_fark_H07"] = max(t4["max_fark_H07"], mk)
            t4["max_dskor_H07"] = max(t4["max_dskor_H07"], ms)
            t4["say_esit"] &= (om.shape[0] == oh.shape[0])
    # ---- U3: normalize/denormalize fark testi (dikdortgen tensorlar; yanlis-boyut negatif kontrolu) ----
    rng = np.random.default_rng(0)
    u3 = {}
    for (H, W) in ((480, 640), (640, 480), (384, 640), (640, 640), (192, 256)):
        n = 40
        x1 = rng.uniform(0, W * 0.7, n); y1 = rng.uniform(0, H * 0.7, n)
        b = np.stack([x1, y1, x1 + rng.uniform(8, W * 0.3, n), y1 + rng.uniform(8, H * 0.3, n)], 1)
        b[:, [2]] = np.minimum(b[:, [2]], W); b[:, [3]] = np.minimum(b[:, [3]], H)
        # uzak/birbirini orten olmayan kutular: sinif = indeks (her kutu ayri sinif -> birlestirme yok)
        v = np.concatenate([b, rng.uniform(0.1, 0.9, (n, 1)), np.arange(n)[:, None] % 20], 1)
        v[:, 5] = np.arange(n)   # her kutu kendi sinifi
        o, _ = wbf_fuse([v], (H, W), 0.6, CONF, "avg")
        mk, _ = _farkli_kutu(o, v)
        o_yanlis, _ = wbf_fuse([v], (H, W), 0.6, CONF, "avg", olcek_hw=(W, H))   # (H,W) yer degisti: negatif kontrol
        mk_y, _ = _farkli_kutu(o_yanlis, v)
        u3[f"{H}x{W}"] = dict(max_fark_px=mk, yanlis_boyut_max_fark_px=mk_y)
    # gercek veride tasma (tensor disi kutu) payi: clip'in koordinati degistirdigi kutu orani
    R.update(dict(
        U1_tek_gorunum=t1, U1_iou_thr_1=t1b, U1_skip_0_01=t1s, U2_ozdes_gorunumler=t2, U3_normalize_denormalize=u3, U4_H07_esdegerlik=t4,
        hw_uyumlu=bool(hw_uyum), hw_ornekler=sorted(set(hw_l))[:6],
        dal_ort_kutu=float(np.mean(br_say)) if br_say else 0.0, dal_300_kesilen_pay=float(np.mean(br_300)) if br_300 else 0.0,
        dal_tensor_disi_kutu_pay=float(np.mean(tasma)) if tasma else 0.0,
        wbf_seviyesi="dal-basina NMS (conf .001, iou .7, multi_label, max_det 300) SONRASI; T1 ise ham dal birlestirme + tek NMS",
    ))
    # gecti/kaldi
    TOL_PX, TOL_S = 1e-2, 1e-4
    gec = {
        "U1_temiz_kutular_ayni": (t1["max_fark_px_temiz"] <= TOL_PX and t1["max_dskor_temiz"] <= TOL_S and t1.get("say_esit_temiz", True)),
        "U1_iou1_ayni": (t1b["max_fark_px"] <= TOL_PX and t1b["max_dskor"] <= TOL_S and t1b["say_esit"]),
        "U1_skip_filtre": (t1s["kutu_esit"] and t1s["max_dskor_temiz"] <= TOL_S),
        "U2_ozdes_gorunumler_tek_ile_ayni": (max(t2["max_fark_iki"], t2["max_fark_uc"]) <= TOL_PX
                                              and max(t2["max_dskor_iki"], t2["max_dskor_uc"]) <= TOL_S and t2["say_esit"]),
        "U3_roundtrip": all(v["max_fark_px"] <= TOL_PX for v in u3.values()),
        "U3_negatif_kontrol_duyarli": any(v["yanlis_boyut_max_fark_px"] > 1.0 for k, v in u3.items() if k.split("x")[0] != k.split("x")[1]),
        "U4_H07_esdegerlik": (t4["max_fark_H07"] <= TOL_PX and t4["max_dskor_H07"] <= TOL_S and t4["say_esit"]),
        "hw_tensorle_uyumlu": bool(hw_uyum),
    }
    R["gecti"] = gec
    R["hepsi_gecti"] = bool(all(gec.values()))
    return R


# --------------------------------------------------------------------------------------------------------------
# Tam val tek gecis
# --------------------------------------------------------------------------------------------------------------
def tam_calis(a, cfgs, t_basla):
    import torch
    import multiprocessing as mp
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import nms
    from ultralytics.utils.metrics import ap_per_class, box_iou
    import degerlendir as D

    D.tta_yamasi_uygula()
    cuda_ok = torch.cuda.is_available()
    dev_arg = 0 if cuda_ok else "cpu"
    ad2cfg = {c["ad"]: c for c in cfgs}
    wbf_cfgs = [c for c in cfgs if c["kind"] == "wbf"]
    tier_ok = {t: True for t in range(0, 4)}
    gecikme = {"n": 0, "wbf_sn": 0.0, "tier_sn": {t: 0.0 for t in range(0, 4)}}
    havuz = None
    if a.isci > 1:
        havuz = mp.get_context("spawn").Pool(a.isci)

    class Acc:
        def __init__(self):
            self.tp, self.conf, self.pcls, self.tcls = [], [], [], []
            self.n_det, self.n_img, self.n_300, self.n_pre = 0, 0, 0, 0
            self.loc_sum, self.loc_n, self.loc_90 = 0.0, 0, 0
            self.bin_conf = {"S": [0.0, 0], "M": [0.0, 0], "L": [0.0, 0]}
            self.bin_gt = {"S": 0, "M": 0, "L": 0}

    accs = {c["ad"]: Acc() for c in cfgs}

    class TaniVal(DetectionValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            self.t_model = model
            self.t_batch_no = 0

        def preprocess(self, batch):
            b = super().preprocess(batch)
            self.t_img = b["img"]
            return b

        def _nms_cat(self, ys):
            return nms.non_max_suppression(torch.cat(ys, -1), CONF, IOU0, nc=0, multi_label=True, max_det=MAXDET)

        def postprocess(self, preds):
            t0 = time.time()
            img = self.t_img
            bs = img.shape[0]
            H, W = int(img.shape[2]), int(img.shape[3])
            out = {}
            stash = {}
            # R0: tek olcek, tek NMS (baseline ile ayni)
            p0 = self.t_model(img, augment=False)
            r0 = nms.non_max_suppression(p0[0] if isinstance(p0, (list, tuple)) else p0, CONF, IOU0, nc=0, multi_label=True, max_det=MAXDET)
            out["R0"] = r0
            for seri, s in (("T1", STD_S), ("T2", UP_S)):
                D.TTA["s"] = list(s)
                D.TTA["stash"] = None
                self.t_model(img, augment=True)
                st = D.TTA["stash"]
                stash[seri] = (list(st["y"]), tuple(st["hw"]))
            D.TTA["s"] = list(STD_S)
            hw = {k: v[1] for k, v in stash.items()}
            dal = {}   # seri -> [dal][img] np
            for seri, (ys, _) in stash.items():
                dal[seri] = [[x.float().cpu().numpy() for x in nms.non_max_suppression(yb, CONF, IOU0, nc=0, multi_label=True, max_det=MAXDET)] for yb in ys]
            # NMS'li referanslar
            for c in cfgs:
                if c["kind"] == "nms_cat":
                    out[c["ad"]] = self._nms_cat([stash[c["seri"]][0][i] for i in c["dallar"]])
                elif c["kind"] == "nms_branch":
                    out[c["ad"]] = self._nms_cat([stash[c["seri"]][0][i] for i in c["dallar"]])
            for c in cfgs:
                if c["kind"] == "clip_of":
                    kl = []
                    for x in out[c["kaynak"]]:
                        x = x.clone()
                        x[:, [0, 2]] = x[:, [0, 2]].clamp(0, W)
                        x[:, [1, 3]] = x[:, [1, 3]].clamp(0, H)
                        kl.append(x)
                    out[c["ad"]] = kl
            # WBF (isci havuzu: goruntu basina bir gorev)
            aktif = [c for c in wbf_cfgs if tier_ok[c["tier"]]]
            t_w = time.time()
            gorevler = []
            for i in range(bs):
                views = {seri: [dal[seri][k][i] for k in range(3)] for seri in dal}
                gorevler.append((views, hw, aktif))
            sonuc = havuz.map(_gorev, gorevler, chunksize=1) if havuz else [_gorev(g) for g in gorevler]
            gecikme["wbf_sn"] += time.time() - t_w
            gecikme["n"] += 1
            dev = img.device
            for c in aktif:
                out[c["ad"]] = [torch.as_tensor(sonuc[i][c["ad"]][0], device=dev) for i in range(bs)]
                for i in range(bs):
                    accs[c["ad"]].n_pre += sonuc[i][c["ad"]][1]
            self.t_out = out
            return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in r0]

        def update_metrics(self, preds, batch):
            self.t_batch_no += 1
            bs = len(preds)
            for si in range(bs):
                self.seen += 1
                pb = self._prepare_batch(si, batch)
                cls = pb["cls"].cpu().numpy()
                gtb, gtc = pb["bboxes"], pb["cls"]
                for ad, lst in self.t_out.items():
                    x = lst[si].float()
                    A = accs[ad]
                    pr = {"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5]}
                    tp = self._process_batch(pr, pb)["tp"]
                    A.tp.append(tp)
                    A.conf.append(x[:, 4].cpu().numpy())
                    A.pcls.append(x[:, 5].cpu().numpy())
                    A.tcls.append(cls)
                    A.n_img += 1
                    A.n_det += int(x.shape[0])
                    A.n_300 += int(x.shape[0] >= MAXDET)
                    # yerellestirme + dal-bazli olcek-skor tanisi
                    if gtb.shape[0] and x.shape[0]:
                        iou = box_iou(gtb, x[:, :4]) * (gtc[:, None] == x[:, 5][None, :]).float()
                        sel = x[:, 4] >= 0.25
                        if sel.any():
                            bi = iou[:, sel].max(0).values
                            m = bi >= 0.5
                            A.loc_sum += float(bi[m].sum()); A.loc_n += int(m.sum()); A.loc_90 += int((bi >= 0.9).sum())
                        mx, arg = iou.max(1)
                        mt = mx >= 0.5
                        sq = ((gtb[:, 2] - gtb[:, 0]) * (gtb[:, 3] - gtb[:, 1])).clamp(min=0).sqrt()
                        for nm, lo, hi in (("S", 0, 32), ("M", 32, 96), ("L", 96, 1e9)):
                            mm = (sq >= lo) & (sq < hi)
                            A.bin_gt[nm] += int(mm.sum())
                            k = mm & mt
                            if k.any():
                                A.bin_conf[nm][0] += float(x[arg[k], 4].sum()); A.bin_conf[nm][1] += int(k.sum())
            # zaman butcesi: ilk 4 batch'ten sonra tier'lari ata
            if self.t_batch_no == 4:
                self._butce()

        def _butce(self):
            toplam_batch = math.ceil(len(self.dataloader.dataset) / self.dataloader.batch_size) if hasattr(self.dataloader, "dataset") else 63
            gecen = time.time() - t_basla
            kalan_sn = a.zaman_butce_dk * 60 - gecen - 120
            per = gecikme["wbf_sn"] / max(gecikme["n"], 1)
            # her tier'in goreli maliyeti ~ konfigurasyon sayisi (skip=.01 daha ucuz; kaba)
            sayi = {t: sum(1 for c in wbf_cfgs if c["tier"] == t) for t in range(4)}
            tum = sum(sayi.values())
            for t in (3, 2, 1):
                aktif_say = sum(sayi[k] for k in range(4) if tier_ok[k])
                tahmin = per * (aktif_say / max(tum, 1)) * (toplam_batch - self.t_batch_no) + 0.0
                if tahmin > kalan_sn and kalan_sn > 0:
                    tier_ok[t] = False
                    print(f"[wbf-tani] ZAMAN BUTCESI: tier {t} atildi (tahmin {tahmin:.0f} sn > kalan {kalan_sn:.0f} sn)")
            print(f"[wbf-tani] batch {self.t_batch_no}: WBF/batch={per:.2f} sn; aktif tier={[t for t in tier_ok if tier_ok[t]]}")

        def get_stats(self):
            return {}

        def print_results(self):
            pass

        def finalize_metrics(self):
            pass

    val_args = dict(data=str(a.data or (WS / "data" / "voc_dev.yaml")), split="val", imgsz=640, batch=a.batch, device=dev_arg, workers=a.workers,
                    plots=False, save_json=False, project=str(a.run_dir), name="val_wbf_tani", exist_ok=True, verbose=False,
                    model=str(W0), augment=False)
    v = TaniVal(args=val_args)
    t_val = time.time()
    v(model=str(W0))
    if havuz:
        havuz.close(); havuz.join()
    print(f"[wbf-tani] val gecisi {time.time() - t_val:.0f} sn")

    # ---- metrikler ----
    sonuc = {}
    for c in cfgs:
        A = accs[c["ad"]]
        if A.n_img == 0:
            sonuc[c["ad"]] = None
            continue
        tp = np.concatenate(A.tp, 0) if A.tp and sum(len(t) for t in A.tp) else np.zeros((0, 10), bool)
        conf = np.concatenate(A.conf); pcls = np.concatenate(A.pcls); tcls = np.concatenate(A.tcls)
        ap = ap_per_class(tp, conf, pcls, tcls)[5]
        sonuc[c["ad"]] = dict(
            map50_95=float(ap.mean()), map50=float(ap[:, 0].mean()), ap75=float(ap[:, 5].mean()), ap90=float(ap[:, 8].mean()),
            n_img=A.n_img, det_ort=A.n_det / A.n_img, pct_300=100.0 * A.n_300 / A.n_img,
            n_once_kesme_ort=(A.n_pre / A.n_img) if c["kind"] == "wbf" else float("nan"),
            loc_iou=(A.loc_sum / A.loc_n) if A.loc_n else float("nan"), loc_n=A.loc_n, loc_frac90=(A.loc_90 / A.loc_n) if A.loc_n else float("nan"),
            skor_S=(A.bin_conf["S"][0] / A.bin_conf["S"][1]) if A.bin_conf["S"][1] else float("nan"),
            skor_M=(A.bin_conf["M"][0] / A.bin_conf["M"][1]) if A.bin_conf["M"][1] else float("nan"),
            skor_L=(A.bin_conf["L"][0] / A.bin_conf["L"][1]) if A.bin_conf["L"][1] else float("nan"),
            kapsam_S=(A.bin_conf["S"][1] / A.bin_gt["S"]) if A.bin_gt["S"] else float("nan"),
            kapsam_M=(A.bin_conf["M"][1] / A.bin_gt["M"]) if A.bin_gt["M"] else float("nan"),
            kapsam_L=(A.bin_conf["L"][1] / A.bin_gt["L"]) if A.bin_gt["L"] else float("nan"),
        )
    return sonuc, [c["ad"] for c in cfgs if not (c["kind"] == "wbf" and not tier_ok[c["tier"]])], dict(tier_ok), gecikme


# --------------------------------------------------------------------------------------------------------------
def yaz_rapor(cfgs, S, atlanan, birim, sure_sn, dosya_md, dosya_csv, cihaz):
    ad2c = {c["ad"]: c for c in cfgs}
    ok = {k: v for k, v in S.items() if v}
    f = lambda x, n=2: "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{n}f}"
    T1, R0v, W1 = ok.get("T1"), ok.get("R0"), ok.get("W1")
    # ---- CSV ----
    with open(dosya_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ad", "grup", "tur", "seri", "dallar", "agirlik", "iou_thr", "skip", "conf_type", "map50_95", "map50", "ap75", "ap90",
                    "d_vs_R0_puan", "d_vs_T1_puan", "det_ort", "pct_300_kesilen", "once_kesme_ort", "loc_iou_ort", "loc_frac_ge90", "skor_S", "skor_M", "skor_L",
                    "kapsam_S", "kapsam_M", "kapsam_L", "durum"])
        for c in cfgs:
            s = ok.get(c["ad"])
            if not s or c["ad"] in atlanan_set(atlanan):
                w.writerow([c["ad"], c["grup"], c["kind"], c["seri"], "".join(map(str, c["dallar"])), c["w"] or "", c["iou_thr"] or "", c["skip"] or "",
                            c["conf_type"] or "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "ATLANDI(zaman butcesi)"])
                continue
            ref_T = ok["T1" if c["seri"] == "T1" else "T2"]
            w.writerow([c["ad"], c["grup"], c["kind"], c["seri"], "".join(map(str, c["dallar"])), c["w"] or "", c["iou_thr"] or "", c["skip"] or "", c["conf_type"] or "",
                        f"{s['map50_95']:.6f}", f"{s['map50']:.6f}", f"{s['ap75']:.6f}", f"{s['ap90']:.6f}",
                        f"{(s['map50_95'] - R0v['map50_95']) * 100:.4f}", f"{(s['map50_95'] - ref_T['map50_95']) * 100:.4f}",
                        f"{s['det_ort']:.1f}", f"{s['pct_300']:.1f}", "" if math.isnan(s["n_once_kesme_ort"]) else f"{s['n_once_kesme_ort']:.1f}",
                        f"{s['loc_iou']:.5f}", f"{s['loc_frac90']:.5f}", f"{s['skor_S']:.4f}", f"{s['skor_M']:.4f}", f"{s['skor_L']:.4f}",
                        f"{s['kapsam_S']:.4f}", f"{s['kapsam_M']:.4f}", f"{s['kapsam_L']:.4f}", "OK"])
    # ---- karar kurallari (mekanik; yorum karar-tahlilcinindir) ----
    grid = [(c, ok[c["ad"]]) for c in cfgs if c["grup"] == "izgara" and c["seri"] == "T1" and c["ad"] in ok and c["ad"] not in atlanan_set(atlanan)]
    grid.append((ad2c["W1"], W1))
    en_iyi_c, en_iyi_s = max(grid, key=lambda t: t[1]["map50_95"])
    d_w1_t1 = (W1["map50_95"] - T1["map50_95"]) * 100
    d_best_t1 = (en_iyi_s["map50_95"] - T1["map50_95"]) * 100
    d_best_w1 = (en_iyi_s["map50_95"] - W1["map50_95"]) * 100
    avg_mx = None
    if "W1_i0.6_s0.001_max" in ok:
        avg_mx = (ok["W1_i0.6_s0.001_max"]["map50_95"] - W1["map50_95"]) * 100
    TOL = 0.5   # puan (seed gurultusu ~0.4; goruntu-bootstrap GA yarim-genisligi ~0.3-0.5)
    yapi = {}
    # (a) mekanizma canli mi?
    d_b0_i1 = (ok["B0_wbf_i1"]["map50_95"] - ok["B0_nms_clip"]["map50_95"]) * 100
    d_b0x3 = (ok["B0x3_wbf"]["map50_95"] - ok["B0_wbf"]["map50_95"]) * 100
    a_ok = bool(birim.get("hepsi_gecti")) and abs(d_b0_i1) < 0.02 and abs(d_b0x3) < 0.05
    yapi["a"] = a_ok
    # (b) etki suruyor mu: izgaranin en iyisi de T1'in altinda mi
    b_suruyor = d_best_t1 < -0.0
    yapi["b"] = b_suruyor
    # (c) nedenler
    c_list = []
    if avg_mx is not None and avg_mx >= TOL:
        c_list.append(f"conf_type=max, avg'dan {avg_mx:+.2f} puan iyi: skorda gorunum-sayisi cezasi (avg: <3 gorunumde gorulen kutu skoru kucuk) siralamayi bozuyor")
    if d_best_w1 >= TOL:
        c_list.append(f"izgara en iyisi ({en_iyi_c['ad']}) W1'den {d_best_w1:+.2f} puan iyi: WBF hiperparametresi (iou_thr/skip/conf_type) varsayilanda uygun degil")
    t_loc = (W1["ap90"] - T1["ap90"]) * 100, (W1["ap75"] - T1["ap75"]) * 100, (W1["map50"] - T1["map50"]) * 100
    if t_loc[2] > 0 and (t_loc[0] < -0.3 or t_loc[1] < -0.3):
        c_list.append(f"W1: AP50 {t_loc[2]:+.2f}, AP75 {t_loc[1]:+.2f}, AP90 {t_loc[0]:+.2f} (T1'e gore): ince konumlama kaybi, kaba bulma kazanci")
    if "W1_w211" in ok and (ok["W1_w211"]["map50_95"] - W1["map50_95"]) * 100 >= TOL:
        c_list.append("agirlik [2,1,1] W1'den iyi: dal kalitesi esit degil (0.83/0.67 dallari zayif)")
    for kk in ("b01", "b02", "b12"):
        if f"W1_{kk}" in ok and f"T1_{kk}_nms" in ok:
            dd = (ok[f"W1_{kk}"]["map50_95"] - ok[f"T1_{kk}_nms"]["map50_95"]) * 100
            c_list.append(f"alt-kume {kk}: WBF - (ham birlestirme+NMS) = {dd:+.2f} puan")
    yapi["c"] = c_list
    # ---- MD ----
    satirlar = []
    P = satirlar.append
    P("# WBF tanisi (hakem B4; beklenti-ihlali.md §4) — YALNIZ VAL, baseline s0 best.pt")
    P("")
    P(f"Uretildi: betik `varyantlar/h07_post/wbf_tani.py`; cihaz {cihaz}; toplam {sure_sn:.0f} sn; val 1000 goruntu (data/voc_dev.yaml); kilitli veriye referans yok.")
    P("")
    P("## 0. Ozet (mekanik; yorum karar-tahlilcinindir)")
    P(f"- Tek gecis R0 yeniden-uretimi: R0 mAP50-95 = {R0v['map50_95']:.5f} (baseline metrics.json {BASE_VAL}); T1 = {T1['map50_95']:.5f} (H07 CSV {H07_REF['T1']}); "
      f"W1 = {W1['map50_95']:.5f} (H07 CSV {H07_REF['W1']}); T2 = {ok['T2']['map50_95']:.5f} (H07 {H07_REF['T2']}); W2 = {ok['W2']['map50_95']:.5f} (H07 {H07_REF['W2']}).")
    P(f"- (a) Mekanizma canli ve dogru mu: **{'EVET' if a_ok else 'HAYIR/BELIRSIZ'}** (birim testleri hepsi gecti={birim.get('hepsi_gecti')}; "
      f"val duzeyi kimlik testi: tek dal WBF(iou_thr~1) - tek dal NMS(kirpilmis) = {d_b0_i1:+.3f} puan (<0.02 beklenir); uc ozdes dal - tek dal WBF = {d_b0x3:+.3f} puan (<0.05 beklenir)). "
      f"Bilgi: tek dal WBF(.6) - tek dal NMS = {(ok['B0_wbf']['map50_95'] - ok['B0_nms']['map50_95']) * 100:+.3f} puan (kirpma + .6<IoU<=.7 birlestirmesi).")
    P(f"- Kirpma kontrolu (WBF kutulari girdi-tensoru sinirina kirpar, NMS kirpmaz): R0_clip - R0 = {(ok['R0_clip']['map50_95'] - ok['R0']['map50_95']) * 100:+.3f} puan; T1_clip - T1 = {(ok['T1_clip']['map50_95'] - ok['T1']['map50_95']) * 100:+.3f} puan.")
    P(f"- (b) Etki suruyor mu: W1 - T1 = {d_w1_t1:+.2f} puan; izgara en iyisi `{en_iyi_c['ad']}` - T1 = {d_best_t1:+.2f} puan -> **{'SURUYOR (izgarada da T1 altinda/esit)' if b_suruyor else 'KAYBOLDU (izgarada T1 ustu)'}**.")
    P("- (c) Neden (mekanik kanit listesi, esik 0.5 puan):")
    for ln in c_list or ["(esigi asan tek neden bulunamadi)"]:
        P(f"  - {ln}")
    P("")
    P("## 1. Kod incelemesi: WBF hangi seviyede?")
    P(f"- {birim['wbf_seviyesi']}.")
    P("- H07 `wbf_goruntuler`: her dal icin `non_max_suppression(conf=.001, iou=.7, multi_label, max_det=300)`; kutular girdi-tensoru koordinatinda (W,H,W,H) ile [0,1]'e normalize + `clip`; "
      "`weighted_boxes_fusion(weights=None, iou_thr=.6, skip_box_thr=.001, conf_type='avg')`; sonuc skora gore siralanir, `[:300]`.")
    P(f"- Dal basina NMS'li kutu sayisi ortalama {birim['dal_ort_kutu']:.0f}; 300 siniri dolan dal orani {birim['dal_300_kesilen_pay'] * 100:.1f}% (<=20 goruntu); tensor-disi (clip'e takilan) kutu orani {birim['dal_tensor_disi_kutu_pay'] * 100:.1f}%.")
    P("- `_clip_augmented` (Ultralytics): olcek-1 dali en kaba (stride-32, buyuk nesne) kafasini, en kucuk-olcek dali en ince (stride-8, kucuk nesne) kafasini kaybeder; "
      "yani her dal her boyuttaki nesneyi goremez. `conf_type=avg` fuzyonu <3 gorunumde gorulen kutuyu (3 dal icinde ort. * gorunum/3) cezalandirir: **yapisal olarak gormeyen dal bile 'yok' sayilir** "
      "(skor boyut-bagimli olceklenir). T2 icin kafa kirpmasi ters yonde (kart: 'tek fark s listesi')." )
    P("")
    P("## 2. Birim testleri (CPU/GPU, <=20 goruntu, gercek model dallari)")
    for k, v in birim["gecti"].items():
        P(f"- {k}: **{'GECTI' if v else 'KALDI'}**")
    t1 = birim["U1_tek_gorunum"]
    P(f"- U1 tek gorunum (iou_thr .6): {t1['n_img']} goruntu; ayni-sinif IoU>.6 cifti olmayan (temiz) {t1['n_temiz']}, cifti olan (birlesen) {t1['n_birlesen']}; temiz goruntulerde "
      f"girdinin tensor-sinirina KIRPILMIS haline gore max koordinat farki {t1['max_fark_px_temiz']:.2e} px (kirpilmamis girdiye gore {t1.get('max_fark_ham_px_temiz', 0):.2f} px: WBF kutulari [0,1]'e kirpar, NMS kirpmaz), max skor farki {t1['max_dskor_temiz']:.2e}; girdi/cikti kutu sayisi {t1['n_kutu_girdi']}/{t1['n_kutu_cikti']} "
      f"(fark = NMS .7 > WBF .6 uyumsuzlugundan: .6<IoU<=.7 ciftleri tek-gorunumde bile birlesir).")
    P(f"- U1 iou_thr~1 (kirpilmis girdiye gore; kirpilmamis girdiye gore max {birim['U1_iou_thr_1'].get('max_fark_ham_px', 0):.2f} px): max fark {birim['U1_iou_thr_1']['max_fark_px']:.2e} px / {birim['U1_iou_thr_1']['max_dskor']:.2e} skor; sayi esit={birim['U1_iou_thr_1']['say_esit']}; skip=0.01 filtresi dogru={birim['U1_skip_0_01']['kutu_esit']}.")
    t2 = birim["U2_ozdes_gorunumler"]
    P(f"- U2 iki/uc ozdes gorunum (tek-gorunum sonucuyla): max fark iki={t2['max_fark_iki']:.2e}/uc={t2['max_fark_uc']:.2e} px, skor {t2['max_dskor_iki']:.2e}/{t2['max_dskor_uc']:.2e}; sayi esit={t2['say_esit']}.")
    P("- U3 normalize/denormalize (dikdortgen tensorlar; her kutu ayri sinif):")
    for k, v in birim["U3_normalize_denormalize"].items():
        P(f"  - {k}: dogru boyut max fark {v['max_fark_px']:.2e} px; (H,W) yer degisince {v['yanlis_boyut_max_fark_px']:.1f} px (negatif kontrol)")
    u4 = birim["U4_H07_esdegerlik"]
    P(f"- U4 bu betigin `wbf_fuse`'u = H07 `wbf_goruntuler` (uc dal, varsayilan): max fark {u4['max_fark_H07']:.2e} px, skor {u4['max_dskor_H07']:.2e}; sayi esit={u4['say_esit']}; hw==tensor.shape={birim['hw_uyumlu']}.")
    P("")
    P("## 3. Tam-val sonuclari (puan = x100; Delta: R0'a ve kendi serisinin ham-birlestirme referansina (T1 / T2) gore)")
    P("")
    P("| ad | grup | dallar | w | iou | skip | conf | mAP50-95 | mAP50 | AP75 | AP90 | d vs R0 | d vs T | det/gor | %300 | on-kesme | loc IoU(conf>=.25, IoU>=.5) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cfgs:
        s = ok.get(c["ad"])
        if not s or c["ad"] in atlanan_set(atlanan):
            P(f"| {c['ad']} | {c['grup']} | {''.join(map(str, c['dallar']))} | | | | | ATLANDI | | | | | | | | | |")
            continue
        refT = ok["T1" if c["seri"] == "T1" else "T2"]
        P(f"| {c['ad']} | {c['grup']} | {''.join(map(str, c['dallar']))} | {c['w'] or ''} | {c['iou_thr'] or ''} | {c['skip'] or ''} | {c['conf_type'] or ''} | "
          f"{f(s['map50_95'])} | {f(s['map50'])} | {f(s['ap75'])} | {f(s['ap90'])} | {(s['map50_95'] - R0v['map50_95']) * 100:+.2f} | {(s['map50_95'] - refT['map50_95']) * 100:+.2f} | "
          f"{s['det_ort']:.0f} | {s['pct_300']:.0f} | {'' if math.isnan(s['n_once_kesme_ort']) else format(s['n_once_kesme_ort'], '.0f')} | {s['loc_iou']:.4f} |")
    P("")
    P("## 4. Boyut-bazli eslesen-GT skoru (GT'ye IoU>=.5 ile eslesen en iyi tahminin ort. guveni; kapsam = eslesen GT orani) — olcek-skor ceza hipotezi")
    P("")
    P("| ad | skor S | skor M | skor L | kapsam S | kapsam M | kapsam L |")
    P("|---|---|---|---|---|---|---|")
    for ad in ("R0", "T1", "W1", "W1_i0.6_s0.001_max", "W1_i0.6_s0.001_box_and_model_avg", "W1_i0.6_s0.001_absent", "B0_nms", "T2", "W2"):
        s = ok.get(ad)
        if s and ad not in atlanan_set(atlanan):
            P(f"| {ad} | {s['skor_S']:.3f} | {s['skor_M']:.3f} | {s['skor_L']:.3f} | {s['kapsam_S']:.3f} | {s['kapsam_M']:.3f} | {s['kapsam_L']:.3f} |")
    P("(S: sqrt(alan)<32 px, M: 32-96, L: >=96; girdi-tensoru pikseli.)")
    P("")
    P("## 5. Yontem notlari")
    P("- Tek val gecisi: her goruntu icin R0, T1 (3 dal), T2 (3 dal) ileri-geciri bir kez; WBF konfigurasyonlari ayni NMS'li dal ciktilari uzerinde isci surecleriyle; metrik "
      "Ultralytics `_process_batch` + `ap_per_class` (H07 ile ayni; mAP50-95 yeniden-uretim kontrolu yukarida).")
    P("- Atlanan (zaman butcesi): " + (", ".join(sorted(atlanan_set(atlanan))) or "yok"))
    Path(dosya_md).write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    return yapi, dict(d_w1_t1=d_w1_t1, d_best_t1=d_best_t1, best=en_iyi_c["ad"], d_best_w1=d_best_w1, avg_mx=avg_mx)


def atlanan_set(atlanan):
    return set(atlanan or [])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mod", default="tam", choices=["birim", "tam"])
    ap.add_argument("--run-dir", default=str(Path(os.environ.get("TEMP", ".")) / "_wbf_tani_tmp"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-img", type=int, default=20)
    ap.add_argument("--data", default=None, help="varsayilan WS/data/voc_dev.yaml (yalniz hata ayiklama icin mini yaml)")
    ap.add_argument("--cpu-deneme", action="store_true", help="tam modu CPU'da (kucuk alt-kume) tesisat testi icin calistir; kuyrukta KULLANILMAZ")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--isci", type=int, default=5)
    ap.add_argument("--zaman-butce-dk", type=float, default=34.0)
    ap.add_argument("--sonuc-md", default=str(WS / "sonuclar" / "wbf-tani.md"))
    ap.add_argument("--sonuc-csv", default=str(WS / "sonuclar" / "wbf-tani.csv"))
    ap.add_argument("--birim-json", default=str(WS / "sonuclar" / "wbf-tani-birim.json"))
    a = ap.parse_args()
    t_basla = time.time()
    import torch
    cuda_ok = torch.cuda.is_available()
    cihaz = "cuda:0" if cuda_ok else "cpu"
    cihaz_str = f"cuda:0 / {torch.cuda.get_device_name(0)}" if cuda_ok else "cpu"
    Path(a.run_dir).mkdir(parents=True, exist_ok=True)
    Path(a.sonuc_md).parent.mkdir(parents=True, exist_ok=True)
    print(f"[wbf-tani] mod={a.mod} cihaz={cihaz_str} agirlik={W0}")
    birim = birim_testleri(a.n_img, cihaz)
    Path(a.birim_json).write_text(json.dumps(birim, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[wbf-tani] birim:", json.dumps(birim["gecti"], ensure_ascii=False), "hepsi_gecti=", birim["hepsi_gecti"])
    if a.mod == "birim":
        t1 = birim["U1_tek_gorunum"]
        print(f"[wbf-tani] U1 temiz={t1['n_temiz']} birlesen={t1['n_birlesen']} max_fark_temiz={t1['max_fark_px_temiz']:.2e} px; kutu girdi/cikti {t1['n_kutu_girdi']}/{t1['n_kutu_cikti']}")
        print(f"[wbf-tani] dal_ort_kutu={birim['dal_ort_kutu']:.0f} dal_300={birim['dal_300_kesilen_pay']:.3f} tasma={birim['dal_tensor_disi_kutu_pay']:.3f}")
        return 0 if birim["hepsi_gecti"] else 3
    if not cuda_ok and not a.cpu_deneme:
        print("[wbf-tani] HATA: tam mod GPU ister (kuyruk --gpu-zorunlu)")
        return 2
    cfgs = konfigurasyonlar()
    S, kosan, tier_ok, gec = tam_calis(a, cfgs, t_basla)
    atlanan = [c["ad"] for c in cfgs if c["kind"] == "wbf" and not tier_ok[c["tier"]]]
    sure = time.time() - t_basla
    yapi, ozet = yaz_rapor(cfgs, S, atlanan, birim, sure, a.sonuc_md, a.sonuc_csv, cihaz_str)
    r0 = S["R0"]["map50_95"]
    r0_fark = r0 - BASE_VAL
    uyum = abs(r0_fark) <= 0.0008
    n_p = None
    try:
        from ultralytics import YOLO
        n_p = int(sum(p.numel() for p in YOLO(str(W0)).model.parameters()))
    except Exception:
        pass
    ag = {f"wbf_{ad}_map50_95": round(s["map50_95"], 6) for ad, s in S.items() if s}
    ag.update({f"wbf_{ad}_ap75": round(s["ap75"], 6) for ad, s in S.items() if s})
    metrics = {
        "val_map50_95": round(r0, 6), "val_map50": round(S["R0"]["map50"], 6),
        "egitim_sure_sn": round(sure, 2), "egitim_yok": 1, "param_sayisi": n_p, "cihaz": cihaz_str,
        "varyant_imzasi": "H07-WBF-tani: baseline s0 best.pt, yalniz val; WBF mekanizma testi + iou_thr/skip/conf_type izgarasi + dal alt-kumeleri (egitimsiz)",
        "varyant_etki_sayaci": int(sum(1 for ad, s in S.items() if s and ad != "R0" and abs(s["map50_95"] - r0) > 1e-9)),
        "wbf_r0_fark": round(r0_fark, 6), "wbf_r0_uyumsuz": int(not uyum),
        "wbf_W1_map50_95": round(S["W1"]["map50_95"], 6), "wbf_T1_map50_95": round(S["T1"]["map50_95"], 6),
        "wbf_W1_h07_fark": round(S["W1"]["map50_95"] - H07_REF["W1"], 6), "wbf_T1_h07_fark": round(S["T1"]["map50_95"] - H07_REF["T1"], 6),
        "wbf_birim_hepsi_gecti": int(birim["hepsi_gecti"]),
        "wbf_a_mekanizma_dogru": int(yapi["a"]), "wbf_b_etki_suruyor": int(yapi["b"]),
        "wbf_izgara_en_iyi_d_T1_puan": round(ozet["d_best_t1"], 4), "wbf_W1_d_T1_puan": round(ozet["d_w1_t1"], 4),
        "wbf_atlanan_konfig": len(atlanan),
        "alt_grup": ag,
    }
    assert not any(k.startswith(("test_", "rep_")) for k in metrics), "kilitli veri anahtari yazilamaz"
    out = Path(a.out) if a.out else Path(a.run_dir) / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[wbf-tani] bitti {sure:.0f} sn; R0 fark {r0_fark:+.6f}; W1={S['W1']['map50_95']:.5f} T1={S['T1']['map50_95']:.5f}; rapor {a.sonuc_md}")
    if not uyum:
        print(f"[wbf-tani] HATA: R0 baseline metrics.json'dan {abs(r0_fark):.6f} > 0.0008 sapti (rc=94)")
        return 94
    return 0


if __name__ == "__main__":
    sys.exit(main())
