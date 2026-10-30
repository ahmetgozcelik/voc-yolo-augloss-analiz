# -*- coding: utf-8 -*-
"""
H07 CPU birim testleri (GPU'suz; CUDA_VISIBLE_DEVICES=-1):
  T1  Gaussian Soft-NMS: elle hesap (ayni kutu, IoU=1, sigma=0.5 -> 0.8*exp(-2)); uzak kutu korunur; esik elemesi
  T2  Yari bolme: sirali stem, cift/tek indeks, ayrik, birlesim = tum liste, 500/500 (1000'lik gercek liste ile)
  T3  TTA yamasi: s=[1,.83,.67] iken ORIJINAL ultralytics _predict_augment ile BIREBIR ayni cikti; T2 (yukari) farkli;
      stash 3 dal icerir
  T4  Paket/bootstrap: dizin dilimleme == naif dilimleme; Paket.map == dogrudan ap_per_class; ayni paketler -> Delta=0;
      guclendirilmis satir -> Delta>0; tohum sabit -> tekrarlanabilir
  T5  WBF: iki ayni dal -> ayni kutular ve ayni skor (avg); gercek model TTA dallari -> sinirlar icinde, skor sirali, <=300
  T6  Soft-NMS uctan uca (gercek model, rastgele goruntu): cikti bicimi (N,6), <=300, skor sirali
  T7  Kaynak denetimi: bu dizinde kilitli veri bolumune referans yok
Kullanim: CUDA_VISIBLE_DEVICES=-1 python test_h07.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

_AD = Path(__file__).resolve().parent
sys.path.insert(0, str(_AD))
import degerlendir as D  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
WS = _AD.parents[1]


def t1():
    b = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
    s = np.array([0.9, 0.8, 0.5])
    idx, sk = D.soft_nms_sinif(b, s, 0.5)
    bekl = [0.9, 0.5, 0.8 * np.exp(-1.0 / 0.5)]
    print(f"T1 soft-nms: idx={idx} skor={[round(x, 6) for x in sk]} beklenen={[round(x, 6) for x in bekl]}")
    assert idx == [0, 2, 1] and np.allclose(sk, bekl, atol=1e-9)
    # kismi ortusme: IoU(0,1)=1/3 -> 0.8*exp(-(1/3)^2/0.5)
    b2 = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=float)  # IoU = 50/150 = 1/3
    idx2, sk2 = D.soft_nms_sinif(b2, np.array([0.9, 0.8]), 0.5)
    bekl2 = 0.8 * np.exp(-((1 / 3) ** 2) / 0.5)
    print(f"T1 kismi ortusme: {sk2[1]:.6f} beklenen {bekl2:.6f}")
    assert abs(sk2[1] - bekl2) < 1e-9
    # esik: cok dusuk skor elenir
    idx3, _ = D.soft_nms_sinif(np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float), np.array([0.5, 0.004]), 0.05)
    assert idx3 == [0], idx3  # 0.004*exp(-1/0.05)=~0 < 0.001 elenir
    # torch sarmali: (N,6) bicimi korunur, <=300
    x = torch.tensor([[0, 0, 10, 10, 0.9, 1], [0, 0, 10, 10, 0.8, 1], [0, 0, 10, 10, 0.7, 2]], dtype=torch.float32)
    o = D.soft_nms_goruntu(x, 0.5)
    print(f"T1 torch sarmali: {o.tolist()}")
    assert o.shape[1] == 6 and o.shape[0] == 3 and (o[:-1, 4] >= o[1:, 4]).all()
    assert abs(o[1, 4].item() - 0.7) < 1e-6   # farkli sinif bozunmaz


def t2():
    liste = [ln.strip() for ln in (WS / "data" / "splits" / "val.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    A, B = D.yarilari_bol(liste)
    sA, sB = {Path(f).name for f in A}, {Path(f).name for f in B}
    print(f"T2 yari bolme: |liste|={len(liste)} |A|={len(A)} |B|={len(B)} kesisim={len(sA & sB)} birlesim={len(sA | sB)}")
    assert len(A) == 500 and len(B) == 500 and not (sA & sB) and (sA | sB) == {Path(f).name for f in liste}
    sirali = sorted(liste, key=lambda f: (Path(f).stem, Path(f).name))
    assert A == sirali[0::2] and B == sirali[1::2]
    A2, B2 = D.yarilari_bol(list(reversed(liste)))
    assert A2 == A and B2 == B, "bolme girdi sirasindan bagimsiz olmali"


def model_yukle():
    from ultralytics.nn.autobackend import AutoBackend
    w = WS / "runs" / "20260930-072103_baseline_s0" / "train" / "weights" / "best.pt"
    ab = AutoBackend(model=str(w), device=torch.device("cpu"), fp16=False)
    ab.eval()
    return ab


def t3(ab):
    from ultralytics.nn.tasks import DetectionModel
    orijinal = DetectionModel._predict_augment
    x = torch.rand(1, 3, 192, 256)
    with torch.inference_mode():
        ref = orijinal(ab.model, x.clone())[0]
    D.tta_yamasi_uygula()
    D.TTA["s"] = list(D.STD_S)
    with torch.inference_mode():
        yeni = DetectionModel._predict_augment(ab.model, x.clone())[0]
    fark = (ref - yeni).abs().max().item()
    print(f"T3 yama(s=std) vs orijinal: sekil {tuple(ref.shape)} {tuple(yeni.shape)} max|fark|={fark:.2e}; stash dal sayisi={len(D.TTA['stash']['y'])} hw={D.TTA['stash']['hw']}")
    assert ref.shape == yeni.shape and fark == 0.0
    assert len(D.TTA["stash"]["y"]) == 3
    D.TTA["s"] = list(D.UP_S)
    with torch.inference_mode():
        up = DetectionModel._predict_augment(ab.model, x.clone())[0]
    print(f"T3 T2(yukari) sekil={tuple(up.shape)} (std {tuple(yeni.shape)}); ortalama |fark| (ayni sekilse)="
          f"{(up - yeni).abs().mean().item() if up.shape == yeni.shape else 'sekil farkli'}")
    assert up.shape != yeni.shape or (up - yeni).abs().max().item() > 0
    D.TTA["s"] = list(D.STD_S)
    return x


def t4():
    rng = np.random.default_rng(0)
    n = 40
    names = [f"im{i}.jpg" for i in range(n)]
    tp_l, conf_l, pcls_l, tcls_l = [], [], [], []
    for i in range(n):
        k = rng.integers(0, 12)
        conf_l.append(rng.random(k))
        pcls_l.append(rng.integers(0, 5, k).astype(float))
        tp_l.append(rng.random((k, 10)) < np.linspace(0.8, 0.1, 10)[None, :])
        tcls_l.append(rng.integers(0, 5, rng.integers(1, 6)).astype(float))
    P = D.Paket(names, tp_l, conf_l, pcls_l, tcls_l)
    idx = rng.integers(0, n, 60)
    naif = np.concatenate([np.arange(P.off[i], P.off[i + 1]) for i in idx])
    assert np.array_equal(P.dizin(idx), naif)
    from ultralytics.utils.metrics import ap_per_class
    dogru = ap_per_class(np.concatenate(tp_l), np.concatenate(conf_l), np.concatenate(pcls_l), np.concatenate(tcls_l))[5]
    m, m50 = P.map()
    print(f"T4 Paket.map={m:.6f} dogrudan={dogru.mean():.6f} map50={m50:.6f}")
    assert abs(m - dogru.mean()) < 1e-12 and abs(m50 - dogru[:, 0].mean()) < 1e-12
    # guclendirilmis satir: tum tp'ler True
    P2 = D.Paket(names, [np.ones_like(t) for t in tp_l], conf_l, pcls_l, tcls_l)
    d1 = D.bootstrap_delta({"R0": P, "X": P2, "Y": P}, ["X", "Y"], 30, 0)
    d2 = D.bootstrap_delta({"R0": P, "X": P2, "Y": P}, ["X", "Y"], 30, 0)
    print(f"T4 bootstrap: X ort Delta={d1['X'].mean():+.4f} (>0), Y ort Delta={d1['Y'].mean():+.4f} (=0); tekrarlanabilir={np.array_equal(d1['X'], d2['X'])}")
    assert (d1["X"] > 0).all() and np.allclose(d1["Y"], 0) and np.array_equal(d1["X"], d2["X"])


def t5(ab, x):
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import nms
    # iki ayni dal -> WBF sonucu ayni kutu/skor
    y = torch.zeros(1, 4 + 20, 50)
    # 5 birbirinden uzak kutu (xywh), farkli skor, sinif 3
    merkez = torch.tensor([[40., 40.], [120., 40.], [200., 40.], [40., 160.], [200., 160.]])
    y[0, 0, :5], y[0, 1, :5] = merkez[:, 0], merkez[:, 1]
    y[0, 2, :5], y[0, 3, :5] = 30.0, 40.0
    y[0, 4 + 3, :5] = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
    stash = {"y": [y.clone(), y.clone()], "hw": (256, 320)}
    tek = nms.non_max_suppression(y, D.CONF, D.IOU0, nc=0, multi_label=True, max_det=300)[0]
    w = D.wbf_goruntuler(stash, nms)[0]
    print(f"T5 WBF ayni iki dal: NMS tek dal {tek.shape[0]} kutu, WBF {w.shape[0]} kutu; skor ort NMS={tek[:, 4].mean():.4f} WBF={w[:, 4].mean():.4f}")
    assert w.shape[0] == tek.shape[0] and abs(w[:, 4].sort(descending=True)[0] - tek[:, 4].sort(descending=True)[0]).max() < 1e-4
    # gercek model dallari
    D.TTA["s"] = list(D.STD_S)
    with torch.inference_mode():
        DetectionModel._predict_augment(ab.model, x.clone())
        out = D.wbf_goruntuler(D.TTA["stash"], nms)[0]
    H, W = D.TTA["stash"]["hw"]
    print(f"T5 gercek model WBF: {out.shape[0]} kutu, max skor={out[:, 4].max().item() if len(out) else 0:.3f}")
    assert out.shape[1] == 6 and out.shape[0] <= 300
    assert (out[:, 4][:-1] >= out[:, 4][1:]).all() if len(out) > 1 else True
    if len(out):
        assert out[:, [0, 2]].min() >= -1e-3 and out[:, [0, 2]].max() <= W + 1e-3 and out[:, [1, 3]].max() <= H + 1e-3


def t6(ab, x):
    cfg = dict(D.R0, soft=0.5)
    with torch.inference_mode():
        preds = ab(x, augment=False)
        o = D.son_isle(cfg, preds)
        n = D.son_isle(D.R0, preds)
    print(f"T6 soft-nms uctan uca: {o[0].shape} <=300 skor sirali={(o[0][:-1, 4] >= o[0][1:, 4]).all().item()} | NMS kutu sayisi {n[0].shape[0]}, soft {o[0].shape[0]}")
    assert o[0].shape[1] == 6 and o[0].shape[0] <= 300
    assert (o[0][:-1, 4] >= o[0][1:, 4]).all() if o[0].shape[0] > 1 else True
    # soft-nms en yuksek skorlu kutuyu korur
    assert abs(o[0][0, 4].item() - n[0][0, 4].item()) < 1e-6


def t7():
    yasak = ["voc_" + "test", "test" + "_kilitli", "rep_" + "test", "splits/" + "test"]
    bulunan = []
    for p in _AD.glob("*.py"):
        if p.name == "test_h07.py":
            continue
        s = p.read_text(encoding="utf-8")
        for y in yasak:
            if y in s:
                bulunan.append((p.name, y))
    print(f"T7 kaynak denetimi (degerlendir.py vb.): yasakli dize bulunan = {bulunan}")
    assert not bulunan


if __name__ == "__main__":
    print("cihaz: cuda_available =", torch.cuda.is_available())
    t1(); t2()
    ab = model_yukle()
    x = t3(ab)
    t4(); t5(ab, x); t6(ab, x); t7()
    print("TUM H07 CPU TESTLERI GECTI")
