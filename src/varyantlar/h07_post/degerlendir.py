# -*- coding: utf-8 -*-
"""
H07 degerlendirme betigi — EGITIMSIZ son-islem / test-zamani artirma satirlari (yalniz VAL; hipotezler/H07.md).

Kapsam: baseline seed0 best.pt uzerinde R0 + 12 satir:
  N50 N60 N80      : NMS iou 0.5/0.6/0.8                      (aile N)
  S03 S05          : Gaussian Soft-NMS sigma 0.3/0.5          (aile S)
  U704 U768 U832   : test imgsz 704/768/832, tek olcek        (aile U)
  T1 T2            : TTA (T1 = Ultralytics [1,.83,.67]+lr-flip; T2 = [1,1.2,1.4]+lr-flip)  (aile T)
  W1 W2            : WBF (iou_thr .6, skip .001, avg) T1 / T2 dallarina (aile W)
Her satir referans hattindan (conf 0.001, iou 0.7, max_det 300, imgsz 640, tek olcek, tek NMS) YALNIZ bir bilesenle
farklidir; satirlar birlestirilmez. Karar kurali (kart): yari-A'da aile-icinde en iyi (Delta>0) satir ADAY; ADAY ->
(a) yari-B Delta >= +0.30 puan, (b) tam-val paired goruntu-bootstrap (B=200) tek-yonlu %90 alt sinir > 0,
(c) baseline seed1 + seed2 best.pt'de ayni satir icin ort Delta >= +0.20 ve iki Delta da > 0 => DOGRULANDI.
W ailesinde yalniz yari-A'da daha iyi olan T satirinin W'si aday olabilir (kart: "T1/T2'den daha iyi olanin
dallarina"); diger W satiri 'ek' olarak raporlanir, aday olmaz.

Degerlendirme motoru: Ultralytics DetectionValidator (rakip_kos.degerlendir ile AYNI argumanlar: batch 16, save_json,
half yok) — bu yuzden R0 dogrudan baseline metrics.json val_map50_95'i (0.62893) yeniden uretir (tolerans 0.0008; asilirsa
kart durur, rc=94). Metrik: native mAP50-95 (ap_per_class), AP_S/M/L: faster-coco-eval (validator). Goruntu-bazli
istatistikler (tp, conf, pred_cls, target_cls) validator.metrics.stats'tan alinir; yarim-bolme ve bootstrap bunlardan
ap_per_class ile yeniden hesaplanir (ayni metrik). Kart metni bootstrap icin "pycocotools yeniden ornekleme" diyordu;
Delta'nin tanimli oldugu native val_map50_95 uzerinde yeniden ornekleme yapildi (pycocotools duplike goruntu
kimligini desteklemez) — README'de belgelendi; COCO AP (faster-coco-eval) tam-val'de ikincil sutun.

Gecikme: batch=1, FP16, ayni oturumda, satirlar ARAKESITLI (3 tur, medyan), 50 isinma + 300 olcum goruntusu
(data/splits/val.txt'in ilk 350'si — rakip_kos ile ayni), ISINMA DISLANIR; olculen sure = onisleme (letterbox+tensor+cihaza
kopya) + model (TTA dahil) + NMS/Soft-NMS/WBF + kutu olcekleme + cikti cpu'ya = UCTAN UCA. Olcum kosul conf=0.001
(degerlendirme isletim noktasi; NMS payi en yuksek), R0_dep (conf 0.25, tek-etiket) ayrica (baseline 8.0 ms ile baglanti).
R0 tekrar CV>%3 ise tur sayisi artirilir (en cok 6).

Bu betik YALNIZ voc_dev.yaml (val listesi) okur; kilitli veri bolumune referans YOKTUR.
Kullanim (kuyruk):
  python varyantlar/h07_post/degerlendir.py --run-dir {RUN_DIR} --out {RUN_DIR}/metrics.json
Kuru calisma (CPU, kucuk altkume): --data <mini.yaml> --bs-B 10 --bs-isci 0 --gecikme-isinma 3 --gecikme-olcum 10 --gecikme-tekrar 1
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

WS = Path(__file__).resolve().parents[2]
CONF = 0.001
IOU0 = 0.7
MAXDET = 300
HAVUZ = 1000          # Soft-NMS aday havuzu: skor>=0.001 adaylarin en yuksek 1000'i (iou=1.0 on-NMS = bastirma yok)
SOFT_ESIK = 0.001     # Soft-NMS skor esigi
R0_TOL = 0.0008
STD_S = [1, 0.83, 0.67]
UP_S = [1, 1.2, 1.4]
TTA = {"s": list(STD_S), "stash": None}   # _predict_augment yamasinin paylasilan durumu

R0 = dict(ad="R0", aile="R0", iou=IOU0, imgsz=640, augment=False, tta_s=None, soft=None, wbf=False)


def _c(ad, aile, **k):
    d = dict(R0)
    d.update(ad=ad, aile=aile)
    d.update(k)
    return d


SATIRLAR = [
    _c("N50", "N", iou=0.5), _c("N60", "N", iou=0.6), _c("N80", "N", iou=0.8),
    _c("S03", "S", soft=0.3), _c("S05", "S", soft=0.5),
    _c("U704", "U", imgsz=704), _c("U768", "U", imgsz=768), _c("U832", "U", imgsz=832),
    _c("T1", "T", augment=True, tta_s=STD_S), _c("T2", "T", augment=True, tta_s=UP_S),
    _c("W1", "W", augment=True, tta_s=STD_S, wbf=True), _c("W2", "W", augment=True, tta_s=UP_S, wbf=True),
]
KIRMIZI_PUAN = {"N": 1.0, "S": 1.0, "W": 1.5, "T": 2.0, "U": 2.0}   # kart: tam-val Delta > bu (puan) => kirmizi bayrak
ALT_AD = {"N": "NMS-IoU", "S": "Soft-NMS", "U": "test-imgsz", "T": "TTA", "W": "WBF"}


def simdi():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def atomik_yaz(yol: Path, veri):
    yol = Path(yol)
    yol.parent.mkdir(parents=True, exist_ok=True)
    tmp = yol.with_suffix(yol.suffix + ".tmp")
    tmp.write_text(json.dumps(veri, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, yol)


# --------------------------------------------------------------------------------------------------------------
# TTA yamasi: DetectionModel._predict_augment ile ayni govde; yalniz (1) olcek listesi TTA["s"]'den gelir,
# (2) dal ciktilari (clip sonrasi) WBF icin TTA["stash"]'e konur. s=STD_S iken orijinalle BIREBIR aynidir (test_h07.py).
# --------------------------------------------------------------------------------------------------------------
def tta_yamasi_uygula():
    from ultralytics.nn.modules import Detect
    from ultralytics.nn.tasks import BaseModel, DetectionModel
    from ultralytics.utils import LOGGER
    from ultralytics.utils.torch_utils import scale_img
    import torch

    def _predict_augment(self, x):
        if getattr(self, "end2end", False) or type(self.model[-1]) is not Detect:
            LOGGER.warning("Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]
        s = TTA["s"]
        f = [None, 3, None]  # flips (2-ud, 3-lr) — orijinalle ayni; yalniz s degisir
        assert len(s) == len(f)
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = BaseModel.predict(self, xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        TTA["stash"] = {"y": list(y), "hw": (int(img_size[0]), int(img_size[1]))}
        return torch.cat(y, -1), None

    DetectionModel._predict_augment = _predict_augment


# --------------------------------------------------------------------------------------------------------------
# Son-islem: NMS / Soft-NMS / WBF (validator ve gecikme motoru AYNI fonksiyonu kullanir)
# --------------------------------------------------------------------------------------------------------------
def _iou_matris(b):
    x1 = np.maximum(b[:, None, 0], b[None, :, 0])
    y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2])
    y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (a[:, None] + a[None, :] - inter + 1e-9)


def soft_nms_sinif(b, s, sigma, esik=SOFT_ESIK):
    """Gaussian Soft-NMS (Bodla 2017), tek sinif: secilen kutunun skoru korunur, kalanlar exp(-IoU^2/sigma) ile azalir;
    skoru esigin altina inen elenir. Donus: (secilen indeksler, bozunmus skorlar) — secim sirasi skor sirasidir."""
    n = len(s)
    iou = _iou_matris(b)
    s = s.astype(np.float64).copy()
    canli = np.ones(n, bool)
    idx, sk = [], []
    for _ in range(n):
        sm = np.where(canli, s, -1.0)
        m = int(sm.argmax())
        if sm[m] < esik:
            break
        idx.append(m)
        sk.append(float(s[m]))
        canli[m] = False
        s = s * np.where(canli, np.exp(-(iou[m] ** 2) / sigma), 1.0)
    return idx, sk


def soft_nms_goruntu(x, sigma):
    """x: torch (N,6) [xyxy, conf, cls] (havuz, bastirmasiz). Donus: ayni bicimde (<=300, skor sirali)."""
    import torch
    if x.shape[0] == 0:
        return x
    xn = x.detach().float().cpu().numpy()
    parcalar = []
    for c in np.unique(xn[:, 5]):
        m = xn[:, 5] == c
        b, s = xn[m, :4], xn[m, 4]
        idx, sk = soft_nms_sinif(b, s, sigma)
        if idx:
            parcalar.append(np.concatenate([b[idx], np.asarray(sk)[:, None], np.full((len(idx), 1), c)], 1))
    if not parcalar:
        return x[:0]
    o = np.concatenate(parcalar, 0)
    o = o[np.argsort(-o[:, 4], kind="stable")][:MAXDET]
    return torch.as_tensor(o, dtype=x.dtype, device=x.device)


def wbf_goruntuler(stash, nms):
    """stash: {'y': [dal tensorleri (bs,4+nc,A_b)], 'hw': (H,W)}. Her dal ayri NMS (conf .001, iou .7, max_det 300),
    sonra goruntu basina WBF (iou_thr .6, skip .001, conf_type avg); normalize koordinat (girdi tensoru H,W)."""
    import torch
    from ensemble_boxes import weighted_boxes_fusion
    H, W = stash["hw"]
    dallar = [nms.non_max_suppression(yb, CONF, IOU0, nc=0, multi_label=True, max_det=MAXDET) for yb in stash["y"]]
    bs = len(dallar[0])
    olcek = np.array([W, H, W, H], dtype=np.float64)
    cikti = []
    for i in range(bs):
        bl, sl, ll = [], [], []
        ref = dallar[0][i]
        for d in dallar:
            xi = d[i].detach().float().cpu().numpy()
            if xi.shape[0] == 0:
                continue
            bl.append(np.clip(xi[:, :4] / olcek, 0.0, 1.0))
            sl.append(xi[:, 4])
            ll.append(xi[:, 5].astype(int))
        if not bl:
            cikti.append(ref[:0, :6])
            continue
        b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=None, iou_thr=0.6, skip_box_thr=CONF, conf_type="avg")
        o = np.concatenate([b * olcek, s[:, None], l[:, None].astype(np.float64)], 1)
        o = o[np.argsort(-o[:, 4], kind="stable")][:MAXDET]
        cikti.append(torch.as_tensor(o, dtype=ref.dtype, device=ref.device))
    return cikti


def son_isle(cfg, preds, conf=CONF, multi_label=True):
    """Ham model ciktisi -> goruntu basina (N,6) tensor listesi (girdi-tensoru koordinati)."""
    from ultralytics.utils import nms
    p = preds[0] if isinstance(preds, (list, tuple)) else preds
    if cfg.get("wbf"):
        return wbf_goruntuler(TTA["stash"], nms)
    if cfg.get("soft"):
        havuz = nms.non_max_suppression(p, conf, 1.0, nc=0, multi_label=True, max_det=HAVUZ)
        return [soft_nms_goruntu(x, cfg["soft"]) for x in havuz]
    return nms.non_max_suppression(p, conf, cfg["iou"], nc=0, multi_label=multi_label, max_det=MAXDET)


# --------------------------------------------------------------------------------------------------------------
# Goruntu-bazli istatistik paketi, mAP, bootstrap
# --------------------------------------------------------------------------------------------------------------
def map_hesapla(tp, conf, pcls, tcls):
    from ultralytics.utils.metrics import ap_per_class
    if len(tcls) == 0:
        return 0.0, 0.0
    if len(conf) == 0:
        return 0.0, 0.0
    ap = ap_per_class(tp, conf, pcls, tcls)[5]
    return float(ap.mean()), float(ap[:, 0].mean())


class Paket:
    """Goruntu-bazli validator istatistikleri; alt-kume / bootstrap icin hizli dilimleme."""

    def __init__(self, names, tp_l, conf_l, pcls_l, tcls_l):
        self.names = list(names)
        self.n = len(self.names)
        self.len = np.array([len(c) for c in conf_l], dtype=np.int64)
        self.off = np.concatenate([[0], np.cumsum(self.len)])
        self.tp = np.concatenate([np.asarray(t, dtype=bool).reshape(-1, 10) for t in tp_l], 0) if self.off[-1] else np.zeros((0, 10), bool)
        self.conf = np.concatenate([np.asarray(c, dtype=float).reshape(-1) for c in conf_l]) if self.off[-1] else np.zeros(0)
        self.pcls = np.concatenate([np.asarray(c, dtype=float).reshape(-1) for c in pcls_l]) if self.off[-1] else np.zeros(0)
        self.tcls_l = [np.asarray(t, dtype=float).reshape(-1) for t in tcls_l]

    def dizin(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        ln = self.len[idx]
        top = int(ln.sum())
        if top == 0:
            return np.zeros(0, np.int64)
        basl = self.off[idx]
        return np.repeat(basl - (np.cumsum(ln) - ln), ln) + np.arange(top)

    def map(self, idx=None):
        if idx is None:
            idx = np.arange(self.n)
        ii = self.dizin(idx)
        tc = np.concatenate([self.tcls_l[i] for i in idx]) if len(idx) else np.zeros(0)
        return map_hesapla(self.tp[ii], self.conf[ii], self.pcls[ii], tc)

    @property
    def n_pred(self):
        return int(self.off[-1])


_BS = {}


def _bs_init(paketler):
    _BS["p"] = paketler


def _bs_tur(args):
    tohum, n = args
    rng = np.random.default_rng(tohum)
    idx = rng.integers(0, n, n)
    return {k: p.map(idx)[0] for k, p in _BS["p"].items()}


def bootstrap_delta(paketler, satirlar, B, isci, tohum0=12345):
    """Eslestirilmis goruntu-bootstrap: her orneklemde R0 ve her aday satir AYNI goruntu dizinleriyle. Donus {satir: Delta dizisi}."""
    n = paketler["R0"].n
    alt = {k: paketler[k] for k in ["R0"] + list(satirlar)}
    gorevler = [(tohum0 + b, n) for b in range(B)]
    if isci and isci > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(isci, initializer=_bs_init, initargs=(alt,)) as pool:
            sonuc = pool.map(_bs_tur, gorevler, chunksize=max(1, B // (isci * 4)))
    else:
        _bs_init(alt)
        sonuc = [_bs_tur(g) for g in gorevler]
    return {k: np.array([s[k] - s["R0"] for s in sonuc]) for k in satirlar}


# --------------------------------------------------------------------------------------------------------------
# Dogruluk satiri (DetectionValidator)
# --------------------------------------------------------------------------------------------------------------
def dogrulayici_sinifi():
    from ultralytics.models.yolo.detect import DetectionValidator

    class HVal(DetectionValidator):
        h07_cfg = None

        def postprocess(self, preds):
            out = son_isle(self.h07_cfg, preds)
            return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in out]

        def update_metrics(self, preds, batch):
            if not hasattr(self, "h07_names"):
                self.h07_names = []
            self.h07_names.extend(Path(f).name for f in batch["im_file"])
            return super().update_metrics(preds, batch)

        def get_stats(self):
            self.h07_stats = {k: list(v) for k, v in self.metrics.stats.items()}
            return super().get_stats()

    return HVal


def satir_kos(w_path, cfg, data_yaml, cihaz_arg, run_dir, batch, workers):
    """Tek konfigurasyonu tam val'de kosar. Donus: {'stats': validator stats dict, 'paket': Paket}."""
    TTA["s"] = list(cfg["tta_s"]) if cfg.get("tta_s") else list(STD_S)
    TTA["stash"] = None
    val_args = dict(
        data=str(data_yaml), split="val", imgsz=cfg["imgsz"], batch=batch, device=cihaz_arg, workers=workers,
        plots=False, save_json=True, project=str(run_dir), name=f"val_{cfg['ad']}", exist_ok=True, verbose=False,
        model=str(w_path),
    )
    if cfg.get("augment"):
        val_args["augment"] = True
    if not cfg.get("soft") and not cfg.get("wbf"):
        val_args["iou"] = cfg["iou"]
    v = dogrulayici_sinifi()(args=val_args)
    v.h07_cfg = cfg
    stats = v(model=str(w_path))
    st = v.h07_stats
    paket = Paket(v.h07_names, st["tp"], st["conf"], st["pred_cls"], st["target_cls"])
    try:  # kayit dosyasi (predictions.json) disk sisirir; silinir
        pj = Path(v.save_dir) / "predictions.json"
        if pj.exists():
            pj.unlink()
    except Exception:
        pass
    return {"stats": stats, "paket": paket}


# --------------------------------------------------------------------------------------------------------------
# Gecikme motoru (batch=1, uctan uca)
# --------------------------------------------------------------------------------------------------------------
class Motor:
    def __init__(self, w_path, device, half):
        import torch
        from ultralytics.nn.autobackend import AutoBackend
        self.torch = torch
        self.device = device
        self.ab = AutoBackend(model=str(w_path), device=device, fp16=bool(half))
        self.ab.eval()
        self.stride = int(self.ab.stride)
        self.half = bool(self.ab.fp16)
        self.cuda = device.type == "cuda"
        self.lb = {}
        self.ab.warmup(imgsz=(1, 3, 640, 640))

    def _letterbox(self, im0, imgsz):
        from ultralytics.data.augment import LetterBox
        lb = self.lb.get(imgsz)
        if lb is None:
            lb = self.lb[imgsz] = LetterBox((imgsz, imgsz), auto=True, stride=self.stride)
        return lb(image=im0)

    def tek(self, im0, cfg, conf=CONF, multi_label=True):
        from ultralytics.utils import ops
        torch = self.torch
        TTA["s"] = list(cfg["tta_s"]) if cfg.get("tta_s") else list(STD_S)
        im = self._letterbox(im0, cfg["imgsz"])
        im = np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))[None]
        x = torch.from_numpy(im).to(self.device)
        x = x.half() if self.half else x.float()
        x = x / 255
        with torch.inference_mode():
            preds = self.ab(x, augment=bool(cfg.get("augment")))
            out = son_isle(cfg, preds, conf=conf, multi_label=multi_label)
            o = out[0]
            o[:, :4] = ops.scale_boxes(x.shape[2:], o[:, :4], im0.shape)
            sonuc = o.float().cpu().numpy()
        if self.cuda:
            torch.cuda.synchronize()
        return sonuc


def gecikme_olc(motor, cfg, imgs, isinma, olcum, conf=CONF, multi_label=True):
    for im in imgs[:isinma]:
        motor.tek(im, cfg, conf, multi_label)
    sure = []
    for im in imgs[isinma:isinma + olcum]:
        t = time.perf_counter()
        motor.tek(im, cfg, conf, multi_label)
        sure.append(time.perf_counter() - t)
    return 1000.0 * float(np.mean(sure)), len(sure)


def gpu_kullanim():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
        u, m = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")]
        return u, m
    except Exception:
        return None, None


# --------------------------------------------------------------------------------------------------------------
def yarilari_bol(names):
    """Sirali stem'e gore deterministik bolme: cift indeks -> A, tek indeks -> B."""
    sirali = sorted(names, key=lambda f: (Path(f).stem, Path(f).name))
    return sirali[0::2], sirali[1::2]


def val_listesi(data_yaml):
    import yaml
    y = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    kok = Path(y["path"])
    lst = kok / y["val"]
    return [ln.strip() for ln in lst.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default=None, help="varsayilan WS/data/voc_dev.yaml (kuru calismada mini yaml)")
    ap.add_argument("--agirlik", action="append", default=[], help="s0=YOL (tekrarlanabilir); varsayilan baseline s0/s1/s2 best.pt")
    ap.add_argument("--sonuc-csv", default=None, help="varsayilan WS/sonuclar/h07_post.csv")
    ap.add_argument("--yari-dizin", default=None, help="varsayilan WS/sonuclar (h07_yari_A.txt / h07_yari_B.txt)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bs-B", type=int, default=200)
    ap.add_argument("--bs-isci", type=int, default=6)
    ap.add_argument("--gecikme-isinma", type=int, default=50)
    ap.add_argument("--gecikme-olcum", type=int, default=300)
    ap.add_argument("--gecikme-tekrar", type=int, default=3)
    ap.add_argument("--gecikme-tekrar-max", type=int, default=6)
    ap.add_argument("--atla-gecikme", action="store_true")
    ap.add_argument("--atla-s12", action="store_true")
    a = ap.parse_args()

    t_basla = time.time()
    run_dir = Path(a.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(a.out).resolve() if a.out else run_dir / "metrics.json"
    kuru = a.data is not None
    data_yaml = Path(a.data).resolve() if a.data else WS / "data" / "voc_dev.yaml"
    csv_yol = Path(a.sonuc_csv) if a.sonuc_csv else WS / "sonuclar" / "h07_post.csv"
    yari_dizin = Path(a.yari_dizin) if a.yari_dizin else WS / "sonuclar"
    progress = run_dir / "progress.json"
    agir = {"s0": WS / "runs" / "20260930-072103_baseline_s0" / "train" / "weights" / "best.pt",
            "s1": WS / "runs" / "20260930-101624_baseline_s1" / "train" / "weights" / "best.pt",
            "s2": WS / "runs" / "20260930-131427_baseline_s2" / "train" / "weights" / "best.pt"}
    for kv in a.agirlik:
        k, v = kv.split("=", 1)
        agir[k] = Path(v)
    bekle_val = {"s0": 0.62893, "s1": 0.635322, "s2": 0.627787}   # baseline metrics.json val_map50_95 (R0 tekrar uretim kontrolu)

    import torch
    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_ok else "cpu")
    device_arg = 0 if cuda_ok else "cpu"
    cihaz_str = f"cuda:0 / {torch.cuda.get_device_name(0)}" if cuda_ok else "cpu"
    print(f"[h07] cihaz: {cihaz_str} | data={data_yaml} | kuru={kuru} | agirliklar={ {k: str(v) for k, v in agir.items()} }")
    gpu_u0, gpu_m0 = gpu_kullanim()
    print(f"[h07] baslangic GPU kullanim={gpu_u0}% bellek={gpu_m0} MiB")

    def prog(adim, **k):
        d = {"adim": adim, "epoch": adim, "zaman": simdi(), **k}
        try:
            atomik_yaz(progress, d)
        except Exception:
            pass

    tta_yamasi_uygula()

    # ---- val listesi, yarim bolme (A/B) ----
    liste = val_listesi(data_yaml)
    yA, yB = yarilari_bol(liste)
    yari_dizin.mkdir(parents=True, exist_ok=True)
    (yari_dizin / "h07_yari_A.txt").write_text("\n".join(yA) + "\n", encoding="utf-8")
    (yari_dizin / "h07_yari_B.txt").write_text("\n".join(yB) + "\n", encoding="utf-8")
    setA = {Path(f).name for f in yA}
    print(f"[h07] val={len(liste)} goruntu; yari A={len(yA)} B={len(yB)} (sirali stem, cift/tek indeks) -> {yari_dizin}")

    # ---- ASAMA 1: s0 uzerinde R0 + 12 satir ----
    w0 = agir["s0"]
    sonuc = {}
    adim = 0
    for cfg in [R0] + SATIRLAR:
        t = time.time()
        r = satir_kos(w0, cfg, data_yaml, device_arg, run_dir, a.batch, a.workers)
        sonuc[cfg["ad"]] = r
        adim += 1
        m = r["stats"].get("metrics/mAP50-95(B)")
        prog(adim, asama="s0_satirlar", satir=cfg["ad"], val_map50_95=float(m), sure_sn=round(time.time() - t, 1))
        print(f"[h07] {cfg['ad']}: map50_95={m:.5f} ap_s={r['stats'].get('metrics/mAP_small(B)')} n_pred/goruntu={r['paket'].n_pred / max(r['paket'].n, 1):.1f} ({time.time() - t:.0f} sn)")

    # ---- R0 yeniden-uretim kontrolu (kart: >0.0008 sapma => kart durur) ----
    r0_val = float(sonuc["R0"]["stats"]["metrics/mAP50-95(B)"])
    r0_kendi, _ = sonuc["R0"]["paket"].map()
    r0_fark = r0_val - bekle_val["s0"]
    print(f"[h07] R0 dogrulama: validator={r0_val:.6f} kendi(ap_per_class)={r0_kendi:.6f} baseline metrics.json={bekle_val['s0']} fark={r0_fark:+.6f}")
    if abs(r0_kendi - r0_val) > 1e-6:
        print(f"[h07] UYARI: kendi ap_per_class hesabi validator'dan {abs(r0_kendi - r0_val):.2e} farkli")
    r0_uyumsuz = (not kuru) and abs(r0_fark) > R0_TOL
    R0p = sonuc["R0"]["paket"]
    idxA = np.array([i for i, nme in enumerate(R0p.names) if nme in setA])
    idxB = np.array([i for i, nme in enumerate(R0p.names) if nme not in setA])
    for k, r in sonuc.items():   # tum satirlar ayni goruntu sirasini kullanmali
        assert r["paket"].names == R0p.names, f"{k}: goruntu sirasi R0'dan farkli"

    satirlar = {}
    r0A, r0B = R0p.map(idxA)[0], R0p.map(idxB)[0]
    for cfg in SATIRLAR:
        ad = cfg["ad"]
        p = sonuc[ad]["paket"]
        st = sonuc[ad]["stats"]
        mA, mB = p.map(idxA)[0], p.map(idxB)[0]
        satirlar[ad] = dict(
            ad=ad, aile=cfg["aile"], map50_95=float(st["metrics/mAP50-95(B)"]), map50=float(st["metrics/mAP50(B)"]),
            ap_s=float(st.get("metrics/mAP_small(B)", float("nan"))), ap_m=float(st.get("metrics/mAP_medium(B)", float("nan"))),
            ap_l=float(st.get("metrics/mAP_large(B)", float("nan"))),
            map_A=float(mA), map_B=float(mB), d_full=float(st["metrics/mAP50-95(B)"]) - r0_val,
            d_A=float(mA - r0A), d_B=float(mB - r0B),
            n_pred_goruntu=p.n_pred / max(p.n, 1),
            ayni_mi=bool(abs(float(st["metrics/mAP50-95(B)"]) - r0_val) < 1e-12 and p.n_pred == R0p.n_pred),
        )
    # W kart kurali: yarim-A'da daha iyi T satirinin W'si aday
    w_secilen = "W1" if satirlar["T1"]["map_A"] >= satirlar["T2"]["map_A"] else "W2"
    for ad in satirlar:
        satirlar[ad]["aday_uygun"] = (satirlar[ad]["aile"] != "W") or (ad == w_secilen)
    adaylar = {}
    for aile in ("N", "S", "U", "T", "W"):
        ak = [s for s in satirlar.values() if s["aile"] == aile and s["aday_uygun"]]
        en = max(ak, key=lambda s: s["d_A"])
        if en["d_A"] > 0:
            adaylar[aile] = en["ad"]
    print(f"[h07] R0: A={r0A:.5f} B={r0B:.5f} | W kart kurali secilen={w_secilen} | adaylar (yari-A en iyi, Delta>0)={adaylar}")
    for s in satirlar.values():
        s["aday"] = adaylar.get(s["aile"]) == s["ad"]

    # ---- bootstrap (tam val, eslestirilmis) ----
    bs = {}
    if adaylar and not r0_uyumsuz:
        t = time.time()
        pk = {k: sonuc[k]["paket"] for k in ["R0"] + list(adaylar.values())}
        prog(adim, asama="bootstrap", B=a.bs_B)
        d = bootstrap_delta(pk, list(adaylar.values()), a.bs_B, a.bs_isci)
        for ad, arr in d.items():
            bs[ad] = dict(alt90=float(np.percentile(arr, 10)), ort=float(arr.mean()), std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                          p_pozitif=float((arr > 0).mean()))
        print(f"[h07] bootstrap B={a.bs_B} ({time.time() - t:.0f} sn): " + "; ".join(f"{k}: alt90={v['alt90']*100:+.3f} ort={v['ort']*100:+.3f} puan" for k, v in bs.items()))
    for s in satirlar.values():
        b = bs.get(s["ad"])
        s["bs_alt90"] = b["alt90"] if b else float("nan")
        s["bs_ort"] = b["ort"] if b else float("nan")

    # ---- ASAMA 2: s1/s2 onayi (aday satirlar + R0) ----
    s12 = {}
    if adaylar and not a.atla_s12 and not r0_uyumsuz:
        for sd in ("s1", "s2"):
            if sd not in agir or not Path(agir[sd]).exists():
                print(f"[h07] UYARI: {sd} agirligi yok ({agir.get(sd)}); (c) degerlendirilemez")
                continue
            r0s = satir_kos(agir[sd], R0, data_yaml, device_arg, run_dir, a.batch, a.workers)
            v0 = float(r0s["stats"]["metrics/mAP50-95(B)"])
            s12[sd] = {"R0": v0, "R0_fark_metrics": v0 - bekle_val.get(sd, v0)}
            print(f"[h07] {sd} R0={v0:.6f} (baseline metrics.json {bekle_val.get(sd)}; fark {v0 - bekle_val.get(sd, v0):+.6f})")
            cfg_by = {c["ad"]: c for c in SATIRLAR}
            for ad in sorted(set(adaylar.values())):
                rs = satir_kos(agir[sd], cfg_by[ad], data_yaml, device_arg, run_dir, a.batch, a.workers)
                vs = float(rs["stats"]["metrics/mAP50-95(B)"])
                s12[sd][ad] = vs - v0
                adim += 1
                prog(adim, asama="s12", satir=ad, agirlik=sd, val_map50_95=vs)
                print(f"[h07] {sd} {ad}: Delta={1e2 * (vs - v0):+.3f} puan")
    for s in satirlar.values():
        d1 = s12.get("s1", {}).get(s["ad"])
        d2 = s12.get("s2", {}).get(s["ad"])
        s["d_s1"] = d1 if d1 is not None else float("nan")
        s["d_s2"] = d2 if d2 is not None else float("nan")

    # ---- gecikme ----
    lat = {}
    lat_tur = {}
    gpu_oncesi = None
    cv_r0 = float("nan")
    if not a.atla_gecikme:
        import cv2
        gu, gm = gpu_kullanim()
        gpu_oncesi = gu
        if cuda_ok and gu is not None and gu > 15:
            print(f"[h07] UYARI: gecikme olcumu oncesi GPU kullanimi %{gu} (>15): olcum paylasilan GPU'da, gecikme satirlari GUVENILMEZ (h07_gecikme_gpu_mesgul=1)")
        prog(adim, asama="gecikme")
        val_list = liste
        n_yuk = a.gecikme_isinma + a.gecikme_olcum
        imgs = [cv2.imread(f) for f in val_list[:n_yuk]]
        imgs = [i for i in imgs if i is not None]
        motor = Motor(w0, device, half=cuda_ok)
        cfgler = [R0] + SATIRLAR
        dep = dict(R0, ad="R0_dep")
        turlar = {c["ad"]: [] for c in cfgler}
        turlar["R0_dep"] = []
        tur = 0
        while True:
            for c in cfgler:
                ms, n = gecikme_olc(motor, c, imgs, a.gecikme_isinma, a.gecikme_olcum)
                turlar[c["ad"]].append(ms)
            ms, n = gecikme_olc(motor, dep, imgs, a.gecikme_isinma, a.gecikme_olcum, conf=0.25, multi_label=False)
            turlar["R0_dep"].append(ms)
            tur += 1
            r0t = turlar["R0"]
            cv_r0 = (statistics.pstdev(r0t) / statistics.mean(r0t)) if len(r0t) > 1 else 0.0
            print(f"[h07] gecikme tur {tur}: R0={r0t[-1]:.3f} ms, R0 CV={cv_r0 * 100:.2f}% | "
                  + " ".join(f"{k}={v[-1]:.1f}" for k, v in turlar.items() if k != 'R0'))
            prog(adim + tur, asama="gecikme", tur=tur)
            if tur >= a.gecikme_tekrar and (cv_r0 <= 0.03 or tur >= a.gecikme_tekrar_max):
                break
        lat_tur = turlar
        lat = {k: float(statistics.median(v)) for k, v in turlar.items()}
        gu1, gm1 = gpu_kullanim()
        print(f"[h07] gecikme bitti; tur={tur} R0 CV={cv_r0 * 100:.2f}% GPU kullanim son={gu1}%")
    for s in satirlar.values():
        s["lat_ms"] = lat.get(s["ad"], float("nan"))
        s["lat_oran"] = (s["lat_ms"] / lat["R0"]) if lat else float("nan")
        s["fps"] = 1000.0 / s["lat_ms"] if lat and s["lat_ms"] > 0 else float("nan")

    # ---- dogrulama kurali ----
    for s in satirlar.values():
        a_ok = s["d_B"] >= 0.0030
        b_ok = (not math.isnan(s["bs_alt90"])) and s["bs_alt90"] > 0
        d12 = [s["d_s1"], s["d_s2"]]
        c_ok = all(not math.isnan(x) for x in d12) and (sum(d12) / 2 >= 0.0020) and all(x > 0 for x in d12)
        s["dogr_a"], s["dogr_b"], s["dogr_c"] = bool(a_ok), bool(b_ok), bool(c_ok)
        s["dogrulandi"] = bool(s["aday"] and a_ok and b_ok and c_ok)
        s["kirmizi_bayrak"] = bool(s["d_full"] * 100 > KIRMIZI_PUAN[s["aile"]])
        oran_ok = (not math.isnan(s["lat_oran"])) and s["lat_oran"] <= 1.10
        if s["dogrulandi"] and s["kirmizi_bayrak"]:
            s["durum"] = "INCELE(kirmizi_bayrak)"
        elif s["dogrulandi"] and oran_ok:
            s["durum"] = "KAZANC-A"
        elif s["dogrulandi"]:
            s["durum"] = "KAZANC-B"
        elif s["aday"]:
            s["durum"] = "aday_dogrulanmadi"
        else:
            s["durum"] = "aday_degil"
        s["tablo_A"] = bool(oran_ok)
        s["tablo_B"] = True
        s["fps_ge30"] = bool((not math.isnan(s["lat_ms"])) and s["lat_ms"] <= 33.3)
    n_dogr = sum(1 for s in satirlar.values() if s["dogrulandi"])
    n_kaz_a = sum(1 for s in satirlar.values() if s["durum"] == "KAZANC-A")
    n_kaz_b = sum(1 for s in satirlar.values() if s["durum"] == "KAZANC-B")
    n_kirmizi = sum(1 for s in satirlar.values() if s["kirmizi_bayrak"])
    if r0_uyumsuz:
        oneri = "DUR(R0_uyumsuz)"
    elif n_kirmizi and n_dogr:
        oneri = "INCELE"
    elif n_kaz_a:
        oneri = "KAZANC-A"
    elif n_kaz_b:
        oneri = "KAZANC-B"
    elif n_dogr == 0:
        oneri = "OLDUR"
    else:
        oneri = "INCELE"

    # ---- CSV ----
    import csv
    kolonlar = ["satir", "aile", "aday", "aday_uygun", "map50_95", "map50", "ap_s", "ap_m", "ap_l", "delta_full_puan", "map_A", "map_B",
                "delta_A_puan", "delta_B_puan", "bs_alt90_puan", "bs_ort_puan", "delta_s1_puan", "delta_s2_puan",
                "dogr_a", "dogr_b", "dogr_c", "dogrulandi", "kirmizi_bayrak", "durum", "lat_ms", "lat_oran", "fps", "fps_ge30", "tablo_A",
                "n_pred_goruntu", "ayni_mi_R0"]
    csv_yol.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_yol, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(kolonlar)
        r0_lat = lat.get("R0", float("nan"))
        w.writerow(["R0", "R0", "", "", f"{r0_val:.6f}", f"{float(sonuc['R0']['stats']['metrics/mAP50(B)']):.6f}",
                    f"{float(sonuc['R0']['stats'].get('metrics/mAP_small(B)', float('nan'))):.6f}",
                    f"{float(sonuc['R0']['stats'].get('metrics/mAP_medium(B)', float('nan'))):.6f}",
                    f"{float(sonuc['R0']['stats'].get('metrics/mAP_large(B)', float('nan'))):.6f}", "0", f"{r0A:.6f}", f"{r0B:.6f}",
                    "0", "0", "", "", "", "", "", "", "", "", "", "referans", f"{r0_lat:.3f}", "1.000",
                    f"{1000 / r0_lat:.2f}" if lat else "", "", "", f"{R0p.n_pred / max(R0p.n, 1):.1f}", ""])
        for s in satirlar.values():
            f4 = lambda x, sc=1.0: "" if (isinstance(x, float) and math.isnan(x)) else f"{x * sc:.4f}"
            w.writerow([s["ad"], s["aile"], int(s["aday"]), int(s["aday_uygun"]), f"{s['map50_95']:.6f}", f"{s['map50']:.6f}",
                        f"{s['ap_s']:.6f}", f"{s['ap_m']:.6f}", f"{s['ap_l']:.6f}", f4(s["d_full"], 100), f"{s['map_A']:.6f}", f"{s['map_B']:.6f}",
                        f4(s["d_A"], 100), f4(s["d_B"], 100), f4(s["bs_alt90"], 100), f4(s["bs_ort"], 100), f4(s["d_s1"], 100), f4(s["d_s2"], 100),
                        int(s["dogr_a"]), int(s["dogr_b"]), int(s["dogr_c"]), int(s["dogrulandi"]), int(s["kirmizi_bayrak"]), s["durum"],
                        "" if math.isnan(s["lat_ms"]) else f"{s['lat_ms']:.3f}", "" if math.isnan(s["lat_oran"]) else f"{s['lat_oran']:.3f}",
                        "" if math.isnan(s["fps"]) else f"{s['fps']:.2f}", int(s["fps_ge30"]), int(s["tablo_A"]),
                        f"{s['n_pred_goruntu']:.1f}", int(s["ayni_mi"])])
    print(f"[h07] CSV: {csv_yol}")

    detay = {"zaman": simdi(), "kuru": kuru, "r0": {"val_map50_95": r0_val, "kendi": r0_kendi, "A": r0A, "B": r0B, "fark_metrics_json": r0_fark,
                                                  "tol": R0_TOL, "uyumsuz": bool(r0_uyumsuz)},
             "w_secilen": w_secilen, "adaylar": adaylar, "bootstrap": bs, "s12": s12, "gecikme_ms_medyan": lat,
             "gecikme_tur": lat_tur, "r0_gecikme_cv": cv_r0, "satirlar": satirlar, "oneri": oneri,
             "gpu_baslangic": [gpu_u0, gpu_m0]}
    atomik_yaz(run_dir / "h07_detay.json", detay)

    # ---- metrics.json (standart sema: val_ anahtarlari; kilitli veri anahtari YOK) ----
    alt_grup = {}
    for s in satirlar.values():
        ad = s["ad"]
        alt_grup[f"h07_{ad}_map50_95"] = round(s["map50_95"], 6)
        alt_grup[f"h07_{ad}_delta_full_puan"] = round(s["d_full"] * 100, 4)
        alt_grup[f"h07_{ad}_delta_A_puan"] = round(s["d_A"] * 100, 4)
        alt_grup[f"h07_{ad}_delta_B_puan"] = round(s["d_B"] * 100, 4)
        alt_grup[f"h07_{ad}_ap_s"] = round(s["ap_s"], 6) if not math.isnan(s["ap_s"]) else 0.0
        if not math.isnan(s["lat_ms"]):
            alt_grup[f"h07_{ad}_lat_ms"] = round(s["lat_ms"], 3)
            alt_grup[f"h07_{ad}_lat_oran"] = round(s["lat_oran"], 4)
        if not math.isnan(s["bs_alt90"]):
            alt_grup[f"h07_{ad}_bs_alt90_puan"] = round(s["bs_alt90"] * 100, 4)
        if not math.isnan(s["d_s1"]):
            alt_grup[f"h07_{ad}_delta_s1_puan"] = round(s["d_s1"] * 100, 4)
            alt_grup[f"h07_{ad}_delta_s2_puan"] = round(s["d_s2"] * 100, 4)
        alt_grup[f"h07_{ad}_dogrulandi"] = int(s["dogrulandi"])
        alt_grup[f"h07_{ad}_kirmizi_bayrak"] = int(s["kirmizi_bayrak"])
    r0st = sonuc["R0"]["stats"]
    n_p = None
    try:
        from ultralytics import YOLO
        n_p = int(sum(p.numel() for p in YOLO(str(w0)).model.parameters()))
    except Exception:
        pass
    metrics = {
        "val_map50_95": round(r0_val, 6), "val_map50": round(float(r0st["metrics/mAP50(B)"]), 6),
        "val_map_small": float(r0st.get("metrics/mAP_small(B)", 0.0)), "val_map_medium": float(r0st.get("metrics/mAP_medium(B)", 0.0)),
        "val_map_large": float(r0st.get("metrics/mAP_large(B)", 0.0)),
        "egitim_sure_sn": round(time.time() - t_basla, 2), "egitim_yok": 1,
        "param_sayisi": n_p, "cihaz": cihaz_str,
        "varyant_imzasi": "H07: egitimsiz son-islem/TTA satirlari (N/S/U/T/W, 12 satir + R0) baseline s0/s1/s2 best.pt uzerinde, yalniz val; her satir referans hattindan tek bilesenle farkli",
        "varyant_etki_sayaci": int(sum(0 if s["ayni_mi"] else 1 for s in satirlar.values())),   # R0'dan farkli cikan satir sayisi
        "h07_r0_fark": round(r0_fark, 6), "h07_r0_uyumsuz": int(r0_uyumsuz),
        "h07_satir_sayisi": len(satirlar), "h07_aday_sayisi": len(adaylar), "h07_dogrulanan_sayisi": int(n_dogr),
        "h07_kazanc_a_sayisi": int(n_kaz_a), "h07_kazanc_b_sayisi": int(n_kaz_b), "h07_kirmizi_bayrak_sayisi": int(n_kirmizi),
        "h07_bootstrap_B": int(a.bs_B), "h07_gecikme_tur": int(len(lat_tur.get("R0", []))), "h07_r0_gecikme_cv": round(cv_r0, 5) if not math.isnan(cv_r0) else 0.0,
        "h07_gpu_baslangic_kullanim": float(gpu_u0) if gpu_u0 is not None else -1.0,
        "h07_gecikme_oncesi_gpu_kullanim": float(gpu_oncesi) if gpu_oncesi is not None else -1.0,
        "h07_gecikme_gpu_mesgul": int(bool(cuda_ok and gpu_oncesi is not None and gpu_oncesi > 15)),
        "h07_oneri_kodu": {"DUR(R0_uyumsuz)": -1, "OLDUR": 0, "INCELE": 1, "KAZANC-B": 2, "KAZANC-A": 3}.get(oneri, 1),
        "alt_grup": alt_grup,
    }
    if lat:
        metrics.update({"latency_toplam_ms": round(lat["R0"], 4), "fps": round(1000.0 / lat["R0"], 3), "latency_R0_dep_ms": round(lat["R0_dep"], 4),
                        "hiz_batch": 1, "hiz_precision_fp16": 1.0 if cuda_ok else 0.0,
                        "hiz_olcum_n_goruntu": int(a.gecikme_olcum), "hiz_isinma_n_goruntu": int(a.gecikme_isinma)})
    assert not any(k.startswith(("test_", "rep_")) for k in metrics), "kilitli veri anahtari yazilamaz"
    atomik_yaz(out_path, metrics)
    print(f"[h07] metrics.json yazildi: {out_path} | oneri={oneri} dogrulanan={n_dogr} kazanc_A={n_kaz_a} kazanc_B={n_kaz_b} kirmizi={n_kirmizi} | toplam {time.time() - t_basla:.0f} sn")
    if r0_uyumsuz:
        print(f"[h07] HATA: R0 baseline metrics.json'dan {abs(r0_fark):.6f} > {R0_TOL} saptı: hat uyumsuz, KART DURUR (rc=94)")
        return 94
    return 0


if __name__ == "__main__":
    sys.exit(main())
