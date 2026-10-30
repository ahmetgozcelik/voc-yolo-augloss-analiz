# -*- coding: utf-8 -*-
"""
H03: kutu-tabanli kucuk-nesne Copy-Paste (ayni goruntuye, cakismasiz). H03 kartinin "Degisen tek sey" tarifi:

  Mosaic'ten sonra, RandomPerspective'ten once (v8_transforms flip-modu konumu) calisir. p=0.5 olasilikla
  goruntude, sqrt(alan) <= 48 px (mosaic tuvalinde) ve baska hicbir GT kutusuyla IoA>0.1 ortusmeyen nesnelerden
  rastgele 1-3 tanesinin kutu-ici piksel yamasi, ayni goruntude mevcut hicbir kutuyla cakismayan (IoU=0,
  <=10 deneme) rastgele bir konuma, YENIDEN OLCEKLEMEDEN, 3 px kenar yumusatmasiyla yapistirilir; sinif etiketi
  ayni kalir. close_mosaic'te p=0 -> labels aynen doner.

Uygulama kararlari (kart sessiz oldugu noktalar; kos.py imzasina ve README'ye de yazilir):
  * Hedef konum mosaic tuvalinin MERKEZ penceresinde ([W/4, 3W/4] x [H/4, 3H/4], = imgsz x imgsz) secilir: sonraki
    RandomPerspective(border=-imgsz/2) tuvalin yalniz bu merkez penceresini egitim goruntusu olarak tutar; tuvalin
    tamamina yapistirma yapilsaydi yapistirmalarin ~%75'i kirpilip kaybolur (etkin tedavi ~4x seyreltilir).
    zone="full" ile kart-harfiyen tuval-geneli secilebilir (ablasyon).
  * Kenar yumusatma yamanin ICINDE uygulanir (alfa: kenardan 1,2,3 px icerde 0.25,0.5,0.75; >=4 px icerde 1);
    yama disina tasmaz -> komsu kutulara/piksellere dokunmaz.
  * Yama kaynagi kutunun floor/ceil tam-sayi dikdortgenidir; etikete KAYNAK kutunun gercek (kesirli) boyutu,
    tam-sayi oteleme ile tasinarak yazilir. Cakisma testi yama dikdortgeni uzerinden (etiket kutusunu kapsar).
  * Min kenar 2 px (dejenere yama olmasin).

Windows spawn: bu modul `ultralytics.data.augment.CopyPaste` ile DEGISTIRILIR; dataset ana surecte kurulur
(v8_transforms ad cozumlemesi ana surecte), BoxCopyPaste ornekleri dataset ile birlikte pickle'lanip
DataLoader isci sureclerine gider; isci unpickle'da `boxcp` modulunu sys.path'ten (spawn, ana surecin
sys.path'ini aktarir) ice aktarir. Sinif `__main__`'de TANIMLI DEGIL (pickle icin sart).

Sayac: BOXCP_SAYAC_DIR ortam degiskeni (kos.py ayarlar; isciler miras alir) tanimliysa her surec
<dir>/<pid>.json dosyasina kumulatif sayaclarini periyodik yazar (kos.py bitimde toplar).
"""
import json
import os
import random
from pathlib import Path

import numpy as np

from ultralytics.utils.instance import Instances
from ultralytics.utils.metrics import bbox_ioa

_SAYAC_ALANLARI = ("cagri", "uygulanan_goruntu", "yapistirilan", "aday_toplam", "yer_bulunamadi")


class BoxCopyPaste:
    """CopyPaste ile ayni yapici imzasi (dataset, pre_transform, p, mode); labels -> labels."""

    def __init__(self, dataset=None, pre_transform=None, p: float = 0.5, mode: str = "flip",
                 max_sqrt_alan: float = 48.0, n_min: int = 1, n_max: int = 3, ioa_esik: float = 0.1,
                 yumusatma: int = 3, deneme: int = 10, zone: str = "center", min_kenar: int = 2) -> None:
        if mode != "flip":
            raise ValueError(f"BoxCopyPaste yalniz mode='flip' konumunu destekler, verilen: {mode}")
        self.p = float(p)
        self.max_sqrt_alan = max_sqrt_alan
        self.n_min, self.n_max = n_min, n_max
        self.ioa_esik = ioa_esik
        self.yumusatma = yumusatma
        self.deneme = deneme
        self.zone = zone
        self.min_kenar = min_kenar
        self.son = []  # test/gorsellestirme: [(kaynak_indeks, yeni_kutu(xyxy), yama_dikdortgeni)]
        self._s = dict.fromkeys(_SAYAC_ALANLARI, 0)
        self._yaz_sayac = 0

    # ---- sayac (yalnizca ortam degiskeni varsa dosyaya) ----
    def _sayac_yaz(self, zorla=False):
        d = os.environ.get("BOXCP_SAYAC_DIR")
        if not d:
            return
        self._yaz_sayac += 1
        if zorla or self._yaz_sayac % int(os.environ.get("BOXCP_FLUSH_N", "50")) == 0:
            try:
                Path(d).mkdir(parents=True, exist_ok=True)
                yol = Path(d) / f"{os.getpid()}.json"
                tmp = yol.with_suffix(".tmp")
                tmp.write_text(json.dumps(self._s), encoding="utf-8")
                os.replace(tmp, yol)
            except OSError:
                pass  # sayac yazimi egitimi asla bozmaz

    def __call__(self, labels):
        if self.p == 0 or random.random() >= self.p:
            if self.p != 0:
                self._s["cagri"] += 1
                self._sayac_yaz()
            return labels
        self._s["cagri"] += 1
        self.son = []
        img = labels["img"]
        H, W = img.shape[:2]
        ins = labels["instances"]
        ins.convert_bbox(format="xyxy")
        ins.denormalize(W, H)  # mosaic ciktisinda zaten piksel/xyxy (no-op)
        b = np.asarray(ins.bboxes, dtype=np.float32)
        n = len(b)
        if n == 0:
            self._sayac_yaz()
            return labels
        w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        ioa = bbox_ioa(b, b)  # ioa[i, j] = kesisim(i, j) / alan(j)
        np.fill_diagonal(ioa, 0.0)
        aday_mask = (np.sqrt(np.maximum(w * h, 0)) <= self.max_sqrt_alan) & (w >= self.min_kenar) & (h >= self.min_kenar) \
            & (ioa.max(0) <= self.ioa_esik)
        aday = np.nonzero(aday_mask)[0]
        self._s["aday_toplam"] += int(len(aday))
        if len(aday) == 0:
            self._sayac_yaz()
            return labels
        k = min(random.randint(self.n_min, self.n_max), len(aday))
        secilen = random.sample([int(i) for i in aday], k)

        # yamalar, degistirilmemis goruntuden (yapistirmalar aday bolgelerine dusmez: IoU=0 sarti)
        yamalar = []
        for j in secilen:
            x1, y1 = int(np.floor(b[j, 0])), int(np.floor(b[j, 1]))
            x2, y2 = int(np.ceil(b[j, 2])), int(np.ceil(b[j, 3]))
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
            if x2 - x1 < self.min_kenar or y2 - y1 < self.min_kenar:
                continue
            yamalar.append((j, (x1, y1, x2, y2), img[y1:y2, x1:x2].copy()))

        out_img = img.copy()
        engeller = [tuple(r) for r in b.tolist()]  # (x1, y1, x2, y2) mevcut GT + yapistirilan yama dikdortgenleri
        yeni_kutular, yeni_cls = [], []
        for j, (sx1, sy1, sx2, sy2), yama in yamalar:
            ph, pw = yama.shape[:2]
            if self.zone == "center":
                lo_x, hi_x = W // 4, (W - W // 4) - pw
                lo_y, hi_y = H // 4, (H - H // 4) - ph
            else:
                lo_x, hi_x, lo_y, hi_y = 0, W - pw, 0, H - ph
            if hi_x < lo_x or hi_y < lo_y:
                self._s["yer_bulunamadi"] += 1
                continue
            yer = None
            for _ in range(self.deneme):
                tx, ty = random.randint(lo_x, hi_x), random.randint(lo_y, hi_y)
                r = (tx, ty, tx + pw, ty + ph)
                if all(min(r[2], e[2]) - max(r[0], e[0]) <= 0 or min(r[3], e[3]) - max(r[1], e[1]) <= 0
                       for e in engeller):
                    yer = (tx, ty)
                    break
            if yer is None:
                self._s["yer_bulunamadi"] += 1
                continue
            tx, ty = yer
            # kenar yumusatma (yama icinde)
            if self.yumusatma > 0:
                dx = np.minimum(np.arange(1, pw + 1), np.arange(pw, 0, -1))
                dy = np.minimum(np.arange(1, ph + 1), np.arange(ph, 0, -1))
                alfa = np.clip(np.minimum(dy[:, None], dx[None, :]) / (self.yumusatma + 1.0), 0.0, 1.0)[..., None]
                hedef = out_img[ty:ty + ph, tx:tx + pw].astype(np.float32)
                yeni = alfa * yama.astype(np.float32) + (1.0 - alfa) * hedef
                out_img[ty:ty + ph, tx:tx + pw] = np.clip(yeni + 0.5, 0, 255).astype(np.uint8)
            else:
                out_img[ty:ty + ph, tx:tx + pw] = yama
            ox, oy = tx - sx1, ty - sy1  # tam-sayi oteleme
            kutu = [b[j, 0] + ox, b[j, 1] + oy, b[j, 2] + ox, b[j, 3] + oy]
            yeni_kutular.append(kutu)
            yeni_cls.append(labels["cls"][j])
            engeller.append((tx, ty, tx + pw, ty + ph))
            self.son.append((j, kutu, (tx, ty, tx + pw, ty + ph)))

        if yeni_kutular:
            tum = np.concatenate([b, np.asarray(yeni_kutular, dtype=np.float32)], axis=0)
            labels["cls"] = np.concatenate([labels["cls"], np.asarray(yeni_cls).reshape(len(yeni_cls), -1)], axis=0)
            labels["instances"] = Instances(tum, ins.segments, ins.keypoints, bbox_format="xyxy", normalized=False)
            labels["img"] = out_img
            self._s["uygulanan_goruntu"] += 1
            self._s["yapistirilan"] += len(yeni_kutular)
        self._sayac_yaz()
        return labels

    def __repr__(self):
        return (f"BoxCopyPaste(p={self.p}, max_sqrt_alan={self.max_sqrt_alan}, n={self.n_min}-{self.n_max}, "
                f"ioa>{self.ioa_esik} ele, yumusatma={self.yumusatma}, deneme={self.deneme}, zone={self.zone})")


def sayac_topla(dizin):
    """BOXCP_SAYAC_DIR altindaki <pid>.json dosyalarini toplar."""
    top = dict.fromkeys(_SAYAC_ALANLARI, 0)
    n_surec = 0
    for f in Path(dizin).glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        n_surec += 1
        for k in _SAYAC_ALANLARI:
            top[k] += int(d.get(k, 0))
    top["surec_sayisi"] = n_surec
    return top


def uygula():
    """v8_transforms'un ad olarak cozdugu yerlesik CopyPaste'i degistirir (dataset kurulmadan ONCE)."""
    import ultralytics.data.augment as augment
    augment.CopyPaste = BoxCopyPaste
    return augment.CopyPaste
