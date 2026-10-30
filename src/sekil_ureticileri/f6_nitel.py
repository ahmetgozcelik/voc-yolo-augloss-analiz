#!/usr/bin/env python
"""F6 nitel panel uretici: VOC07 test'ten kural-tabanli secilmis 6 goruntu x {GT, baseline s0, H03 s0, baseline+TTA s0}.

Girdi (WS = deney_workspace/2-1-...): runs/<id>/test_paket.npz (gorunti-duzeyi AP icin), runs/<id>/test_preds.json
(cizilecek tespitler, skor>=0.25), data/VOC/images/test2007/*.jpg, data/VOC/labels/test2007/*.txt, data/splits/test.txt.
Cikti: figures/tiles/f6/r<k>_{gt,base,h03,tta}.jpg (raster = yalniz piksel icerigi; etiketler TikZ'te),
       figures/data/f6_panel.dat (noktalı virgul ayracli; TikZ/pgfplotstable okur), figures/data/f6_secim_meta.json.

KARE SECIM OLCUTU (veriye bakmadan once yazildi; notes/figure-plan.md F6 ile ayni; caption'a girer):
  Aday havuzu: test goruntuleri, GT >= 2 nesne, uc yontem de >= 1 tespit (skor>=0.25) uretiyor (kapsama tabani),
               ve rakibin (baseline s0) goruntu-duzeyi AP50-95'i kendi ceyrekler-arasi araliginda [Q1,Q3]
               (rakip o karede normal davraniyor; bunlar saglandiktan sonra secim yapilir).
  Secim: (i) TTA-baseline gorunti AP farkinin en buyuk 2'si, (ii) en kucuk (en negatif) 1'i,
         (iii) kucuk nesne (sqrt(alan) < 32 px, orijinal px) iceren adaylar arasinda H03-baseline farkinin en buyuk 1'i
         ve en kucuk 1'i, (iv) kalan havuzdan numpy default_rng(0) ile rastgele 1.
Goruntu-duzeyi AP50-95: GT sinifi U (skor>=0.25 ile tahmin edilen sinif) kumesi uzerinde sinif-ortalamasi; her sinif icin
  tp bayraklari (npz, Ultralytics eslemesi) ile 101-nokta (Ultralytics compute_ap 'interp') AP, 10 IoU esigi ortalamasi.
Hicbir deger elle girilmez; hepsi npz/preds/etiketten hesaplanir.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
MW = Path(__file__).resolve().parents[2]
WS = MW.parents[1] / "deney_workspace" / "2-1-enhancing-real-time-object-detection-on-pascal-v"
RUNS = {"base": "20261001-212345_baseline-test_s0", "h03": "20261001-212720_h03_boxcp-test_s0",
        "tta": "20261001-213042_baseline-tta-test_s0"}
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
OP = 0.25
TILE_W, TILE_H = 480, 360
BGR = {"tp": (159, 84, 0), "fp": (43, 24, 178), "fn": (0, 140, 255), "gt_in": (255, 255, 255), "gt_out": (0, 0, 0)}


def ap101(tp_sorted, n_gt):
    if n_gt == 0:
        return 0.0
    tpc = np.cumsum(tp_sorted)
    fpc = np.cumsum(~tp_sorted)
    rec = tpc / n_gt
    prec = tpc / (tpc + fpc)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def image_ap(z):
    """z: npz -> goruntu basina AP50-95 dizisi (4952,) ve OP'de tespit sayisi."""
    ln = z["len"]
    tl = z["tlen"]
    po = np.concatenate(([0], np.cumsum(ln)))
    go = np.concatenate(([0], np.cumsum(tl)))
    out = np.zeros(len(ln))
    nop = np.zeros(len(ln), int)
    for i in range(len(ln)):
        p0, p1 = po[i], po[i + 1]
        conf = z["conf"][p0:p1]
        pcls = z["pcls"][p0:p1].astype(int)
        tp = z["tp"][p0:p1]
        gcls = z["tcls"][go[i]:go[i + 1]].astype(int)
        nop[i] = int((conf >= OP).sum())
        cl = set(gcls.tolist()) | set(pcls[conf >= OP].tolist())
        if not cl:
            out[i] = np.nan
            continue
        aps = []
        for c in cl:
            m = pcls == c
            ng = int((gcls == c).sum())
            if m.sum() == 0:
                aps.append(0.0)
                continue
            o = np.argsort(-conf[m], kind="stable")
            tpc = tp[m][o]
            aps.append(np.mean([ap101(tpc[:, t], ng) for t in range(10)]))
        out[i] = float(np.mean(aps))
    return out, nop


def load_gt(stem):
    p = WS / "data" / "VOC" / "images" / "test2007" / f"{stem}.jpg"
    W, H = Image.open(p).size
    boxes = []
    for line in open(WS / "data" / "VOC" / "labels" / "test2007" / f"{stem}.txt", encoding="utf-8"):
        t = line.split()
        if len(t) < 5:
            continue
        c = int(t[0])
        xc, yc, w, h = (float(v) for v in t[1:5])
        boxes.append((c, (xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H))
    return p, W, H, boxes


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def classify(dets, gts):
    """skor>=OP tespitler (cls,x1,y1,x2,y2,score) -> TP/FP listesi; kacirilan GT listesi (IoU>=0.5, sinif eslemeli, greedy skor sirasi)."""
    used = set()
    res = []
    for d in sorted(dets, key=lambda r: -r[5]):
        best, bj = 0.0, -1
        for j, g in enumerate(gts):
            if j in used or g[0] != d[0]:
                continue
            v = iou(d[1:5], g[1:5])
            if v > best:
                best, bj = v, j
        if best >= 0.5:
            used.add(bj)
            res.append((d, "tp"))
        else:
            res.append((d, "fp"))
    missed = [g for j, g in enumerate(gts) if j not in used]
    return res, missed


def draw(img, items, thick=2):
    for (x1, y1, x2, y2), col in items:
        cv2.rectangle(img, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), col, thick, cv2.LINE_AA)


def tile(path, W, H, items_fn):
    im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    s = min(TILE_W / W, TILE_H / H)
    nw, nh = int(round(W * s)), int(round(H * s))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((TILE_H, TILE_W, 3), 128, np.uint8)
    ox, oy = (TILE_W - nw) // 2, (TILE_H - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = im
    sub = canvas[oy:oy + nh, ox:ox + nw]
    items_fn(sub, s)
    return canvas


def main():
    z = {k: np.load(WS / "runs" / rid / "test_paket.npz", allow_pickle=True) for k, rid in RUNS.items()}
    names = [str(n) for n in z["base"]["names"]]
    assert all([str(n) for n in z[k]["names"]] == names for k in z)
    ap, nop, nops = {}, {}, {}
    for k in z:
        ap[k], nop[k] = image_ap(z[k])
    ngt = z["base"]["tlen"]
    valid = (ngt >= 1) & ~np.isnan(ap["base"])
    q1, q3 = np.percentile(ap["base"][valid], [25, 75])
    cand = (ngt >= 2) & (nop["base"] >= 1) & (nop["h03"] >= 1) & (nop["tta"] >= 1) & (ap["base"] >= q1) & (ap["base"] <= q3) & valid
    idx = np.where(cand)[0]
    print(f"havuz: {len(idx)} / {len(names)} goruntu; baseline goruntu-AP Q1={q1:.3f}, Q3={q3:.3f}")
    d_tta = ap["tta"] - ap["base"]
    d_h03 = ap["h03"] - ap["base"]
    # kucuk nesne bayragi (yalniz aday goruntuler)
    small = {}
    for i in idx:
        _, W, H, boxes = load_gt(Path(names[i]).stem)
        small[i] = any(((b[3] - b[1]) * (b[4] - b[2])) < 32 ** 2 for b in boxes)
    chosen = []

    def pick(order, label, pool=None):
        for i in order:
            if i in [c[0] for c in chosen]:
                continue
            if pool is not None and i not in pool:
                continue
            chosen.append((int(i), label))
            return

    key = lambda arr, sgn: sorted(idx, key=lambda i: (sgn * -arr[i], names[i]))   # sgn=+1 buyukten kucuge
    pick(key(d_tta, +1), "TTA kazancı 1")
    pick(key(d_tta, +1), "TTA kazancı 2")
    pick(key(d_tta, -1), "TTA kaybı")
    spool = {i for i in idx if small[i]}
    pick(key(d_h03, +1), "H03 kazancı (küçük)", spool)
    pick(key(d_h03, -1), "H03 kaybı (küçük)", spool)
    rng = np.random.default_rng(0)
    rest = [i for i in idx if i not in [c[0] for c in chosen]]
    pick([int(rng.choice(rest))], "Rastgele (tohum 0)")
    assert len(chosen) == 6
    out_tiles = MW / "figures" / "tiles" / "f6"
    out_tiles.mkdir(parents=True, exist_ok=True)
    preds = {k: json.load(open(WS / "runs" / rid / "test_preds.json", encoding="utf-8")) for k, rid in RUNS.items()}
    byimg = {k: {} for k in preds}
    for k, lst in preds.items():
        for r in lst:
            if r["score"] >= OP:
                byimg[k].setdefault(r["file_name"], []).append(
                    (int(r["category_id"]) - 1, r["bbox"][0], r["bbox"][1], r["bbox"][0] + r["bbox"][2], r["bbox"][1] + r["bbox"][3], r["score"]))
    rows = ["satir;kriter;goruntu;n_gt;n_kucuk;ap_base;ap_h03;ap_tta;tp_base;fp_base;fn_base;tp_h03;fp_h03;fn_h03;tp_tta;fp_tta;fn_tta;gt_siniflar"]
    for r, (i, label) in enumerate(chosen, start=1):
        stem = Path(names[i]).stem
        path, W, H, gts = load_gt(stem)
        cnt = {}
        for g in gts:
            cnt[CLASSES[g[0]]] = cnt.get(CLASSES[g[0]], 0) + 1
        gtl = ", ".join(f"{c} x{n}" if n > 1 else c for c, n in sorted(cnt.items()))
        nsm = sum(1 for b in gts if ((b[3] - b[1]) * (b[4] - b[2])) < 32 ** 2)

        def gt_fn(sub, s, gts=gts):
            for g in gts:
                bx = (g[1] * s, g[2] * s, g[3] * s, g[4] * s)
                draw(sub, [(bx, BGR["gt_out"])], 4)
                draw(sub, [(bx, BGR["gt_in"])], 2)
        cv2.imencode(".jpg", tile(path, W, H, gt_fn), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(out_tiles / f"r{r}_gt.jpg"))
        stat = {}
        for k in ("base", "h03", "tta"):
            res, missed = classify(byimg[k].get(names[i], []), gts)

            def det_fn(sub, s, res=res, missed=missed):
                draw(sub, [((g[1] * s, g[2] * s, g[3] * s, g[4] * s), BGR["fn"]) for g in missed], 2)
                draw(sub, [((d[1] * s, d[2] * s, d[3] * s, d[4] * s), BGR["fp"]) for d, t in res if t == "fp"], 2)
                draw(sub, [((d[1] * s, d[2] * s, d[3] * s, d[4] * s), BGR["tp"]) for d, t in res if t == "tp"], 2)
            cv2.imencode(".jpg", tile(path, W, H, det_fn), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(out_tiles / f"r{r}_{k}.jpg"))
            stat[k] = (sum(1 for _, t in res if t == "tp"), sum(1 for _, t in res if t == "fp"), len(missed))
        rows.append(";".join([str(r), label, stem, str(len(gts)), str(nsm), f"{ap['base'][i]:.3f}", f"{ap['h03'][i]:.3f}", f"{ap['tta'][i]:.3f}"] +
                             [str(v) for k in ("base", "h03", "tta") for v in stat[k]] + [gtl]))
        print(rows[-1])
    (MW / "figures" / "data").mkdir(parents=True, exist_ok=True)
    (MW / "figures" / "data" / "f6_panel.dat").write_text("\n".join(rows) + "\n", encoding="utf-8")
    meta = dict(n_goruntu=len(names), n_havuz=int(len(idx)), q1_baseline_goruntu_ap=float(q1), q3_baseline_goruntu_ap=float(q3),
                n_kucuk_iceren_aday=int(len(spool)), op_conf=OP, runs=RUNS,
                ortalama_goruntu_ap={k: float(np.nanmean(ap[k][valid])) for k in ap},
                ort_delta_tta_havuz=float(np.mean(d_tta[idx])), ort_delta_h03_havuz=float(np.mean(d_h03[idx])),
                ort_delta_tta_tum=float(np.nanmean((ap["tta"] - ap["base"])[valid])), ort_delta_h03_tum=float(np.nanmean((ap["h03"] - ap["base"])[valid])),
                kazanc_orani_tta_tum=float(np.mean((ap["tta"] - ap["base"])[valid] > 1e-9)), kayip_orani_tta_tum=float(np.mean((ap["tta"] - ap["base"])[valid] < -1e-9)),
                kazanc_orani_h03_tum=float(np.mean((ap["h03"] - ap["base"])[valid] > 1e-9)), kayip_orani_h03_tum=float(np.mean((ap["h03"] - ap["base"])[valid] < -1e-9)),
                secilenler=[dict(satir=r + 1, kriter=l, goruntu=names[i], d_tta=float(d_tta[i]), d_h03=float(d_h03[i])) for r, (i, l) in enumerate(chosen)])
    (MW / "figures" / "data" / "f6_secim_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print("tamam:", {k: meta[k] for k in ("n_havuz", "ort_delta_tta_tum", "ort_delta_h03_tum", "kazanc_orani_tta_tum", "kayip_orani_tta_tum")})


if __name__ == "__main__":
    main()
