"""KITTI değerlendirmesi: DontCare kuralıyla COCO tarzı mAP (ön-kayıt: ON-KAYIT-ek-deney.md, Veri bölümü).

Kural (hakem tur 2 düzeltmesi): DontCare bölgeleri COCOeval'e her sınıf için `iscrowd=1` (yok-say) yer gerçeği olarak
eklenir. COCOeval tespitleri ÖNCE gerçek nesnelerle eşleştirir; yalnız EŞLEŞMEMİŞ bir tespit bir DontCare bölgesiyle
IoA (kesişim / tespit alanı) >= o anki IoU eşiği ise yok sayılır (ne TP ne FP). Bu, KITTI resmî değerlendirmesinin
"önce eşleştir, eşleşmeyeni DontCare'de yok say" mantığının COCO eşik taramasına uyarlamasıdır (resmî sabit eşik
yerine 0.50:0.95 taraması). Yer gerçeği nesneleri = YOLO etiketleri.
Metrik: pycocotools COCOeval, maxDets=300 (Ultralytics max_det ile aynı), AP@[.5:.95], AP50, AP_S/M/L.

Kullanım:
  python scripts/ek_deney/kitti_degerlendir.py --preds RUN/val_preds.json --liste data/kitti/splits/val.txt \
         --dontcare data/kitti/dontcare.json --out RUN/kitti_val.json
  --dontcare verilmezse kural uygulanmaz (yalnız Ultralytics metriğiyle tutarlılık sınaması için).
Tahmin dosyası Ultralytics save_json çıktısıdır (category_id = sınıf + 1, bbox = xywh piksel).
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def gt_kur(liste, nc):
    images, anns, kimlik = [], [], {}
    for k, p in enumerate(liste):
        p = Path(p)
        w, h = Image.open(p).size
        iid = k + 1
        kimlik[p.stem] = iid
        images.append({"id": iid, "file_name": p.name, "width": w, "height": h})
        lp = Path(str(p).replace("images", "labels")).with_suffix(".txt")
        if lp.exists():
            for l in lp.read_text().splitlines():
                if not l.strip():
                    continue
                c, x, y, bw, bh = l.split()[:5]
                bw, bh = float(bw) * w, float(bh) * h
                x0, y0 = float(x) * w - bw / 2, float(y) * h - bh / 2
                anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": int(c) + 1,
                             "bbox": [x0, y0, bw, bh], "area": bw * bh, "iscrowd": 0})
    return {"images": images, "annotations": anns, "categories": [{"id": k + 1} for k in range(nc)]}, kimlik


def degerlendir(preds, liste, dontcare=None, nc=8):
    gt, kimlik = gt_kur(liste, nc)
    boyut = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}
    ters = {v: k for k, v in kimlik.items()}
    # eşleştirme dosya adı gövdesiyle (Ultralytics image_id sayısal olmayan adlarda metindir)
    preds = [dict(d, image_id=kimlik[Path(d["file_name"]).stem]) for d in preds if Path(d["file_name"]).stem in kimlik]
    n_dc = 0
    if dontcare is not None:
        for iid, (w, h) in boyut.items():
            for x0, y0, x1, y1 in dontcare.get(ters[iid], []):
                bw, bh = (x1 - x0) * w, (y1 - y0) * h
                if bw <= 0 or bh <= 0:
                    continue
                n_dc += 1
                for k in range(nc):  # DontCare sınıf-bağımsız → her sınıf için yok-say bölgesi
                    gt["annotations"].append({"id": len(gt["annotations"]) + 1, "image_id": iid, "category_id": k + 1,
                                              "bbox": [x0 * w, y0 * h, bw, bh], "area": bw * bh, "iscrowd": 1})
    with contextlib.redirect_stdout(io.StringIO()):
        C = COCO()
        C.dataset = gt
        C.createIndex()
        R = C.loadRes([{k: d[k] for k in ("image_id", "category_id", "bbox", "score")} for d in preds]) if preds else None
        E = COCOeval(C, R, "bbox")
        E.params.maxDets = [1, 10, 300]
        E.evaluate()
        E.accumulate()
        E.summarize()
    # IoU=0.50, alan=all, maxDet=300'de DontCare nedeniyle yok sayılan (eşleşmemiş) tespit sayısı
    yok = 0
    if dontcare is not None:
        for e in E.evalImgs:
            if e is not None and e["aRng"] == E.params.areaRng[0] and e["maxDet"] == 300:
                yok += int(np.asarray(e["dtIgnore"])[0].sum())
    P = E.eval["precision"]  # [T, R, K, A, M]; A: all/small/medium/large; M: maxDets indeksi 2 = 300

    def ap(t=None, a=0):
        x = P[:, :, :, a, 2] if t is None else P[t, :, :, a, 2]
        x = x[x > -1]
        return round(float(x.mean()), 6) if x.size else float("nan")
    def ap_k(ks):  # ON-KAYIT v5b Ö3: sınıf alt kümesi mAP50-95 (ikincil)
        if max(ks) >= P.shape[2]:  # KITTI 8 sınıfı dışı (birim testleri)
            return float("nan")
        x = P[:, :, ks, 0, 2]
        x = x[x > -1]
        return round(float(x.mean()), 6) if x.size else float("nan")

    return {"map50_95": ap(), "map50": ap(0), "map75": ap(5),
            "map50_95_3sinif": ap_k([0, 3, 5]), "map50_95_7sinif": ap_k([0, 1, 2, 3, 5, 6, 7]),
            "ap50_95_sinif": [ap_k([k]) for k in range(nc)],
            "map_small": ap(None, 1), "map_medium": ap(None, 2), "map_large": ap(None, 3),
            "n_goruntu": len(gt["images"]), "n_gt": sum(a["iscrowd"] == 0 for a in gt["annotations"]),
            "n_tahmin": len(preds), "n_dontcare_bolge": n_dc, "dontcare_yok_sayilan_iou50": yok,
            "dontcare_kurali": "coco-crowd" if dontcare is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--liste", required=True)
    ap.add_argument("--dontcare", default=None)
    ap.add_argument("--nc", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    liste = [l.strip() for l in Path(a.liste).read_text(encoding="utf-8").splitlines() if l.strip()]
    preds = json.loads(Path(a.preds).read_text())
    dc = json.loads(Path(a.dontcare).read_text()) if a.dontcare else None
    r = degerlendir(preds, liste, dc, a.nc)
    print(json.dumps(r, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
