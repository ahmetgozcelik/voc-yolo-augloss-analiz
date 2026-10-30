#!/usr/bin/env python
"""K3.5 hata analizi (YALNIZ VAL; test dosyalarina hic dokunmaz).

Kullanim (WS kokunden):
  env/.venv/Scripts/python.exe scripts/analiz/hata_analizi.py \
      [--run-dir runs/<id>] [--out-md ...] [--out-csv ...] [--op-conf 0.25] [--seed-glob 'runs/*_baseline_s*']

Girdi : <run-dir>/val_preds.json (COCO-json, conf=0.001, max_det=300), metrics.json (yalniz val
        anahtarlari; 'rep_*' anahtarlari yuklenir yuklenmez atilir), train/results.csv;
        data/splits/val.txt + YOLO etiketleri, data/voc_dev.yaml, data/sinif_dagilimi.csv.
Cikti : WS/sonuclar/hata-analizi.md, hata-kirilim.csv (varsayilan kosum icin); baska kosumda
        dosya adlarina _<run_id> eklenir. Ayrica <out-md>.json (seed-arasi kiyas icin ozet sayilar).
GPU kullanmaz, model calistirmaz.

Yontem: eslestirme pycocotools COCOeval (maxDets=300, IoU 0.50:0.05:0.95, alan araliklari COCO
S<32^2<=M<96^2<=L, alan = kutu alani, goruntuler ORIJINAL cozunurlukte). Hata turu ayristirmasi (TIDE
benzeri, IoU=0.5 esleşmesine göre; 'same-class-once' oncelik):
  TP   : COCOeval'de IoU>=0.5 ile eslesti
  Dupe : eslesmedi, ayni sinif GT ile IoU>=0.5 (GT'yi daha yuksek puanli kutu almis)
  Cls  : eslesmedi, baska sinif GT ile IoU>=0.5 (ayni sinif GT ile >=0.5 yok)
  Loc  : ayni sinif GT ile 0.1<=IoU<0.5
  Both : baska sinif GT ile 0.1<=IoU<0.5 (ayni sinif GT ile >=0.1 yok)
  Bkg  : hicbir GT ile IoU>=0.1 yok
  Miss : hicbir kutu (herhangi skor) tarafindan IoU>=0.5 ile eslesmeyen GT
ΔAP: TIDE tarzi oracle; hata duzeltilip degerlendirme yeniden kosulur (Cls: sinif dogrulanir,
Loc: kutu GT'ye yapistirilir, Both/Dupe/Bkg: kutu silinir, Miss: GT silinir).
"""
import argparse, contextlib, glob, io, json, math, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WS = Path(__file__).resolve().parents[2]
DEFAULT_RUN = WS / "runs" / "20260930-072103_baseline_s0"
TYPES = ["TP", "Dupe", "Cls", "Loc", "Both", "Bkg"]
S_MAX, M_MAX = 32 ** 2, 96 ** 2


# ----------------------------------------------------------------------------- yardimcilar
def md_table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def fm(x, d=3):
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return "–"
    return f"{x:.{d}f}"


def pct(x, d=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}%"


def size_bin(area):
    return "S" if area < S_MAX else ("M" if area < M_MAX else "L")


def load_metrics_val_only(path):
    """metrics.json'u yukler, TEST anahtarlarini ('rep_*') hemen atar."""
    m = json.load(open(path, encoding="utf-8"))
    m = {k: v for k, v in m.items() if not k.startswith("rep_")}
    if isinstance(m.get("alt_grup"), dict):
        m["alt_grup"] = {k: v for k, v in m["alt_grup"].items() if not k.startswith("rep_")}
    return m


def quiet_eval(cg, dets, maxdet=300):
    dets = [dict(d) for d in dets]
    with contextlib.redirect_stdout(io.StringIO()):
        if not dets:
            return None
        cd = cg.loadRes(dets)
        E = COCOeval(cg, cd, "bbox")
        E.params.maxDets = [1, 10, maxdet]
        E.evaluate()
        E.accumulate()
    return E


def make_gt(images, anns, cats):
    cg = COCO()
    cg.dataset = {"images": images, "annotations": [dict(a) for a in anns], "categories": cats}
    with contextlib.redirect_stdout(io.StringIO()):
        cg.createIndex()
    return cg


def ap_arr(E, t=None, a=0):
    """per-class AP (K,) ; t=None -> IoU ortalamasi; t=IoU indeksi; a=alan indeksi. Bos sinif=nan."""
    p = E.eval["precision"][:, :, :, a, 2]
    if t is not None:
        p = p[t:t + 1]
    out = np.full(p.shape[2], np.nan)
    for k in range(p.shape[2]):
        v = p[:, :, k]
        v = v[v > -1]
        if v.size:
            out[k] = v.mean()
    return out


def recall_arr(E, t=0, a=0):
    r = E.eval["recall"][t, :, a, 2]
    return np.where(r > -1, r, np.nan)


def iou_mat(a, b):
    """a (N,4) xyxy, b (M,4) xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def slope_per10(y):
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0] * 10)


# ----------------------------------------------------------------------------- veri yukleme
def load_gt(list_path=None, yaml_path=None):
    """Varsayilan: val (voc_dev.yaml + splits/val.txt). --etiket-listesi/--data-yaml ile baska bolme (ornek: test)."""
    names_d = yaml.safe_load(open(yaml_path or (WS / "data" / "voc_dev.yaml"), encoding="utf-8"))["names"]
    names = [names_d[i] for i in range(len(names_d))]
    files = [l.strip() for l in open(list_path or (WS / "data" / "splits" / "val.txt"), encoding="utf-8") if l.strip()]
    images, anns, stem2id = [], [], {}
    gid = 0
    for i, p in enumerate(files):
        stem = Path(p).stem
        W, H = Image.open(p).size
        images.append({"id": i + 1, "file_name": stem + ".jpg", "width": W, "height": H})
        stem2id[stem] = i + 1
        lab = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
        if lab.exists():
            for line in open(lab, encoding="utf-8"):
                t = line.split()
                if len(t) < 5:
                    continue
                c = int(t[0]); xc, yc, w, h = (float(v) for v in t[1:5])
                x1 = (xc - w / 2) * W; y1 = (yc - h / 2) * H; bw = w * W; bh = h * H
                gid += 1
                anns.append({"id": gid, "image_id": i + 1, "category_id": c + 1, "bbox": [x1, y1, bw, bh],
                             "area": bw * bh, "iscrowd": 0})
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    return names, images, anns, cats, stem2id


def load_dets(run_dir, stem2id, preds_path=None):
    raw = json.load(open(preds_path or (run_dir / "val_preds.json"), encoding="utf-8"))
    # Ultralytics COCO-json: sinif kimligi 1-tabanli (class_map=range(1,nc+1)); 0 gorulurse 0-tabanli say
    off = 1 if min(int(r["category_id"]) for r in raw) == 0 else 0
    dets = []
    for r in raw:
        stem = Path(r["file_name"]).stem
        dets.append({"image_id": stem2id[stem], "category_id": int(r["category_id"]) + off,
                     "bbox": [float(v) for v in r["bbox"]], "score": float(r["score"])})
    return dets


# ----------------------------------------------------------------------------- cekirdek analiz
def core(names, images, anns, cats, dets, op_conf):
    K = len(names)
    nG, nD = len(anns), len(dets)
    cg = make_gt(images, anns, cats)
    E = quiet_eval(cg, dets)
    R = {"E": E, "cg": cg}
    # ---- esleme tablolari
    A = len(E.params.areaRng); I = len(E.params.imgIds)
    mt = np.zeros((nD, 10), bool); dt_gt50 = np.zeros(nD, int)
    gmt = np.zeros((nG, 10), bool); gt_dt50 = np.zeros(nG, int)
    for k in range(K):
        for i in range(I):
            ev = E.evalImgs[k * A * I + 0 * I + i]
            if ev is None:
                continue
            for d, did in enumerate(ev["dtIds"]):
                mt[did - 1] = ev["dtMatches"][:, d] > 0
                dt_gt50[did - 1] = int(ev["dtMatches"][0, d])
            for g, gid_ in enumerate(ev["gtIds"]):
                gmt[gid_ - 1] = ev["gtMatches"][:, g] > 0
                gt_dt50[gid_ - 1] = int(ev["gtMatches"][0, g])
    # ---- dizilar
    db = np.array([d["bbox"] for d in dets]); dxy = db.copy(); dxy[:, 2:] += dxy[:, :2]
    dimg = np.array([d["image_id"] for d in dets]); dcls = np.array([d["category_id"] for d in dets]) - 1
    dsc = np.array([d["score"] for d in dets]); darea = db[:, 2] * db[:, 3]
    gb = np.array([a["bbox"] for a in anns]); gxy = gb.copy(); gxy[:, 2:] += gxy[:, :2]
    gimg = np.array([a["image_id"] for a in anns]); gcls = np.array([a["category_id"] for a in anns]) - 1
    garea = gb[:, 2] * gb[:, 3]
    d_by = {}; g_by = {}
    for j, im in enumerate(dimg): d_by.setdefault(im, []).append(j)
    for j, im in enumerate(gimg): g_by.setdefault(im, []).append(j)
    iou_same = np.zeros(nD); iou_oth = np.zeros(nD); oth_cls = -np.ones(nD, int); same_gt = -np.ones(nD, int)
    oth_gt = -np.ones(nD, int); iou_match = np.zeros(nD)
    g_same_det = np.zeros(nG); g_oth_det = np.zeros(nG)          # GT'ye en yuksek IoU (kutu, ayni/baska sinif)
    g_nb_same = np.zeros(nG); g_nb_any = np.zeros(nG)             # baska GT komsusuyla max IoU
    g_nb_same_gt = -np.ones(nG, int)
    g_dup_cover = np.zeros(nG, bool)                              # eslesmeyen GT'ye IoU>=.5 ayni sinif baska-GT'ye eslesmis kutu
    for im in range(1, len(images) + 1):
        dj = np.array(d_by.get(im, []), int); gj = np.array(g_by.get(im, []), int)
        if len(gj) == 0:
            continue
        if len(gj) > 1:
            gg = iou_mat(gxy[gj], gxy[gj]); np.fill_diagonal(gg, 0)
            same = gcls[gj][:, None] == gcls[gj][None, :]
            g_nb_any[gj] = gg.max(1)
            gs = np.where(same, gg, 0); g_nb_same[gj] = gs.max(1); g_nb_same_gt[gj] = gj[gs.argmax(1)]
        if len(dj) == 0:
            continue
        M = iou_mat(dxy[dj], gxy[gj])  # (nd, ng)
        same = dcls[dj][:, None] == gcls[gj][None, :]
        Ms = np.where(same, M, 0); Mo = np.where(~same, M, 0)
        iou_same[dj] = Ms.max(1); same_gt[dj] = gj[Ms.argmax(1)]
        iou_oth[dj] = Mo.max(1); oth_gt[dj] = gj[Mo.argmax(1)]; oth_cls[dj] = gcls[gj][Mo.argmax(1)]
        g_same_det[gj] = Ms.max(0); g_oth_det[gj] = Mo.max(0)
        for r, dd in enumerate(dj):
            if dt_gt50[dd] > 0:
                iou_match[dd] = M[r, dt_gt50[dd] - 1 - gj[0]]
        Mhit = (Ms >= 0.5)
        for c, g in enumerate(gj):
            if gt_dt50[g] == 0 and Mhit[:, c].any():
                g_dup_cover[g] = True
    # ---- tur atama
    typ = np.empty(nD, object)
    tp = mt[:, 0]
    for j in range(nD):
        if tp[j]: typ[j] = "TP"
        elif iou_same[j] >= 0.5: typ[j] = "Dupe"
        elif iou_oth[j] >= 0.5: typ[j] = "Cls"
        elif iou_same[j] >= 0.1: typ[j] = "Loc"
        elif iou_oth[j] >= 0.1: typ[j] = "Both"
        else: typ[j] = "Bkg"
    missed = gt_dt50 == 0
    gscore = np.zeros(nG)
    for g in range(nG):
        if gt_dt50[g] > 0: gscore[g] = dsc[gt_dt50[g] - 1]
    R.update(dict(mt=mt, gmt=gmt, dt_gt50=dt_gt50, gt_dt50=gt_dt50, dxy=dxy, dimg=dimg, dcls=dcls, dsc=dsc,
                  darea=darea, gxy=gxy, gimg=gimg, gcls=gcls, garea=garea, iou_same=iou_same, iou_oth=iou_oth,
                  oth_cls=oth_cls, same_gt=same_gt, oth_gt=oth_gt, iou_match=iou_match, typ=typ, missed=missed,
                  gscore=gscore, g_same_det=g_same_det, g_oth_det=g_oth_det, g_nb_same=g_nb_same, g_nb_any=g_nb_any,
                  g_dup_cover=g_dup_cover, g_nb_same_gt=g_nb_same_gt, nG=nG, nD=nD, K=K))
    return R


def oracles(names, images, anns, cats, dets, R):
    """TIDE tarzi oracle ΔAP (AP50 ve AP50-95, sinif ortalamasi + sinif basina AP50)."""
    typ = R["typ"]; out = {}
    base = R["E"]

    def summ(E):
        a50 = ap_arr(E, 0); a95 = ap_arr(E, None)
        return np.nanmean(a50), np.nanmean(a95), a50

    out["base"] = summ(base)
    # Cls
    d2 = []
    for j, d in enumerate(dets):
        d = dict(d)
        if typ[j] == "Cls": d["category_id"] = int(R["oth_cls"][j]) + 1
        d2.append(d)
    out["Cls"] = summ(quiet_eval(R["cg"], d2))
    # Loc
    d2 = []
    for j, d in enumerate(dets):
        d = dict(d)
        if typ[j] == "Loc":
            b = anns[R["same_gt"][j]]["bbox"]; d["bbox"] = list(b)
        d2.append(d)
    out["Loc"] = summ(quiet_eval(R["cg"], d2))
    for t in ("Both", "Dupe", "Bkg"):
        out[t] = summ(quiet_eval(R["cg"], [d for j, d in enumerate(dets) if typ[j] != t]))
    # Miss: GT sil
    keep = [a for g, a in enumerate(anns) if not R["missed"][g]]
    out["Miss"] = summ(quiet_eval(make_gt(images, keep, cats), dets))
    # tum FP kaldir (TP disinda her sey)
    out["FP_hepsi"] = summ(quiet_eval(R["cg"], [d for j, d in enumerate(dets) if typ[j] == "TP"]))
    # FP + Cls/Loc duzelt + Miss sil ~ 1.0; yazilmaz.
    # TP kutularini GT'ye yapistir (kutu kusursuz, AP50-95 -> AP50 sinirina gider)
    d2 = []
    for j, d in enumerate(dets):
        d = dict(d)
        if typ[j] == "TP": d["bbox"] = list(anns[R["dt_gt50"][j] - 1]["bbox"])
        d2.append(d)
    out["Kutu_kusursuz"] = summ(quiet_eval(R["cg"], d2))
    # IoU-farkindali siralama: TP puanlarini sinif basina IoU sirasina gore yeniden dagit
    d2 = [dict(d) for d in dets]
    for k in range(R["K"]):
        idx = np.where((R["dcls"] == k) & (typ == "TP"))[0]
        if len(idx) < 2: continue
        sc = np.sort(R["dsc"][idx])[::-1]; order = idx[np.argsort(-R["iou_match"][idx], kind="stable")]
        for s, j in zip(sc, order): d2[j]["score"] = float(s)
    out["TP_IoU_sirali"] = summ(quiet_eval(R["cg"], d2))
    return out



def load_difficult(images, names):
    """Val goruntulerinin XML'inden difficult==1 nesneler (etiketlerde yok). Kutu: 1-tabanli -> 0-tabanli (xmin-1, xmax-1)."""
    import xml.etree.ElementTree as ET
    root = WS / "data" / "VOC" / "images" / "VOCdevkit"
    out = []
    for im in images:
        stem = Path(im["file_name"]).stem
        for yr in ("VOC2007", "VOC2012"):
            xp = root / yr / "Annotations" / (stem + ".xml")
            if xp.exists():
                for o in ET.parse(xp).getroot().findall("object"):
                    if int(o.find("difficult").text) == 1 and o.find("name").text in names:
                        b = o.find("bndbox")
                        out.append((im["id"], names.index(o.find("name").text), float(b.find("xmin").text) - 1, float(b.find("ymin").text) - 1,
                                    float(b.find("xmax").text) - 1, float(b.find("ymax").text) - 1))
                break
    return out

# ----------------------------------------------------------------------------- bolumler
def build_report(run_dir, args):
    names, images, anns, cats, stem2id = load_gt()
    dets = load_dets(run_dir, stem2id)
    K = len(names); nG = len(anns)
    metrics = load_metrics_val_only(run_dir / "metrics.json")
    R = core(names, images, anns, cats, dets, args.op_conf)
    E = R["E"]; typ = R["typ"]; dsc = R["dsc"]
    ORA = oracles(names, images, anns, cats, dets, R)
    summ = {}
    L = []  # markdown satirlari
    op = args.op_conf
    L.append(f"# Hata analizi (K3.5) — baseline `{run_dir.name}`\n")
    L.append(f"*Üretici: `scripts/analiz/hata_analizi.py` · yalnız VAL (1000 görüntü, {nG} nesne; difficult nesneler etiketlerde yok) · "
             f"tahmin sayısı {len(dets)} (conf=0.001, max_det=300, NMS IoU=0.7) · eşleştirme: pycocotools COCOeval · "
             f"işletim noktası skor≥{op}.*\n")
    L.append("**Sızıntı notu:** Bu analizde test kümesi (VOC07 test), `rep_test*` dosyaları ve `metrics.json` içindeki `rep_*` anahtarları "
             "hiç kullanılmadı; betik bu anahtarları yükler yüklemez atar. Buradan çıkan hipotezler yalnız val bilgisine dayanır.\n")

    # ---- 0 genel doğrulama
    a50 = ap_arr(E, 0); a75 = ap_arr(E, 5); a95 = ap_arr(E, None)
    m50, m75, m95 = np.nanmean(a50), np.nanmean(a75), np.nanmean(a95)
    summ.update(map50=m50, map75=m75, map5095=m95)
    L.append("## 0. Genel ölçüler ve doğrulama\n")
    rows = [["mAP50-95", fm(m95, 4), fm(metrics.get("val_map50_95"), 4), fm(m95 - metrics.get("val_map50_95", np.nan), 4)],
            ["mAP50", fm(m50, 4), fm(metrics.get("val_map50"), 4), fm(m50 - metrics.get("val_map50", np.nan), 4)],
            ["mAP75 (yalnız pycocotools)", fm(m75, 4), "–", "–"]]
    L.append(md_table(["ölçü", "pycocotools (bu betik)", "Ultralytics metrics.json (best.pt val)", "fark"], rows))
    L.append(f"\nAP50 ile AP50-95 arasındaki fark {100 * (m50 - m95):.1f} puan; AP50→AP75 farkı {100 * (m50 - m75):.1f} puan. "
             "pycocotools ile Ultralytics AP'si arasındaki küçük fark interpolasyon/eşleştirme (Ultralytics kendi IoU-matrisi eşleşmesini kullanır) kaynaklıdır.\n")

    # ---- 1 sınıf bazlı
    dist = pd.read_csv(WS / "data" / "sinif_dagilimi.csv")
    tr = dist[dist["split"] == "train"].set_index("class")
    va = dist[dist["split"] == "val"].set_index("class")
    cls_df = pd.DataFrame({"sinif": names, "AP50_95": a95, "AP50": a50, "AP75": a75,
                           "train_nesne": [int(tr.loc[n, "objects"]) for n in names],
                           "val_nesne": [int(va.loc[n, "objects"]) for n in names],
                           "train_kucuk_oran": [tr.loc[n, "small_lt32sq"] / tr.loc[n, "objects"] for n in names]})
    cls_df["AP50_AP75"] = cls_df["AP50"] - cls_df["AP75"]
    srt = cls_df.sort_values("AP50_95")
    rho95 = spearmanr(cls_df["train_nesne"], cls_df["AP50_95"]); rho50 = spearmanr(cls_df["train_nesne"], cls_df["AP50"])
    rng = np.random.default_rng(0)
    perm = np.array([spearmanr(rng.permutation(cls_df["train_nesne"].values), cls_df["AP50_95"].values)[0] for _ in range(5000)])
    p_perm = float((np.abs(perm) >= abs(rho95[0])).mean())
    rho_small = spearmanr(cls_df["train_kucuk_oran"], cls_df["AP50_95"])
    mean_area = [float(np.mean([math.sqrt(R["garea"][g]) for g in range(nG) if R["gcls"][g] == k])) for k in range(K)]
    cls_df["gt_medyan_kenar_px"] = [float(np.median([math.sqrt(R["garea"][g]) for g in range(nG) if R["gcls"][g] == k])) for k in range(K)]
    srt = cls_df.sort_values("AP50_95")
    rho_size = spearmanr(cls_df["gt_medyan_kenar_px"], cls_df["AP50_95"])
    summ.update(rho_freq_ap5095=float(rho95[0]), rho_freq_ap50=float(rho50[0]), weakest5=list(srt["sinif"][:5]))
    L.append("## 1. Sınıf başına AP ve sınıf frekansı\n")
    rows = [[r.sinif, fm(r.AP50_95, 3), fm(r.AP50, 3), fm(r.AP75, 3), fm(r.AP50_AP75, 3), r.train_nesne, r.val_nesne,
             fm(r.gt_medyan_kenar_px, 0)] for r in srt.itertuples()]
    L.append(md_table(["sınıf (AP50-95 artan)", "AP50-95", "AP50", "AP75", "AP50−AP75", "train nesne", "val nesne", "val GT medyan kenar px"], rows))
    weak = srt.head(5)
    L.append(f"\nEn zayıf 5 sınıf (AP50-95): " + ", ".join(f"{r.sinif} {r.AP50_95:.3f}" for r in weak.itertuples()) +
             f". Sınıf frekansı (train nesne) ile AP50-95 Spearman ρ={rho95[0]:.2f} (permütasyon p={p_perm:.3f}), AP50 ile ρ={rho50[0]:.2f}; "
             f"sınıf başına train küçük-nesne oranı ile ρ={rho_small[0]:.2f}; val GT medyan kenar uzunluğu ile ρ={rho_size[0]:.2f} (n=20 sınıf, gürültülü). "
             + ("Frekans AP'nin zayıf belirleyicisi (|ρ|<0.3): zayıf sınıflarda sorun veri azlığından çok nesne boyutu/görünümü/bağlamıdır.\n" if abs(rho95[0]) < 0.3 else "Frekans AP ile anlamlı ilişkili: seyrek sınıflar daha zayıf.\n"))

    # ---- 2 boyut
    Ag = {"all": 0, "S": 1, "M": 2, "L": 3}
    size_rows = []; size_sum = {}
    gsize = np.array([size_bin(a) for a in R["garea"]])
    for s, a in Ag.items():
        if s == "all":
            n = nG
        else:
            n = int((gsize == s).sum())
        ap95s = np.nanmean(ap_arr(E, None, a)); ap50s = np.nanmean(ap_arr(E, 0, a)); rec = np.nanmean(recall_arr(E, 0, a))
        rec_all = np.nanmean(E.eval["recall"][:, :, a, 2][E.eval["recall"][:, :, a, 2] > -1])
        if s == "all": msk = np.ones(nG, bool)
        else: msk = gsize == s
        rec_op = float((R["gscore"][msk] >= op).mean()) if msk.any() else float("nan")
        size_rows.append([s, n, fm(ap95s, 3), fm(ap50s, 3), fm(rec, 3), fm(rec_all, 3), fm(rec_op, 3)])
        size_sum[s] = dict(n=n, ap5095=float(ap95s), ap50=float(ap50s), rec50=float(rec), rec_op=rec_op)
    summ["boyut"] = size_sum
    fine = [(0, 16 ** 2, "<16²"), (16 ** 2, 32 ** 2, "16²–32²"), (32 ** 2, 64 ** 2, "32²–64²"), (64 ** 2, 96 ** 2, "64²–96²"),
            (96 ** 2, 192 ** 2, "96²–192²"), (192 ** 2, 1e12, "≥192²")]
    frows = []
    for lo, hi, lab in fine:
        msk = (R["garea"] >= lo) & (R["garea"] < hi)
        if msk.sum() == 0:
            frows.append([lab, 0, "–", "–", "–"]); continue
        frows.append([lab, int(msk.sum()), pct(float((~R["missed"][msk]).mean())), pct(float((R["gscore"][msk] >= op).mean())),
                      pct(float(R["gmt"][msk][:, 5].mean()))])
    L.append("## 2. Nesne boyutu kırılımı (orijinal çözünürlük; COCO S<32²≤M<96²≤L)\n")
    L.append(md_table(["boyut", "GT sayısı", "AP50-95", "AP50", "<e-posta> (tüm conf, sınıf ort.)", "<e-posta>:0.95 (sınıf ort.)", f"<e-posta> skor≥{op}"], size_rows))
    L.append("\nİnce alan aralıkları (GT düzeyi, eşleşme bayrağı):\n")
    L.append(md_table(["alan (px²)", "GT", "<e-posta> (tüm conf)", f"<e-posta> skor≥{op}", "<e-posta> (tüm conf)"], frows))
    # sinif x boyut AP
    srows = []
    for k in range(K):
        row = [names[k]]
        for s in ("S", "M", "L"):
            n = int(((gsize == s) & (R["gcls"] == k)).sum())
            apv = ap_arr(E, None, Ag[s])[k]
            row += [n, fm(apv, 2) if n >= 5 else "–"]
        srows.append(row)
    L.append("\nSınıf × boyut AP50-95 (n<5 ise '–'):\n")
    L.append(md_table(["sınıf", "n_S", "AP_S", "n_M", "AP_M", "n_L", "AP_L"], srows))
    sS, sL = size_sum["S"], size_sum["L"]
    L.append(f"\nKüçük nesneler (n={sS['n']}, val GT'nin %{100 * sS['n'] / nG:.1f}'i): AP50-95 {sS['ap5095']:.3f} vs büyük {sL['ap5095']:.3f} "
             f"(fark {100 * (sL['ap5095'] - sS['ap5095']):.1f} puan); <e-posta> {sS['rec50']:.3f} vs {sL['rec50']:.3f}. Not: ölçüm orijinal piksel; "
             "eğitim/çıkarım 640 px letterbox olduğundan (~1.3× büyütme) S nesneler modelde ≈<41 px görünür. Val'de S örneği az olduğundan S-AP güven aralığı geniştir.\n")

    # ---- 3 hata türü
    tcount = {t: int((typ == t).sum()) for t in TYPES}
    tcount["Miss"] = int(R["missed"].sum())
    op_mask = dsc >= op
    tcount_op = {t: int(((typ == t) & op_mask).sum()) for t in TYPES}
    tcount_op["Miss"] = int((R["gscore"] < op).sum())       # op'ta bulunamayan GT (eşleşmemiş + düşük skorlu)
    b50 = ORA["base"][0]; b95 = ORA["base"][1]
    orows = []
    for t in ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss", "FP_hepsi", "Kutu_kusursuz", "TP_IoU_sirali"):
        o50, o95, _ = ORA[t]
        orows.append([t, fm(100 * (o50 - b50), 2), fm(100 * (o95 - b95), 2)])
    summ["oracle"] = {t: dict(d_ap50=float(ORA[t][0] - b50), d_ap5095=float(ORA[t][1] - b95)) for t in ORA if t != "base"}
    summ["hata_sayilari"] = tcount; summ["hata_sayilari_op"] = tcount_op
    L.append("## 3. Hata türü ayrıştırması (TIDE benzeri)\n")
    L.append("Sayılar: tüm tahminler (conf≥0.001) ve işletim noktası (skor≥" + str(op) + "). `Miss` tüm conf'ta hiçbir kutuyla IoU≥0.5 eşleşmeyen GT; "
             "işletim noktası sütununda ise o eşikte bulunamayan GT (eşleşmeyen + düşük skorla eşleşen).\n")
    rows = [[t, tcount[t], pct(tcount[t] / len(dets)) if t != "Miss" else pct(tcount[t] / nG) + " (GT'nin)", tcount_op[t]] for t in TYPES + ["Miss"]]
    L.append(md_table(["tür", "sayı (tüm conf)", "oran", f"sayı skor≥{op}"], rows))
    L.append("\nOracle ΔAP (puan): ilgili hata türü düzeltilirse mAP ne kadar artardı (AP50 ve AP50-95, sınıf ortalaması). "
             f"Taban: AP50={100 * b50:.2f}, AP50-95={100 * b95:.2f}.\n")
    L.append(md_table(["düzeltilen", "ΔAP50 (puan)", "ΔAP50-95 (puan)"], orows))
    # ---- difficult etkisi (protokol): etiketlerde olmayan difficult nesneleri bulan FP'ler
    dif = load_difficult(images, names)
    hit = np.zeros(len(dets), bool)
    dif_by = {}
    for d in dif: dif_by.setdefault(d[0], []).append(d)
    for j in np.where(typ != "TP")[0]:
        for d in dif_by.get(int(R["dimg"][j]), []):
            if d[1] == R["dcls"][j] and iou_mat(R["dxy"][j:j + 1], np.array([[d[2], d[3], d[4], d[5]]]))[0, 0] >= 0.5:
                hit[j] = True; break
    Ed = quiet_eval(R["cg"], [d for j, d in enumerate(dets) if not hit[j]])
    d50 = np.nanmean(ap_arr(Ed, 0)) - b50; d95 = np.nanmean(ap_arr(Ed, None)) - b95
    summ["difficult"] = dict(n_difficult=len(dif), fp_hit=int(hit.sum()), fp_hit_op=int((hit & op_mask).sum()), d_ap50=float(d50), d_ap5095=float(d95))
    L.append(f"\nProtokol etkisi: val'de etiketlerden çıkarılmış difficult nesne sayısı {len(dif)} (XML). Bunlardan biriyle aynı sınıf IoU≥0.5 örtüşen ve eşleşmemiş (FP sayılan) tahmin: "
             f"{int(hit.sum())} (skor≥{op}: {int((hit & op_mask).sum())}; işletim noktası FP'lerinin {pct((hit & op_mask).sum() / max(((typ != 'TP') & op_mask).sum(), 1))}'i). "
             f"Bu tahminler devkit gibi yok sayılsaydı ΔAP50 = {100 * d50:+.2f}, ΔAP50-95 = {100 * d95:+.2f} puan (Ultralytics protokolünün val üzerinde ölçülmüş cezası; makalede belirtilmeli, Bkg/Loc/Dupe sayılarının bir kısmı bu yüzden yapaydır).\n")
    L.append("\nNot: `Dupe` ve `Loc` ΔAP50-95 değerleri IoU=0.5 tanımına göredir; bu kutular daha yüksek IoU eşiklerinde TP olabildiği için "
             "silme oracle'ı (Dupe) AP50-95'i düşürebilir; Loc oracle'ı kutuyu GT'ye yapıştırdığı için AP50-95 kazancı AP50'den büyüktür.\n")
    dsz = np.array([size_bin(a) for a in R["darea"]]); gsz = np.array([size_bin(a) for a in R["garea"]])
    szrows = []
    for t in ("TP", "Dupe", "Cls", "Loc", "Both", "Bkg"):
        szrows.append([t + f" (skor≥{op}, tahmin kutusu alanı)"] + [int(((typ == t) & op_mask & (dsz == s_)).sum()) for s_ in ("S", "M", "L")])
    szrows.append([f"Miss (skor≥{op} bulunamayan GT, GT alanı)"] + [int(((R["gscore"] < op) & (gsz == s_)).sum()) for s_ in ("S", "M", "L")])
    szrows.append(["GT sayısı"] + [int((gsz == s_).sum()) for s_ in ("S", "M", "L")])
    L.append("\nHata türü × boyut (işletim noktası; tam uzun biçim `hata-kirilim.csv`):\n")
    L.append(md_table(["tür", "S", "M", "L"], szrows))
    top = sorted(((t, ORA[t][0] - b50) for t in ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss")), key=lambda x: -x[1])
    L.append(f"\nAP50'ye en çok mal olan hata türü: {top[0][0]} (+{100 * top[0][1]:.2f}), sonra {top[1][0]} (+{100 * top[1][1]:.2f}) ve {top[2][0]} (+{100 * top[2][1]:.2f}). "
             "Oracle'lar birbirinden bağımsız ölçüldü, toplamları ΔAP'ye eşit değildir (etkileşim). `Kutu_kusursuz` = tüm TP kutuları GT'ye yapıştırılırsa AP50-95'in erişeceği üst sınır; "
             "`TP_IoU_sirali` = yalnız TP'lerin skorları IoU sırasına göre dağıtılırsa (FP skorları aynı) kazanç, yani IoU-farkında skorlamanın (VFL/QFL/IoU-aware) teorik payı.\n")

    # sinif basina ΔAP50 (oracle) ve sinif x hata
    pc_rows = []
    for k in range(K):
        pc_rows.append([names[k]] + [fm(100 * (ORA[t][2][k] - ORA["base"][2][k]), 1) for t in ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss")])
    L.append("\nSınıf başına ΔAP50 (puan, oracle):\n")
    L.append(md_table(["sınıf", "Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"], pc_rows))
    # FN alt türleri
    mi = np.where(R["missed"])[0]
    sub_cls = sum(1 for g in mi if R["g_oth_det"][g] >= 0.5 and R["g_same_det"][g] < 0.5)
    sub_loc = sum(1 for g in mi if R["g_same_det"][g] >= 0.1 and R["g_same_det"][g] < 0.5 and R["g_oth_det"][g] < 0.5)
    sub_dup = sum(1 for g in mi if R["g_dup_cover"][g])
    sub_bg = len(mi) - sub_cls - sub_loc - sub_dup
    L.append(f"\nKaçırılan GT ({len(mi)}) alt kırılımı: yalnız başka sınıfla örtüşen kutu var (sınıf karışıklığı) {sub_cls}; "
             f"aynı sınıf kutu var ama IoU∈[0.1,0.5) (lokalizasyon) {sub_loc}; aynı sınıf IoU≥0.5 kutu var ama başka GT'ye atanmış (atama çakışması/örtüşme) {sub_dup}; "
             f"hiç kutu yok (gerçek kaçırma) {max(sub_bg, 0)} (kalan sınıflama kesin değil: bu üç grup kesişebilir).\n")
    summ["miss_alt"] = dict(sinif=sub_cls, lok=sub_loc, atama=sub_dup, gercek=max(sub_bg, 0))

    # ---- CSV (uzun)
    rows_csv = []
    dsize = np.array([size_bin(a) for a in R["darea"]])
    for k in range(K):
        for s in ("tum", "S", "M", "L"):
            for t in TYPES:
                mk = (R["dcls"] == k) & (typ == t) & ((dsize == s) if s != "tum" else True)
                dap = float(ORA[t][2][k] - ORA["base"][2][k]) if (s == "tum" and t in ("Cls", "Loc", "Both", "Dupe", "Bkg")) else float("nan")
                rows_csv.append([names[k], s, t, int(mk.sum()), int((mk & op_mask).sum()), dap])
            mk = (R["gcls"] == k) & R["missed"] & ((gsize == s) if s != "tum" else True)
            mko = (R["gcls"] == k) & (R["gscore"] < op) & ((gsize == s) if s != "tum" else True)
            dap = float(ORA["Miss"][2][k] - ORA["base"][2][k]) if s == "tum" else float("nan")
            rows_csv.append([names[k], s, "Miss", int(mk.sum()), int(mko.sum()), dap])
    csv_df = pd.DataFrame(rows_csv, columns=["sinif", "boyut", "hata_turu", "sayi", f"sayi_skor_ge_{op}", "dAP50_oracle"])

    # ---- 4 lokalizasyon
    tp_idx = np.where(typ == "TP")[0]
    ious = R["iou_match"][tp_idx]
    edges = np.arange(0.5, 1.0001, 0.05)
    h_all, _ = np.histogram(ious, bins=edges); h_op, _ = np.histogram(R["iou_match"][tp_idx[dsc[tp_idx] >= op]], bins=edges)
    hrows = [[f"{edges[i]:.2f}–{edges[i + 1]:.2f}", int(h_all[i]), pct(h_all[i] / h_all.sum()), int(h_op[i]), pct(h_op[i] / max(h_op.sum(), 1))] for i in range(len(h_all))]
    L.append("## 4. Lokalizasyon kalitesi\n")
    L.append(md_table(["IoU aralığı", "TP (tüm conf)", "oran", f"TP skor≥{op}", "oran"], hrows))
    iou_op = R["iou_match"][tp_idx[dsc[tp_idx] >= op]]
    summ["tp_iou"] = dict(ort=float(ious.mean()), ort_op=float(iou_op.mean()), p_lt075=float((iou_op < 0.75).mean()))
    sp = spearmanr(dsc[tp_idx], ious)
    L.append(f"\nTP IoU ortalaması {ious.mean():.3f} (skor≥{op}: {iou_op.mean():.3f}); işletim noktası TP'lerinin %{100 * (iou_op < 0.75).mean():.1f}'i IoU<0.75. "
             f"TP'lerde skor–IoU Spearman ρ={sp[0]:.2f} (skorun konum kalitesini ne kadar taşıdığı).\n")
    crow = cls_df.sort_values("AP50_AP75", ascending=False)
    miou = [float(np.mean(R["iou_match"][tp_idx][R["dcls"][tp_idx] == k])) for k in range(K)]
    rows = [[r.sinif, fm(r.AP50, 3), fm(r.AP75, 3), fm(100 * r.AP50_AP75, 1), fm(miou[names.index(r.sinif)], 3)] for r in crow.itertuples()]
    L.append("Sınıf başına AP50 − AP75 (büyükten küçüğe) ve TP ort. IoU:\n")
    L.append(md_table(["sınıf", "AP50", "AP75", "AP50−AP75 (puan)", "TP ort. IoU"], rows))
    L.append(f"\nAP50→AP75 düşüşü sınıf ortalaması {100 * (m50 - m75):.1f} puan; en büyük düşüş " + ", ".join(f"{r.sinif} ({100 * r.AP50_AP75:.0f})" for r in crow.head(3).itertuples()) +
             ". Kutu-regresyon (IoU/DFL/box kazancı) manipülasyonlarının hedef alanı budur.\n")

    # ---- 5 karışıklık
    # GT'ye en yuksek skorlu kutu (IoU>=0.5, skor>=op)
    order = np.argsort(-dsc)
    conf_mat = np.zeros((K + 1, K + 1), int)  # satir=GT sinifi (+arka plan), sutun=tahmin (+kacirilan)
    ok_idx = order[dsc[order] >= op]
    by_img = {}
    for j in ok_idx: by_img.setdefault(R["dimg"][j], []).append(j)
    g_by = {}
    for g in range(nG): g_by.setdefault(R["gimg"][g], []).append(g)
    for im, gj in g_by.items():
        dj = np.array(by_img.get(im, []), int)
        for g in gj:
            if len(dj) == 0:
                conf_mat[R["gcls"][g], K] += 1; continue
            M = iou_mat(R["gxy"][g:g + 1], R["dxy"][dj])[0]
            cand = np.where(M >= 0.5)[0]
            if len(cand) == 0:
                conf_mat[R["gcls"][g], K] += 1
            else:
                # dogru sinif kutu varsa dogru; yoksa en yuksek skorlu yanlis sinif
                corr = [c for c in cand if R["dcls"][dj[c]] == R["gcls"][g]]
                pick = dj[corr[0]] if corr else dj[cand[0]]  # dj skor sirali
                conf_mat[R["gcls"][g], R["dcls"][pick]] += 1
    pairs = []
    for a in range(K):
        for b in range(K):
            if a != b and conf_mat[a, b] > 0:
                pairs.append((names[a], names[b], int(conf_mat[a, b]), conf_mat[a, b] / max(conf_mat[a].sum(), 1)))
    pairs.sort(key=lambda x: -x[2])
    sym = {}
    for a, b, n, _ in pairs:
        key = tuple(sorted((a, b))); sym[key] = sym.get(key, 0) + n
    symr = sorted(sym.items(), key=lambda x: -x[1])[:8]
    cm_df = pd.DataFrame(conf_mat, index=names + ["arka_plan/kacirilan"], columns=names + ["kacirilan"])
    L.append("## 5. Sınıf karışıklığı\n")
    L.append(f"İşletim noktası (skor≥{op}, IoU≥0.5): her GT için eşleşen en yüksek skorlu kutu (doğru sınıf varsa o). Matris dosyası: `{args.out_md.stem}_karisiklik.csv`.\n")
    L.append(md_table(["GT sınıfı → tahmin", "sayı", "GT'nin payı"], [[f"{a} → {b}", n, pct(r)] for a, b, n, r in pairs[:12]]))
    L.append("\nÇift bazında (yönsüz) toplam: " + "; ".join(f"{a}/{b} {n}" for (a, b), n in symr) + ".\n")
    n_cls_op = int(sum(n for _, _, n, _ in pairs)); n_gt = nG
    cl = {}
    for j in np.where((typ == "Cls") & (dsc >= 0.05))[0]:
        key = (names[R["oth_cls"][j]], names[R["dcls"][j]]); cl[key] = cl.get(key, 0) + 1
    clt = sorted(cl.items(), key=lambda x: -x[1])[:10]
    L.append("\nTüm-conf alternatif görünüm: `Cls` türü tahminler (IoU≥0.5 başka sınıf GT, skor≥0.05), GT sınıfı → tahmin:\n")
    L.append(md_table(["GT → tahmin", "tahmin sayısı"], [[f"{a} → {b}", n] for (a, b), n in clt]))
    can = [("cat", "dog"), ("cow", "sheep"), ("chair", "diningtable"), ("chair", "sofa"), ("horse", "cow"), ("bus", "train"), ("motorbike", "bicycle")]
    L.append("\nKanonik çiftler (yönsüz; `Cls` tahmin sayısı skor≥0.05 | işletim noktası GT sayısı): " +
             "; ".join(f"{a}/{b} {cl.get((a, b), 0) + cl.get((b, a), 0)} | {int(conf_mat[names.index(a), names.index(b)] + conf_mat[names.index(b), names.index(a)])}" for a, b in can) + ".\n")
    L.append(f"İşletim noktasında GT'lerin {pct(n_cls_op / n_gt)}'i başka sınıf olarak etiketlendi ({n_cls_op}/{n_gt}); {pct(conf_mat[:, K].sum() / n_gt)}'i hiç bulunamadı. "
             f"Sınıf karışıklığı oracle ΔAP50 = {100 * (ORA['Cls'][0] - b50):.2f} puan (§3; Cls, altı hata türü içinde {1 + sum(1 for t in ('Loc','Both','Dupe','Bkg','Miss') if ORA[t][0] > ORA['Cls'][0])}. sırada); işletim noktasında karışan GT oranı yalnız {pct(n_cls_op / n_gt)}, yani sorun sınıf ayrımından çok kutu/skor düzeyindedir.\n")
    summ["karisiklik_ust"] = [(f"{a}->{b}", n) for a, b, n, _ in pairs[:5]]

    # ---- 6 kalabalık
    cnt = np.bincount(R["gimg"], minlength=len(images) + 1)[R["gimg"]]
    groups = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3–4"), (5, 8, "5–8"), (9, 999, "≥9")]
    # bos gt'li goruntu yok; goruntu basina FP (op)
    fp_op = ((typ != "TP") & op_mask)
    fp_img = np.bincount(R["dimg"][fp_op], minlength=len(images) + 1)
    grows = []
    n_img_grp = {}
    gcount_img = np.bincount(R["gimg"], minlength=len(images) + 1)
    for lo, hi, lab in groups:
        gm = (cnt >= lo) & (cnt <= hi)
        im_ids = np.where((gcount_img >= lo) & (gcount_img <= hi))[0]
        im_ids = im_ids[im_ids > 0]
        grows.append([lab, len(im_ids), int(gm.sum()), pct(float((~R["missed"][gm]).mean())), pct(float((R["gscore"][gm] >= op).mean())),
                      pct(float(R["gmt"][gm][:, 5].mean())), fm(float(fp_img[im_ids].mean()), 2)])
    L.append("## 6. Kalabalık ve örtüşme\n")
    L.append(md_table(["nesne/görüntü", "görüntü", "GT", "<e-posta> (tüm conf)", f"<e-posta> skor≥{op}", "<e-posta> (tüm conf)", f"FP/görüntü (skor≥{op})"], grows))
    ov = [(0, 0.1, "<0.1 (komşu yok)"), (0.1, 0.3, "0.1–0.3"), (0.3, 0.5, "0.3–0.5"), (0.5, 0.7, "0.5–0.7"), (0.7, 1.01, "≥0.7")]
    orows2 = []
    for lo, hi, lab in ov:
        gm = (R["g_nb_same"] >= lo) & (R["g_nb_same"] < hi)
        orows2.append([lab, int(gm.sum()), pct(float((~R["missed"][gm]).mean())) if gm.any() else "–",
                       pct(float((R["gscore"][gm] >= op).mean())) if gm.any() else "–"])
    L.append("\nAynı sınıftan başka GT ile en yüksek IoU (örtüşme) gruplarına göre recall:\n")
    L.append(md_table(["komşu GT IoU", "GT", "<e-posta> (tüm conf)", f"<e-posta> skor≥{op}"], orows2))
    # cift analizi
    pr_done = set(); pair_stats = {"iki": 0, "bir": 0, "hic": 0}; pair_hi = {"iki": 0, "bir": 0, "hic": 0}
    for g in range(nG):
        h = R["g_nb_same_gt"][g]
        if h < 0 or R["g_nb_same"][g] < 0.3: continue
        key = (min(g, h), max(g, h))
        if key in pr_done: continue
        pr_done.add(key)
        f = int(~R["missed"][g]) + int(~R["missed"][h])
        nm = {2: "iki", 1: "bir", 0: "hic"}[f]
        pair_stats[nm] += 1
        if R["g_nb_same"][g] >= 0.7: pair_hi[nm] += 1
    L.append(f"\nAynı sınıf GT çiftleri (IoU≥0.3): {sum(pair_stats.values())} çift; ikisi de bulundu {pair_stats['iki']}, yalnız biri {pair_stats['bir']}, hiçbiri {pair_stats['hic']}. "
             f"IoU≥0.7 (NMS eşiği üstü, doğrudan NMS etkilenir): {sum(pair_hi.values())} çift, ikisi de bulundu {pair_hi['iki']}, yalnız biri {pair_hi['bir']}, hiçbiri {pair_hi['hic']}. "
             f"Kaçırılan GT'lerin {sub_dup}/{len(mi)}'i aynı sınıf IoU≥0.5 bir kutuya sahip ama o kutu başka GT'ye atanmış (olası NMS/atama kaybı).\n")
    L.append(f"Kalabalıkla recall düşüşü: tek nesneli görüntüde {grows[0][4]} iken ≥9 nesnelide {grows[-1][4]} (skor≥{op}); tüm-conf recall {grows[0][3]} → {grows[-1][3]}, <e-posta> {grows[0][5]} → {grows[-1][5]}. "
             "Çiftlerde tek-bulunma ve atama çakışması sayıları küçükse kayıp NMS'ten değil düşük skor/zayıf konumdan gelir.\n")
    summ["kalabalik"] = {g[0]: dict(n_img=g[1], recall=g[3]) for g in grows}
    summ["nms_ciftleri"] = pair_stats

    # ---- 7 kalibrasyon
    bins = np.linspace(0, 1, 11)
    tpf = (typ == "TP").astype(float)
    tp75 = R["mt"][:, 5].astype(float)
    crows = []; ece_all = 0.0; ece_01 = 0.0; n01 = int((dsc >= 0.1).sum())
    for i in range(10):
        m = (dsc >= bins[i]) & (dsc < bins[i + 1] if i < 9 else dsc <= 1.0)
        if m.sum() == 0:
            continue
        prec = tpf[m].mean(); ms = dsc[m].mean()
        crows.append([f"{bins[i]:.1f}–{bins[i + 1]:.1f}", int(m.sum()), fm(ms, 3), fm(prec, 3), fm(tp75[m].mean(), 3),
                      fm(float(R["iou_match"][m & (typ == 'TP')].mean()) if (m & (typ == 'TP')).any() else float('nan'), 3), fm(ms - prec, 3)])
        ece_all += m.sum() / len(dsc) * abs(ms - prec)
        if bins[i] >= 0.1: ece_01 += m.sum() / n01 * abs(ms - prec)
    # sinif-bazli ECE (skor>=0.1)
    L.append("## 7. Kalibrasyon (skor vs gerçek precision)\n")
    L.append("Tahmin 'doğru' = IoU≥0.5 ile eşleşen (<e-posta>); ayrıca IoU≥0.75 için precision. Not: YOLO sınıf skoru olasılık değil, TAL hizalama hedefli (IoU'ya bağlı) bir skordur; "
             "düşük skor bölgesindeki düşük precision 'aday havuzu' doğasından gelir.\n")
    L.append(md_table(["skor aralığı", "tahmin", "ort. skor", "<e-posta>", "<e-posta>", "TP ort. IoU", "skor−precision"], crows))
    L.append(f"\nECE (tüm tahminler, 10 eşit genişlikli kutu) = {ece_all:.3f}; ECE (skor≥0.1) = {ece_01:.3f}. "
             "`skor−precision` pozitifse aşırı güvenli, negatifse eksik güvenli. ")
    hi_m = dsc >= 0.5
    p50h, p75h = tpf[hi_m].mean(), tp75[hi_m].mean()
    gap_hi = np.mean([float(r[6]) for r in crows[5:]])
    L.append(f"Skor≥0.5 tahminlerin <e-posta> = {pct(p50h)}, <e-posta> = {pct(p75h)} (fark {100 * (p50h - p75h):.1f} puan). Skor 0.5 üstü kutularda (skor−precision) ortalaması {gap_hi:+.3f}: "
             + ("model yüksek skorlarda eksik güvenli (skor < gerçek <e-posta>), sıralama iyi; asıl boşluk <e-posta> ile @0.75 arasındaki konum kalitesidir. " if gap_hi < -0.02 else "model yüksek skorlarda aşırı güvenli. ")
             + f"TP'lerde skor–IoU Spearman ρ={sp[0]:.2f}: skor konum kalitesini kısmen taşıyor. TP_IoU_sirali oracle'ı (§3) IoU-farkında skorlamanın teorik payını sayısallaştırıyor.\n")
    summ["kalibrasyon"] = dict(ece_all=float(ece_all), ece_01=float(ece_01), prec50_hi=float(tpf[hi_m].mean()), prec75_hi=float(tp75[hi_m].mean()),
                               rho_skor_iou=float(sp[0]))

    # ---- 8 egitim dinamigi
    res = pd.read_csv(run_dir / "train" / "results.csv"); res.columns = [c.strip() for c in res.columns]
    if len(res) < 30:
        L.append("## 8. Eğitim dinamiği\n")
        L.append("Koşum 30 epoch'ten kısa: eğim/close_mosaic analizi anlamsız, atlandı.\n")
    else:
        y = res["metrics/mAP50-95(B)"].values; n = len(y)
        last20 = y[-20:]; s20 = slope_per10(last20); s10pre = slope_per10(y[-30:-10]) if n >= 30 else float("nan")
        cm_ep = 90; pre = y[n - 20:n - 10]; post = y[n - 10:]
        best_ep = int(np.argmax(y)) + 1
        tl = {c: res[f"train/{c}_loss"].values for c in ("box", "cls", "dfl")}
        vl = {c: res[f"val/{c}_loss"].values for c in ("box", "cls", "dfl")}
        jump = {c: tl[c][n - 10] / tl[c][n - 11] - 1 for c in tl}
        drows = [[c, fm(tl[c][0], 3), fm(tl[c][n - 11], 3), fm(tl[c][-1], 3), fm(vl[c][-1], 3), fm(vl[c][-1] / tl[c][-1], 2), fm(vl[c][n - 11] / tl[c][n - 11], 2), pct(jump[c])] for c in tl]
        summ["egitim"] = dict(slope_last20=s20, best_ep=best_ep, max=float(y.max()), last=float(y[-1]), pre_cm=float(pre.mean()), post_cm=float(post.mean()),
                              std_last20=float(np.std(last20)))
        L.append("## 8. Eğitim dinamiği (results.csv)\n")
        rows = [["epoch sayısı", n], ["en iyi epoch (mAP50-95)", f"{best_ep} ({y.max():.4f})"], ["son epoch mAP50-95", fm(float(y[-1]), 4)],
                ["epoch 50 / 80 / 90 / 100", " / ".join(fm(float(y[e - 1]), 4) for e in (50, 80, 90, 100))],
                ["son 20 epoch eğimi (puan/10 epoch)", fm(100 * s20, 3)], ["önceki 20 epoch (ep 71-90) eğimi", fm(100 * s10pre, 3)],
                ["son 20 epoch std (puan)", fm(100 * float(np.std(last20)), 3)],
                ["ep 81-90 ort. vs ep 91-100 ort. (close_mosaic=10)", f"{fm(float(pre.mean()), 4)} vs {fm(float(post.mean()), 4)} (Δ {100 * (post.mean() - pre.mean()):+.2f} puan)"],
                ["mAP50 son epoch", fm(float(res['metrics/mAP50(B)'].values[-1]), 4)],
                ["results.csv en iyi ep mAP50-95 vs metrics.json (best.pt yeniden val)", f"{fm(float(y.max()), 4)} vs {fm(metrics.get('val_map50_95'), 4)} (fark {100 * (y.max() - metrics.get('val_map50_95', np.nan)):.2f} puan; ölçüm-protokolü gürültü tabanı)"]]
        L.append(md_table(["gösterge", "değer"], rows))
        L.append("\nLoss tablosu (train loss mosaic açıkken ep1/ep90, kapalıyken ep100; val/train oranı):\n")
        L.append(md_table(["loss", "train ep1", "train ep90", "train ep100", "val ep100", "val/train ep100", "val/train ep90", "train loss ep90→91 değişim"], drows))
        tot = tl["box"][-1] + tl["cls"][-1] + tl["dfl"][-1]
        w = (tl["box"][-1] / tot, tl["cls"][-1] / tot, tl["dfl"][-1] / tot)
        w1 = tl["box"][0] + tl["cls"][0] + tl["dfl"][0]
        L.append(f"\nKayıp payı (results.csv loss_items kazançlı; loss.py:469-471 — önceki sürüm kazançları iki kez uyguluyordu, 2026-10-01 düzeltildi) ep100: box {pct(w[0])}, cls {pct(w[1])}, dfl {pct(w[2])} (ep1: box {pct(tl['box'][0] / w1)}, cls {pct(tl['cls'][0] / w1)}, dfl {pct(tl['dfl'][0] / w1)}). "
                 f"Son 20 epoch mAP eğimi {100 * s20:+.3f} puan/10 ep (std {100 * np.std(last20):.2f}) → " + ("doygun (eğim seed gürültüsü altında)" if abs(100 * s20) < 0.15 else "hâlâ eğimli") +
                 f". close_mosaic sonrası (ep 91-100) ortalama {100 * (post.mean() - pre.mean()):+.2f} puan. Train loss mosaic kapanınca ani düşer (cls {pct(jump['cls'])}); bu rejim değişimi, train/val farkının bir kısmının augmentation kaynaklı olduğunu gösterir. "
                 f"val cls loss son 20 epoch {vl['cls'][n - 20]:.3f}→{vl['cls'][-1]:.3f} ({'yükselmiyor' if vl['cls'][-1] <= vl['cls'][n - 20] else 'yükseliyor'}): val loss bozulmadığı için klasik aşırı öğrenme yok, ama mosaic kapalıyken val/train oranı cls {vl['cls'][-1] / tl['cls'][-1]:.2f}, box {vl['box'][-1] / tl['box'][-1]:.2f}: train kaybı val kaybından ayrışıyor (genelleme açığı); val mAP doygun.\n")

    # ---- 9 seed
    L.append("## 9. Seed varyansı\n")
    rows = []; grp = {}
    for d in sorted(glob.glob(str(WS / args.seed_glob))):
        mp = Path(d) / "metrics.json"
        if not mp.exists(): continue
        mm = load_metrics_val_only(mp)
        ep = mm.get("epochs_calisilan")
        rows.append([Path(d).name, mm.get("seed"), ep, fm(mm.get("val_map50_95"), 4), fm(mm.get("val_map50"), 4), fm(mm.get("val_map_small"), 3), fm(mm.get("val_map_medium"), 3), fm(mm.get("val_map_large"), 3)])
        grp.setdefault(ep, []).append(mm.get("val_map50_95"))
    L.append(md_table(["koşum", "seed", "epoch", "mAP50-95", "mAP50", "mAP_S", "mAP_M", "mAP_L"], rows))
    srows = []
    for ep, v in sorted(grp.items(), key=lambda x: (x[0] is None, x[0])):
        v = np.array([x for x in v if x is not None])
        srows.append([ep, len(v), fm(float(v.mean()), 4), fm(float(v.std(ddof=1)), 4) if len(v) > 1 else "n<2"])
    L.append("\nEpoch grubuna göre (aynı epoch sayısı, seed'ler arası; n≥2 iken std):\n")
    L.append(md_table(["epoch", "n koşum", "ort. mAP50-95", "std"], srows))
    L.append("\n*Tam (100 ep) koşumlarda seed1/seed2 bittiğinde şef bu betiği yeniden koşturunca tablo otomatik dolar (`--seed-glob`). Ön-uçuş (5 ep) iki seed yalnız erken işarettir; "
             "5 epoch'ta mAP henüz yakınsamadığından tam koşum seed std'si için temsilî değildir.*\n")

    # ---- 10 hız
    L.append("## 10. Hız profili (metrics.json)\n")
    keys = [("param_sayisi", "parametre"), ("gflops", "GFLOPs (640)"), ("latency_preprocess_ms", "ön-işleme ms"), ("latency_inference_ms", "çıkarım ms"),
            ("latency_postprocess_nms_ms", "NMS/son-işlem ms"), ("latency_toplam_ms", "toplam ms"), ("fps", "FPS"), ("hiz_batch", "batch"),
            ("hiz_precision_fp16", "FP16 (1=evet)"), ("hiz_olcum_n_goruntu", "ölçüm görüntüsü"), ("hiz_isinma_n_goruntu", "ısınma görüntüsü"), ("egitim_sure_sa", "eğitim süresi (saat)")]
    L.append(md_table(["gösterge", "değer"], [[lab, metrics.get(k)] for k, lab in keys]))
    inf = metrics["latency_inference_ms"]; pre_ms = metrics["latency_preprocess_ms"]; nms = metrics["latency_postprocess_nms_ms"]; tot = metrics["latency_toplam_ms"]
    def fps_of(ms): return 1000.0 / ms
    vb = [["eğitim-zamanı (aug/loss) varyantları", "aynı", fm(tot, 2), fm(fps_of(tot), 0), "çıkarım grafiği değişmez; hız bütçesi ≈ 0"],
          ["Soft-NMS/DIoU-NMS (son-işlem ×1–5 varsayımı)", "NMS ×1–5", f"{pre_ms + inf + nms:.2f}–{pre_ms + inf + 5 * nms:.2f}", f"{fps_of(pre_ms + inf + 5 * nms):.0f}–{fps_of(tot):.0f}", "tahmin (ölçülmedi)"],
          ["TTA (Ultralytics augment, ~3 ölçek+flip; literatür çarpanı 3.6× çıkarım)", "çıkarım ×3.6 (literatür)", f"{pre_ms + 3.6 * inf + nms:.2f}", f"{fps_of(pre_ms + 3.6 * inf + nms):.0f}", "tahmin; literatur-cipa.md §4 (Ultralytics TTA dokümanı)"],
          ["YOLO11s'e geçiş (bağlam satırı)", "≈3.4× GFLOPs", "–", "–", "ölçülmedi"]]
    L.append("\nVaryant hız bütçesi (yalnız tahmin; gerçek ölçüm varyant koşumunda yapılır):\n")
    L.append(md_table(["varyant sınıfı", "çarpan", "tahmini toplam ms", "tahmini FPS", "not"], vb))
    L.append(f"\nBaseline batch=1 FP16 toplam {tot:.2f} ms ({metrics['fps']:.0f} FPS; çıkarım {inf:.2f} ms, NMS {nms:.2f} ms = toplamın %{100 * nms / tot:.0f}'i). 'Gerçek zamanlı' için ≥30 FPS kabul edilirse (33 ms) "
             f"~{33.3 / tot:.1f}× gecikme payı vardır; TTA bile (~{fps_of(pre_ms + 3.6 * inf + nms):.0f} FPS tahmini) eşiğin üstünde kalabilir, ancak hız-doğruluk satırı ayrı raporlanmalıdır. "
             "Not: hız ölçümü GPU'nun paylaşıldığı bir anda alınmış olabilir (eğitim kuyruğu); karşılaştırmalı hız ölçümü aynı koşulda yapılmalıdır.\n")

    return "\n".join(L), csv_df, cm_df, summ


# ----------------------------------------------------------------------------- hafif rapor (baska bolme: test)
def build_report_lean(preds_path, args):
    """--preds verildiginde: yalniz cekirdek hata-turu kirilimi + oracle dAP (sinif/boyut CSV'si val ile ayni sema).
    Egitim dinamigi, difficult, sinif dagilimi vb. bolumler (val'e ozgu girdiler) YOK. Betik test_preds.json uzerinde
    yalniz CPU'da calisir; model/GPU yok. Sonuc KESIFSEL ve post-hoc'tur (K5 test kilidi acildiktan sonra; secim/karar yok)."""
    names, images, anns, cats, stem2id = load_gt(args.etiket_listesi, args.data_yaml)
    dets = load_dets(None, stem2id, Path(preds_path))
    K = len(names); nG = len(anns); op = args.op_conf
    R = core(names, images, anns, cats, dets, op)
    E = R["E"]; typ = R["typ"]; dsc = R["dsc"]
    ORA = oracles(names, images, anns, cats, dets, R)
    a50 = ap_arr(E, 0); a75 = ap_arr(E, 5); a95 = ap_arr(E, None)
    summ = dict(map50=float(np.nanmean(a50)), map75=float(np.nanmean(a75)), map5095=float(np.nanmean(a95)),
                n_goruntu=len(images), n_gt=nG, n_tahmin=len(dets), kaynak_tahmin=str(preds_path), op_conf=op)
    gsize = np.array([size_bin(a) for a in R["garea"]]); dsize = np.array([size_bin(a) for a in R["darea"]])
    Ag = {"all": 0, "S": 1, "M": 2, "L": 3}; size_sum = {}
    for s_, a in Ag.items():
        n = nG if s_ == "all" else int((gsize == s_).sum())
        size_sum[s_] = dict(n=n, ap5095=float(np.nanmean(ap_arr(E, None, a))), ap50=float(np.nanmean(ap_arr(E, 0, a))))
    summ["boyut"] = size_sum
    tcount = {t: int((typ == t).sum()) for t in TYPES}; tcount["Miss"] = int(R["missed"].sum())
    op_mask = dsc >= op
    tcount_op = {t: int(((typ == t) & op_mask).sum()) for t in TYPES}; tcount_op["Miss"] = int((R["gscore"] < op).sum())
    b50, b95 = ORA["base"][0], ORA["base"][1]
    summ["oracle"] = {t: dict(d_ap50=float(ORA[t][0] - b50), d_ap5095=float(ORA[t][1] - b95)) for t in ORA if t != "base"}
    summ["hata_sayilari"] = tcount; summ["hata_sayilari_op"] = tcount_op
    rows_csv = []
    for k in range(K):
        for s_ in ("tum", "S", "M", "L"):
            for t in TYPES:
                mk = (R["dcls"] == k) & (typ == t) & ((dsize == s_) if s_ != "tum" else True)
                dap = float(ORA[t][2][k] - ORA["base"][2][k]) if (s_ == "tum" and t in ("Cls", "Loc", "Both", "Dupe", "Bkg")) else float("nan")
                rows_csv.append([names[k], s_, t, int(mk.sum()), int((mk & op_mask).sum()), dap])
            mk = (R["gcls"] == k) & R["missed"] & ((gsize == s_) if s_ != "tum" else True)
            mko = (R["gcls"] == k) & (R["gscore"] < op) & ((gsize == s_) if s_ != "tum" else True)
            dap = float(ORA["Miss"][2][k] - ORA["base"][2][k]) if s_ == "tum" else float("nan")
            rows_csv.append([names[k], s_, "Miss", int(mk.sum()), int(mko.sum()), dap])
    csv_df = pd.DataFrame(rows_csv, columns=["sinif", "boyut", "hata_turu", "sayi", f"sayi_skor_ge_{op}", "dAP50_oracle"])
    L = [f"# Hata analizi (hafif, test) — `{Path(preds_path).parent.name}`\n",
         f"*Üretici: `scripts/analiz/hata_analizi.py --preds` · {len(images)} görüntü, {nG} nesne, {len(dets)} tahmin (conf≥0.001) · "
         f"eşleştirme pycocotools COCOeval · işletim noktası skor≥{op}. KEŞİFSEL, post-hoc: K5 test kilidi açıldıktan sonra mevcut tahminlerden "
         f"CPU'da üretildi; hiçbir seçim/karar bu çıktıya dayanmaz.*\n",
         f"pycocotools mAP50-95={100 * summ['map5095']:.2f}, mAP50={100 * summ['map50']:.2f}, mAP75={100 * summ['map75']:.2f}.\n",
         "## Hata türü sayıları\n",
         md_table(["tür", "sayı (tüm conf)", f"sayı skor≥{op}"], [[t, tcount[t], tcount_op[t]] for t in TYPES + ["Miss"]]) + "\n",
         "## Oracle ΔAP (puan)\n",
         md_table(["düzeltilen", "ΔAP50", "ΔAP50-95"], [[t, fm(100 * (ORA[t][0] - b50), 2), fm(100 * (ORA[t][1] - b95), 2)]
                                                         for t in ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss", "FP_hepsi", "Kutu_kusursuz", "TP_IoU_sirali")]) + "\n",
         f"Taban: AP50={100 * b50:.2f}, AP50-95={100 * b95:.2f}. Not: Dupe/Loc ΔAP50-95 IoU=0.5 tanımına göre.\n"]
    return "\n".join(L), csv_df, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN))
    ap.add_argument("--out-md", default=None); ap.add_argument("--out-csv", default=None)
    ap.add_argument("--op-conf", type=float, default=0.25)
    ap.add_argument("--seed-glob", default="runs/*_baseline_s*")
    ap.add_argument("--hipotez-md", default=str(WS / "sonuclar" / "hata-analizi.hipotez.md"),
                    help="varsa rapora eklenir (yalnız varsayılan baseline koşumu için)")
    ap.add_argument("--preds", default=None, help="verilirse HAFIF mod: bu COCO-json tahmin dosyasi (ornek test_preds.json) analiz edilir")
    ap.add_argument("--etiket-listesi", default=None, help="hafif mod: goruntu yolu listesi (ornek data/splits/test.txt)")
    ap.add_argument("--data-yaml", default=None, help="hafif mod: sinif adlari yaml (ornek data/voc_test.yaml)")
    ap.add_argument("--out-tag", default=None, help="hafif mod: cikti adi etiketi (varsayilan: tahmin dosyasinin klasor adi)")
    args = ap.parse_args()
    if args.preds:
        if not (args.etiket_listesi and args.data_yaml):
            ap.error("--preds icin --etiket-listesi ve --data-yaml zorunlu")
        pp = Path(args.preds); pp = pp if pp.is_absolute() else (WS / pp).resolve()
        tag = args.out_tag or pp.parent.name
        out_dir = WS / "sonuclar"; out_dir.mkdir(exist_ok=True)
        md, csv_df, summ = build_report_lean(pp, args)
        omd = out_dir / f"hata-analizi-test_{tag}.md"
        omd.write_text(md, encoding="utf-8")
        csv_df.to_csv(out_dir / f"hata-kirilim-test_{tag}.csv", index=False, encoding="utf-8")
        json.dump(summ, open(str(omd) + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
        print("yazildi:", omd); return
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute(): run_dir = (WS / run_dir).resolve()
    default = run_dir.resolve() == DEFAULT_RUN.resolve()
    suf = "" if default else f"_{run_dir.name}"
    out_dir = WS / "sonuclar"; out_dir.mkdir(exist_ok=True)
    args.out_md = Path(args.out_md) if args.out_md else out_dir / f"hata-analizi{suf}.md"
    args.out_csv = Path(args.out_csv) if args.out_csv else out_dir / f"hata-kirilim{suf}.csv"
    md, csv_df, cm_df, summ = build_report(run_dir, args)
    if default and Path(args.hipotez_md).exists():
        md += "\n" + open(args.hipotez_md, encoding="utf-8").read()
    args.out_md.write_text(md, encoding="utf-8")
    csv_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    cm_df.to_csv(args.out_md.with_name(args.out_md.stem + "_karisiklik.csv"), encoding="utf-8")
    json.dump(summ, open(str(args.out_md) + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print("yazildi:", args.out_md, args.out_csv)


if __name__ == "__main__":
    main()
