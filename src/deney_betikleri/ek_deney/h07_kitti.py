"""H07 (eğitimsiz çıkarım) — KITTI val, B s0–2 best.pt (ON-KAYIT-ek-deney.md "Eğitimsiz çıkarım (H07)"). WS kökünden:
    python scripts/ek_deney/h07_kitti.py [--model yolo11s|yolo26n|hepsi] [--data YAML --liste TXT --agirlik s0=PT ...]
YOLO11s: ilk deneyin satırları (R0, N50/N60/N80, S03/S05, U704/U768/U832, T1/T2, W1/W2) — varyantlar/h07_post/degerlendir.py'nin
  doğrulayıcı alt sınıfı ve son-işlemi (son_isle) birebir kullanılır.
YOLO26n: NMS'siz (end2end) → N/S/W UYGULANAMAZ; R0, U704/U768/U832 ve T1/T2 (augment destekliyorsa; hata verirse "uygulanamaz").
Her satır: predictions.json → kitti_degerlendir (DontCare = COCO crowd). Gecikme: batch 1, FP16, 50 ısınma + 300 KITTI val görüntüsü,
Ultralytics predictor speed (ön-işleme+çıkarım+son-işlem); yalnız R0 ve seçilen satır için.
Seçim (teste girecek satır): s0–2 ortalama DontCare mAP50-95'i en yüksek satır (R0 hariç); eşitlikte düşük gecikmeli. Keşifsel.
Çıktı: sonuclar/ek_deney/h07_kitti.json + .md.
"""
import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import kitti_degerlendir as kd  # noqa: E402

OUT = WS / "sonuclar/ek_deney"


def h07():
    spec = importlib.util.spec_from_file_location("h07_degerlendir", str(WS / "varyantlar/h07_post/degerlendir.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.tta_yamasi_uygula()
    return m


def satir_val(H, cfg, w, data, proje, cihaz, end2end):
    """Bir satırı val'de koşar, predictions.json'u döndürür."""
    from ultralytics.models.yolo.detect import DetectionValidator
    H.TTA["s"] = list(cfg["tta_s"]) if cfg.get("tta_s") else list(H.STD_S)
    H.TTA["stash"] = None
    a = dict(data=str(data), split="val", imgsz=cfg["imgsz"], batch=16, device=cihaz, workers=4, plots=False,
             save_json=True, project=str(proje), name=f"val_{cfg['ad']}", exist_ok=True, verbose=False, model=str(w))
    if cfg.get("augment"):
        a["augment"] = True
    if end2end:
        v = DetectionValidator(args=a)
    else:
        if not cfg.get("soft") and not cfg.get("wbf"):
            a["iou"] = cfg["iou"]
        v = H.dogrulayici_sinifi()(args=a)
        v.h07_cfg = cfg
    v(model=str(w))
    pj = Path(v.save_dir) / "predictions.json"
    preds = json.loads(pj.read_text())
    shutil.rmtree(v.save_dir, ignore_errors=True)
    return preds


def gecikme(H, w, cfg, liste, cihaz, end2end, isinma=50, olcum=300):
    """Batch 1, FP16, görüntü başına uçtan uca süre (ms, ort.). Düzeltme 2026-10-22: önceki sürüm yol listesini
    predict'e tek kaynak olarak veriyordu; Ultralytics listeyi TEK batch'te işler (LoadPilAndNumpy) → batch 1 değildi,
    U832'de OOM ile düştü (hiçbir gecikme değeri yazılmadan). YOLO11s: VOC ölçümünün motoru (Motor/gecikme_olc) birebir.
    YOLO26n (NMS'siz; son_isle NMS uygular, uygulanamaz): görüntü başına predict, perf_counter + cuda senkronu."""
    import time
    import cv2
    import torch
    imgs = [cv2.imread(p) for p in liste[:isinma + olcum]]
    if not end2end:
        dev = torch.device(f"cuda:{cihaz}" if cihaz != "cpu" else "cpu")
        motor = H.Motor(w, dev, half=dev.type == "cuda")
        return H.gecikme_olc(motor, cfg, imgs, isinma, olcum)[0]
    from ultralytics import YOLO
    m = YOLO(str(w))
    TTA_s = list(cfg["tta_s"]) if cfg.get("tta_s") else list(H.STD_S)
    kw = dict(imgsz=cfg["imgsz"], batch=1, half=cihaz != "cpu", device=cihaz, verbose=False, save=False,
              conf=0.001, max_det=300, augment=bool(cfg.get("augment")))
    sure = []
    for i, im in enumerate(imgs):
        H.TTA["s"] = TTA_s
        t = time.perf_counter()
        m.predict(source=im, **kw)
        if cihaz != "cpu":
            torch.cuda.synchronize()
        if i >= isinma:
            sure.append(time.perf_counter() - t)
    return 1000.0 * float(np.mean(sure))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hepsi")
    ap.add_argument("--data", default=str(WS / "data/kitti_dev.yaml"))
    ap.add_argument("--liste", default=str(WS / "data/kitti/splits/val.txt"))
    ap.add_argument("--agirlik", action="append", default=[], help="model:s=PT (kuru test için)")
    ap.add_argument("--cihaz", default="0")
    ap.add_argument("--atla-gecikme", action="store_true")
    ap.add_argument("--out", default=str(OUT / "h07_kitti.json"))
    a = ap.parse_args()
    cihaz = a.cihaz if a.cihaz == "cpu" else int(a.cihaz)
    H = h07()
    liste = [l for l in Path(a.liste).read_text().split() if l]
    dc = json.loads((WS / "data/kitti/dontcare.json").read_text())
    sys.path.insert(0, str(Path(__file__).parent))
    import analiz_ek as A
    modeller = A.MODELLER if a.model == "hepsi" else [a.model]
    elle = {}
    for x in a.agirlik:
        k, p = x.split("=", 1)
        elle[k] = Path(p)
    sonuc = {}
    for m in modeller:
        end2end = m == "yolo26n"
        satirlar = [H.R0] + [c for c in H.SATIRLAR if not end2end or c["ad"][0] in "UT"]
        agirlik = {}
        for s in (0, 1, 2):
            if f"{m}:{s}" in elle:
                agirlik[s] = elle[f"{m}:{s}"]
            else:
                rd = A.kosum(m, "B", s)
                if rd:
                    agirlik[s] = rd / "train/weights/best.pt"
        tablo = {}
        for cfg in satirlar:
            vals = {}
            for s, w in agirlik.items():
                try:
                    preds = satir_val(H, cfg, w, a.data, OUT / "h07_tmp", cihaz, end2end)
                    vals[s] = kd.degerlendir(preds, liste, dc, 8)["map50_95"]
                except Exception as e:  # YOLO26n augment desteklemezse
                    vals = {"uygulanamaz": f"{type(e).__name__}: {str(e)[:160]}"}
                    break
            tablo[cfg["ad"]] = vals
            print(m, cfg["ad"], vals, flush=True)
        r0 = tablo.get("R0", {})
        adaylar = {k: v for k, v in tablo.items() if k != "R0" and "uygulanamaz" not in v and v}
        ozet = {k: {"ort": float(np.mean([v[s] for s in v])), "delta_puan": float(100 * np.mean([v[s] - r0[s] for s in v if s in r0]))}
                for k, v in adaylar.items()}
        secilen = max(ozet, key=lambda k: ozet[k]["ort"]) if ozet else None
        gec = {}
        if not a.atla_gecikme and secilen and 0 in agirlik:
            cfgs = {c["ad"]: c for c in satirlar}
            gec = {"R0_ms": gecikme(H, agirlik[0], cfgs["R0"], liste, cihaz, end2end),
                   f"{secilen}_ms": gecikme(H, agirlik[0], cfgs[secilen], liste, cihaz, end2end)}
            gec["oran"] = gec[f"{secilen}_ms"] / gec["R0_ms"]
        sonuc[m] = {"satirlar": tablo, "ozet": ozet, "secilen_test_satiri": secilen, "gecikme_s0": gec,
                    "uygulanamaz_aileler": ["N", "S", "W"] if end2end else []}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["# H07 — KITTI val (keşifsel)\n"]
    for m, r in sonuc.items():
        md.append(f"\n## {m}  (seçilen test satırı: {r['secilen_test_satiri']}; gecikme {r['gecikme_s0']})\n\n| Satır | Δ puan (R0'a göre, s0–2 ort.) |\n|---|---|")
        for k, v in r["ozet"].items():
            md.append(f"| {k} | {v['delta_puan']:+.2f} |")
        for k, v in r["satirlar"].items():
            if "uygulanamaz" in v:
                md.append(f"| {k} | uygulanamaz ({v['uygulanamaz'][:60]}) |")
    Path(a.out).with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
