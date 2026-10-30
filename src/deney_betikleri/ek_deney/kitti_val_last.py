"""Hakem tur 3 B3(c) tanısı (keşifsel, val; test kilidine dokunmaz). WS kökünden:
    python scripts/ek_deney/kitti_val_last.py
Her KITTI koşumunun last.pt'si KITTI val'de değerlendirilir (DontCare kurallı + DontCare'siz). best.pt'nin DontCare kurallı
val değeri val_preds.json'dan (analiz_ek.val ile aynı) alınır. Soru: best.pt seçimi (Ultralytics, DontCare'siz val uygunluğu)
DontCare kurallı val'de de best > last veriyor mu? Veriyorsa erken tepe gerçek genelleme açığı; vermiyorsa seçim ölçütü
uyumsuzluğu. Çıktı: sonuclar/ek_deney/val-last-tani.{json,md}
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import kitti_degerlendir as kd  # noqa: E402

OUT = WS / "sonuclar/ek_deney"


def main():
    from ultralytics.models.yolo.detect import DetectionValidator
    liste = [l for l in (WS / "data/kitti/splits/val.txt").read_text().split() if l]
    dc = json.loads((WS / "data/kitti/dontcare.json").read_text())
    kos = json.loads((OUT / "val-analiz.json").read_text(encoding="utf-8"))["kosumlar"]
    onceki = OUT / "val-last-tani.json"
    r = json.loads(onceki.read_text(encoding="utf-8")) if onceki.exists() else {}
    for ad, run in kos.items():
        if ad in r:
            continue
        rd = WS / "runs" / run
        best = kd.degerlendir(json.loads((rd / "val_preds.json").read_text()), liste, dc, 8)
        v = DetectionValidator(args=dict(data=str(WS / "data/kitti_dev.yaml"), split="val", imgsz=640, batch=16, device=0,
                                         plots=False, save_json=True, project=str(OUT / "val_last_tmp"), name=run,
                                         exist_ok=True, verbose=False, model=str(rd / "train/weights/last.pt")))
        v(model=str(rd / "train/weights/last.pt"))
        preds = json.loads((Path(v.save_dir) / "predictions.json").read_text())
        shutil.rmtree(v.save_dir, ignore_errors=True)
        mt = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))
        r[ad] = {"run_id": run, "en_iyi_epoch": mt.get("en_iyi_epoch"),
                 "best_dc": best["map50_95"], "best_dcsiz": mt["val_map50_95"],
                 "last_dc": kd.degerlendir(preds, liste, dc, 8)["map50_95"],
                 "last_dcsiz": kd.degerlendir(preds, liste, None, 8)["map50_95"]}
        onceki.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        print(ad, r[ad], flush=True)
    md = ["# KITTI val — best.pt / last.pt, DontCare kurallı ve DontCare'siz (hakem tur 3 B3c; keşifsel)\n",
          "| Koşum | en iyi ep | best DC | last DC | best−last DC (puan) | best DC'siz | last DC'siz | best−last DC'siz (puan) |",
          "|---|---|---|---|---|---|---|---|"]
    for ad, x in r.items():
        md.append(f"| {ad} | {x['en_iyi_epoch']} | {x['best_dc']:.4f} | {x['last_dc']:.4f} | {100 * (x['best_dc'] - x['last_dc']):+.2f} "
                  f"| {x['best_dcsiz']:.4f} | {x['last_dcsiz']:.4f} | {100 * (x['best_dcsiz'] - x['last_dcsiz']):+.2f} |")
    md.append("\n## Model başına özet (ortalama best−last, puan)\n\n| Model | n | DC | DC'siz | best−last DC > 0 olan koşum |\n|---|---|---|---|---|")
    for m in ("yolo26n", "yolo11s"):
        xs = [x for a, x in r.items() if a.startswith(m + "/")]
        if xs:
            d1 = [100 * (x["best_dc"] - x["last_dc"]) for x in xs]
            d2 = [100 * (x["best_dcsiz"] - x["last_dcsiz"]) for x in xs]
            md.append(f"| {m} | {len(xs)} | {np.mean(d1):+.2f} | {np.mean(d2):+.2f} | {sum(d > 0 for d in d1)}/{len(xs)} |")
    (OUT / "val-last-tani.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[-4:]))


if __name__ == "__main__":
    main()
