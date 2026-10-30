"""train1000 tanısı (hakem tur 4 Ö3): verilen B koşumlarının best/last.pt'si 1000 eğitim görüntüsünde (Ultralytics val, DontCare'siz).
WS kökünden: python sonuclar/ek_deney/tani/train1000_degerlendir_v2.py RUN_ID [RUN_ID ...]"""
import json, sys
from pathlib import Path
from ultralytics import YOLO
if __name__ == "__main__":
    T = Path("sonuclar/ek_deney/tani"); out = T / "train1000_sonuc.json"
    r = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for run in sys.argv[1:]:
        for w in ("best.pt", "last.pt"):
            m = YOLO(f"runs/{run}/train/weights/{w}")
            v = m.val(data=str(T / "tren_ws.yaml"), split="val", imgsz=640, batch=16, device=0, plots=False, verbose=False,
                      project=str(T / "tmp"), name=run + w, exist_ok=True, workers=0)
            r[f"{run}/{w}"] = {"map50_95": round(float(v.box.map), 4), "map50": round(float(v.box.map50), 4),
                               "sinif": [round(float(x), 3) for x in v.box.maps]}
            print("SONUC", run, w, r[f"{run}/{w}"], flush=True)
    out.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
