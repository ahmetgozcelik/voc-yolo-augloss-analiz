"""Ek deney kuyruklarını üretir (ON-KAYIT-ek-deney.md). WS kökünden:
    python scripts/ek_deney/kuyruk_uret.py duman      → kuyruk/ek-duman.jsonl     (8 kart × 2 model, fraction 0.05, 1 epoch)
    python scripts/ek_deney/kuyruk_uret.py onucus     → kuyruk/ek-onucus.jsonl    (B × 2 model, 2 epoch, tam veri)
    python scripts/ek_deney/kuyruk_uret.py yolo26n    → kuyruk/ek-yolo26n.jsonl   (8 kart × 3 seed, 100 epoch)
    python scripts/ek_deney/kuyruk_uret.py yolo11s    → kuyruk/ek-yolo11s.jsonl   (8 kart × 3 seed, 100 epoch)
    python scripts/ek_deney/kuyruk_uret.py h05pos     → kuyruk/ek-h05pos.jsonl    (YOLO11n × VOC, H05-pos α=1 γ=0 + H05-neg poz=1, 3 seed)
    python scripts/ek_deney/kuyruk_uret.py dumancp    → kuyruk/ek-duman-cp.jsonl  (yalnız H03/H08 × 2 model duman)
Ortak: 640 px, batch 16 (nbs 64), workers 4, optimizer=MuSGD lr0=0.01 momentum=0.937 warmup_bias_lr=0.0
(ilk deneyde `auto`nun fiilen ürettiği: trainer.py:1171-1172 MuSGD/0.01 + warmup_bias_lr=0; ısınma momentumu args.momentum=0.937'ye çeker).
"""
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
PY = (WS / "env/.venv/Scripts/python.exe").as_posix()  # Windows CreateProcess göreli yürütülebiliri cwd ile çözmez
KARTLAR = {  # kart → (varyant adı, betik)
    "B": ("baseline", "scripts/rakip_kos.py"),
    "H01": ("h01_multiscale25", "varyantlar/h01_multiscale25/kos.py"),
    "H02": ("h02_innerciou", "varyantlar/h02_innerciou/kos.py"),
    "H03": ("h03_boxcp", "varyantlar/h03_boxcp/kos.py"),
    "H04": ("h04_scale25", "varyantlar/h04_scale25/kos.py"),
    "H05": ("h05_vfl", "varyantlar/h05_vfl/kos.py"),
    "H06": ("h06_box10", "varyantlar/h06_box10/kos.py"),
    "H08": ("h08_boxcp_doz", "varyantlar/h08_boxcp_doz/kos.py"),
}
KART_OVERRIDE = {"H01": ["multi_scale=0.25"], "H03": ["copy_paste=0.5"], "H04": ["scale=0.25"],
                 "H06": ["box=10.0"], "H08": ["copy_paste=1.0"]}  # ilk deney kuyruklarıyla birebir (k4-*.jsonl)
MODEL = {"yolo26n": "rakip/weights/yolo26n.pt", "yolo11s": "rakip/weights/yolo11s.pt"}
KURAL = json.dumps({"nan_kes": True, "min_adim": 20, "alt_sinir": {"val_map50_95": 0.02}, "sabit_sn": 1200})
SURE_DK = {"yolo26n": 150, "yolo11s": 260}  # tam koşum zaman sınırı (tahminin ≈2 katı); ön-uçuştan sonra güncellenir


def satir(kart, model, seed, epochs, fraction, asama, etiket, kural=KURAL, zaman=None, ek=()):
    var, betik = KARTLAR[kart]
    komut = [PY, betik, "--seed", str(seed), "--epochs", str(epochs), "--fraction", str(fraction),
             "--imgsz", "640", "--batch", "16", "--workers", "4",
             "--model", MODEL[model], "--data", "data/kitti_dev.yaml", "--hiz-txt", "data/kitti/splits/val.txt",
             "--override", "optimizer=MuSGD", "--override", "lr0=0.01", "--override", "momentum=0.937",
             "--override", "warmup_bias_lr=0.0",
             *[x for o in KART_OVERRIDE.get(kart, []) for x in ("--override", o)],
             *ek, "--run-dir", "{RUN_DIR}", "--out", "{RUN_DIR}/metrics.json"]
    return {"varyant": f"ek_{model}_{var}", "seed": str(seed), "etiket": etiket, "hipotez": None if kart == "B" else kart,
            "ebeveyn": None, "gpu_zorunlu": True, "asama": asama, "zaman_siniri_dk": zaman or SURE_DK[model],
            "kural": kural, "komut": komut}


def main():
    hedef = sys.argv[1]
    rows = []
    if hedef == "duman":
        for m in MODEL:
            for k in KARTLAR:
                rows.append(satir(k, m, 0, 1, 0.05, 1, f"EK duman {m} {k} close_mosaic=0 (kod yolu; kanıt değil)", kural=None,
                                  zaman=20, ek=("--override", "close_mosaic=0")))
    elif hedef == "onucus":
        for m in MODEL:
            rows.append(satir("B", m, 0, 2, 1.0, 2, f"EK ön-uçuş {m} B close_mosaic=0 (2 epoch; süre/bellek; kanıt değil)",
                              kural=None, zaman=60, ek=("--override", "close_mosaic=0")))
    elif hedef in MODEL:
        for s in (0, 1, 2):          # seed-dış döngü: kesintide her seed'in tam kart seti önce biter
            for k in KARTLAR:
                rows.append(satir(k, hedef, s, 100, 1.0, 3, f"EK tam {hedef} {k} s{s} (ON-KAYIT-ek-deney)"))
    elif hedef == "h05pos":  # H05-pos (α=1 γ=0) ve H05-neg (poz=1), seed-dış döngü; ilk deney kurulumu (auto optimizer)
        for s in (0, 1, 2):
            for ad, hip, ek, etk in (("h05pos_vfl_a1g0", "H05-pos", ["--vfl-alpha", "1", "--vfl-gamma", "0"], "α=1 γ=0"),
                                     ("h05neg_vfl_poz1", "H05-neg", ["--vfl-poz", "1"], "α=0.75 γ=2 poz=1")):
                rows.append({"varyant": ad, "seed": str(s), "hipotez": hip,
                             "etiket": f"EK {hip} YOLO11n×VOC {etk} s{s} (val-only; VOC test AÇILMAZ)",
                             "ebeveyn": None, "gpu_zorunlu": True, "asama": 3, "zaman_siniri_dk": 300, "kural": KURAL,
                             "komut": [PY, "varyantlar/h05_vfl/kos.py", *ek,
                                       "--seed", str(s), "--epochs", "100", "--fraction", "1.0", "--imgsz", "640",
                                       "--batch", "16", "--workers", "4", "--run-dir", "{RUN_DIR}",
                                       "--out", "{RUN_DIR}/metrics.json"]})
    elif hedef == "dumanyeni":  # hakem tur 3 koşul (a): başlık sayaçları + H05-neg kod yolu
        for m in MODEL:
            for k in ("H02", "H05"):
                rows.append(satir(k, m, 0, 1, 0.05, 1, f"EK duman-yeni {m} {k} close_mosaic=0 (başlık sayacı; kanıt değil)",
                                  kural=None, zaman=20, ek=("--override", "close_mosaic=0")))
        for ad, hip, ek in (("h05pos_vfl_a1g0", "H05-pos", ["--vfl-alpha", "1", "--vfl-gamma", "0"]),
                            ("h05neg_vfl_poz1", "H05-neg", ["--vfl-poz", "1"])):
            rows.append({"varyant": ad, "seed": "0", "hipotez": hip, "etiket": f"EK duman-yeni {hip} YOLO11n×VOC (kanıt değil)",
                         "ebeveyn": None, "gpu_zorunlu": True, "asama": 1, "zaman_siniri_dk": 20, "kural": None,
                         "komut": [PY, "varyantlar/h05_vfl/kos.py", *ek, "--seed", "0", "--epochs", "1", "--fraction", "0.05",
                                   "--imgsz", "640", "--batch", "16", "--workers", "4", "--override", "close_mosaic=0",
                                   "--run-dir", "{RUN_DIR}", "--out", "{RUN_DIR}/metrics.json"]})
    elif hedef == "dumancp":
        for m in MODEL:
            for k in ("H03", "H08"):
                rows.append(satir(k, m, 0, 1, 0.05, 1, f"EK duman {m} {k} close_mosaic=0 (copy-paste kod yolu; kanıt değil)",
                                  kural=None, zaman=20, ek=("--override", "close_mosaic=0")))
    else:
        raise SystemExit(__doc__)
    ad = {"dumanyeni": "ek-duman-yeni", "dumancp": "ek-duman-cp", "duman": "ek-duman", "onucus": "ek-onucus", "yolo26n": "ek-yolo26n", "yolo11s": "ek-yolo11s", "h05pos": "ek-h05pos"}[hedef]
    p = WS / f"kuyruk/{ad}.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{p.relative_to(WS)}: {len(rows)} satır")


if __name__ == "__main__":
    main()
