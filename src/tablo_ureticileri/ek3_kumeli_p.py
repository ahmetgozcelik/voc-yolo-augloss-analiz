#!/usr/bin/env python
"""Hakem tur 3 Ö4(c): YOLO11s H01 icin kumeli bootstrap tek yonlu p (Delta<=0) — YALNIZ CPU (pycocotools/numpy; GPU yok).
analiz_ek.py test() ile AYNI dizi (ayni rng tohumu=0, ayni grup cekimleri, ayni ap_agirlikli); yalniz tek model/kart cifti.
Tutarlilik sinamasi: yeniden hesaplanan %95 GA, test-analiz.json'daki kumeli_boot_ga95 ile esit olmali (uretilen JSON'da kontrol alani).
Calistirma (WS venv): env/.venv/Scripts/python.exe <MW>/evidence/scripts/ek3_kumeli_p.py [--B 2000] [--model yolo11s] [--kart H01]
Cikti: <MW>/evidence/ek3_kumeli_p_<model>_<kart>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
MW = Path(__file__).resolve().parents[2]
WS = MW.parents[1] / "deney_workspace" / "2-1-enhancing-real-time-object-detection-on-pascal-v"
sys.path.insert(0, str(WS / "scripts" / "ek_deney"))
import analiz_ek as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--model", default="yolo11s")
ap.add_argument("--kart", default="H01")
a = ap.parse_args()
T = WS / "sonuclar" / "ek_deney" / "test"
liste = (WS / "data/kitti/splits/test.txt").read_text().split()
dc = json.loads((WS / "data/kitti/dontcare.json").read_text())
grup = A.test_gruplari()
gruplar = sorted(set(grup.values()))
uyeler = {g: [k for k, v in grup.items() if v == g] for g in gruplar}
sira = [Path(p).stem for p in liste]
gidx = {g: np.array([sira.index(i) for i in uyeler[g]]) for g in gruplar}

prep = {}
for f in sorted(T.glob("*.json")):
    if f.name == "ACILDI.json":
        continue
    r = json.loads(f.read_text(encoding="utf-8"))
    m = "yolo26n" if "_ek_yolo26n_" in r["run_id"] else "yolo11s"
    kart = next(k for k, v in A.VARY.items() if r["varyant"] == f"ek_{m}_{v}")
    if m == a.model and kart in ("B", a.kart):
        kay = A.eval_kayitlari(json.loads((T / r["run_id"] / "test_preds.json").read_text()), liste, dc)
        prep[(kart, int(r["seed"]))] = A.hazirla(kay, sira)
assert len(prep) == 6, sorted(prep)

rng = np.random.default_rng(0)
cekim = []
for _ in range(a.B):
    w = np.zeros(len(sira))
    for g in rng.choice(gruplar, size=len(gruplar), replace=True):
        w[gidx[g]] += 1
    cekim.append(w)
t0 = time.time()
boot = []
b_ap = {s: [A.ap_agirlikli(prep[("B", s)], w) for w in cekim] for s in (0, 1, 2)}
k_ap = {s: [A.ap_agirlikli(prep[(a.kart, s)], w) for w in cekim] for s in (0, 1, 2)}
boot = np.array([np.mean([100 * (k_ap[s][j] - b_ap[s][j]) for s in (0, 1, 2)]) for j in range(a.B)])
ga = [round(float(np.quantile(boot, 0.025)), 3), round(float(np.quantile(boot, 0.975)), 3)]
ref = json.loads((WS / "sonuclar/ek_deney/test-analiz.json").read_text(encoding="utf-8"))[a.model][a.kart]["kumeli_boot_ga95"]
out = {"model": a.model, "kart": a.kart, "B": a.B, "rng_seed": 0, "n_grup": len(gruplar),
       "ga95_yeniden": ga, "ga95_kaynak_test_analiz": ref, "ga_tutarli": ga == ref,
       "boot_ort": float(boot.mean()), "boot_sd": float(boot.std(ddof=1)),
       "p_tek_yonlu_le0": float((boot <= 0).mean()), "p_tek_yonlu_le0_add1": float(((boot <= 0).sum() + 1) / (a.B + 1)),
       "q10": float(np.quantile(boot, 0.10)), "q05": float(np.quantile(boot, 0.05)), "q025": float(np.quantile(boot, 0.025)),
       "sure_sn": round(time.time() - t0, 1),
       "yontem": "13 test sahne grubu uzerinde kumeli bootstrap; tek yonlu p = P*(Delta*<=0) (yuzdelik/basit); analiz_ek.py test() ile ayni rng/cekim; tohum varyansi icermez"}
(MW / "evidence" / f"ek3_kumeli_p_{a.model}_{a.kart}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
