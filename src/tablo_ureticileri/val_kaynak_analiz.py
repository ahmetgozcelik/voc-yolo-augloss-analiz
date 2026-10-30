#!/usr/bin/env python
"""H03 val->test isaret degisimi icin KESIFSEL kontrol (Faz 2.5, post-hoc; karar/secim yok): val = 303 VOC2007-trainval + 697 VOC2012-trainval goruntusu
(WS/data/splits/split_meta.json), test = yalniz VOC2007 test. H03'un val Δ'sini goruntu kaynagina gore ayirir (pycocotools; mevcut val_preds.json uzerinde, CPU).
Cikti: MW/evidence/val_kaynak_analiz.json (evidence builder ingest eder). Yeni egitim/cikarim YOK.
Calistirma: <WS>/env/.venv/Scripts/python.exe evidence/scripts/val_kaynak_analiz.py
"""
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

MW = Path(__file__).resolve().parents[2]
WS = MW.parents[1] / "deney_workspace" / "2-1-enhancing-real-time-object-detection-on-pascal-v"
sys.path.insert(0, str(WS / "scripts" / "analiz"))
import hata_analizi as H  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RUNS = {("baseline", 0): "20260930-072103_baseline_s0", ("baseline", 1): "20260930-101624_baseline_s1", ("baseline", 2): "20260930-131427_baseline_s2",
        ("h03", 0): "20260930-231243_h03_boxcp_s0", ("h03", 1): "20261001-021011_h03_boxcp_s1", ("h03", 2): "20261001-050625_h03_boxcp_s2"}
names, images, anns, cats, stem2id = H.load_gt()
cg = H.make_gt(images, anns, cats)
ids07 = [im["id"] for im in images if not re.match(r"^\d{4}_", im["file_name"])]
ids12 = [im["id"] for im in images if re.match(r"^\d{4}_", im["file_name"])]
print("val 07:", len(ids07), "12:", len(ids12))
res = {}
for (arm, s), rid in RUNS.items():
    dets = H.load_dets(WS / "runs" / rid, stem2id)
    with contextlib.redirect_stdout(io.StringIO()):
        cd = cg.loadRes([dict(d) for d in dets])
    for tag, ids in (("07", ids07), ("12", ids12), ("tum", [im["id"] for im in images])):
        with contextlib.redirect_stdout(io.StringIO()):
            E = H.COCOeval(cg, cd, "bbox")
            E.params.imgIds = ids
            E.params.maxDets = [1, 10, 300]
            E.evaluate()
            E.accumulate()
        res[f"{arm}.s{s}.{tag}"] = {"map50_95": float(np.nanmean(H.ap_arr(E, None))) * 100, "map50": float(np.nanmean(H.ap_arr(E, 0))) * 100}
    print(arm, s, {k: round(v["map50_95"], 3) for k, v in res.items() if k.startswith(f"{arm}.s{s}.")})
d = {}
for tag in ("07", "12", "tum"):
    for m in ("map50_95", "map50"):
        ds = [res[f"h03.s{s}.{tag}"][m] - res[f"baseline.s{s}.{tag}"][m] for s in range(3)]
        d[f"{tag}.{m}"] = {"dseed": ds, "ort": float(np.mean(ds)), "sd": float(np.std(ds, ddof=1))}
out = {"n_07": len(ids07), "n_12": len(ids12), "kaynak": "WS/runs/<baseline|h03 val kosumlari>/val_preds.json; WS/data/splits/val.txt; pycocotools COCOeval (imgIds alt kumeleri)",
       "kosumlar": {f"{a}.s{s}": r for (a, s), r in RUNS.items()}, "ham": res, "delta_h03": d}
(MW / "evidence" / "val_kaynak_analiz.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
for k, v in d.items():
    print(k, [round(x, 3) for x in v["dseed"]], round(v["ort"], 3), round(v["sd"], 3))
