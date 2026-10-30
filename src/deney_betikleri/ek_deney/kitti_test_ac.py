"""KITTI TEST KİLİDİ — TEK AÇILIŞ (ON-KAYIT-ek-deney.md "Test kilidi"). WS kökünden:
    python scripts/ek_deney/kitti_test_ac.py --onay TEST-KILIDI-AC [--kismi]
Korumalar:
  * --onay TEST-KILIDI-AC zorunlu; sonuclar/ek_deney/test/ACILDI.json varsa (önceki açılış) ÇALIŞMAZ.
  * --devam (ON-KAYIT v5b): yarıda kalan AYNI açılış, ACILDI.json'da dondurulan plan ve best/last sha256'larıyla sürer;
    tamamlanmış koşum dosyaları atlanır. Yeni koşum eklenemez.
  * ek-yolo26n + ek-yolo11s kuyruklarındaki her (varyant, seed) için .done rc=0 ve sağlık kapısından geçmiş bir koşum
    bulunmalı; eksik varsa durur (--kismi: eksikler "koşulmadı" diye kaydedilir — kısma kararı sonrası).
  * Açılıştan ÖNCE ON-KAYIT ve DONDURMA.sha256 hash'leri + dondurulmuş bölme denetimi gunluk/deney-gunlugu.md'ye yazılır.
Her koşum için: best.pt → DetectionValidator(kitti_test.yaml, split=val, imgsz 640, batch 16, save_json) → test_preds.json
→ kitti_degerlendir (DontCare'li birincil + DontCare'siz ikincil). Çıktı: sonuclar/ek_deney/test/<run_id>.json.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import kitti_degerlendir as kd  # noqa: E402
import saglik_kapisi as sk  # noqa: E402

OUT = WS / "sonuclar/ek_deney/test"
KUYRUKLAR = ["ek-yolo26n", "ek-yolo11s"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def kosum_bul(varyant, seed):
    """Varyant+seed için en son, rc=0, sağlık kapısından geçen tam koşum."""
    adaylar = sorted((WS / "runs").glob(f"*_{varyant}_s{seed}"), reverse=True)
    for rd in adaylar:
        d = rd / ".done"
        if not d.exists() or json.loads(d.read_text(encoding="utf-8")).get("rc") != 0:
            continue
        cfg = json.loads((rd / "config.json").read_text(encoding="utf-8"))
        if cfg.get("asama") != 3:
            continue
        if sk.denetle(rd, tam=True):
            continue
        return rd
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onay", required=True)
    ap.add_argument("--kismi", action="store_true")
    ap.add_argument("--devam", action="store_true", help="yarıda kalan AYNI açılışı ACILDI.json'daki dondurulmuş planla sürdür")
    a = ap.parse_args()
    if a.onay != "TEST-KILIDI-AC":
        sys.exit("onay metni yanlış")
    if a.devam:
        # ON-KAYIT v5b (hakem tur 4 Ö6): devam = aynı açılış; plan ve ağırlık hash'leri değişmemiş olmalı
        if not (OUT / "ACILDI.json").exists():
            sys.exit("--devam: ACILDI.json yok (açılış hiç başlamadı)")
        kayit = json.loads((OUT / "ACILDI.json").read_text(encoding="utf-8"))
        plan = []
        for x in kayit["plan"]:
            rd = WS / "runs" / x["run_id"]
            for k in ("best", "last"):
                if sha(rd / f"train/weights/{k}.pt") != x[f"{k}_sha256"]:
                    sys.exit(f"--devam: {x['run_id']} {k}.pt hash'i değişmiş")
            plan.append((x["varyant"], x["seed"], rd))
        g = WS / "gunluk/deney-gunlugu.md"
        g.write_text(g.read_text(encoding="utf-8").rstrip("\n") + "\n\n"
                     f"**KITTI TEST AÇILIŞI — devam** {datetime.now().isoformat(timespec='seconds')} (aynı açılış, aynı plan)\n",
                     encoding="utf-8")
    else:
        if (OUT / "ACILDI.json").exists():
            sys.exit("TEST DAHA ÖNCE AÇILDI — ikinci açılış yasak (ON-KAYIT); yarıda kaldıysa --devam")
        # 1) koşum envanteri
        plan, eksik = [], []
        for q in KUYRUKLAR:
            for l in (WS / f"kuyruk/{q}.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(l)
                rd = kosum_bul(r["varyant"], r["seed"])
                (plan if rd else eksik).append((r["varyant"], r["seed"], rd))
        if eksik and not a.kismi:
            sys.exit(f"{len(eksik)} koşum eksik/sağlıksız: {[(v, s) for v, s, _ in eksik][:10]} … (--kismi yalnız kısma kararıyla)")
        # açılıştan ÖNCE: her planlı koşumun best.pt ve last.pt'si var olmalı (yarıda düşmeyi önler)
        yok = [f"{rd.name}/{k}.pt" for _, _, rd in plan for k in ("best", "last") if not (rd / f"train/weights/{k}.pt").exists()]
        if yok:
            sys.exit(f"ağırlık dosyası eksik: {yok[:10]}")
        # 2) dondurma denetimi + günlük kaydı (açılıştan ÖNCE)
        don = (WS / "data/kitti/DONDURMA.sha256").read_text().split("\n")
        for satir in filter(None, don):
            h, ad = satir.split(maxsplit=1)
            if sha(WS / ad.lstrip("*")) != h:
                sys.exit(f"DONDURMA ihlali: {ad}")
        on = sha(WS / "hipotezler/ek_deney/ON-KAYIT-ek-deney.md")
        dd = sha(WS / "data/kitti/DONDURMA.sha256")
        zaman = datetime.now().isoformat(timespec="seconds")
        OUT.mkdir(parents=True, exist_ok=True)
        kayit = {"acilis": zaman, "on_kayit_sha256": on, "dondurma_sha256": dd, "kosum": len(plan),
                 "koşulmadı": [f"{v}_s{s}" for v, s, _ in eksik],
                 "plan": [{"run_id": rd.name, "varyant": v, "seed": s,
                           "best_sha256": sha(rd / "train/weights/best.pt"), "last_sha256": sha(rd / "train/weights/last.pt")}
                          for v, s, rd in plan]}
        (OUT / "ACILDI.json").write_text(json.dumps(kayit, ensure_ascii=False, indent=1), encoding="utf-8")
        g = WS / "gunluk/deney-gunlugu.md"
        g.write_text(g.read_text(encoding="utf-8").rstrip("\n") + "\n\n"
                     f"**KITTI TEST AÇILIŞI** {zaman} — ON-KAYIT sha256 {on}; DONDURMA sha256 {dd}; {len(plan)} koşum; "
                     f"koşulmadı: {len(eksik)}\n", encoding="utf-8")
    # 3) değerlendirme
    from ultralytics.models.yolo.detect import DetectionValidator
    liste = [l for l in (WS / "data/kitti/splits/test.txt").read_text().split() if l]
    dc = json.loads((WS / "data/kitti/dontcare.json").read_text())
    def tek(agirlik, ad):
        val = DetectionValidator(args=dict(data=str(WS / "data/kitti_test.yaml"), split="val", imgsz=640, batch=16,
                                           device=0, plots=False, save_json=True, project=str(OUT), name=ad,
                                           exist_ok=True, verbose=False, model=str(agirlik)))
        st = val(model=str(agirlik))
        preds = json.loads((Path(val.save_dir) / "predictions.json").read_text())
        return val, st, preds

    for v, s, rd in plan:
        best = rd / "train/weights/best.pt"
        vd = OUT / rd.name
        if (OUT / f"{rd.name}.json").exists():  # --devam: tamamlanmış koşum atlanır
            continue
        val, st, preds = tek(best, rd.name)
        sonuc = {"run_id": rd.name, "varyant": v, "seed": s, "best_pt_sha256": sha(best),
                 "ultralytics_test_map50_95": round(float(st.get("metrics/mAP50-95(B)", float("nan"))), 6),
                 "dontcare": kd.degerlendir(preds, liste, dc, 8), "dontcaresiz": kd.degerlendir(preds, liste, None, 8)}
        shutil.move(str(Path(val.save_dir) / "predictions.json"), str(vd / "test_preds.json"))
        # ON-KAYIT v5 madde 2: last.pt duyarlılığı (ikincil, keşifsel; aynı tek açılış)
        last = rd / "train/weights/last.pt"
        vl, _, preds_l = tek(last, rd.name + "_last")
        sonuc["last"] = {"last_pt_sha256": sha(last), "dontcare": kd.degerlendir(preds_l, liste, dc, 8),
                         "dontcaresiz": kd.degerlendir(preds_l, liste, None, 8)}
        shutil.move(str(Path(vl.save_dir) / "predictions.json"), str(vd / "test_preds_last.json"))
        shutil.rmtree(vl.save_dir, ignore_errors=True)
        (OUT / f"{rd.name}.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(rd.name, sonuc["dontcare"]["map50_95"], "last", sonuc["last"]["dontcare"]["map50_95"], flush=True)
    # 4) Model başına en çok bir H07 satırı (val'de seçilen; keşifsel) — B s0–2 best.pt
    h7 = WS / "sonuclar/ek_deney/h07_kitti.json"
    if h7.exists():
        import h07_kitti as h7m
        H = h7m.h07()
        cfgs = {c["ad"]: c for c in [H.R0] + H.SATIRLAR}
        for m, r in json.loads(h7.read_text(encoding="utf-8")).items():
            sec = r.get("secilen_test_satiri")
            if not sec:
                continue
            for v, s, rd in plan:
                if v != f"ek_{m}_baseline" or (OUT / f"h07/{m}_{sec}_s{s}.json").exists():
                    continue
                preds = h7m.satir_val(H, cfgs[sec], rd / "train/weights/best.pt", WS / "data/kitti_test.yaml",
                                      OUT / "h07_tmp", 0, m == "yolo26n")
                sonuc = {"run_id": rd.name, "model": m, "seed": s, "h07_satir": sec,
                         "dontcare": kd.degerlendir(preds, liste, dc, 8), "dontcaresiz": kd.degerlendir(preds, liste, None, 8)}
                (OUT / "h07").mkdir(exist_ok=True)
                (OUT / f"h07/{m}_{sec}_s{s}.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
                print("H07", m, sec, s, sonuc["dontcare"]["map50_95"], flush=True)
    else:
        print("UYARI: h07_kitti.json yok — H07 test satırı değerlendirilmedi")
    print("TEST AÇILIŞI TAMAM", len(plan))


if __name__ == "__main__":
    main()
