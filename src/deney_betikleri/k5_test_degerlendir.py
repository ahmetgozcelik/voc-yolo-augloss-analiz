#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
k5_test_degerlendir.py — K5 EGITIMSIZ degerlendirme (test kilidi acildiktan sonra, bir kez).

Egitim YOK: verilen best.pt'yi bir veri yaml'inda degerlendirir ve standart metrics.json yazar.
rakip_kos.py'nin degerlendir() fonksiyonuyla AYNI DetectionValidator ayarlari (split=val [yaml'in val'i],
imgsz 640, batch 16, conf/iou varsayilan 0.001/0.7, half varsayilan, save_json) kullanilir; --tta-t1 yalniz
H07 T1 yapilandirmasini (Ultralytics [1,.83,.67] + lr-flip, h07_post/degerlendir.py tta_yamasi_uygula) ekler.

Anahtar oneki (--onek):
  test : gercek K5 kosumu (data = data/voc_test.yaml, varyant adi '-test' son ekli ZORUNLU)
  val  : kuru test (data = voc_dev.yaml veya onun alt kumesi; test yaml'i YASAK)
Tutarlilik korumasi: onek=test <=> yaml adi voc_test.yaml; onek=val iken voc_test.yaml reddedilir.

Yazilanlar ({RUN_DIR}):
  metrics.json      : <onek>_map50_95, _map50, _map75, _precision, _recall, _map_small/_medium/_large (AP_S/M/L),
                      latency_* / fps (batch=1, FP16, h07 uctan-uca protokolu; olcum goruntuleri data/splits/val.txt'in
                      ilk 350'si — TEST GORUNTUSU KULLANILMAZ), egitim_sure_sn (= degerlendirme suresi, egitim_yok=1),
                      param_sayisi, cihaz, alt_grup (ap50_<sinif>, ap5095_<sinif>)
  <onek>_preds.json : validator predictions.json kopyasi (COCO-json: image_id, category_id, bbox, score)
  <onek>_paket.npz  : goruntu-duzeyi istatistik (tp, conf, pred_cls, target_cls, uzunluklar, goruntu adlari) —
                      eslestirilmis goruntu-bootstrap icin (h07 Paket ile ayni bicim)

Kullanim (kuyruk):
  python WS/scripts/k5_test_degerlendir.py --weights <best.pt> --varyant-adi baseline-test \
      --data WS/data/voc_test.yaml --onek test --seed 0 --out {RUN_DIR}/metrics.json [--tta-t1]
Kuru test (CPU, VAL): CUDA_VISIBLE_DEVICES=-1 ... --data WS/data/voc_dev.yaml --onek val --varyant-adi baseline-test
"""
import argparse
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

WS = Path(__file__).resolve().parents[1]
H07_YOL = WS / "varyantlar" / "h07_post" / "degerlendir.py"


def atomik_yaz(yol: Path, veri):
    yol = Path(yol)
    yol.parent.mkdir(parents=True, exist_ok=True)
    tmp = yol.with_suffix(yol.suffix + ".tmp")
    tmp.write_text(json.dumps(veri, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, yol)


def h07_yukle():
    spec = importlib.util.spec_from_file_location("h07_degerlendir", str(H07_YOL))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gpu_kullanim():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
        u, m = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")]
        return u, m
    except Exception:
        return None, None


def f6(x):
    return round(float(x), 6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--varyant-adi", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--onek", choices=["test", "val"], default="test")
    ap.add_argument("--tta-t1", action="store_true", help="H07 T1: Ultralytics TTA [1,.83,.67]+lr-flip")
    ap.add_argument("--seed", type=int, default=None, help="yalniz kayit icin (egitim seed'i)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gecikme-isinma", type=int, default=50)
    ap.add_argument("--gecikme-olcum", type=int, default=300)
    ap.add_argument("--gecikme-tekrar", type=int, default=3)
    ap.add_argument("--atla-gecikme", action="store_true")
    a = ap.parse_args()

    t_basla = time.time()
    onek = a.onek
    out_path = Path(a.out).resolve()
    run_dir = out_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    w_path = Path(a.weights).resolve()
    data_yaml = Path(a.data).resolve()

    # ---- tutarlilik korumalari (kilit disiplini) ----
    test_yaml_mi = data_yaml.name == "voc_test.yaml"
    if onek == "test" and not test_yaml_mi:
        sys.exit("[k5] HATA: --onek test yalniz data/voc_test.yaml ile kullanilir")
    if onek == "val" and test_yaml_mi:
        sys.exit("[k5] HATA: kuru test (--onek val) icin voc_test.yaml YASAK")
    if onek == "test" and not a.varyant_adi.endswith("-test"):
        sys.exit("[k5] HATA: test koşumunun varyant adi '-test' son ekli olmali")
    if not w_path.exists():
        sys.exit(f"[k5] HATA: agirlik yok: {w_path}")

    import torch
    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_ok else "cpu")
    device_arg = 0 if cuda_ok else "cpu"
    cihaz_str = f"cuda:0 / {torch.cuda.get_device_name(0)}" if cuda_ok else "cpu"
    print(f"[k5] varyant={a.varyant_adi} onek={onek} tta_t1={a.tta_t1} data={data_yaml} weights={w_path} cihaz={cihaz_str}")
    gpu_u0, gpu_m0 = gpu_kullanim()
    print(f"[k5] baslangic GPU kullanim={gpu_u0}% bellek={gpu_m0} MiB")

    h07 = h07_yukle()
    h07.tta_yamasi_uygula()   # yalniz DetectionModel._predict_augment'i degistirir; augment=False iken devreye girmez
    h07.TTA["s"] = list(h07.STD_S)
    h07.TTA["stash"] = None
    cfg = dict(h07.R0)
    if a.tta_t1:
        cfg = dict(h07.R0, ad="T1", aile="T", augment=True, tta_s=list(h07.STD_S))

    from ultralytics.models.yolo.detect import DetectionValidator
    import ultralytics as _ul

    class KVal(DetectionValidator):
        """DetectionValidator + goruntu adi ve goruntu-duzeyi istatistik yakalama (hesaplamayi degistirmez)."""

        def update_metrics(self, preds, batch):
            if not hasattr(self, "k5_names"):
                self.k5_names = []
            self.k5_names.extend(Path(f).name for f in batch["im_file"])
            return super().update_metrics(preds, batch)

        def get_stats(self):
            self.k5_stats = {k: list(v) for k, v in self.metrics.stats.items()}
            return super().get_stats()

    # rakip_kos.degerlendir ile ayni argumanlar (+ workers, + augment yalniz TTA'da)
    val_args = dict(
        data=str(data_yaml), split="val", imgsz=a.imgsz, batch=a.batch, device=device_arg, workers=a.workers,
        plots=False, save_json=True, project=str(run_dir), name=f"{onek}_degerlendir", exist_ok=True, verbose=False,
        model=str(w_path),
    )
    if a.tta_t1:
        val_args["augment"] = True
    v = KVal(args=val_args)
    stats = v(model=str(w_path))
    box = v.metrics.box
    print(f"[k5] {onek}_map50_95={stats.get('metrics/mAP50-95(B)'):.6f} map50={stats.get('metrics/mAP50(B)'):.6f}")

    per_ap50, per_ap5095 = {}, {}
    names = v.metrics.names
    for i, c in enumerate(v.metrics.ap_class_index):
        _p, _r, ap50, ap = v.metrics.class_result(i)
        per_ap50[names[int(c)]] = f6(ap50)
        per_ap5095[names[int(c)]] = f6(ap)

    # predictions.json -> <onek>_preds.json
    src = Path(v.save_dir) / "predictions.json"
    if src.exists():
        shutil.copy(src, run_dir / f"{onek}_preds.json")
        print(f"[k5] {onek}_preds.json yazildi")
    else:
        print(f"[k5] UYARI: {src} bulunamadi")

    # goruntu-duzeyi paket
    st = v.k5_stats
    P = h07.Paket(v.k5_names, st["tp"], st["conf"], st["pred_cls"], st["target_cls"])
    kendi_m, _ = P.map()
    print(f"[k5] kendi ap_per_class mAP50-95={kendi_m:.6f} (validator ile fark {abs(kendi_m - float(stats['metrics/mAP50-95(B)'])):.2e})")
    np.savez_compressed(
        run_dir / f"{onek}_paket.npz", tp=P.tp, conf=P.conf, pcls=P.pcls, len=P.len,
        tcls=np.concatenate(P.tcls_l) if P.tcls_l else np.zeros(0), tlen=np.array([len(t) for t in P.tcls_l], dtype=np.int64),
        names=np.array(P.names),
    )

    # ---- gecikme (h07 uctan uca protokolu; olcum goruntuleri val.txt'in ilk 350'si) ----
    hiz = {}
    gpu_oncesi = None
    if not a.atla_gecikme:
        import cv2
        gu, _gm = gpu_kullanim()
        gpu_oncesi = gu
        if cuda_ok and gu is not None and gu > 15:
            print(f"[k5] UYARI: gecikme oncesi GPU kullanimi %{gu} (>15): gecikme GUVENILMEZ (k5_gecikme_gpu_mesgul=1)")
        val_txt = WS / "data" / "splits" / "val.txt"
        liste = [ln.strip() for ln in val_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        n_yuk = a.gecikme_isinma + a.gecikme_olcum
        imgs = [cv2.imread(f) for f in liste[:n_yuk]]
        imgs = [i for i in imgs if i is not None]
        motor = h07.Motor(w_path, device, half=cuda_ok)
        dep = dict(cfg, ad=cfg["ad"] + "_dep")
        tur_ms, tur_dep = [], []
        for _ in range(max(1, a.gecikme_tekrar)):
            ms, n_olc = h07.gecikme_olc(motor, cfg, imgs, a.gecikme_isinma, a.gecikme_olcum)
            tur_ms.append(ms)
            ms2, _n = h07.gecikme_olc(motor, dep, imgs, a.gecikme_isinma, a.gecikme_olcum, conf=0.25, multi_label=False)
            tur_dep.append(ms2)
        lat = float(statistics.median(tur_ms))
        lat_dep = float(statistics.median(tur_dep))
        cv = (statistics.pstdev(tur_ms) / statistics.mean(tur_ms)) if len(tur_ms) > 1 else 0.0
        hiz = {
            "latency_toplam_ms": round(lat, 4), "fps": round(1000.0 / lat, 3),
            "latency_dep_ms": round(lat_dep, 4), "fps_dep": round(1000.0 / lat_dep, 3),
            "latency_tur_cv": round(cv, 5),
            "hiz_batch": 1, "hiz_precision_fp16": 1.0 if cuda_ok else 0.0,
            "hiz_olcum_n_goruntu": int(n_olc), "hiz_isinma_n_goruntu": int(a.gecikme_isinma),
            "hiz_tur": len(tur_ms),
            "k5_gecikme_gpu_mesgul": int(bool(cuda_ok and gpu_oncesi is not None and gpu_oncesi > 15)),
        }
        print(f"[k5] gecikme (uctan uca, conf .001) medyan={lat:.3f} ms ({1000 / lat:.1f} FPS); dep(conf .25)={lat_dep:.3f} ms; CV={cv * 100:.2f}%")

    n_p = None
    try:
        from ultralytics import YOLO
        n_p = int(sum(p.numel() for p in YOLO(str(w_path)).model.parameters()))
    except Exception as e:
        print(f"[k5] param_sayisi cikarilamadi: {e}")

    alt_grup = {f"ap50_{k}": x for k, x in per_ap50.items()}
    alt_grup.update({f"ap5095_{k}": x for k, x in per_ap5095.items()})

    s = stats
    metrics = {
        f"{onek}_map50_95": f6(s["metrics/mAP50-95(B)"]),
        f"{onek}_map50": f6(s["metrics/mAP50(B)"]),
        f"{onek}_map75": f6(box.map75),
        f"{onek}_precision": f6(s["metrics/precision(B)"]),
        f"{onek}_recall": f6(s["metrics/recall(B)"]),
        f"{onek}_map_small": f6(s.get("metrics/mAP_small(B)", 0.0)),
        f"{onek}_map_medium": f6(s.get("metrics/mAP_medium(B)", 0.0)),
        f"{onek}_map_large": f6(s.get("metrics/mAP_large(B)", 0.0)),
        f"{onek}_n_goruntu": int(P.n),
        "egitim_sure_sn": round(time.time() - t_basla, 2), "egitim_yok": 1,
        "param_sayisi": n_p, "cihaz": cihaz_str,
        "varyant_adi": a.varyant_adi, "seed": a.seed if a.seed is not None else -1,
        "tta_t1": int(a.tta_t1), "weights": str(w_path), "ultralytics_surum": _ul.__version__,
        "k5_gpu_baslangic_kullanim": float(gpu_u0) if gpu_u0 is not None else -1.0,
        "varyant_imzasi": f"K5 egitimsiz degerlendirme ({'T1 TTA' if a.tta_t1 else 'tek olcek'}), onek={onek}, rakip_kos.degerlendir ile ayni DetectionValidator ayarlari",
        "alt_grup": alt_grup,
    }
    metrics = {k: x for k, x in metrics.items() if x is not None}
    metrics.update(hiz)
    # onek=val iken kilitli veri anahtari (test_/rep_) uretilmemis olmali; onek=test iken yalniz test_ var
    yasak = "test_" if onek == "val" else "val_"
    assert not any(k.startswith((yasak, "rep_")) for k in metrics), "onek tutarsizligi"
    atomik_yaz(out_path, metrics)
    print(f"[k5] metrics.json yazildi: {out_path} :: {onek}_map50_95={metrics[f'{onek}_map50_95']} toplam {time.time() - t_basla:.0f} sn")


if __name__ == "__main__":
    main()
