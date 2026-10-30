#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rakip_kos.py — Metrik köprüsü (K3): rakip Ultralytics YOLO11n (pinli ultralytics==8.4.165,
commit 004db34b12f65fa5a160de2b1090f6943106bd35, bkz. WS/rakip/) çağırır ve sonucu standart
koşum sözleşmesine (deney-sozlesmesi.md §K3) uygun {RUN_DIR}/metrics.json'a yazar. RAKİP KODUNA
DOKUNULMAZ (R2) — yalnız `ultralytics` paketinin genel API'si (YOLO, DetectionValidator)
kullanılır.

Kullanım (kosum.py ile, {RUN_DIR} yer tutucusu kosum.py tarafından çözülür):
  python WS/scripts/rakip_kos.py --seed 0 --epochs 100 --imgsz 640 --batch 16 --workers 8 \
      --run-dir {RUN_DIR} --out {RUN_DIR}/metrics.json [--rep-test] \
      [--fraction 0.1] [--override mosaic=0.0] [--override close_mosaic=0]

Windows: spawn ile çok süreçli veri yükleyici kullanıldığı için tüm çalışma kodu
`if __name__ == "__main__":` koruması ALTINDA çağrılır (ana giriş noktası budur).

Yazılan dosyalar ({RUN_DIR} altında):
  - metrics.json      : düz sözlük (val_*, egitim_sure_sn, cihaz, alt_grup — deney-sozlesmesi §K3)
  - progress.json     : her epoch sonunda {"adim","epoch","loss","val_map50_95",...} —
                        kosum.py --kural canlı erken durdurma bunu okur (kosum-izleme.md K6.4)
  - val_preds.json    : best.pt'nin voc_dev.yaml val kümesi üzerindeki tahminleri (COCO-json
                        biçiminde bbox/score/category_id/file_name) — K3.5 hata analizi girdisi
  - test_kilitli.json : yalnız --rep-test verilirse; kilitli VOC07 test sonucu (ayrı, denetlenebilir
                        kopya). AYRICA metrics.json'a `rep_test_*` anahtarlarıyla da yazılır —
                        bu genel "test metriğini K5'e kadar yazma" kuralının kasıtlı, dokümante
                        edilmiş bir istisnasıdır: orkestrasyon_durumu.md "Şef protokol kararları"
                        → "Baseline best.pt 07 test'te YALNIZ replikasyon için bir kez ölçülür →
                        metrics anahtarı rep_test_*". Bu anahtar yalnız `--rep-test` bayrağıyla
                        (yalnız baseline koşumlarında, K4 varyantlarında ASLA) üretilir ve K4
                        seçim metriği HER ZAMAN val_map50_95'tir (replikasyon_kapisi.py bunu okur).
  - rep_test_preds.json : yalnız --rep-test; kilitli test tahminleri.

Sınıf başı AP50 / AP50-95 metrics.json'ın "alt_grup" alanına (tek seviye iç içe, sözleşmeye
uygun) `ap50_<sinif>` / `ap5095_<sinif>` anahtarlarıyla yazılır; --rep-test ile `rep_ap50_<sinif>` /
`rep_ap5095_<sinif>` eklenir.

Hız ölçümü: batch=1, FP16, 50 ısınma + 300 ölçüm görüntüsü (data/splits/val.txt'den, sırayla),
preprocess/inference/postprocess(NMS dahil) ayrı ms/görüntü + toplam + fps (Ultralytics
predictor.speed, bkz. notlar/literatur-cipa.md §3.7).
"""
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# WS/scripts/rakip_kos.py -> WS kökü
WS_KOK = Path(__file__).resolve().parents[1]


def atomik_yaz(yol: Path, veri):
    yol.parent.mkdir(parents=True, exist_ok=True)
    tmp = yol.with_suffix(yol.suffix + ".tmp")
    tmp.write_text(json.dumps(veri, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, yol)


def parse_override(pairs):
    """--override key=val listesini ultralytics train()/val() kwargs'ına çevirir."""
    out = {}
    for kv in pairs or []:
        if "=" not in kv:
            print(f"[rakip_kos] UYARI: --override '{kv}' yok sayıldı (key=val bekleniyor)")
            continue
        k, v = kv.split("=", 1)
        k, v = k.strip(), v.strip()
        low = v.lower()
        if low in ("true", "false"):
            val = low == "true"
        elif low in ("none", "null"):
            val = None
        else:
            try:
                val = int(v)
            except ValueError:
                try:
                    val = float(v)
                except ValueError:
                    val = v
        out[k] = val
    return out


def read_lines(p: Path):
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def temizle_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0, help="train alt kümesi oranı (duman/ön-uçuş)")
    ap.add_argument("--data", default=None, help="varsayılan: WS/data/voc_dev.yaml")
    ap.add_argument("--test-data", default=None, help="varsayılan: WS/data/voc_test.yaml (yalnız --rep-test)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=None, help="varsayılan: WS/rakip/weights/yolo11n.pt")
    ap.add_argument("--override", action="append", default=[],
                    help="ultralytics train() ek override, key=val (tekrarlanabilir; K4 varyantları kullanır)")
    ap.add_argument("--rep-test", action="store_true",
                    help="best.pt'yi kilitli VOC07 test'te BİR KEZ ölç (yalnız baseline replikasyonu — şef kararı)")
    ap.add_argument("--out", default=None, help="metrics.json hedefi (varsayılan {run-dir}/metrics.json)")
    ap.add_argument("--hiz-txt", default=None,
                    help="hız ölçümü görüntü listesi (varsayılan: WS/data/splits/val.txt = VOC val; ek deney: KITTI val)")
    a = ap.parse_args()

    run_dir = Path(a.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(a.out).resolve() if a.out else run_dir / "metrics.json"

    data_path = Path(a.data).resolve() if a.data else (WS_KOK / "data" / "voc_dev.yaml")
    test_data_path = Path(a.test_data).resolve() if a.test_data else (WS_KOK / "data" / "voc_test.yaml")
    model_path = Path(a.model).resolve() if a.model else (WS_KOK / "rakip" / "weights" / "yolo11n.pt")

    print(f"[rakip_kos] WS_KOK={WS_KOK}")
    print(f"[rakip_kos] data={data_path} test_data={test_data_path} model={model_path} run_dir={run_dir}")
    print(f"[rakip_kos] seed={a.seed} epochs={a.epochs} imgsz={a.imgsz} batch={a.batch} "
          f"fraction={a.fraction} workers={a.workers} rep_test={a.rep_test} override={a.override}")

    import torch  # geç import
    cuda_ok = torch.cuda.is_available()
    device_str = f"cuda:0 / {torch.cuda.get_device_name(0)}" if cuda_ok else "cpu"
    device_arg = 0 if cuda_ok else "cpu"
    print(f"[rakip_kos] cihaz: {device_str} (torch {torch.__version__}, cuda derleme {torch.version.cuda})")
    if not cuda_ok:
        # Sessiz CPU düşüşü YASAK (hesaplama-kaynagi.md) — açıkça yazılıyor; kosum.py --gpu-zorunlu
        # zaten bu koşumu hiç başlatmamalıydı; burada da cihaz metrics.json'a doğru yazılacak.
        print("[rakip_kos] UYARI: CUDA YOK. Bu koşum CPU'da sürecek (10-100x yavaş). "
              "--gpu-zorunlu ile başlatılmadıysa kontrol edin.")

    import random
    random.seed(a.seed)
    try:
        import numpy as np
        np.random.seed(a.seed)
    except Exception:
        pass
    torch.manual_seed(a.seed)

    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    import ultralytics as _ul

    progress_path = run_dir / "progress.json"

    def on_fit_epoch_end(trainer):
        """5-10 satırlık progress yazıcı — rakip koduna DOKUNMADAN, callback API ile (R2)."""
        try:
            m = trainer.metrics or {}
            tloss = trainer.tloss
            loss_val = None
            if tloss is not None:
                try:
                    loss_val = float(tloss.mean().item()) if hasattr(tloss, "mean") else float(sum(tloss) / len(tloss))
                except Exception:
                    loss_val = None
            prog = {
                "adim": int(trainer.epoch) + 1,
                "epoch": int(trainer.epoch) + 1,
                "epochs_total": int(trainer.epochs),
                "loss": loss_val,
                "val_map50_95": m.get("metrics/mAP50-95(B)"),
                "val_map50": m.get("metrics/mAP50(B)"),
                "val_precision": m.get("metrics/precision(B)"),
                "val_recall": m.get("metrics/recall(B)"),
                "epoch_time_sn": float(trainer.epoch_time) if trainer.epoch_time else None,
                "zaman": datetime.now().isoformat(timespec="seconds"),
            }
            atomik_yaz(progress_path, prog)
        except Exception as e:
            print(f"[rakip_kos] progress yazım hatası (yok sayıldı, eğitim etkilenmedi): {e}")

    # ---- Eğitim ----
    model = YOLO(str(model_path))
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    overrides = parse_override(a.override)
    train_kwargs = dict(
        data=str(data_path), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers,
        fraction=a.fraction, seed=a.seed, deterministic=True, cache=False, exist_ok=True,
        amp=True, val=True, plots=False, project=str(run_dir), name="train", device=device_arg,
        verbose=True,
    )
    train_kwargs.update(overrides)
    print(f"[rakip_kos] train_kwargs={train_kwargs}")

    t0 = time.time()
    model.train(**train_kwargs)
    # Ek deney sağlık kapısı: Ultralytics OOM'da batch'i sessizce yarılar (trainer.py:558-591) → son batch kaydı
    try:
        batch_son = int(getattr(model.trainer, "batch_size", a.batch))
    except Exception:
        batch_son = None
    gpu_bellek_max_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3) if cuda_ok else None
    egitim_sure_sn = time.time() - t0
    print(f"[rakip_kos] eğitim bitti: {egitim_sure_sn:.1f} sn")

    train_dir = run_dir / "train"
    best_pt = train_dir / "weights" / "best.pt"
    if not best_pt.exists():
        last_pt = train_dir / "weights" / "last.pt"
        print(f"[rakip_kos] UYARI: best.pt yok ({best_pt}); last.pt kullanılıyor: {last_pt}")
        best_pt = last_pt

    # ---- results.csv'den epoch süresi + en iyi epoch ----
    results_csv = train_dir / "results.csv"
    epoch_sn_ort, best_epoch = None, None
    if results_csv.exists():
        import csv
        with open(results_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        zamanlar = []
        for r in rows:
            v = r.get("time")
            if v not in (None, ""):
                try:
                    zamanlar.append(float(v))
                except ValueError:
                    pass
        if len(zamanlar) >= 2:
            diffs = [b - a2 for a2, b in zip(zamanlar, zamanlar[1:]) if b > a2]
            if diffs:
                epoch_sn_ort = sum(diffs) / len(diffs)
        elif len(zamanlar) == 1:
            epoch_sn_ort = zamanlar[0]
        if rows:
            key95 = next((k for k in rows[0] if "mAP50-95" in k), None)
            if key95:
                try:
                    vals = [float(r[key95]) for r in rows]
                    best_epoch = int(vals.index(max(vals)) + 1)
                except Exception:
                    pass

    # ---- Değerlendirme: best.pt ile voc_dev.yaml val (DetectionValidator DOĞRUDAN — S/M/L kırılımı
    #      için `stats` dict'i almak gerekiyor; model.val() bunu döndürmüyor, yalnız DetMetrics
    #      döndürüyor. Bu, rakip koduna dokunmadan genel API'nin kullanılmasıdır (R2). ----
    def degerlendir(yaml_yolu: Path, ad: str):
        val_args = dict(
            data=str(yaml_yolu), split="val", imgsz=a.imgsz, batch=a.batch, device=device_arg,
            plots=False, save_json=True, project=str(run_dir), name=ad, exist_ok=True, verbose=False,
            model=str(best_pt),
        )
        validator = DetectionValidator(args=val_args)
        stats = validator(model=str(best_pt))
        per_ap50, per_ap5095 = {}, {}
        try:
            names = validator.metrics.names
            for i, c in enumerate(validator.metrics.ap_class_index):
                p, r, ap50, ap = validator.metrics.class_result(i)
                cname = names[int(c)]
                per_ap50[cname] = round(float(ap50), 6)
                per_ap5095[cname] = round(float(ap), 6)
        except Exception as e:
            print(f"[rakip_kos] {ad}: sınıf başı AP çıkarılamadı: {e}")
        return stats, per_ap50, per_ap5095, validator.save_dir

    dev_stats, dev_ap50, dev_ap5095, dev_save_dir = degerlendir(data_path, "val_dev")

    val_preds_src = dev_save_dir / "predictions.json"
    val_preds_dst = run_dir / "val_preds.json"
    if val_preds_src.exists():
        shutil.copy(val_preds_src, val_preds_dst)
        print(f"[rakip_kos] val_preds.json yazıldı: {val_preds_dst}")
    else:
        print(f"[rakip_kos] UYARI: {val_preds_src} bulunamadı, val_preds.json yazılamadı")

    # ---- Hız ölçümü: batch=1, 640, FP16, 50 ısınma + 300 ölçüm (data/splits/val.txt) ----
    val_txt = Path(a.hiz_txt).resolve() if a.hiz_txt else (WS_KOK / "data" / "splits" / "val.txt")
    hiz = {}
    try:
        val_list = read_lines(val_txt)
        warmup_imgs = val_list[:50]
        measure_imgs = val_list[50:350] if len(val_list) >= 350 else val_list[50:]
        speed_model = YOLO(str(best_pt))
        if warmup_imgs:
            list(speed_model.predict(source=warmup_imgs, imgsz=a.imgsz, batch=1, half=cuda_ok,
                                      device=device_arg, verbose=False, save=False, stream=True))
        if measure_imgs:
            list(speed_model.predict(source=measure_imgs, imgsz=a.imgsz, batch=1, half=cuda_ok,
                                      device=device_arg, verbose=False, save=False, stream=True))
            sp = speed_model.predictor.speed  # {'preprocess','inference','postprocess'} ms/görüntü
            toplam_ms = sum(sp.values())
            hiz = {
                "latency_preprocess_ms": round(float(sp.get("preprocess", 0.0)), 4),
                "latency_inference_ms": round(float(sp.get("inference", 0.0)), 4),
                "latency_postprocess_nms_ms": round(float(sp.get("postprocess", 0.0)), 4),
                "latency_toplam_ms": round(float(toplam_ms), 4),
                "fps": round(float(1000.0 / toplam_ms), 3) if toplam_ms > 0 else None,
                "hiz_olcum_n_goruntu": len(measure_imgs),
                "hiz_isinma_n_goruntu": len(warmup_imgs),
                "hiz_precision_fp16": 1.0 if cuda_ok else 0.0,
                "hiz_batch": 1,
            }
    except Exception as e:
        print(f"[rakip_kos] hız ölçümü başarısız (yok sayıldı): {e}")

    n_p, gflops = None, None
    try:
        info = model.model.info(detailed=False, verbose=True, imgsz=a.imgsz)
        if info:
            _n_l, n_p, _n_g, gflops = info
    except Exception as e:
        print(f"[rakip_kos] model.info başarısız: {e}")

    alt_grup = {}
    for k, v in dev_ap50.items():
        alt_grup[f"ap50_{k}"] = v
    for k, v in dev_ap5095.items():
        alt_grup[f"ap5095_{k}"] = v

    metrics = temizle_none({
        "val_map50_95": round(float(dev_stats.get("metrics/mAP50-95(B)", float("nan"))), 6),
        "val_map50": round(float(dev_stats.get("metrics/mAP50(B)", float("nan"))), 6),
        "val_precision": round(float(dev_stats.get("metrics/precision(B)", float("nan"))), 6),
        "val_recall": round(float(dev_stats.get("metrics/recall(B)", float("nan"))), 6),
        "val_fitness": round(float(dev_stats.get("fitness", float("nan"))), 6) if "fitness" in dev_stats else None,
        "val_map_small": dev_stats.get("metrics/mAP_small(B)"),
        "val_map_medium": dev_stats.get("metrics/mAP_medium(B)"),
        "val_map_large": dev_stats.get("metrics/mAP_large(B)"),
        "batch_son": batch_son,
        "gpu_bellek_max_gb": gpu_bellek_max_gb,
        "egitim_sure_sn": round(egitim_sure_sn, 2),
        "egitim_sure_sa": round(egitim_sure_sn / 3600.0, 4),
        "epoch_sn_ort": round(epoch_sn_ort, 2) if epoch_sn_ort else None,
        "en_iyi_epoch": best_epoch,
        "epochs_calisilan": a.epochs,
        "seed": a.seed,
        "param_sayisi": int(n_p) if n_p is not None else None,
        "gflops": round(float(gflops), 3) if gflops is not None else None,
        "cihaz": device_str,
        "ultralytics_surum": _ul.__version__,
        "results_csv": str(results_csv),
        "best_pt": str(best_pt),
        "alt_grup": alt_grup,
    })
    metrics.update(temizle_none(hiz))

    # ---- Yalnız --rep-test: kilitli VOC07 test, BİR KEZ (şef kararı) ----
    if a.rep_test:
        rep_stats, rep_ap50, rep_ap5095, rep_save_dir = degerlendir(test_data_path, "rep_test")
        for k, v in rep_ap50.items():
            metrics["alt_grup"][f"rep_ap50_{k}"] = v
        for k, v in rep_ap5095.items():
            metrics["alt_grup"][f"rep_ap5095_{k}"] = v
        rep_dict = temizle_none({
            "rep_test_map50_95": round(float(rep_stats.get("metrics/mAP50-95(B)", float("nan"))), 6),
            "rep_test_map50": round(float(rep_stats.get("metrics/mAP50(B)", float("nan"))), 6),
            "rep_test_precision": round(float(rep_stats.get("metrics/precision(B)", float("nan"))), 6),
            "rep_test_recall": round(float(rep_stats.get("metrics/recall(B)", float("nan"))), 6),
            "rep_test_map_small": rep_stats.get("metrics/mAP_small(B)"),
            "rep_test_map_medium": rep_stats.get("metrics/mAP_medium(B)"),
            "rep_test_map_large": rep_stats.get("metrics/mAP_large(B)"),
        })
        test_lock = dict(rep_dict)
        test_lock["rep_test_per_class_ap50"] = rep_ap50
        test_lock["rep_test_per_class_ap50_95"] = rep_ap5095
        test_lock["not"] = ("Şef kararı (orkestrasyon_durumu.md): yalnız baseline, yalnız bir kez. "
                             "K4 varyantlarında bu bayrak KULLANILMAZ; seçim metriği val_map50_95'tir.")
        atomik_yaz(run_dir / "test_kilitli.json", test_lock)
        metrics.update(rep_dict)  # şef kararınca replikasyon kapısı bu anahtarları metrics.json'da bekliyor

        rep_preds_src = rep_save_dir / "predictions.json"
        rep_preds_dst = run_dir / "rep_test_preds.json"
        if rep_preds_src.exists():
            shutil.copy(rep_preds_src, rep_preds_dst)

    atomik_yaz(out_path, metrics)
    ozet = {k: metrics[k] for k in ("val_map50_95", "val_map50", "egitim_sure_sn", "cihaz") if k in metrics}
    print(f"[rakip_kos] metrics.json yazıldı: {out_path} :: {ozet}")


if __name__ == "__main__":
    main()
