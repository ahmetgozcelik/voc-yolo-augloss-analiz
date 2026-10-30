"""Ek deney sağlık kapısı (ON-KAYIT-ek-deney.md "Sağlık kapısı"). WS kökünden:
    python scripts/ek_deney/saglik_kapisi.py [--desen "ek_yolo*"] [--tam]
Her koşum için: metrics.json var; batch_son == 16 ve stdout.log'da "Reducing to batch" yok; H02/H05 başlık sayacı
(yolo26n: 2, yolo11s / VOC yolo11n: 1); H03/H08 boxcp_cagri > 0; --tam ise: VOC koşumlarında en iyi epoch >= 20,
KITTI (_ek_) koşumlarında ON-KAYIT v5b ölçütü (ogrenme_denetle; en iyi epoch < 20 yalnız uyarı).
Çıktı: satır başına GEÇTİ/KALDI + nedenler; rc = kalan koşum sayısı (0 = hepsi geçti).
"""
import argparse
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]


def denetle(rd: Path, tam: bool):
    neden = []
    mp = rd / "metrics.json"
    if not mp.exists():
        return ["metrics.json yok"]
    m = json.loads(mp.read_text(encoding="utf-8"))
    ad = rd.name
    if m.get("batch_son") not in (None, 16):
        neden.append(f"batch_son={m.get('batch_son')}")
    if m.get("batch_son") is None:
        neden.append("batch_son kaydı yok (eski koşucu)")
    log = rd / "stdout.log"
    if log.exists() and "Reducing to batch" in log.read_text(encoding="utf-8", errors="replace"):
        neden.append("stdout: Reducing to batch")
    beklenen_bas = 2 if "yolo26n" in ad else 1
    if "h02_innerciou" in ad and m.get("inner_bas_sayisi") != beklenen_bas:
        neden.append(f"inner_bas_sayisi={m.get('inner_bas_sayisi')} (beklenen {beklenen_bas})")
    if ("h05_vfl" in ad or "h05pos" in ad or "h05neg" in ad) and m.get("vfl_bas_sayisi") != beklenen_bas:
        neden.append(f"vfl_bas_sayisi={m.get('vfl_bas_sayisi')} (beklenen {beklenen_bas})")
    if ("h03_boxcp" in ad or "h08_boxcp" in ad) and not (m.get("boxcp_cagri") or 0) > 0:
        neden.append("boxcp_cagri=0")
    if tam and "_ek_" not in ad and (m.get("en_iyi_epoch") or 0) < 20:
        neden.append(f"en_iyi_epoch={m.get('en_iyi_epoch')} < 20")
    if tam and "_ek_" in ad:  # KITTI: ON-KAYIT v5/v5b madde 1 (en iyi epoch yalnız uyarı)
        neden += ogrenme_denetle(rd, m)
    return neden


def _csv(rd: Path):
    import csv
    rc = rd / "train/results.csv"
    if not rc.exists():
        return None
    return [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(rc.open(encoding="utf-8"))]


def _b_s0(rd: Path):
    """Aynı modelin tamamlanmış tam B s0 koşumu (LR planı referansı)."""
    m = "yolo26n" if "_ek_yolo26n_" in rd.name else "yolo11s"
    for p in sorted((WS / "runs").glob(f"*_ek_{m}_baseline_s0"), reverse=True):
        d = p / ".done"
        if d.exists() and json.loads(d.read_text(encoding="utf-8")).get("rc") == 0 \
                and json.loads((p / "config.json").read_text(encoding="utf-8")).get("asama") == 3:
            return p
    return None


def ogrenme_denetle(rd: Path, m: dict):
    """ON-KAYIT v5b (hakem tur 4 Ö2) geçerlilik ölçütü, sonuçtan bağımsız:
    results.csv tam 100 satır; tüm kayıp/mAP sütunları sonlu; LR sütunları aynı modelin B s0'ı ile epoch başına aynı;
    ep81–90 ort. eğitim (box+cls) kaybı < ep11–20 ort.; en iyi val mAP50-95 >= 0.02."""
    import math
    satir = _csv(rd)
    if satir is None:
        return ["results.csv yok"]
    neden = []
    if len(satir) != 100:
        neden.append(f"results.csv {len(satir)} satır (100 değil)")
    sut = [k for k in satir[0] if k.endswith("_loss") or k.startswith("metrics/")]
    if any(not math.isfinite(float(r[k])) for r in satir for k in sut):
        neden.append("NaN/inf")
    elif len(satir) >= 90:
        kayip = [float(r["train/box_loss"]) + float(r["train/cls_loss"]) for r in satir]
        if not sum(kayip[80:90]) < sum(kayip[10:20]):
            neden.append("eğitim kaybı ep81–90 ≥ ep11–20")
    b = _b_s0(rd)
    if b is not None and b != rd:
        ref = _csv(b)
        lr = [k for k in satir[0] if k.startswith("lr/")]
        if ref is None or len(ref) != len(satir) or any(
                abs(float(r[k]) - float(q[k])) > 1e-9 + 1e-6 * abs(float(q[k])) for r, q in zip(satir, ref) for k in lr):
            neden.append(f"LR planı B s0 ({b.name}) ile aynı değil")
    if (m.get("val_map50_95") or 0) < 0.02:
        neden.append(f"val_map50_95={m.get('val_map50_95')} < 0.02")
    return neden


def uyari(rd: Path):
    """Raporlanan (geçersizlik değil) uyarılar."""
    mp = rd / "metrics.json"
    if "_ek_" not in rd.name or not mp.exists():
        return []
    e = json.loads(mp.read_text(encoding="utf-8")).get("en_iyi_epoch") or 0
    return [f"uyarı: en_iyi_epoch={e} < 20"] if e < 20 else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desen", default="*_ek_*")
    ap.add_argument("--tam", action="store_true", help="tam koşum ölçütleri (en iyi epoch >= 20)")
    a = ap.parse_args()
    kalan = 0
    for rd in sorted((WS / "runs").glob(a.desen)):
        d = rd / ".done"
        if not d.exists():
            continue
        if json.loads(d.read_text(encoding="utf-8")).get("rc") != 0:
            continue
        n = denetle(rd, a.tam)
        kalan += bool(n)
        print(f"{'KALDI' if n else 'GEÇTİ'}  {rd.name}  {'; '.join(n + (uyari(rd) if a.tam else []))}")
    sys.exit(kalan)


if __name__ == "__main__":
    main()
