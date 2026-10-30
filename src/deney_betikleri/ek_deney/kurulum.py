"""Ek deney ortam kurulumu (Windows / Linux).

WS kökünden çalıştırılır:
    python ek_deney/kurulum.py              # tam kurulum
    python ek_deney/kurulum.py --kontrol    # yalnız doğrulama (kurulum yapmaz)

Adımlar (idempotent, tekrar çalıştırılabilir):
  1. env/.venv  : eğitim ortamı (torch + requirements.lock, ilk deneyle aynı sürümler)
  2. env/.venv-oi : Open Images indirme ortamı (fiftyone; eğitim ortamından ayrı)
  3. rakip/ultralytics : pinli referans klonu (v8.4.165 = 004db34)
  4. rakip/weights : yolo11n.pt (sha256 doğrulamalı) + yolo11s.pt (sha256 kaydedilir)
  5. data/ içindeki eski makine yollarını bu makinenin WS köküne çevirir
  6. GPU / CUDA / ultralytics doğrulaması → env/kurulum-ek-deney.md
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
WIN = os.name == "nt"
ESKI_KOK = "<WS>"
UL_TAG, UL_COMMIT = "v8.4.165", "004db34b12f65fa5a160de2b1090f6943106bd35"
TORCH, TV = "2.14.0", "0.29.0"
AGIRLIK = {
    "yolo11n.pt": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "yolo11s.pt": None,  # ek deneyde ilk kez indirilir; hash rapora yazılır
}
AGIRLIK_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/{}"
RAPOR = []


def log(s):
    print(s, flush=True)
    RAPOR.append(s)


def kos(cmd, **kw):
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def venv_py(ad):
    return WS / "env" / ad / ("Scripts/python.exe" if WIN else "bin/python")


def venv_kur(ad):
    py = venv_py(ad)
    if not py.exists():
        kos([sys.executable, "-m", "venv", WS / "env" / ad])
    kos([py, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    return py


def gpu_cc():
    """nvidia-smi'den compute capability (ör. 12.0) ve kart adı."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
        ad, cc, mem, drv = [x.strip() for x in out.splitlines()[0].split(",")]
        return ad, float(cc), mem, drv
    except Exception as e:  # noqa: BLE001
        return None, None, None, str(e)


def torch_index(cc):
    # CUDA 13 tekerlekleri sm_75 ve üstünü destekler (RTX 20xx ve sonrası). Daha eski kart → cu126.
    return "cu130" if cc is None or cc >= 7.5 else "cu126"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def egitim_ortami(index):
    py = venv_kur(".venv")
    try:
        kos([py, "-m", "pip", "install", "-q", f"torch=={TORCH}", f"torchvision=={TV}",
             "--index-url", f"https://download.pytorch.org/whl/{index}"])
    except subprocess.CalledProcessError:
        log(f"UYARI: torch {TORCH}/{index} bulunamadı; aynı index'in en güncel sürümü kuruluyor (SAPMA, rapora işlenir)")
        kos([py, "-m", "pip", "install", "-q", "torch", "torchvision",
             "--index-url", f"https://download.pytorch.org/whl/{index}"])
    kilit = [l.strip() for l in (WS / "env/requirements.lock").read_text().splitlines()
             if l.strip() and not l.startswith(("torch==", "torchvision=="))]
    gecici = WS / "env/_kilit_torchsuz.txt"
    gecici.write_text("\n".join(kilit) + "\n")
    kos([py, "-m", "pip", "install", "-q", "-r", gecici])
    gecici.unlink()
    return py


def oi_ortami():
    py = venv_kur(".venv-oi")
    kos([py, "-m", "pip", "install", "-q", "fiftyone==1.22.1", "pandas", "pyyaml", "pillow", "imagehash"])
    return py


def ultralytics_klon():
    hedef = WS / "rakip/ultralytics"
    if not (hedef / ".git").exists():
        kos(["git", "clone", "--depth", "1", "--branch", UL_TAG,
             "https://github.com/ultralytics/ultralytics.git", hedef])
    head = subprocess.run(["git", "-C", hedef, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    log(f"ultralytics klon HEAD={head} {'EŞLEŞTİ' if head == UL_COMMIT else 'UYUŞMAZLIK!'}")
    if head != UL_COMMIT:
        raise SystemExit("rakip/ultralytics pinli commit değil")


def agirliklar():
    d = WS / "rakip/weights"
    d.mkdir(parents=True, exist_ok=True)
    for ad, beklenen in AGIRLIK.items():
        p = d / ad
        if not p.exists():
            log(f"indiriliyor {ad}")
            urllib.request.urlretrieve(AGIRLIK_URL.format(ad), p)
        h = sha256(p)
        (d / f"{ad}.sha256").write_text(h + "\n")
        durum = "kayıt" if beklenen is None else ("EŞLEŞTİ" if h == beklenen else "UYUŞMAZLIK!")
        log(f"{ad} sha256={h} {durum}")
        if beklenen and h != beklenen:
            raise SystemExit(f"{ad} hash uyuşmazlığı")


def yol_duzelt():
    """data/*.yaml ve data/splits/*.txt içindeki eski makine kökünü bu makinenin WS köküne çevirir."""
    yeni = WS.as_posix()
    n = 0
    for p in list((WS / "data").glob("*.yaml")) + list((WS / "data/splits").glob("*.txt")):
        s = p.read_text(encoding="utf-8")
        if ESKI_KOK in s:
            p.write_text(s.replace(ESKI_KOK, yeni), encoding="utf-8")
            n += 1
    log(f"yol düzeltme: {n} dosya ({ESKI_KOK} → {yeni})")


def dogrula():
    py = venv_py(".venv")
    kod = ("import torch, torchvision, ultralytics; c=torch.cuda.is_available(); "
           "x=torch.nn.Conv2d(3,16,3).cuda()(torch.randn(2,3,64,64).cuda()).shape if c else None; "
           "print(torch.__version__, torchvision.__version__, ultralytics.__version__, 'cuda', c, "
           "torch.cuda.get_device_name(0) if c else '-', x)")
    out = subprocess.run([py, "-c", kod], capture_output=True, text=True)
    log("doğrulama: " + (out.stdout.strip() or out.stderr.strip()[-500:]))
    pyoi = venv_py(".venv-oi")
    if pyoi.exists():
        o2 = subprocess.run([pyoi, "-c", "import fiftyone; print('fiftyone', fiftyone.__version__)"],
                            capture_output=True, text=True)
        log("oi ortamı: " + (o2.stdout.strip() or o2.stderr.strip()[-300:]))
    return out.returncode == 0 and "cuda True" in out.stdout


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--kontrol", action="store_true", help="yalnız doğrulama")
    ap.add_argument("--torch-index", default=None, help="cu130 / cu126 (varsayılan: GPU'ya göre)")
    a = ap.parse_args()
    ad, cc, mem, drv = gpu_cc()
    disk = shutil.disk_usage(WS).free / 1e9
    log(f"# Ek deney kurulum raporu — {datetime.now():%Y-%m-%d %H:%M}")
    log(f"OS {platform.platform()} · Python {platform.python_version()} · GPU {ad} cc={cc} {mem} sürücü {drv} · boş disk {disk:.0f} GB")
    if disk < 60:
        log("UYARI: boş disk < 60 GB (OI-VOC20 + VOC + koşumlar için önerilen alt sınır)")
    if not a.kontrol:
        idx = a.torch_index or torch_index(cc)
        log(f"torch index: {idx}")
        egitim_ortami(idx)
        oi_ortami()
        ultralytics_klon()
        agirliklar()
        yol_duzelt()
    ok = dogrula()
    log(f"SONUÇ: {'HAZIR' if ok else 'HATA — CUDA/torch doğrulaması geçmedi'}")
    (WS / "env/kurulum-ek-deney.md").write_text("\n\n".join(RAPOR) + "\n", encoding="utf-8")
    (WS / "env/kurulum-ek-deney.json").write_text(json.dumps(
        {"gpu": ad, "cc": cc, "vram": mem, "surucu": drv, "disk_gb": round(disk), "hazir": ok}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
