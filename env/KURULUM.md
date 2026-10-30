# Ortam Kurulumu — 2.1 PASCAL VOC (hedef A: bu makine, GPU)

Kurulum tarihi: 2026-09-29 · Makine: <makine> (Windows 10, i5-12400F, 15.9 GB RAM, RTX 5050 8 GB, sürücü 610.47)
Ölçüm: `env/ortam-yerel.md` (ortam_raporu.py rc=0)

## Komutlar (WS kökünden, Git Bash)
```bash
py -3.11 -m venv env/.venv                       # uv yok; sistem Python 3.11.0 ile venv
V=env/.venv/Scripts/python.exe
$V -m pip install --upgrade pip
$V -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
$V -m pip install ultralytics pycocotools ensemble-boxes scipy pandas matplotlib
$V -m pip freeze > env/requirements.lock
```

## Doğrulama çıktısı
```
torch 2.14.0+cu130 tv 0.29.0+cu130 ultralytics 8.4.165 cuda True conv ok (64, 16, 318, 318)
```
RTX 5050 = Blackwell (sm_120); cu130 tekerleği sm_120 içeriyor. cu121 gibi eski index KULLANILMAZ (sm_120 yok).

## Windows notları
- `PYTHONIOENCODING=utf-8`; yollar tırnak içinde.
- Ultralytics dataloader Windows'ta spawn kullanır → eğitim betikleri `if __name__ == "__main__":` korumalı olmalı; `workers` ≤ 6 (12 mantıksal çekirdek, 16 GB RAM).
- `cache=ram` KULLANILMAZ (VOC ~16.5k görüntü RAM'i doldurur); `cache=False` veya `cache=disk`.
- 8 GB VRAM: YOLO11n/s 640px batch 16–32 sığar; RT-DETR-R50 bu kartta pratik değil (rakip olarak YOLO11 seçilmesi önerilir).
