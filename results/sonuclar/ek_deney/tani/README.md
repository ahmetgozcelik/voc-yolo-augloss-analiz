# Hakem tur 4 ucuz denetimleri (test kilidine dokunmaz) — 2026-10-19
1. YOLO26n B s0 (20261019-124541) ağırlıkları, train bölmesinden seed 0 ile çekilen 1000 görüntüde (`train1000.txt`, `train1000_degerlendir.py`; Ultralytics val, DontCare'siz):
   best.pt mAP50-95 0.5606 (mAP50 0.8336) | last.pt 0.7842 (0.9609). Aynı ağırlıklar val'de 0.1789 / ≈0.153.
   Sınıf başı (best): car .704 van .688 truck .732 ped .396 sit .408 cyc .382 tram .577 misc .598.
   → train ≫ val: veri/etiket hattı çalışıyor; düşük val, sahne-ayrık genelleme açığı ve aşırı uyumdur (best→last train +22 puan, val −3 puan).
2. Gerçek kutu çizimi (`gt_ciz.py`, 3 val + 3 train görüntüsü, `gt_kutu_ornek.jpg`): kutular nesnelerle hizalı, sınıf adları doğru.
