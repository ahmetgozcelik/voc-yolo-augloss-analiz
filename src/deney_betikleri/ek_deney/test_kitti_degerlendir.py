"""kitti_degerlendir.py birim testleri (sentetik görüntü + etiket). Çalıştır: python scripts/ek_deney/test_kitti_degerlendir.py"""
import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from kitti_degerlendir import degerlendir  # noqa: E402


def kur(kok):
    (kok / "images").mkdir(parents=True)
    (kok / "labels").mkdir(parents=True)
    p = kok / "images/000001.png"
    Image.new("RGB", (1000, 400)).save(p)
    # GT: sınıf 0, piksel kutu (100,100)-(200,300)
    (kok / "labels/000001.txt").write_text("0 0.15 0.5 0.1 0.5\n")
    return [str(p)]


def test():
    with tempfile.TemporaryDirectory() as t:
        liste = kur(Path(t))
        dogru = {"image_id": 1, "file_name": "000001.png", "category_id": 1, "bbox": [100, 100, 100, 200], "score": 0.9}
        yanlis_dc = {"image_id": 1, "file_name": "000001.png", "category_id": 1, "bbox": [700, 100, 100, 100], "score": 0.95}
        dc = {"000001": [[0.65, 0.2, 0.85, 0.6]]}  # piksel (650,80)-(850,240) → yanlis_dc'nin IoA'sı 1.0

        r = degerlendir([dogru], liste, None, nc=1)
        assert abs(r["map50_95"] - 1.0) < 1e-6, r
        # DontCare içindeki yanlış-pozitif (daha yüksek skor) kural yokken AP'yi düşürür, kuralla atılır
        r0 = degerlendir([dogru, yanlis_dc], liste, None, nc=1)
        r1 = degerlendir([dogru, yanlis_dc], liste, dc, nc=1)
        assert r0["map50_95"] < 0.99, r0
        assert r1["dontcare_yok_sayilan_iou50"] == 1 and abs(r1["map50_95"] - 1.0) < 1e-6, r1
        # Doğru tespit DontCare bölgesiyle örtüşse bile ATILMAZ (önce gerçek nesneyle eşleşir)
        dc_gt = {"000001": [[0.05, 0.2, 0.25, 0.8]]}  # GT kutusunu (100,100)-(200,300) tamamen kapsar
        r2 = degerlendir([dogru], liste, dc_gt, nc=1)
        assert abs(r2["map50_95"] - 1.0) < 1e-6 and r2["dontcare_yok_sayilan_iou50"] == 0, r2
        # DontCare dışındaki yanlış-pozitif yok sayılmaz
        uzak = dict(yanlis_dc, bbox=[400, 300, 50, 50])
        r3 = degerlendir([dogru, uzak], liste, dc, nc=1)
        assert r3["dontcare_yok_sayilan_iou50"] == 0, r3
    print("test_kitti_degerlendir: 5/5 GEÇTİ (DontCare = coco-crowd yok-say)")


if __name__ == "__main__":
    test()
