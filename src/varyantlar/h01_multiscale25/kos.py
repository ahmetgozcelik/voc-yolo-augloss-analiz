# -*- coding: utf-8 -*-
"""
H01 sarmalayici: YALNIZ damga + gozlem. Egitimi degistiren tek sey komut satirindaki
`--override multi_scale=0.25` (rakip_kos.py -> model.train(multi_scale=0.25)); bu dosya hicbir egitim
davranisini degistirmez. Gozlem: DetectionTrainer.preprocess_batch sonrasi batch boyutu kaydedilir
(RNG tuketmez) -> multi_scale'in gercekten devrede oldugunun kaniti (480-800 px arasi degisen boyutlar).

Kullanim (kuyruk satiri):
  python varyantlar/h01_multiscale25/kos.py --seed S --epochs E ... --override multi_scale=0.25 \
      --run-dir {RUN_DIR} --out {RUN_DIR}/metrics.json
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_ortak"))
import vortak  # noqa: E402

BOYUT = Counter()  # (H,W) -> batch sayisi (yalniz ana surec; preprocess_batch ana surecte calisir)
_PATCHED = False


def yama_uygula():
    global _PATCHED
    if _PATCHED:
        return
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    orig = DetectionTrainer.preprocess_batch

    def gozlemli(self, batch):
        b = orig(self, batch)
        BOYUT[tuple(int(x) for x in b["img"].shape[-2:])] += 1
        return b

    DetectionTrainer.preprocess_batch = gozlemli
    _PATCHED = True


def toplayici():
    if not BOYUT:
        return {"varyant_etki_sayaci": 0}
    kenarlar = [max(h, w) for (h, w) in BOYUT]
    ms_disi = sum(n for (h, w), n in BOYUT.items() if max(h, w) != 640)
    return {
        "varyant_etki_sayaci": int(ms_disi),          # 640 disi boyutta islenen egitim batch'i sayisi
        "ms_batch_sayisi": int(sum(BOYUT.values())),
        "ms_boyut_min": int(min(kenarlar)),
        "ms_boyut_max": int(max(kenarlar)),
        "ms_farkli_boyut_sayisi": int(len(BOYUT)),
    }


if __name__ == "__main__":
    if not any(a.replace(" ", "") == "multi_scale=0.25" for a in sys.argv):
        sys.exit("[h01] HATA: komutta '--override multi_scale=0.25' yok; bu varyant o override ile kosar")
    yama_uygula()
    sys.exit(vortak.calistir(
        "H01: multi_scale=0.25 (baseline 0.0) - batch boyutu 480-800 px arasi rastgele (32'nin kati); "
        "baska her sey baseline ile ayni; dogrulama 640 px",
        toplayici))
