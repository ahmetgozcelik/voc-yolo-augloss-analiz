# -*- coding: utf-8 -*-
"""
H04 sarmalayici: YALNIZ damga + gozlem. Egitimi degistiren tek sey komut satirindaki
`--override scale=0.25` (rakip_kos.py -> model.train(scale=0.25)); bu dosya hicbir egitim davranisini
degistirmez ve RNG tuketmez. Gozlem: DetectionTrainer.preprocess_batch sonrasi, egitim veri kumesinin
transform zincirindeki RandomPerspective.scale degeri okunur (ana surecteki dataset nesnesi) ->
scale'in gercekten 0.25 oldugunun kaniti (varyant_etki_sayaci = RandomPerspective.scale==0.25 olan
egitim batch'i sayisi; trainer.args.scale != 0.25 ise 0 kalir -> beklenti kapisi IHLAL verir).

Kullanim (kuyruk satiri):
  python varyantlar/h04_scale25/kos.py --seed S --epochs E ... --override scale=0.25 \
      --run-dir {RUN_DIR} --out {RUN_DIR}/metrics.json
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_ortak"))
import vortak  # noqa: E402

GOZLEM = Counter()  # (args_scale, rp_scale) -> batch sayisi (yalniz ana surec)
_PATCHED = False


def rp_scale_bul(transforms):
    """Ic ice Compose zincirinde RandomPerspective.scale degerini bulur (yoksa None)."""
    if transforms is None:
        return None
    if type(transforms).__name__ == "RandomPerspective":
        return float(transforms.scale)
    for t in getattr(transforms, "transforms", []) or []:
        r = rp_scale_bul(t)
        if r is not None:
            return r
    return None


def yama_uygula():
    global _PATCHED
    if _PATCHED:
        return
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    orig = DetectionTrainer.preprocess_batch

    def gozlemli(self, batch):
        b = orig(self, batch)
        try:
            rp = rp_scale_bul(getattr(self.train_loader.dataset, "transforms", None))
            GOZLEM[(round(float(self.args.scale), 4), None if rp is None else round(rp, 4))] += 1
        except Exception:
            GOZLEM[("hata", None)] += 1
        return b

    DetectionTrainer.preprocess_batch = gozlemli
    _PATCHED = True


def toplayici():
    tutan = sum(n for (a, r), n in GOZLEM.items() if a == 0.25 and r == 0.25)
    toplam = sum(GOZLEM.values())
    return {
        "varyant_etki_sayaci": int(tutan),          # args.scale==0.25 VE RandomPerspective.scale==0.25 olan egitim batch'i
        "scale_batch_sayisi": int(toplam),
        "scale_args": float(max((a for (a, r) in GOZLEM if a != "hata"), default=-1.0)),
        "scale_rp_farkli_deger_sayisi": int(len({r for (a, r) in GOZLEM})),
    }


if __name__ == "__main__":
    if not any(a.replace(" ", "") == "scale=0.25" for a in sys.argv):
        sys.exit("[h04] HATA: komutta '--override scale=0.25' yok; bu varyant o override ile kosar")
    yama_uygula()
    sys.exit(vortak.calistir(
        "H04: mosaic/afin olcek jitter scale=0.25 (baseline 0.5): RandomPerspective olcek carpani U[0.75,1.25] "
        "(baseline U[0.5,1.5]); multi_scale/mosaic/mixup/close_mosaic/kayip kazanclari ve diger her sey baseline ile ayni",
        toplayici))
