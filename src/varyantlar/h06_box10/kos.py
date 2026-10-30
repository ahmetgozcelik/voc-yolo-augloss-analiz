# -*- coding: utf-8 -*-
"""
H06 sarmalayici: YALNIZ damga + gozlem. Egitimi degistiren tek sey komut satirindaki
`--override box=10.0` (rakip_kos.py -> model.train(box=10.0)); bu dosya hicbir egitim davranisini degistirmez
ve RNG tuketmez. Gozlem: v8DetectionLoss.get_assigned_targets_and_loss sarilir (sonuc aynen dondurulur); egitimde
(grad acik) cagri basina self.hyp.box okunur -> kayip kazancinin gercekten 10.0 oldugunun kaniti
(varyant_etki_sayaci = hyp.box==10.0 iken egitim kaybi cagrisi sayisi; ana surec, DataLoader iscilerinde degil).
Kart: trainer.args.box != 10.0 ise kart gecersiz -> sayac 0 kalir, beklenti kapisi IHLAL verir.

Kullanim (kuyruk satiri):
  python varyantlar/h06_box10/kos.py --seed S --epochs E ... --override box=10.0 \
      --run-dir {RUN_DIR} --out {RUN_DIR}/metrics.json
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_ortak"))
import vortak  # noqa: E402

GOZLEM = Counter()  # (hyp.box, hyp.cls, hyp.dfl) -> egitim kayip cagrisi
VAL_CAGRI = [0]
_PATCHED = False


def yama_uygula():
    global _PATCHED
    if _PATCHED:
        return
    import torch
    from ultralytics.utils.loss import v8DetectionLoss
    orig = v8DetectionLoss.get_assigned_targets_and_loss

    def gozlemli(self, preds, batch):
        sonuc = orig(self, preds, batch)
        if torch.is_grad_enabled():
            GOZLEM[(float(self.hyp.box), float(self.hyp.cls), float(self.hyp.dfl))] += 1
        else:
            VAL_CAGRI[0] += 1
        return sonuc

    v8DetectionLoss.get_assigned_targets_and_loss = gozlemli
    _PATCHED = True


def toplayici():
    tutan = sum(n for (b, c, d), n in GOZLEM.items() if b == 10.0 and c == 0.5 and d == 1.5)
    if GOZLEM:
        (b, c, d), _ = GOZLEM.most_common(1)[0]
    else:
        b = c = d = -1.0
    return {
        "varyant_etki_sayaci": int(tutan),  # hyp.box==10.0 (cls 0.5, dfl 1.5 sabit) egitim kayip cagrisi
        "box_gain": float(b), "cls_gain": float(c), "dfl_gain": float(d),
        "box_egitim_kayip_cagri": int(sum(GOZLEM.values())),
        "box_val_kayip_cagri": int(VAL_CAGRI[0]),
    }


if __name__ == "__main__":
    if not any(a.replace(" ", "") in ("box=10", "box=10.0") for a in sys.argv):
        sys.exit("[h06] HATA: komutta '--override box=10.0' yok; bu varyant o override ile kosar")
    yama_uygula()
    sys.exit(vortak.calistir(
        "H06: kutu kaybi kazanci box=10.0 (baseline 7.5); cls=0.5, dfl=1.5, CIoU bicimi, TAL ve diger her sey baseline ile ayni",
        toplayici))
