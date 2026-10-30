# -*- coding: utf-8 -*-
"""
H02 giris noktasi: BboxLoss.forward'i Inner-CIoU (ratio=0.75) ile degistirir, sonra scripts/rakip_kos.py'yi
AYNI CLI argumanlariyla cagirir (metrics.json sozlesmesi ayni). rakip_kos.py'ye / ultralytics'e dokunulmaz.
Kayip ana surecte hesaplanir (DataLoader iscileri yalniz veri uretir) -> Windows spawn'da yama iscilerde gerekmez.

Istege bagli: --inner-ratio R (ablasyon; varsayilan 0.75) - rakip_kos'a iletilmeden once argv'den cikarilir.
  python varyantlar/h02_innerciou/kos.py --seed S --epochs E ... --run-dir {RUN_DIR} --out {RUN_DIR}/metrics.json
"""
import sys
from pathlib import Path

_AD = Path(__file__).resolve().parent
sys.path.insert(0, str(_AD.parent / "_ortak"))
sys.path.insert(0, str(_AD))
import vortak  # noqa: E402
import innerciou_patch as patch  # noqa: E402


def toplayici():
    s = patch.SAYAC
    ek = {
        "varyant_etki_sayaci": int(s["cagri"]),  # yamali BboxLoss.forward EGITIM cagri sayisi (ana surec, grad acik)
        "inner_val_cagri": int(s["cagri_val"]),
        "inner_ratio": float(patch.RATIO),
        "inner_bas_sayisi": len(patch.BASLAR),  # eğitimde yamalı forward'ın çağrıldığı BboxLoss örneği (YOLO26: 2)
    }
    if s["fark_olcum_n"]:
        ek["inner_iou_fark_ort"] = round(s["fark_toplam"] / s["fark_olcum_n"], 6)  # ort (IoU - IoU_inner), orneklenmis
    return ek


if __name__ == "__main__":
    r = vortak.argv_cikar("--inner-ratio")
    oran = patch.uygula(r if r is not None else 0.75)
    sys.exit(vortak.calistir(
        f"H02: BboxLoss IoU terimi CIoU -> Inner-CIoU (ratio={oran}): loss_iou=(1-CIoU)+IoU-IoU_inner; "
        "DFL/TAL/kazanclar/diger her sey baseline ile ayni",
        toplayici))
