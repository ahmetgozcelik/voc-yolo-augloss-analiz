# -*- coding: utf-8 -*-
"""
H08 giris noktasi (H03b, yuksek doz): ultralytics.data.augment.CopyPaste -> boxcp_doz.BoxCopyPasteDoz
(H03 BoxCopyPaste, max_sqrt_alan=96), sonra scripts/rakip_kos.py AYNI CLI argumanlariyla +
`--override copy_paste=1.0` ile cagrilir. rakip_kos.py'ye, ultralytics'e ve h03_boxcp'ye dokunulmaz.
Windows spawn: atama modul import zamaninda (if __name__ disinda), sinif boxcp_doz modulunde.

  python varyantlar/h08_boxcp_doz/kos.py --seed S --epochs E ... --override copy_paste=1.0 \
      [--override close_mosaic=0] --run-dir {RUN_DIR} --out {RUN_DIR}/metrics.json
"""
import os
import sys
from pathlib import Path

_AD = Path(__file__).resolve().parent
sys.path.insert(0, str(_AD.parent / "_ortak"))
sys.path.insert(0, str(_AD))
import vortak  # noqa: E402
import boxcp_doz  # noqa: E402

boxcp_doz.uygula()

_SAYAC_DIZIN = None
_P = 1.0


def toplayici():
    if _SAYAC_DIZIN is None:
        return {"varyant_etki_sayaci": 0}
    t = boxcp_doz.sayac_topla(_SAYAC_DIZIN)
    cagri = max(int(t["cagri"]), 1)
    return {
        "varyant_etki_sayaci": int(t["yapistirilan"]),
        "boxcp_uygulanan_goruntu": int(t["uygulanan_goruntu"]),
        "boxcp_cagri": int(t["cagri"]),
        "boxcp_aday_toplam": int(t["aday_toplam"]),
        "boxcp_yer_bulunamadi": int(t["yer_bulunamadi"]),
        "boxcp_surec_sayisi": int(t["surec_sayisi"]),
        # H08 doz sayaclari (kart: doz = yapistirilan nesne / mosaic cagrisi; H03 referansi 0.142)
        "boxcp_doz_nesne_per_mosaic": t["yapistirilan"] / cagri,
        "boxcp_uygulanan_pay": t["uygulanan_goruntu"] / cagri,
        "boxcp_yer_bulunamadi_pay": t["yer_bulunamadi"] / max(int(t["aday_toplam"]), 1),
        # varyant_imzasi dogrulamasi icin sayisal parametre kopyalari
        "boxcp_p": float(_P),
        "boxcp_max_sqrt_alan": float(boxcp_doz.MAX_SQRT_ALAN),
    }


def main():
    global _SAYAC_DIZIN, _P
    if not any(a.replace(" ", "") == "copy_paste=1.0" for a in sys.argv):
        sys.exit("[h08] HATA: komutta '--override copy_paste=1.0' yok; bu varyant o override ile kosar "
                 "(H03'un 0.5'i ile karistirma)")
    zone = "center"
    run_dir = Path(vortak.argv_deger("--run-dir", "."))
    _SAYAC_DIZIN = str(run_dir.resolve() / "boxcp_sayac")
    os.makedirs(_SAYAC_DIZIN, exist_ok=True)
    for f in Path(_SAYAC_DIZIN).glob("*.json"):
        f.unlink()
    os.environ["BOXCP_SAYAC_DIR"] = _SAYAC_DIZIN
    return vortak.calistir(
        f"H08: YUKSEK DOZLU kutu-tabanli copy-paste (p={_P}, max_sqrt_alan={boxcp_doz.MAX_SQRT_ALAN:g}, 1-3 nesne, "
        f"IoA>0.1 ele, cakismasiz, yeniden olcekleme yok, 3px yumusatma, hedef bolge={zone}); H03 ile fark yalniz "
        "p 0.5->1.0 ve max_sqrt_alan 48->96; mosaic sonrasi/RandomPerspective oncesi; baska her sey baseline ile ayni",
        toplayici)


if __name__ == "__main__":
    sys.exit(main())
