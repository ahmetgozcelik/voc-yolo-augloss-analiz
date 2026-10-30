# -*- coding: utf-8 -*-
"""
H03 giris noktasi: ultralytics.data.augment.CopyPaste -> boxcp.BoxCopyPaste (kutu-tabanli kucuk-nesne
copy-paste), sonra scripts/rakip_kos.py AYNI CLI argumanlariyla + `--override copy_paste=0.5` ile cagrilir.
rakip_kos.py'ye / ultralytics'e dokunulmaz.

Windows spawn: ATAMA MODUL IMPORT ZAMANINDA (asagida, `if __name__` DISINDA) yapilir; DataLoader isci
sureclerinde __mp_main__ olarak bu dosya yeniden ice aktarilir -> yama iscilerde de etkindir. Asil
guvence: dataset ana surecte kurulur ve BoxCopyPaste ornekleri dataset ile pickle'lanip iscilere gider
(sinif boxcp modulunde, __main__'de degil).

  python varyantlar/h03_boxcp/kos.py --seed S --epochs E ... --override copy_paste=0.5 \
      --run-dir {RUN_DIR} --out {RUN_DIR}/metrics.json
Hedef bolge: tuvalin merkez penceresi (zone="center"; boxcp.py aciklamasina bakin).
"""
import os
import sys
import tempfile
from pathlib import Path

_AD = Path(__file__).resolve().parent
sys.path.insert(0, str(_AD.parent / "_ortak"))
sys.path.insert(0, str(_AD))
import vortak  # noqa: E402
import boxcp  # noqa: E402

# Import-zamani yama (ana surec + spawn isci __mp_main__ yeniden importu)
boxcp.uygula()

_SAYAC_DIZIN = None


def toplayici():
    if _SAYAC_DIZIN is None:
        return {"varyant_etki_sayaci": 0}
    t = boxcp.sayac_topla(_SAYAC_DIZIN)
    return {
        "varyant_etki_sayaci": int(t["yapistirilan"]),       # yapistirilan toplam nesne (yaklasik: <=50*isci eksik olabilir)
        "boxcp_uygulanan_goruntu": int(t["uygulanan_goruntu"]),
        "boxcp_cagri": int(t["cagri"]),
        "boxcp_aday_toplam": int(t["aday_toplam"]),
        "boxcp_yer_bulunamadi": int(t["yer_bulunamadi"]),
        "boxcp_surec_sayisi": int(t["surec_sayisi"]),
    }


def main():
    global _SAYAC_DIZIN
    if not any(a.replace(" ", "") == "copy_paste=0.5" for a in sys.argv):
        sys.exit("[h03] HATA: komutta '--override copy_paste=0.5' yok; bu varyant o override ile kosar")
    zone = "center"
    run_dir = Path(vortak.argv_deger("--run-dir", "."))
    _SAYAC_DIZIN = str(run_dir.resolve() / "boxcp_sayac")
    os.makedirs(_SAYAC_DIZIN, exist_ok=True)
    for f in Path(_SAYAC_DIZIN).glob("*.json"):
        f.unlink()
    os.environ["BOXCP_SAYAC_DIR"] = _SAYAC_DIZIN  # isciler miras alir
    return vortak.calistir(
        f"H03: kutu-tabanli kucuk-nesne copy-paste (p=0.5, 1-3 nesne, sqrt(alan)<=48, IoA>0.1 ele, cakismasiz, "
        f"yeniden olcekleme yok, 3px yumusatma, hedef bolge={zone}); mosaic sonrasi/RandomPerspective oncesi; "
        "baska her sey baseline ile ayni",
        toplayici)


if __name__ == "__main__":
    sys.exit(main())
