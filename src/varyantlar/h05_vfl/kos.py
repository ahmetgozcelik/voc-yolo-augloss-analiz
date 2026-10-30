# -*- coding: utf-8 -*-
"""
H05 giris noktasi: v8DetectionLoss cls terimini BCE -> Varifocal agirlikli BCE (alpha=0.75, gamma=2) yapar, sonra
scripts/rakip_kos.py'yi AYNI CLI argumanlariyla cagirir (metrics.json sozlesmesi ayni). rakip_kos.py'ye /
ultralytics'e dokunulmaz. Kayip ana surecte hesaplanir (DataLoader iscileri yalniz veri uretir).

Istege bagli (ablasyon): --vfl-alpha A, --vfl-gamma G (varsayilan 0.75 / 2.0), --vfl-poz q|1 (varsayilan q;
  1 = H05-neg: pozitifler agirliksiz) - rakip_kos'a iletilmeden cikarilir.
  python varyantlar/h05_vfl/kos.py --seed S --epochs E ... --run-dir {RUN_DIR} --out {RUN_DIR}/metrics.json
"""
import sys
from pathlib import Path

_AD = Path(__file__).resolve().parent
sys.path.insert(0, str(_AD.parent / "_ortak"))
sys.path.insert(0, str(_AD))
import vortak  # noqa: E402
import vfl_patch as patch  # noqa: E402


def toplayici():
    s = patch.SAYAC
    ek = {
        "varyant_etki_sayaci": int(s["cagri"]),  # VFL modulu EGITIM cagri sayisi (ana surec, grad acik)
        "vfl_val_cagri": int(s["cagri_val"]),
        "vfl_alpha": float(patch.ALPHA),
        "vfl_gamma": float(patch.GAMMA),
        "vfl_poz_agirlik": patch.POZ,
        "vfl_bas_sayisi": len(patch.BASLAR),  # eğitimde VFL'nin çağrıldığı kayıp örneği sayısı (YOLO26: 2)
    }
    if s["grad_olcum_n"]:
        n = s["grad_olcum_n"]
        ek["vfl_grad_norm_ort"] = round(s["grad_vfl_toplam"] / n, 6)
        ek["bce_grad_norm_ort"] = round(s["grad_bce_toplam"] / n, 6)  # ayni (p,t) icin BCE gradyan normu (ornekleme)
        if s["grad_bce_toplam"] > 0:
            ek["vfl_bce_grad_orani"] = round(s["grad_vfl_toplam"] / s["grad_bce_toplam"], 6)
        if s["kayip_bce_toplam"] > 0:
            ek["vfl_bce_kayip_orani"] = round(s["kayip_vfl_toplam"] / s["kayip_bce_toplam"], 6)
        ek["vfl_olcum_n"] = int(n)
    return ek


if __name__ == "__main__":
    a = vortak.argv_cikar("--vfl-alpha")
    g = vortak.argv_cikar("--vfl-gamma")
    pz = vortak.argv_cikar("--vfl-poz")
    alpha, gamma = patch.uygula(a, g, pz)
    sys.exit(vortak.calistir(
        f"H05: cls kaybi BCE -> Varifocal agirlikli BCE (alpha={alpha}, gamma={gamma}, poz={patch.POZ}): "
        f"w=alpha*sigmoid(p)^gamma*(1-y)+{'q*y' if patch.POZ == 'q' else 'y'}, "
        "y=1[q>0], q=TAL target_scores; normalizasyon (target_scores_sum) ve cls=0.5 kazanci, TAL, kutu/DFL kayiplari ve "
        "diger her sey baseline ile ayni",
        toplayici))
