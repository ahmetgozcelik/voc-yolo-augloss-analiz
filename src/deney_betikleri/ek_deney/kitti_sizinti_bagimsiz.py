"""Bölmeler arası bağımsız sızıntı istatistikleri (hakem tur 2–3; kitti_hazirla.py'den SONRA koşulur). WS kökünden:
    python scripts/ek_deney/kitti_sizinti_bagimsiz.py
NCC (96×29 gri) bölmeler arası en yüksek değer / %99'luk / ≥0.90 sayısı; dHash (8×8) Hamming ≤ 2 çiftleri ve NCC'leri.
Yazar: data/kitti/bolme_raporu.json → "sizinti_bagimsiz"; data/kitti/dhash_ciftleri.json.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

D = Path(__file__).resolve().parents[2] / "data/kitti"


def nv(p):
    a = np.asarray(Image.open(p).convert("L").resize((96, 29), Image.BILINEAR), dtype=np.float32).ravel()
    a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


def dh(p, n=8):
    a = np.asarray(Image.open(p).convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def main():
    S = {b: (D / f"splits/{b}.txt").read_text().split() for b in ("train", "val", "test")}
    V = {b: np.stack([nv(p) for p in S[b]]) for b in S}
    H = {b: np.stack([dh(p) for p in S[b]]) for b in S}
    out, ciftler = {}, []
    for a, b in (("test", "train"), ("val", "train"), ("test", "val")):
        C = V[a] @ V[b].T
        m = C.max(1)
        x = np.unpackbits(H[a][:, None, :] ^ H[b][None, :, :], axis=-1).sum(-1)
        ii, jj = np.nonzero(x <= 2)
        for i, j in zip(ii, jj):
            ciftler.append({"a": a, "b": b, "ia": Path(S[a][i]).stem, "ib": Path(S[b][j]).stem,
                            "ham": int(x[i, j]), "ncc": round(float(C[i, j]), 3)})
        out[f"{a}-{b}"] = {"ncc_max": round(float(m.max()), 3), "ncc_p99": round(float(np.quantile(m, 0.99)), 3),
                           "ncc_ge090": int((m >= 0.90).sum()), "dhash_le2_cift": int(len(ii)),
                           "dhash_ciftleri_ncc_max": round(float(max([C[i, j] for i, j in zip(ii, jj)], default=0)), 3)}
    r = json.loads((D / "bolme_raporu.json").read_text(encoding="utf-8"))
    r["sizinti_bagimsiz"] = out
    (D / "bolme_raporu.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    ciftler.sort(key=lambda c: -c["ncc"])
    (D / "dhash_ciftleri.json").write_text(json.dumps(ciftler, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
