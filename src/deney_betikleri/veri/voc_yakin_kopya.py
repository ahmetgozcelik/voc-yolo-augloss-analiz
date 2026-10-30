"""Yakın-kopya taraması (train∪val × test): 32x32 gri küçük resim, sıfır-ortalama birim-norm vektör,
kosinüs (=NCC). Ayrıca yatay-ayna versiyonu. Eşikler: NCC>=0.95 (aday), >=0.98 (güçlü aday).
Test yalnız istatistik amaçlı okunur. Sonuç data/sizinti_raporu.json'a 'ncc_*' anahtarlarıyla eklenir.
Kullanım: env/.venv/Scripts/python.exe scripts/veri/voc_yakin_kopya.py
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path(__file__).resolve().parents[2]
D = WS / "data"
def load(lst):
    return [l.strip() for l in (D / "splits" / lst).read_text(encoding="utf-8").splitlines() if l.strip()]
def vec(p, flip=False):
    im = Image.open(p); im.draft("L", (64, 64))
    a = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32)
    if flip: a = a[:, ::-1]
    a = a.ravel() - a.mean(); n = np.linalg.norm(a)
    return a / n if n > 1e-6 else a
pool = load("train.txt") + load("val.txt"); test = load("test.txt")
P = np.stack([vec(p) for p in pool]); T = np.stack([vec(p) for p in test])
Tf = np.stack([vec(p, True) for p in test])
best = np.full(len(pool), -1.0, np.float32); arg = np.zeros(len(pool), int); fl = np.zeros(len(pool), bool)
for i in range(0, len(pool), 2000):
    S = P[i:i+2000] @ T.T; Sf = P[i:i+2000] @ Tf.T
    m, mf = S.max(1), Sf.max(1)
    use_f = mf > m
    best[i:i+2000] = np.where(use_f, mf, m)
    arg[i:i+2000] = np.where(use_f, Sf.argmax(1), S.argmax(1)); fl[i:i+2000] = use_f
rel = lambda s: s.split("/data/VOC/")[1]
det = [{"pool": rel(pool[k]), "test": rel(test[arg[k]]), "ncc": round(float(best[k]), 4), "flipped": bool(fl[k])}
       for k in np.argsort(-best)[:15]]
res = {"ncc_method": "32x32 gri, sıfır-ortalama birim-norm, kosinüs; yatay ayna dahil; her havuz görüntüsü için en yakın test",
       "ncc_ge_0.95": int((best >= 0.95).sum()), "ncc_ge_0.98": int((best >= 0.98).sum()),
       "ncc_ge_0.99": int((best >= 0.99).sum()), "ncc_top15": det,
       "ncc_pool_best_quantiles": {q: round(float(np.quantile(best, q)), 4) for q in (0.5, 0.99, 0.999)}}
r = json.loads((D / "sizinti_raporu.json").read_text(encoding="utf-8")); r.update(res)
(D / "sizinti_raporu.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))
