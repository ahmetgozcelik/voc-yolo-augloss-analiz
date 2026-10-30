"""On-eleme 'sans' olasiliklarinin yontem denetimi (kilit sonrasi, kesifsel). build_evidence.py L1052-1085'teki Monte Carlo'yu AYNI rng tohumu ve
girdilerle (evidence.json dval.map50_95.<kart>.sd) yeniden uretir ve tek bir duyarlilik ekler: kabul kuralindaki '>=2 seed Delta>0' kosulu."""
import json, math
from pathlib import Path
import numpy as np
from scipy import stats

E = {r["anahtar"]: r["deger"] for r in json.loads((Path(__file__).parents[1] / "evidence.json").read_text(encoding="utf-8"))["kayitlar"]}
ELENEN = ["h01", "h02", "h04", "h05", "h06", "h08"]
T90 = stats.t.ppf(0.90, 2)
NMC = 400000
rng = np.random.default_rng(20261003)  # build_evidence.py ile ayni tohum ve cizim sirasi
pi_a = pi_b = 1.0
sat = {}
for a in ELENEN:
    s = E[f"dval.map50_95.{a}.sd"]
    x = rng.normal(0.0, s, (NMC, 3))
    m, sd = x.mean(1), x.std(1, ddof=1)
    t_ok = m / (sd / math.sqrt(3)) > T90
    acc_a = float(np.mean((m >= 0.40) & t_ok))                      # build_evidence.py kurali (>=2 seed kosulu YOK)
    acc_b = float(np.mean((m >= 0.40) & t_ok & ((x > 0).sum(1) >= 2)))  # + ontanimli '>=2 seed Delta>0' kosulu
    pi_a *= 1 - acc_a
    pi_b *= 1 - acc_b
    sat[a] = {"sd": round(s, 4), "kabul_kodun_kurali": acc_a, "kabul_2seed_kosuluyla": acc_b}
cik = {"nmc": NMC, "rng_tohum": 20261003, "kart": sat, "P_6_6_kodun_kurali": pi_a, "P_6_6_2seed_kosuluyla": pi_b,
       "kayitli_E2560": E["onelem.sans_isabet_6_6.mc"]}
print(json.dumps(cik, indent=1))
