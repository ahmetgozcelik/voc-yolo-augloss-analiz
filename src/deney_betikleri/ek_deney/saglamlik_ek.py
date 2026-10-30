"""KITTI test sağlamlık analizleri (kilit SONRASI, keşifsel; etiket: kilit_sonrasi_kesifsel).
Dış değerlendirmenin istediği analizler, MEVCUT test tahminlerinden (yeni test açılışı yok, yeni eğitim yok, GPU yok).
Kaynak: sonuclar/ek_deney/test/<run_id>/test_preds.json (best.pt) ve test_preds_last.json (last.pt),
        data/kitti/{splits/test.txt, dontcare.json, test_gruplari.json}, değerlendirici = analiz_ek.py + kitti_degerlendir.py.
Kullanım (WS kökünden):
    python scripts/ek_deney/saglamlik_ek.py hesapla [--isci 6]   # koşum başına AP dizilerini önbelleğe yazar (yeniden başlatılabilir)
    python scripts/ek_deney/saglamlik_ek.py topla                 # önbellekten sonuclar/ek_deney/saglamlik-analiz.{json,md}
Yöntem:
  * Her koşum için (görüntü-başı COCO eşleşmeleri bir kez; analiz_ek.eval_kayitlari) sınıf-başı AP (8) hesaplanır:
    tam örneklem, B=2000 grup-çekimi, tek-grup (13) ve bir-grup-dışarıda (13). AP = COCO 101 nokta, DontCare=crowd.
  * Hiyerarşik bootstrap: her çoğaltmada (j) (i) tohumlar 3 tohumdan yerine koyarak çekilir, (ii) 13 test sahne grubu yerine koyarak
    çekilir (grup çekimi tüm tohum/kartlar için AYNI: eşli). Δ_j = çekilen tohumların (kart − B) AP farkı ortalaması (puan).
    %95 GA = yüzdelik [2.5, 97.5]; P(Δ<=0) = çoğaltmalarda Δ_j <= 0 oranı. B=2000, rng tohumu sabit.
  * Metrik alt kümeleri: 8 sınıf; tram'sız 7 (tram=idx 6 çıkar); ps+misc'siz 6 (person_sitting=4, misc=7 çıkar).
    (Not: değerlendiricideki eski 'map50_95_7sinif' person_sitting'i çıkarır; burada 'tramsiz7' ayrı bir tanımdır.)
"""
import argparse
import sys as _s
_s.stdout.reconfigure(encoding="utf-8", errors="replace")
import json
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import analiz_ek as ae  # noqa: E402
import kitti_degerlendir as kd  # noqa: E402

T = WS / "sonuclar/ek_deney/test"
TMP = WS / "sonuclar/ek_deney/saglamlik_tmp"
OUT = WS / "sonuclar/ek_deney"
B_BOOT = 2000
SEED_GRUP = 20261024
SEED_TOH = 20261025
SINIF = ["car", "van", "truck", "pedestrian", "person_sitting", "cyclist", "tram", "misc"]
ALT = {"s8": list(range(8)), "tramsiz7": [0, 1, 2, 3, 4, 5, 7], "ps_misc_siz6": [0, 1, 2, 3, 5, 6]}
KARTLAR = ["H01", "H02", "H03", "H04", "H05", "H06", "H08"]
REC = np.linspace(0, 1, 101)


def kosumlar():
    liste = []
    for f in sorted(T.glob("*.json")):
        if f.name == "ACILDI.json":
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        m = "yolo26n" if "_ek_yolo26n_" in r["run_id"] else "yolo11s"
        kart = next(k for k, v in ae.VARY.items() if r["varyant"] == f"ek_{m}_{v}")
        liste.append({"run_id": r["run_id"], "model": m, "kart": kart, "seed": int(r["seed"]),
                      "best": r["dontcare"]["map50_95"], "last": r["last"]["dontcare"]["map50_95"]})
    return liste


def ap_sinif(prep, w):
    """Sınıf başına AP (IoU 0.50:0.95 ortalaması); w = görüntü başına çekilme sayısı. Sonuç (8,), npos=0 -> NaN.
    analiz_ek.ap_agirlikli ile sayısal olarak aynı (eşik ekseninde vektörleştirilmiş)."""
    out = np.full(8, np.nan)
    for k in range(1, 9):
        im, tp, ig, npos_i = prep[k]
        npos = float((npos_i * w).sum())
        if npos == 0:
            continue
        wd = w[im]
        sel = wd > 0
        if not sel.any():
            out[k - 1] = 0.0
            continue
        tp_s = tp[:, sel]
        ww = wd[sel][None, :] * (~ig[:, sel])
        tps = np.cumsum(ww * tp_s, axis=1)
        fps = np.cumsum(ww * ~tp_s, axis=1)
        rc = tps / npos
        pr = tps / np.maximum(tps + fps, np.spacing(1))
        pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
        toplam = 0.0
        for t in range(10):
            ind = np.searchsorted(rc[t], REC, side="left")
            ok = ind < pr.shape[1]
            q = np.zeros(101)
            q[ok] = pr[t, ind[ok]]
            toplam += q.mean()
        out[k - 1] = toplam / 10
    return out


def agirliklar():
    grup = ae.test_gruplari()
    gruplar = sorted(set(grup.values()))
    liste = (WS / "data/kitti/splits/test.txt").read_text().split()
    sira = [Path(p).stem for p in liste]
    gidx = {g: np.array([i for i, s in enumerate(sira) if grup[s] == g]) for g in gruplar}
    rng = np.random.default_rng(SEED_GRUP)
    cekim = np.zeros((B_BOOT, len(sira)))
    for j in range(B_BOOT):
        for g in rng.choice(gruplar, size=len(gruplar), replace=True):
            cekim[j, gidx[g]] += 1
    tek = np.zeros((len(gruplar), len(sira)))
    loo = np.ones((len(gruplar), len(sira)))
    for i, g in enumerate(gruplar):
        tek[i, gidx[g]] = 1
        loo[i, gidx[g]] = 0
    return liste, sira, gruplar, cekim, tek, loo


def isle(arg):
    spec, ck = arg
    yol = TMP / f"{spec['run_id']}_{ck}.npz"
    if yol.exists():
        return spec["run_id"], ck, "onbellek"
    liste, sira, gruplar, cekim, tek, loo = agirliklar()
    dc = json.loads((WS / "data/kitti/dontcare.json").read_text())
    dosya = "test_preds.json" if ck == "best" else "test_preds_last.json"
    kay = ae.eval_kayitlari(json.loads((T / spec["run_id"] / dosya).read_text()), liste, dc)
    prep = ae.hazirla(kay, sira)
    tam = ap_sinif(prep, np.ones(len(sira)))
    boot = np.array([ap_sinif(prep, w) for w in cekim])
    grp = np.array([ap_sinif(prep, w) for w in tek])
    lo = np.array([ap_sinif(prep, w) for w in loo])
    np.savez_compressed(yol, tam=tam, boot=boot, grp=grp, loo=lo, kayitli_map=spec[ck], map_tam=np.nanmean(tam))
    return spec["run_id"], ck, f"map_tam={np.nanmean(tam):.6f} kayitli={spec[ck]:.6f}"


def hesapla(isci):
    from multiprocessing import Pool
    TMP.mkdir(parents=True, exist_ok=True)
    isler = [(s, ck) for s in kosumlar() for ck in ("best", "last")]
    t0 = time.time()
    with Pool(isci) as p:
        for i, r in enumerate(p.imap_unordered(isle, isler), 1):
            print(f"[{i}/{len(isler)} {time.time() - t0:.0f}s] {r}", flush=True)


def altkume(ap, idx):
    """ap[..., 8] -> sınıf alt kümesi ortalaması (NaN'ları yok sayar), eksen -1."""
    with np.errstate(all="ignore"):
        return np.nanmean(ap[..., idx], axis=-1)


def topla():
    liste = kosumlar()
    veri = {}
    for s in liste:
        for ck in ("best", "last"):
            z = np.load(TMP / f"{s['run_id']}_{ck}.npz")
            veri[(s["model"], s["kart"], s["seed"], ck)] = {k: z[k] for k in ("tam", "boot", "grp", "loo")} | {
                "kayitli": float(z["kayitli_map"]), "map_tam": float(z["map_tam"])}
    # yeniden üretim denetimi: önbellekteki tam-örneklem mAP == kayıtlı test mAP50-95
    fark = max(abs(v["map_tam"] - v["kayitli"]) for v in veri.values())
    liste_, sira, gruplar, cekim, tek, loo = agirliklar()
    n_img_g = [int(tek[i].sum()) for i in range(len(gruplar))]
    # grup başına sınıf sayıları (YOLO etiketleri)
    sayim = np.zeros((len(gruplar), 8), int)
    grup = ae.test_gruplari()
    gi = {g: i for i, g in enumerate(gruplar)}
    for p in liste_:
        lp = Path(str(p).replace("images", "labels")).with_suffix(".txt")
        if lp.exists():
            for l in lp.read_text().splitlines():
                if l.strip():
                    sayim[gi[grup[Path(p).stem]], int(l.split()[0])] += 1
    rng = np.random.default_rng(SEED_TOH)
    secim = rng.integers(0, 3, size=(B_BOOT, 3))  # her çoğaltma için tohum çekimi (yerine koyarak)
    sonuc = {"yontem": {"B": B_BOOT, "rng_grup": SEED_GRUP, "rng_tohum": SEED_TOH, "grup_sayisi": len(gruplar),
                        "yeniden_uretim_max_fark_map": fark, "etiket": "kilit_sonrasi_kesifsel",
                        "alt_kumeler": {k: [SINIF[i] for i in v] for k, v in ALT.items()}},
             "gruplar": {g: {"goruntu": n_img_g[i], "sinif_sayilari": dict(zip(SINIF, map(int, sayim[i])))}
                         for i, g in enumerate(gruplar)}, "sonuc": {}}
    for m in ("yolo11s", "yolo26n"):
        for ck in ("best", "last"):
            for k in KARTLAR:
                ent = {}
                for ad, idx in ALT.items():
                    def ms(anahtar, s, kart):
                        return altkume(veri[(m, kart, s, ck)][anahtar], idx)
                    dtam = np.array([100 * (ms("tam", s, k) - ms("tam", s, "B")) for s in range(3)])
                    dboot = np.array([100 * (ms("boot", s, k) - ms("boot", s, "B")) for s in range(3)])  # (3,B)
                    j = np.arange(B_BOOT)
                    hier = np.nanmean(np.stack([dboot[secim[:, c], j] for c in range(3)]), axis=0)
                    grup_only = np.nanmean(dboot, axis=0)
                    # yalnız tohum bootstrap'ı (grup sabit): tam örneklemde tohum çekimi
                    toh_only = np.array([np.mean(dtam[secim[b]]) for b in range(B_BOOT)])
                    q = lambda x: [round(float(np.nanquantile(x, 0.025)), 3), round(float(np.nanquantile(x, 0.975)), 3)]
                    e = {"delta_ort": round(float(dtam.mean()), 3), "delta_seed": [round(float(x), 3) for x in dtam],
                         "hier_ga95": q(hier), "hier_p_le0": round(float(np.mean(hier <= 0)), 4),
                         "hier_ort": round(float(np.nanmean(hier)), 3),
                         "grup_ga95": q(grup_only), "grup_p_le0": round(float(np.mean(grup_only <= 0)), 4),
                         "tohum_ga95": q(toh_only), "tohum_p_le0": round(float(np.mean(toh_only <= 0)), 4)}
                    if ad == "s8":
                        # sahne grubu başına Δ (tek-grup mAP; tohum ortalaması) ve bir-grup-dışarıda katkısı
                        dg = np.array([[100 * (ms("grp", s, k)[g] - ms("grp", s, "B")[g]) for g in range(len(gruplar))]
                                       for s in range(3)])
                        dgm = np.nanmean(dg, axis=0)
                        dlo = np.array([[100 * (ms("loo", s, k)[g] - ms("loo", s, "B")[g]) for g in range(len(gruplar))]
                                        for s in range(3)]).mean(axis=0)
                        katki = dtam.mean() - dlo  # grup g dahil olduğunda toplam Δ'nın ne kadarı (puan)
                        e["grup"] = {g: {"delta_tek_grup": None if np.isnan(dgm[i]) else round(float(dgm[i]), 3),
                                         "delta_bir_grup_disarida": round(float(dlo[i]), 3),
                                         "katki_puan": round(float(katki[i]), 3)} for i, g in enumerate(gruplar)}
                        e["pozitif_grup"] = int(np.sum(dgm > 0))
                        e["negatif_grup"] = int(np.sum(dgm < 0))
                        e["bos_veya_sifir_grup"] = int(np.sum(~(dgm != 0) | np.isnan(dgm)))
                        e["pozitif_grup_loo_katki"] = int(np.sum(katki > 0))
                        # sınıf başına Δ (tam örneklem)
                        e["sinif_delta"] = {SINIF[c]: round(float(np.mean([100 * (veri[(m, k, s, ck)]['tam'][c] - veri[(m, 'B', s, ck)]['tam'][c])
                                                                      for s in range(3)])), 3) for c in range(8)}
                    ent[ad] = e
                sonuc["sonuc"][f"{m}/{ck}/{k}"] = ent
    (OUT / "saglamlik-analiz.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "saglamlik-analiz.md").write_text(md_uret(sonuc), encoding="utf-8")
    print(md_uret(sonuc))


def md_uret(S):
    s = ["# KITTI test sağlamlık analizleri (kilit SONRASI, keşifsel; etiket: kilit_sonrasi_kesifsel)\n",
         f"Yeniden üretim denetimi: önbellekteki tam-örneklem mAP50-95 ile kayıtlı test değerleri arasındaki en büyük mutlak fark = "
         f"{S['yontem']['yeniden_uretim_max_fark_map']:.2e}.\n",
         f"Hiyerarşik bootstrap: B={S['yontem']['B']}; tohum (3, yerine koyarak) sonra sahne grubu (13, yerine koyarak); Δ = kart − B (puan). "
         "P(Δ≤0) = çoğaltmalarda Δ≤0 oranı. tek-grup/loo tanımları json'da.\n"]
    for ck in ("best", "last"):
        for m in ("yolo11s", "yolo26n"):
            s.append(f"\n## {m} — {ck}.pt\n\n| Kart | Δ ort | hiyer. GA95 | P(Δ≤0) | yalnız-grup GA95 | yalnız-tohum GA95 | tram'sız Δ [hiyer. GA] | 6 sınıf Δ [hiyer. GA] | poz. grup/13 |\n|---|---|---|---|---|---|---|---|---|")
            for k in KARTLAR:
                e = S["sonuc"][f"{m}/{ck}/{k}"]
                a, b, c = e["s8"], e["tramsiz7"], e["ps_misc_siz6"]
                s.append(f"| {k} | {a['delta_ort']:+.2f} | [{a['hier_ga95'][0]:+.2f}, {a['hier_ga95'][1]:+.2f}] | {a['hier_p_le0']:.3f} "
                         f"| [{a['grup_ga95'][0]:+.2f}, {a['grup_ga95'][1]:+.2f}] | [{a['tohum_ga95'][0]:+.2f}, {a['tohum_ga95'][1]:+.2f}] "
                         f"| {b['delta_ort']:+.2f} [{b['hier_ga95'][0]:+.2f}, {b['hier_ga95'][1]:+.2f}] "
                         f"| {c['delta_ort']:+.2f} [{c['hier_ga95'][0]:+.2f}, {c['hier_ga95'][1]:+.2f}] | {a['pozitif_grup']}/13 |")
    s.append("\n## Test grupları: görüntü ve sınıf sayıları (YOLO etiketleri)\n\n| Grup | görüntü | " + " | ".join(SINIF) + " |\n|---|---|" + "---|" * 8)
    for g, v in S["gruplar"].items():
        s.append(f"| {g} | {v['goruntu']} | " + " | ".join(str(v["sinif_sayilari"][c]) for c in SINIF) + " |")
    for ck in ("best",):
        for m in ("yolo11s", "yolo26n"):
            for k in KARTLAR:
                e = S["sonuc"][f"{m}/{ck}/{k}"]["s8"]
                s.append(f"\n### {m} {k} ({ck}.pt): grup başına Δ (puan)\n\n| Grup | Δ tek-grup | Δ bu grup dışarıda | katkı (puan) |\n|---|---|---|---|")
                for g, v in e["grup"].items():
                    d = v["delta_tek_grup"]
                    s.append(f"| {g} | {'--' if d is None else format(d, '+.2f')} | {v['delta_bir_grup_disarida']:+.2f} | {v['katki_puan']:+.2f} |")
                s.append(f"\nSınıf başına Δ (tam örneklem): " + ", ".join(f"{c} {e['sinif_delta'][c]:+.2f}" for c in SINIF))
    return "\n".join(s) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("komut", choices=["hesapla", "topla"])
    ap.add_argument("--isci", type=int, default=6)
    a = ap.parse_args()
    hesapla(a.isci) if a.komut == "hesapla" else topla()
