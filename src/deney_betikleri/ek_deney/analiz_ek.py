"""Ek deney analizi (ON-KAYIT-ek-deney.md "Uç noktalar ve istatistik", "Mekanizma kartları"). WS kökünden:
    python scripts/ek_deney/analiz_ek.py val        # KITTI val (ikincil): DontCare'li mAP, eşli Δ, Holm-5 / Holm-2
    python scripts/ek_deney/analiz_ek.py mekanizma  # H05-pos / H05-neg sınıflaması (VOC val, ilk deney B ve H05'e göre)
    python scripts/ek_deney/analiz_ek.py test       # KITTI test (birincil; yalnız kitti_test_ac.py'den SONRA)
Çıktılar: sonuclar/ek_deney/{val,mekanizma,test}-analiz.{json,md}
Test GA'sı: sahne grubu düzeyinde kümeli bootstrap (B=2000; görüntü-başı COCO eşleşmeleri bir kez hesaplanır,
yeniden örneklemede AP yalnız bu kayıtlardan 101 nokta interpolasyonla yeniden toplanır).
"""
import contextlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import kitti_degerlendir as kd  # noqa: E402
import saglik_kapisi as sk  # noqa: E402

OUT = WS / "sonuclar/ek_deney"
MODELLER = ["yolo26n", "yolo11s"]
VARY = {"B": "baseline", "H01": "h01_multiscale25", "H02": "h02_innerciou", "H03": "h03_boxcp", "H04": "h04_scale25",
        "H05": "h05_vfl", "H06": "h06_box10", "H08": "h08_boxcp_doz"}
IYILESME = ["H01", "H03", "H04", "H06", "H08"]
YONLU = ["H02", "H05"]


def holm(ps):
    k = len(ps)
    sira = np.argsort(ps)
    adj = np.empty(k)
    m = 0.0
    for r, i in enumerate(sira):
        m = max(m, (k - r) * ps[i])
        adj[i] = min(1.0, m)
    return adj.tolist()


def tek_yonlu_t(d, yon):
    d = np.asarray(d, float)
    if len(d) < 2 or np.allclose(d.std(ddof=1), 0):
        return float("nan")
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    return float(stats.t.sf(t, len(d) - 1) if yon > 0 else stats.t.cdf(t, len(d) - 1))


def ga95(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return [float(d.mean() - h), float(d.mean() + h)]


def kosum(model, kart, seed):
    rd = None
    for p in sorted((WS / "runs").glob(f"*_ek_{model}_{VARY[kart]}_s{seed}"), reverse=True):
        d = p / ".done"
        if d.exists() and json.loads(d.read_text(encoding="utf-8")).get("rc") == 0 \
                and json.loads((p / "config.json").read_text(encoding="utf-8")).get("asama") == 3 \
                and not sk.denetle(p, tam=True):
            rd = p
            break
    return rd


def aile_tablosu(deger):
    """deger[(model, kart, seed)] = mAP (0-1). Δ puan (×100) seed-eşli."""
    sonuc = {}
    for m in MODELLER:
        satir = {}
        for k in IYILESME + YONLU:
            d = [100 * (deger[(m, k, s)] - deger[(m, "B", s)]) for s in (0, 1, 2)
                 if (m, k, s) in deger and (m, "B", s) in deger]
            if len(d) < 2:
                satir[k] = {"n": len(d)}
                continue
            yon = 1 if k in IYILESME else -1
            satir[k] = {"n": len(d), "delta_seed": [round(x, 3) for x in d], "ort": round(float(np.mean(d)), 3),
                        "sd": round(float(np.std(d, ddof=1)), 3), "ga95": [round(x, 3) for x in ga95(d)],
                        "p_tek": tek_yonlu_t(d, yon)}
        for aile, yon in ((IYILESME, 1), (YONLU, -1)):
            ks = [k for k in aile if "p_tek" in satir.get(k, {}) and not math.isnan(satir[k]["p_tek"])]
            for k, pa in zip(ks, holm([satir[k]["p_tek"] for k in ks])):
                satir[k]["p_holm"] = round(pa, 4)
                if yon > 0:
                    satir[k]["kabul"] = bool(satir[k]["ort"] >= 0.40 and pa < 0.10)
                else:
                    satir[k]["replike"] = bool(satir[k]["ort"] < 0 and pa < 0.10)
        sd_b = [satir[k]["sd"] for k in IYILESME if "sd" in satir.get(k, {})]
        satir["_kritik_etki_ort_puan"] = round(stats.t.ppf(0.98, 2) / math.sqrt(3) * float(np.mean(sd_b)), 3) if sd_b else None
        sonuc[m] = satir
    return sonuc


def yaz(ad, veri, md):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{ad}-analiz.json").write_text(json.dumps(veri, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / f"{ad}-analiz.md").write_text(md, encoding="utf-8")
    print(md)


def md_tablo(baslik, sonuc, metrik):
    s = [f"# {baslik}\n", f"Metrik: {metrik}; Δ = kart − B (aynı seed), puan. İyileşme ailesi Holm-5 (kabul: Δ≥+0.40 ve p_holm<0.10); "
         "yönlü aile Holm-2 (replike: Δ<0 ve p_holm<0.10).\n"]
    for m, satir in sonuc.items():
        s.append(f"\n## {m}  (kritik etki ≈ {satir.get('_kritik_etki_ort_puan')} puan)\n\n| Kart | n | Δ seed | Δ ort ± sd | %95 GA | p tek | p Holm | karar |\n|---|---|---|---|---|---|---|---|")
        for k in IYILESME + YONLU:
            r = satir.get(k, {})
            if "ort" not in r:
                s.append(f"| {k} | {r.get('n', 0)} | — | — | — | — | — | eksik |")
                continue
            karar = ("KABUL" if r.get("kabul") else "kabul ölçütü sağlanmadı") if k in IYILESME else \
                ("REPLİKE" if r.get("replike") else "replike olmadı")
            s.append(f"| {k} | {r['n']} | {r['delta_seed']} | {r['ort']:+.2f} ± {r['sd']:.2f} | [{r['ga95'][0]:+.2f}, {r['ga95'][1]:+.2f}] "
                     f"| {r['p_tek']:.3f} | {r.get('p_holm', float('nan')):.3f} | {karar} |")
    return "\n".join(s) + "\n"


def val():
    liste = (WS / "data/kitti/splits/val.txt").read_text().split()
    dc = json.loads((WS / "data/kitti/dontcare.json").read_text())
    deger, deger_dcsiz, kimlik, tani = {}, {}, {}, {}
    alt = {"map50_95_3sinif": {}, "map50_95_7sinif": {}, "egri_ep11_90": {}}
    for m in MODELLER:
        for k in VARY:
            for s in (0, 1, 2):
                rd = kosum(m, k, s)
                if not rd:
                    continue
                preds = json.loads((rd / "val_preds.json").read_text())
                e = kd.degerlendir(preds, liste, dc, 8)
                deger[(m, k, s)] = e["map50_95"]
                mt = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))
                deger_dcsiz[(m, k, s)] = mt["val_map50_95"]
                kimlik[f"{m}/{k}/s{s}"] = rd.name
                # ON-KAYIT v5b Ö3: koşum başına tanı (betimsel)
                eg = [float(r["metrics/mAP50-95(B)"]) for r in sk._csv(rd)]
                tani[f"{m}/{k}/s{s}"] = {"en_iyi_epoch": mt.get("en_iyi_epoch"), "egitim_ici_en_iyi": round(max(eg), 6),
                                         "yeniden_deg": mt["val_map50_95"], "fark_puan": round(100 * (max(eg) - mt["val_map50_95"]), 3),
                                         "egri_ep11_90": round(float(np.mean(eg[10:90])), 6), "son_epoch": round(eg[-1], 6)}
                alt["map50_95_3sinif"][(m, k, s)] = e["map50_95_3sinif"]
                alt["map50_95_7sinif"][(m, k, s)] = e["map50_95_7sinif"]
                alt["egri_ep11_90"][(m, k, s)] = tani[f"{m}/{k}/s{s}"]["egri_ep11_90"]
    r = {"dontcare": aile_tablosu(deger), "dontcaresiz_ultralytics": aile_tablosu(deger_dcsiz), "kosumlar": kimlik,
         "tani": tani, "ikincil_betimsel": {ad: betimsel_delta(v) for ad, v in alt.items()}}
    md = md_tablo("KITTI val (İKİNCİL, keşifsel)", r["dontcare"], "val mAP50-95 (DontCare=COCO crowd)")
    md += "\n# Koşum tanısı (v5b Ö3)\n\n| Koşum | en iyi ep | eğitim-içi en iyi | yeniden değ. | fark (puan) | ep11–90 ort | son ep |\n|---|---|---|---|---|---|---|\n"
    for ad, t in tani.items():
        md += (f"| {ad} | {t['en_iyi_epoch']} | {t['egitim_ici_en_iyi']:.4f} | {t['yeniden_deg']:.4f} | {t['fark_puan']:+.2f} "
               f"| {t['egri_ep11_90']:.4f} | {t['son_epoch']:.4f} |\n")
    md += md_betimsel(r["ikincil_betimsel"])
    yaz("val", r, md)


def betimsel_delta(deger):
    """Karar üretmeyen tohum-eşli Δ (puan): {model: {kart: {delta_seed, ort, sd}}}."""
    out = {}
    for m in MODELLER:
        for k in IYILESME + YONLU:
            d = [100 * (deger[(m, k, s)] - deger[(m, "B", s)]) for s in (0, 1, 2)
                 if (m, k, s) in deger and (m, "B", s) in deger]
            if len(d) >= 2:
                out.setdefault(m, {})[k] = {"delta_seed": [round(x, 3) for x in d], "ort": round(float(np.mean(d)), 3),
                                            "sd": round(float(np.std(d, ddof=1)), 3)}
    return out


def md_betimsel(veri):
    s = "\n# İkincil metrikler (betimsel; Holm/karar yok)\n\n| Metrik | Model | Kart | Δ seed | Δ ort ± sd |\n|---|---|---|---|---|\n"
    for ad, v in veri.items():
        for m, kartlar in v.items():
            for k, r in kartlar.items():
                s += f"| {ad} | {m} | {k} | {r['delta_seed']} | {r['ort']:+.2f} ± {r['sd']:.2f} |\n"
    return s


def mekanizma():
    def bul(desen):
        out = {}
        for s in (0, 1, 2):
            c = [p for p in sorted((WS / "runs").glob(desen.format(s=s)), reverse=True)
                 if (p / ".done").exists() and json.loads((p / ".done").read_text(encoding="utf-8")).get("rc") == 0
                 and json.loads((p / "config.json").read_text(encoding="utf-8")).get("asama") == 3
                 and "114442" not in p.name]
            if c:
                out[s] = json.loads((c[0] / "metrics.json").read_text(encoding="utf-8"))["val_map50_95"]
        return out
    # ilk deney B: k3-baseline-v2 koşumları (ek_ ön ekli KITTI koşumları hariç)
    B = {}
    for s in (0, 1, 2):
        c = [p for p in sorted((WS / "runs").glob(f"2026093*_baseline_s{s}"), reverse=True)
             if (p / ".done").exists() and json.loads((p / ".done").read_text(encoding="utf-8")).get("rc") == 0
             and json.loads((p / "config.json").read_text(encoding="utf-8")).get("asama") == 3]
        if c:
            B[s] = json.loads((c[0] / "metrics.json").read_text(encoding="utf-8"))["val_map50_95"]
    H05 = {s: v for s, v in bul("2026100*_h05_vfl_s{s}").items()}
    POS, NEG = bul("*_h05pos_vfl_a1g0_s{s}"), bul("*_h05neg_vfl_poz1_s{s}")
    seeds = sorted(set(B) & set(H05) & set(POS) & set(NEG))
    d = {ad: [100 * (X[s] - B[s]) for s in seeds] for ad, X in (("H05", H05), ("pos", POS), ("neg", NEG))}
    r = {"seed": seeds, "B": B, "delta": {k: [round(x, 3) for x in v] for k, v in d.items()}}
    if len(seeds) >= 2:
        for ad in ("pos", "neg"):
            r[ad] = {"ort": round(float(np.mean(d[ad])), 3), "p_tek_neg": tek_yonlu_t(d[ad], -1),
                     "katki": bool(tek_yonlu_t(d[ad], -1) < 0.10 and np.mean(d[ad]) < 0),
                     "f_betimsel": round(float(np.mean(d[ad]) / np.mean(d["H05"])), 3)}
        I = np.array(d["H05"]) - (np.array(d["pos"]) + np.array(d["neg"]))
        se = float(I.std(ddof=1) / math.sqrt(len(I)))
        r["etkilesim"] = {"I_seed": [round(x, 3) for x in I], "I_ort": round(float(I.mean()), 3), "SE": round(se, 3),
                          "toplamsal": bool(abs(I.mean()) <= 2 * se)}
    md = ("# H05 mekanizma (YOLO11n × VOC val; ilk deney B ve H05'e göre)\n\n```json\n"
          + json.dumps(r, ensure_ascii=False, indent=1) + "\n```\n")
    yaz("mekanizma", r, md)


# ---- test: kümeli bootstrap için görüntü-başı COCO eşleşmeleri ----
def eval_kayitlari(preds, liste, dc):
    """COCOeval.evaluate() sonrası alan=all, maxDet=300 için görüntü×sınıf kayıtları."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt, kimlik = kd.gt_kur(liste, 8)
    boyut = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}
    ters = {v: k for k, v in kimlik.items()}
    preds = [dict(p, image_id=kimlik[Path(p["file_name"]).stem]) for p in preds if Path(p["file_name"]).stem in kimlik]
    for iid, (w, h) in boyut.items():
        for x0, y0, x1, y1 in dc.get(ters[iid], []):
            bw, bh = (x1 - x0) * w, (y1 - y0) * h
            if bw > 0 and bh > 0:
                for k in range(8):
                    gt["annotations"].append({"id": len(gt["annotations"]) + 1, "image_id": iid, "category_id": k + 1,
                                              "bbox": [x0 * w, y0 * h, bw, bh], "area": bw * bh, "iscrowd": 1})
    with contextlib.redirect_stdout(io.StringIO()):
        C = COCO()
        C.dataset = gt
        C.createIndex()
        E = COCOeval(C, C.loadRes([{k: p[k] for k in ("image_id", "category_id", "bbox", "score")} for p in preds]), "bbox")
        E.params.maxDets = [1, 10, 300]
        E.evaluate()
    kay = {}
    for e in E.evalImgs:
        if e is None or e["aRng"] != E.params.areaRng[0] or e["maxDet"] != 300:
            continue
        kay[(ters[e["image_id"]], e["category_id"])] = (np.asarray(e["dtScores"]), np.asarray(e["dtMatches"]) > 0,
                                                        np.asarray(e["dtIgnore"]).astype(bool),
                                                        int((~np.asarray(e["gtIgnore"]).astype(bool)).sum()))
    return kay


def ap_kayitlardan(kay, goruntuler):
    """COCOeval.accumulate ile aynı 101 nokta AP (alan=all); goruntuler tekrar içerebilir."""
    rec_th = np.linspace(0, 1, 101)
    ap_t = []
    for t in range(10):
        ap_k = []
        for k in range(1, 9):
            sc, tp, ig, npos = [], [], [], 0
            for g in goruntuler:
                r = kay.get((g, k))
                if r is None:
                    continue
                sc.append(r[0]); tp.append(r[1][t]); ig.append(r[2][t]); npos += r[3]
            if npos == 0:
                continue
            if sc:
                s = np.concatenate(sc); m = np.concatenate(tp); i = np.concatenate(ig)
                o = np.argsort(-s, kind="mergesort"); m, i = m[o], i[o]
                tps = np.cumsum(m & ~i); fps = np.cumsum(~m & ~i)
            else:
                tps = fps = np.zeros(0)
            rc = tps / npos
            pr = tps / np.maximum(tps + fps, np.spacing(1))
            for j in range(len(pr) - 1, 0, -1):
                pr[j - 1] = max(pr[j - 1], pr[j])
            q = np.zeros(101)
            ind = np.searchsorted(rc, rec_th, side="left")
            ok = ind < len(pr)
            q[ok] = pr[ind[ok]]
            ap_k.append(q.mean())
        ap_t.append(np.mean(ap_k) if ap_k else np.nan)
    return float(np.nanmean(ap_t))


def hazirla(kay, goruntu_sirasi):
    """Sınıf başına tespitleri skora göre bir kez sıralar (ağırlıklı/tekrarlı yeniden örnekleme için)."""
    gi = {g: i for i, g in enumerate(goruntu_sirasi)}
    prep = {}
    for k in range(1, 9):
        sc, tp, ig, im = [], [], [], []
        npos = np.zeros(len(goruntu_sirasi))
        for (g, kk), r in kay.items():
            if kk != k or g not in gi:
                continue
            npos[gi[g]] += r[3]
            if len(r[0]):
                sc.append(r[0]); tp.append(r[1]); ig.append(r[2]); im.append(np.full(len(r[0]), gi[g]))
        if sc:
            s_ = np.concatenate(sc); o = np.argsort(-s_, kind="mergesort")
            prep[k] = (np.concatenate(im)[o], np.concatenate(tp, axis=1)[:, o], np.concatenate(ig, axis=1)[:, o], npos)
        else:
            prep[k] = (np.zeros(0, int), np.zeros((10, 0), bool), np.zeros((10, 0), bool), npos)
    return prep


def ap_agirlikli(prep, w):
    """w: görüntü başına çekilme sayısı (tekrarlı örnekleme); COCO 101 nokta AP (alan=all), sonuç ap_kayitlardan ile aynı."""
    rec_th = np.linspace(0, 1, 101)
    ap = np.full((10, 8), np.nan)
    for k in range(1, 9):
        im, tp, ig, npos_i = prep[k]
        npos = float((npos_i * w).sum())
        if npos == 0:
            continue
        wd = w[im]
        for t in range(10):
            keep = (~ig[t]) & (wd > 0)
            if not keep.any():
                ap[t, k - 1] = 0.0
                continue
            ww = wd[keep]; m = tp[t][keep]
            tps = np.cumsum(ww * m); fps = np.cumsum(ww * ~m)
            rc = tps / npos
            pr = np.maximum.accumulate((tps / np.maximum(tps + fps, np.spacing(1)))[::-1])[::-1]
            ind = np.searchsorted(rc, rec_th, side="left")
            q = np.zeros(101); ok = ind < len(pr); q[ok] = pr[ind[ok]]
            ap[t, k - 1] = q.mean()
    return float(np.nanmean(np.nanmean(ap, axis=1)))


def test_gruplari():
    yol = WS / "data/kitti/test_gruplari.json"
    if yol.exists():
        return json.loads(yol.read_text())
    import kitti_hazirla as kh
    img, drv = kh.goruntuler(), kh.drive_esle()
    grup, *_ = kh.sahne_gruplari(img, drv)
    t = {Path(p).stem: grup[drv[int(Path(p).stem)]] for p in (WS / "data/kitti/splits/test.txt").read_text().split()}
    yol.write_text(json.dumps(t))
    return t


def test(B_boot=2000):
    T = OUT / "test"
    if not (T / "ACILDI.json").exists():
        sys.exit("Test henüz açılmadı (kitti_test_ac.py).")
    liste = (WS / "data/kitti/splits/test.txt").read_text().split()
    dc = json.loads((WS / "data/kitti/dontcare.json").read_text())
    grup = test_gruplari()
    gruplar = sorted(set(grup.values()))
    uyeler = {g: [k for k, v in grup.items() if v == g] for g in gruplar}
    deger, kayitlar, deger_last, test_alt = {}, {}, {}, {}
    for f in sorted(T.glob("*.json")):
        if f.name == "ACILDI.json":
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        m = "yolo26n" if "_ek_yolo26n_" in r["run_id"] else "yolo11s"
        kart = next(k for k, v in VARY.items() if r["varyant"] == f"ek_{m}_{v}")
        s = int(r["seed"])
        deger[(m, kart, s)] = r["dontcare"]["map50_95"]
        if "last" in r:  # ON-KAYIT v5 madde 2 (ikincil)
            deger_last[(m, kart, s)] = r["last"]["dontcare"]["map50_95"]
        for ad in ("map50_95_3sinif", "map50_95_7sinif"):  # v5b Ö3 (ikincil, betimsel)
            if ad in r["dontcare"]:
                test_alt.setdefault(ad, {})[(m, kart, s)] = r["dontcare"][ad]
        kayitlar[(m, kart, s)] = eval_kayitlari(json.loads((T / r["run_id"] / "test_preds.json").read_text()), liste, dc)
    sonuc = aile_tablosu(deger)
    sira = [Path(p).stem for p in liste]
    gidx = {g: np.array([sira.index(i) for i in uyeler[g]]) for g in gruplar}
    prep = {key: hazirla(v, sira) for key, v in kayitlar.items()}
    rng = np.random.default_rng(0)
    cekim = []
    for _ in range(B_boot):  # tüm kartlar AYNI grup çekimlerini kullanır (eşli)
        w = np.zeros(len(sira))
        for g in rng.choice(gruplar, size=len(gruplar), replace=True):
            w[gidx[g]] += 1
        cekim.append(w)
    for m in MODELLER:
        b_ap = {s: [ap_agirlikli(prep[(m, "B", s)], w) for w in cekim] for s in (0, 1, 2) if (m, "B", s) in prep}
        for k in IYILESME + YONLU:
            if "ort" not in sonuc[m].get(k, {}):
                continue
            ss = [s for s in (0, 1, 2) if (m, k, s) in prep and s in b_ap]
            k_ap = {s: [ap_agirlikli(prep[(m, k, s)], w) for w in cekim] for s in ss}
            boot = [np.mean([100 * (k_ap[s][j] - b_ap[s][j]) for s in ss]) for j in range(B_boot)]
            sonuc[m][k]["kumeli_boot_ga95"] = [round(float(np.quantile(boot, 0.025)), 3), round(float(np.quantile(boot, 0.975)), 3)]
    md = md_tablo("KITTI TEST (BİRİNCİL)", sonuc, "test mAP50-95 (DontCare=COCO crowd)")
    if deger_last:  # ON-KAYIT v5b Ö5/Ö7: betimsel; Holm/karar YOK; sağlamlık etiketi = best ve last Δ̄ aynı işaretli mi
        sonuc_last = {}
        md += ("\n# KITTI TEST — last.pt duyarlılığı (İKİNCİL, betimsel; karar üretmez)\n\n"
               "| Model | Kart | Δ best ort | Δ last seed | Δ last ort ± sd | sağlamlık |\n|---|---|---|---|---|---|\n")
        for m in MODELLER:
            for k in IYILESME + YONLU:
                d = [100 * (deger_last[(m, k, s)] - deger_last[(m, "B", s)]) for s in (0, 1, 2)
                     if (m, k, s) in deger_last and (m, "B", s) in deger_last]
                if len(d) < 2 or "ort" not in sonuc[m].get(k, {}):
                    continue
                ob = sonuc[m][k]["ort"]
                et = "seçimden bağımsız" if np.sign(np.mean(d)) == np.sign(ob) else "best.pt seçimine duyarlı"
                sonuc_last.setdefault(m, {})[k] = {"delta_seed": [round(x, 3) for x in d], "ort": round(float(np.mean(d)), 3),
                                                   "sd": round(float(np.std(d, ddof=1)), 3), "saglamlik": et}
                sonuc[m][k]["saglamlik"] = et
                md += (f"| {m} | {k} | {ob:+.2f} | {[round(x, 2) for x in d]} | {np.mean(d):+.2f} ± "
                       f"{np.std(d, ddof=1):.2f} | {et} |\n")
        sonuc["_last_pt_ikincil"] = sonuc_last
    if test_alt:
        sonuc["_ikincil_betimsel"] = {ad: betimsel_delta(v) for ad, v in test_alt.items()}
        md += md_betimsel(sonuc["_ikincil_betimsel"])
    yaz("test", sonuc, md)


def h05c():
    """ON-KAYIT-H05c.md: VFL + cls 1.47. D_s = Δc,s − ΔH05,s (tek yönlü t, D>0, α=0.10); R = (Δ̄c − Δ̄H05)/(−Δ̄H05)."""
    def bul(desen):
        out = {}
        for s in (0, 1, 2):
            c = [p for p in sorted((WS / "runs").glob(desen.format(s=s)), reverse=True)
                 if (p / ".done").exists() and json.loads((p / ".done").read_text(encoding="utf-8")).get("rc") == 0
                 and json.loads((p / "config.json").read_text(encoding="utf-8")).get("asama") == 3
                 and "114442" not in p.name]
            if c:
                out[s] = (c[0], json.loads((c[0] / "metrics.json").read_text(encoding="utf-8")))
        return out
    B = {s: m["val_map50_95"] for s, (_, m) in bul("2026093*_baseline_s{s}").items()}
    H = bul("2026100*_h05_vfl_s{s}")
    C = bul("*_h05c_vfl_cls147_s{s}")
    seeds = sorted(set(B) & set(H) & set(C))
    dH = [100 * (H[s][1]["val_map50_95"] - B[s]) for s in seeds]
    dC = [100 * (C[s][1]["val_map50_95"] - B[s]) for s in seeds]
    D = [c - h for c, h in zip(dC, dH)]
    r = {"seed": seeds, "kosumlar": {s: C[s][0].name for s in seeds},
         "delta_H05": [round(x, 3) for x in dH], "delta_c": [round(x, 3) for x in dC], "D": [round(x, 3) for x in D]}
    if len(seeds) >= 2:
        p = tek_yonlu_t(D, +1)
        R = (np.mean(dC) - np.mean(dH)) / (-np.mean(dH))
        if np.mean(D) > 0 and p < 0.10 and R >= 0.75:
            karar = "büyüklük kaybı açıklar"
        elif np.mean(D) > 0 and p < 0.10 and R >= 0.25:
            karar = "büyüklük kaybın bir kısmını açıklar (kalan pay biçime bağlanır; hipotez)"
        else:
            karar = "büyüklük kaybı açıklamaz (biçim hipotezi güçlenir, kanıtlanmaz)"
        oran = [C[s][1].get("vfl_bce_grad_orani") for s in seeds]
        r.update({"delta_c_ort": round(float(np.mean(dC)), 3), "delta_c_sd": round(float(np.std(dC, ddof=1)), 3),
                  "D_ort": round(float(np.mean(D)), 3), "D_sd": round(float(np.std(D, ddof=1)), 3), "p_tek": p,
                  "R": round(float(R), 3), "karar": karar, "iyilesme_iddiasi_degil": bool(np.mean(dC) > 0.40),
                  "vfl_bce_grad_orani": oran,
                  "etkin_grad_orani": [round(o * 1.47 / 0.5, 3) if o else None for o in oran],
                  "en_iyi_epoch": {"H05c": [C[s][1].get("en_iyi_epoch") for s in seeds],
                                   "H05": [H[s][1].get("en_iyi_epoch") for s in seeds]}})
    md = ("# H05c — VFL + cls 1.47 (YOLO11n × VOC val; ON-KAYIT-H05c.md; keşifsel)\n\n```json\n"
          + json.dumps(r, ensure_ascii=False, indent=1) + "\n```\n")
    yaz("h05c", r, md)


if __name__ == "__main__":
    {"val": val, "mekanizma": mekanizma, "test": test, "h05c": h05c}[sys.argv[1]]()
