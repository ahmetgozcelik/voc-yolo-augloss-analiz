#!/usr/bin/env python
"""notes/derivations.md uretici: turetilmis degerlerin formulleri ve girdi kayitlari (id + anahtar + deger). Degerler evidence.json'dan okunur."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "figures" / "scripts"))
import _ev as E  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _derivations_tur2 import SEC2  # noqa: E402

K = E.load()


def row(key):
    r = K[key]
    d = r["deger"]
    dv = f"{d:.4f}" if isinstance(d, float) else str(d)
    return f"| `{key}` | {r['id']} | {dv} {r['birim']} | {r['kaynak_dosya']} |"


SEC = [
    ("D1. Seed-eşli Δ (val): kart − baseline, aynı seed",
     "Δ_s = M_kart,s − M_baseline,s  (puan = ×100). H03 için ortalama ve s.d. (ddof=1) 3 seed üzerinden; p = `scipy.stats.ttest_rel` (df=2, çift yönlü); tek yönlü p = p_çift/2 (Δ>0).",
     ["dval.map50_95.h03.s0", "dval.map50_95.h03.s1", "dval.map50_95.h03.s2", "dval.map50_95.h03.ort", "dval.map50_95.h03.sd", "dval.map50_95.h03.p_cift", "dval.map50_95.h03.p_tek",
      "dval.map50.h03.ort", "dval.map50.h03.sd", "dval.map50.h03.p_cift"]),
    ("D2. Test Δ ve istatistik (WS/sonuclar/test-analiz.json'dan doğrudan)",
     "Δ_s = M_kart,s − M_baseline,s (test); p = seed-eşli t (df=2); Holm: 2 karşılaştırma (H03, TTA; birincil metrik); %95 GA: görüntü-düzeyi eşli bootstrap (B=1000), 3 seed birleşik.",
     ["dtest.map50_95.h03.ort", "dtest.map50_95.h03.sd", "dtest.map50_95.h03.p", "holm.h03", "boot.h03.ci_lo", "boot.h03.ci_hi", "dtest.map50_95.tta.ort", "dtest.map50_95.tta.sd",
      "dtest.map50_95.tta.p", "holm.tta", "boot.tta.ci_lo", "boot.tta.ci_hi"]),
    ("D3. TTA val Δ (3 baseline best.pt) ve p",
     "Δ_s: `h07_post.csv` T1 satırı (delta_full_puan, delta_s1_puan, delta_s2_puan; her biri kendi best.pt'sinin R0'ına göre); ort/s.d.; p = tek-örnek t (H0: Δ=0, df=2; türetilmiş, keşifsel).",
     ["tta_val.delta.s0", "tta_val.delta.s1", "tta_val.delta.s2", "tta_val.delta.ort", "tta_val.delta.sd", "tta_val.delta.p_cift"]),
    ("D4. Gecikme oranları (oturum etiketli)",
     "Test: ort(TTA) / ort(baseline) aynı K5 oturumunda (3 seed). Val: `h07_post.csv` lat_oran (aynı oturum, R0=1). Kart gecikmesi: kart/baseline (egitim sonu olcumu).",
     ["test.latency_ms.baseline.ort", "test.latency_ms.tta.ort", "test.latency_orani.tta", "tta_val.latency_orani", "test.latency_orani.h03", "val.latency_orani.h08", "tta.beklenen_piksel_orani"]),
    ("D5. Seed gürültüsü",
     "s.d. = std(ddof=1) 3 seed; oran = s.d._val / s.d._test; F = oran²; p = P(F(2,2) > F) (tek yönlü; çift yönlü 2×).",
     ["gurultu.val_sd_baseline", "gurultu.test_sd_baseline", "gurultu.oran", "gurultu.f_orani", "gurultu.f_p_tek", "gurultu.val_ap_s_aralik", "gurultu.korelasyon_val_test", "gurultu.korelasyon_val_test_h03"]),
    ("D6. H03 val→test tutarlılığı",
     "Fark_s = Δ_val,s − Δ_test,s; tek-örnek t (df=2). mAP50 için aynı.",
     ["tutarlilik.h03.val_test_fark_ort", "tutarlilik.h03.val_test_fark_sd", "tutarlilik.h03.val_test_fark_p", "tutarlilik.h03.map50_fark_ort", "tutarlilik.h03.map50_fark_p"]),
    ("D7. Val çözünürlüğü: görüntü örnekleme SE tahmini ve ön-eleme kapısı",
     "SE_test = ortalama((CI_hi − CI_lo)/(2·1.96)) seed başına bootstrap; SE_val ≈ SE_test·√(4952/1000) (iid görüntü varsayımı, TAHMİN). Yanlış-pozitif = 1 − Φ(0.30/σ), σ ∈ {H03 val Δ s.d., SE_val}.",
     ["turet.se_test_per_seed_h03", "turet.se_val_per_seed_h03_tahmin", "turet.on_eleme_yanlis_pozitif.gozlenen_sd", "turet.on_eleme_yanlis_pozitif.goruntu_se_tahmini", "turet.min_tespit_edilebilir_etki"]),
    ("D8. Çoklu karşılaştırma (12 keşifsel test)",
     "12 düzeltmesiz p (H03, TTA × {mAP50-95, mAP50, mAP75, AP_S, AP_M, AP_L}); Holm (adım-aşağı, monoton).",
     ["holm12.min", "holm12.tta.map50_95", "holm12.tta.ap_m", "holm12.tta.ap_l", "holm12.h03.ap_m"]),
    ("D9. Bütçe",
     "GPU sa = Σ sure_sn / 3600 (`.done`); tur bazında ve kümülatif; sure_sn yoksa 0 (yetim). Tur eşlemesi kuyruk adından: k3*/manuel→0, k4-tur1+k4-h03-seed→1, k4-tur2+k4-tur2b→2, k4-tur3a→3, k5-test→K5.",
     ["butce.tur.0.sure_sa", "butce.tur.1.sure_sa", "butce.tur.2.sure_sa", "butce.tur.3.sure_sa", "butce.tur.4.sure_sa", "butce.kumulatif.2", "butce.toplam_sa", "butce.toplam_kosum", "butce.kanit_kosum"]),
    ("D10. Doz",
     "doz = varyant_etki_sayaci / boxcp_cagri (nesne/mosaic); çarpan = doz_H08 / ort(doz_H03).",
     ["doz.nesne_per_mosaic.h03.s0", "doz.h03_ort", "doz.nesne_per_mosaic.h08.s0", "doz.carpan_h08_h03"]),
    ("D11. Kayıp payları (baseline s0, ep 100)",
     "pay_k = train/k_loss / (box+cls+dfl) (results.csv `loss_items` zaten kazançlıdır; hata-analizi.md ile uyumlu).",
     ["loss_pay.box.ep100.baseline.s0", "loss_pay.cls.ep100.baseline.s0", "loss_pay.dfl.ep100.baseline.s0", "loss_pay.box.ep1.baseline.s0", "loss_pay.cls.ep1.baseline.s0", "loss_pay.dfl.ep1.baseline.s0"]),
    ("D12. H07 türetilmiş",
     "ΔAP_x = AP_x(satır) − AP_x(R0); W1−T1 ve AP50 farkları.",
     ["h07.U768.d_ap_s", "h07.U768.d_ap_l", "h07.W1_eksi_T1.delta", "h07.W1_eksi_R0.map50", "h07.T1_eksi_R0.map50"]),
    ("D13. H03 val kaynağa göre ayrım (keşifsel)",
     "val_preds.json üzerinde pycocotools; image alt kümeleri VOC07 (303) ve VOC12 (697); Δ = H03 − baseline (aynı seed).",
     ["valkaynak.h03.07.map50_95.ort", "valkaynak.h03.07.map50_95.sd", "valkaynak.h03.12.map50_95.ort", "valkaynak.h03.12.map50_95.sd", "valkaynak.h03.07.map50.ort", "valkaynak.h03.07.map50.sd",
      "valkaynak.h03.12.map50.ort", "valkaynak.h03.12.map50.sd"]),
    ("D14. Test işletim noktası ve görüntü-düzeyi (nitel panel havuzu)",
     "Görüntü-düzeyi AP50-95: npz `tp/conf/pcls`, GT sınıfı ∪ (skor≥0.25 sınıfları), 101-nokta AP; ortalama Δ tüm geçerli görüntülerde.",
     ["dtest.precision.tta.ort", "dtest.recall.tta.ort", "dtest.precision.tta.p", "f6.ort_delta_tta_tum", "f6.kazanc_orani_tta_tum", "f6.kayip_orani_tta_tum", "f6.ort_delta_h03_tum", "f6.n_havuz"]),
    ("D15. Seed-düzeyi t aralığı (Faz 5 tur 1, hakem Ö2; build_derivations.py'ye taşındı)",
     "Eğitim müdahalesinde belirsizlik tohumlar arasıdır; görüntü-bootstrap GA'sı bunu içermez. Ortalama Δ ± t_{0.975,2} · s_Δ / √n, n=3, df=2, t_{0.975,2}=4.302653. Girdiler test-analiz.json'dan (dtest.*.ort/sd). "
     "H03 aralığı +0.30 marjını kapsar; yarı genişlik ≈ %50 güçte kritik etki (α=0.05, df=2; s_Δ üç tohumdan tahmin edildiği için belirsiz).",
     ["turet.t_krit_df2", "dtest.map50_95.h03.ort", "dtest.map50_95.h03.sd", "turet.h03_t_aralik_lo", "turet.h03_t_aralik_hi", "dtest.map50_95.tta.ort", "dtest.map50_95.tta.sd",
      "turet.tta_t_aralik_lo", "turet.tta_t_aralik_hi"]),
    ("D16. Replikasyon çıpası duyarlılığı (Faz 5 tur 1, hakem Ö7)",
     "Yalnız YOLO11n çıpaları (61.0 ve 61.3) alınırsa kabul aralığının üst sınırı 61.3 + 1.2 (tolerans) = 62.5 olur; baseline test mAP50-95 bu sınırın üstündedir (SARI). Fark = baseline − 61.3.",
     ["rep.map50_95.baseline.ort", "replik.fark_yolo11n_cipa_ust", "replik.yolo11n_cipa_ust_sinir"]),
    ("D17. n=3 genişlemesi (P1, hakem B3): seed-eşli Δ, ort ± s.d., p, t aralığı, ön-eleme isabeti",
     "Δ_s = M_kart,s − M_baseline,s (s=0,1,2; puan = ×100; AP75: pycocotools). ort ± s.d. (ddof=1); p_çift = `scipy.stats.ttest_1samp(Δ, 0)` (= seed-eşli t, df=2); p_tek (H1: Δ>0) = p_çift/2 (ort>0) ya da 1 − p_çift/2; "
     "aralık = ort ± t_{0.975,2}·s.d./√3. Kabul ölçütü (ön-kayıtlı): ort Δ ≥ +0.40 VE p_tek < 0.10. Ön-eleme: seed-0 Δ ≥ +0.30. Holm-7: yedi kartın p_çift'i (mAP50-95) üzerinde (keşifsel). "
     "Tüm değerler `WS/sonuclar/n3-analiz.json` ile 1e-6 içinde çapraz doğrulanır (üretici assert). Aşağıda H05 (kırmızı bayrak) ve H08 (doz) için örnek girdiler.",
     ["dval.map50_95.h05.s0", "dval.map50_95.h05.s1", "dval.map50_95.h05.s2", "dval.map50_95.h05.ort", "dval.map50_95.h05.sd", "dval.map50_95.h05.p_cift", "dval.map50_95.h05.t_lo", "dval.map50_95.h05.t_hi",
      "dval.map50_95.h05.holm7", "mek.vfl_bce_grad_orani.h05.s0", "mek.vfl_bce_grad_orani.h05.s1", "mek.vfl_bce_grad_orani.h05.s2",
      "dval.map50_95.h02.ort", "dval.map50_95.h02.p_cift", "dval.map50_95.h02.holm7", "dval.map50_95.h08.ort", "dval.map50.h08.ort", "dval.map75.h03.ort",
      "n3.kabul_sayisi", "n3.on_eleme.gecen_sayisi", "n3.on_eleme.elenen_sayisi", "n3.on_eleme.dogru_sayisi", "tur.1.en_iyi_delta_n3", "tur.2.en_iyi_delta_n3", "tur.3.en_iyi_delta_n3",
      "tur.durma_kurali_n3_tetik"]),
    ("D18. n=3: doz, gecikme gürültüsü ve bütçe (tur 5 = k5-ek-seed)",
     "doz_H08 = ort(varyant_etki_sayaci / boxcp_cagri) 3 seed; çarpan = doz_H08 / doz_H03 (H08 doz ile aday havuzunu birlikte değiştirir). Gecikme oranı = ort(kart)/ort(baseline) (3 seed; aynı mimari). "
     "Bütçe: Σ sure_sn/3600 (`.done`; 47 çekirdek + 13 k5-ek-seed = 60 koşum); günlük sayacı 76.1 sa ayrı kayıtta (.done toplamından ≈0.6 sa fazla).",
     ["doz.h08_ort", "doz.carpan_h08_h03_n3", "gurultu.latency_orani_n3_min", "gurultu.latency_orani_n3_max", "gurultu.latency_tek_kosum_max_oran", "butce.tur.5.sure_sa", "butce.kumulatif.5",
      "butce.n3.toplam_sa", "butce.n3.toplam_kosum", "butce.n3.kanit_kosum", "butce.n3.ek_egitim_sa", "butce.n3.gunluk_toplam_sa"]),
]
SEC += SEC2
L = ["# Türetilmiş değerler (Faz 2) — formül + girdi kayıtları", "",
     "Üretici: `evidence/scripts/build_derivations.py` (değerler `evidence/evidence.json`'dan okunur; elle değiştirmeyin). Birim: puan = ×100.", ""]
for title, formula, keys in SEC:
    L.append(f"## {title}")
    L.append(f"**Formül/yöntem:** {formula}\n")
    L.append("| anahtar | id | değer | kaynak dosya |")
    L.append("|---|---|---|---|")
    for k in keys:
        if k in K:
            L.append(row(k))
        else:
            L.append(f"| `{k}` | – | (kayıt yok) | – |")
    L.append("")
(E.MW / "notes" / "derivations.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("derivations.md yazildi")
