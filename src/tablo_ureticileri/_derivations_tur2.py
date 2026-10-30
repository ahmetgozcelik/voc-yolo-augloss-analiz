"""Faz 5 tur-2 (hakem N1/N2/N6) turetilmis degerlerin D19-D22 bolumleri; build_derivations.py bunu SEC'e ekler."""
ARMS = ["h01", "h02", "h03", "h04", "h05", "h06", "h08"]
SIX = ["h01", "h02", "h04", "h05", "h06", "h08"]

SEC2 = [
    ("D19. Kritik etki (güç) — kart başına, ortak ve test (Faz 5 tur 2, hakem N1)",
     "Eşli t (df=2, α=0.05 çift yönlü), n=3; s_Δ = seed-eşli Δ'nın s.d.'si (kart: `dval.map50_95.<kart>.sd`; ortak: s_Δ,ort = √(mean_a s_Δ,a²), 7 kart, df=14, eşit varyans kabulü; test: `dtest.map50_95.<h03|tta>.sd`). "
     "(i) %50 güç (geleneksel; = %95 t aralığının yarı genişliği): δ50 = t_{0.975,2}·s_Δ/√3, t_{0.975,2}=4.3027. "
     "(ii) %80 güç, yaklaşık: δ80 ≈ (t_{0.975,2}+t_{0.80,2})·s_Δ/√3 = 5.363·s_Δ/√3; normal yaklaşım (z_{0.975}+z_{0.80}) = 2.802·s_Δ/√3 df=2'de iyimserdir (yalnız ortak değerde bilgi için gösterilir). "
     "(iii) %80 güç, kesin (s_Δ bilinen kabul): merkezi olmayan t (df=2), nc=δ√3/s_Δ; P(|T_nc|>t_{0.975,2})=0.80 ⇒ nc=5.653 (`guc.nc_gucu80_nct`), δ=nc·s_Δ/√3. Aynı çözüm %50 için nc=3.629 verir (t aralığı yarı genişliği gerçekte %50'nin biraz üstünde güce karşılık gelir; geleneksel adlandırma korunur). "
     "(iv) Ortak s_Δ belirsizliği: %95 GA çarpanı [√(14/χ²_{0.975,14}), √(14/χ²_{0.025,14})] = [0.73, 1.58]; δ50 GA'sı bu çarpanla ölçeklenir. Tek kartın s_Δ'sı df=2'den (çarpan ≈[0.52, 6.3]) geldiği için kart başına değerler çok belirsizdir. "
     "(v) Kabul eşiği: ön-kayıtlı kural ort Δ≥+0.40 VE tek yönlü p<0.10 ⇒ kartın kabul alabilmesi için gereken en küçük ortalama = max(0.40, t_{0.90,2}·s_Δ/√3), t_{0.90,2}=1.886. "
     "(vi) Eşdeğerlik (TOST benzeri): %95 t aralığının [−0.30,+0.30] içinde kalması — hiçbir kartta sağlanmaz. Uyarı: kartların Δ'ları aynı baseline koşumlarını paylaşır, s_Δ'lar bağımsız değildir.",
     ["guc.nc_gucu50_nct", "guc.nc_gucu80_nct", "guc.t_toplam_gucu80_yaklasik", "guc.z_toplam_gucu80_normal"]
     + [f"guc.val.{a}.{s}" for a in ARMS for s in ("kritik50", "kritik50_nct", "kritik80_t_yaklasik", "kritik80_nct", "kabul_esik_ort")]
     + ["guc.val.ortak_sd_delta", "guc.val.ortak.kritik50", "guc.val.ortak.kritik50_alt", "guc.val.ortak.kritik50_ust", "guc.val.ortak.kritik80_t_yaklasik", "guc.val.ortak.kritik80_nct",
        "guc.val.ortak.kritik80_normal", "guc.test.h03.kritik50", "guc.test.h03.kritik80_nct", "guc.test.h03.kritik80_t_yaklasik", "guc.test.tta.kritik50", "guc.test.tta.kritik80_nct",
        "guc.val.aralik_030_kapsayan_kart", "guc.val.aralik_040_kapsayan_kart", "guc.val.aralik_marj030_icinde_kart", "turet.min_tespit_edilebilir_etki"]),
    ("D20. Ön-eleme isabeti: bilgisiz kapı altında şans olasılığı (Faz 5 tur 2, hakem N2)",
     "Soru: kapı hiç bilgi taşımasa da altı elenen kartın hepsinin n=3'te kabul almaması ne kadar olasıdır? "
     "(A) Hakem formülü: null kartın kapıyı geçme olasılığı p=1−Φ(0.30/σ) (E1371: 0.105, σ=gözlenen seed s.d.; E1372: 0.275, σ=görüntü-SE tahmini); P(6/6 eleme)=(1−p)^6 = 0.145 (p=0.275) … 0.512 (p=0.105); hakemin 0.15–0.51 aralığı DOĞRULANDI. "
     "(B) Kart başına s_Δ ile: p_a=1−Φ(0.30/s_Δ,a) (gerçek etki 0, seed-0 Δ~N(0,s_Δ,a²)); P=Π_{6 elenen}(1−p_a). "
     "(C) 'İsabet'in kendi tanımı (elenen kart n=3'te de kabul almaz) ve bilgisiz kapı (seed-0 Δ ile n=3 kararı bağımsız), gerçek etki 0: Monte Carlo (400 000 tekrar/kart, rng seed 20261003, Δ_s~N(0,s_Δ,a²), 3 seed; kabul = ort≥+0.40 ve t>t_{0.90,2}); P=Π_{6 elenen}(1−a_a). "
     "Yorum: 6/6 şansla beklenebilir bir sonuçtur (A–C: 0.15–0.77); kapının bilgisini kanıtlamaz. Kapı hiçbir gerçek pozitif görmediği için duyarlılığı ölçülemez.",
     ["onelem.sans_6_6.alt_sinir", "onelem.sans_6_6.ust_sinir", "turet.on_eleme_yanlis_pozitif.gozlenen_sd", "turet.on_eleme_yanlis_pozitif.goruntu_se_tahmini"]
     + [f"onelem.gec_olasilik.{a}" for a in ARMS] + ["onelem.sans_6_6.kart_sd"] + [f"onelem.n3_kabul_h0.{a}" for a in SIX] + ["onelem.sans_isabet_6_6.mc"]),
    ("D21. Bağımsız seed 1–2 ortalaması (Faz 5 tur 2, hakem N2)",
     "n=3 ortalaması eleme kararını belirleyen seed 0'ı içerir (elenen kartlarda seed-0 Δ<+0.30 ile kesilmiş ⇒ aşağı yanlı; H03'te ≥+0.30 ile seçilmiş ⇒ yukarı yanlı). Yanlısız tahmin yalnız bağımsız tohumlardır: "
     "ort_{s1,s2} = (Δ_1+Δ_2)/2 (n=2; s.d./p raporlanmaz, betimseldir). Girdi: `dval.map50_95.<kart>.s1/.s2`.",
     [f"bagimsiz.val_map50_95.{a}.ort_s12" for a in ARMS] + ["bagimsiz.on_eleme_esigi_ustu_sayisi", "dval.map50_95.h06.s0", "dval.map50_95.h06.s1", "dval.map50_95.h06.s2"]),
    ("D22. Renk/anlamlılık kuralı ve önceden tanımlı Holm aileleri (Faz 5 tur 2, hakem N6)",
     "\\iyi/\\kotu YALNIZ Holm-düzeltilmiş anlamlılıkta: (a) doğrulama ailesi = 7 eğitim kartının mAP50-95 seed-eşli çift yönlü p'leri, Holm-7 (`dval.map50_95.<kart>.holm7`); "
     "(b) test ailesi = K5 planının 12'li ailesi (H03, TTA × 6 metrik), Holm-12 (`holm12.*`; H03 mAP50-95 de aileye dahildir, konservatif) VE %95 bootstrap GA sıfırı kesmez. "
     "Düzeltmesiz p<0.05 olup Holm sonrası anlamlı olmayan hücre renklenmez, † ile işaretlenir (doğrulama: H02 p=0.041, Holm-7 0.244; test: H03 AP_M/AP_L, TTA mAP50-95/mAP75/AP_M/AP_L). "
     "Doğrulamada Holm-7<0.05 yalnız H05 (0.031) ⇒ \\kotu. TTA'nın doğrulama Δ'sı (ham p=0.075) 7'li ailenin üyesi değildir, renklenmez ve † almaz.",
     [f"dval.map50_95.{a}.holm7" for a in ARMS] + ["dval.map50_95.h02.p_cift", "dval.map50_95.h05.p_cift", "tta_val.delta.p_cift", "holm12.min", "holm12.h03.map50_95", "holm12.tta.map50_95",
                                                    "dtest.map50_95.tta.p", "dtest.ap_m.h03.p", "dtest.ap_l.h03.p", "dtest.map75.tta.p", "dtest.ap_m.tta.p", "dtest.ap_l.tta.p"]),
]
