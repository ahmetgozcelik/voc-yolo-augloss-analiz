#!/usr/bin/env python
"""Makale tablolari uretici (makale-uret Faz 2). Hucreler YALNIZ evidence/evidence.json kayitlarindan gelir.

Iskelet: figure-style.md A4 (tek sutun Springer sn-jnl: `table` + `tabular*{\\textwidth}{@{\\extracolsep{\\fill}}..}` veya `tabularx{\\textwidth}`,
\\small, caption ustte, booktabs). Pre-attentive kodlama (kural 7 / kalite-kurallari S8): her sutun `yon` ("yuksek"|"dusuk") tasir;
\\iyi{} / \\kotu{} yone ve ANLAMLILIK kriterine gore KODLA atanir (elle degil). Baseline/referans satiri notr.
Anlamlilik kriterleri (on-kayitli; caption/dipnotta anlatilir) -- Faz 5 tur-2 (hakem N6) sonrasi, TEK KURAL: \iyi/\kotu YALNIZ Holm-duzeltilmis anlamlilikta.
  - test (H03, TTA)            : K5 planinin 12'li ailesinde (H03 mAP50-95 dahil) Holm-12 p < 0.05 VE %95 bootstrap GA sifiri kesmiyor (hicbiri -> RENKSIZ)
  - val (7 egitim karti, n=3)  : onceden tanimli 7'li aile, Holm-7 p < 0.05 -> yalniz H05 \kotu; H02 (ham p=0.041, Holm-7 0.244) RENKSIZ + dagger
  - duzeltmesiz p<0.05 olup Holm sonrasi anlamli olmayan hucre: renksiz, `$^\dagger$` isareti (T2, T3, T5).
  - H07 satirlari              : mavi = on-kayitli DOGRULANDI; kirmizi = betimsel esik (Δ <= -0.5 ve iki yarida da negatif); gecikme orani > 1.10 kirmizi (+ tek-kosum gurultu isareti)
T5 Holm sutunu K5 planinin 12'li ailesine hizali (revizyon tur-1'de elle yapilan duzeltme buraya islendi; yeniden uretimde kaybolmaz).
Caption butcesi: tablo <= 25 kelime (figure-style E2); dipnot <= 40 kelime (tur-2 hakem N9); uretici dogrular.
Revizyon-v2 (Asama 3; yalniz duzen/terim): T_voc_ana (tab:voc-ana; ana metin = T2+T3+T5 mAP50-95 test satirlari), T_ekb_test_ikincil (tab:ekb-test-ikincil),
T_ekb_mutlak (tab:ekb-mutlak); terimler baseline -> taban model, TUT/OLDUR/INCELE -> SURDUR/KAPAT/INCELE, on-eleme gec/ele -> SURDUR/KAPAT. Yeni hesap yok.
Calistirma (MW kokunden): python evidence/scripts/build_tables.py   (cikti yalniz evidence/tables; paper/tables'a kopyalanir)
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "figures" / "scripts"))
import _ev as E  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = E.MW / "evidence" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
v = E.v
MIN = "$-$"
PM = " $\\pm$ "
SD_BASE = v("gurultu.val_sd_baseline")
BUDGET = v("on_kayit.gercek_zaman_orani")


# ------------------------------------------------------------------------------------------- bicim yardimcilari
def num(x, d=2, sign=False):
    s = f"{abs(x):.{d}f}"
    if x < 0 and float(s) != 0.0:
        return MIN + s
    if sign:
        return "+" + s
    return s


def pm(m, s, d=2, sign=False):
    return num(m, d, sign) + PM + f"{s:.{d}f}"


def pval(p):
    return f"{p:.3f}" if p >= 0.001 else "$<$0.001"


def mark(txt, kind):
    return {"iyi": "\\iyi{%s}", "kotu": "\\kotu{%s}", None: "%s"}[kind] % txt


def direction(value, ref, yon, sig):
    """Metrik yonune gore kodla: sig (anlamli) degilse notr. yon: 'yuksek' (buyuk iyi) | 'dusuk' (kucuk iyi)."""
    if not sig or value == ref:
        return None
    better = value > ref if yon == "yuksek" else value < ref
    return "iyi" if better else "kotu"


def esc(t):
    t = str(t)
    for a, b in (("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")):
        t = t.replace(a, b)
    t = t.replace("->", "$\\to$").replace("<=", "$\\le$").replace("~", "$\\sim$")
    return t


def wc(t):
    return len(re.sub(r"\\[a-zA-Z]+\{?|[{}$]", " ", t).split())


NW = {}
DAG = "$^\\dagger$"


def table(label, caption, colspec, header, rows, notes=None, cap_max=25, tabcolsep=None, notes_max=40):
    n_words = wc(caption)
    assert n_words <= cap_max, f"{label}: caption {n_words} kelime > {cap_max}"
    if notes:
        NW[label] = wc(notes)
        assert NW[label] <= notes_max, f"{label}: dipnot {NW[label]} kelime > {notes_max}"
    use_x = "X" in colspec or "Y" in colspec
    ncol = len(re.findall(r"[lcrXY]", re.sub(r">\{[^}]*\}", "", colspec)))
    L = ["\\begin{table}[!htbp]", "\\centering\\small" + (f"\\setlength{{\\tabcolsep}}{{{tabcolsep}pt}}" if tabcolsep else ""), f"\\caption{{{caption}}}\\label{{{label}}}"]
    L.append(f"\\begin{{tabularx}}{{\\textwidth}}{{{colspec}}}" if use_x else f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{colspec}}}")
    L.append("\\toprule")
    for h in header:
        L.append(h + " \\\\")
    L.append("\\midrule")
    for r in rows:
        if r == "MID":
            L.append("\\midrule")
        else:
            L.append(r + " \\\\")
    L.append("\\bottomrule")
    if notes:
        L.append("\\addlinespace[3pt]")
        L.append(f"\\multicolumn{{{ncol}}}{{@{{}}p{{\\dimexpr\\textwidth-2\\tabcolsep\\relax}}@{{}}}}{{{notes}}} \\\\")
    L.append("\\end{tabularx}" if use_x else "\\end{tabular*}")
    L.append("\\end{table}")
    return "\n".join(L) + "\n", n_words


GENIS = {"T1_replikasyon.tex", "T_voc_ana.tex"}   # iki sütunlu dergi sürümünde tam genişlik (table*)


def write(name, content):
    if name in GENIS:
        content = content.replace("\\begin{table}", "\\begin{table*}").replace("\\end{table}", "\\end{table*}")
    (OUT / name).write_text("% uretici: evidence/scripts/build_tables.py <- evidence/evidence.json (elle duzenleme bir sonraki uretimde kaybolur)\n" + content, encoding="utf-8")


WORDS = {}

# =========================================================================================== T1 replikasyon
rows = []
for s in range(3):
    rows.append(f"Bu çalışma, YOLO11n (COCO ön-eğitimli, 100 ep), tohum {s} & {num(v(f'rep.map50_95.baseline.s{s}'))} & {num(v(f'rep.map50.baseline.s{s}'))}")
rows.append(f"\\textbf{{Bu çalışma, ort.{PM}s.d. (n=3)}} & {pm(v('rep.map50_95.baseline.ort'), v('rep.map50_95.baseline.sd'))} & {pm(v('rep.map50.baseline.ort'), v('rep.map50.baseline.sd'))}")
rows.append("MID")
for k, ad, cite in (("zhang_yolo11n", "YOLO11n sıfırdan, 500 ep", "zhang_2025_adlyolo"), ("lin_yolo11n", "YOLO11-N, 600 ep (genel ayar)", "lin_2025_yolomaster"),
                    ("zhang_yolov8n", "YOLOv8n sıfırdan, 500 ep", "zhang_2025_adlyolo"), ("elhamdouni_yolov8n", "YOLOv8n, 200 ep", "elhamdouni_2025_yolov8nvoc"),
                    ("lin_yolo11s", "YOLO11s tam ince ayar (kontrol noktası testte seçilmiş)", "lin_2026_yolopeft")):
    rows.append(f"Literatür: {ad}{DAG if k == 'elhamdouni_yolov8n' else ''}~\\cite{{{cite}}} & {num(v(f'lit.map50_95.{k}'), 1 if k != 'lin_yolo11s' else 2)} & {num(v(f'lit.map50.{k}'), 1 if k != 'lin_yolo11s' else 2)}")
rows.append("MID")
rows.append(f"Kabul aralığı (literatür aralığı $\\pm$ tolerans){DAG} & {num(v('rep.kabul.map50_95.alt'),1)}--{num(v('rep.kabul.map50_95.ust'),1)} & {num(v('rep.kabul.map50.alt'),1)}--{num(v('rep.kabul.map50.ust'),1)}")
rows.append(f"Makullük kapısı kararı & \\multicolumn{{2}}{{c}}{{{v('rep.karar').replace('YESIL', 'YEŞİL')}}}")
t, w = table("tab:replikasyon",
             "Taban model makullük denetimi: VOC07 test mAP50-95 ve mAP50 [puan]. Literatür satırları aynı koşulda üretilmemiştir; kaynakları ayrı olarak anılır. Renk kodlaması uygulanmaz (referans değerler).",
             ">{\\raggedright\\arraybackslash}X c c", ["Satır & mAP50-95 [puan] & mAP50 [puan]"], rows,
             notes=f"{DAG} Üst çıpa 63.5 (kabul üst ucu {num(v('rep.kabul.map50_95.ust'),1)}) YOLOv8n'e aittir ve muhtemelen VOC07 testinde seçilmiş kontrol noktasındandır; "
                   "değer kaynağın 4.4. bölümünde metin olarak doğrulandı, tablo hücresi okunamadı. "
                   "Diğer literatür satırları da başlatma, epoch ve kontrol noktası seçiminde farklıdır; yalnız bağlamdır. "
                   "Bu çalışmanın değerleri 101 noktalı COCO AP'sidir ve \\texttt{difficult} nesneler hariç tutulmuştur; "
                   "VOC07 11 noktalı devkit değerleriyle~\\cite{everingham_2010_voc} doğrudan kıyaslanamaz.", notes_max=60)
write("T1_replikasyon.tex", t)
WORDS["T1"] = w

# =========================================================================================== T2 leaderboard
# P1 (n=3): TUM egitim kartlari dogrulamada 3 seed; test yalniz baseline, H03, TTA. (Cekirdek kartlar: dval.* / val.* ort/sd/p_cift kayitlari)
ARMS = [("h01", "H01 multi\\_scale"), ("h02", "H02 Inner-CIoU"), ("h03", "H03 copy-paste"), ("h04", "H04 scale"),
        ("h05", "H05 Varifocal"), ("h06", "H06 kutu kazancı"), ("h08", "H08 yüksek doz")]
SIG_P = 0.05
ON_ELEME_ETIKET = {1: "SÜRDÜR", 0: "KAPAT"}   # revizyon-v2 terimleri (ozgun: gec / ele; kart kararlari TUT/OLDUR/INCELE -> SURDUR/KAPAT/INCELE)


def sig_val(arm, m="map50_95"):
    """val: onceden tanimli 7'li aile (7 egitim karti, mAP50-95), Holm-7 p<0.05 (hakem N6: renk YALNIZ Holm-duzeltilmis anlamlilikta)."""
    return v(f"dval.{m}.{arm}.holm7") < SIG_P


def dag_val(arm, m="map50_95"):
    """val: duzeltmesiz p<0.05 ama Holm-7 sonrasi anlamli degil -> renksiz + dagger."""
    return v(f"dval.{m}.{arm}.p_cift") < SIG_P and not sig_val(arm, m)


def sig_test(arm):
    """test: K5 planinin 12'li ailesinde (H03 mAP50-95 dahil) Holm-12 p<0.05 VE %95 bootstrap GA sifiri kesmiyor."""
    return v(f"holm12.{arm}.map50_95") < SIG_P and (v(f"boot.{arm}.ci_lo") > 0 or v(f"boot.{arm}.ci_hi") < 0)


def dag_test(arm, m="map50_95"):
    """test: duzeltmesiz p<0.05 ama Holm-12 sonrasi anlamli degil."""
    return v(f"dtest.{m}.{arm}.p") < SIG_P and not (m == "map50_95" and sig_test(arm)) and v(f"holm12.{arm}.{m}") >= SIG_P


# sutun tanimi: yon "yuksek" (buyuk iyi)
rows = [f"Taban model (n=3) & {pm(v('val.map50_95.baseline.ort'), v('val.map50_95.baseline.sd'))} & -- & -- & {pm(v('test.map50_95.baseline.ort'), v('test.map50_95.baseline.sd'))} & --"]
base_val = v("val.map50_95.baseline.ort")
for arm, ad in ARMS:
    mv, sv = v(f"val.map50_95.{arm}.ort"), v(f"val.map50_95.{arm}.sd")
    dm, ds_, pc = v(f"dval.map50_95.{arm}.ort"), v(f"dval.map50_95.{arm}.sd"), v(f"dval.map50_95.{arm}.p_cift")
    s_ = sig_val(arm)
    c_val = mark(pm(mv, sv), direction(mv, base_val, "yuksek", s_))
    c_dval = mark(pm(dm, ds_, sign=True), direction(dm, 0, "yuksek", s_))
    c_p = pval(pc) + (DAG if dag_val(arm) else "")
    if arm == "h03":
        tm = v("test.map50_95.h03.ort")
        st = sig_test("h03")
        c_test = mark(pm(tm, v("test.map50_95.h03.sd")), direction(tm, v("test.map50_95.baseline.ort"), "yuksek", st))
        dt = v("dtest.map50_95.h03.ort")
        c_dtest = mark(pm(dt, v("dtest.map50_95.h03.sd"), sign=True), direction(dt, 0, "yuksek", st)) + (DAG if dag_test("h03") else "")
    else:
        c_test = c_dtest = "--"
    rows.append(f"{ad} & {c_val} & {c_dval} & {c_p} & {c_test} & {c_dtest}")
tm, dt = v("test.map50_95.tta.ort"), v("dtest.map50_95.tta.ort")
st = sig_test("tta")
rows.append(f"Taban model + TTA, eğitimsiz & -- & {pm(v('tta_val.delta.ort'), v('tta_val.delta.sd'), sign=True)} & {pval(v('tta_val.delta.p_cift'))} & "
            f"{mark(pm(tm, v('test.map50_95.tta.sd')), direction(tm, v('test.map50_95.baseline.ort'), 'yuksek', st))} & "
            f"{mark(pm(dt, v('dtest.map50_95.tta.sd'), sign=True), direction(dt, 0, 'yuksek', st))}{DAG if dag_test('tta') else ''}")
t, w = table("tab:leaderboard",
             "Doğrulama (n=3) ve test mAP50-95 [puan] (ort. $\\pm$ s.d.), taban modele göre $\\Delta$. $p$: çift yönlü eşli $t$. Renk: Holm-anlamlı (mavi iyi, kırmızı kötü).",
             ">{\\raggedright\\arraybackslash}X c c c c c",
             ["Yöntem & Doğr. mAP50-95 & $\\Delta$ doğr. & $p$ & Test mAP50-95 & $\\Delta$ test"], rows,
             notes=f"Renk için doğrulamada yedi kartlık Holm-7, testte 12'li Holm ve sıfırı kesmeyen GA gerekir. {DAG}: düzeltmesiz $p<0.05$, Holm sonrası anlamlı değil. "
                   f"``--'': test açılmadı. TTA gecikmesi {v('test.latency_orani.tta'):.2f}$\\times$ (Tablo~\\ref{{tab:h07}}).",
             tabcolsep=3)
write("T2_leaderboard.tex", t)
WORDS["T2"] = w

# =========================================================================================== T3 ablasyon / negatif sonuc (n=3)
rows = []
for arm, ad in ARMS:
    A = arm.upper()
    assert v(f"abl.{A}.n_seed") == 3, A
    assert v(f"n3.kabul.{arm}") == 0, A                      # tum kartlar OLDUR (kabul olcutu karsilanmadi)
    dm, sdv = v(f"dval.map50_95.{arm}.ort"), v(f"dval.map50_95.{arm}.sd")
    s_ = sig_val(arm)
    dcell = mark(pm(dm, sdv, sign=True), direction(dm, 0, "yuksek", s_))
    pcell = pval(v(f"dval.map50_95.{arm}.p_cift")) + (DAG if dag_val(arm) else "")
    icell = f"[{num(v(f'dval.map50_95.{arm}.t_lo'), 2, True)}, {num(v(f'dval.map50_95.{arm}.t_hi'), 2, True)}]"
    s0 = v(f"dval.map50_95.{arm}.s0")
    gecti = ON_ELEME_ETIKET[int(v(f"n3.on_eleme.{arm}.gecti"))]
    kab = "; kabul yok" if gecti == "SÜRDÜR" else ""                # H03: kapiyi gecti ama n=3'te kabul almadi
    m12 = v(f"bagimsiz.val_map50_95.{arm}.ort_s12")
    degisen = re.sub(r"\s*\([^)]*\)", "", str(v(f"abl.{A}.degisen_tek_sey"))).replace("box kazancı", "kutu kazancı")   # parantez ayrintilari (doz, boyut) metinde/F5'te; terim birligi (K7)
    rows.append(f"{A} & {esc(degisen)} & {dcell} & {pcell} & {icell} & {num(s0, sign=True)} $\\to$ {gecti}{kab} & {num(m12, sign=True)}")
pmin = min((v(f"dval.map50_95.{a}.p_tek"), a) for a, _ in ARMS)
t, w = table("tab:ablasyon",
             "Ablasyon, n=3 tohum (doğrulama): taban modele göre eşli $\\Delta$ mAP50-95 [puan]; yedi kart KAPAT. Renk: yalnız Holm-7 anlamlı (mavi iyi, kırmızı kötü).",
             "l >{\\raggedright\\arraybackslash}X c c c >{\\raggedright\\arraybackslash}p{2.3cm} c",
             ["Kart & Değişen tek şey & $\\Delta$ ort.$\\pm$s.d. & $p$ (çift) & \\%95 $t$ aralığı & Ön-eleme (s0 $\\Delta$) & $\\Delta$ tohum 1--2"], rows,
             notes=f"Kabul: ort. $\\Delta\\ge+{v('on_kayit.tut_esigi'):.2f}$ ve tek yönlü $p<{v('on_kayit.tut_p_tek'):.2f}$; hiçbiri sağlamadı. Ön-eleme: s0 $\\Delta\\ge+{v('on_kayit.on_eleme_esigi'):.2f}$. "
                   f"Tohum 1--2: eleme tohumu s0 hariç bağımsız ortalama. {DAG}: düzeltmesiz $p<0.05$ (H02: Holm-7 {v('dval.map50_95.h02.holm7'):.3f}).",
             tabcolsep=3)
write("T3_ablasyon.tex", t)
WORDS["T3"] = w

# =========================================================================================== T4 H07 gecikme-bedelli cikarim
DEF = {"R0": "Referans (NMS 0.7, 640 px)", "N50": "NMS IoU 0.5", "N60": "NMS IoU 0.6", "N80": "NMS IoU 0.8", "S03": "Soft-NMS $\\sigma$=0.3",
       "S05": "Soft-NMS $\\sigma$=0.5", "U704": "test imgsz 704", "U768": "test imgsz 768", "U832": "test imgsz 832",
       "T1": "TTA 1, 0.83, 0.67 + flip", "T2": "TTA 1, 1.2, 1.4 + flip", "W1": "WBF (T1 dalları)", "W2": "WBF (T2 dalları)"}
DURUM = {"referans": "referans", "aday_degil": "aday değil", "KAZANC-B": "KAZANÇ-B"}
NOISE_MAX = v("gurultu.latency_tek_kosum_max_oran")      # tek-kosum olcum gurultusu ust siniri (24 egitim kosumu, max/medyan)
rows = []
for r in ["R0", "N50", "N60", "N80", "S03", "S05", "U704", "U768", "U832", "T1", "T2", "W1", "W2"]:
    d, dA, dB = v(f"h07.{r}.delta"), v(f"h07.{r}.delta_A"), v(f"h07.{r}.delta_B")
    ratio = v(f"h07.{r}.lat_oran")
    if r == "R0":
        kd = kr = None
    else:
        good = v(f"h07.{r}.dogrulandi") == 1
        bad = d <= -0.5 and dA < 0 and dB < 0
        kd = "iyi" if good else ("kotu" if bad else None)
        in_noise = round(ratio, 2) <= round(NOISE_MAX, 2)         # tek-kosum gurultusu icinde -> renksiz + dagger (hakem N9)
        kr = None if in_noise else direction(ratio, BUDGET, "dusuk", ratio > BUDGET)
    rcell = mark(num(ratio, 2), kr) + (DAG if (r != "R0" and ratio > BUDGET and kr is None) else "")
    rows.append(f"{r} & {DEF[r]} & {mark(num(d, sign=True) if r != 'R0' else '0.00', kd)} & {num(v(f'h07.{r}.d_ap_s'), sign=True) if r != 'R0' else '0.00'} & "
                f"{num(v(f'h07.{r}.d_ap_l'), sign=True) if r != 'R0' else '0.00'} & {num(v(f'h07.{r}.lat_ms'), 2)} & {rcell} & {DURUM[v(f'h07.{r}.durum')]}")
rows.append("MID")
st = sig_test("tta")
rt = v("test.latency_orani.tta")
rows.append(f"T1 test & TTA, 3 tohum, test & {mark(pm(v('dtest.map50_95.tta.ort'), v('dtest.map50_95.tta.sd'), sign=True), direction(v('dtest.map50_95.tta.ort'), 0, 'yuksek', st))}{DAG if dag_test('tta') else ''} & "
            f"{num(v('dtest.ap_s.tta.ort'), sign=True)} & {num(v('dtest.ap_l.tta.ort'), sign=True)} & "
            f"{num(v('test.latency_ms.tta.ort'), 2)} & {mark(num(rt, 2), direction(rt, BUDGET, 'dusuk', rt > BUDGET))} & KAZANÇ-B")
t, w = table("tab:h07",
             "H07 eğitimsiz çıkarım satırları (doğrulama, tohum 0) ve TTA test satırı: $\\Delta$ mAP50-95 ve gecikme. Mavi doğrulandı; kırmızı $\\Delta\\le-0.5$ ya da $>$1.10$\\times$.",
             "l >{\\raggedright\\arraybackslash}X c c c c c l",
             ["Satır & Tanım & $\\Delta$ mAP50-95 & $\\Delta$ AP$_S$ & $\\Delta$ AP$_L$ & [ms] & [$\\times$] & Durum"], rows, tabcolsep=3,
             notes=f"Doğrulama: taban model tohum-0 \\texttt{{best.pt}} (R0 = {v('h07.R0.lat_ms'):.2f} ms); TTA test: kendi oturumu (taban model {v('test.latency_ms.baseline.ort'):.2f} ms). "
                   f"Kırmızı $\\Delta$ iki doğrulama yarısında da negatif (GA yok). {DAG}: oran tek-koşum gürültüsü ({NOISE_MAX:.2f}$\\times$) içinde (renksiz) ya da $\\Delta$ düzeltmesiz $p<0.05$.")
write("T4_h07_gecikme.tex", t)
WORDS["T4"] = w

# =========================================================================================== T5 test istatistigi
# Holm-12 sutunu K5 planinin 12'li ailesidir (H03 mAP50-95 dahil, konservatif); hicbir hucre Holm-12 sonrasi anlamli degil -> renk yok; ham p<0.05 -> dagger.
METR = [("map50_95", "mAP50-95"), ("map50", "mAP50"), ("map75", "mAP75"), ("ap_s", "AP$_S$"), ("ap_m", "AP$_M$"), ("ap_l", "AP$_L$")]
rows = []
for arm, ad in (("h03", "H03 kutu copy-paste"), ("tta", "Taban model + TTA")):
    for i, (m, mad) in enumerate(METR):
        dm, sdv, p = v(f"dtest.{m}.{arm}.ort"), v(f"dtest.{m}.{arm}.sd"), v(f"dtest.{m}.{arm}.p")
        pc = pval(p) + (DAG if dag_test(arm, m) else "")
        if m == "map50_95":
            sg = sig_test(arm)
            kd = direction(dm, 0, "yuksek", sg)
            holm = mark(pval(v(f"holm12.{arm}.{m}")), "iyi" if (sg and kd == "iyi") else ("kotu" if (sg and kd == "kotu") else None))
            ci = f"[{num(v(f'boot.{arm}.ci_lo'), 2, True)}, {num(v(f'boot.{arm}.ci_hi'), 2, True)}]"
            cell_d = mark(pm(dm, sdv, sign=True), kd)
        else:
            holm = pval(v(f"holm12.{arm}.{m}"))
            ci = "--"
            cell_d = pm(dm, sdv, sign=True)
        rows.append(f"{ad if i == 0 else ''} & {mad} & {cell_d} & {pc} & {holm} & {ci}")
    lo, hi = v(f"turet.{arm}_t_aralik_lo"), v(f"turet.{arm}_t_aralik_hi")
    rows.append(f" & mAP50-95 ($t$) & -- & -- & -- & [{num(lo, 2, True)}, {num(hi, 2, True)}]")
    rows.append("MID")
rows.append(f"H03 (doğrulama) & mAP50-95 & {pm(v('dval.map50_95.h03.ort'), v('dval.map50_95.h03.sd'), sign=True)} & {pval(v('dval.map50_95.h03.p_cift'))} & -- & --")
rows.append(f" & mAP50 & {pm(v('dval.map50.h03.ort'), v('dval.map50.h03.sd'), sign=True)} & {pval(v('dval.map50.h03.p_cift'))} & -- & --")
rows.append(f"TTA (doğrulama, 3 \\texttt{{best.pt}}) & mAP50-95 & {pm(v('tta_val.delta.ort'), v('tta_val.delta.sd'), sign=True)} & {pval(v('tta_val.delta.p_cift'))} & -- & --")
t, w = table("tab:test-istatistik",
             "Taban modele göre $\\Delta$ [puan] (tohum-eşli): test n=3 ve doğrulama. $p$: tohum-eşli t, çift yönlü; Holm-12: 12 test; GA: görüntü-örnekleme bootstrap'ı.",
             ">{\\raggedright\\arraybackslash}X l c c c c",
             ["Karşılaştırma & Metrik & $\\Delta$ ort.$\\pm$s.d. & $p$ & Holm-12 $p$ & \\%95 GA"], rows, tabcolsep=3,
             notes=f"Holm-12: K5 ailesi (H03, TTA $\\times$ 6 metrik; H03 mAP50-95 dahil, konservatif); en küçüğü {v('holm12.min'):.3f}, renk yok. "
                   f"GA tohum varyansını içermez ($B$=1000); ($t$) satırı tohum-düzeyi $t$ aralığıdır. {DAG}: düzeltmesiz $p<0.05$, Holm sonrası anlamlı değil.")
write("T5_test_istatistik.tex", t)
WORDS["T5"] = w

# =========================================================================================== T6 deney kaydi
KART = {"baseline": "taban model", "baseline-test": "taban model (test)", "H03-test": "H03 (test)", "TTA-test": "TTA (test)", "kalibrasyon": "kalibrasyon", "kontrol (baseline_cm0)": "kontrol (mosaic hep açık)", "H07-WBF tani": "H07 WBF tanısı"}
TUR_AD = {0: "Tur 0", 1: "Tur 1", 2: "Tur 2", 3: "Tur 3", 4: "K5", 5: "Ek tohum"}
SIX = ("H01", "H02", "H04", "H05", "H06", "H08")
KARAR = {"baseline": v("rep.karar").replace("YESIL", "YEŞİL"), "kalibrasyon": "kapı", "kontrol (baseline_cm0)": "kontrol", "H07": "KAZANÇ-B (TTA)",
         "baseline-test": "değerlendirme", "H03-test": "değerlendirme", "TTA-test": "değerlendirme", "H03": "KAPAT (3 tohum ort.)"}
for a in SIX:
    KARAR[a] = "KAPAT (ön-eleme, tohum 0)"
KARAR5 = {a: "ön-eleme doğrulandı (n=3)" for a in SIX}
KARAR5["H07-WBF tani"] = "tanı"
lst = json.loads(v("butce.kart_listesi")) + json.loads(v("butce.kart_listesi_n3"))
rows = []
last_t = None
for t_, k in lst:
    if last_t is not None and t_ != last_t:
        rows.append("MID")
    nk, n3, sa = int(v(f"butce.kart.{t_}.{k}.n_kosum")), int(v(f"butce.kart.{t_}.{k}.n_asama3")), v(f"butce.kart.{t_}.{k}.sure_sa")
    ng = int(v(f"butce.kart.{t_}.{k}.n_gecersiz"))
    ks = f"{nk}" + (f" ({ng} geçersiz)" if ng else "")
    karar = KARAR5.get(k, "") if t_ == 5 else KARAR.get(k, "")
    rows.append(f"{TUR_AD[t_] if t_ != last_t else ''} & {KART.get(k) or esc(k)} & {ks} & {n3} & {num(sa, 2)} & {karar}")
    last_t = t_
rows.append("MID")
rows.append(f"\\textbf{{Toplam}} & & \\textbf{{{int(v('butce.n3.toplam_kosum'))}}} & \\textbf{{{int(v('butce.n3.kanit_kosum'))}}} & \\textbf{{{num(v('butce.n3.toplam_sa'), 2)}}} & hedef {num(v('butce.hedef'), 0)} sa, tavan {num(v('butce.tavan'), 0)} sa")
t, w = table("tab:deney-kaydi",
             f"Deney kaydı: tur ve kart başına koşum damgası sayısı, kanıt niteliğindeki aşama-3 koşumları ve GPU süresi [sa]; toplam {int(v('butce.n3.toplam_kosum'))} koşum.",
             "l >{\\raggedright\\arraybackslash}X c c c l",
             ["Tur & Kart / grup & Koşum & Aşama-3 & GPU [sa] & Karar"], rows, tabcolsep=3,
             notes="Koşum: duman testi, kısa ön deneme ve tam aşama damgaları. Geçersiz: yetim süreç (rc=96), güvenlikle durmuş H05 (rc=95); süreye dahil, kanıta değil. "
                   "K5: eğitimsiz test. Ek tohum: yalnız doğrulama. Satır toplamları yuvarlama nedeniyle toplamdan $\\pm$0.02 sa ayrılabilir; "
                   "toplamlar yuvarlanmamış sürelerdendir.")
write("T6_deney_kaydi.tex", t)
WORDS["T6"] = w


# =========================================================================================== revizyon-v2: birlesik VOC tablosu + Ek B tablolari
# YENI HESAP YOK: asagidaki uc uretici T2/T3/T5'in ayni evidence anahtarlarini, ayni bicim yardimcilarini ve ayni renk kurallarini
# (sig_val/dag_val: Holm-7; sig_test/dag_test: Holm-12 + GA) kullanir; yalniz satir/sutun secimi ve etiketler farklidir.
def _pmk(m, s, sign=False):
    """pm() ile ayni sayi; dar tablo icin $\\pm$ cevresinde bosluksuz (yalniz bicim)."""
    return pm(m, s, 2, sign).replace(PM, "$\\pm$")


def _test_hucreleri(arm):
    """T5'in mAP50-95 test satiri (Δ ± s.d., p, Holm-12, %95 bootstrap GA); renk/dagger T5 ile ayni kural."""
    dm, sdv = v(f"dtest.map50_95.{arm}.ort"), v(f"dtest.map50_95.{arm}.sd")
    sg = sig_test(arm)
    kd = direction(dm, 0, "yuksek", sg)
    c_d = mark(_pmk(dm, sdv, sign=True), kd)
    c_p = pval(v(f"dtest.map50_95.{arm}.p")) + (DAG if dag_test(arm) else "")
    c_h = mark(pval(v(f"holm12.{arm}.map50_95")), kd if sg else None)
    c_ga = f"[{num(v(f'boot.{arm}.ci_lo'), 2, True)}, {num(v(f'boot.{arm}.ci_hi'), 2, True)}]"
    return [c_d, f"{c_p} / {c_h}", c_ga]


def voc_ana():
    """Ana metin VOC tablosu (tab:voc-ana) = T3 (dogrulama Δ, p, t aralığı, ön-eleme) + T5 mAP50-95 test satırları (H03, TTA) + T2 TTA doğrulama satırı."""
    rows = []
    for arm, ad in ARMS:
        A = arm.upper()
        dm, sdv = v(f"dval.map50_95.{arm}.ort"), v(f"dval.map50_95.{arm}.sd")
        dcell = mark(_pmk(dm, sdv, sign=True), direction(dm, 0, "yuksek", sig_val(arm)))
        pcell = pval(v(f"dval.map50_95.{arm}.p_cift")) + (DAG if dag_val(arm) else "")
        icell = f"[{num(v(f'dval.map50_95.{arm}.t_lo'), 2, True)}, {num(v(f'dval.map50_95.{arm}.t_hi'), 2, True)}]"
        ecell = f"{num(v(f'dval.map50_95.{arm}.s0'), sign=True)} $\\to$ {ON_ELEME_ETIKET[int(v(f'n3.on_eleme.{arm}.gecti'))]}"
        if int(v(f'n3.on_eleme.{arm}.gecti')):
            ecell += "; kabul yok"   # H03: ön-elemeyi geçti, n=3'te kabul ölçütünü sağlamadı (orijinal T3 hücresi)
        degisen = re.sub(r"\s*\([^)]*\)", "", str(v(f"abl.{A}.degisen_tek_sey"))).replace("box kazancı", "kutu kazancı")
        tc = _test_hucreleri(arm) if arm == "h03" else ["--"] * 3
        rows.append(" & ".join([A, esc(degisen), dcell, pcell, icell, ecell] + tc))
    rows.append("MID")
    rows.append(" & ".join(["TTA", "taban model + TTA (eğitimsiz)", _pmk(v("tta_val.delta.ort"), v("tta_val.delta.sd"), sign=True),
                            pval(v("tta_val.delta.p_cift")), "--", "--"] + _test_hucreleri("tta")))   # T2 TTA satiri ile ayni (renk/dagger yok)
    t, w = table("tab:voc-ana",
                 f"VOC: taban modele göre tohum-eşli $\\Delta$ mAP50-95 [puan], n=3. Taban model: doğrulama {pm(v('val.map50_95.baseline.ort'), v('val.map50_95.baseline.sd'))}, "
                 f"test {pm(v('test.map50_95.baseline.ort'), v('test.map50_95.baseline.sd'))}. Renk: yalnız Holm-anlamlı (mavi iyi, kırmızı kötü).",
                 "@{}l >{\\raggedright\\arraybackslash}X c c >{\\centering\\arraybackslash}p{1.15cm} >{\\centering\\arraybackslash}p{1.4cm} c "
                 ">{\\centering\\arraybackslash}p{1.3cm} >{\\centering\\arraybackslash}p{1.15cm}@{}",   # aralik/karar hucreleri iki satira kirilir (tam genislik, \\small)
                 ["& & \\multicolumn{4}{c}{Doğrulama} & \\multicolumn{3}{c}{Test} \\\\ \\cmidrule(lr){3-6}\\cmidrule(l){7-9}\n"
                  "Kart & Değişen tek şey & $\\Delta\\pm$s.d. & $p$ & \\%95 $t$ aralığı & Ön-eleme s0 $\\Delta\\to$ karar & $\\Delta\\pm$s.d. & $p$ / \\mbox{Holm-12} & \\%95 GA"], rows,
                 notes=f"$p$: çift yönlü eşli $t$. Kabul: $\\bar\\Delta\\ge+{v('on_kayit.tut_esigi'):.2f}$ ve tek yönlü $p<{v('on_kayit.tut_p_tek'):.2f}$; "
                       f"ön-eleme: s0 $\\Delta\\ge+{v('on_kayit.on_eleme_esigi'):.2f}$. Renk: doğrulamada Holm-7, testte Holm-12 ve sıfırı kesmeyen GA (görüntü bootstrap'ı). "
                       f"{DAG}: düzeltmesiz $p<0.05$. "
                       "Özgün karar etiketleri TUT/ÖLDÜR/İNCELE, burada SÜRDÜR/KAPAT/İNCELE.",
                 tabcolsep=2)
    write("T_voc_ana.tex", t)
    WORDS["T_voc_ana"] = w


def ekb_test_ikincil():
    """Ek B (tab:ekb-test-ikincil): T5'in YALNIZ ikincil metrik satırları (mAP50, mAP75, AP_S/M/L); mAP50-95 ve (t) satırları tab:voc-ana'da/çıkarıldı.
    GA sütunu bu satırlarda T5'te de boştu (``--'') -> sütun düşürüldü. H03 doğrulama mAP50 satırı T5'teki gibi (ikincil metrik)."""
    rows = []
    for arm, ad in (("h03", "H03 kutu copy-paste"), ("tta", "Taban model + TTA")):
        for i, (m, mad) in enumerate([x for x in METR if x[0] != "map50_95"]):
            dm, sdv, p = v(f"dtest.{m}.{arm}.ort"), v(f"dtest.{m}.{arm}.sd"), v(f"dtest.{m}.{arm}.p")
            rows.append(f"{ad + ' (test)' if i == 0 else ''} & {mad} & {pm(dm, sdv, sign=True)} & {pval(p) + (DAG if dag_test(arm, m) else '')} & {pval(v(f'holm12.{arm}.{m}'))}")
        rows.append("MID")
    rows.append(f"H03 (doğrulama) & mAP50 & {pm(v('dval.map50.h03.ort'), v('dval.map50.h03.sd'), sign=True)} & {pval(v('dval.map50.h03.p_cift'))} & --")
    t, w = table("tab:ekb-test-ikincil",
                 "VOC ikincil ölçütler: taban modele göre tohum-eşli $\\Delta$ [puan], n=3. $p$: çift yönlü eşli $t$; Holm-12: 12 testlik aile.",
                 ">{\\raggedright\\arraybackslash}X l c c c",
                 ["Karşılaştırma & Metrik & $\\Delta$ ort.$\\pm$s.d. & $p$ & Holm-12 $p$"], rows, tabcolsep=3,
                 notes=f"mAP50-95 satırları: Tablo~\\ref{{tab:voc-ana}}. Holm-12: K5 ailesi (H03, TTA $\\times$ 6 metrik; H03 mAP50-95 dahil, konservatif); "
                       f"en küçüğü {v('holm12.min'):.3f}, renk yok. {DAG}: düzeltmesiz $p<0.05$, Holm sonrası anlamlı değil.")
    write("T_ekb_test_ikincil.tex", t)
    WORDS["T_ekb_test_ikincil"] = w


def ekb_mutlak():
    """Ek B (tab:ekb-mutlak): birleşik tablodan çıkan sütunlar -- mutlak doğrulama/test mAP50-95 (T2) ve Δ tohum 1--2 (T3); renk T2 ile aynı."""
    base_val, base_test = v("val.map50_95.baseline.ort"), v("test.map50_95.baseline.ort")
    rows = [f"Taban model & {pm(base_val, v('val.map50_95.baseline.sd'))} & {pm(base_test, v('test.map50_95.baseline.sd'))} & --", "MID"]
    for arm, ad in ARMS:
        mv = v(f"val.map50_95.{arm}.ort")
        c_val = mark(pm(mv, v(f"val.map50_95.{arm}.sd")), direction(mv, base_val, "yuksek", sig_val(arm)))
        if arm == "h03":
            tm = v("test.map50_95.h03.ort")
            c_test = mark(pm(tm, v("test.map50_95.h03.sd")), direction(tm, base_test, "yuksek", sig_test("h03")))
        else:
            c_test = "--"
        rows.append(f"{ad} & {c_val} & {c_test} & {num(v(f'bagimsiz.val_map50_95.{arm}.ort_s12'), sign=True)}")
    rows.append("MID")
    tm = v("test.map50_95.tta.ort")
    rows.append(f"Taban model + TTA (eğitimsiz) & -- & {mark(pm(tm, v('test.map50_95.tta.sd')), direction(tm, base_test, 'yuksek', sig_test('tta')))} & --")
    t, w = table("tab:ekb-mutlak",
                 "VOC mutlak mAP50-95 [puan] (ort. $\\pm$ s.d., n=3) ve tohum 1--2 ortalama $\\Delta$. Renk: yalnız Holm-anlamlı (kırmızı kötü).",
                 ">{\\raggedright\\arraybackslash}X c c c",
                 ["Yöntem & Doğrulama mAP50-95 & Test mAP50-95 & $\\Delta$ tohum 1--2"], rows, tabcolsep=3,
                 notes="$\\Delta$ tohum 1--2: ön-eleme tohumu s0 hariç bağımsız ortalama (doğrulama). Test yalnız taban model, H03 ve TTA için açıldı. "
                       "Renk ölçütü: doğrulamada Holm-7, testte Holm-12 ve sıfırı kesmeyen GA (Tablo~\\ref{tab:voc-ana}).")
    write("T_ekb_mutlak.tex", t)
    WORDS["T_ekb_mutlak"] = w


VOC_KISA = {"h01": "multi-scale 0.25", "h02": "CIoU $\\to$ Inner-CIoU", "h03": "kutu copy-paste",
            "h04": "ölçek genliği 0.25", "h05": "BCE $\\to$ Varifocal", "h06": "kutu kazancı 10",
            "h08": "copy-paste, doz"}   # tam tanımlar Tablo 1'de


def voc_ana_panelli():
    """tab:voc-ana, iki panel (yalnız düzen; hücreler voc_ana() ile aynı kaynak ve biçimlerle): (a) doğrulama, (b) test.
    Dokuz sütunlu tek tabloda hücreler üç satıra kırılıyordu; paneller tam genişlikte ve kırılmasız."""
    cap = (f"VOC: taban modele göre tohum-eşli $\\Delta$ mAP50-95 [puan], n=3. Taban model: doğrulama "
           f"{pm(v('val.map50_95.baseline.ort'), v('val.map50_95.baseline.sd'))}, test "
           f"{pm(v('test.map50_95.baseline.ort'), v('test.map50_95.baseline.sd'))}. Renk: yalnız Holm-anlamlı (mavi iyi, kırmızı kötü).")
    assert wc(cap) <= 25, wc(cap)
    notes = (f"$p$: çift yönlü eşli $t$. Kabul: $\\bar\\Delta\\ge+{v('on_kayit.tut_esigi'):.2f}$ ve tek yönlü $p<{v('on_kayit.tut_p_tek'):.2f}$; "
             f"ön-eleme: s0 $\\Delta\\ge+{v('on_kayit.on_eleme_esigi'):.2f}$. Renk: doğrulamada Holm-7, testte Holm-12 ve sıfırı kesmeyen GA (görüntü bootstrap'ı). "
             f"{DAG}: düzeltmesiz $p<0.05$. Özgün karar etiketleri TUT/ÖLDÜR/İNCELE, burada SÜRDÜR/KAPAT/İNCELE.")
    NW["tab:voc-ana"] = wc(notes)
    assert NW["tab:voc-ana"] <= 40, NW["tab:voc-ana"]
    X = ">{\\raggedright\\arraybackslash\\hyphenpenalty=10000\\exhyphenpenalty=10000}X"
    L = ["\\begin{table}[!htbp]", "\\centering\\small\\setlength{\\tabcolsep}{4pt}", f"\\caption{{{cap}}}\\label{{tab:voc-ana}}",
         f"\\begin{{tabularx}}{{\\textwidth}}{{@{{}}l {X} c c c c@{{}}}}", "\\toprule",
         "\\multicolumn{6}{@{}l}{\\textit{(a) Doğrulama}} \\\\",
         "Kart & Değişen & $\\Delta\\pm$s.d. & $p$ & \\%95 $t$ aralığı & Ön-eleme (s0) \\\\", "\\midrule"]
    for arm, ad in ARMS:
        dm, sdv = v(f"dval.map50_95.{arm}.ort"), v(f"dval.map50_95.{arm}.sd")
        dcell = mark(_pmk(dm, sdv, sign=True), direction(dm, 0, "yuksek", sig_val(arm)))
        pcell = pval(v(f"dval.map50_95.{arm}.p_cift")) + (DAG if dag_val(arm) else "")
        icell = f"[{num(v(f'dval.map50_95.{arm}.t_lo'), 2, True)}, {num(v(f'dval.map50_95.{arm}.t_hi'), 2, True)}]"
        gecti = int(v(f"n3.on_eleme.{arm}.gecti"))
        ecell = f"{num(v(f'dval.map50_95.{arm}.s0'), sign=True)} $\\to$ " + ("geçti; KAPAT" if gecti else "KAPAT")
        L.append(" & ".join([arm.upper(), VOC_KISA[arm], dcell, pcell, icell, ecell]) + " \\\\")
    L.append("\\midrule")
    L.append(" & ".join(["TTA", "TTA (eğitimsiz)", _pmk(v("tta_val.delta.ort"), v("tta_val.delta.sd"), sign=True),
                         pval(v("tta_val.delta.p_cift")), "--", "--"]) + " \\\\")
    L += ["\\midrule", "\\multicolumn{6}{@{}l}{\\textit{(b) Test (kilitli VOC07 test; yalnız teste giren karşılaştırmalar)}} \\\\",
          "Kart & Karşılaştırma & $\\Delta\\pm$s.d. & $p$ & Holm-12 $p$ & \\%95 GA \\\\", "\\midrule"]
    for arm, ad in (("h03", "kutu copy-paste"), ("tta", "TTA (eğitimsiz)")):
        c_d, c_ph, c_ga = _test_hucreleri(arm)
        c_p, c_h = c_ph.split(" / ")
        L.append(" & ".join([arm.upper(), ad, c_d, c_p, c_h, c_ga]) + " \\\\")
    L += ["\\bottomrule", "\\addlinespace[3pt]",
          f"\\multicolumn{{6}}{{@{{}}p{{\\dimexpr\\textwidth-2\\tabcolsep\\relax}}@{{}}}}{{{notes}}} \\\\",
          "\\end{tabularx}", "\\end{table}"]
    write("T_voc_ana.tex", "\n".join(L) + "\n")
    WORDS["T_voc_ana"] = wc(cap)


voc_ana()          # eski tek panelli sürüm (dosyası aşağıda panelli sürümle üzerine yazılır)
voc_ana_panelli()
ekb_test_ikincil()
ekb_mutlak()
print("tablolar yazildi (caption kelime):", {k: w for k, w in WORDS.items()}, "| dipnot kelime:", NW)
