#!/usr/bin/env python
"""KITTI ek genelleme deneyi tablo ureticisi: T7 (test), T8 (ikincil betimsel kiyaslar; ek), T9 (kosum basina en iyi epoch; ek),
T10 (ikincil test metrikleri + val Δ + egitim-ici/yeniden degerlendirme farki; ek), T11 (sinif basi AP50-95 Δ; ek), T12 (saglamlik: hiyerarsik bootstrap GA, tram-siz/6 sinif Δ, grup sayisi; kilit sonrasi kesifsel; ek5.sag.*).
Hucreler YALNIZ evidence/evidence.json `ek.*`, `ek3.*` ve `ek4.*` kayitlarindan gelir (ek3: build_ek_evidence_tur3.py; ek4: build_ek_evidence_tur4.py).

Iskelet: build_tables.py ile ayni (figure-style A4): `table` + `tabularx{\\textwidth}`, \\small, caption ustte, booktabs, tam genislik.
Renk kurali (pre-attentive; kalite-kurallari S8; sutun yonu `yon`'dan KODLA):
  T7: \\iyi{} YALNIZ on-kayitli kabul olcutunu saglayan satirda (yon=yuksek, p_Holm<0.10 ve Δ>=+0.40); \\kotu{} yalniz anlamli kotu.
      Kumeli GA / t-GA / last.pt sutunlari betimsel: renk yok.
  T8: betimsel (Holm/karar yok) -> RENK YOK (anlamlilik sinanmadi).
  T9: en iyi epoch == 1 -> yalniz egik (\\textit{}; erken tepe uyarisi; renk YOK: \\kotu{} diger tablolarda "kotu fark" anlamina gelir, K10); diger hucreler notr.
  T12: renk YALNIZ tabloda GOSTERILEN hiyerarsik GA (s8) sifiri disladiginda (Δ, GA, P sutunlari); tram'siz/6 sinif sutunlarinda GA gosterilmedigi icin renk yok.
Caption <= 25 kelime, dipnot <= 40 kelime (figure-style E2); uretici dogrular.
Calistirma (MW kokunden): python evidence/scripts/build_ek_tables.py   [-> evidence/tables ve paper/tables: T7_kitti_test.tex, T8_kitti_ikincil.tex, T9_kitti_epoch.tex]
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "figures" / "scripts"))
import _ev as E  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
v = E.v
MIN = "$-$"
PM = " $\\pm$ "
MODELS = [("yolo26n", "YOLO26n"), ("yolo11s", "YOLO11s")]
CARDS = [("H01", "H01 multi-scale"), ("H02", "H02 Inner-CIoU"), ("H03", "H03 kutu copy-paste"), ("H04", "H04 mosaic scale"),
         ("H05", "H05 Varifocal"), ("H06", "H06 kutu kazancı"), ("H08", "H08 yüksek doz")]
KARAR_KISA = {"kabul ölçütü sağlanmadı": "ölçüt sağlanmadı"}  # evidence'taki tam metin: 'kabul ölçütü sağlanmadı' (dipnotta tanimli)
SAG_KISA = {"seçimden bağımsız": "bağımsız", "best.pt seçimine duyarlı": "duyarlı"}


def num(x, d=2, sign=False):
    s = f"{abs(x):.{d}f}"
    if x < 0 and float(s) != 0.0:
        return MIN + s
    return ("+" + s) if sign else s


def pm(m, s, d=2, sign=False):
    return num(m, d, sign) + PM + f"{s:.{d}f}"


PMC = r"$\pm$"  # bosluksuz (dar tablo)


def pmc(m, s, d=2, sign=False):
    return num(m, d, sign) + PMC + f"{s:.{d}f}"


def pval(p):
    return f"{p:.3f}" if p >= 0.001 else "$<$0.001"


def mark(txt, kind):
    return {"iyi": "\\iyi{%s}", "kotu": "\\kotu{%s}", "egik": "\\textit{%s}", None: "%s"}[kind] % txt


def direction(value, ref, yon, sig):
    if not sig or value == ref:
        return None
    better = value > ref if yon == "yuksek" else value < ref
    return "iyi" if better else "kotu"


def wc(t):
    return len(re.sub(r"\\[a-zA-Z]+\{?|[{}$]", " ", t).split())


GENIS = {"T7_kitti_test.tex"}   # iki sütunlu dergi sürümünde tam genişlik (table*)


def write(name, content):
    if name in GENIS:
        content = content.replace("\\begin{table}", "\\begin{table*}").replace("\\end{table}", "\\end{table*}")
    for out in (E.MW / "evidence" / "tables", E.MW / "paper" / "tables"):
        out.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(content, encoding="utf-8")


def skeleton(caption, label, colspec, header, rows, notes, ncol, gen, sep=2, yer="!htbp"):
    L = [f"\\begin{{table}}[{yer}]", f"\\centering\\small\\setlength{{\\tabcolsep}}{{{sep}pt}}", f"\\caption{{{caption}}}\\label{{{label}}}",
         f"\\begin{{tabularx}}{{\\textwidth}}{{{colspec}}}", "\\toprule"] + header + ["\\midrule"]
    for r in rows:
        L.append("\\midrule" if r == "MID" else r + " \\\\")
    L += ["\\bottomrule", "\\addlinespace[3pt]", f"\\multicolumn{{{ncol}}}{{@{{}}p{{\\dimexpr\\textwidth-2\\tabcolsep\\relax}}@{{}}}}{{{notes}}} \\\\", "\\end{tabularx}", "\\end{table}"]
    assert f"{sep}pt" in L[1]
    return f"% uretici: evidence/scripts/build_ek_tables.py <- evidence/evidence.json ({gen}; elle duzenleme bir sonraki uretimde kaybolur)\n" + "\n".join(L) + "\n"


# =========================================================================================================== T7
def t7():
    from scipy import stats as _st
    tq = _st.t.ppf(0.975, 2) / 3 ** 0.5  # tohum-duzeyi t-GA yari genisligi = (t_{0.975,df=2}/sqrt(3)) * s  (n=3)
    for m, _ in MODELS:  # dipnottaki formulun kayitli t-GA ile birebir tuttugunu dogrula
        for c, _ in CARDS:
            k = f"ek.test.{m}.{c}"
            assert abs((v(f"{k}.ga_hi") - v(f"{k}.delta_ort")) - tq * v(f"{k}.delta_sd")) < 2e-3, k
    rows = []
    for mi, (m, mad) in enumerate(MODELS):
        if mi:
            rows.append("MID")
        rows.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textbf{{{mad}}} (kritik etki $\approx$ {v(f'ek.test.{m}.kritik_etki'):.2f} puan)}}")
        for c, cad in CARDS:
            k = f"ek.test.{m}.{c}"
            kar = v(f"{k}.karar")
            sig = kar in ("KABUL", "replike oldu")
            dm, ds = v(f"{k}.delta_ort"), v(f"{k}.delta_sd")
            kind = direction(dm, 0, "yuksek", sig)
            cell_d = mark(pm(dm, ds, sign=True), kind)
            kga = f"[{num(v(f'{k}.boot_ga_lo'), 2, True)}, {num(v(f'{k}.boot_ga_hi'), 2, True)}]"
            ph = mark(pval(v(f"{k}.p_holm")), kind)
            kr = mark(KARAR_KISA.get(kar, kar), kind)
            rows.append(f"{cad} & {cell_d} & {kga} & {ph} & {kr} & {num(v(f'{k}.last_delta_ort'), 2, True)} & {SAG_KISA[v(f'{k}.saglamlik')]}")
    caption = (r"KITTI test (13 sahne grubu, DontCare kurallı mAP50-95): kart $-$ taban model $\Delta$ [puan], $n=3$ tohum-eşli. "
               r"Mavi+kalın: önceden belirlenmiş kabul ölçütünü sağlayan kart.")
    notes = (r"Kabul: $\Delta\ge+0.40$, $p_{\mathrm{Holm}}<0.10$ (Holm-5; H02, H05 yönlü Holm-2; model başına). "
             r"Kümeli GA: 13 grupta bootstrap ($B=2000$), tohum varyansını içermez. "
             rf"Tohum $t$-GA $=\bar\Delta\pm{tq:.2f}\,s$. "
             r"$\Delta$ last betimsel; sağlamlık: best/last işareti aynıysa bağımsız.")
    assert wc(caption) <= 25, f"T7 caption {wc(caption)} kelime"
    assert wc(notes) <= 40, f"T7 dipnot {wc(notes)} kelime"
    colspec = r">{\raggedright\arraybackslash}X c c c l c c"   # Karar hücresi tek satır (iki sütunlu sürümde table*)
    header = [r"Kart & $\Delta$ test $\pm$ s.d. & Kümeli GA & $p_{\mathrm{Holm}}$ & Karar & $\Delta$ \texttt{last} & Sağlamlık \\"]
    write("T7_kitti_test.tex", skeleton(caption, "tab:kitti-test", colspec, header, rows, notes, 7, "ek.test.*"))
    print(f"T7 yazildi (caption {wc(caption)} kelime, dipnot {wc(notes)} kelime)")


# =========================================================================================================== T8
def t8():
    rows = []
    for mi, (m, mad) in enumerate(MODELS):
        if mi:
            rows.append("MID")
        rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textbf{{{mad}}}}}")
        for c, cad in CARDS:
            def cell(base):
                return pm(v(f"{base}.delta_ort"), v(f"{base}.delta_sd"), sign=True)
            rows.append(" & ".join([cad,
                                    cell(f"ek3.test.{m}.{c}.dcsiz"),
                                    cell(f"ek.test.{m}.{c}.sinif3"),
                                    cell(f"ek.test.{m}.{c}.sinif7"),
                                    cell(f"ek.val.{m}.{c}.egri_ep11_90")]))
    nobj = {s: v(f"ek3.b2.test_nesne.{s}") for s in ("person_sitting", "tram", "truck")}
    caption = ("KITTI önceden belirlenmiş ikincil betimsel kıyaslar: kart $-$ taban model $\\Delta$ ortalama $\\pm$ s.d. [puan], $n=3$ tohum-eşli; Holm yok, renk yok.")
    notes = ("Test (\\texttt{best.pt}): DontCare'siz mAP50-95; DontCare kurallı alt kümeler: car, pedestrian, cyclist (3 sınıf) ve person\\_sitting hariç 7 sınıf. "
             f"Val: eğitim eğrisinin (Ultralytics, DontCare'siz) 11.--90. epoch ortalaması. Test nesne sayısı: person\\_sitting {nobj['person_sitting']}, tram {nobj['tram']}, truck {nobj['truck']}.")
    assert wc(caption) <= 25, f"T8 caption {wc(caption)} kelime"
    assert wc(notes) <= 40, f"T8 dipnot {wc(notes)} kelime"
    colspec = ">{\\raggedright\\arraybackslash}X c c c c"
    header = ["\\multicolumn{1}{@{}l}{} & \\multicolumn{3}{c}{Test $\\Delta$} & Val $\\Delta$ \\\\", "\\cmidrule(lr){2-4}",
              "Kart & DontCare'siz & 3 sınıf & 7 sınıf & ep11--90 eğri \\\\"]
    write("T8_kitti_ikincil.tex", skeleton(caption, "tab:kitti-ikincil", colspec, header, rows, notes, 5, "ek.test.*, ek.val.*, ek3.test.*, ek3.b2.*", sep=6))
    print(f"T8 yazildi (caption {wc(caption)} kelime, dipnot {wc(notes)} kelime)")


# =========================================================================================================== T10
def t10():
    mets = [("map50", "mAP50"), ("map_small", "AP$_S$"), ("map_medium", "AP$_M$"), ("map_large", "AP$_L$")]
    rows = []
    for mi, (m, mad) in enumerate(MODELS):
        if mi:
            rows.append("MID")
        rows.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textbf{{{mad}}}}}")
        # baseline: mutlak degerler (3 tohum ortalamasi +- s.d.); val delta yok; ic-yeniden fark B icin de tanimli
        cells = ["B"] + [num(v(f"ek4.test.{m}.B.{mk}.ort")) + PMC + f"{v(f'ek4.test.{m}.B.{mk}.sd'):.2f}" for mk, _ in mets] + ["--"]
        cells.append(pmc(v(f"ek4.val.{m}.B.egitim_ici_eksi_yeniden_ort"), v(f"ek4.val.{m}.B.egitim_ici_eksi_yeniden_sd"), sign=True))
        rows.append(" & ".join(cells))
        for c, _ in CARDS:
            cells = [c] + [pmc(v(f"ek4.test.{m}.{c}.{mk}.delta_ort"), v(f"ek4.test.{m}.{c}.{mk}.delta_sd"), sign=True) for mk, _ in mets]
            cells.append(pmc(v(f"ek.val.{m}.{c}.delta_ort"), v(f"ek.val.{m}.{c}.delta_sd"), sign=True))
            cells.append(pmc(v(f"ek4.val.{m}.{c}.egitim_ici_eksi_yeniden_ort"), v(f"ek4.val.{m}.{c}.egitim_ici_eksi_yeniden_sd"), sign=True))
            rows.append(" & ".join(cells))
    caption = (r"KITTI ikincil ölçütler (\texttt{best.pt}): kart $-$ taban model $\Delta$ ortalama $\pm$ s.d. [puan], $n=3$ tohum-eşli; test DontCare kurallı. "
               r"Betimsel, renk yok.")
    notes = (r"Kartlar Tablo~\ref{tab:kitti-test}'dedir. Val $\Delta$: DontCare kurallı mAP50-95. İç$-$yeniden: eğitim-içi en iyi val (Ultralytics) $-$ aynı ağırlığın COCO-tarzı DontCare'siz yeniden değerlendirmesi. "
             r"B satırı (taban model) mutlak değerdir; AP$_S$/AP$_M$/AP$_L$ COCO alan sınıflarıdır.")
    assert wc(caption) <= 25, f"T10 caption {wc(caption)} kelime"
    assert wc(notes) <= 40, f"T10 dipnot {wc(notes)} kelime"
    colspec = r">{\raggedright\arraybackslash}X c c c c c c"
    header = [r"\multicolumn{1}{@{}l}{} & \multicolumn{4}{c}{Test $\Delta$} & \multicolumn{2}{c}{Val [puan]} \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
              r"Kart & mAP50 & AP$_S$ & AP$_M$ & AP$_L$ & $\Delta$ mAP50-95 & İç$-$yeniden \\"]
    write("T10_kitti_test_ek.tex", skeleton(caption, "tab:kitti-test-ek", colspec, header, rows, notes, 7, "ek4.test.*, ek4.val.*, ek.val.*", sep=3))
    print(f"T10 yazildi (caption {wc(caption)} kelime, dipnot {wc(notes)} kelime)")


# =========================================================================================================== T11
def t11():
    sinif = [("car", "car"), ("van", "van"), ("truck", "truck"), ("pedestrian", "ped."), ("person_sitting", "sit."), ("cyclist", "cyc."), ("tram", "tram"), ("misc", "misc")]
    rows = []
    for mi, (m, mad) in enumerate(MODELS):
        if mi:
            rows.append("MID")
        rows.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\textbf{{{mad}}}}}")
        rows.append(" & ".join(["B"] + [num(v(f"ek4.test.{m}.B.sinif.{s}.ort"), 1) for s, _ in sinif]))
        for c, _ in CARDS:
            rows.append(" & ".join([c] + [pm(v(f"ek4.test.{m}.{c}.sinif.{s}.delta_ort"), v(f"ek4.test.{m}.{c}.sinif.{s}.delta_sd"), 1, True).replace(PM, "$\\pm$") for s, _ in sinif]))
    nobj = {s: int(v(f"ek3.b2.test_nesne.{s}")) for s, _ in sinif}
    caption = (r"KITTI test: sınıf başı AP50-95 (DontCare kurallı, \texttt{best.pt}) kart $-$ taban model $\Delta$ ortalama $\pm$ s.d. [puan], $n=3$ tohum-eşli. "
               r"Betimsel, renk yok.")
    notes = (r"Kartlar Tablo~\ref{tab:kitti-test}'dedir; sit. = person\_sitting. B satırı mutlak AP. Test nesne sayısı: " + ", ".join(f"{k} {nobj[s]}" for s, k in sinif) + ".")
    assert wc(caption) <= 25, f"T11 caption {wc(caption)} kelime"
    assert wc(notes) <= 40, f"T11 dipnot {wc(notes)} kelime"
    colspec = r">{\raggedright\arraybackslash}X " + "c " * 7 + "c"
    header = ["Kart & " + " & ".join(k for _, k in sinif) + r" \\"]
    write("T11_kitti_sinif.tex", skeleton(caption, "tab:kitti-sinif", colspec, header, rows, notes, 9, "ek4.test.*.sinif.*, ek3.b2.test_nesne.*", sep=2))
    print(f"T11 yazildi (caption {wc(caption)} kelime, dipnot {wc(notes)} kelime)")


# =========================================================================================================== T9
def t9():
    arms = [("B", "B taban model")] + [(c, cad) for c, cad in CARDS]
    rows = []
    n_ep1 = {}
    for mi, (m, mad) in enumerate(MODELS):
        if mi:
            rows.append("MID")
        rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textbf{{{mad}}}}}")
        for c, cad in arms:
            cells = [cad]
            for i in range(3):
                e = v(f"ek.epoch.{m}.{c}.s{i}")
                f = v(f"ek3.val.{m}.{c}.s{i}.best_eksi_last")
                cells.append(mark(str(int(e)), "egik" if e == 1 else None))
                cells.append(num(f, 1, True))
            rows.append(" & ".join(cells))
        n_ep1[m] = (int(v(f"ek.epoch.{m}.tum.n_ep1")), int(v(f"ek.epoch.{m}.tum.n_kosum")))
    caption = ("KITTI: koşum başına en iyi epoch (\\texttt{best.pt}) ve val best$-$last farkı [puan]; 48 koşum, tohum s0--s2. "
               "Eğik: epoch 1.")
    notes = (f"Val farkı: eğitim-içi en iyi $-$ son epoch mAP50-95 (Ultralytics, DontCare'siz). En iyi epoch 1 olan koşum: "
             f"YOLO26n {n_ep1['yolo26n'][0]}/{n_ep1['yolo26n'][1]}, YOLO11s {n_ep1['yolo11s'][0]}/{n_ep1['yolo11s'][1]}.")
    assert wc(caption) <= 25, f"T9 caption {wc(caption)} kelime"
    assert wc(notes) <= 40, f"T9 dipnot {wc(notes)} kelime"
    colspec = ">{\\raggedright\\arraybackslash}X c c c c c c"
    header = ["\\multicolumn{1}{@{}l}{} & \\multicolumn{2}{c}{s0} & \\multicolumn{2}{c}{s1} & \\multicolumn{2}{c}{s2} \\\\",
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
              "Kol & Epoch & Fark & Epoch & Fark & Epoch & Fark \\\\"]
    write("T9_kitti_epoch.tex", skeleton(caption, "tab:kitti-epoch", colspec, header, rows, notes, 7, "ek.epoch.*, ek3.val.*", sep=4))
    print(f"T9 yazildi (caption {wc(caption)} kelime, dipnot {wc(notes)} kelime)")


# =========================================================================================================== T12
def t12():
    """Saglamlik (kilit sonrasi, kesifsel): hiyerarsik bootstrap GA, P(Δ<=0), tram'siz 7 sinif Δ, 6 sinif Δ, pozitif grup /13. Kaynak: ek5.sag.*."""
    def sig(m, c, sub):
        lo, hi = v(f"ek5.sag.{m}.best.{c}.{sub}.hier_lo"), v(f"ek5.sag.{m}.best.{c}.{sub}.hier_hi")
        return "iyi" if lo > 0 else ("kotu" if hi < 0 else None)  # yon=yuksek; yalniz GA sifiri DISLADIGINDA renk

    rows = []
    for mi, (m, mad) in enumerate(MODELS):
        if mi:
            rows.append("MID")
        rows.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textbf{{{mad}}}}}")
        for c, cad in CARDS:
            k = f"ek5.sag.{m}.best.{c}"
            kd = sig(m, c, "s8")
            cells = [cad,
                     mark(num(v(f"{k}.s8.delta"), 2, True), kd),
                     mark(f"[{num(v(f'{k}.s8.hier_lo'), 2, True)}, {num(v(f'{k}.s8.hier_hi'), 2, True)}]", kd),
                     mark(f"{v(f'{k}.s8.hier_p_le0'):.3f}", kd),
                     num(v(f"{k}.tramsiz7.delta"), 2, True),   # alt-kume GA tabloda yok -> renk yok (Ö3)
                     num(v(f"{k}.ps_misc_siz6.delta"), 2, True),
                     f"{int(v(f'{k}.s8.poz_grup'))}/13"]
            rows.append(" & ".join(cells))
    caption = (r"KITTI test sağlamlık (kilit sonrası, keşifsel): kart $-$ taban model $\Delta$ [puan]; tohum ve 13 sahne grubunda hiyerarşik bootstrap ($B=2000$). "
               r"Mavi+kalın: GA sıfırı dışlar.")
    notes = (r"$P(\Delta\le0)$: bootstrap'ta $\Delta\le0$ oranı. Tram'sız: 7 sınıf; 6 sınıf: person\_sitting ve misc hariç; bu iki sütunda renk yoktur (GA gösterilmez). "
             r"Poz. grup: tohum-ortalama $\Delta>0$ olan sahne grubu. \texttt{best.pt}. H02, H05: yönlü kartlar (beklenen $\Delta<0$).")
    assert wc(caption) <= 25, f"T12 caption {wc(caption)} kelime"
    assert wc(notes) <= 40, f"T12 dipnot {wc(notes)} kelime"
    colspec = r">{\raggedright\arraybackslash}X c c c c c c"
    header = [r"Kart & $\Delta$ & Hiyerarşik GA95 & $P(\Delta\le0)$ & Tram'sız $\Delta$ & 6 sınıf $\Delta$ & Poz. grup \\"]
    write("T12_kitti_saglamlik.tex", skeleton(caption, "tab:kitti-saglamlik", colspec, header, rows, notes, 7, "ek5.sag.*", sep=3, yer="p"))  # tek başına float sayfası (aksi hâlde 1.9 pt taşma)
    print(f"T12 yazildi (caption {wc(caption)} kelime, dipnot {wc(notes)} kelime)")


if __name__ == "__main__":
    t7()
    t8()
    t9()
    t10()
    t11()
    t12()
