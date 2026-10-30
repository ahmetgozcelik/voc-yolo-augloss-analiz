"""\\ref / \\eqref sonrası Türkçe ekleri, referansın derlenmiş numarasının okunuşuna göre düzeltir (ünlü uyumu ve
ünsüz sertleşmesi). MW kökünden: python evidence/scripts/ref_ek_duzelt.py [--yaz]
Numaralar paper/main.aux'tan okunur (önce derleyin). --yaz yoksa yalnız raporlar.
"""
import re
import sys
from pathlib import Path

MW = Path(__file__).resolve().parents[2]
TEX = MW / "paper" / "main.tex"
AUX = MW / "paper" / "main.aux"

# sayı/harf okunuşunun son sözcüğü
BIRLER = {0: "sıfır", 1: "bir", 2: "iki", 3: "üç", 4: "dört", 5: "beş", 6: "altı", 7: "yedi", 8: "sekiz", 9: "dokuz"}
ONLAR = {1: "on", 2: "yirmi", 3: "otuz", 4: "kırk", 5: "elli", 6: "altmış", 7: "yetmiş", 8: "seksen", 9: "doksan"}
HARF = {c: c.lower() for c in "ABCDEFG"}  # Ek A → "a" okunur; A: "a" (kalın, ünlüyle biter)


def son_sozcuk(numara):
    son = re.split(r"[.\-]", numara.strip())[-1]
    rakam = re.search(r"(\d+)$", son)  # "A5" gibi ek etiketlerinde okunuş rakamla biter
    if rakam:
        son = rakam.group(1)
    if son.isdigit():
        n = int(son)
        if n == 0:
            return BIRLER[0]
        if n % 100 == 0:
            return "yüz"
        if n % 10:
            return BIRLER[n % 10]
        return ONLAR[(n // 10) % 10]
    if son and son[-1].upper() in HARF:
        return {"a": "a", "b": "be", "c": "ce", "d": "de", "e": "e", "f": "fe", "g": "ge"}[son[-1].lower()]
    return None


KALIN, INCE = set("aıou"), set("eiöü")
SERT = set("fstkçşhp")


def ozellik(soz):
    unluler = [c for c in soz if c in KALIN | INCE]
    son_unlu = unluler[-1]
    return {"kalin": son_unlu in KALIN, "yuvarlak": son_unlu in "ouöü", "son_unlu_mu": soz[-1] in KALIN | INCE,
            "sert": soz[-1] in SERT, "son_unlu": son_unlu}


def ek_uret(eski, soz):
    o = ozellik(soz)
    a = "a" if o["kalin"] else "e"
    D = "t" if o["sert"] else "d"
    i4 = {"a": "ı", "ı": "ı", "o": "u", "u": "u", "e": "i", "i": "i", "ö": "ü", "ü": "ü"}[o["son_unlu"]]
    e = eski
    # bulunma (+ki / +dir) ve ayrılma
    m = re.fullmatch(r"[dt][ae](ki|d[ıiuü]r|n)?", e)
    if m:
        kuyruk = m.group(1) or ""
        if kuyruk.startswith("d") and kuyruk != "n":
            kuyruk = "d" + ("ı" if a == "a" else "i") + "r"  # -dır, önceki -da/-de ekine uyar
        return D + a + kuyruk
    # yönelme
    if re.fullmatch(r"y?[ae]", e):
        return ("y" if o["son_unlu_mu"] else "") + a
    # ilgi
    if re.fullmatch(r"n?[ıiuü]n", e):
        return ("n" if o["son_unlu_mu"] else "") + i4 + "n"
    # belirtme
    if re.fullmatch(r"y?[ıiuü]", e):
        return ("y" if o["son_unlu_mu"] else "") + i4
    return None


def main():
    yaz = "--yaz" in sys.argv
    numara = {}
    for aux in (AUX, AUX.parent / "cevrimici_ek.aux"):  # çevrimiçi ek etiketleri de (xr-hyper)
        if aux.exists():
            numara.update(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux.read_text(encoding="utf-8", errors="replace")))
    desen = re.compile(r"(\\(?:eq)?ref\{([^}]+)\}\)?)(['’])([a-zçğıöşü]+)")
    degisen = []

    def degistir(m):
        etiket, kesme, eski = m.group(2), m.group(3), m.group(4)
        no = numara.get(etiket)
        if no is None:
            return m.group(0)
        soz = son_sozcuk(no)
        yeni = ek_uret(eski, soz) if soz else None
        if yeni and yeni != eski:
            degisen.append((etiket, no, eski, yeni))
            return m.group(1) + kesme + yeni
        return m.group(0)

    for tex in [TEX] + sorted(TEX.parent.glob("ek_*.tex")):
        s = tex.read_text(encoding="utf-8")
        s2 = desen.sub(degistir, s)
        if yaz and s2 != s:
            tex.write_text(s2, encoding="utf-8", newline="\n")
    for d in degisen:
        print(f"{d[0]} ({d[1]}): '{d[2]} → '{d[3]}")
    print(f"toplam {len(degisen)} değişiklik")


if __name__ == "__main__":
    main()
