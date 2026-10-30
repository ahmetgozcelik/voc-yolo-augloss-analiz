# -*- coding: utf-8 -*-
"""
vortak.py — varyant sarmalayıcıları (varyantlar/h0X_*/kos.py) için ORTAK yardımcılar.

Yalnız damga + gözlem: `scripts/rakip_kos.py` (baseline koşucusu) DEĞİŞTİRİLMEDEN aynı CLI argümanlarıyla
çağrılır; eğitim davranışını yalnız varyantın KENDİ yaması değiştirir. Bu modül şunları yapar:
  1. {run-dir}/config.json'a "varyant_imzasi" (değişen tek şeyin açık ifadesi) yazar,
  2. rakip_kos.main()'i çağırır (sys.argv olduğu gibi),
  3. bitince metrics.json'a "varyant_imzasi" + "varyant_etki_sayaci" (yamanın DEVREYE GİRDİĞİNİN sayısal
     kanıtı; 0 ise "bit-bit baseline ile aynı" şüphesi — beklenti kapısı `ikincil_aralik alt=1` ile yakalar)
     ve varyanta özgü düz sayısal anahtarları ekler.
"""
import json
import os
import sys
from pathlib import Path

WS_KOK = Path(__file__).resolve().parents[2]
SCRIPTS = WS_KOK / "scripts"


def argv_deger(flag, default=None, argv=None):
    argv = sys.argv if argv is None else argv
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


def argv_cikar(flag, argv=None):
    """`flag değer` veya `flag=değer` çiftini argv'den (yerinde) çıkarır; değeri döndürür (yoksa None).
    rakip_kos.py'nin argparse'ı bilmediği varyant bayraklarını ona iletmemek için."""
    argv = sys.argv if argv is None else argv
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            v = argv[i + 1]
            del argv[i:i + 2]
            return v
        if a.startswith(flag + "="):
            v = a.split("=", 1)[1]
            del argv[i]
            return v
    return None


def atomik_yaz(yol: Path, veri):
    yol = Path(yol)
    yol.parent.mkdir(parents=True, exist_ok=True)
    tmp = yol.with_suffix(yol.suffix + ".tmp")
    tmp.write_text(json.dumps(veri, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, yol)


def calistir(imza: str, toplayici=None):
    """imza: değişen tek şeyin açık ifadesi. toplayici: () -> {anahtar: sayı} (varyanta özgü düz metrikler;
    'varyant_etki_sayaci' zorunlu anahtar)."""
    run_dir = Path(argv_deger("--run-dir", ".")).resolve()
    out = Path(argv_deger("--out")).resolve() if argv_deger("--out") else run_dir / "metrics.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = run_dir / "config.json"
    if cfg.exists():
        try:
            c = json.loads(cfg.read_text(encoding="utf-8"))
            c["varyant_imzasi"] = imza
            atomik_yaz(cfg, c)
        except Exception as e:  # config yazımı eğitimi engellemez
            print(f"[vortak] config.json imza yazılamadı (yok sayıldı): {e}")
    print(f"[vortak] varyant_imzasi: {imza}")

    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import rakip_kos  # noqa: E402  (main() yalnız çağrılınca çalışır)
    rakip_kos.main()

    if not out.exists():
        print("[vortak] UYARI: metrics.json yok; damga yazılamadı")
        return 1
    m = json.loads(out.read_text(encoding="utf-8"))
    m["varyant_imzasi"] = imza
    ek = toplayici() if toplayici else {}
    if "varyant_etki_sayaci" not in ek:
        ek["varyant_etki_sayaci"] = 0
    m.update(ek)
    atomik_yaz(out, m)
    print(f"[vortak] metrics.json damgalandı: varyant_etki_sayaci={m['varyant_etki_sayaci']} ek={ek}")
    return 0
