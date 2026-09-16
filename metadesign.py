"""E9 — HAFİF CLI BAŞLATICI (Obsidian dışı / hızlı kullanım).

GUI yerine: 19 otomasyon üretecini terminalden çalıştır. Obsidian otomasyonu
(form→çalıştır→rapor) asıl arayüzümüz; bu, hızlı/betiksel kullanım içindir.

Kullanım (.venv, rcwa_py klasöründen):
  python metadesign.py --list                       # deney tiplerini listele
  python metadesign.py metalens_design_full D_lens=6 f=6 material=TiO2_t
  python metadesign.py grating period=1.2 duty=0.5
  python metadesign.py --menu                        # etkileşimli menü
"""
import sys, os
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _parse_kv(args):
    fm = {}
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            fm[k] = v   # run_experiment.num() sayıya çevirir; string alanlar aynen kalır
    return fm


def main(argv):
    import run_experiment as R
    names = list(R.RUNNERS.keys())
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); print("Tipler:", ", ".join(names)); return 0
    if argv[0] == "--list":
        print("Mevcut deney tipleri (%d):" % len(names))
        for n in names:
            print("  -", n)
        return 0
    if argv[0] == "--menu":
        for i, n in enumerate(names):
            print(f"[{i:2d}] {n}")
        try:
            sel = int(input("Seç (numara): ").strip()); name = names[sel]
        except Exception:
            print("Geçersiz seçim."); return 1
        raw = input("Parametreler (key=val boşlukla, boş=varsayılan): ").strip()
        fm = _parse_kv(raw.split()) if raw else {}
    else:
        name = argv[0]
        if name not in R.RUNNERS:
            print(f"Bilinmeyen tip: {name}\nTipler: {', '.join(names)}"); return 1
        fm = _parse_kv(argv[1:])
    slug = "CLI_" + name
    print(f"Çalışıyor: {name}  parametre={fm or 'varsayılan'}")
    fig, body, summary = R.RUNNERS[name](fm, slug)
    print("\n=== SONUÇ ===\n" + body)
    print(f"\nÖzet: {summary}")
    print(f"Figür: {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
