"""Tidy3D bağlantı doğrulama yardımcısı.

ÖNEMLİ: API anahtarını buraya YAZMA. Anahtarı bir kez terminalden gir:
    tidy3d configure            # (CLI) anahtarı sorar
  ya da:
    python -c "import tidy3d.web as web; web.configure('SENIN_API_ANAHTARIN')"
Anahtar ~/.tidy3d/config'e yazılır (bu dosyayı paylaşma/gitleme).

Sonra bu betik bağlantıyı ve kredini doğrular:
    python examples/tidy3d_setup.py
"""
import sys
try:
    import tidy3d.web as web
except ImportError:
    sys.exit("Tidy3D yok: pip install tidy3d")

print("Bağlantı test ediliyor...")
try:
    web.test()   # anahtar geçerliyse sessiz/başarılı
    print("✓ Bağlantı OK.")
    try:
        acc = web.account()
        print("Hesap / kredi:", acc)
    except Exception as e:
        print("(hesap bilgisi alınamadı:", e, ")")
except Exception as e:
    print("✗ Bağlantı başarısız:", e)
    print("Önce anahtarı yapılandır:  tidy3d configure   (ya da web.configure('...'))")
