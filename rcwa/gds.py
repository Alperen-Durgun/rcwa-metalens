"""GDS-II dışa aktarma — Madde 2 (üretim köprüsü).
Tasarlanan sütun haritasını üretilebilir GDS dosyasına çevirir (litografi maskesi).

YAZIM: saf-Python GDSII yazıcı (`gdswriter.write_gds`) — Unicode/Türkçe yol güvenli,
gdstk gerektirmez, geçici dosya/taşıma yok. OKUMA (doğrulama): gdstk (varsa).
"""
import numpy as np
from .shapes import polygon


def layout_to_gds(placements, filepath, cell_name="METALENS", layer=1, datatype=0, unit=1e-6):
    """placements: [{x,y,shape,params,angle}] (um). GDS dosyası yazar (saf-Python).
    unit=1e-6 -> koordinatlar mikron. Döndürür: yazılan polygon sayısı."""
    from .gdswriter import write_gds
    return write_gds(placements, filepath, cell_name=cell_name, layer=layer,
                     datatype=datatype, unit=unit)


def read_gds_summary(filepath):
    """GDS'i geri okuyup (hücre sayısı, polygon sayısı) döndürür — doğrulama için.
    gdstk C-okuyucu Unicode yolu açamaz; Unicode yolda ÖNCEDEN ASCII temp'e kopyala
    (böylece gdstk hiç Unicode yol denemez → hata mesajı çıkmaz)."""
    import gdstk, os
    tmp = None; path = filepath
    if not all(ord(c) < 128 for c in filepath):
        import tempfile, shutil
        fd, tmp = tempfile.mkstemp(suffix=".gds"); os.close(fd)
        shutil.copy(filepath, tmp); path = tmp
    lib = gdstk.read_gds(path)
    if tmp:
        os.remove(tmp)
    npoly = sum(len(c.polygons) for c in lib.cells)
    return {"cells": len(lib.cells), "polygons": npoly}
