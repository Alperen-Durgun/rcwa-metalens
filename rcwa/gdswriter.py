"""Saf-Python GDSII yazıcı (Unicode-yol güvenli).

gdstk'nın C `fopen`'ı Windows'ta Türkçe/Unicode karakterli yolu (ör. 'Kişisel_Merkez'
içindeki 'ş') açamaz. Bu modül GDSII ikili biçimini doğrudan Python'un `open()`'ıyla
yazar → hiçbir kod-sayfası kısıtı yok, hiçbir geçici dosya/taşıma gerekmez.

GDSII kayıt-tabanlı, big-endian bir biçimdir. Referans: Calma GDSII Stream Format.
Doğrulama: yazılan dosya gdstk ile geri okunup poligon sayısı/geometri karşılaştırılır
(test_gdswriter_roundtrip).
"""
import struct
import datetime
import numpy as np
from .shapes import polygon


def _rec(rtype, dtype, data=b""):
    return struct.pack(">HBB", 4 + len(data), rtype, dtype) + data


def _real8(value):
    """GDSII 8-byte gerçek sayı (excess-64 üs, taban-16 mantis)."""
    if value == 0:
        return b"\x00" * 8
    sign = 0
    if value < 0:
        sign = 0x80; value = -value
    exponent = 64
    while value >= 1.0:
        value /= 16.0; exponent += 1
    while value < 1.0 / 16.0:
        value *= 16.0; exponent -= 1
    mantissa = int(round(value * (1 << 56)))
    if mantissa >= (1 << 56):                      # yuvarlama taşması
        mantissa >>= 4; exponent += 1
    return bytes([sign | (exponent & 0x7F)]) + mantissa.to_bytes(7, "big")


def _ascii(s):
    b = s.encode("ascii", "replace")
    if len(b) % 2:
        b += b"\x00"
    return b


def _timestamp():
    t = datetime.datetime.now()
    vals = [t.year, t.month, t.day, t.hour, t.minute, t.second] * 2
    return struct.pack(">12h", *vals)


def write_gds(placements, filepath, cell_name="METALENS", layer=1, datatype=0,
              unit=1e-6, precision=1e-9, n_circle=48):
    """placements: [{x,y,shape,params,angle}] (um). Saf-Python GDSII yazar.
    unit=1e-6 (um kullanıcı birimi), precision=1e-9 (nm veritabanı birimi).
    Döndürür: yazılan poligon (BOUNDARY) sayısı."""
    dbscale = unit / precision                     # 1um -> 1000 db birimi (nm)
    ts = _timestamp()
    out = bytearray()
    out += _rec(0x00, 0x02, struct.pack(">h", 600))                 # HEADER v600
    out += _rec(0x01, 0x02, ts)                                     # BGNLIB
    out += _rec(0x02, 0x06, _ascii("LIB"))                          # LIBNAME
    out += _rec(0x03, 0x05, _real8(precision / unit) + _real8(precision))  # UNITS
    out += _rec(0x05, 0x02, ts)                                     # BGNSTR
    out += _rec(0x06, 0x06, _ascii(cell_name))                     # STRNAME
    # sabit kayıtlar (her poligonda aynı) — önceden hazırla
    r_bnd = _rec(0x08, 0x00); r_lay = _rec(0x0D, 0x02, struct.pack(">h", layer))
    r_dt = _rec(0x0E, 0x02, struct.pack(">h", datatype)); r_end = _rec(0x11, 0x00)
    # dairesel sütunları (angle=0) VEKTÖRİZE işle (hız): büyük lens için kritik
    circ = [p for p in placements if p.get("shape", "circle") == "circle" and float(p.get("angle", 0.0)) == 0.0]
    other = [p for p in placements if not (p.get("shape", "circle") == "circle" and float(p.get("angle", 0.0)) == 0.0)]
    n = 0
    if circ:
        ang = np.linspace(0, 2 * np.pi, n_circle, endpoint=False)
        uc = np.stack([np.cos(ang), np.sin(ang)], axis=1)           # (n_circle,2)
        cx = np.array([p["x"] for p in circ]); cy = np.array([p["y"] for p in circ])
        rad = np.array([float(np.atleast_1d(p["params"])[0]) for p in circ]) / 2.0
        # (M, n_circle+1, 2): merkez + yarıçap*birim-çember, kapalı
        vx = cx[:, None] + rad[:, None] * uc[None, :, 0]
        vy = cy[:, None] + rad[:, None] * uc[None, :, 1]
        vx = np.concatenate([vx, vx[:, :1]], axis=1); vy = np.concatenate([vy, vy[:, :1]], axis=1)
        ix = np.round(vx * dbscale).astype(">i4"); iy = np.round(vy * dbscale).astype(">i4")
        M, P = ix.shape
        inter = np.empty((M, P, 2), ">i4"); inter[:, :, 0] = ix; inter[:, :, 1] = iy
        xy_bytes = inter.reshape(M, P * 2).tobytes()                # tüm poligonların XY'si
        xy_hdr = struct.pack(">HBB", 4 + P * 2 * 4, 0x10, 0x03)     # XY başlığı (sabit uzunluk)
        stride = P * 2 * 4
        chunks = []
        for k in range(M):
            chunks.append(r_bnd); chunks.append(r_lay); chunks.append(r_dt)
            chunks.append(xy_hdr); chunks.append(xy_bytes[k * stride:(k + 1) * stride]); chunks.append(r_end)
        out += b"".join(chunks); n += M
    for pl in other:                                                # dairesel-olmayan / döndürülmüş
        verts = np.asarray(polygon(pl["shape"], pl["params"], center=(pl["x"], pl["y"]),
                                   angle=pl.get("angle", 0.0), n_circle=n_circle), float)
        xy = np.vstack([verts, verts[:1]])
        out += r_bnd + r_lay + r_dt + _rec(0x10, 0x03, np.round(xy * dbscale).astype(">i4").tobytes()) + r_end
        n += 1
    out += _rec(0x07, 0x00)                                         # ENDSTR
    out += _rec(0x04, 0x00)                                         # ENDLIB
    with open(filepath, "wb") as f:                                # Python open -> Unicode güvenli
        f.write(out)
    return n
