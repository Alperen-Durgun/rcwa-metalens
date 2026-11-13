"""Pure-Python GDSII writer (Unicode-path safe).

gdstk's C `fopen` cannot open paths containing non-ASCII/Unicode characters on Windows
(e.g. accented letters in a folder name). This module writes the GDSII binary format
directly with Python's `open()` → no code-page restriction, no temporary file/move needed.

GDSII is a record-based, big-endian format. Reference: Calma GDSII Stream Format.
Verification: the written file is read back with gdstk and the polygon count/geometry
is compared (test_gdswriter_roundtrip).
"""
import struct
import datetime
import numpy as np
from .shapes import polygon


def _rec(rtype, dtype, data=b""):
    return struct.pack(">HBB", 4 + len(data), rtype, dtype) + data


def _real8(value):
    """GDSII 8-byte real number (excess-64 exponent, base-16 mantissa)."""
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
    if mantissa >= (1 << 56):                      # rounding overflow
        mantissa >>= 4; exponent += 1
    return bytes([sign | (exponent & 0x7F)]) + mantissa.to_bytes(7, "big")


def _ascii(s):
    b = s.encode("ascii", "replace")
    if len(b) % 2:
        b += b"\x00"
    return b


def _timestamp(t=None):
    """BGNLIB/BGNSTR date record. t=None: now; a datetime: that date (reproducible files)."""
    t = datetime.datetime.now() if t is None else t
    vals = [t.year, t.month, t.day, t.hour, t.minute, t.second] * 2
    return struct.pack(">12h", *vals)


def _atom_key(p):
    par = tuple(np.round(np.atleast_1d(p["params"]).astype(float), 6))
    return (p.get("shape", "circle"), par, round(float(p.get("angle", 0.0)), 9))


def write_gds(placements, filepath, cell_name="METALENS", layer=1, datatype=0,
              unit=1e-6, precision=1e-9, n_circle=48, hierarchical=True, timestamp=None):
    """placements: [{x,y,shape,params,angle}] (um). Writes a GDSII file in pure Python.
    unit=1e-6 (um user unit), precision=1e-9 (nm database unit).

    hierarchical=True (default): every DISTINCT meta-atom (shape, size, rotation) is written
    once as its own cell and the lens is a list of cell references (SREF). A metalens uses
    only a few dozen distinct atoms, so the file shrinks from ~415 to ~30 bytes per pillar
    (>10x) and loads much faster in layout tools. hierarchical=False writes one flat
    polygon per pillar (some simple viewers/converters prefer that).
    timestamp: None writes the current date; a datetime writes that date instead, so the same
    layout gives a byte-identical file (used by the recipe pipeline).
    Returns: number of pillars written (= polygons after flattening)."""
    dbscale = unit / precision                     # 1um -> 1000 db units (nm)
    ts = _timestamp(timestamp)
    r_bnd = _rec(0x08, 0x00); r_lay = _rec(0x0D, 0x02, struct.pack(">h", layer))
    r_dt = _rec(0x0E, 0x02, struct.pack(">h", datatype)); r_end = _rec(0x11, 0x00)

    def boundary(verts):
        xy = np.vstack([verts, verts[:1]])
        return r_bnd + r_lay + r_dt + _rec(0x10, 0x03, np.round(xy * dbscale).astype(">i4").tobytes()) + r_end

    n = len(placements)
    with open(filepath, "wb") as f:                # Python open -> Unicode-path safe
        f.write(_rec(0x00, 0x02, struct.pack(">h", 600)))                        # HEADER v600
        f.write(_rec(0x01, 0x02, ts))                                            # BGNLIB
        f.write(_rec(0x02, 0x06, _ascii("LIB")))                                 # LIBNAME
        f.write(_rec(0x03, 0x05, _real8(precision / unit) + _real8(precision)))  # UNITS
        if hierarchical and n:
            keys = {}; idx = np.empty(n, np.int64)
            for i, p in enumerate(placements):
                idx[i] = keys.setdefault(_atom_key(p), len(keys))
            prefix = cell_name[:20]
            names = [_ascii(f"{prefix}_{k:06d}") for k in range(len(keys))]      # fixed length
            for (shape, par, ang), k in keys.items():                            # one cell per atom
                verts = np.asarray(polygon(shape, par if len(par) > 1 else par[0], center=(0.0, 0.0),
                                           angle=ang, n_circle=n_circle), float)
                f.write(_rec(0x05, 0x02, ts) + _rec(0x06, 0x06, names[k]) + boundary(verts) + _rec(0x07, 0x00))
            f.write(_rec(0x05, 0x02, ts) + _rec(0x06, 0x06, _ascii(cell_name)))  # top cell
            ln = len(names[0])                     # SREF record: 4 + (4+ln) + 12 + 4 bytes
            tmpl = _rec(0x0A, 0x00) + _rec(0x12, 0x06, names[0]) + _rec(0x10, 0x03, b"\x00" * 8) + r_end
            table = np.frombuffer(b"".join(names), np.uint8).reshape(len(names), ln)
            xy = np.array([[p["x"], p["y"]] for p in placements], float)
            step = 200000                                                        # bounded memory
            for s in range(0, n, step):
                e = min(n, s + step)
                arr = np.tile(np.frombuffer(tmpl, np.uint8), (e - s, 1))
                arr[:, 8:8 + ln] = table[idx[s:e]]
                arr[:, 12 + ln:20 + ln] = np.round(xy[s:e] * dbscale).astype(">i4").view(np.uint8).reshape(e - s, 8)
                f.write(arr.tobytes())
            f.write(_rec(0x07, 0x00))                                            # ENDSTR
        else:
            f.write(_rec(0x05, 0x02, ts) + _rec(0x06, 0x06, _ascii(cell_name)))
            # circular pillars (angle=0) are processed VECTORIZED (speed)
            is_c = [p.get("shape", "circle") == "circle" and float(p.get("angle", 0.0)) == 0.0 for p in placements]
            circ = [p for p, c in zip(placements, is_c) if c]
            if circ:
                ang = np.linspace(0, 2 * np.pi, n_circle, endpoint=False)
                cx = np.array([p["x"] for p in circ]); cy = np.array([p["y"] for p in circ])
                rad = np.array([float(np.atleast_1d(p["params"])[0]) for p in circ]) / 2.0
                vx = cx[:, None] + rad[:, None] * np.cos(ang)[None, :]
                vy = cy[:, None] + rad[:, None] * np.sin(ang)[None, :]
                vx = np.concatenate([vx, vx[:, :1]], axis=1); vy = np.concatenate([vy, vy[:, :1]], axis=1)
                Mc, Pn = vx.shape
                inter = np.empty((Mc, Pn, 2), ">i4")
                inter[:, :, 0] = np.round(vx * dbscale); inter[:, :, 1] = np.round(vy * dbscale)
                head = r_bnd + r_lay + r_dt + struct.pack(">HBB", 4 + Pn * 8, 0x10, 0x03)
                rec = np.empty((Mc, len(head) + Pn * 8 + len(r_end)), np.uint8)
                rec[:, :len(head)] = np.frombuffer(head, np.uint8)
                rec[:, len(head):len(head) + Pn * 8] = inter.view(np.uint8).reshape(Mc, Pn * 8)
                rec[:, len(head) + Pn * 8:] = np.frombuffer(r_end, np.uint8)
                f.write(rec.tobytes())
            for pl, c in zip(placements, is_c):                                  # non-circular / rotated
                if not c:
                    f.write(boundary(np.asarray(polygon(pl["shape"], pl["params"], center=(pl["x"], pl["y"]),
                                                        angle=pl.get("angle", 0.0), n_circle=n_circle), float)))
            f.write(_rec(0x07, 0x00))                                            # ENDSTR
        f.write(_rec(0x04, 0x00))                                                # ENDLIB
    return n


