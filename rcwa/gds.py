"""GDS-II export — Item 2 (fabrication bridge).
Converts the designed pillar map into a manufacturable GDS file (lithography mask).

WRITE: pure-Python GDSII writer (`gdswriter.write_gds`) — Unicode/non-ASCII path
safe, no gdstk required, no temp file/move. READ (validation): gdstk (if available).
"""
import numpy as np
from .shapes import polygon


def layout_to_gds(placements, filepath, cell_name="METALENS", layer=1, datatype=0, unit=1e-6):
    """placements: [{x,y,shape,params,angle}] (um). Writes a GDS file (pure-Python).
    unit=1e-6 -> coordinates in microns. Returns: number of polygons written."""
    from .gdswriter import write_gds
    return write_gds(placements, filepath, cell_name=cell_name, layer=layer,
                     datatype=datatype, unit=unit)


def read_gds_summary(filepath):
    """Reads the GDS back and returns (cell count, polygon count) — for validation.
    The gdstk C reader cannot open a Unicode path; for a Unicode path, copy to an
    ASCII temp file FIRST (so gdstk never sees a Unicode path → no error message)."""
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
