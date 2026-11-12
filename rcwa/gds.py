"""GDS-II export — Item 2 (fabrication bridge).
Converts the designed pillar map into a manufacturable GDS file (lithography mask).

WRITE: pure-Python GDSII writer (`gdswriter.write_gds`) — Unicode/non-ASCII path
safe, no gdstk required, no temp file/move. READ (validation): gdstk (if available).
"""


def layout_to_gds(placements, filepath, cell_name="METALENS", layer=1, datatype=0, unit=1e-6,
                  hierarchical=True):
    """placements: [{x,y,shape,params,angle}] (um). Writes a GDS file (pure-Python).
    unit=1e-6 -> coordinates in microns. hierarchical=True stores each distinct meta-atom once
    and references it (>10x smaller files); False writes flat polygons.
    Returns: number of pillars written."""
    from .gdswriter import write_gds
    return write_gds(placements, filepath, cell_name=cell_name, layer=layer,
                     datatype=datatype, unit=unit, hierarchical=hierarchical)


def read_gds_summary(filepath):
    """Reads the GDS back -> {cells, polygons, references}; 'polygons' counts the FLATTENED
    layout (references expanded), so flat and hierarchical files compare equal.
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
    npoly = sum(len(c.get_polygons()) for c in lib.top_level())
    return {"cells": len(lib.cells), "polygons": npoly,
            "references": sum(len(c.references) for c in lib.cells)}
