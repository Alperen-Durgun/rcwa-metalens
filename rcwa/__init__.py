"""Öğretici RCWA kütüphanesi — 1B ve 2B (TE + TM).
1B/2B çözücü, kararlı otomatik çözücü, dispersif malzeme, önbellek.
"""
from .convmat import convmat1d, convmat2d
from .redheffer import redheffer_star
from .layer import homogeneous_modes, grating_modes, modes_from_ER, modes_from_ER_cached, layer_smatrix
from .solver import solve_rcwa_1d
from .solver2d import solve_rcwa_2d
from .stable import solve_1d_auto, solve_2d_auto
from .solver1d_scalar import solve_scalar_1d
from . import materials
from . import cache
from . import metrics
from . import shapes
from . import gds
from . import fullwave
from .parallel import parallel_map

__all__ = ["convmat1d", "convmat2d", "redheffer_star", "homogeneous_modes",
           "grating_modes", "modes_from_ER", "layer_smatrix",
           "solve_rcwa_1d", "solve_rcwa_2d", "solve_1d_auto", "solve_2d_auto", "solve_scalar_1d",
           "materials", "cache", "metrics", "shapes", "gds", "fullwave", "modes_from_ER_cached", "parallel_map"]
__version__ = "0.7.0"
