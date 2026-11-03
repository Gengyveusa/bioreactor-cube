"""1D depth-column grid — v1 spatial substrate.

Index 0 is the saliva-facing top; index (N-1) is the gingival sulcus / junctional
epithelium at the bottom. The O2 / heme gradient runs along this axis, which is
the dominant axis per Layer 3.
"""
from __future__ import annotations
from typing import Iterable
import numpy as np

from .base import SpatialGrid, Position


CHEMISTRY_FIELDS = (
    "O2", "pH", "sugar", "heme",
    "lactate", "NO", "H2S", "NH3",
    "propionate", "butyrate", "acetate",
    "nitrate", "nitrite",   # enterosalivary nitrogen / NO axis
)


class SpatialGrid1D(SpatialGrid):
    def __init__(self, depth_cells: int = 50, cell_size_um: float = 20.0):
        self._N = int(depth_cells)
        self._cell_size_um = cell_size_um
        # Fields: dict[name -> ndarray of shape (N,)]
        self._fields: dict[str, np.ndarray] = {
            name: np.zeros(self._N, dtype=np.float64) for name in CHEMISTRY_FIELDS
        }
        # Sensible initial conditions:
        # O2 decreasing from saliva (top) to depth — exponential decay
        depth_idx = np.arange(self._N)
        self._fields["O2"][:] = 8.0 * np.exp(-depth_idx / 8.0)
        # pH neutral
        self._fields["pH"][:] = 6.8
        # No sugar, no heme initially — Stephan curve and bleed are events
        # No metabolites initially — agents will produce them

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._N,)

    @property
    def n_cells(self) -> int:
        return self._N

    def positions(self) -> Iterable[Position]:
        for i in range(self._N):
            yield (i,)

    def get_neighbors(self, pos: Position, include_self: bool = True) -> list[Position]:
        (i,) = pos
        out: list[Position] = []
        if include_self:
            out.append((i,))
        if i > 0:
            out.append((i - 1,))
        if i < self._N - 1:
            out.append((i + 1,))
        return out

    def get_field(self, name: str) -> np.ndarray:
        return self._fields[name]

    def get_field_at(self, name: str, pos: Position) -> float:
        return float(self._fields[name][pos[0]])

    def add_to_field_at(self, name: str, pos: Position, delta: float) -> None:
        self._fields[name][pos[0]] += delta

    def depth_fraction(self, pos: Position) -> float:
        if self._N <= 1:
            return 0.0
        return pos[0] / (self._N - 1)

    def is_top_boundary(self, pos: Position) -> bool:
        return pos[0] == 0

    def is_bottom_boundary(self, pos: Position) -> bool:
        return pos[0] == self._N - 1

