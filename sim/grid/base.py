"""SpatialGrid abstract base — the 1D-now/3D-later seam.

Position is a tuple. For 1D: (depth,). For 3D: (x, y, depth).
Agents and chemistry never reach inside the grid directly; they go through this
interface. To upgrade to 3D, write a SpatialGrid3D that implements these methods
on a 3D ndarray. Nothing else changes.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable
import numpy as np


Position = tuple   # alias — (depth,) or (x, y, depth)


class SpatialGrid(ABC):
    """Abstract interface for any spatial substrate."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Cell count per axis."""

    @property
    @abstractmethod
    def n_cells(self) -> int:
        """Total number of cells (product of shape)."""

    @abstractmethod
    def positions(self) -> Iterable[Position]:
        """Iterate over every grid position."""

    @abstractmethod
    def get_neighbors(self, pos: Position, include_self: bool = True) -> list[Position]:
        """von Neumann neighborhood positions."""

    @abstractmethod
    def get_field(self, name: str) -> np.ndarray:
        """Return the full field array (mutable view)."""

    @abstractmethod
    def get_field_at(self, name: str, pos: Position) -> float:
        """Read one cell of a field."""

    @abstractmethod
    def add_to_field_at(self, name: str, pos: Position, delta: float) -> None:
        """Add delta to one cell of a field (source/sink emission)."""

    @abstractmethod
    def depth_fraction(self, pos: Position) -> float:
        """0.0 = saliva surface, 1.0 = junctional epithelium / bottom boundary.

        Agents use this to compare against species.preferred_depth.
        """

    @abstractmethod
    def is_top_boundary(self, pos: Position) -> bool: ...

    @abstractmethod
    def is_bottom_boundary(self, pos: Position) -> bool: ...

