"""CubeGrid — the 3-D representative-volume-element (RVE) voxel substrate for the gingival-pocket
biofilm engine ("the Cube").

Implements the existing ``SpatialGrid`` ABC (the 1D-now/3D-later seam in ``sim/grid/base.py``), so
the existing agents, chemistry, and host couplings plug straight in — nothing else has to change.

Geometry — a unit cell of pocket biofilm, six named faces (the surfaces are where the action is):

    x = 0      ENAMEL_PELLICLE   hard tissue: acquired pellicle (adhesion API) + biomineralization
    x = nx-1   SOFT_TISSUE       host: keratinocytes / fibroblasts / blood + nerve / immune
    y = 0,ny-1 ADJACENT_BIOFILM  continuation into surrounding biofilm  (y axis is PERIODIC)
    z = 0      PERIODIC          deep unit-cell closure
    z = nz-1   ORAL_CAVITY       exposed: saliva flow, pH, nutrients, O2 (the open top)

Addressing: every voxel has a coordinate (i, j, k) AND a linear ``index``; ``index`` <-> ``coord``
plus per-face distance make each pixel relatable to every other (coordinate + full-graph addressing).

Per-voxel state (the "state calculation associated with each pixel"):
  * chemistry fields  — O2, pH, NO, nitrate/nitrite, sugar, lactate, heme, Ca, PO4, ... (Layer 3)
  * microbial composition — biomass per species (Layer 1 nodes)
  * structure — EPS matrix density + mineralized fraction (biomineralization)
Network role (frontier / connector / keystone / commensal) is derived from composition + the
coaggregation graph; it is not stored, it is computed (Layer 1, in context — see the connectome).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from sim.grid.base import Position, SpatialGrid
from sim.grid.grid_1d import CHEMISTRY_FIELDS as _CHEM_1D
from sim.species import CATALOG

# Chemistry = the 1-D field set + the biomineralization substrates delivered at the enamel face,
# + "peptide", the pocket's OTHER nutrient economy and the one the deep zone actually runs on.
# Gingival crevicular fluid is a serum transudate, so the red complex is fed protein by the HOST
# rather than sugar by the diet. Cube-only; the 1-D sim keeps its own field list untouched.
CHEMISTRY_FIELDS: tuple[str, ...] = _CHEM_1D + ("Ca", "PO4", "peptide", "hemoprotein", "urea")
STRUCTURE_FIELDS: tuple[str, ...] = ("eps", "mineral")          # EPS matrix density, mineral fraction
DEFAULT_SPECIES_IDS: tuple[str, ...] = tuple(s.id for s in CATALOG)


class CubeGrid(SpatialGrid):
    """A 3-D voxel lattice (nx × ny × nz) carrying chemistry, composition, and structure per voxel."""

    def __init__(self, nx: int = 50, ny: int = 50, nz: int = 50, *,
                 cell_size_um: float = 20.0,
                 species_ids: Sequence[str] = DEFAULT_SPECIES_IDS) -> None:
        if min(nx, ny, nz) < 1:
            raise ValueError("cube dimensions must be >= 1")
        self._shape = (int(nx), int(ny), int(nz))
        self.cell_size_um = float(cell_size_um)
        self.species_ids = tuple(species_ids)
        self._sidx = {sid: n for n, sid in enumerate(self.species_ids)}
        self._fields: dict[str, np.ndarray] = {
            name: np.zeros(self._shape, dtype=np.float64)
            for name in CHEMISTRY_FIELDS + STRUCTURE_FIELDS
        }
        # composition[i, j, k, s] = biomass of species s at voxel (i, j, k)
        self._composition = np.zeros(self._shape + (len(self.species_ids),), dtype=np.float64)
        self._init_chemistry()

    def _init_chemistry(self) -> None:
        """O2 enters at the oral-cavity face (z = nz-1) and decays into depth; pH neutral."""
        _, _, nz = self._shape
        k = np.arange(nz)
        d_from_oral = (nz - 1) - k                     # 0 at the oral face, large deep
        o2_col = 8.0 * np.exp(-d_from_oral / 8.0)
        self._fields["O2"][:, :, :] = o2_col[np.newaxis, np.newaxis, :]
        self._fields["pH"][:] = 6.8

    # ---------------------------------------------------------------- SpatialGrid ABC
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def n_cells(self) -> int:
        return int(np.prod(self._shape))

    def positions(self) -> Iterable[Position]:
        nx, ny, nz = self._shape
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    yield (i, j, k)

    def get_neighbors(self, pos: Position, include_self: bool = True) -> list[Position]:
        """6-neighbor (von Neumann): x bounded (enamel/soft tissue), y periodic (adjacent biofilm),
        z bounded (oral top open, deep bottom closed)."""
        i, j, k = pos
        nx, ny, nz = self._shape
        out: list[Position] = [(i, j, k)] if include_self else []
        if i > 0:
            out.append((i - 1, j, k))
        if i < nx - 1:
            out.append((i + 1, j, k))
        if ny > 1:                                     # y periodic — adjacent-biofilm continuation
            out.append((i, (j - 1) % ny, k))
            out.append((i, (j + 1) % ny, k))
        if k > 0:
            out.append((i, j, k - 1))
        if k < nz - 1:
            out.append((i, j, k + 1))
        return out

    def get_field(self, name: str) -> np.ndarray:
        return self._fields[name]

    def get_field_at(self, name: str, pos: Position) -> float:
        return float(self._fields[name][pos])

    def add_to_field_at(self, name: str, pos: Position, delta: float) -> None:
        self._fields[name][pos] += delta

    def depth_fraction(self, pos: Position) -> float:
        nz = self._shape[2]
        return 0.0 if nz <= 1 else 1.0 - pos[2] / (nz - 1)   # 0 = oral surface, 1 = deep

    def is_top_boundary(self, pos: Position) -> bool:
        return pos[2] == self._shape[2] - 1                  # oral-cavity face

    def is_bottom_boundary(self, pos: Position) -> bool:
        return pos[2] == 0                                   # deep / periodic face

    # ---------------------------------------------------------------- addressing
    def index(self, pos: Position) -> int:
        """Linear address of a voxel — relatable to every other voxel (coordinate + index)."""
        i, j, k = pos
        _, ny, nz = self._shape
        return (i * ny + j) * nz + k

    def coord(self, index: int) -> Position:
        """Inverse of :meth:`index`."""
        _, ny, nz = self._shape
        k = index % nz
        j = (index // nz) % ny
        i = index // (nz * ny)
        return (i, j, k)

    # ---------------------------------------------------------------- composition (Layer 1 nodes)
    def composition_at(self, pos: Position) -> dict[str, float]:
        return {sid: float(self._composition[pos + (n,)]) for sid, n in self._sidx.items()}

    def add_biomass_at(self, pos: Position, species_id: str, delta: float) -> None:
        self._composition[pos + (self._sidx[species_id],)] += delta

    def total_biomass(self) -> np.ndarray:
        """(nx, ny, nz) total microbial biomass per voxel."""
        return self._composition.sum(axis=3)

    @property
    def composition(self) -> np.ndarray:
        """(nx, ny, nz, n_species) biomass array (mutable view) — for vectorized seeding/solve."""
        return self._composition

    def species_index(self, species_id: str) -> int:
        return self._sidx[species_id]

