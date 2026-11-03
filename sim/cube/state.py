"""Per-voxel state — read the full 'state calculation associated with each pixel', with its address
and its distance to every named surface. The addressable per-voxel state map is the Cube's output.
"""

from __future__ import annotations

from dataclasses import dataclass

from sim.cube.faces import FACES
from sim.cube.grid import CHEMISTRY_FIELDS, STRUCTURE_FIELDS, CubeGrid


@dataclass(frozen=True)
class VoxelState:
    """The complete state of one voxel + its address, relatable to all others via ``index``."""

    address: tuple[int, int, int]
    index: int
    chemistry: dict[str, float]        # O2, pH, NO, nitrate/nitrite, sugar, lactate, heme, Ca, PO4, ...
    composition: dict[str, float]      # species_id -> biomass (only present species)
    structure: dict[str, float]        # eps, mineral
    total_biomass: float
    dominant_species: str | None
    face_distance: dict[str, int]      # lattice distance to each named face (the surface relation)


def read_voxel(grid: CubeGrid, pos: tuple[int, int, int]) -> VoxelState:
    """Compute the full state of one voxel — the unit of the addressable 3-D state map."""
    chem = {f: grid.get_field_at(f, pos) for f in CHEMISTRY_FIELDS}
    struct = {f: grid.get_field_at(f, pos) for f in STRUCTURE_FIELDS}
    comp_all = grid.composition_at(pos)
    comp = {sid: b for sid, b in comp_all.items() if b > 0}
    total = float(sum(comp_all.values()))
    dominant = max(comp, key=comp.get) if comp else None

    face_dist: dict[str, int] = {}
    for face in FACES:
        n = grid.shape[face.axis]
        idx = pos[face.axis]
        face_dist[face.label] = int(idx if face.side == 0 else (n - 1 - idx))

    return VoxelState(
        address=tuple(pos),
        index=grid.index(pos),
        chemistry=chem,
        composition=comp,
        structure=struct,
        total_biomass=total,
        dominant_species=dominant,
        face_distance=face_dist,
    )

