"""Tests for the Cube — the 3-D RVE voxel substrate, addressing, and the six-face surfaces module."""

from __future__ import annotations

import numpy as np

from sim.cube import (
    FACES,
    CubeGrid,
    FaceConditions,
    FaceKind,
    apply_face_boundaries,
    face_distance,
    read_voxel,
)
from sim.grid.base import SpatialGrid


def test_cubegrid_is_a_spatialgrid_and_has_shape() -> None:
    g = CubeGrid(4, 5, 6)
    assert isinstance(g, SpatialGrid)          # plugs into the existing engine via the ABC
    assert g.shape == (4, 5, 6)
    assert g.n_cells == 120


def test_addressing_roundtrips_for_every_voxel() -> None:
    g = CubeGrid(3, 4, 5)
    seen = set()
    for pos in g.positions():
        idx = g.index(pos)
        assert g.coord(idx) == pos             # coordinate <-> linear index is a bijection
        seen.add(idx)
    assert seen == set(range(g.n_cells))       # every voxel has a unique address


def test_neighbors_y_periodic_x_and_z_bounded() -> None:
    g = CubeGrid(4, 4, 4)
    # interior voxel: 6 neighbors + self
    assert len(g.get_neighbors((1, 1, 1))) == 7
    # y wraps (adjacent-biofilm continuation): j=0 neighbors include j=ny-1
    ny = g.shape[1]
    ns = g.get_neighbors((1, 0, 1), include_self=False)
    assert (1, ny - 1, 1) in ns
    # x does NOT wrap (enamel / soft tissue are real walls): i=0 has no i=nx-1 neighbor
    assert (g.shape[0] - 1, 0, 1) not in g.get_neighbors((0, 0, 1), include_self=False)


def test_oxygen_gradient_high_at_oral_face_low_deep() -> None:
    g = CubeGrid(2, 2, 10)
    o2 = g.get_field("O2")
    assert o2[0, 0, -1] > o2[0, 0, 0]          # oral face (z=nz-1) richer than the deep face (z=0)
    assert g.depth_fraction((0, 0, g.shape[2] - 1)) == 0.0   # oral surface = depth 0
    assert g.depth_fraction((0, 0, 0)) == 1.0               # deep = depth 1


def test_six_faces_are_distinct_and_named() -> None:
    assert len(FACES) == 6
    kinds = [f.kind for f in FACES]
    assert FaceKind.ENAMEL_PELLICLE in kinds
    assert FaceKind.SOFT_TISSUE in kinds
    assert FaceKind.ORAL_CAVITY in kinds
    assert kinds.count(FaceKind.ADJACENT_BIOFILM) == 2       # two adjacent-biofilm faces


def test_surfaces_module_applies_each_face_boundary() -> None:
    g = CubeGrid(5, 5, 5)
    apply_face_boundaries(g, FaceConditions())
    # oral cavity (z = nz-1): O2 saturated, nitrate delivered
    assert np.allclose(g.get_field("O2")[:, :, -1], 8.0)
    assert np.allclose(g.get_field("nitrate")[:, :, -1], 0.5)
    # soft tissue (x = nx-1): serum pH 7.4 (away from the shared oral edge z=nz-1, where oral wins)
    assert np.allclose(g.get_field("pH")[-1, :, :-1], 7.4)
    # enamel (x = 0): Ca/PO4 supersaturation for biomineralization, anoxic wall (off the oral edge)
    assert np.allclose(g.get_field("Ca")[0, :, :], 1.0)        # oral doesn't set Ca → whole face holds
    assert np.allclose(g.get_field("O2")[0, :, :-1], 0.0)


def test_face_distance_is_zero_on_the_face_and_grows_inward() -> None:
    g = CubeGrid(6, 6, 6)
    enamel = next(f for f in FACES if f.kind is FaceKind.ENAMEL_PELLICLE)  # x=0
    d = face_distance(g, enamel)
    assert np.all(d[0, :, :] == 0)             # zero on the enamel face
    assert np.all(d[-1, :, :] == g.shape[0] - 1)  # max at the opposite (soft-tissue) face


def test_read_voxel_gives_address_state_and_surface_relations() -> None:
    g = CubeGrid(4, 4, 4)
    g.add_biomass_at((1, 1, 1), g.species_ids[0], 0.7)
    g.add_biomass_at((1, 1, 1), g.species_ids[1], 0.3)
    g.get_field("mineral")[1, 1, 1] = 0.2
    vs = read_voxel(g, (1, 1, 1))
    assert vs.address == (1, 1, 1) and vs.index == g.index((1, 1, 1))
    assert vs.dominant_species == g.species_ids[0]
    assert vs.total_biomass == 1.0
    assert vs.structure["mineral"] == 0.2
    assert "pH" in vs.chemistry
    assert set(vs.face_distance) >= {"enamel_pellicle", "soft_tissue", "oral_cavity"}

