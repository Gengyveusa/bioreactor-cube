"""The Cube — a 3-D representative-volume-element voxel engine for the gingival-pocket biofilm.

Built on the existing ``sim.grid.SpatialGrid`` ABC (the 1D-now/3D-later seam). v1 = the lattice +
addressing (``CubeGrid``), the six-face surfaces module (``faces``), and the per-voxel state map
(``state``). See ``CUBE-DESIGN.md`` for the architecture and roadmap.
"""

from sim.cube.faces import (
    FACES,
    Face,
    FaceConditions,
    FaceKind,
    apply_face_boundaries,
    face_distance,
    face_influence,
)
from sim.cube.grid import (
    CHEMISTRY_FIELDS,
    DEFAULT_SPECIES_IDS,
    STRUCTURE_FIELDS,
    CubeGrid,
)
from sim.cube.state import VoxelState, read_voxel

__all__ = [
    "CubeGrid",
    "CHEMISTRY_FIELDS",
    "STRUCTURE_FIELDS",
    "DEFAULT_SPECIES_IDS",
    "FACES",
    "Face",
    "FaceKind",
    "FaceConditions",
    "apply_face_boundaries",
    "face_distance",
    "face_influence",
    "VoxelState",
    "read_voxel",
]

