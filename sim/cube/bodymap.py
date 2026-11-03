"""The connected body-map — the systemic thesis (Layer-7 propagation) realised spatially.

Cubes are nodes; the BLOODSTREAM is the edge. Each tick every cube advances its own Epoch-4 loop, then
each edge carries the source cube's systemic efflux (e.g. the oral cube's OMV load crossing its
soft-tissue/gum face) into the destination cube as an arrival (e.g. inflaming the artery's endothelium).
A dysbiotic oral cube therefore drives a *distant* artery cube toward plaque — oral → systemic, in space.

This is not a metaphor in the data: the dose that moves is the oral cube's real OMV efflux, and the
artery's plaque is the real emergent output of its own information-primary loop responding to it.
"""

from __future__ import annotations

from sim.cube.fused import CubeEpoch4
from sim.cube.grid import CubeGrid
from sim.cube.tissues import CubeTissue


class BodyMap:
    """A graph of tissue cubes linked by the bloodstream."""

    def __init__(self) -> None:
        self.cubes: dict[str, CubeEpoch4] = {}
        self.edges: list[tuple[str, str, float]] = []

    def place(self, name: str, tissue: CubeTissue, dim: int = 24, seed: bool = True) -> CubeEpoch4:
        """Drop a cube of `tissue` into the body at `name` (liver/brain/artery/oral…)."""
        grid = CubeGrid(dim, dim, dim, species_ids=tissue.cell_ids)
        eng = CubeEpoch4(grid, tissue)
        if seed:
            eng.seed_uniform()
        self.cubes[name] = eng
        return eng

    def link(self, src: str, dst: str, k: float = 1.0) -> None:
        """A bloodstream edge: src's efflux arrives at dst, scaled by transfer coefficient k."""
        self.edges.append((src, dst, k))

    def step(self, dt: float = 0.1) -> None:
        for eng in self.cubes.values():
            eng.step(dt)
        for src, dst, k in self.edges:
            self.cubes[dst].receive(self.cubes[src].efflux() * k)

    def run(self, steps: int, dt: float = 0.1, sample_every: int = 20) -> list[dict]:
        traj: list[dict] = []
        for t in range(1, steps + 1):
            self.step(dt)
            if t % sample_every == 0 or t == steps:
                snap: dict = {"t": round(t * dt, 1)}
                for name, eng in self.cubes.items():
                    r = eng.readout()
                    snap[name] = {k: v for k, v in r.items() if k != "fields"}
                traj.append(snap)
        return traj

