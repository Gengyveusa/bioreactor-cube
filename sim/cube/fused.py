"""Fused engine — the Epoch-4 loop running ON the 3-D cube, now TISSUE-DRIVEN. (Design increments 2–3.)

The cube's per-voxel chemistry is the Terrain (T); per-voxel-per-cell expression is the Information
substrate (the thing that moves); biomass (composition) is what Energy grows; the readout is ψ. Spatial
structure is NOT placed — it EMERGES from terrain-gated expression (Information → T → E → I → Δ → ψ).

The engine is universal; a CubeTissue (sim/cube/tissues.py) supplies the cell roster, genomes, terrain,
transitions, and ψ — so the SAME engine runs 'oral' or 'artery' (or any tissue). Operators stay pinned:
    T  _expressibility(cid)        genome niche/demand vs local terrain fields  -> 0..1
    E  step(): express→grow→shed ; tissue.update_terrain (E→T) ; tissue.react (transitions)
    I  per-cell defense -> persistence ; predation deletes the un-expressible & un-defended
    Δ  step()                       one tick
    ψ  readout()                    tissue-specific system state
"""

from __future__ import annotations

import numpy as np

from sim.cube.grid import CubeGrid
from sim.cube.tissues import ORAL_TISSUE, CubeTissue, oral_prepare_terrain

DECAY = 0.2   # predation/clearance on un-expressible, un-defended information

# Re-exported for callers/tests that prepare an oral grid directly.
prepare_terrain = oral_prepare_terrain


class CubeEpoch4:
    """Runs the information-primary loop over a CubeGrid, parameterised by a CubeTissue."""

    def __init__(self, grid: CubeGrid, tissue: CubeTissue = ORAL_TISSUE) -> None:
        self.grid = grid
        self.tissue = tissue
        self.species = list(grid.species_ids)
        self.genomes = tissue.genomes
        self.expression = np.full(grid.composition.shape, 0.2, dtype=np.float64)
        self.extra: dict[str, np.ndarray] = {}     # tissue-specific terrain fields (oxLDL/shear/inflammation)
        self.t = 0.0                               # elapsed model time — a timed challenge needs a clock
        # [I] Host-immune clearance, per species id. None means the host is not modelled at all,
        # which is the pre-existing behaviour and stays the default: an unmeasured patient must not
        # silently acquire an immune system. Set via sim.cube.host.attach_host.
        self.mortality_mult: dict | None = None
        # [L5] The phage war. None means phages are not modelled, which is how every result before
        # sim/cube/phage.py existed was produced, so it stays the default.
        self.phage = None
        # Coaggregation dependence: late colonisers build on whoever is already there rather than
        # on clean pellicle. Its own flag rather than a property of the phage system, because it is
        # a property of the COMMUNITY and is needed by experiments that involve no phage at all.
        # Off by default: switching it on changes every existing result.
        self.coaggregation = False
        # [L1 in] Immigration from the salivary planktonic pool. None means an absorbing zero — a
        # species at zero can never return — which is the shipped behaviour and what every ablation
        # experiment in the harness relies on.
        self.dispersal = None
        tissue.prepare(self)

    def field(self, name: str) -> np.ndarray:
        """Terrain accessor: tissue extra-field if present, else the grid chemistry field."""
        return self.extra[name] if name in self.extra else self.grid.get_field(name)

    def seed_uniform(self) -> None:
        """Seed every cell thin & everywhere; let terrain-gated expression carve the structure."""
        comp = self.grid.composition
        comp[:] = 0.0
        for n, cid in enumerate(self.species):
            comp[..., n] = 0.02 * self.tissue.abundance.get(cid, 0.02)

    def _expressibility(self, cid: str) -> np.ndarray:
        """T — the gate as a field: gaussian niche windows × monotone demands vs local terrain."""
        g = self.genomes[cid]
        e = np.ones(self.grid.shape)
        for f, (opt, w) in g.get("niche", {}).items():
            e = e * np.exp(-((self.field(f) - opt) / w) ** 2)
        for f, strength in g.get("demand", {}).items():
            e = e * ((1.0 - strength) + strength * np.clip(self.field(f), 0.0, 1.0))
        return np.clip(e, 0.0, 1.0)

    def step(self, dt: float = 0.1) -> None:
        """Δ — one tick: gate (T) → express+grow (E) → predation by (1-T)(1-I) → terrain & transitions."""
        self.t += dt
        comp = self.grid.composition
        total = comp.sum(axis=3)
        # [I] Keystone subversion of host clearance. Computed ONCE per tick, before any species is
        # updated, so every species sees the same immune state — the keystone disables clearance
        # for the whole neighbourhood, which is the point. Only meaningful when a host is modelled.
        subvert = None
        if self.mortality_mult is not None:
            from sim.cube.host import clearance_scale
            subvert = clearance_scale(self)
        for n, cid in enumerate(self.species):
            g = self.genomes[cid]
            ex = self._expressibility(cid)                              # T
            m = self.expression[..., n]
            m += dt * 3.0 * (ex - m)
            np.clip(m, 0.0, 1.0, out=m)                                 # E: expression relaxes toward T
            b = comp[..., n]
            # Coaggregation: a late coloniser cannot build on clean pellicle, it builds on whoever
            # is already there. Off unless a PhageSystem asks for it, because switching it on
            # changes every existing result — see sim.cube.phage.coaggregation_gate.
            gate = 1.0
            if self.coaggregation:
                from sim.cube.phage import coaggregation_gate
                gate = coaggregation_gate(self, cid)
            comp[..., n] = b + dt * g["growth"] * m * gate * b * (1.0 - total)  # E: grow (logistic)
            persist = 0.5 + 0.5 * g["defense"]                         # I
            # Host-immune clearance scales the deletion term. The keystones evade most of it (see
            # sim.cube.host.KEYSTONE_EVASION), which is why a competent immune system alone does
            # not resolve a red-complex pocket — it prunes everything the keystones out-survive.
            clear = 1.0 if self.mortality_mult is None else self.mortality_mult.get(cid, 1.0)
            if subvert is not None:
                # Clearance above the engine's own baseline is HOST clearance, and host clearance is
                # what the keystone disables. The baseline predation stays: subversion blunts the
                # immune system, it does not make the biofilm immortal.
                clear = 1.0 + (clear - 1.0) * subvert
            comp[..., n] -= dt * DECAY * clear * (1.0 - ex) * (1.0 - persist) * comp[..., n]
        # [L1 in] Arrival, before predation: a cell that lands this tick is subject to the same
        # clearance as one that was already here, rather than getting a free pass for a step.
        if self.dispersal is not None:
            self.dispersal.step(self, dt)
        np.clip(comp, 0.0, None, out=comp)
        # [L5] Phage predation runs AFTER growth and before the terrain feedback, so a burst this
        # tick reshapes the matrix and chemistry next tick rather than instantly.
        if self.phage is not None:
            self.phage.step(self, dt)
        self.tissue.update_terrain(self, dt)                           # E reshapes T (the feedback)
        self.tissue.react(self, dt)                                    # E: tissue-specific transitions

    def readout(self) -> dict:
        """ψ — the tissue's system state."""
        return self.tissue.readout(self)

    def efflux(self) -> float:
        """[L7 out] systemic load leaving this cube into the bloodstream."""
        return self.tissue.efflux(self)

    def receive(self, influx: float) -> None:
        """[L7 in] systemic load arriving from the bloodstream."""
        self.tissue.receive(self, influx)


def run_fused(tissue: CubeTissue = ORAL_TISSUE, dim: int = 32, steps: int = 120, dt: float = 0.1,
              sample_every: int = 20):
    """Build a cube for the tissue, seed thin, run the Epoch-4 loop; return (engine, ψ trajectory)."""
    grid = CubeGrid(dim, dim, dim, species_ids=tissue.cell_ids)
    eng = CubeEpoch4(grid, tissue)
    eng.seed_uniform()
    traj = []
    for t in range(1, steps + 1):
        eng.step(dt)
        if t % sample_every == 0 or t == steps:
            r = eng.readout()
            traj.append({"t": round(t * dt, 1), **{k: v for k, v in r.items() if k != "fields"}})
    return eng, traj

