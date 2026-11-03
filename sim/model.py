"""OralBiofilmModel — the Mesa orchestrator.

Each tick:
1. BoundaryConditions.apply() — saliva inflow at top, heme at bottom
2. agent.step() for every live agent — they read chem, grow, emit, divide, OMV
3. Diffusion1D.step() — chemistry diffuses + decays
4. Observables computed and pushed into trajectory
"""
from __future__ import annotations
import random
import itertools

from mesa import Model

from .config import SimConfig
from .grid import SpatialGrid, SpatialGrid1D
from .chemistry import Diffusion1D, BoundaryConditions
from .agents import BacterialAgent, OMVEmission
from .species import CATALOG, BY_ID
from .observables import observe, Trajectory


class OralBiofilmModel(Model):
    def __init__(self, cfg: SimConfig):
        super().__init__()
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.tick: int = 0
        self._next_id = itertools.count()
        # Spatial substrate
        if cfg.grid.kind == "1d":
            self.grid_obj: SpatialGrid = SpatialGrid1D(
                depth_cells=cfg.grid.depth_cells,
                cell_size_um=cfg.grid.cell_size_um,
            )
        else:
            raise NotImplementedError(f"grid.kind={cfg.grid.kind} not implemented yet")
        # Chemistry
        self.diffusion = Diffusion1D(cfg.chemistry)
        self.boundaries = BoundaryConditions(cfg.chemistry)
        # Agents — Mesa's BaseScheduler isn't great for large counts; use our own list
        self.agents_alive: list[BacterialAgent] = []
        self.omv_log: list[tuple[int, OMVEmission]] = []
        self.trajectory = Trajectory()

        self._seed_inoculum()

    def _seed_inoculum(self) -> None:
        """Saliva-broadcast inoculum: pioneers seeded everywhere shallow,
        anaerobes seeded broadly across their preferred band with extra mass
        deep (where the chemistry envelope can actually receive them).
        """
        per = self.cfg.agents.initial_inoculum_per_species
        seeding = self.cfg.agents.seeding   # {species_id: relative_abundance} | None (patient panel)
        n_panel = len(seeding) if seeding else 0
        N = self.grid_obj.shape[0]
        for sp in CATALOG:
            # Patient seeding: panel species scaled by abundance (relative to uniform),
            # species absent from the panel keep the baseline inoculum.
            if seeding and sp.id in seeding:
                count = max(0, round(per * seeding[sp.id] * n_panel))
            else:
                count = per
            lo, hi = sp.preferred_depth
            i_lo = max(0, int(lo * (N - 1)))
            i_hi = min(N - 1, int(hi * (N - 1)))
            # Spread across the FULL band — not just one random point
            for k in range(count):
                if i_hi > i_lo:
                    # Stratified placement plus jitter
                    frac = (k + 0.5) / count
                    depth_idx = int(i_lo + frac * (i_hi - i_lo))
                    depth_idx = max(0, min(N - 1,
                                           depth_idx + self.rng.randint(-1, 1)))
                else:
                    depth_idx = i_lo
                pos = (depth_idx,)
                a = BacterialAgent(model=self, species=sp, position=pos)
                self.agents_alive.append(a)

        # Pre-condition the chemistry: small baseline heme at the deepest 5 cells
        # so heme-obligate species (Pg) don't die before bleed events reach them.
        # Biological justification: even healthy gingival sulci have trace heme
        # from minor mechanical irritation.
        for i in range(max(0, N - 5), N):
            self.grid_obj.add_to_field_at("heme", (i,), 0.15)

    def spawn_daughter(self, parent: BacterialAgent) -> None:
        """Place a daughter in a neighbor cell of the parent."""
        nbrs = self.grid_obj.get_neighbors(parent.position, include_self=True)
        # Prefer staying within preferred depth band
        sp = parent.species
        N = self.grid_obj.shape[0]
        lo, hi = sp.preferred_depth
        i_lo, i_hi = int(lo * (N - 1)), int(hi * (N - 1))
        candidates = [p for p in nbrs if i_lo <= p[0] <= i_hi] or nbrs
        pos = self.rng.choice(candidates)
        a = BacterialAgent(model=self, species=sp, position=pos)
        self.agents_alive.append(a)

    def step(self) -> None:
        self.tick += 1
        # 1. Boundaries (top inflow, bottom heme)
        self.boundaries.apply(self.grid_obj, self.tick)
        # 2. Agents — shuffle to remove order bias
        self.rng.shuffle(self.agents_alive)
        for a in list(self.agents_alive):
            a.step()
        # Compost the dead
        self.agents_alive = [a for a in self.agents_alive if a.alive]
        # 3. Chemistry diffusion
        self.diffusion.step(self.grid_obj)
        # 4. Observe
        if self.tick % self.cfg.log_every == 0:
            self.trajectory.ticks.append(observe(self))

    def run(self, ticks: int | None = None) -> Trajectory:
        n = ticks if ticks is not None else self.cfg.ticks
        for _ in range(n):
            self.step()
        return self.trajectory

