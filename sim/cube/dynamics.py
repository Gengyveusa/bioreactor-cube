"""Time-playback dynamics — stepped growth + chemistry + biomineralization, captured as frames.

The flagship scenario: *Corynebacterium matruchotii* (the Mineralizer) colonizes the enamel surface
by **rapid binary fission** (a much higher growth rate than the community), then — once dense — runs
its **biomineralization program**: hydroxyapatite (mineral) nucleates from the enamel-supplied
Ca/PO4 wherever the colony passes a density threshold (calculus formation, Welch 2016).

Real coupling: logistic growth × the species' viability fitness, biomass spread by diffusion, and
mineral = k · max(0, Cma − threshold) · Ca · PO4. Frames are sampled point sets the viewer scrubs.
"""

from __future__ import annotations

import numpy as np

from sim.cube.faces import apply_face_boundaries
from sim.cube.grid import CubeGrid
from sim.cube.sample import (
    DEFAULT_ABUNDANCE,
    _axis_norm,
    _depth_field,
    _diffuse,
    build_sample,
    seed_composition,
    solve_snapshot,
)
from sim.species import CATALOG
from sim.species_extended import EXTENDED

CMA = next(s for s in EXTENDED if s.id == "Cma")
DEMO_IDS = [s.id for s in CATALOG] + ["Cma"]
CMA_COLOR = "#f0d27a"        # Mineralizer
MINERAL_COLOR = "#dbeafe"    # hydroxyapatite crystal
HOURS_PER_STEP = 0.5


def _fitness(grid: CubeGrid, s) -> np.ndarray:
    o2 = grid.get_field("O2")
    depth = _depth_field(grid.shape)
    o2_w = max(1.0, (s.envelope.o2[1] - s.envelope.o2[0]) / 2.0)
    f = np.exp(-((o2 - s.envelope.o2[2]) / o2_w) ** 2)
    c = 0.5 * (s.preferred_depth[0] + s.preferred_depth[1])
    return f * np.exp(-((depth - c) / 0.2) ** 2)


def matruchotii_demo(dim: int = 40, steps: int = 72, capture: int = 36, seed: int = 0):
    """Run the colonization → biomineralization scenario; return (grid, frames) where each frame is
    (Cma biomass field, mineral field, step index)."""
    grid = CubeGrid(dim, dim, dim, species_ids=DEMO_IDS)
    seed_composition(grid, DEFAULT_ABUNDANCE, seed)
    solve_snapshot(grid, 30)

    nx, ny, nz = grid.shape
    comp, idx = grid.composition, grid.species_index
    # a small Cma colony seeded on the enamel surface (x≈0) at shallow depth
    x_n = _axis_norm(nx)[:, np.newaxis, np.newaxis]
    depth = _depth_field(grid.shape)
    seed_field = np.exp(-((x_n - 0.06) / 0.08) ** 2) * np.exp(-((depth - 0.12) / 0.12) ** 2)
    comp[..., idx("Cma")] = 0.04 * np.broadcast_to(seed_field, grid.shape)

    fits = {s.id: _fitness(grid, s) for s in CATALOG}
    fits["Cma"] = _fitness(grid, CMA)
    rates = {s.id: 0.08 for s in CATALOG}
    rates["Cma"] = 0.35                       # fast binary fission — the whole point
    cap, k_min, thresh = 2.0, 0.06, 0.25
    Ca, PO4, mineral = grid.get_field("Ca"), grid.get_field("PO4"), grid.get_field("mineral")

    frames = []
    cap_steps = set(np.linspace(0, steps - 1, capture).astype(int))
    cma_i = idx("Cma")
    for step in range(steps):
        total = comp.sum(axis=3)
        for sid in DEMO_IDS:
            si = idx(sid)
            b = comp[..., si]
            comp[..., si] = b + rates[sid] * fits[sid] * b * (1.0 - total / cap)
        comp[..., cma_i] = 0.85 * comp[..., cma_i] + 0.15 * _diffuse(comp[..., cma_i])  # colony spread
        np.clip(comp, 0.0, None, out=comp)
        # biomineralization program — triggers where the colony is dense
        cma = comp[..., cma_i]
        m_new = k_min * np.maximum(0.0, cma - thresh) * Ca * PO4
        mineral += m_new
        Ca -= 0.2 * m_new
        PO4 -= 0.2 * m_new
        np.clip(Ca, 0.0, None, out=Ca)
        np.clip(PO4, 0.0, None, out=PO4)
        apply_face_boundaries(grid)            # keep Ca/PO4 supplied at the enamel face
        if step in cap_steps:
            frames.append((comp[..., cma_i].copy(), mineral.copy(), step))
    return grid, frames


def _sample(field: np.ndarray, max_n: int, shape, rng, count_k: float) -> list:
    flat = field.flatten()
    tot = float(flat.sum())
    if tot <= 0:
        return []
    n = int(np.clip(tot * count_k, 15, max_n))
    n = min(n, int((flat > 0).sum()))
    chosen = rng.choice(flat.size, size=n, replace=False, p=flat / tot)
    nx, ny, nz = shape
    out = []
    for fi in chosen:
        i, j, k = (int(x) for x in np.unravel_index(fi, shape))
        out.append([round(i / (nx - 1), 3), round(j / (ny - 1), 3), round(k / (nz - 1), 3)])
    return out


def build_demo(dim: int = 40, field_n: int = 16, n_points: int = 1800, max_edges: int = 280,
               steps: int = 72, capture: int = 36, max_cma: int = 1300, max_min: int = 900,
               seed: int = 0) -> dict:
    """Full payload for the viewer: a static background snapshot + the matruchotii time-frames."""
    data = build_sample(dim=dim, field_n=field_n, n_points=n_points, max_edges=max_edges, seed=seed)
    grid, frames = matruchotii_demo(dim=dim, steps=steps, capture=capture, seed=seed)
    rng = np.random.default_rng(seed + 1)
    cma_tot = max(float(c.sum()) for c, _, _ in frames) or 1.0
    min_tot = max(float(m.sum()) for _, m, _ in frames) or 1.0
    kc, km = max_cma / cma_tot, max_min / min_tot

    frame_data = []
    for cma, mineral, step in frames:
        mtot = float(mineral.sum())
        frame_data.append({
            "t": round(step * HOURS_PER_STEP, 1),
            "cma": _sample(cma, max_cma, grid.shape, rng, kc),
            "mineral": _sample(mineral, max_min, grid.shape, rng, km),
            "cma_n": int((cma > 0.05).sum()),
            "min": round(mtot, 1),
        })
    data["demo"] = {
        "frames": frame_data,
        "cma_color": CMA_COLOR,
        "mineral_color": MINERAL_COLOR,
        "min_max": round(min_tot, 1),
    }
    return data

