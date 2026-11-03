"""Tests for the fused engine — the Epoch-4 loop on the 3-D cube.

Headline: with everything seeded THIN and EVERYWHERE, the spatial structure must EMERGE from
terrain-gated expression alone — deep anaerobes (red complex) end up deep, surface aerobes end up
shallow. Nothing is placed; the Information→T→E→I→Δ→ψ loop carves it."""

from __future__ import annotations

import numpy as np

from sim.cube.fused import CubeEpoch4, prepare_terrain, run_fused
from sim.cube.grid import CubeGrid


def _weighted_mean_depth(eng, predicate) -> float:
    """Biomass-weighted mean k-index (0 = deep face, nz-1 = oral face) over species matching predicate."""
    comp = eng.grid.composition
    nz = eng.grid.shape[2]
    k = np.arange(nz)[np.newaxis, np.newaxis, :]
    num = den = 0.0
    for n, sid in enumerate(eng.species):
        if predicate(eng.genome[sid]):
            b = comp[..., n]
            num += float((b * k).sum())
            den += float(b.sum())
    return num / den if den else float("nan")


def test_fused_runs_and_information_moves() -> None:
    eng, traj = run_fused(dim=20, steps=60)
    assert traj and {"dysbiosis", "NO", "omv_flux", "biomass"} <= set(traj[-1])
    assert eng.expression.max() > 0.25          # expression rose from the 0.2 seed where terrain permits


def test_terrain_is_set_before_the_loop() -> None:
    g = CubeGrid(12, 12, 12)
    prepare_terrain(g)
    o2 = g.get_field("O2")
    assert o2[:, :, -1].mean() > o2[:, :, 0].mean()    # O2 high at oral face, low deep (terrain T)


def _species_depth(eng, sid) -> float:
    comp = eng.grid.composition
    nz = eng.grid.shape[2]
    k = np.arange(nz)[np.newaxis, np.newaxis, :]
    b = comp[..., eng.species.index(sid)]
    return float((b * k).sum() / max(b.sum(), 1e-9))


def test_spatial_structure_EMERGES_from_terrain_gating() -> None:
    """The whole point: seeded UNIFORM, a surface aerobe ends up shallow and a deep anaerobe ends up
    deep — purely because terrain (the O2 the aerobes themselves consumed) gates expression. Nothing
    was placed; the Information→T→E→I→Δ→ψ loop carved it."""
    eng, _ = run_fused(dim=26, steps=180)
    # oral face = high k, deep face = low k.
    aerobe = _species_depth(eng, "Sa")           # S. sanguinis — O2-loving pioneer
    anaerobe = _species_depth(eng, "Vp")         # Veillonella — obligate anaerobe
    pg = _species_depth(eng, "Pg")               # P. gingivalis — red-complex anaerobe
    assert aerobe > anaerobe                      # the aerobe sits shallower than the anaerobe
    assert aerobe > pg                            # and shallower than the red-complex keystone


def test_psi_fields_are_3d_and_consistent() -> None:
    eng, _ = run_fused(dim=16, steps=40)
    r = eng.readout()
    assert r["fields"]["dysbiosis"].shape == eng.grid.shape
    assert 0.0 <= r["dysbiosis"] <= 1.0
    assert r["NO"] >= 0.0 and r["omv_flux"] >= 0.0

