"""Grid-convergence regression tests.

These lock in a property that was violated for the whole life of the project before 2026-08-01:
readouts must describe the tissue, not the lattice. The measured spread of `rupture_index` across
dim 10-24 was 183% (0.126 at dim 10, 0.99 at dim 24) with the biology, dose and duration held fixed,
which meant the published "0.36 vs 0.10" diseased-vs-healthy figure sat entirely inside the range
the mesh alone produced.

Two independent causes, both fixed:
  1. DIMENSIONAL. `mmp` was an extensive sum over voxels while `cap` and `foam_frac` were intensive
     means; their product is a function of voxel count. Every state readout is now intensive.
  2. TRANSPORT. One application of the diffusion stencil moves information one CELL, so refining
     the mesh silently slowed physical diffusion. Diffusion is now scaled by (n/n_ref)^2.
"""

from __future__ import annotations

from sim.cube.convergence import run_coupled, sweep, variation
from sim.cube.ecm import DIFFUSION_REF_DIM, diffusion_iters

# Lattice sizes to compare. Kept small so the suite stays fast; the full sweep lives in
# sim/cube/convergence.py and goes to dim 32.
DIMS = (10, 16, 22)
STEPS = 120

# Ceiling on acceptable spread. Anything above this is reporting the grid.
TOL = 0.15


def _series():
    return sweep(DIMS, steps=STEPS)


def test_diffusion_iters_scales_quadratically():
    """Physical diffusivity is held fixed by applying the stencil (n/n_ref)^2 times."""
    assert diffusion_iters(DIFFUSION_REF_DIM) == 1
    assert diffusion_iters(2 * DIFFUSION_REF_DIM) == 4
    assert diffusion_iters(3 * DIFFUSION_REF_DIM) == 9
    assert diffusion_iters(1) >= 1                      # never zero


def test_headline_readouts_are_mesh_independent():
    """The numbers anyone would quote must not move when only the lattice changes."""
    s = _series()
    offenders = {k: round(variation(s, k), 3)
                 for k in ("rupture_index", "endothelial_activation", "ecm_density",
                           "plaque_density", "biomass", "foam_fraction")
                 if variation(s, k) > TOL}
    assert not offenders, "mesh-dependent readouts: %s" % offenders


def test_state_readouts_are_intensive_not_extensive():
    """An extensive sum grows ~dim^3. Catch a regression by its scaling signature directly."""
    small = run_coupled(10, steps=STEPS)
    large = run_coupled(22, steps=STEPS)
    for key in ("biomass", "plaque_density", "mmp_activity", "inflammation_load"):
        a, b = float(small[key]), float(large[key])
        if a <= 0:
            continue
        # dim 22 has ~10.6x the voxels of dim 10. An extensive quantity would grow by that much.
        assert b / a < 3.0, "%s looks extensive: %.4g -> %.4g across a 10.6x voxel increase" % (key, a, b)


def test_diseased_healthy_contrast_survives_refinement():
    """The contrast is the thing that gets quoted, so it is the thing that must hold up."""
    ratios = []
    for d in DIMS:
        dis = run_coupled(d, dysbiotic=True, steps=STEPS)["rupture_index"]
        hlt = run_coupled(d, dysbiotic=False, steps=STEPS)["rupture_index"]
        assert dis > hlt > 0.0, "dysbiotic must exceed healthy at dim %d (%.4g vs %.4g)" % (d, dis, hlt)
        ratios.append(dis / hlt)
    spread = (max(ratios) - min(ratios)) / (sum(ratios) / len(ratios))
    assert spread < 0.6, "contrast ratio itself is mesh-dependent: %s" % [round(r, 2) for r in ratios]


def test_rupture_index_is_not_called_a_risk():
    """It is a dimensionless composite with an arbitrary scale, not a probability of anything.

    The name is load-bearing: `rupture_risk` invited readers to treat an uncalibrated index as an
    event probability. If someone renames it back, this test should stop them.
    """
    r = run_coupled(12, steps=60)
    assert "rupture_index" in r
    assert "rupture_risk" not in r

