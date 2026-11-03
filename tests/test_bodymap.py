"""Tests for the 2nd tissue (artery/atheroma) + the connected body-map.

The headline (the systemic thesis, realised spatially): a DYSBIOTIC, bleeding oral cube sheds more
heme-induced OMV into the bloodstream than a HEALTHY one, and that distant load drives a coupled
ARTERY cube toward endothelial activation and foam-cell formation — a monotone oral→systemic dose-
response. Nothing about plaque is hand-set; it is the artery's own information-primary loop responding."""

from __future__ import annotations

from sim.cube.bodymap import BodyMap
from sim.cube.fused import run_fused
from sim.cube.tissues import ARTERY_TISSUE, ORAL_HEALTHY, ORAL_TISSUE


def test_artery_tissue_runs_standalone() -> None:
    eng, traj = run_fused(ARTERY_TISSUE, dim=16, steps=60, sample_every=60)
    r = traj[-1]
    # plaque_DENSITY, not burden: every state readout is now intensive, so it describes the tissue
    # rather than the lattice. See sim/cube/convergence.py for why that matters.
    assert {"plaque_density", "foam_fraction", "endothelial_activation", "inflammation_load"} <= set(r)
    assert eng.readout()["biomass"] > 0.0
    assert "Foam" in eng.species and "EC" in eng.species          # the artery roster, not the oral one


def test_oral_efflux_is_higher_when_dysbiotic() -> None:
    """Hemin from a bleeding pocket induces OMV shedding → more systemic load into the bloodstream."""
    d, _ = run_fused(ORAL_TISSUE, dim=18, steps=100, sample_every=100)
    h, _ = run_fused(ORAL_HEALTHY, dim=18, steps=100, sample_every=100)
    assert d.efflux() > h.efflux() > 0.0


def _coupled_artery(oral_tissue, dim=18, steps=130, k=2.5) -> dict:
    bm = BodyMap()
    bm.place("oral", oral_tissue, dim=dim)
    bm.place("artery", ARTERY_TISSUE, dim=dim)
    bm.link("oral", "artery", k=k)                                 # the bloodstream edge
    return bm.run(steps, sample_every=steps)[-1]["artery"]


def test_body_map_dose_response_oral_drives_artery() -> None:
    """THE thesis: dysbiotic oral → more endothelial activation + foam than healthy oral, same artery."""
    dysbiotic = _coupled_artery(ORAL_TISSUE)
    healthy = _coupled_artery(ORAL_HEALTHY)
    assert dysbiotic["endothelial_activation"] > healthy["endothelial_activation"]
    assert dysbiotic["foam_fraction"] >= healthy["foam_fraction"]
    assert dysbiotic["inflammation_load"] > healthy["inflammation_load"]


def test_no_bloodstream_link_means_no_distant_activation() -> None:
    """Cut the edge (k=0): the artery gets no oral load, so endothelial activation stays at baseline,
    well below the coupled-dysbiotic case. The effect is carried by the bloodstream, not assumed."""
    coupled = _coupled_artery(ORAL_TISSUE, k=2.5)
    uncoupled = _coupled_artery(ORAL_TISSUE, k=0.0)
    assert coupled["endothelial_activation"] > uncoupled["endothelial_activation"]

