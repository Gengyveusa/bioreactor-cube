"""Layer-6 host coupling tests — the host as a combatant.

The load-bearing assertions here are the NEGATIVE ones. A host layer that quietly gives every
existing run an immune system would invalidate every result already recorded, so each coupling is
tested for "changes nothing when unmeasured" as carefully as for "does something when measured".
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.cube.faces import FaceConditions
from sim.cube.grid import CubeGrid
from sim.cube.host import KEYSTONE_IDS, HostState, attach_host
from sim.intake.adapters import from_myperiopath
from sim.intake.apply import recession_mm, to_face_conditions, to_host_state


def test_unmeasured_host_changes_absolutely_nothing():
    """The whole existing corpus of runs depends on this."""
    intake = from_myperiopath({"Pg": 1e5}, probing_depth_mm=6.0, bleeding_on_probing=0.6)
    state = to_host_state(intake)
    assert state.empty
    assert state.mortality_multipliers(["Pg", "Sa"]) == {}

    base = FaceConditions(serum_heme=0.6)
    assert state.apply_to_conditions(base) is base          # not even a copy


def test_cal_minus_pd_is_the_thing_cal_adds():
    """Two sites with identical probe readings and different histories must stop being identical."""
    intact = from_myperiopath({"Pg": 1e5}, probing_depth_mm=5.0, clinical_attachment_mm=5.0)
    destroyed = from_myperiopath({"Pg": 1e5}, probing_depth_mm=5.0, clinical_attachment_mm=8.0)

    assert recession_mm(intact) == 0.0
    assert recession_mm(destroyed) == 3.0
    assert to_host_state(destroyed).tissue_breakdown > to_host_state(intact).tissue_breakdown

    # PD still owns the geometry — CAL must not silently start resizing the lattice.
    from sim.intake.apply import to_grid_shape
    assert to_grid_shape(intact) == to_grid_shape(destroyed)


def test_cal_without_pd_is_ignored_rather_than_guessed():
    assert recession_mm(from_myperiopath({"Pg": 1e5}, clinical_attachment_mm=8.0)) == 0.0
    assert recession_mm(from_myperiopath({"Pg": 1e5}, probing_depth_mm=5.0)) == 0.0


def test_recession_never_goes_negative():
    """CAL < PD happens in real charts (gingival overgrowth, measurement error). Do not invent
    negative tissue loss from it."""
    weird = from_myperiopath({"Pg": 1e5}, probing_depth_mm=8.0, clinical_attachment_mm=5.0)
    assert recession_mm(weird) == 0.0


def test_systemic_markers_are_averaged_not_summed():
    """hs-CRP, IL-6 and TNF-alpha are correlated readouts of ONE tone. Summing would let a full
    panel triple the same signal purely because more boxes were ticked."""
    one = to_host_state(from_myperiopath({"Pg": 1e5}, hs_crp=10.0))
    three = to_host_state(from_myperiopath({"Pg": 1e5}, hs_crp=10.0, il6=10.0, tnf_alpha=20.0))
    assert one.inflammation == pytest.approx(1.0, abs=1e-6)
    assert three.inflammation <= 1.0


def test_breakdown_takes_the_worse_of_active_and_historical():
    """aMMP-8 is the current RATE; recession is what is already GONE. A quiet mouth with old
    recession is not actively breaking down, so these do not add."""
    old = to_host_state(from_myperiopath({"Pg": 1e5}, probing_depth_mm=3.0,
                                         clinical_attachment_mm=8.0))
    active = to_host_state(from_myperiopath({"Pg": 1e5}, ammp8=45.0))
    both = to_host_state(from_myperiopath({"Pg": 1e5}, probing_depth_mm=3.0,
                                          clinical_attachment_mm=8.0, ammp8=45.0))
    assert both.tissue_breakdown == max(old.tissue_breakdown, active.tissue_breakdown)
    assert both.tissue_breakdown <= 1.0


def test_hba1c_reads_as_impairment_not_capacity():
    """Diabetes impairs neutrophil function, so a HIGHER HbA1c must mean LESS clearance."""
    well = to_host_state(from_myperiopath({"Pg": 1e5}, hba1c=5.2))
    poor = to_host_state(from_myperiopath({"Pg": 1e5}, hba1c=9.1))
    assert well.antimicrobial_capacity > poor.antimicrobial_capacity
    assert 0.0 <= poor.antimicrobial_capacity <= 1.0


def test_keystones_evade_most_of_the_clearance():
    """The mechanism that makes a competent immune system insufficient: gingipains degrade
    complement and immunoglobulin, so the red complex keeps only a fraction of the pressure.

    The consequence is counterintuitive and worth having pinned in a test — host clearance prunes
    the COMMENSALS harder than the keystone it is aimed at.
    """
    state = to_host_state(from_myperiopath({"Pg": 1e5}, hba1c=5.0))
    mult = state.mortality_multipliers(["Pg", "Td", "Tf", "Sa", "Vp"])
    for k in KEYSTONE_IDS:
        assert mult[k] < mult["Sa"]
    assert mult["Sa"] == mult["Vp"]


def test_inflammation_adds_to_bop_heme_rather_than_replacing_it():
    bop_only = to_face_conditions(from_myperiopath({"Pg": 1e5}, bleeding_on_probing=0.6))
    with_tone = to_face_conditions(from_myperiopath({"Pg": 1e5}, bleeding_on_probing=0.6,
                                                    hs_crp=8.0))
    assert with_tone.serum_heme > bop_only.serum_heme
    assert with_tone.serum_heme <= 1.0


def test_tissue_breakdown_deepens_the_anoxic_niche():
    base = to_face_conditions(from_myperiopath({"Pg": 1e5}))
    broken = to_face_conditions(from_myperiopath({"Pg": 1e5}, ammp8=45.0))
    assert broken.serum_O2 < base.serum_O2
    assert broken.serum_O2 >= 0.0


def test_a_consumed_marker_is_disqualified_as_a_validation_target():
    """The anti-circularity guard, made structural rather than a comment. A marker is an input or
    an outcome; using one as both is assuming what you claim to predict."""
    state = to_host_state(from_myperiopath({"Pg": 1e5}, hs_crp=4.0, hba1c=6.0))
    assert "hs_crp" in state.consumed and "hba1c" in state.consumed
    assert "hs_crp" not in state.validation_targets()
    assert "hba1c" not in state.validation_targets()
    assert "il6" in state.validation_targets()          # never spent, still honest to score against

    untouched = to_host_state(from_myperiopath({"Pg": 1e5}))
    assert set(untouched.validation_targets()) == set(HostState.KNOWN)


def test_engine_mortality_hook_defaults_to_inert_and_binds_when_set():
    from sim.cube.fused import CubeEpoch4

    eng = CubeEpoch4(CubeGrid(nx=6, ny=6, nz=6))
    assert eng.mortality_mult is None                    # stock predation, unchanged

    attach_host(eng, to_host_state(from_myperiopath({"Pg": 1e5})))
    assert eng.mortality_mult is None                    # unmeasured host stays inert

    attach_host(eng, to_host_state(from_myperiopath({"Pg": 1e5}, hba1c=5.0)))
    assert eng.mortality_mult is not None
    assert eng.mortality_mult["Pg"] < eng.mortality_mult["Sa"]


def test_host_clearance_actually_moves_biomass_in_a_run():
    """End to end: the coupling must reach the composition array, not just the config object."""
    from sim.cube.fused import CubeEpoch4

    def run(intake):
        g = CubeGrid(nx=8, ny=8, nz=8)
        eng = CubeEpoch4(g)
        eng.seed_uniform()
        if intake is not None:
            attach_host(eng, to_host_state(intake))
        for _ in range(12):
            eng.step(0.1)
        return float(g.composition.sum())

    stock = run(None)
    cleared = run(from_myperiopath({"Pg": 1e5}, hba1c=5.0))
    assert cleared < stock, "a competent host must remove biomass the stock engine keeps"

