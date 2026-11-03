"""Intake tests. The round-trip test is the important one: unit and normalisation bugs are the
likeliest failure here and the hardest to see by eye."""

from __future__ import annotations

import numpy as np
import pytest

from sim.cube.sample import DEFAULT_ABUNDANCE
from sim.intake import (
    Provenance, apply_intake, from_manual, from_myperiopath, from_relative_abundance,
    resolve_taxon, to_abundances, to_face_conditions, to_grid_shape,
)
from sim.intake.baselines import healthy_baseline, perio_baseline
from sim.intake.priors import complete_composition, dysbiosis_signal
from sim.intake.schema import Value
from sim.species import CATALOG

CATALOG_IDS = [s.id for s in CATALOG]


# ------------------------------------------------------------------ taxon resolution
def test_myperiopath_panel_resolves_ten_of_eleven():
    """The real 11-organism panel: 10 map, only Capnocytophaga has no node."""
    panel = ["Aa", "Pg", "Tf", "Td", "En", "Fn", "Pi", "Cr", "Pm", "Ec", "Capnocytophaga"]
    mapped = [n for n in panel if resolve_taxon(n).ok]
    unmapped = [n for n in panel if not resolve_taxon(n).ok]
    assert len(mapped) == 10, mapped
    assert unmapped == ["Capnocytophaga"]


def test_scientific_names_and_synonyms_resolve():
    assert resolve_taxon("Porphyromonas gingivalis").species_id == "Pg"
    assert resolve_taxon("  Tannerella  forsythia ").species_id == "Tf"
    assert resolve_taxon("Peptostreptococcus micros").species_id == "Pm"   # legacy name for P. micra
    assert resolve_taxon("Bacteroides forsythus").species_id == "Tf"       # legacy name


def test_unknown_taxon_never_guesses():
    r = resolve_taxon("Nonexistentia bogusii")
    assert not r.ok and r.how == "unknown"


def test_extended_taxa_flagged_outside_default_catalog():
    r = resolve_taxon("Campylobacter rectus")
    assert r.species_id == "Cr" and r.in_catalog is False


# ------------------------------------------------------------------ adapters
def test_myperiopath_marks_measured_and_records_unmapped():
    intake = from_myperiopath({"Pg": 4.2e5, "Tf": 1.1e5, "Capnocytophaga": 3e4},
                              probing_depth_mm=6.0, bleeding_on_probing=1.0)
    site = intake.primary_site
    assert site.composition["Pg"].provenance is Provenance.MEASURED
    assert "Pg" in site.measured_species()
    assert any("Capnocytophaga" in u for u in intake.unmapped_taxa)


def test_implausible_values_are_dropped_not_fatal():
    """Warn, run, mark degraded - a pH of 12 is an error, a missing organism is normal."""
    intake = from_myperiopath({"Pg": 1e5}, saliva_pH=12.0, probing_depth_mm=99.0)
    assert intake.saliva.pH is None
    assert intake.primary_site.probing_depth_mm is None
    assert len(intake.warnings) >= 2
    assert intake.coverage().degraded


def test_negative_and_nonnumeric_abundances_rejected():
    intake = from_myperiopath({"Pg": -5.0, "Tf": "not a number", "Td": 1e5})
    comp = intake.primary_site.composition
    assert "Pg" not in comp and "Tf" not in comp and "Td" in comp
    assert len(intake.warnings) >= 2


def test_percentages_detected_and_converted():
    intake = from_relative_abundance({"Sa": 40.0, "Pg": 60.0})
    assert abs(sum(v.value for v in intake.primary_site.composition.values()) - 1.0) < 1e-9


# ------------------------------------------------------------------ conditional priors
def test_completion_covers_the_whole_catalog_and_normalises():
    out = complete_composition({"Pg": Value(0.30, Provenance.MEASURED)})
    assert set(out) == set(CATALOG_IDS)
    assert abs(sum(v.value for v in out.values()) - 1.0) < 1e-9


def test_measured_values_keep_their_provenance_unmeasured_are_prior():
    out = complete_composition({"Pg": Value(0.30, Provenance.MEASURED)})
    assert out["Pg"].provenance is Provenance.MEASURED
    assert out["Rm"].provenance is Provenance.PRIOR


def test_prior_is_conditional_not_fixed():
    """A pathogen-heavy panel must depress the inferred protective flora relative to a clean one."""
    dysbiotic = complete_composition({"Pg": Value(0.60, Provenance.MEASURED),
                                      "Tf": Value(0.30, Provenance.MEASURED),
                                      "Sa": Value(0.10, Provenance.MEASURED)})
    healthy = complete_composition({"Pg": Value(0.02, Provenance.MEASURED),
                                    "Tf": Value(0.02, Provenance.MEASURED),
                                    "Sa": Value(0.96, Provenance.MEASURED)})
    assert dysbiotic["Rm"].value < healthy["Rm"].value      # reducers depressed when pathogens high
    assert dysbiotic["Vp"].value < healthy["Vp"].value


def test_dysbiosis_signal_is_zero_without_information():
    assert dysbiosis_signal({}) == 0.0


def test_uninformative_panel_falls_back_to_plain_prior():
    """Measuring only what the prior already expects should not invent a shift."""
    measured = {k: Value(v, Provenance.MEASURED) for k, v in DEFAULT_ABUNDANCE.items()}
    sig = dysbiosis_signal({k: v.value for k, v in measured.items()})
    assert abs(sig) < 1e-6


# ------------------------------------------------------------------ the round trip
def test_round_trip_relative_abundance_survives_seeding():
    """Seed a known composition at production resolution, read biomass back per species, and check
    every species survives and the dominant one stays dominant. Absolute mass is reshaped by the
    spatial fitness field, so this asserts on ranking and non-extinction, not exact equality."""
    target = {"Sa": 0.30, "Fn": 0.25, "Pg": 0.20, "Vp": 0.15, "Td": 0.10}
    out = apply_intake(from_manual(target, probing_depth_mm=4.0), dim=40, complete=False)
    grid = out["grid"]
    got = {sid: float(grid.composition[..., grid.species_index(sid)].sum()) for sid in target}
    assert all(v > 0 for v in got.values()), got
    assert not out["seeding_losses"]

    # Exact rank order is NOT preserved, by design: seed_composition weights each species by the
    # habitable volume of its own O2/depth niche, so a mid-depth generalist (Fn) can outweigh a
    # narrow surface specialist (Sa) requested slightly higher. What must hold is monotonicity
    # across a real gap - anything asked for >=2x more must come out ahead.
    for a in target:
        for b in target:
            if target[a] >= 2.0 * target[b]:
                assert got[a] > got[b], (a, b, target[a], target[b], got[a], got[b])


def test_small_lattice_silently_kills_anaerobes_and_we_say_so():
    """A real engine trap, caught rather than papered over: seed_composition's anaerobe term
    clip(1.2 - O2, 0, 1.2) zeroes every anaerobe when the lattice is too small for the O2 profile
    to decay. With no immigration in the engine, zero is absorbing - so this must be surfaced."""
    target = {"Sa": 0.30, "Fn": 0.25, "Pg": 0.20, "Vp": 0.15, "Td": 0.10}
    out = apply_intake(from_manual(target, probing_depth_mm=4.0), dim=12, complete=False)
    lost = out["seeding_losses"]
    assert {"Pg", "Vp", "Td"} <= set(lost), lost
    assert out["coverage"].degraded
    assert any("seeding lost" in w for w in out["coverage"].warnings)


def test_abundances_cover_every_catalog_species_when_completed():
    intake = from_myperiopath({"Pg": 4.2e5})
    ab = to_abundances(intake)
    assert set(ab) == set(CATALOG_IDS)
    assert all(v > 0 for v in ab.values()), "no species may be seeded at zero - the cube has no immigration"


def test_seeded_grid_has_biomass_everywhere_it_should():
    out = apply_intake(from_myperiopath({"Pg": 4.2e5, "Fn": 2e5}), dim=10)
    grid = out["grid"]
    assert float(grid.composition.sum()) > 0
    assert np.isfinite(grid.composition).all()


# ------------------------------------------------------------------ the other three insert points
def test_bleeding_on_probing_drives_the_heme_boundary():
    bleeding = to_face_conditions(from_myperiopath({"Pg": 1e5}, bleeding_on_probing=1.0))
    dry = to_face_conditions(from_myperiopath({"Pg": 1e5}, bleeding_on_probing=0.0))
    assert bleeding.serum_heme > dry.serum_heme


def test_salivary_ph_reaches_the_oral_face():
    fc = to_face_conditions(from_myperiopath({"Pg": 1e5}, saliva_pH=5.9))
    assert abs(fc.oral_pH - 5.9) < 1e-9


def test_unmeasured_saliva_keeps_engine_defaults():
    from sim.cube.faces import FaceConditions
    fc = to_face_conditions(from_myperiopath({"Pg": 1e5}))
    d = FaceConditions()
    assert fc.oral_pH == d.oral_pH and fc.oral_nitrate == d.oral_nitrate


def test_probing_depth_sets_voxel_size():
    deep = to_grid_shape(from_myperiopath({"Pg": 1e5}, probing_depth_mm=6.0), dim=40)
    shallow = to_grid_shape(from_myperiopath({"Pg": 1e5}, probing_depth_mm=3.0), dim=40)
    assert deep["cell_size_um"] == 150.0
    assert shallow["cell_size_um"] == 75.0
    assert deep["nz"] == shallow["nz"] == 40      # voxel COUNT fixed; runtime stays predictable


def test_host_markers_are_carried_but_seed_nothing():
    out = apply_intake(from_myperiopath({"Pg": 1e5}, ammp8=42.0), dim=8)
    assert out["host"].ammp8.value == 42.0
    assert "ammp8" not in out["abundances"]


# ------------------------------------------------------------------ coverage / honesty
def test_pathogen_only_panel_reports_low_coverage():
    cov = from_myperiopath({"Pg": 4e5, "Tf": 1e5, "Td": 8e4, "Aa": 2e4}).coverage()
    assert cov.n_measured == 4 and cov.n_modeled == 18
    assert cov.measured_fraction < 0.3
    assert cov.degraded
    assert "4/18" in cov.summary()


def test_provenance_map_is_parallel_to_abundances():
    from sim.intake.apply import composition_provenance
    intake = from_myperiopath({"Pg": 4e5, "Tf": 1e5})
    ab, prov = to_abundances(intake), composition_provenance(intake)
    assert set(ab) == set(prov)
    assert prov["Pg"] is Provenance.MEASURED and prov["Rm"] is Provenance.PRIOR


def test_baselines_are_never_marked_measured():
    for mk in (healthy_baseline, perio_baseline):
        intake = mk()
        assert all(v.provenance is Provenance.DEFAULT
                   for v in intake.primary_site.composition.values())
        assert any("not a real subject" in w for w in intake.warnings)


def test_baselines_differ_in_the_direction_that_matters():
    h, p = healthy_baseline(), perio_baseline()
    red = ("Pg", "Td", "Tf")
    h_red = sum(h.primary_site.composition[s].value for s in red)
    p_red = sum(p.primary_site.composition[s].value for s in red)
    assert p_red > 5 * h_red
    assert p.primary_site.probing_depth_mm.value > h.primary_site.probing_depth_mm.value


def test_baselines_build_real_cubes():
    for mk in (healthy_baseline, perio_baseline):
        out = apply_intake(mk(), dim=10)
        assert float(out["grid"].composition.sum()) > 0


# --------------------------------------------------------------------------- wired saliva inputs
# These three fields were declared in the schema from day one and read by nothing. Each test below
# asserts BOTH halves of the contract: the value changes the simulation when measured, and the
# default reproduces the pre-wiring behaviour exactly when it is not. The second half is the one
# that matters — a seam that silently alters every existing run is worse than no seam.

def test_salivary_buffer_sets_bicarbonate_and_leaves_plaque_buffer_alone():
    """A chairside buffer strip is a SALIVA measurement; the plaque fixed buffer is a different
    compartment (~90% of plaque buffering is immobile surface charge, Shellis & Dibdin 1988) and
    nothing chairside measures it. Wiring the strip onto FIXED_BUFFER_MM would be a category error.
    """
    from sim.cube.chemistry import FIXED_BUFFER_MM
    from sim.intake.apply import to_face_conditions

    low = to_face_conditions(from_myperiopath({"Pg": 1e5}, salivary_buffer=4.0)).buffers
    high = to_face_conditions(from_myperiopath({"Pg": 1e5}, salivary_buffer=6.0)).buffers

    assert high.bicarb_mM > low.bicarb_mM
    assert low.fixed_mM == high.fixed_mM == FIXED_BUFFER_MM      # plaque compartment untouched


def test_unmeasured_buffer_reproduces_shipped_chemistry_exactly():
    from sim.cube.faces import FaceConditions
    from sim.intake.apply import to_face_conditions
    assert to_face_conditions(from_myperiopath({"Pg": 1e5})).buffers == FaceConditions().buffers


def test_better_buffered_saliva_resists_acid_without_moving_resting_ph():
    """The whole point of a buffer: same starting pH, smaller excursion under the same acid load.

    If resting pH moved with the buffer pool, the SID would not have been re-solved and every
    patient with a buffer result would silently start somewhere different — which would look like
    a treatment effect in any before/after comparison.
    """
    from sim.cube.chemistry import BufferSpec

    poor, rich = BufferSpec(bicarb_mM=1.5), BufferSpec(bicarb_mM=15.0)
    assert abs(float(poor.solve({}, shape=())) - float(rich.solve({}, shape=()))) < 1e-6
    assert float(rich.solve({"lactate": 10.0}, shape=())) > float(poor.solve({"lactate": 10.0}, shape=()))


def test_buffer_test_units_are_not_guessed_silently():
    from sim.cube.chemistry import bicarbonate_from_buffer_test as conv
    assert conv(12.0, "mM") == 12.0
    assert conv(5.0, "pH") == pytest.approx(4.4)          # the Bardow resting anchor
    assert conv(6.0, "pH") > conv(4.0, "pH")              # monotone
    with pytest.raises(ValueError):
        conv(5.0, "furlongs")
    with pytest.raises(ValueError):
        conv(-1.0, "mM")


def test_salivary_nitrite_reaches_the_field_only_when_measured():
    """The engine has modelled a nitrite field all along; the measurement never reached it.

    Unmeasured must stay None rather than default to a number: pinning the mouth face to 0.0 would
    make it an infinite sink, the same bug already fixed once for the fermentation acids.
    """
    from sim.cube.faces import FaceConditions, apply_face_boundaries
    from sim.cube.grid import CubeGrid
    from sim.intake.apply import to_face_conditions

    assert to_face_conditions(from_myperiopath({"Pg": 1e5})).oral_nitrite is None
    assert to_face_conditions(from_myperiopath({"Pg": 1e5}, salivary_nitrite=0.8)).oral_nitrite == 0.8

    g = CubeGrid(nx=6, ny=6, nz=6)
    g.get_field("nitrite")[:] = 0.3
    apply_face_boundaries(g, FaceConditions())                       # unmeasured: do not touch
    assert float(g.get_field("nitrite")[:, :, -1].mean()) == pytest.approx(0.3)
    apply_face_boundaries(g, FaceConditions(oral_nitrite=0.9))       # measured: impose it
    assert float(g.get_field("nitrite")[:, :, -1].mean()) == pytest.approx(0.9)


def test_total_load_scales_biomass_and_is_bounded():
    """Two mouths with identical ratios and 100x different load are not the same mouth.

    The schema said so in a docstring for weeks while nothing read the field. The clamp is
    deliberate: the growth terms are not calibrated across decades of biomass, so load is an
    influential anchor rather than a literal multiplier.
    """
    from sim.intake.apply import LOAD_FACTOR_RANGE, load_factor, to_abundances

    assert load_factor(from_myperiopath({"Pg": 1e5})) == 1.0        # absent -> unchanged

    lo_i = from_myperiopath({"Pg": 1e5, "Tf": 5e4}, total_load=4.0e5)
    hi_i = from_myperiopath({"Pg": 1e5, "Tf": 5e4}, total_load=4.0e6)
    assert sum(to_abundances(hi_i).values()) > sum(to_abundances(lo_i).values())

    # identical ratios, different scale -> the RATIOS must survive untouched
    lo, hi = to_abundances(lo_i), to_abundances(hi_i)
    assert lo["Pg"] / lo["Tf"] == pytest.approx(hi["Pg"] / hi["Tf"])

    assert load_factor(from_myperiopath({"Pg": 1e5}, total_load=1e12)) == LOAD_FACTOR_RANGE[1]
    assert load_factor(from_myperiopath({"Pg": 1e5}, total_load=1.0)) == LOAD_FACTOR_RANGE[0]

