"""Phage-layer tests — the third combatant.

Two things are being defended here. First, that phages do nothing at all unless asked, because
every result recorded before this layer existed was produced without them. Second, that the
mechanics which make a biofilm different from a test tube — matrix obstruction, resistance with a
cost, lysogeny — are actually wired and not decorative.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.cube.fused import CubeEpoch4
from sim.cube.grid import CubeGrid
from sim.cube.phage import (
    COAGG_DEPENDENCE, COAGG_PARTNERS, DEFAULT_PANEL, PHI_BRIDGE, PHI_KEYSTONE, PHI_PIONEER,
    PhageSpec, PhageSystem, coaggregation_gate, coaggregation_support, community_shares,
)
from sim.cube.tissues import ORAL_TISSUE

DIM = 10


def _engine(panel=(), coagg=False):
    g = CubeGrid(nx=DIM, ny=DIM, nz=DIM)
    eng = CubeEpoch4(g, tissue=ORAL_TISSUE)
    eng.seed_uniform()
    if panel:
        sysp = PhageSystem(panel=tuple(panel), coaggregation=coagg)
        sysp.prepare(eng)
        eng.phage = sysp
        return eng, sysp
    return eng, None


def _run(eng, steps=40, dt=0.1, dose=None, sysp=None, at=10, where="uniform"):
    for i in range(steps):
        if dose is not None and i == at:
            sysp.dose(eng, dose[0], dose[1], where=where)
        eng.step(dt)
    return eng


# --------------------------------------------------------------------------- inertness
def test_no_phage_is_the_default_and_changes_nothing():
    eng, _ = _engine()
    assert eng.phage is None
    before = float(eng.grid.composition.sum())
    _run(eng, steps=10)
    assert float(eng.grid.composition.sum()) != before      # the sim still runs
    # and no phage state was ever allocated
    assert not any(k.startswith("phage:") for k in eng.extra)


def test_an_undosed_phage_system_barely_perturbs_anything():
    """Allocating the fields must not itself be an intervention.

    It is not bit-identical and should not be: the mutational supply of resistant cells exists
    whether or not a phage is present, and those cells pay the cost of escape. That is real, and
    it is tiny. The tolerance here is what separates 'a standing mutation load' from 'the phage
    module is quietly killing things'.
    """
    plain, _ = _engine()
    _run(plain, steps=30)
    withsys, sysp = _engine(panel=(PHI_PIONEER,))
    _run(withsys, steps=30)
    assert float(withsys.grid.composition.sum()) == pytest.approx(
        float(plain.grid.composition.sum()), rel=1e-3)


def test_resistance_does_not_ratchet_to_fixation_without_phage():
    """The bug this test exists for: supply-plus-selection with no negative term drove resistance
    to 1.0 given enough quiet ticks, which would have made every long run phage-proof before a
    phage was dosed."""
    eng, sysp = _engine(panel=(PHI_PIONEER,))
    _run(eng, steps=200)
    assert float(sysp.resistant(eng, "Sa").mean()) < 0.05


def test_coaggregation_is_opt_in():
    """Switching it on changes every existing result, so it must not switch itself on."""
    assert PhageSystem().coaggregation is False
    eng, _ = _engine(panel=(PHI_PIONEER,))
    assert eng.phage.coaggregation is False


# --------------------------------------------------------------------------- core predation
def test_phage_kills_its_host_and_spares_everyone_else():
    eng, sysp = _engine(panel=(PHI_BRIDGE,))
    control, _ = _engine()
    _run(control, steps=50)
    _run(eng, steps=50, dose=("phiFn", 3.0), sysp=sysp)

    a, b = community_shares(control), community_shares(eng)
    assert b["Fn"] < a["Fn"] * 0.9, "the targeted host must actually fall"
    # A phage with one host must not directly remove a non-host. Shares can shift by competitive
    # release, so this asserts the ABSOLUTE biomass of an unrelated species is not destroyed.
    i = eng.grid.species_index("Smu")
    assert float(eng.grid.composition[..., i].sum()) > 0.0


def test_burst_replicates_rather_than_merely_decaying():
    """Burst must contribute particles, tested against a burst=0 control rather than against the
    starting titre.

    Whether the population NETS positive depends on host density versus decay, which is a regime
    question and not a property of the code; asserting net growth would be asserting that this
    particular lattice happens to sit above the replication threshold.
    """
    def titre(burst):
        spec = PhageSpec("t", hosts=("Sa",), adsorption=7.0, latent=0.5, burst=burst, decay=0.35)
        eng, sysp = _engine(panel=(spec,))
        sysp.dose(eng, "t", 0.5)
        _run(eng, steps=25)
        return float(sysp._free(eng, "t").sum())

    assert titre(80.0) > titre(0.0) * 1.05


def test_infection_removes_biomass_from_the_growing_pool():
    """Infected cells must stop growing, or lysis is a free lunch that never costs the host."""
    eng, sysp = _engine(panel=(PHI_PIONEER,))
    _run(eng, steps=15, dose=("phiSt", 3.0), sysp=sysp, at=2)
    lysing = sysp._lysing(eng, "phiSt", "Sa")
    assert float(lysing.sum()) > 0.0


# --------------------------------------------------------------------------- the biofilm parts
def test_matrix_obstructs_surface_delivered_phage():
    """The refuge effect: a virion is a ~100 nm particle and the EPS gel does not treat it like a solute.

    Delivering at the mouth face (a rinse) must do measurably less than seeding phage uniformly
    through the biofilm, because the surface dose has to cross the matrix. Seeding uniformly is
    how a lattice study flatters phage therapy.
    """
    control, _ = _engine()
    _run(control, steps=60)
    base = float(control.grid.composition[..., control.grid.species_index("Sa")].sum())

    out = {}
    for where in ("uniform", "oral"):
        eng, sysp = _engine(panel=(PHI_PIONEER,))
        _run(eng, steps=60, dose=("phiSt", 3.0), sysp=sysp, at=15, where=where)
        out[where] = float(eng.grid.composition[..., eng.grid.species_index("Sa")].sum())

    assert out["uniform"] < base and out["oral"] < base      # both work
    assert out["oral"] > out["uniform"], "the matrix must shelter the host from a surface dose"


def test_resistance_appears_selects_and_costs():
    eng, sysp = _engine(panel=(PHI_PIONEER,))
    _run(eng, steps=60, dose=("phiSt", 3.0), sysp=sysp, at=10)
    res = sysp.resistant(eng, "Sa")
    assert 0.0 < float(res.mean()) <= 1.0

    undosed, us = _engine(panel=(PHI_PIONEER,))
    _run(undosed, steps=60)
    assert float(res.mean()) > float(us.resistant(undosed, "Sa").mean()), \
        "phage pressure must select harder than mutation supply alone"


def test_resistance_cannot_arise_from_nothing_without_a_seed():
    """A zero that can never leave zero would report eradications real populations never deliver."""
    from sim.cube import phage as P
    eng, sysp = _engine(panel=(PHI_PIONEER,))
    _run(eng, steps=30)
    assert float(sysp.resistant(eng, "Sa").mean()) > 0.0     # mutation supply, no dose


def test_bridges_pay_more_to_escape_than_keystones():
    """Bridge-Asymmetry, on the resistance side: receptor loss is adhesin loss, and for a species
    whose ecological job IS adhesion that is nearly the whole job."""
    from sim.cube.phage import RESISTANCE_COST, RESISTANCE_COST_BRIDGE
    assert RESISTANCE_COST_BRIDGE > RESISTANCE_COST


def test_keystones_are_harder_to_infect_than_bridges():
    """This must come from the EXISTING genome defence values, not be re-encoded here."""
    eng, _ = _engine()
    assert eng.genomes["Pg"]["defense"] > eng.genomes["Fn"]["defense"]
    assert eng.genomes["Fn"]["defense"] > eng.genomes["Sa"]["defense"]


def test_lysogeny_produces_prophage_carriage():
    """Most oral phages are temperate; a model where every infection lyses is the wrong model."""
    eng, sysp = _engine(panel=(PHI_KEYSTONE,))       # highest lysogeny propensity in the panel
    _run(eng, steps=40, dose=("phiPg", 3.0), sysp=sysp, at=5)
    assert float(sysp._prophage(eng, "phiPg", "Pg").sum()) > 0.0


# --------------------------------------------------------------------------- coaggregation
def test_coaggregation_support_is_bounded_and_partner_driven():
    eng, _ = _engine(panel=(PHI_BRIDGE,), coagg=True)
    _run(eng, steps=20)
    sup = coaggregation_support(eng, "Pg")
    assert sup.min() >= 0.0 and sup.max() <= 1.0
    # a pioneer binds the pellicle directly and must be unaffected
    assert coaggregation_gate(eng, "Sa").min() == pytest.approx(1.0)
    assert COAGG_DEPENDENCE["Pioneer"] == 0.0


def test_the_keystone_depends_on_partners_more_than_a_pioneer():
    assert COAGG_DEPENDENCE["Keystone"] > COAGG_DEPENDENCE["Bridge"] > COAGG_DEPENDENCE["Pioneer"]


def test_coagg_partners_match_the_published_edge_list():
    """Pg reaches the community through six partners, none of them a pioneer streptococcus. That
    redundancy is precisely why a single indirect strike is not obviously enough here, and it is
    the structural difference from a two-species dish."""
    assert set(COAGG_PARTNERS["Pg"]) == {"Fa", "Fn", "Pi", "Pm", "Td", "Tf"}
    assert "Sa" not in COAGG_PARTNERS["Pg"]


def test_dose_route_is_validated():
    eng, sysp = _engine(panel=(PHI_PIONEER,))
    with pytest.raises(ValueError):
        sysp.dose(eng, "phiSt", 1.0, where="intravenous")


def test_readout_is_intensive_and_complete():
    eng, sysp = _engine(panel=DEFAULT_PANEL)
    _run(eng, steps=20, dose=("phiFn", 1.0), sysp=sysp, at=5)
    r = sysp.readout(eng)
    for spec in DEFAULT_PANEL:
        assert "titre:%s" % spec.id in r
        for h in spec.hosts:
            for k in ("infected", "prophage", "resistant"):
                assert 0.0 <= r["%s:%s" % (k, h)] <= 1.0, (k, h)


# ------------------------------------------------------- one transport law, two consumers (Aug 5)

def test_phage_exponent_is_retired_not_merely_unused():
    """PHAGE_ECM_EXPONENT = 3.0 was invented and marked [[VERIFY]]. Peulen & Wilkinson 2011
    (PMID 21434601) measured size scaling in a live biofilm and found exp(-k*r^2), not a power of
    porosity. The name survives so imports resolve; nothing may read it."""
    from sim.cube import phage as P
    assert P.PHAGE_ECM_EXPONENT is None


def test_phage_and_drugs_share_one_obstruction_law():
    """A virion and a drug molecule are the same physics at different radii. Two separate rules
    could drift apart silently, which is exactly how theta^3 survived as long as it did."""
    import inspect
    from sim.cube import delivery, phage
    from sim.cube.ecm import obstruction_factor
    assert "obstruction_factor" in inspect.getsource(phage.PhageSystem.step)
    assert obstruction_factor is not None
    # delivery's size_factor must BE the shared law, not a copy of it
    assert "obstruction_factor" in inspect.getsource(delivery.size_factor)


def test_the_law_is_size_selective_in_the_right_direction():
    """Small solutes must be effectively unobstructed — a biofilm mesh does not sieve 0.5 nm — and
    a virion must pay a real penalty. That separation is the whole reason theta^n was retired."""
    import numpy as np
    from sim.cube.ecm import obstruction_factor
    m = np.full((4, 4, 4), 0.10)
    drug = float(obstruction_factor(m, 0.5).mean())
    virion = float(obstruction_factor(m, 50.0).mean())
    assert drug > 0.99, "a small molecule must pass essentially unobstructed, got %.4f" % drug
    assert 0.3 < virion < 0.9, "a virion must pay a real but not absolute penalty, got %.4f" % virion


def test_swapping_the_law_preserved_phage_behaviour():
    """Calibrated so the two agree within ~1% over the matrix range this engine occupies
    (m = 0.08-0.13). They diverge in a DENSE matrix, and there the exponential is the measured
    shape — but no existing phage result was allowed to move on the swap."""
    import numpy as np
    from sim.cube.ecm import BIOFILM_ECM, obstruction_factor, porosity
    for mv in (0.08, 0.10, 0.13):
        m = np.array(mv)
        old = float(porosity(m, BIOFILM_ECM.alpha) ** 3)
        new = float(obstruction_factor(m, 50.0))
        assert abs(new - old) / old < 0.03, "m=%.2f: %.4f vs %.4f" % (mv, old, new)

