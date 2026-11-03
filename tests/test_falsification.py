"""Falsification-harness tests.

These do NOT assert that the model is correct — it mostly is not. They assert that the harness
runs, that it is honest about what it measures, and that the recorded status of each check is the
status the code actually produces. A test failing here means a check CHANGED, which is exactly the
event worth being told about: either something regressed, or something got fixed and the scoreboard
needs updating to say so.
"""

from __future__ import annotations

import pytest

from sim.cube.falsification import CHECKS, run_all


def test_harness_runs_and_returns_every_check():
    r = run_all(verbose=False)
    assert set(r) == {c.id for c in CHECKS}
    assert len(r) == len(CHECKS)
    assert len({c.id for c in CHECKS}) == len(CHECKS), "duplicate check id"


def test_every_status_is_one_of_the_three():
    r = run_all(verbose=False)
    for cid, res in r.items():
        assert res.status in ("PASS", "FAIL", "BLOCKED"), (cid, res.status)


def test_recorded_status_matches_reality():
    """The `known` field is the scoreboard's memory. Drift means something moved."""
    r = run_all(verbose=False)
    drift = {c.id: (c.known, r[c.id].status) for c in CHECKS if c.known != r[c.id].status}
    assert not drift, ("check status changed (known -> now); update `known` if this is an "
                       "improvement: %s" % drift)


def test_every_check_carries_a_citation_and_a_claim():
    """A falsification target without a source is an opinion."""
    for c in CHECKS:
        assert c.citation.strip(), c.id
        assert c.claim.strip(), c.id
        assert len(c.citation) > 20, ("citation too thin to be checkable: %s" % c.id)


def test_failures_explain_themselves():
    """A red light must say WHY, or it cannot direct the next piece of work."""
    r = run_all(verbose=False)
    for c in CHECKS:
        res = r[c.id]
        if res.status in ("FAIL", "BLOCKED"):
            assert res.note.strip() or res.expected.strip(), c.id


def test_ph_is_dynamic_and_sugar_is_no_longer_inert():
    """pH became a solved field on 2026-08-01, and this records what that changed.

    Before: pH took exactly 2 values across the cube, nothing ate sugar, nothing made acid, and a
    10x dietary sugar change moved composition by EXACTLY 0.00000. F6 passed as the pH-clamped
    control arm (Bradshaw 1989 found the same null with pH held), and F7 was BLOCKED.

    Now: pH is solved from charge balance over lactate/acetate against salivary phosphate and
    bicarbonate buffers, so F7 passes on its own terms. F6 still passes, but for a weaker reason —
    the shift is small rather than absent. That is the honest state: the sugar -> acid -> pH ->
    growth chain is wired end to end, and its effect size is modest because the emergent pH span is
    ~0.7 units rather than the 1.5-2 of a real Stephan challenge.
    """
    r = run_all(verbose=False)
    assert r["F7"].status == "PASS", "pH should be a solved field now"
    assert "distinct values" in r["F7"].measured

    # F6 was reframed once pH became dynamic: it now tests Bradshaw's TREATMENT arm (sugar selects
    # for aciduric species) rather than the pH-clamped control arm. Assert the semantics, not the
    # wording — parsing a float out of the message text is how this test broke the first time.
    assert r["F6"].status == "PASS", "aciduric selection under acid should reproduce"
    assert "aciduric" in r["F6"].measured.lower()


# --------------------------------------------------------------- the protein economy (2026-08-02)
# The pocket runs two nutrient economies and the engine modelled only one. These pin the second.

def _severe(**kw):
    from sim.cube.faces import FaceConditions
    base = dict(serum_heme=1.0, oral_sugar=0.0, serum_peptide=2.5)
    base.update(kw)
    return FaceConditions(**base)


def _run_pocket(cond, steps=120, dim=12):
    import numpy as np
    from sim.cube.fused import CubeEpoch4
    from sim.cube.grid import CubeGrid
    from sim.cube.tissues import oral_tissue
    g = CubeGrid(nx=dim, ny=dim, nz=dim)
    eng = CubeEpoch4(g, tissue=oral_tissue(cond))
    eng.seed_uniform()
    for _ in range(steps):
        eng.step(0.1)
    return g, eng


def test_no_gcf_means_no_proteolysis():
    """serum_peptide defaults to 0, so the protein economy stays switched off without GCF.

    This used to also assert NH3 == 0, which was correct when proteolysis was the ONLY ammonia
    source and is wrong now. Saliva carries 3-5 mmol/L urea whether or not the pocket is
    inflamed (PMID 9438574), so ureolytic ammonia is expected in a sugar-fed cube with no GCF at
    all — that is the alkali generation which gives a Stephan curve its recovery limb. The claim
    worth defending is that no CREVICULAR PROTEIN means no proteolysis.
    """
    from sim.cube.faces import FaceConditions
    g, _ = _run_pocket(FaceConditions(serum_heme=1.0, oral_sugar=0.3), steps=40)
    assert float(g.get_field("peptide").sum()) == 0.0
    assert float(g.get_field("hemoprotein").sum()) == 0.0


def test_gcf_makes_the_deep_pocket_alkaline_against_an_acid_surface():
    """The measured signature of a periodontal pocket: saccharolytic plaque near 6.5 at the mouth,
    proteolytic depth at 7.4-8.5 (Eggert 1991; Bickel & Cimasoni 1985). One cube, both."""
    import numpy as np
    g, _ = _run_pocket(_severe())
    ph = g.get_field("pH")
    surface = float(np.median(ph[:, :, -4:]))
    deep = float(np.median(ph[-4:, :, :4]))
    assert deep > surface + 0.1, "the depth must be more alkaline than the surface"
    assert 7.0 < deep < 8.6, "deep pocket should land in the measured band, got %.2f" % deep


def test_proteolysis_is_net_alkaline_not_net_acid():
    """The bug this pins: SCFAs were converted at the LACTATE scale while ammonia had its own,
    six-fold apart, which made protein fermentation acidify the pocket. Same reaction, one scale."""
    import numpy as np
    from sim.cube.tissues import NH3_YIELD, SCFA_YIELD
    assert NH3_YIELD > SCFA_YIELD, "deamination must out-produce its own acid on a common scale"
    lo, _ = _run_pocket(_severe(serum_peptide=0.5))
    hi, _ = _run_pocket(_severe(serum_peptide=4.0))
    assert float(np.median(hi.get_field("pH")[-4:, :, :4])) > \
           float(np.median(lo.get_field("pH")[-4:, :, :4])), "more GCF must mean more alkaline"


def test_ammonia_is_a_cation_in_the_charge_balance():
    from sim.cube.chemistry import BufferSpec
    b = BufferSpec()
    assert float(b.solve({"lactate": 10.0}, shape=(), ammonia_mM=20.0)) > \
           float(b.solve({"lactate": 10.0}, shape=(), ammonia_mM=0.0))


def test_only_deaminating_species_make_ammonia():
    """E. corrodens ferments peptide with no NH3 byproduct in the catalog and must not silently
    alkalinise the pocket."""
    from sim.cube.tissues import _NH3_PRODUCERS, _PROTEOLYTIC
    assert "Ec" in _PROTEOLYTIC
    assert "Ec" not in _NH3_PRODUCERS
    assert {"Pg", "Td", "Tf", "Fn", "Pi"} <= set(_NH3_PRODUCERS)


def test_gcf_buffer_blends_toward_serum_without_moving_its_own_resting_ph():
    """GCF is a serum transudate: five times the bicarbonate of saliva, resting at 7.4. Adding that
    bicarbonate WITHOUT its counter-cation would acidify the depth instead of alkalinising it."""
    import numpy as np
    from sim.cube.chemistry import (SALIVA_BICARB_MM, SERUM_BICARB_MM, SERUM_PH, sid_field,
                                    solve_ph)
    assert SERUM_BICARB_MM > SALIVA_BICARB_MM * 4
    for target, bic in ((6.8, SALIVA_BICARB_MM), (SERUM_PH, SERUM_BICARB_MM)):
        sid = sid_field(target, bicarb_mM=bic)
        got = float(solve_ph({}, sid, bicarb_mM=bic, shape=()))
        assert abs(got - target) < 1e-3, (target, got)


# ------------------------------------------------------------------ the oxygen envelope (2026-08-02)

def test_red_complex_envelopes_span_their_measured_habitat():
    """Mettraux, Gusberti & Graf 1984 (J Periodontol 55:516-21, PMID 6592325) measured pO2 at the
    BASE of 111 untreated 5-10 mm pockets: 5-27 mmHg, mean 13.3 mmHg = 1.75% atm, and concluded
    the pocket "does not represent a completely anaerobic environment".

    These organisms are routinely isolated from exactly that environment, so an envelope that caps
    them below it excludes them from their own habitat. The catalog used to cap Pg at 0.2% atm.
    """
    from sim.species import BY_ID
    mean_base_pct_atm = 13.3 / 760.0 * 100.0        # 1.75
    assert BY_ID["Pg"].envelope.o2[1] >= mean_base_pct_atm
    for sid in ("Td", "Tf"):
        assert BY_ID[sid].envelope.o2[1] >= 1.0


def test_the_envelope_max_is_load_bearing():
    """The shipped width was max(1.0, (hi-lo)/2), which floored every tight envelope at 1.0 and made
    the catalog's own numbers inert: Pg at 0.2 and Td at 0.1 produced an identical curve, and
    widening either changed nothing."""
    from sim.cube.tissues import EDGE_VIABILITY, _niche_width
    import numpy as np
    narrow = _niche_width(0.0, 2.0, 0.0)
    wide = _niche_width(0.0, 8.0, 0.0)
    assert wide > narrow, "a wider envelope must give a wider niche"
    # viability at the envelope edge is EDGE_VIABILITY by construction, above the resolution floor
    assert np.exp(-((8.0 - 0.0) / wide) ** 2) == pytest.approx(EDGE_VIABILITY, rel=1e-6)


def test_niche_width_floor_is_a_resolution_limit_not_a_biology_claim():
    """Removing the floor entirely broke mesh-independence: rupture_index and oral_efflux were
    still falling monotonically at dim 24. The floor is what the lattice can resolve."""
    from sim.cube.tissues import NICHE_WIDTH_FLOOR, _niche_width
    assert _niche_width(0.0, 0.5, 0.1) == NICHE_WIDTH_FLOOR       # narrower than resolvable
    assert _niche_width(0.0, 2.0, 0.0) > NICHE_WIDTH_FLOOR        # resolvable, so derived


def test_the_keystone_grows_in_a_dysbiotic_pocket_and_not_a_healthy_one():
    """The ecological switch the whole model exists to represent. Measured LOCALLY in the deep
    pocket, because a keystone confined to the pocket base SHOULD have a low cube-wide share."""
    import numpy as np
    from sim.cube.faces import FaceConditions
    from sim.cube.fused import DECAY

    DEEP = (slice(-4, None), slice(None), slice(0, 4))

    def deep_growth_ratio(cond):
        g, eng = _run_pocket(cond, steps=140, dim=12)
        n = g.species_index("Pg")
        ex = eng._expressibility("Pg")
        total = g.composition.sum(axis=3)
        gn = eng.genomes["Pg"]
        persist = 0.5 + 0.5 * gn["defense"]
        ratio = (gn["growth"] * eng.expression[..., n] * (1.0 - total)) / \
                np.maximum(DECAY * (1.0 - ex) * (1.0 - persist), 1e-12)
        return float(np.median(ratio[DEEP]))

    healthy = deep_growth_ratio(FaceConditions(oral_sugar=0.3))
    severe = deep_growth_ratio(_severe())
    assert healthy < 0.5, "a healthy sugar-fed pocket must not sustain the keystone"
    assert severe > healthy * 3, "the dysbiotic pocket must be a different regime, not a nudge"


# ------------------------------------------------------- the regime pairing (F9/F10/F11, 2026-08-02)

def test_regime_paired_checks_are_registered():
    from sim.cube.falsification import CHECKS
    ids = {c.id for c in CHECKS}
    assert {"F9", "F10", "F11"} <= ids


def test_the_dysbiotic_regime_is_actually_a_different_regime():
    """F9-F11 are only meaningful if DYSBIOTIC differs from the cube F1-F3 run."""
    from sim.cube.falsification import DYSBIOTIC
    from sim.cube.sample import BLEEDING_POCKET
    assert DYSBIOTIC.serum_peptide > 0 and BLEEDING_POCKET.serum_peptide == 0
    assert DYSBIOTIC.oral_sugar == 0.0 and BLEEDING_POCKET.oral_sugar > 0


def test_deep_slice_is_local_and_scales_with_the_lattice():
    """Cube-wide averaging is what made a real local effect read as the wrong sign in F2."""
    from sim.cube.falsification import _deep_slice
    for dim in (8, 16, 32):
        sl = _deep_slice(dim)
        k = max(2, dim // 4)
        assert sl[0] == slice(dim - k, None) and sl[2] == slice(0, k)


def test_coaggregation_is_reachable_without_a_phage():
    """It is a property of the COMMUNITY. Requiring a PhageSystem to switch it on would mean no
    phage-free experiment could ever use it."""
    from sim.cube.fused import CubeEpoch4
    from sim.cube.grid import CubeGrid
    eng = CubeEpoch4(CubeGrid(nx=6, ny=6, nz=6))
    assert eng.coaggregation is False        # still off by default
    eng.coaggregation = True
    eng.step(0.1)                            # must not raise


def test_a_phage_system_asking_for_coaggregation_still_gets_it():
    from sim.cube.fused import CubeEpoch4
    from sim.cube.grid import CubeGrid
    from sim.cube.phage import PHI_BRIDGE, PhageSystem
    eng = CubeEpoch4(CubeGrid(nx=6, ny=6, nz=6))
    PhageSystem(panel=(PHI_BRIDGE,), coaggregation=True).prepare(eng)
    assert eng.coaggregation is True


# ------------------------------------------------- keystone mechanism + F12/F13 pairing (2026-08-02)

def test_bleeding_delivers_hemoprotein_not_only_free_heme():
    """Blood carries haemoglobin; heme has to be prised out of it, and oxyhaemoglobin must first be
    oxidised to methaemoglobin before it will come (Smalley 2011, PLoS One 6:e17182, PMID 21390208).
    A face that emits only free heme hands the red complex the resource its proteases exist for."""
    from sim.cube.faces import FaceConditions, apply_face_boundaries
    from sim.cube.grid import CubeGrid
    g = CubeGrid(nx=6, ny=6, nz=6)
    apply_face_boundaries(g, FaceConditions(serum_hemoprotein=1.5))
    assert float(g.get_field("hemoprotein")[-1, :, :].mean()) == pytest.approx(1.5)


def test_heme_liberation_is_weighted_to_who_has_the_protease():
    """P. gingivalis gingipains are the characterised route; P. intermedia's interpain A is named in
    the same review as oxidising haemoglobin; dentilisin and karilysin are far less active."""
    from sim.cube.tissues import HEME_LIBERATORS
    assert HEME_LIBERATORS["Pg"] > HEME_LIBERATORS["Pi"] > HEME_LIBERATORS["Td"]
    assert set(HEME_LIBERATORS) <= {"Pg", "Pi", "Td", "Tf"}


def test_liberated_heme_is_a_public_good_not_a_private_one():
    """It enters the shared field, so any heme-demanding species can use what the keystone frees.
    That is the whole point of calling it a public good."""
    from sim.species import BY_ID
    demanders = [s for s in ("Td", "Tf", "Pi", "Hp", "Ec") if BY_ID[s].envelope.heme >= 1]
    assert len(demanders) >= 4, "the heme-dependent guild must be more than the liberators"


def test_a_linear_public_good_cannot_make_a_keystone():
    """The structural finding, pinned so it is not rediscovered.

    Heme liberation is wired and measured, and sweeping its rate 22x moves deep-pocket heme from
    0.083 to 0.092. The keystone's active biomass is ~7e-4 per voxel, so ANY term linear in it is a
    few-percent effect by construction. F11 therefore stays FAIL, and the note records why.
    """
    from sim.cube.falsification import CHECKS
    f11 = next(c for c in CHECKS if c.id == "F11")
    assert f11.known == "FAIL"


def test_f13_is_paired_to_the_sugar_economy_not_the_dysbiotic_one():
    """Regime pairing means the regime the phenomenon LIVES in, which is not the same regime for
    every check. A protein-fed pocket has almost no lactate, so pairing the Veillonella crossfeed to
    DYSBIOTIC would be pairing it to a cube where the substrate is absent."""
    from sim.cube.falsification import DYSBIOTIC, SUGAR_RICH
    assert SUGAR_RICH.oral_sugar > DYSBIOTIC.oral_sugar
    assert DYSBIOTIC.oral_sugar == 0.0          # a protein-fed pocket has no lactate economy
    assert SUGAR_RICH.serum_peptide == 0.0      # and the sugar cube has no protein economy


# ------------------------------------------------ keystone subversion + arrival (2026-08-02, late)

def test_subversion_is_switch_like_not_linear():
    """The whole reason this mechanism can make a keystone where a public good could not.
    Hajishengallis 2011 (PMID 22036469) is about 'very low colonization levels', so the response
    must saturate early rather than scale with abundance."""
    from sim.cube.host import SUBVERSION_HALF, SUBVERSION_MAX
    assert 0.0 < SUBVERSION_HALF < 1e-2, "half-max must be at a LOW abundance to be a switch"
    assert 0.0 < SUBVERSION_MAX < 1.0, "complement is degraded, not abolished"


def test_subversion_acts_on_the_whole_community_not_just_the_keystone():
    """A keystone disables the clearance holding everyone down. If it only protected itself it
    would be an evader, which the model already had (KEYSTONE_EVASION) and which is not a keystone."""
    import inspect
    from sim.cube.host import clearance_scale
    src = inspect.getsource(clearance_scale)
    assert "species" not in src, "clearance_scale must be a field, not per-species"


def test_no_host_means_nothing_to_subvert():
    """The conjunctive requirement: with no host clearance the subversion term must be inert."""
    from sim.cube.fused import CubeEpoch4
    from sim.cube.grid import CubeGrid
    eng = CubeEpoch4(CubeGrid(nx=6, ny=6, nz=6))
    eng.seed_uniform()
    assert eng.mortality_mult is None
    before = float(eng.grid.composition.sum())
    eng.step(0.1)
    assert float(eng.grid.composition.sum()) != before      # the sim still runs, unaffected


def test_dispersal_defeats_the_absorbing_zero():
    """The gap _clean_surface apologises for: a species at exactly zero could never return."""
    from sim.cube.dispersal import Dispersal
    from sim.cube.fused import CubeEpoch4
    from sim.cube.grid import CubeGrid

    g = CubeGrid(nx=8, ny=8, nz=8)
    eng = CubeEpoch4(g)
    g.composition[:] = 0.0
    eng.step(0.1)
    assert float(g.composition.sum()) == 0.0, "without dispersal, zero must stay zero"

    eng.dispersal = Dispersal()
    for _ in range(20):
        eng.step(0.1)
    assert float(g.composition.sum()) > 0.0, "with dispersal, a bare surface must be colonisable"


def test_dispersal_respects_knockouts():
    """Every ablation in the harness depends on knockout being permanent. Immigration must not
    quietly undo them."""
    from sim.cube.dispersal import Dispersal
    from sim.cube.fused import CubeEpoch4
    from sim.cube.grid import CubeGrid

    g = CubeGrid(nx=8, ny=8, nz=8)
    eng = CubeEpoch4(g)
    eng.seed_uniform()
    n = g.species_index("Fn")
    g.composition[..., n] = 0.0
    eng.dispersal = Dispersal(blocked=frozenset({"Fn"}))
    for _ in range(20):
        eng.step(0.1)
    assert float(g.composition[..., n].sum()) == 0.0


def test_pioneers_bind_pellicle_and_late_colonizers_do_not():
    """The asymmetry that makes succession fall out instead of being scheduled."""
    from sim.cube.dispersal import attachment_gate
    from sim.cube.fused import CubeEpoch4
    from sim.cube.grid import CubeGrid

    g = CubeGrid(nx=8, ny=8, nz=8)
    eng = CubeEpoch4(g)
    g.composition[:] = 0.0                      # bare surface: no partners for anyone
    assert float(attachment_gate(eng, "Sa").max()) > 0.5, "a pioneer must colonise bare pellicle"
    assert float(attachment_gate(eng, "Pg").max()) < 0.05, "the keystone must not"


# ------------------------------------------------------ grounded corrections (2026-08-02, evening)

def test_the_cube_represents_a_human_periodontitis_pocket():
    """OPERATOR DECISION (2026-08-04): the cube represents a HUMAN PERIODONTITIS pocket, so Pg is
    seeded at the measured human figure of ~0.8-0.9%, not the <0.01% murine level at which the
    keystone effect was originally demonstrated.

    Both numbers come from Hajishengallis et al. 2011 (PMID 22036469) and they answer different
    questions. <0.01% is where keystone-ness was DEFINED, in mice. ~0.8% of total clones is what an
    active human lesion actually carries. The rest of this cube — GCF flow, pocket depth, the
    proteolytic economy — is built for a human diseased pocket, so the human figure is the coherent
    choice and the murine one would make the community incoherent with its own environment.

    The consequence is recorded rather than hidden: F11's keystone index divides by this share, so
    at 0.9% the index is roughly ten times lower than it would be at the murine level. F11 failing
    its absolute bar is therefore partly a statement about which pocket we chose to model, and the
    note says so. Changing this number is a decision about what the cube IS, not a tuning knob.
    """
    from sim.cube.tissues import _ORAL_ABUND
    share = _ORAL_ABUND["Pg"] / sum(_ORAL_ABUND.values())
    assert 0.005 < share < 0.02, (
        "Pg at %.2f%% — the cube is specified as a human periodontitis pocket (~0.8-0.9%%). "
        "Changing this is an operator decision, not a fix." % (100 * share))


def test_alkali_generation_exists_at_all():
    """The model had no base source, so a Stephan curve could only fall. Acidogens make the fall;
    alkali generators make the rise (PMID 2675801, 12097354, 9602287)."""
    from sim.cube.tissues import ADS_VMAX, ARGINOLYTIC, NH3_PER_UREA, UREOLYTIC, UREOLYSIS_VMAX
    assert UREOLYSIS_VMAX > 0 and ADS_VMAX > 0
    assert NH3_PER_UREA == 2.0, "urea hydrolysis yields two ammonia — why it beats arginine per mole"
    assert "An" in UREOLYTIC, "A. naeslundii is the classic plaque urease organism"
    assert {"Sa", "Sm"} <= set(ARGINOLYTIC), "the sanguinis/mitis group carries ADS"


def test_urea_reaches_the_plaque_it_has_to_neutralise():
    """Ureolysis is useless if its substrate never leaves the boundary voxel — which is exactly
    what happened the first time, because urea was not in the ECM-gated transport list."""
    import inspect
    from sim.cube.tissues import _oral_update_terrain
    src = inspect.getsource(_oral_update_terrain)
    i = src.index("apply_ecm(")
    assert '"urea"' in src[i:i + 400]


def test_bulk_clearance_is_off_by_default_and_is_a_geometry_statement():
    """A 6 mm pocket is genuinely diffusion-isolated at its base; a 100 um supragingival film is
    not (McNee & Geddes, PMID 6818937 — t1/2 about six seconds). The knob represents the thin case
    and must not silently apply to the pocket."""
    from sim.cube.faces import FaceConditions
    assert FaceConditions().bulk_clearance == 0.0


def test_antimicrobial_kill_is_differentiated():
    """F4's blocker in one line: a uniform multiplicative kill cannot change relative abundance."""
    from sim.cube.antimicrobial import apply_antimicrobial
    from sim.cube.falsification import _dysbiotic_engine, _run
    eng = _run(_dysbiotic_engine(), 60)
    survival = apply_antimicrobial(eng, dose=4.0)
    assert max(survival.values()) - min(survival.values()) > 0.05
    assert survival["Sa"] < survival["Ca"], "a cationic agent should spare the fungus most"


def test_crossfeed_demand_is_near_obligate_but_not_absolute():
    """Verified: growth strictly conditional on a partner (PMID 18805978), lactate an obligate
    energy source (PMID 16887614), described as sole carbon source (PMID 31235964). Not 1.0,
    because V. parvula also ferments malate (PMID 1645132) and eats Actinomyces succinate
    (PMID 6935027) — routes this model does not represent."""
    from sim.cube.tissues import _ORAL_GENOMES
    d = _ORAL_GENOMES["Vp"]["demand"]["lactate_sat"]
    assert 0.9 <= d < 1.0


def test_succession_labels_were_verified_not_assumed():
    """VERIFIED 2026-08-04. Five succession labels were suspected of disagreeing with COAGG_EDGES;
    all five were checked against the literature with an adversarial audit, and none changed.

    Pinned because the suspicion is easy to re-derive and wrong. V. parvula in particular is a
    strict anaerobe AND a genuine early coloniser, so 'anaerobe therefore late' must not be
    reintroduced as a heuristic. See the block comment on Species.succession.
    """
    from sim.species import BY_ID
    assert BY_ID["Vp"].succession == "early"      # confirmed strong, PMID 25917902 / 3178207
    assert BY_ID["Fa"].succession == "late"       # confirmed moderate
    assert BY_ID["Pm"].succession == "late"       # challenge to this was rejected on audit
    assert BY_ID["Lb"].succession == "late"       # challenge rejected — screening-panel artifact


def test_veillonella_still_needs_a_partner_to_attach():
    """The four-tier nuance that survived: Veillonella is 'early' but not 'initial' — it does not
    bind the acquired pellicle independently (PMID 18805978, 20154130). Dispersal must keep making
    it find a partner, or it becomes a pioneer it is not."""
    from sim.cube.dispersal import PELLICLE_BINDERS
    from sim.species import BY_ID
    assert BY_ID["Vp"].archetype not in PELLICLE_BINDERS
    assert BY_ID["Sa"].archetype in PELLICLE_BINDERS

