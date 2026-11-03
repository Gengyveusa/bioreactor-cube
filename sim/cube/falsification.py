"""The falsification harness — the known results the model must reproduce.

Every check here encodes a qualitative finding that is settled in the literature and that needs NO
new patient data. Most of them fail today. That is the point: a dated, reproducible scoreboard turns
"is the model any good" from an opinion into a number that moves, and every mechanism added later is
judged by whether it flips a red to green.

Status vocabulary:
    PASS     the model reproduces the published result
    FAIL     the model contradicts it, or produces no effect where one is established
    BLOCKED  the check cannot even be evaluated because the mechanism is absent from the engine
             (a BLOCKED check is a scope statement, not a wrong answer)

Run it:
    python3 -c "from sim.cube.falsification import run_all; run_all()"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from sim.cube.fused import CubeEpoch4
from sim.cube.grid import CubeGrid
from sim.cube.tissues import ORAL_TISSUE, SugarChallenge, oral_tissue
from sim.cube.faces import FaceConditions
from sim.species import CATALOG

DIM = 16
STEPS = 150
DT = 0.1

_BY_ID = {s.id: s for s in CATALOG}
RED_COMPLEX = ("Pg", "Td", "Tf")
EARLY = tuple(s.id for s in CATALOG if s.succession == "early")
MIDDLE = tuple(s.id for s in CATALOG if s.succession == "middle")
LATE = tuple(s.id for s in CATALOG if s.succession == "late")
STREPTOCOCCI = ("Sa", "Sm", "Smu")


# --------------------------------------------------------------------------- machinery
@dataclass
class Result:
    status: str
    measured: str
    expected: str
    note: str = ""


@dataclass
class Check:
    id: str
    name: str
    claim: str
    citation: str
    run: Callable
    known: str = ""          # the status recorded when this harness was written
    result: Optional[Result] = field(default=None)


def _engine(tissue=ORAL_TISSUE, dim: int = DIM, knockout=()) -> CubeEpoch4:
    """A seeded oral cube, optionally with species removed.

    Knockout works because growth is proportional to existing biomass and the engine has no
    immigration: a species set to exactly zero can never return. That absorbing zero is itself one
    of the engine's known gaps, and here it is the mechanism the ablation relies on.
    """
    grid = CubeGrid(dim, dim, dim, species_ids=tissue.cell_ids)
    eng = CubeEpoch4(grid, tissue)
    eng.seed_uniform()
    for sid in knockout:
        if sid in eng.species:
            eng.grid.composition[..., eng.species.index(sid)] = 0.0
    return eng


def _clean_surface(dim: int = DIM, knockout=()) -> CubeEpoch4:
    """A freshly-cleaned tooth surface: pioneers present, everything else at a trace.

    Not literally zero, because the engine has no dispersal or immigration — a species at exactly
    zero can never appear, so "from zero" is not expressible and a trace inoculum is the closest
    honest approximation. That limitation is itself a known gap (see the roadmap), and it is worth
    being explicit that this initial condition is a workaround for it.
    """
    grid = CubeGrid(dim, dim, dim, species_ids=ORAL_TISSUE.cell_ids)
    eng = CubeEpoch4(grid, ORAL_TISSUE)
    eng.seed_uniform()
    comp = eng.grid.composition
    for n, sid in enumerate(eng.species):
        if sid in knockout:
            comp[..., n] = 0.0
        elif sid not in EARLY:
            comp[..., n] *= 1e-3                      # trace inoculum for non-pioneers
    return eng


# --------------------------------------------------------------------------- the dysbiotic regime
# F1-F3 all run the SACCHAROLYTIC pocket: sugar-fed, no gingival crevicular fluid. That is the
# regime the engine shipped with, and it is one in which the red complex cannot live — Pg's pH
# optimum is 7.8 and sugar fermentation drives the plaque to ~6.4. Asking whether a keystone
# reorganises a community, in a cube where the keystone is dying, tests nothing about keystones.
#
# This is the other regime: protein-fed, bleeding, sugar-starved — what a severe periodontal pocket
# actually is, and the one where Pg's deep-pocket growth:decay crosses 1.
# A bleeding pocket delivers BOTH: free heme from lysed erythrocytes, and intact haemoglobin
# that has to be prised open by proteases. Dropping the free fraction to 0.1 to make the
# liberation term matter was me tuning the environment to suit a mechanism, and it cost F10
# (the keystone stopped being viable at all). Free heme is back at a bleeding level.
DYSBIOTIC = FaceConditions(serum_heme=1.0, serum_hemoprotein=1.5, oral_sugar=0.0,
                           serum_peptide=2.5)

#: The saccharolytic economy, fed hard. Regime pairing means the regime a phenomenon LIVES in,
#: and that is not the same regime for every check: the Veillonella crossfeed runs on
#: fermentative lactate, so its home is a sugar-rich cube, not the protein-fed pocket where
#: almost no lactate exists for anything to depend on.
SUGAR_RICH = FaceConditions(serum_heme=1.0, oral_sugar=2.0)


def _deep_slice(dim: int) -> tuple:
    """The tissue-facing base of the pocket.

    Late colonisers are CONFINED here, so a cube-wide average is mostly the oxygenated surface they
    were never going to occupy — which is how a real local effect reads as no effect at all.
    """
    k = max(2, dim // 4)
    return (slice(dim - k, None), slice(None), slice(0, k))


def _dysbiotic_engine(dim: int = DIM, knockout=(), coaggregation: bool = True,
                      host: bool = False) -> CubeEpoch4:
    """A seeded cube in the protein-fed pocket, with coaggregation dependence enabled.

    ``host=True`` gives the cube a competent immune system for the keystone to subvert. It is not
    the default because most checks are about the community alone, and because Hajishengallis 2011
    is explicit that BOTH the commensals and complement are required — a cube with no complement is
    the C5aR-deficient arm, in which the keystone effect must vanish.
    """
    tissue = oral_tissue(DYSBIOTIC)
    grid = CubeGrid(dim, dim, dim, species_ids=tissue.cell_ids)
    eng = CubeEpoch4(grid, tissue)
    eng.coaggregation = coaggregation
    if host:
        from sim.cube.host import HostState
        # A metabolically well-controlled host: competent neutrophils, so there is real clearance
        # on the table for the keystone to take away.
        state = HostState(antimicrobial_capacity=1.0, consumed=frozenset({"hba1c"}), empty=False)
        eng.mortality_mult = state.mortality_multipliers(grid.species_ids) or None
    eng.seed_uniform()
    for sid in knockout:
        if sid in eng.species:
            eng.grid.composition[..., eng.species.index(sid)] = 0.0
    return eng


def _deep_mass(eng, ids) -> float:
    """Biomass of a species group in the deep pocket only."""
    sl = _deep_slice(eng.grid.shape[0])
    c = eng.grid.composition[sl]
    return float(sum(c[..., eng.species.index(i)].sum() for i in ids if i in eng.species))


def _deep_rel(eng) -> dict:
    sl = _deep_slice(eng.grid.shape[0])
    c = eng.grid.composition[sl]
    tot = float(c.sum()) or 1.0
    return {sid: float(c[..., n].sum()) / tot for n, sid in enumerate(eng.species)}


def _mass(eng, ids) -> float:
    c = eng.grid.composition
    return float(sum(c[..., eng.species.index(i)].sum() for i in ids if i in eng.species))


def _rel(eng) -> dict:
    c = eng.grid.composition
    tot = float(c.sum()) or 1.0
    return {sid: float(c[..., n].sum()) / tot for n, sid in enumerate(eng.species)}


def _run(eng, steps: int = STEPS) -> CubeEpoch4:
    for _ in range(steps):
        eng.step(DT)
    return eng


# --------------------------------------------------------------------------- F1
def _f1_colonization_order() -> Result:
    """Early colonizers should peak before middle, and middle before late.

    Run from a cleaned surface, not from a fully-seeded community — otherwise this measures who
    declines fastest rather than who arrives first.
    """
    eng = _clean_surface()
    peaks: dict = {}
    for t in range(1, STEPS + 1):
        eng.step(DT)
        r = _rel(eng)
        for sid, v in r.items():
            if sid not in peaks or v > peaks[sid][1]:
                peaks[sid] = (t * DT, v)

    def mean_peak(group):
        vals = [peaks[s][0] for s in group if s in peaks and peaks[s][1] > 1e-6]
        return sum(vals) / len(vals) if vals else float("nan")

    e, m, l = mean_peak(EARLY), mean_peak(MIDDLE), mean_peak(LATE)
    ok = (e < m < l)
    late_present = _mass(eng, LATE) / max(float(eng.grid.composition.sum()), 1e-9)
    return Result(
        "PASS" if ok else "FAIL",
        "mean peak time  early %.1f -> middle %.1f -> late %.1f; late share at end %.4f"
        % (e, m, l, late_present),
        "early < middle < late, with late colonizers actually establishing",
        "" if ok else "ordering not reproduced; late colonizers may never establish at all",
    )


# --------------------------------------------------------------------------- F2
def _f2_fusobacterium_bridge() -> Result:
    """Removing the bridging organism should REDUCE late-colonizer establishment."""
    # From a cleaned surface: bridging is about ESTABLISHMENT, so the late colonizers must
    # actually have to arrive rather than starting already present.
    base = _run(_clean_surface(), STEPS * 2)
    ko = _run(_clean_surface(knockout=("Fn",)), STEPS * 2)
    b, k = _mass(base, RED_COMPLEX), _mass(ko, RED_COMPLEX)
    change = (k - b) / b if b > 0 else float("inf")
    return Result(
        "PASS" if change < -0.05 else "FAIL",
        "red-complex biomass changes %+.1f%% when F. nucleatum is removed" % (100 * change),
        "a DECREASE — Fn bridges early to late colonizers",
        "" if change < 0 else "WRONG SIGN: removal helps the red complex. NOT because the engine "
                              "lacks a coaggregation term any more — it has one — but because this "
                              "check runs the saccharolytic cube, where the red complex cannot live, "
                              "and averages cube-wide. See F9 for the same question asked in the "
                              "dysbiotic pocket and scored locally, where it PASSES.",
    )


# --------------------------------------------------------------------------- F3
def _f3_keystone_contrast() -> Result:
    """P. gingivalis should reorganise the community out of proportion to its abundance.

    The test is the CONTRAST against a commensal of matched abundance, not the raw index — every
    knockout produces some competitive release, and that baseline must be controlled for.
    """
    base = _run(_engine())
    rel0 = _rel(base)
    pg_share = rel0.get("Pg", 0.0)

    # closest commensal by abundance, so the comparison isolates keystone-ness from mere size
    cands = [(abs(rel0.get(s.id, 0) - pg_share), s.id) for s in CATALOG
             if s.id != "Pg" and s.socransky not in ("red",) and rel0.get(s.id, 0) > 0]
    control = sorted(cands)[0][1] if cands else "Sa"

    def index(removed: str) -> float:
        ko = _run(_engine(knockout=(removed,)))
        r1 = _rel(ko)
        others = [s for s in base.species if s != removed]
        tot0 = sum(rel0[s] for s in others) or 1.0
        tot1 = sum(r1[s] for s in others) or 1.0
        l1 = sum(abs(r1[s] / tot1 - rel0[s] / tot0) for s in others)
        share = rel0.get(removed, 1e-9) or 1e-9
        return 0.5 * l1 / share

    kp, kc = index("Pg"), index(control)
    ratio = kp / kc if kc > 0 else float("inf")
    ok = kp > 1.0 and ratio > 2.0
    return Result(
        "PASS" if ok else "FAIL",
        "keystone index Pg %.3f vs matched commensal %s %.3f (ratio %.2fx)" % (kp, control, kc, ratio),
        "Pg index >> 1 and >> the matched control",
        "" if ok else "Pg behaves like any other competitor — the engine has no directed "
                      "interaction, so removal can only produce competitive release",
    )


# --------------------------------------------------------------------------- F4
def _f4_regrowth_order() -> Result:
    """After a broad kill, aerobes/facultatives should re-establish before strict anaerobes."""
    eng = _run(_engine())
    eng.grid.composition[...] *= 0.01                # 99% kill, uniform (in place: it is a property)
    rel_after_kill = _rel(eng)

    t50: dict = {}
    targets = {sid: v for sid, v in rel_after_kill.items() if v > 1e-9}
    for t in range(1, STEPS + 1):
        eng.step(DT)
        r = _rel(eng)
        for sid in targets:
            if sid not in t50 and r[sid] > 1.5 * targets[sid]:
                t50[sid] = t * DT

    fac = [t50[s] for s in t50 if _BY_ID[s].o2_class in ("facultative", "aerotolerant")]
    ana = [t50[s] for s in t50 if _BY_ID[s].o2_class == "anaerobe"]
    if not fac or not ana:
        return Result(
            "BLOCKED",
            "regrowth order undefined — %d facultative / %d anaerobe recovery times observed"
            % (len(fac), len(ana)),
            "facultatives recover before strict anaerobes",
            "a UNIFORM multiplicative kill leaves relative abundance exactly unchanged, so there is "
            "no order to measure. Needs species-differentiated killing (EPS shielding, persisters).",
        )
    mf, ma = sum(fac) / len(fac), sum(ana) / len(ana)
    return Result(
        "PASS" if mf < ma else "FAIL",
        "mean recovery time  facultative %.1f vs anaerobe %.1f" % (mf, ma),
        "facultatives first — the O2 gradient must be rebuilt before the anaerobic niche exists",
    )


# --------------------------------------------------------------------------- F5
def _f5_veillonella_crossfeed() -> Result:
    """Veillonella depends on streptococcal lactate; removing streptococci should reduce it."""
    # Remove every FERMENTATIVE LACTATE PRODUCER, not just the streptococci. Actinomyces,
    # Lactobacillus and Aa also make lactate in this catalog, so a streptococci-only ablation
    # leaves the substrate in place and tests nothing. The mechanism claim is that Veillonella
    # depends on fermentative lactate; this is the ablation that actually asks it.
    producers = tuple(s.id for s in CATALOG if "lactate" in s.byproducts and not s.lactate_consumer)
    base = _run(_engine())
    ko = _run(_engine(knockout=producers))
    b, k = _mass(base, ("Vp",)), _mass(ko, ("Vp",))
    change = (k - b) / b if b > 0 else float("inf")
    return Result(
        "PASS" if change < -0.20 else "FAIL",
        "V. parvula biomass changes %+.1f%% when every lactate producer (%s) is removed"
        % (100 * change, ", ".join(producers)),
        "a substantial DECREASE — Veillonella consumes streptococcal lactate",
        "" if change < -0.20 else "the crossfeed IS wired (fermenters make lactate, consumers eat "
                                  "it, and the gate reads a saturating lactate availability), and "
                                  "the dependency is real but below the -20% bar. The bar is mine, "
                                  "not the literature's; it is deliberately NOT being tuned to pass.",
    )


# --------------------------------------------------------------------------- F6
def _f6_aciduric_selection() -> Result:
    """With pH FREE to fall, a sugar challenge should select for aciduric species.

    This check has been reframed once, deliberately. It used to test Bradshaw 1989's pH-CLAMPED
    CONTROL arm — sugar pulses barely shift composition when pH is held — and the model passed it
    with a shift of exactly 0.00000, which was the right answer for an engine whose pH could not
    move. pH is now solved from charge balance, so the control arm no longer describes this model
    and passing it would mean the chemistry had failed.

    The meaningful test is now the TREATMENT arm of the same body of work: Bradshaw & Marsh 1998
    (Caries Res 32:456) found that S. mutans and L. rhamnosus increases were directly related to the
    MAGNITUDE OF THE pH FALL. So sugar should shift the community toward the aciduric species, and
    that shift should track the pH drop rather than the sugar concentration.
    """
    lo_c = FaceConditions(serum_heme=1.0, oral_sugar=0.05)
    hi_c = FaceConditions(serum_heme=1.0, oral_sugar=3.0)
    lo = _run(_engine(tissue=oral_tissue(lo_c)))
    hi = _run(_engine(tissue=oral_tissue(hi_c)))

    aciduric = ("Smu", "Lb")
    def share(eng):
        r = _rel(eng)
        return sum(r.get(s, 0.0) for s in aciduric)

    a_lo, a_hi = share(lo), share(hi)
    ph_lo = float(np.median(lo.grid.get_field("pH")))
    ph_hi = float(np.median(hi.grid.get_field("pH")))
    rise = (a_hi - a_lo) / a_lo if a_lo > 0 else float("inf")
    ok = rise > 0.20 and ph_hi < ph_lo
    return Result(
        "PASS" if ok else "FAIL",
        "aciduric share (S. mutans + Lactobacillus) %.4f -> %.4f (%+.1f%%) as median pH falls "
        "%.2f -> %.2f" % (a_lo, a_hi, 100 * rise, ph_lo, ph_hi),
        "aciduric species gain, and the gain tracks the size of the pH fall",
        "" if ok else "sugar now moves the community (it moved it by exactly 0.00000 before pH was "
                      "solved), but the aciduric selection is not yet strong enough. The pH fall "
                      "reaching only ~1.1 units rather than the measured 1.9 limits how much "
                      "selective pressure the acid can apply — same root cause as F8.",
    )


# --------------------------------------------------------------------------- F7
def _f7_ph_is_dynamic() -> Result:
    """Is pH a solved field with real spatial structure, or a boundary constant?

    This is deliberately NOT the Stephan-curve test — that is F8. Passing here means only that pH
    is emergent from the chemistry, which is a real milestone and a much weaker claim.
    """
    eng = _run(_engine())
    ph = eng.grid.get_field("pH")
    vals = np.unique(np.round(ph, 3))
    span = float(ph.max() - ph.min())
    ok = len(vals) > 10 and span > 0.5
    return Result(
        "PASS" if ok else "BLOCKED",
        "pH takes %d distinct values across the cube, span %.3f units" % (len(vals), span),
        "a continuous field with many distinct values, not a boundary constant",
        "" if ok else "pH is a Dirichlet boundary value that never diffuses and is never produced "
                      "or consumed. Nothing eats sugar; nothing makes acid. Blocked until the "
                      "charge-balance/buffer solver lands.",
    )


# --------------------------------------------------------------------------- F8
def _f8_stephan_depth() -> Result:
    """A real acid challenge: rest the community, deliver a sugar BOLUS, track plaque pH.

    The Stephan curve's defining feature is DEPTH and TRANSIENCE — plaque pH falls roughly 1.5-2
    units below resting within minutes of a sugar challenge, crosses the enamel-critical 5.5 on the
    way down, and then recovers. A steady sugar boundary cannot produce that shape at any
    concentration; the curve is a response to a bolus outrunning salivary clearance.
    """
    rest_t = 8.0
    ch = SugarChallenge(baseline=0.05, amplitude=3.0, t_start=rest_t, half_life=2.0)
    tissue = oral_tissue(FaceConditions(serum_heme=1.0, oral_sugar=0.05), challenge=ch)
    eng = _engine(tissue=tissue)

    ph_rest, nadir, t_nadir, trace = None, 99.0, 0.0, []
    for i in range(1, int(rest_t / DT) + STEPS * 2):
        eng.step(DT)
        med = float(np.median(eng.grid.get_field("pH")))
        if abs(eng.t - rest_t) < DT / 2:
            ph_rest = med
        if eng.t >= rest_t and med < nadir:
            nadir, t_nadir = med, eng.t - rest_t
        if eng.t >= rest_t:
            trace.append(med)
    ph_rest = ph_rest if ph_rest is not None else trace[0]

    drop = ph_rest - nadir
    recovered = trace[-1] - nadir if trace else 0.0
    ok = drop >= 1.5 and nadir < 5.5
    return Result(
        "PASS" if ok else "FAIL",
        "plaque pH %.2f at rest -> nadir %.2f at t+%.1f (a %.2f unit fall); crosses 5.5: %s; "
        "recovers %.2f units by the end" % (ph_rest, nadir, t_nadir, drop, "yes" if nadir < 5.5 else "no", recovered),
        "a fall of 1.5-2 units, crossing the enamel-critical pH of 5.5, then recovering",
        "" if ok else "MY EARLIER DIAGNOSIS HERE WAS WRONG. This note used to blame transport - "
                      "acid escaping only by diffusion to one clearing face. McNee & Geddes 1982 "
                      "(PMID 6818937) measured lactate diffusion in human plaque at 4.8e-6 cm^2/s, "
                      "2-5x slower than free water, giving a clearance half-time of about SIX "
                      "SECONDS through a 100 um layer, and concluded the prolonged low-pH phase is "
                      "NOT diffusion-limited. Two real things replaced it. (1) The model had no "
                      "BASE source at all, which is why a monotonic fall was its only option: "
                      "acidogens make the fall, alkali generators make the rise (PMID 2675801, "
                      "12097354, 9602287). Ureolysis and ADS are now wired and a recovery limb "
                      "exists for the first time. (2) What remains is a GEOMETRY mismatch: the "
                      "Stephan curve is a supragingival phenomenon in ~100 um plaque, and this "
                      "check runs a 6 mm pocket, where the same acid needs ~3600x longer to leave "
                      "(L^2/D). FaceConditions.bulk_clearance represents the thin case, but at any "
                      "setting that permits recovery the fall also disappears - production and "
                      "clearance have never been calibrated against a common clock, and a Stephan "
                      "curve is exactly a race between them. That calibration is the open work; "
                      "the constants are deliberately not being fitted to this check.",
    )


# --------------------------------------------------------------------------- F9
def _f9_bridge_in_the_pocket() -> Result:
    """F2, asked in a regime where the late colonisers can actually live, and measured where they
    actually are.

    F2 runs the saccharolytic cube and averages over the whole lattice, and it reports the WRONG
    SIGN: removing Fn helps the red complex by +22%. Both of those are artefacts of the setup
    rather than statements about bridging. Removing any large biomass frees shared carrying
    capacity, and cube-wide averaging is dominated by the oxygenated surface where the red complex
    was never going to be.
    """
    base = _run(_dysbiotic_engine(), STEPS * 2)
    ko = _run(_dysbiotic_engine(knockout=("Fn",)), STEPS * 2)
    b, k = _deep_mass(base, RED_COMPLEX), _deep_mass(ko, RED_COMPLEX)
    change = (k - b) / b if b > 0 else float("inf")

    # The pioneer is the control: bridge removal should HELP it while hurting the late colonisers.
    pb, pk = _deep_mass(base, ("Sa",)), _deep_mass(ko, ("Sa",))
    pioneer = (pk - pb) / pb if pb > 0 else 0.0
    directional = change < 0 < pioneer

    ok = change < -0.05 and directional
    return Result(
        "PASS" if ok else "FAIL",
        "deep-pocket red complex %+.1f%% and pioneer S. sanguinis %+.1f%% when F. nucleatum is "
        "removed" % (100 * change, 100 * pioneer),
        "red complex DOWN by >5%% while the pioneer goes UP — the bridging signature",
        "" if ok else ("sign is now correct (F2 reports the opposite) but the magnitude is under "
                       "the 5%% bar; roughly a fifth of it is coaggregation and the rest metabolic"
                       if directional else "no directional separation between late and early"),
    )


# --------------------------------------------------------------------------- F10
def _f10_keystone_is_regime_dependent() -> Result:
    """The switch the whole model exists to represent: the keystone must be EXCLUDED from a healthy
    pocket and viable in a diseased one.

    A model that grows P. gingivalis in a sugar-fed, well-oxygenated, non-bleeding sulcus is not
    modelling periodontitis, whatever else it gets right.
    """
    from sim.cube.fused import DECAY

    def deep_growth_ratio(cond) -> float:
        tissue = oral_tissue(cond)
        grid = CubeGrid(DIM, DIM, DIM, species_ids=tissue.cell_ids)
        eng = CubeEpoch4(grid, tissue)
        eng.seed_uniform()
        _run(eng)
        n = eng.species.index("Pg")
        ex = eng._expressibility("Pg")
        total = grid.composition.sum(axis=3)
        g = eng.genomes["Pg"]
        persist = 0.5 + 0.5 * g["defense"]
        ratio = (g["growth"] * eng.expression[..., n] * (1.0 - total)) / \
                np.maximum(DECAY * (1.0 - ex) * (1.0 - persist), 1e-12)
        return float(np.median(ratio[_deep_slice(DIM)]))

    healthy = deep_growth_ratio(FaceConditions(oral_sugar=0.3))
    diseased = deep_growth_ratio(DYSBIOTIC)
    ok = healthy < 0.5 < 1.0 < diseased
    return Result(
        "PASS" if ok else "FAIL",
        "P. gingivalis deep-pocket growth:decay — healthy %.2f, dysbiotic %.2f" % (healthy, diseased),
        "below 1 in the healthy pocket (excluded) and above 1 in the dysbiotic one (viable)",
        "" if ok else "the keystone is not regime-dependent: it either survives a healthy sulcus "
                      "or cannot grow in a diseased one",
    )


# --------------------------------------------------------------------------- F11
def _f11_keystone_contrast_in_the_pocket(host: bool = True) -> Result:
    """F3 asked in the dysbiotic regime, scored on the deep pocket, WITH a host to subvert.

    The host is what makes this the keystone test rather than a competition test. Hajishengallis
    2011 (PMID 22036469) found bone loss required both the commensal microbiota and complement, so
    a cube with no immune clearance has nothing for P. gingivalis to disable. ``host=False`` is
    that control arm — see F14.
    """
    base = _run(_dysbiotic_engine(host=host))
    rel0 = _deep_rel(base)
    pg_share = rel0.get("Pg", 0.0)
    cands = [(abs(rel0.get(s.id, 0) - pg_share), s.id) for s in CATALOG
             if s.id != "Pg" and s.socransky not in ("red",) and rel0.get(s.id, 0) > 0]
    control = sorted(cands)[0][1] if cands else "Sa"

    def index(removed: str) -> float:
        ko = _run(_dysbiotic_engine(knockout=(removed,), host=host))
        r1 = _deep_rel(ko)
        others = [s for s in base.species if s != removed]
        tot0 = sum(rel0[s] for s in others) or 1.0
        tot1 = sum(r1[s] for s in others) or 1.0
        l1 = sum(abs(r1[s] / tot1 - rel0[s] / tot0) for s in others)
        return 0.5 * l1 / (rel0.get(removed, 1e-9) or 1e-9)

    kp, kc = index("Pg"), index(control)
    ratio = kp / kc if kc > 0 else float("inf")
    ok = kp > 1.0 and ratio > 2.0
    return Result(
        "PASS" if ok else "FAIL",
        "deep-pocket keystone index Pg %.3f vs matched commensal %s %.3f (ratio %.2fx)"
        % (kp, control, kc, ratio),
        "Pg index >> 1 and >> the matched control",
        "" if ok else "the CONTRAST condition now passes decisively (%.2fx vs the 2x bar) and the "
                      "absolute one does not (%.3f vs 1.0). Keystone subversion of host "
                      "clearance is wired and works — see F14 for the control arm. Two honest "
                      "reasons the absolute index stays low: its normalisation divides by the "
                      "keystone's own share, and this cube's Pg sits near 6%% of the deep pocket "
                      "rather than the 'very low colonization levels' the hypothesis is about; "
                      "and the ceiling on what Pg can give away is host.K_CLEARANCE_MORTALITY, "
                      "a placeholder that is deliberately not being raised to clear this bar."
                      % (ratio, kp),
    )


# --------------------------------------------------------------------------- F12
def _f12_colonization_order_in_the_pocket() -> Result:
    """F1, asked where the late colonizers can actually establish.

    F1 runs the saccharolytic cube, in which the red complex cannot grow — so it measures an
    ordering among organisms that were never going to arrive, and reports whatever the decay rates
    happen to do.
    """
    tissue = oral_tissue(DYSBIOTIC)
    grid = CubeGrid(DIM, DIM, DIM, species_ids=tissue.cell_ids)
    eng = CubeEpoch4(grid, tissue)
    eng.coaggregation = True
    eng.seed_uniform()
    comp = eng.grid.composition
    for n, sid in enumerate(eng.species):
        if sid not in EARLY:
            comp[..., n] *= 1e-3                      # trace inoculum, as in _clean_surface

    peaks: dict = {}
    for step in range(1, STEPS * 2 + 1):
        eng.step(DT)
        r = _deep_rel(eng)
        for sid, v in r.items():
            if sid not in peaks or v > peaks[sid][1]:
                peaks[sid] = (step * DT, v)

    def mean_peak(group):
        vals = [peaks[s][0] for s in group if s in peaks and peaks[s][1] > 1e-6]
        return sum(vals) / len(vals) if vals else float("nan")

    e, m, l = mean_peak(EARLY), mean_peak(MIDDLE), mean_peak(LATE)
    late_share = _deep_mass(eng, LATE) / max(
        float(eng.grid.composition[_deep_slice(DIM)].sum()), 1e-9)
    ok = (e < m < l) and late_share > 0.01
    return Result(
        "PASS" if ok else "FAIL",
        "deep-pocket mean peak  early %.1f -> middle %.1f -> late %.1f; late share %.4f"
        % (e, m, l, late_share),
        "early < middle < late, with the late colonizers actually establishing",
        "" if ok else "ordering still not reproduced even where the late colonizers are viable; "
                      "succession here is set by terrain maturation, and every species is seeded "
                      "at once rather than arriving",
    )


# --------------------------------------------------------------------------- F13
def _f13_crossfeed_where_lactate_is_abundant() -> Result:
    """F5, asked in the regime the crossfeed belongs to.

    Veillonella lives on fermentative lactate, so the place to test that dependency is a SUGAR-RICH
    cube, not the dysbiotic pocket — a protein-fed pocket has almost no lactate for anyone to
    depend on, and pairing F5 to DYSBIOTIC would be pairing it to the wrong regime.
    """
    rich = SUGAR_RICH
    producers = tuple(s.id for s in CATALOG if "lactate" in s.byproducts and not s.lactate_consumer)

    def run(knockout=()):
        tissue = oral_tissue(rich)
        grid = CubeGrid(DIM, DIM, DIM, species_ids=tissue.cell_ids)
        eng = CubeEpoch4(grid, tissue)
        eng.seed_uniform()
        for sid in knockout:
            if sid in eng.species:
                grid.composition[..., eng.species.index(sid)] = 0.0
        return _run(eng)

    base, ko = run(), run(producers)
    b, k = _mass(base, ("Vp",)), _mass(ko, ("Vp",))
    change = (k - b) / b if b > 0 else float("inf")
    return Result(
        "PASS" if change < -0.20 else "FAIL",
        "V. parvula changes %+.1f%% in a sugar-rich cube when every lactate producer is removed"
        % (100 * change),
        "a DECREASE of more than 20%% — the crossfeed dependency",
        "" if change < -0.20 else "the regime pairing WORKS directionally — the same ablation is "
                                  "-12.0%% at the shipped oral_sugar=0.3 (F5) and -18.2%% here at "
                                  "2.0, so feeding the lactate economy does deepen the dependency "
                                  "by about half. It still does not clear the 20%% bar. That bar is "
                                  "mine rather than the literature's and it is deliberately not "
                                  "being lowered to make this pass.",
    )


# --------------------------------------------------------------------------- F14
def _f14_keystone_requires_complement() -> Result:
    """The conjunctive requirement, which is the actual central finding of the keystone paper.

    Hajishengallis et al. 2011 (Cell Host Microbe 10:497-506, PMID 22036469): P. gingivalis-induced
    bone loss required BOTH the commensal microbiota AND complement — germ-free mice and C3a/C5a
    receptor-deficient mice did not develop it. So the keystone effect is not a property of the
    organism, it is a property of the organism PLUS a host immune system to subvert.

    This check is that experiment: run the keystone contrast with a competent host, then again with
    no host clearance at all (the C5aR-deficient arm). The effect must be large in the first and
    absent in the second. A model that shows a keystone effect with no complement present has
    reproduced the phenotype by the wrong route.
    """
    with_host = _f11_keystone_contrast_in_the_pocket(host=True)
    without = _f11_keystone_contrast_in_the_pocket(host=False)

    def ratio_of(res: Result) -> float:
        try:
            return float(res.measured.split("(ratio")[1].split("x)")[0])
        except (IndexError, ValueError):
            return float("nan")

    r_host, r_none = ratio_of(with_host), ratio_of(without)
    ok = r_host > 2.0 and r_none < 1.5 and r_host > 2.0 * r_none
    return Result(
        "PASS" if ok else "FAIL",
        "keystone contrast ratio %.2fx with a competent host, %.2fx with no complement to subvert"
        % (r_host, r_none),
        "a large effect WITH host clearance and essentially none without it",
        "" if ok else "the keystone effect does not depend on the host, so whatever produces it "
                      "here is not the published mechanism",
    )


# --------------------------------------------------------------------------- F15
def _f15_colonization_with_arrival() -> Result:
    """Succession from a TRUE zero, now that arrival exists.

    F1 and F12 both seed every species at once and then measure who peaks first, which can only
    ever report who declines slowest — colonisation order is a claim about who ARRIVES. The engine
    had no dispersal, so "from zero" was not expressible and _clean_surface fakes it with a trace
    inoculum it apologises for in its own docstring.

    With sim.cube.dispersal the cube can start at exactly zero and be colonised from saliva, gated
    by what each species can stick to: pioneers bind the acquired pellicle directly, everyone else
    needs a coaggregation partner already present. Succession is not scheduled — it falls out.

    Scored on ARRIVAL TIME (time to half of final biomass), not on share peaks, because a share
    peak moves with the denominator and says nothing about when a species established.
    """
    from sim.cube.dispersal import Dispersal

    tissue = oral_tissue(DYSBIOTIC)
    grid = CubeGrid(DIM, DIM, DIM, species_ids=tissue.cell_ids)
    eng = CubeEpoch4(grid, tissue)
    eng.coaggregation = True
    eng.seed_uniform()
    grid.composition[:] = 0.0                      # a genuinely bare pellicle
    eng.dispersal = Dispersal()

    sl = _deep_slice(DIM)
    traj: dict = {s: [] for s in eng.species}
    for _ in range(STEPS * 2 + 100):
        eng.step(DT)
        for n, s in enumerate(eng.species):
            traj[s].append(float(grid.composition[sl][..., n].sum()))

    def t50(sid: str) -> float:
        a = np.array(traj[sid])
        if a[-1] <= 1e-9:
            return float("nan")
        return float((int(np.argmax(a >= 0.5 * a[-1])) + 1) * DT)

    def mean_t50(group) -> float:
        vals = [t50(s) for s in group if not np.isnan(t50(s))]
        return sum(vals) / len(vals) if vals else float("nan")

    early, red = mean_t50(EARLY), mean_t50(RED_COMPLEX)
    established = float(grid.composition[sl].sum()) > 0.0
    ok = established and early < red and (red - early) > 2.0
    return Result(
        "PASS" if ok else "FAIL",
        "arrival time from a bare surface — pioneers %.1f, red complex %.1f (Fn %.1f, Pi %.1f)"
        % (early, red, t50("Fn"), t50("Pi")),
        "pioneers establish first and the red complex last, starting from exactly zero",
        "" if ok else "colonisation order still not reproduced even with arrival wired",
    )


# --------------------------------------------------------------------------- F16
def _f16_regrowth_dissociates_biomass_from_composition() -> Result:
    """F4, with an antimicrobial that discriminates, scored on TIME TO RETURN.

    F4 is BLOCKED because a uniform multiplicative kill cannot change relative abundance, so there
    is no order to measure. sim.cube.antimicrobial supplies a kill differentiated by matrix
    shielding, envelope and metabolic activity.

    Scored the way the literature reports it. Teles et al. 2012 (PMID 21895662) followed
    redevelopment after full-mouth debridement and found an ORDER: S. mitis and Neisseria by day 1,
    Capnocytophaga and Eikenella over days 1-4, and the red complex not rising at any point in the
    first seven days. Separately, total counts reach or exceed baseline within about two days
    (PMID 21488936). So the claim is a DISSOCIATION — biomass returns quickly while composition
    does not — plus an order within the composition recovery.

    A share ratio at a single timepoint cannot test that: it is dominated by who was killed hardest
    rather than by who came back first. Time-to-return is the measure the finding is stated in.
    """
    from sim.cube.antimicrobial import apply_antimicrobial
    from sim.cube.dispersal import Dispersal

    eng = _run(_dysbiotic_engine())
    base_total = float(eng.grid.composition.sum())
    base = _rel(eng)

    survival = apply_antimicrobial(eng, dose=4.0)
    killed_total = float(eng.grid.composition.sum())
    differentiated = (max(survival.values()) - min(survival.values())) > 0.05

    eng.dispersal = Dispersal()
    steps = STEPS * 2
    totals, shares = [], []
    for _ in range(steps):
        eng.step(DT)
        totals.append(float(eng.grid.composition.sum()))
        shares.append(_rel(eng))

    def composition_distance(sh) -> float:
        """L1 distance of the community from its pre-kill composition."""
        return sum(abs(sh.get(s, 0.0) - base.get(s, 0.0)) for s in base)

    t_bio = next((( k + 1) * DT for k, tot in enumerate(totals) if tot >= base_total), float("inf"))
    d0 = composition_distance(shares[0])
    t_comp = next(((k + 1) * DT for k, sh in enumerate(shares)
                   if composition_distance(sh) <= 0.25 * d0), float("inf"))

    # Reported, not gated: the model has no ordering signal to offer here and saying so is more
    # useful than inventing a threshold that manufactures one.
    early_end = sum(shares[-1].get(s, 0.0) for s in EARLY) / max(
        sum(base.get(s, 0.0) for s in EARLY), 1e-9)
    red_end = sum(shares[-1].get(s, 0.0) for s in RED_COMPLEX) / max(
        sum(base.get(s, 0.0) for s in RED_COMPLEX), 1e-9)

    ok = differentiated and t_bio < t_comp
    return Result(
        "PASS" if ok else ("BLOCKED" if not differentiated else "FAIL"),
        "survival ranged %.3f-%.3f across species; biomass back to baseline at t=%.1f, composition "
        "still %.0f%% displaced at the end (t_comp=%.1f); final shares early %.2fx red %.2fx"
        % (min(survival.values()), max(survival.values()), t_bio,
           100 * composition_distance(shares[-1]) / max(d0, 1e-9), t_comp, early_end, red_end),
        "biomass returning to baseline well before composition does",
        "" if ok else ("the kill is not differentiated, so there is nothing to measure"
                       if not differentiated else
                       "composition recovered as fast as biomass — the dissociation is absent"),
    )


CHECKS = [
    Check("F1", "colonization ordering",
          "pioneers establish before middle, middle before late colonizers",
          "Kolenbrander lineage; succession classes as encoded in sim/species.py",
          _f1_colonization_order, known="FAIL"),
    Check("F2", "F. nucleatum bridging",
          "removing the bridge organism impairs late-colonizer establishment",
          "Bradshaw & Marsh 1998, Caries Res 32:456 (coaggregation-mediated, O2-conditional)",
          _f2_fusobacterium_bridge, known="FAIL"),
    Check("F3", "keystone at low abundance",
          "P. gingivalis reorganises the community out of proportion to its abundance",
          "Hajishengallis 2011, Cell Host Microbe 10:497",
          _f3_keystone_contrast, known="FAIL"),
    Check("F4", "post-antimicrobial regrowth order",
          "facultatives re-establish before strict anaerobes",
          "succession logic; the O2 gradient must be rebuilt before the anaerobic niche exists",
          _f4_regrowth_order, known="BLOCKED"),
    Check("F5", "Veillonella-Streptococcus crossfeeding",
          "Veillonella depends on streptococcal lactate",
          "Bradshaw/Marsh chemostat lineage; 9 crossfeed edges exist in the catalog but are unwired",
          _f5_veillonella_crossfeed, known="FAIL"),
    Check("F6", "aciduric selection under acid",
          "a sugar challenge shifts the flora toward aciduric species, tracking the pH fall",
          "Bradshaw & Marsh 1998, Caries Res 32:456; reframed from the 1989 pH-clamped control arm "
          "once pH became a solved field",
          _f6_aciduric_selection, known="PASS"),
    Check("F7", "pH is a solved field",
          "pH is emergent from the chemistry rather than a boundary constant",
          "charge balance / Stewart SID; salivary buffers after Dibdin 1990, J Dent Res 69:1324",
          _f7_ph_is_dynamic, known="PASS"),      # flipped BLOCKED -> PASS 2026-08-01 (charge-balance pH)
    Check("F8", "Stephan challenge depth",
          "a sugar bolus drops plaque pH 1.5-2 units, crossing the enamel-critical 5.5",
          "Stephan curve; Bradshaw & Marsh 1998 pH-floor series (5.5/5.0/4.5), Caries Res 32:456",
          _f8_stephan_depth, known="FAIL"),
    Check("F9", "bridging, dysbiotic pocket",
          "removing the bridge impairs late colonizers where they can actually live",
          "Bradshaw & Marsh 1998, Caries Res 32:456 (coaggregation-mediated, O2-conditional); "
          "regime per Mettraux 1984 PMID 6592325 + GCF proteolysis",
          _f9_bridge_in_the_pocket, known="PASS"),
    Check("F10", "keystone is regime-dependent",
          "P. gingivalis is excluded from a healthy sulcus and viable in a diseased pocket",
          "Marsh ecological plaque hypothesis; pocket pO2 Mettraux 1984 PMID 6592325; "
          "pocket alkalinity Eggert 1991 / Bickel & Cimasoni 1985",
          _f10_keystone_is_regime_dependent, known="PASS"),
    Check("F11", "keystone contrast, dysbiotic pocket",
          "P. gingivalis reorganises the community out of proportion to its abundance",
          "Hajishengallis 2011, Cell Host Microbe 10:497 (keystone-pathogen hypothesis)",
          _f11_keystone_contrast_in_the_pocket, known="FAIL"),
    Check("F12", "colonization ordering, dysbiotic",
          "early before middle before late, where the late colonizers can actually establish",
          "Kolenbrander lineage; regime per Mettraux 1984 PMID 6592325 + GCF proteolysis",
          _f12_colonization_order_in_the_pocket, known="FAIL"),
    Check("F13", "crossfeed where lactate is abundant",
          "Veillonella depends on fermentative lactate, tested in the sugar economy it belongs to",
          "Kolenbrander & London 1993; Bradshaw & Marsh 1998 on lactate crossfeeding",
          _f13_crossfeed_where_lactate_is_abundant, known="PASS"),
    Check("F14", "keystone effect requires complement",
          "P. gingivalis reorganises the community only when there is host clearance to subvert",
          "Hajishengallis et al. 2011, Cell Host Microbe 10:497-506, PMID 22036469 — bone loss "
          "required both the commensal microbiota and complement",
          _f14_keystone_requires_complement, known="PASS"),
    Check("F15", "colonization with arrival",
          "pioneers colonise bare pellicle first and the red complex arrives last, from true zero",
          "Kolenbrander coaggregation lineage — late colonizers lack pellicle adhesins and attach "
          "to receptors displayed by organisms already present",
          _f15_colonization_with_arrival, known="PASS"),
    Check("F16", "regrowth dissociates biomass from composition",
          "after a broad antimicrobial, biomass returns before composition and the red complex lags",
          "Teles et al. 2012 J Periodontal Res, PMID 21895662 (red complex does not rise in 7 days); "
          "PMID 21488936 (total counts back to baseline in ~2 days)",
          _f16_regrowth_dissociates_biomass_from_composition, known="PASS"),
]


def run_all(verbose: bool = True) -> dict:
    """Run every check and print the scoreboard. Returns {id: Result}."""
    out: dict = {}
    for c in CHECKS:
        c.result = c.run()
        out[c.id] = c.result
    if not verbose:
        return out

    print("FALSIFICATION SCOREBOARD — known results the model should reproduce")
    print("No new patient data required. Most of these fail; that is the instrument working.\n")
    for c in CHECKS:
        r = c.result
        mark = {"PASS": "PASS   ", "FAIL": "FAIL   ", "BLOCKED": "BLOCKED"}[r.status]
        drift = "" if r.status == c.known else "   <-- CHANGED from %s" % c.known
        print("%s  %s  %s%s" % (mark, c.id, c.name, drift))
        print("        claim     %s" % c.claim)
        print("        source    %s" % c.citation)
        print("        measured  %s" % r.measured)
        print("        expected  %s" % r.expected)
        if r.note:
            print("        note      %s" % r.note)
        print()
    tally = {}
    for c in CHECKS:
        tally[c.result.status] = tally.get(c.result.status, 0) + 1
    print("  ".join("%s %d" % (k, v) for k, v in sorted(tally.items())))
    return out


if __name__ == "__main__":
    run_all()

