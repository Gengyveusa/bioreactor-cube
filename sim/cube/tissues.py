"""Tissue plugins for the fused cube engine (Epoch-4) — what makes a cube 'oral' or 'artery'.

The engine (sim/cube/fused.py) is universal: it runs the generic operators (gate T, express+grow E,
recognize I, step Δ) over whatever cell roster and terrain a CubeTissue supplies. A tissue provides:
  genomes  — per cell-type Information seed: a niche (gaussian viability windows) + demands (monotone
             requirements) + growth/defense + tissue traits. This is the gate's content.            [T]
  prepare  — set up the initial terrain (incl. any extra fields the tissue needs).                   [T]
  update_terrain — how Energy reshapes Terrain each tick (consumption, diffusion, resolution).    [E→T]
  react    — tissue-specific Information transitions (e.g. monocyte → foam cell).                     [E]
  readout  — the tissue-specific ψ (system state).                                                   [ψ]
  efflux / receive — the body-map ports: what leaves into the bloodstream / what arrives from it.  [L7]

Both oral and artery are instances behind this one interface — that is the 'place anywhere' promise.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace, field

import numpy as np

from sim.cube.ecm import (ARTERY_ECM, BIOFILM_ECM, BRAIN_ECM, ECMSpec, LIVER_ECM, _diff3, apply_ecm, porosity)
from sim.cube.chemistry import (MM_PER_UNIT, SALIVA_SID_MM, SERUM_BICARB_MM, SERUM_PH,
                                sid_field, solve_ph)
from sim.cube.faces import FaceConditions, apply_face_boundaries
from sim.cube.sample import BLEEDING_POCKET
from sim.species import CATALOG

HEALTHY = FaceConditions()   # default mouth: no bleeding (serum_heme=0), no sugar challenge


@dataclass
class CubeTissue:
    name: str
    cell_ids: list[str]
    genomes: dict[str, dict]
    abundance: dict[str, float]
    prepare: Callable             # (eng) -> None
    update_terrain: Callable      # (eng, dt) -> None      [E→T]
    readout: Callable             # (eng) -> dict          [ψ]
    react: Callable = lambda eng, dt: None        # [E] tissue transitions
    efflux: Callable = lambda eng: 0.0            # [L7 out] systemic load into bloodstream
    receive: Callable = lambda eng, influx: None  # [L7 in] arrival from bloodstream
    ecm: ECMSpec | None = None                    # the tissue's characterised extracellular matrix


# ===================================================================== ORAL BIOFILM (instance #1)
# P. gingivalis was seeded at 0.08 — 8.3% of the community — and that is an order of magnitude too
# high. The keystone-pathogen hypothesis is definitionally about LOW abundance: Hajishengallis et al.
# 2011 (Cell Host Microbe 10:497-506, PMID 22036469) demonstrated the effect with Pg "4 to 5 log
# units lower than the total bacterial counts (<0.01% of the total)", restated in the 2012 review as
# "very low colonization levels (<0.01% of the total bacterial count)" (Nat Rev Microbiol 10:717-25,
# PMID 22941505). Those are MURINE figures and must not be reused as human ones; the same paper
# gives ~0.8% of total clones in active human periodontitis and <=0.03% in another human study.
#
# 0.008 puts Pg at ~0.9% of the seeded community — the human periodontitis figure, which is the
# right target for a cube representing a diseased pocket. This matters beyond realism: the keystone
# index in F3/F11 DIVIDES by the organism's own share, so seeding Pg ten times too high was
# penalising the model by a factor of ten for a number the catalog chose.
_ORAL_ABUND = {
    "Sa": 0.12, "Sm": 0.10, "An": 0.06, "Hp": 0.04, "Rm": 0.07, "Vp": 0.05, "Fn": 0.12, "Pi": 0.04,
    "Ec": 0.03, "Pg": 0.008, "Td": 0.06, "Tf": 0.06, "Aa": 0.02, "Smu": 0.03, "Lb": 0.02, "Fa": 0.02,
    "Pm": 0.02, "Ca": 0.02,
}


# A gaussian niche needs a width, and the width has to MEAN something. The original
# max(1.0, (hi-lo)/2) floored every width at 1.0, which quietly deleted the difference between a
# strict anaerobe and a tolerant one: P. gingivalis at max 0.2% atm and T. denticola at 0.1% got
# the identical curve, and widening either envelope changed nothing at all.
#
# Definition used instead: the envelope's MAX is the concentration at which viability has fallen to
# EDGE_VIABILITY. exp(-((hi-opt)/w)^2) = EDGE_VIABILITY solves to w = (hi-opt)/sqrt(-ln(EDGE)).
# Now the catalog's envelope numbers are load-bearing, which is the only reason to have them.
EDGE_VIABILITY = 0.10
_EDGE_K = float(np.sqrt(-np.log(EDGE_VIABILITY)))     # ~1.517

# A RESOLUTION floor, not a biological claim, and the distinction matters.
#
# The cube spans a whole pocket in 10-32 voxels, so a voxel is 200-600 um while the O2 field runs
# from ~8 to 0 across it — a step of roughly 0.5-0.8 %atm per voxel. A niche band narrower than
# that cannot be resolved: refining the mesh keeps finding more of it, and the readouts never
# converge. Removing the old floor entirely made exactly that happen (rupture_index and oral_efflux
# were still falling monotonically at dim 24, variation 0.36 and 0.58).
#
# So the floor stays, at the coarsest resolvable band rather than at an arbitrary 1.0. The honest
# consequence: species whose true envelope is narrower than this — V. parvula, F. alocis, P. micra,
# all at (0, 0.5) — are modelled as MORE oxygen-tolerant than they are. Fixing that needs a finer
# lattice over a shallower slab, not a smaller number here. [[VERIFY against a resolved run]]
NICHE_WIDTH_FLOOR = 1.0


def _niche_width(lo: float, hi: float, opt: float, floor: float = NICHE_WIDTH_FLOOR) -> float:
    """Gaussian width such that viability is :data:`EDGE_VIABILITY` at the envelope edge.

    Example: _niche_width(0, 2.0, 0) -> ~1.32, so P. gingivalis is still ~17% viable at the
    13.3 mmHg (1.75% atm) mean pO2 Mettraux measured at the base of untreated pockets.
    """
    span = max(abs(hi - opt), abs(opt - lo))
    return max(floor, span / _EDGE_K)


def _oral_genome(s) -> dict:
    return {
        # O2 width is envelope-derived; pH width stays the shipped constant 1.0. Deriving the pH
        # width too was scope creep and it cost the mesh-independence guarantee: pH spans about one
        # unit across the cube, so an envelope-derived width of ~0.5 is again sub-voxel, and
        # rupture_index and oral_efflux stopped converging (0.33 and 0.53 variation). O2 is what was
        # blocking the keystone; pH was not, so pH is left alone.
        "niche": {"O2": (s.envelope.o2[2], _niche_width(*s.envelope.o2[:2], s.envelope.o2[2])),
                  "pH": (s.envelope.ph[2], 1.0)},
        # Demands are MONOTONE requirements on the terrain. Heme for the red complex; lactate for
        # the crossfeeders, which is what makes Veillonella actually DEPEND on streptococcal output
        # rather than merely consume it. Without this the crossfeed edge is bookkeeping.
        #
        # The lactate demand was 0.80, leaving Veillonella 20% of its expression with no lactate at
        # all. The literature says the dependence is far closer to obligate: growth in a
        # reconstructed plaque community is "strictly conditional on the presence of a partner"
        # (Chalmers et al., PMID 18805978), lactate is an obligate energy source in defined medium
        # (Durant et al., PMID 16887614), and it is described as the sole carbon source (PMID
        # 31235964). 0.95 leaves a 5% floor rather than 20% — not zero, because V. parvula does have
        # alternative substrates: it ferments malate (PMID 1645132) and consumes Actinomyces-derived
        # succinate (PMID 6935027), neither of which this model represents.
        "demand": dict(({"heme": 0.65} if s.envelope.heme >= 1 else {}),
                       **({"lactate_sat": 0.95} if s.lactate_consumer else {})),
        "growth": 0.16 if s.archetype == "Cariogenic" else 0.12,
        "defense": 0.7 if s.archetype == "Keystone" else (0.5 if s.role == "pathobiont" else 0.35),
        "traits": {"no_yield": max(0.0, s.NO_yield), "ves": s.vesiculation_rate,
                   "red": 1.0 if s.socransky == "red" else 0.0},
    }


_ORAL_GENOMES = {s.id: _oral_genome(s) for s in CATALOG}


def oral_prepare_terrain(grid, iters: int = 15, cond=BLEEDING_POCKET) -> None:
    """Initial oral terrain: heme spreads from the gum, nitrate from the mouth; steep O2 gradient kept.
    The anoxic deep zone is NOT painted — it emerges from aerobic O2 consumption (E reshaping T)."""
    apply_face_boundaries(grid, cond)
    for _ in range(iters):
        for f in ("heme", "nitrate"):
            grid.get_field(f)[:] = _diff3(grid.get_field(f))
        apply_face_boundaries(grid, cond)


@dataclass(frozen=True)
class SugarChallenge:
    """A timed sugar bolus — the stimulus a Stephan curve is the response to.

    A STEADY sugar boundary cannot produce a Stephan curve no matter how high you set it, because
    the curve is a transient: sugar arrives, is fermented to acid faster than saliva clears it, pH
    falls to a nadir, and then recovers as the sugar runs out. Modelling the arrival as a decaying
    bolus is what makes the nadir and the recovery limb exist at all.

    Example: SugarChallenge(t_start=5.0).at(5.0) -> baseline + amplitude
    """

    baseline: float = 0.05          # inter-meal residual
    amplitude: float = 3.0          # height of the bolus above baseline
    t_start: float = 0.0            # when the challenge is delivered
    half_life: float = 2.0          # oral sugar clearance half-life, model time units [[VERIFY]]

    def at(self, t: float) -> float:
        """Sugar concentration at the mouth-facing boundary at model time t."""
        if t < self.t_start:
            return self.baseline
        return self.baseline + self.amplitude * 0.5 ** ((t - self.t_start) / self.half_life)


# ---- fermentation constants. Uncalibrated placeholders that set how hard a sugar load bites. [[VERIFY]]
SUGAR_UPTAKE = 2.2          # max specific sugar uptake by fermenters
KS_SUGAR = 0.25             # Monod half-saturation for sugar
LACTATE_UPTAKE = 1.4        # max specific lactate uptake by lactate consumers (Veillonella)
KS_LACTATE = 0.20
KS_LACTATE_GATE = 0.010    # half-saturation for lactate AVAILABILITY as seen by the gate [[VERIFY]]
ACID_YIELD = 0.9            # mol acid per mol sugar consumed (homofermentative ~2, heterofermentative less)
ACID_PH_OPTIMUM = 6.0       # aggregate pH optimum for glycolysis across the fermenter pool [[VERIFY]]

# Who does what, read straight off the species catalog rather than hard-coded.
# ---------------------------------------------------------------- proteolysis constants
# The pocket runs two nutrient economies. The plaque surface is SACCHAROLYTIC — dietary sugar to
# lactate, pH down. The deep pocket is PROTEOLYTIC — host protein from gingival crevicular fluid to
# ammonia and short-chain fatty acids, pH UP. The red complex belongs to the second one, which is
# why an acidic cube could never grow it: Pg's pH optimum is 7.8 and saccharolytic plaque sits near
# 6.5. Modelling only the sugar economy made the keystone's own niche unreachable.
PEPTIDE_UPTAKE = 1.1         # max specific peptide uptake by proteolytic species
KS_PEPTIDE = 0.22            # Monod half-saturation for peptide
NH3_YIELD = 0.75             # NH3 released per unit peptide fermented (deamination is near-total)
SCFA_YIELD = 0.45            # butyrate/propionate/acetate per unit peptide
H2S_YIELD = 0.05             # sulfur amino acids are a small fraction of protein

#: Dimensionless proteolytic field -> millimolar. ONE conversion for every product of peptide
#: fermentation, and that is the load-bearing part.
#:
#: The first version of this used MM_PER_UNIT (227, calibrated for LACTATE from a sugar bolus) for
#: the SCFAs while ammonia had its own smaller constant. Same reaction, two scales, six-fold apart —
#: which made protein fermentation come out net ACIDIC. That is backwards: proteolysis is the
#: alkalinising half of the pocket's metabolism and is why a deep lesion is not acidic.
#:
#: Deamination releases roughly one NH3 per amino acid while the carbon skeleton becomes roughly
#: one short-chain fatty acid, so the yields must be compared on ONE molar scale. At pocket pH the
#: SCFAs (pKa ~4.8) are fully dissociated anions and the ammonia is fully NH4+ cation, so the net
#: charge per unit peptide is simply (NH3_YIELD - SCFA_YIELD) — positive, hence alkaline.
#: The absolute scale is calibrated to land severe pockets near the measured band. [[VERIFY]]
PEPTIDE_MM_PER_UNIT = 60.0

# ---------------------------------------------------------------- gingipain heme liberation
# The keystone mechanism, and the reason a 4%-abundance organism can be load-bearing.
#
# Heme is an essential growth factor for the red complex, and the pocket's supply is not free heme —
# it is HEMOPROTEIN in gingival crevicular fluid and erythrocytes (Smalley & Olczak 2017, Mol Oral
# Microbiol 32:1-23, PMID 26662717). P. gingivalis acquires heme "mainly from haemoglobin via the
# sequential actions of the R- and K-specific gingipain proteases" (Smalley et al. 2011, PLoS One
# 6:e17182, PMID 21390208), and that extraction happens OUTSIDE the cell, on immobilised
# hemoglobin. So the heme a gingipain frees is a PUBLIC GOOD: every heme-requiring organism in the
# neighbourhood can take it, whether or not it could have freed any itself.
#
# That is what makes removal of the keystone disproportionate. Pg is not merely a competitor for
# heme, it is the SUPPLIER, and Td/Tf/Pi/Hp/Ec all carry a heme demand they cannot satisfy alone.
#
# Weights are ordinal, not measured. Pg's gingipains are the characterised route; Prevotella
# intermedia's interpain A is named in the same review as oxidising hemoglobin and so gets a real
# but smaller share; T. denticola's dentilisin and T. forsythia's karilysin/mirolysin are different
# proteases with far less hemoglobin activity. [[VERIFY the weights]]
HEME_LIBERATORS = {"Pg": 1.0, "Pi": 0.35, "Td": 0.15, "Tf": 0.15}
HEME_LIBERATION = 0.9        # heme freed per unit protease activity per unit hemoprotein [[VERIFY]]
HEME_SUBSTRATE_COST = 0.15   # fraction of the peptide pool consumed in freeing it

# ---------------------------------------------------------------- alkali generation
# THE MISSING HALF OF THE STEPHAN CURVE, and not the half I expected.
#
# F8's failure note blamed transport: acid escaping only by diffusion to one clearing face. The
# literature says that is wrong. McNee & Geddes (1982, PMID 6818937) measured lactate diffusion in
# human plaque at 4.8e-6 cm^2/s — only 2-5x slower than free water — giving a clearance half-time
# of about SIX SECONDS through a 100 um layer, and concluded the prolonged low-pH phase is NOT
# diffusion-limited. Acid leaves plaque fast.
#
# What produces the recovery limb is BASE. Acidogens alone make only the fall; the rise needs
# alkali-generating organisms (Wijeyeweera & Kleinberg 1989 PMID 2675801; Kleinberg 2002 PMID
# 12097354). A pure-acidogen biofilm sits permanently acidic and does not self-neutralise until an
# alkali generator is added (Bradshaw 1998 PMID 9602287). The model had NO base source at all, so a
# monotonic fall was the only outcome available to it.
#
# Two routes, both real and both present in this catalog:
#   UREOLYSIS   urea + H2O -> 2 NH3 + CO2. Two ammonia per urea, so the more potent route per mole.
#               Km 2.12 mmol/L, Vmax 0.11 umol/min/mg dry wt (Sissons 1998, PMID 9438574).
#   ADS         arginine -> citrulline + NH3, in the sanguinis/mitis streptococci. Roughly half
#               urea's pH response per mole (Van Wuyckhuyse 1995, PMID 8067919).
UREA_KM = 2.12 / 60.0          # Michaelis constant on the proteolytic millimolar scale
UREOLYSIS_VMAX = 2.2           # specific rate — the one fitted constant here [[VERIFY]]
NH3_PER_UREA = 2.0             # stoichiometric; why urea beats arginine per mole
ADS_VMAX = 0.9                 # arginine deiminase, ~half urea's potency per mole [[VERIFY]]

#: Oral urease producers. A. naeslundii is the classic plaque urease organism.
UREOLYTIC = {"An": 1.0, "Hp": 0.4}

#: Arginine-deiminase-positive: the sanguinis/mitis group. This is why a plaque dominated by
#: S. mutans (no ADS) acidifies and stays acidic while a sanguinis-rich plaque recovers.
ARGINOLYTIC = {"Sa": 1.0, "Sm": 1.0, "An": 0.3}

_FERMENTERS = tuple(s.id for s in CATALOG if s.primary_substrate == "sugar")
_PROTEOLYTIC = tuple(s.id for s in CATALOG if s.primary_substrate in ("peptide", "amino_acid"))
_NH3_PRODUCERS = tuple(s.id for s in CATALOG if "NH3" in s.byproducts)
_H2S_PRODUCERS = tuple(s.id for s in CATALOG if "H2S" in s.byproducts)
_BUTYRATE_PRODUCERS = tuple(s.id for s in CATALOG if "butyrate" in s.byproducts)
_PROPIONATE_PRODUCERS = tuple(s.id for s in CATALOG if "propionate" in s.byproducts)
_LACTATE_PRODUCERS = tuple(s.id for s in CATALOG if "lactate" in s.byproducts)
_ACETATE_PRODUCERS = tuple(s.id for s in CATALOG if "acetate" in s.byproducts)
_LACTATE_CONSUMERS = tuple(s.id for s in CATALOG if s.lactate_consumer)


def _oral_update_terrain(eng, dt: float, cond=BLEEDING_POCKET, challenge=None) -> None:
    comp = eng.grid.composition
    o2 = eng.grid.get_field("O2")
    act = comp * eng.expression                        # per-species ACTIVE biomass
    active = act.sum(axis=3)
    o2 -= dt * 1.4 * active * o2                       # aerobic O2 consumption → anoxic deep emerges
    np.clip(o2, 0.0, None, out=o2)

    # ---------------------------------------------------------------- fermentation
    # Sugar -> organic acids -> pH. Before this existed, sugar and lactate were inert fields that
    # nothing produced or consumed, so a 10x dietary sugar change moved the community by exactly
    # zero. Which is the correct answer for a pH-clamped system (Bradshaw 1989) and the wrong one
    # for a mouth.
    def _biomass(ids):
        out = np.zeros(o2.shape)
        for sid in ids:
            if sid in eng.species:
                out = out + act[..., eng.species.index(sid)]
        return out

    sugar = eng.grid.get_field("sugar")
    lactate = eng.grid.get_field("lactate")
    acetate = eng.grid.get_field("acetate")

    # Glycolysis is throttled by the pH it creates. Head, Devine & Marsh 2017 (Sci Rep 7:17413,
    # PMID 29234121) give the rate as
    #     r = m . mu_max . [Gl]/([Gl]+K_Gl) . 4[H+]K_acid/([H+]+K_acid)^2
    # The last factor is a bell peaking at exactly 1 when [H+] = K_acid, and it is what makes the
    # nadir a NADIR rather than a monotonic slide: as pH falls past the community optimum, acid
    # production shuts itself down. K_acid here is an AGGREGATE over the fermenter pool (Head splits
    # it into aciduric pH 5 and non-aciduric pH 7); per-species optima would be better. [[VERIFY]]
    h_ion = 10.0 ** (-eng.grid.get_field("pH"))
    k_acid = 10.0 ** (-ACID_PH_OPTIMUM)
    ph_brake = 4.0 * h_ion * k_acid / (h_ion + k_acid) ** 2

    uptake = dt * SUGAR_UPTAKE * _biomass(_FERMENTERS) * sugar / (sugar + KS_SUGAR) * ph_brake
    uptake = np.minimum(uptake, sugar)                 # cannot eat more than is there
    sugar -= uptake

    # Split the acid yield between lactate and acetate by who is actually present and active.
    lp, ap = _biomass(_LACTATE_PRODUCERS), _biomass(_ACETATE_PRODUCERS)
    share = lp / np.maximum(lp + ap, 1e-12)
    lactate += ACID_YIELD * uptake * share
    acetate += ACID_YIELD * uptake * (1.0 - share)

    # Crossfeeding: Veillonella lives on streptococcal lactate and converts it to weaker acids.
    # This is the term that makes removing the streptococci actually cost Veillonella something.
    l_up = dt * LACTATE_UPTAKE * _biomass(_LACTATE_CONSUMERS) * lactate / (lactate + KS_LACTATE)
    l_up = np.minimum(l_up, lactate)
    lactate -= l_up
    acetate += 0.5 * l_up                              # lactate -> propionate/acetate, less acidic

    # ---------------------------------------------------------------- proteolysis
    # The OTHER economy. GCF protein -> deamination -> NH3 (alkaline) + short-chain fatty acids.
    # Unlike glycolysis this has no pH brake pushing it down: proteolysis raises pH toward its own
    # optimum, so it is self-REINFORCING where glycolysis is self-limiting. That asymmetry is the
    # thermodynamic reason a pocket, once proteolytic, tends to stay proteolytic.
    peptide = eng.grid.get_field("peptide")
    nh3 = eng.grid.get_field("NH3")
    butyrate = eng.grid.get_field("butyrate")
    propionate = eng.grid.get_field("propionate")
    h2s = eng.grid.get_field("H2S")

    p_up = dt * PEPTIDE_UPTAKE * _biomass(_PROTEOLYTIC) * peptide / (peptide + KS_PEPTIDE)
    p_up = np.minimum(p_up, peptide)
    peptide -= p_up

    # Ammonia is credited only to the species that actually deaminate; E. corrodens ferments
    # peptide without an NH3 byproduct in the catalog and must not silently alkalinise the pocket.
    prot = _biomass(_PROTEOLYTIC)
    nh3_share = _biomass(_NH3_PRODUCERS) / np.maximum(prot, 1e-12)
    nh3 += NH3_YIELD * p_up * nh3_share
    h2s += H2S_YIELD * p_up * (_biomass(_H2S_PRODUCERS) / np.maximum(prot, 1e-12))

    bp, pp = _biomass(_BUTYRATE_PRODUCERS), _biomass(_PROPIONATE_PRODUCERS)
    b_share = bp / np.maximum(bp + pp, 1e-12)
    butyrate += SCFA_YIELD * p_up * b_share
    propionate += SCFA_YIELD * p_up * (1.0 - b_share)

    # --- alkali generation: the recovery limb.
    # Ureolysis and ADS are the only base sources in plaque, and without them a Stephan curve can
    # only fall. Both are Monod-saturating in their substrate and proportional to the ACTIVE
    # biomass of the organisms that carry the enzyme — which is why a sanguinis-rich plaque
    # recovers and a mutans-dominated one stays acidic.
    urea = eng.grid.get_field("urea")
    ureolytic = np.zeros(o2.shape)
    for sid, wgt in UREOLYTIC.items():
        if sid in eng.species:
            ureolytic = ureolytic + wgt * act[..., eng.species.index(sid)]
    u_up = dt * UREOLYSIS_VMAX * ureolytic * urea / (urea + UREA_KM)
    u_up = np.minimum(u_up, urea)
    urea -= u_up
    nh3 += NH3_PER_UREA * u_up

    # ADS runs on salivary/GCF arginine, which is part of the same peptide pool the proteolytic
    # economy draws on, so it competes for that substrate rather than having its own field.
    arginolytic = np.zeros(o2.shape)
    for sid, wgt in ARGINOLYTIC.items():
        if sid in eng.species:
            arginolytic = arginolytic + wgt * act[..., eng.species.index(sid)]
    a_up = dt * ADS_VMAX * arginolytic * peptide / (peptide + KS_PEPTIDE)
    a_up = np.minimum(a_up, 0.25 * peptide)     # arginine is a minority of the pool
    peptide -= a_up
    nh3 += a_up

    # --- gingipain heme liberation: the keystone's public good.
    # Heme is freed from GCF hemoprotein by bacterial proteases, weighted by who actually has the
    # hemoglobin-degrading machinery. The freed heme enters the shared field, so the whole
    # heme-requiring guild feeds on what the keystone unlocks.
    liberase = np.zeros(o2.shape)
    for sid, w in HEME_LIBERATORS.items():
        if sid in eng.species:
            liberase = liberase + w * act[..., eng.species.index(sid)]
    hemoprotein = eng.grid.get_field("hemoprotein")
    freed = np.minimum(dt * HEME_LIBERATION * liberase * hemoprotein, hemoprotein)
    heme = eng.grid.get_field("heme")
    heme += freed
    hemoprotein -= freed

    np.clip(sugar, 0.0, None, out=sugar)
    np.clip(lactate, 0.0, None, out=lactate)
    np.clip(acetate, 0.0, None, out=acetate)
    np.clip(peptide, 0.0, None, out=peptide)
    np.clip(nh3, 0.0, None, out=nh3)
    np.clip(heme, 0.0, None, out=heme)
    np.clip(hemoprotein, 0.0, None, out=hemoprotein)

    # Saturating lactate AVAILABILITY for the fitness gate. The gate reads a field against a [0,1]
    # clip, so a raw solute that lives at ~0.02 can never express a real dependency — it would move
    # the gate by under 2% between "abundant" and "absent". Monod-saturating it first is both the
    # physically standard form and the only way the crossfeed dependency can actually bind.
    eng.extra["lactate_sat"] = lactate / (lactate + KS_LACTATE_GATE)

    # the extracellular space: bacteria SECRETE the EPS matrix; it GATES transport and stores heme
    # (the characterised oral ECM — see ecm.BIOFILM_ECM). Sugar and the acids ride the same gate.
    apply_ecm(eng, dt, ("O2", "heme", "nitrate", "sugar", "lactate", "acetate",
                        "peptide", "hemoprotein", "urea", "NH3", "butyrate", "propionate"),
              BIOFILM_ECM)
    # A timed bolus overrides the steady sugar boundary; dt enables finite (not infinite) clearance.
    if challenge is not None:
        cond = replace(cond, oral_sugar=challenge.at(eng.t))
    apply_face_boundaries(eng.grid, cond, dt=dt)

    # ---------------------------------------------------------------- pH by charge balance
    ph = eng.grid.get_field("pH")
    # Buffers ride in on the boundary conditions, so a patient's measured salivary buffering lands
    # here. The default BufferSpec is the shipped literature mix, so an intake with no buffer test
    # solves exactly as it did before this seam existed.
    # THE POCKET IS NOT ONE FLUID. The mouth-facing surface is saliva — weakly buffered, resting
    # near 6.8. The tissue-facing depth is gingival crevicular fluid, which is a serum transudate:
    # buffered five times harder and resting at 7.4. Treating the whole cube as saliva was what
    # made the deep pocket acidic and the keystone's own niche unreachable.
    #
    # So the buffer mix is a FIELD, blended by how serum-like each voxel is, and the SID is
    # re-solved per voxel against that voxel's resting pH — otherwise adding serum bicarbonate
    # without its counter-cation would acidify the depth instead of alkalinising it.
    b = cond.buffers
    gcf = np.clip(peptide / (peptide + KS_PEPTIDE), 0.0, 1.0)
    bicarb = b.bicarb_mM + gcf * (SERUM_BICARB_MM - b.bicarb_mM)
    rest = b.resting_pH + gcf * (SERUM_PH - b.resting_pH)
    sid = sid_field(rest, phosphate_mM=b.phosphate_mM, bicarb_mM=bicarb, fixed_mM=b.fixed_mM)

    # Both economies now meet in one charge balance: the saccharolytic acids pull pH down, the
    # proteolytic ammonia pushes it up, and the host's own buffer decides the baseline each is
    # working against. Which wins is SPATIAL, and that is the point.
    # Two economies, two scales, and they must not be crossed: lactate/acetate are the sugar
    # economy at MM_PER_UNIT; butyrate/propionate/ammonia are the protein economy at
    # PEPTIDE_MM_PER_UNIT. Mixing them is what made proteolysis acidify the pocket.
    ph[:] = solve_ph(
        {"lactate": lactate * MM_PER_UNIT, "acetate": acetate * MM_PER_UNIT,
         "butyrate": butyrate * PEPTIDE_MM_PER_UNIT,
         "propionate": propionate * PEPTIDE_MM_PER_UNIT},
        sid, phosphate_mM=b.phosphate_mM, bicarb_mM=bicarb, fixed_mM=b.fixed_mM,
        ammonia_mM=nh3 * PEPTIDE_MM_PER_UNIT, shape=ph.shape)
    # Saliva and serum are externally buffered reservoirs, so the two open faces keep their imposed
    # pH; everywhere else pH is now emergent from what the community actually made.
    ph[:, :, -1] = cond.oral_pH
    ph[-1, :, :] = cond.serum_pH


def _oral_omv_field(eng) -> np.ndarray:
    """Per-voxel OMV shedding. Hemin INDUCES vesiculation (P. gingivalis sheds far more OMV in a
    bleeding, heme-rich pocket) — so shedding is heme-gated, not just biomass-proportional."""
    heme = np.clip(eng.grid.get_field("heme"), 0.0, 1.0)
    boost = 0.2 + 0.8 * heme
    comp = eng.grid.composition
    base = sum(comp[..., n] * _ORAL_GENOMES[c]["traits"]["ves"] * eng.expression[..., n]
               for n, c in enumerate(eng.species))
    return base * boost


def _oral_readout(eng) -> dict:
    comp = eng.grid.composition
    total = comp.sum(axis=3)
    red = sum(comp[..., n] for n, c in enumerate(eng.species) if _ORAL_GENOMES[c]["traits"]["red"] > 0)
    no = sum(comp[..., n] * _ORAL_GENOMES[c]["traits"]["no_yield"] * eng.expression[..., n]
             for n, c in enumerate(eng.species))
    omv = _oral_omv_field(eng)
    m = eng.extra.get("ecm", np.full(total.shape, BIOFILM_ECM.base_density))
    theta = porosity(m, BIOFILM_ECM.alpha)
    bound = eng.extra.get("ecm_bound_heme", np.zeros(total.shape))
    return {
        "dysbiosis": round(float(red.sum() / max(total.sum(), 1e-9)), 3),
        "NO": round(float(np.sum(no)), 3),
        "omv_flux": round(float(np.sum(omv)), 3),
        "biomass": round(float(total.mean()), 4),               # density (intensive)
        "ecm_density": round(float(m.mean()), 3),              # the extracellular matrix (EPS)
        "ecm_porosity": round(float(theta.mean()), 3),         # how open the wire is to transport
        "ecm_heme_bound": round(float(bound.mean()), 4),       # heme stored in the matrix (density)
        # 6 dp, not 4. Correcting P. gingivalis from 8.3% to its measured ~0.9% took gingipain density
    # to ~4.5e-5, which rounds to a flat 0.0 at four places — a readout that displays ABSENT for a
    # quantity that is merely small. The keystone's whole claim is that it matters at low abundance,
    # so a precision that hides it at low abundance defeats the readout's purpose.
    "gingipain": round(float(eng.extra.get("enzyme", np.zeros(1)).mean()), 6),  # red-complex protease (density)
        "fields": {"dysbiosis": red / np.maximum(total, 1e-9), "NO": no, "omv": omv,
                   "eps": m, "porosity": theta},
    }


def _oral_efflux(eng) -> float:
    """Systemic OMV load entering circulation: heme-induced OMV weighted toward the vascularized
    SOFT-TISSUE (gum) face — the OMV port of Layer-7 — normalised per unit gum area (intensive)."""
    nx, ny, nz = eng.grid.shape
    gum_weight = np.linspace(0.0, 1.0, nx)[:, None, None]      # 1 at the soft-tissue face
    # MEAN, not sum: this is a flux DENSITY. A sum over voxels scales with lattice size, which
    # would make every downstream systemic effect a function of resolution rather than of biology.
    return float((_oral_omv_field(eng) * gum_weight).mean())


def _oral_prepare(eng, cond) -> None:
    """Relax the terrain, then publish the derived fields the fitness gate reads.

    lactate_sat must exist before the first expressibility evaluation, because the gate runs ahead
    of the terrain update on step 1.
    """
    oral_prepare_terrain(eng.grid, cond=cond)
    lac = eng.grid.get_field("lactate")
    eng.extra["lactate_sat"] = lac / (lac + KS_LACTATE_GATE)


def oral_tissue(cond=BLEEDING_POCKET, challenge=None) -> CubeTissue:
    """Oral biofilm tissue. cond=BLEEDING_POCKET → dysbiotic/bleeding; cond=HEALTHY → healthy mouth."""
    return CubeTissue(
        name="oral biofilm" + ("" if cond is BLEEDING_POCKET else " (healthy)"),
        cell_ids=[s.id for s in CATALOG],
        genomes=_ORAL_GENOMES,
        abundance=_ORAL_ABUND,
        prepare=lambda eng: _oral_prepare(eng, cond),
        update_terrain=lambda eng, dt: _oral_update_terrain(eng, dt, cond=cond, challenge=challenge),
        readout=_oral_readout,
        efflux=_oral_efflux,
        ecm=BIOFILM_ECM,
    )


ORAL_TISSUE = oral_tissue(BLEEDING_POCKET)        # default: dysbiotic, bleeding pocket
ORAL_HEALTHY = oral_tissue(HEALTHY)               # control: healthy mouth


# ===================================================================== ARTERY / ATHEROMA (instance #2)
# Geometry reuse: z = nz-1 (the 'oral' face) is the BLOOD LUMEN; just below = intima (plaque forms
# here); deep = media/adventitia (smooth muscle). Cells: endothelium, recruited monocytes, foam
# cells (lipid-laden macrophages), smooth-muscle cells. Extra terrain fields: oxLDL, shear, inflammation.
_ARTERY_ABUND = {"EC": 0.30, "Mono": 0.05, "Foam": 0.0, "SMC": 0.20}

# Presentation scaling for the artery rupture INDEX. No derivation, no units, no calibration to any
# human event rate. It exists only so the index spans a readable range. Changing it changes every
# absolute value and no relative comparison, which is precisely the point. [[VERIFY]]
RUPTURE_SCALE = 1.0e5

_ARTERY_GENOMES = {
    # endothelium — lines the lumen (high O2); its expression IS endothelial activation/dysfunction,
    # driven by the inflammatory load arriving from the bloodstream (the ψ readout).
    "EC":   {"niche": {"O2": (7.0, 2.5)}, "demand": {"inflammation": 0.85},
             "growth": 0.04, "defense": 0.8, "traits": {}},
    # monocyte/macrophage — RECRUITED by inflammation across the lumen→intima; weakly retained.
    "Mono": {"niche": {"O2": (3.0, 3.5)}, "demand": {"inflammation": 0.9},
             "growth": 0.20, "defense": 0.4, "traits": {}},
    # foam cell — forms from monocytes that engulf oxLDL (see react); persists → the lipid core.
    "Foam": {"niche": {"O2": (1.0, 2.0)}, "demand": {"oxLDL": 0.8},
             "growth": 0.05, "defense": 0.85, "traits": {}},
    # smooth muscle — in the media; proliferates with inflammation → neointima / fibrous cap.
    "SMC":  {"niche": {"O2": (2.0, 3.0)}, "demand": {"inflammation": 0.5},
             "growth": 0.10, "defense": 0.6, "traits": {}},
}


def _artery_prepare(eng) -> None:
    g = eng.grid
    nz = g.shape[2]
    zf = np.arange(nz)[None, None, :] / (nz - 1)               # 0 deep (media) … 1 lumen
    g.get_field("O2")[:] = 8.0 * zf + 0.3                      # O2 high at lumen, low in media
    eng.extra["oxLDL"] = np.exp(-((zf - 0.8) / 0.15) ** 2) * np.ones(g.shape)   # subendothelial intima
    eng.extra["oxLDL0"] = eng.extra["oxLDL"].copy()           # the supplied lipid pool (chronic)
    eng.extra["shear"] = (zf * np.ones(g.shape)).copy()       # protective shear, high at the lumen
    eng.extra["inflammation"] = np.zeros(g.shape)             # raised only by the bloodstream port


def _artery_update_terrain(eng, dt: float) -> None:
    comp = eng.grid.composition
    o2 = eng.grid.get_field("O2")
    active = (comp * eng.expression).sum(axis=3)
    o2 -= dt * 1.0 * active * o2
    np.clip(o2, 0.0, None, out=o2)
    o2[:] = _diff3(o2)
    o2[:, :, -1] = 8.0                                          # lumen O2 resupply
    eng.extra["oxLDL"] += dt * 0.1 * (eng.extra["oxLDL0"] - eng.extra["oxLDL"])  # chronic lipid supply
    eng.extra["inflammation"] *= (1.0 - dt * 0.06)             # inflammation resolves slowly
    # the dense fibrous vascular ECM (collagen, λ=2.2) gates transport — low D_eff TRAPS oxLDL in the
    # intima (proteoglycan retention) and slows inflammation penetration. SMCs build the cap.
    apply_ecm(eng, dt, ("inflammation", "oxLDL"), ARTERY_ECM)


def _artery_react(eng, dt: float) -> None:
    """[E] monocyte → foam-cell differentiation where oxLDL is high (information/phenotype transition)."""
    comp = eng.grid.composition
    mi, fi = eng.species.index("Mono"), eng.species.index("Foam")
    flux = dt * 1.2 * comp[..., mi] * np.clip(eng.field("oxLDL"), 0, 1) * eng.expression[..., mi]
    comp[..., mi] -= flux
    comp[..., fi] += flux
    np.clip(comp, 0.0, None, out=comp)


def _artery_receive(eng, influx: float) -> None:
    """[L7 in] systemic OMV / inflammatory load arrives at the lumen and inflames the endothelium.

    A SURFACE FLUX, so it must deposit a fixed amount per unit area per unit time regardless of how
    the lattice is cut. Writing ``+= influx`` into the boundary layer alone does not: that layer is
    one voxel thick, so as nz rises the layer thins and the amount actually delivered falls like
    1/nz. Scaling by nz keeps the delivered dose resolution-independent.
    """
    nz = eng.grid.shape[2]
    eng.extra["inflammation"][:, :, -1] += influx * nz


def _artery_readout(eng) -> dict:
    comp = eng.grid.composition
    total = comp.sum(axis=3)
    foam = comp[..., eng.species.index("Foam")]
    smc = comp[..., eng.species.index("SMC")]
    ec = comp[..., eng.species.index("EC")]
    ec_expr = eng.expression[..., eng.species.index("EC")]
    plaque = foam + smc
    m = eng.extra.get("ecm", np.full(total.shape, ARTERY_ECM.base_density))
    enz = eng.extra.get("enzyme", np.zeros(total.shape))
    cap = float(m.mean())
    mmp = float(enz.mean())                                     # macrophage MMP-2/9 activity DENSITY
    foam_frac = float(foam.sum() / max(total.sum(), 1e-9))       # a ratio: already intensive
    # A DIMENSIONLESS COMPOSITE INDEX, not a probability and not a hazard. All three factors are
    # now intensive, so the index describes the tissue rather than the mesh. RUPTURE_SCALE is a
    # presentation constant with NO derivation — it sets the range, nothing more. Only RELATIVE
    # comparisons (diseased vs healthy, before vs after) carry meaning; the absolute value does
    # not, and it is not calibrated against any human event rate. [[VERIFY]]
    rupture_index = round(min(1.0, RUPTURE_SCALE * mmp * foam_frac / max(cap, 0.05)), 4)
    return {
        "plaque_density": round(float(plaque.mean()), 4),
        "foam_fraction": round(foam_frac, 3),
        "endothelial_activation": round(float((ec_expr * ec).sum() / max(ec.sum(), 1e-9)), 3),
        "inflammation_load": round(float(eng.extra["inflammation"].mean()), 4),
        "ecm_density": round(cap, 3),                          # fibrous cap (collagen) — SMC build, MMP thin
        "ecm_porosity": round(float(porosity(m, ARTERY_ECM.alpha).mean()), 3),
        "mmp_activity": round(mmp, 4),                         # macrophage MMP-2/9 (density)
        "rupture_index": rupture_index,                        # composite index, NOT a probability
        "biomass": round(float(total.mean()), 4),
        "fields": {"plaque": plaque, "inflammation": eng.extra["inflammation"]},
    }


ARTERY_TISSUE = CubeTissue(
    name="artery / atheroma",
    cell_ids=["EC", "Mono", "Foam", "SMC"],
    genomes=_ARTERY_GENOMES,
    abundance=_ARTERY_ABUND,
    prepare=_artery_prepare,
    update_terrain=_artery_update_terrain,
    readout=_artery_readout,
    react=_artery_react,
    receive=_artery_receive,
    ecm=ARTERY_ECM,
)


# ===================================================================== BRAIN / NEUROVASCULAR UNIT (instance #3)
# The other oral→systemic destination (periodontitis ↔ Alzheimer's: P. gingivalis gingipains, BBB
# disruption, neuroinflammation, impaired glymphatic clearance → amyloid). Geometry: z = nz-1 (the
# 'oral' face) is the BLOOD CAPILLARY / BBB (where the systemic load arrives); deep = the parenchyma.
# Cells: brain endothelium (BBB), pericytes, astrocytes (AQP4 → glymphatic flow), microglia, neurons.
# Extra fields: inflammation (arriving), amyloid (accumulates when clearance fails), glymph (clearance).
_BRAIN_ABUND = {"BEC": 0.18, "Peri": 0.12, "Astro": 0.22, "Micro": 0.10, "Neuron": 0.30}

_BRAIN_GENOMES = {
    # brain endothelium — the BBB lining at the capillary; activated by the arriving inflammatory load.
    "BEC":    {"niche": {"O2": (7.0, 2.5)}, "demand": {"inflammation": 0.6},
               "growth": 0.04, "defense": 0.85, "traits": {}},
    # pericyte — wraps the capillary, supports the barrier.
    "Peri":   {"niche": {"O2": (4.5, 3.0)}, "growth": 0.06, "defense": 0.7, "traits": {}},
    # astrocyte — end-feet on the vessel; AQP4 drives glymphatic clearance (computed in update_terrain).
    "Astro":  {"niche": {"O2": (3.5, 3.0)}, "growth": 0.08, "defense": 0.65, "traits": {}},
    # microglia — resident immune; ACTIVATED by inflammation → neuroinflammation.
    "Micro":  {"niche": {"O2": (2.5, 3.5)}, "demand": {"inflammation": 0.9},
               "growth": 0.14, "defense": 0.45, "traits": {}},
    # neuron — the parenchyma; viability falls as amyloid accumulates (the amyloid niche gate).
    "Neuron": {"niche": {"O2": (3.0, 2.5), "amyloid": (0.0, 0.6)},
               "growth": 0.05, "defense": 0.7, "traits": {}},
}


def _brain_prepare(eng) -> None:
    g = eng.grid
    nz = g.shape[2]
    zf = np.arange(nz)[None, None, :] / (nz - 1)               # 0 parenchyma … 1 capillary/BBB
    g.get_field("O2")[:] = 8.0 * zf + 0.3
    eng.extra["inflammation"] = np.zeros(g.shape)
    eng.extra["amyloid"] = np.full(g.shape, 0.1)               # low baseline amyloid
    eng.extra["glymph"] = np.zeros(g.shape)


def _brain_update_terrain(eng, dt: float) -> None:
    comp = eng.grid.composition
    o2 = eng.grid.get_field("O2")
    active = (comp * eng.expression).sum(axis=3)
    o2 -= dt * 1.0 * active * o2
    np.clip(o2, 0.0, None, out=o2)
    o2[:] = _diff3(o2)
    o2[:, :, -1] = 8.0                                          # capillary O2 resupply
    eng.extra["inflammation"] *= (1.0 - dt * 0.06)
    # glymphatic clearance = astrocyte AQP4 activity, STRONGLY suppressed by inflammation (the mechanism:
    # neuroinflammation impairs perivascular CSF flow). Healthy → high clearance; inflamed → near zero.
    astro = comp[..., eng.species.index("Astro")]
    supp = np.clip(1.0 - 4.0 * eng.extra["inflammation"], 0.03, 1.0)
    glymph = 0.6 * supp * np.clip(0.3 + 2.0 * astro, 0.0, 1.5)
    eng.extra["glymph"] = glymph
    # amyloid: produced at baseline, cleared ∝ glymphatic flow → accumulates when clearance fails
    am = eng.extra["amyloid"]
    am += dt * (0.05 - glymph * am)
    np.clip(am, 0.0, None, out=am)
    # the soft hyaluronan/lectican brain ECM (λ≈1.6, astrocyte-built) gates inflammation + amyloid transport
    apply_ecm(eng, dt, ("inflammation", "amyloid"), BRAIN_ECM)


def _brain_receive(eng, influx: float) -> None:
    """[L7 in] systemic OMV / inflammatory load arrives at the capillary and disrupts the BBB."""
    eng.extra["inflammation"][:, :, -1] += influx


def _brain_readout(eng) -> dict:
    comp = eng.grid.composition
    total = comp.sum(axis=3)
    infl = eng.extra["inflammation"]
    glymph = eng.extra.get("glymph", np.zeros_like(total))
    amy = eng.extra["amyloid"]
    mi = eng.species.index("Micro")
    micro_act = float((comp[..., mi] * eng.expression[..., mi]).mean())
    enz = eng.extra.get("enzyme", np.zeros(total.shape))
    mmp = float(enz.mean())
    bbb = 1.0 / (1.0 + 8.0 * float(infl.mean()) + 25.0 * mmp)   # inflammation + MMP-9 degrade the BBB
    m = eng.extra.get("ecm", np.full(total.shape, BRAIN_ECM.base_density))
    return {
        "bbb_integrity": round(bbb, 3),
        "glymphatic_clearance": round(float(glymph.mean()), 4),
        "amyloid_load": round(float(amy.mean()), 4),
        "neuroinflammation": round(micro_act, 3),
        "inflammation_load": round(float(infl.mean()), 4),
        "ecm_density": round(float(m.mean()), 3),              # hyaluronan/lectican gel — astrocyte-built
        "ecm_porosity": round(float(porosity(m, BRAIN_ECM.alpha).mean()), 3),
        "mmp_activity": round(float(enz.mean()), 4),           # microglial MMP-9 / ADAMTS (density)
        "biomass": round(float(total.mean()), 4),
        "fields": {"amyloid": amy, "glymph": glymph, "inflammation": infl},
    }


BRAIN_TISSUE = CubeTissue(
    name="brain / neurovascular unit",
    cell_ids=["BEC", "Peri", "Astro", "Micro", "Neuron"],
    genomes=_BRAIN_GENOMES,
    abundance=_BRAIN_ABUND,
    prepare=_brain_prepare,
    update_terrain=_brain_update_terrain,
    readout=_brain_readout,
    receive=_brain_receive,
    ecm=BRAIN_ECM,
)


# ===================================================================== LIVER / HEPATIC LOBULE (instance #4)
# The first-pass filter for gut/portal blood — so translocating oral/gut bacteria & OMVs hit it directly
# (periodontitis ↔ NAFLD/NASH → fibrosis). Geometry: z = nz-1 = the SINUSOID blood inlet (load arrives);
# deep = the pericentral, low-O2 zone. Cells: hepatocytes, Kupffer macrophages, sinusoidal endothelium
# (LSEC), and stellate cells (which build the fibrotic ECM). Extra fields: inflammation, lipid (steatosis).
_LIVER_ABUND = {"Hep": 0.40, "Kup": 0.12, "LSEC": 0.18, "Stel": 0.12}

_LIVER_GENOMES = {
    # sinusoidal endothelium — lines the inlet (high O2); activated by the arriving load.
    "LSEC": {"niche": {"O2": (7.0, 2.5)}, "demand": {"inflammation": 0.6},
             "growth": 0.04, "defense": 0.8, "traits": {}},
    # Kupffer macrophage — ACTIVATED by inflammation → hepatic inflammation (and MMP secretion).
    "Kup":  {"niche": {"O2": (4.0, 3.0)}, "demand": {"inflammation": 0.9},
             "growth": 0.14, "defense": 0.4, "traits": {}},
    # hepatocyte — the parenchymal workhorse; viability falls as LIPID (steatosis) accumulates (lipotoxicity).
    "Hep":  {"niche": {"O2": (4.0, 4.0), "lipid": (0.0, 0.8)},
             "growth": 0.05, "defense": 0.7, "traits": {}},
    # stellate cell — sits perisinusoidal in the space of Disse (where the load arrives); ACTIVATED by
    # inflammation → deposits the fibrotic collagen (the liver ECM); persists.
    "Stel": {"niche": {"O2": (5.0, 3.0)}, "demand": {"inflammation": 0.85},
             "growth": 0.16, "defense": 0.85, "traits": {}},
}


def _liver_prepare(eng) -> None:
    g = eng.grid
    nz = g.shape[2]
    zf = np.arange(nz)[None, None, :] / (nz - 1)               # 0 pericentral … 1 sinusoid (periportal)
    g.get_field("O2")[:] = 8.0 * zf + 0.3                      # the portal→central O2 zonation
    eng.extra["inflammation"] = np.zeros(g.shape)
    eng.extra["lipid"] = np.full(g.shape, 0.1)                 # low baseline fat


def _liver_update_terrain(eng, dt: float) -> None:
    comp = eng.grid.composition
    o2 = eng.grid.get_field("O2")
    active = (comp * eng.expression).sum(axis=3)
    o2 -= dt * 1.0 * active * o2
    np.clip(o2, 0.0, None, out=o2)
    o2[:] = _diff3(o2)
    o2[:, :, -1] = 8.0                                          # sinusoid O2 resupply
    eng.extra["inflammation"] *= (1.0 - dt * 0.06)
    # steatosis: lipid accumulates with inflammation / metabolic stress, cleared by baseline turnover
    lip = eng.extra["lipid"]
    lip += dt * (0.02 + 0.12 * np.clip(eng.extra["inflammation"], 0, 1) - 0.08 * lip)
    np.clip(lip, 0.0, None, out=lip)
    eng.extra["lipid"][:] = _diff3(lip)
    # the space-of-Disse ECM — gates inflammation transport
    apply_ecm(eng, dt, ("inflammation",), LIVER_ECM)
    # fibrotic deposition: perisinusoidal stellate cells, ACTIVATED by the local inflammatory signal,
    # lay down collagen; TIMP-restrained slow regression. This is the fibrosis that accumulates with the load.
    m = eng.extra["ecm"]
    m += dt * (0.04 * np.clip(eng.extra["inflammation"], 0, 1) - 0.012 * (m - LIVER_ECM.base_density))
    np.clip(m, 0.0, None, out=m)


def _liver_receive(eng, influx: float) -> None:
    """[L7 in] portal/systemic load arrives at the sinusoid and inflames the lobule."""
    eng.extra["inflammation"][:, :, -1] += influx


def _liver_readout(eng) -> dict:
    comp = eng.grid.composition
    total = comp.sum(axis=3)
    hi, ki, si = eng.species.index("Hep"), eng.species.index("Kup"), eng.species.index("Stel")
    hep, hep_expr = comp[..., hi], eng.expression[..., hi]
    stel = comp[..., si]
    infl = eng.extra["inflammation"]
    lipid = eng.extra["lipid"]
    m = eng.extra.get("ecm", np.full(total.shape, LIVER_ECM.base_density))
    hep_fn = float((hep * hep_expr).sum() / max(hep.sum(), 1e-9))        # mean hepatocyte activity (detox)
    fibrosis = float((m * stel).sum() / max(stel.sum(), 1e-9))          # collagen density in stellate territory
    return {
        "hepatocyte_function": round(hep_fn, 3),
        "steatosis": round(float((lipid * hep).sum() / max(hep.sum(), 1e-9)), 3),
        "fibrosis": round(fibrosis, 3),                                 # perisinusoidal stellate-built collagen
        "hepatic_inflammation": round(float((comp[..., ki] * eng.expression[..., ki]).mean()), 4),
        "inflammation_load": round(float(infl.mean()), 4),
        "ecm_density": round(float(m.mean()), 3),
        "ecm_porosity": round(float(porosity(m, LIVER_ECM.alpha).mean()), 3),
        "biomass": round(float(total.mean()), 4),
        "fields": {"lipid": lipid, "inflammation": infl, "fibrosis": m},
    }


LIVER_TISSUE = CubeTissue(
    name="liver / hepatic lobule",
    cell_ids=["Hep", "Kup", "LSEC", "Stel"],
    genomes=_LIVER_GENOMES,
    abundance=_LIVER_ABUND,
    prepare=_liver_prepare,
    update_terrain=_liver_update_terrain,
    readout=_liver_readout,
    receive=_liver_receive,
    ecm=LIVER_ECM,
)

