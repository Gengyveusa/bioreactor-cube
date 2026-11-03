"""Layer-6 host immunity, wired into the CUBE — the host as a combatant, not a scoreboard.

``sim/host.py`` designed three host couplings and has had zero callers since it was written; it
also targets ``SimConfig``, which is the 1-D simulator, not the cube. This module is the cube's
version, and it exists because the host is a *player* in the pocket, not a readout of it.

THE THREE AXES (same decomposition as sim/host.py, applied to cube surfaces):

1. ``inflammation``          systemic tone (hs-CRP / IL-6 / TNF-alpha)
                             -> raises serum heme at the soft-tissue face.
                             Inflamed gingiva bleeds; heme is what feeds the Pg bloom.

2. ``tissue_breakdown``      active collagen destruction (aMMP-8) PLUS the attachment already lost
                             (CAL - PD = recession)
                             -> deepens the anoxic niche by lowering serum-face O2 delivery.

3. ``antimicrobial_capacity`` neutrophil-mediated clearance, read from HbA1c
                             -> per-species mortality, SUBVERTED by keystone gingipains.

WHY THIS IS NOT CIRCULAR. Feeding a marker in as an input and then scoring the model against that
same marker would be assuming what you claim to predict. The guard is structural rather than a
comment: :class:`HostState` records which markers it consumed, and :meth:`validation_targets`
returns only the ones it did not. A marker can be an input or an outcome. Never both.

The distinction that makes the inputs legitimate: these three axes are measurements of HOST STATE
(how inflamed, how broken down, how competent the neutrophils are), while the cube's systemic
outputs are inflammatory efflux and arterial consequence. Host susceptibility in, disease
trajectory out. What would be circular is inputting serum CRP and reporting inflammatory load.

ALL GAINS ARE PLACEHOLDERS pending cohort calibration, exactly as in sim/host.py. The marker ->
axis anchors below are real clinical cutpoints; the interpolation between them is a calibration
choice. [[VERIFY the curves before any clinical reading]]
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------- coupling gains
K_INFLAMMATION_HEME = 0.45   # serum-face heme added per unit inflammation  [[VERIFY]]
K_TISSUE_O2 = 0.60           # serum-face O2 removed per unit tissue_breakdown  [[VERIFY]]
K_CLEARANCE_MORTALITY = 0.30 # extra mortality multiplier per unit capacity (matches sim/host.py)
KEYSTONE_EVASION = 0.20      # fraction of clearance the red complex still receives

#: Red-complex keystones. Gingipains degrade complement and immunoglobulin and blunt neutrophil
#: killing, so these three evade most of whatever clearance the host can mount. This is the
#: mechanism that makes a competent immune system insufficient on its own.
KEYSTONE_IDS = ("Pg", "Td", "Tf")


# --------------------------------------------------------------------------- keystone subversion
# THE mechanism, and the one a linear public good could never be.
#
# Hajishengallis et al. 2011 (Cell Host Microbe 10:497-506, PMID 22036469): P. gingivalis "at very
# low colonization levels, triggers changes to the amount and composition of the oral commensal
# microbiota leading to inflammatory periodontal bone loss", and critically "the commensal
# microbiota AND complement were both required" — germ-free mice and C3a/C5a-receptor-deficient
# mice did not develop bone loss.
#
# Two things follow, and both are structural rather than numeric:
#
# 1. IT IS A SWITCH, NOT A SUPPLY. The effect is saturating in Pg's abundance, so "very low
#    colonization levels" already deliver most of it. A term LINEAR in a 4%-abundance organism is a
#    4% effect by construction, which is exactly why heme liberation could not produce a keystone
#    at any rate constant (see tissues.HEME_LIBERATION and F11's note).
#
# 2. IT ACTS ON THE HOST, NOT THE COMMUNITY DIRECTLY. Pg does not feed or poison its neighbours; it
#    disables the clearance that was holding ALL of them down. So the beneficiary is the whole
#    local community, and removing Pg restores clearance for everyone — a disproportionate,
#    community-wide effect from a low-abundance organism.
#
# The conjunctive requirement is also a free falsification control: with no host clearance to
# subvert, the keystone effect must VANISH. That is the C5aR-deficient arm, and it is testable.
KEYSTONE_SUBVERTERS = {"Pg": 1.0}

#: Active biomass at which subversion is half-maximal. SMALL on purpose — that is what makes this a
#: switch rather than a dose-response, and it is the whole content of "low-abundance". [[VERIFY]]
SUBVERSION_HALF = 3.0e-4

#: Ceiling on how much host clearance a fully-subverted neighbourhood loses. Not 1.0: complement is
#: degraded, not abolished, and neutrophils still arrive. [[VERIFY]]
SUBVERSION_MAX = 0.85


def subversion_field(eng) -> np.ndarray:
    """0..1 per-voxel: how far local complement-mediated clearance has been subverted.

    Saturating in the subverter's ACTIVE biomass, so a small keystone population already produces
    most of the effect. Returns zeros when no subverter is present.

    Example: subversion_field(eng) is ~0.5 where Pg's active biomass is SUBVERSION_HALF
    """
    act = eng.grid.composition * eng.expression
    s = np.zeros(eng.grid.shape)
    for sid, w in KEYSTONE_SUBVERTERS.items():
        if sid in eng.species:
            s = s + w * act[..., eng.species.index(sid)]
    return s / (s + SUBVERSION_HALF)


def clearance_scale(eng) -> np.ndarray:
    """Multiplier on host clearance after keystone subversion — 1.0 where the host is unimpaired."""
    return 1.0 - SUBVERSION_MAX * subversion_field(eng)


# --------------------------------------------------------------------------- marker -> axis
# Anchors are (measured value, normalized 0-1). The ENDPOINTS are clinical cutpoints in real use;
# the straight lines between them are the calibration choice.
#
#   hs-CRP   AHA/CDC cardiovascular risk strata: <1 low, 1-3 average, >3 high. Above ~10 mg/L the
#            reading reflects acute infection rather than chronic tone and should be re-drawn, so
#            the curve saturates there rather than continuing to climb.
#   IL-6     healthy adults typically sit under ~2 pg/mL.
#   TNF-a    healthy adults roughly 0.5-8 pg/mL; assay-dependent, hence the wide top anchor.
#   aMMP-8   10 ng/mL is the established chairside threshold for ACTIVE collagen degradation
#            (PerioSafe / ORALyzer); 20+ is strongly positive.
#   HbA1c    <5.7 normal, 5.7-6.4 prediabetes, >=6.5 diabetes. This one is INVERTED: it measures
#            IMPAIRMENT of neutrophil function, so a higher value means less clearance.
_CRP_ANCHORS = ((0.5, 0.0), (1.0, 0.15), (3.0, 0.55), (10.0, 1.0))
_IL6_ANCHORS = ((1.0, 0.0), (3.5, 0.5), (10.0, 1.0))
_TNF_ANCHORS = ((2.0, 0.0), (8.0, 0.5), (20.0, 1.0))
_AMMP8_ANCHORS = ((5.0, 0.0), (10.0, 0.4), (20.0, 0.7), (50.0, 1.0))
_HBA1C_IMPAIRMENT = ((5.4, 0.0), (5.7, 0.1), (6.5, 0.4), (8.0, 0.7), (11.0, 1.0))

#: Recession (CAL - PD) in mm mapped to a historical tissue-loss floor. 5 mm of lost attachment is
#: severe by any staging system, so the curve saturates there. [[VERIFY]]
_RECESSION_ANCHORS = ((0.0, 0.0), (2.0, 0.35), (5.0, 1.0))


def _interp(value: float, anchors) -> float:
    xs = [a for a, _ in anchors]
    ys = [b for _, b in anchors]
    return float(np.interp(float(value), xs, ys))


def _val(marker) -> Optional[float]:
    """Unwrap an intake ``Value`` (or a bare number, or None) without importing the intake package."""
    if marker is None:
        return None
    v = getattr(marker, "value", marker)
    return None if v is None else float(v)


@dataclass(frozen=True)
class HostState:
    """The three normalized host axes, plus a record of which markers were spent to build them.

    Example: HostState.from_markers(host, recession_mm=2.0).tissue_breakdown > 0.0  -> True
    """

    inflammation: float = 0.0
    tissue_breakdown: float = 0.0
    antimicrobial_capacity: float = 0.0
    #: Markers consumed as INPUT. Anything here is disqualified as a validation target.
    consumed: frozenset = field(default_factory=frozenset)
    #: True when nothing was measured, so callers can assert "changes nothing".
    empty: bool = True

    # Every marker this module knows how to read. validation_targets() is the complement.
    KNOWN = ("hs_crp", "il6", "tnf_alpha", "ammp8", "hba1c")

    @classmethod
    def from_markers(cls, markers, *, recession_mm: float = 0.0) -> "HostState":
        """Build the axes from an intake :class:`~sim.intake.schema.HostMarkers` plus recession.

        Unmeasured markers contribute nothing, so an intake with no labs yields an all-zero state
        and the cube runs exactly as it did before this module existed.
        """
        consumed: set = set()
        infl_terms = []
        for name, anchors in (("hs_crp", _CRP_ANCHORS), ("il6", _IL6_ANCHORS),
                              ("tnf_alpha", _TNF_ANCHORS)):
            v = _val(getattr(markers, name, None)) if markers is not None else None
            if v is not None:
                infl_terms.append(_interp(v, anchors))
                consumed.add(name)
        # The three systemic markers are correlated readouts of ONE underlying tone, so they are
        # averaged rather than summed. Summing would let a full panel triple the same signal.
        inflammation = float(np.mean(infl_terms)) if infl_terms else 0.0

        breakdown_terms = []
        ammp8 = _val(getattr(markers, "ammp8", None)) if markers is not None else None
        if ammp8 is not None:
            breakdown_terms.append(_interp(ammp8, _AMMP8_ANCHORS))
            consumed.add("ammp8")
        if recession_mm > 0:
            breakdown_terms.append(_interp(recession_mm, _RECESSION_ANCHORS))
            consumed.add("clinical_attachment_mm")
        # aMMP-8 is the CURRENT rate of destruction; recession is what has ALREADY been destroyed.
        # The site's breakdown state is the worse of the two, not their sum: a quiet mouth with old
        # recession is not actively breaking down, and an active mouth with no recession yet is.
        tissue_breakdown = max(breakdown_terms) if breakdown_terms else 0.0

        capacity = 0.0
        hba1c = _val(getattr(markers, "hba1c", None)) if markers is not None else None
        if hba1c is not None:
            # Measuring HbA1c reveals a host defence the cube otherwise models as entirely absent.
            # A competent host therefore looks BETTER once measured, which is correct: the
            # unmeasured default is not "healthy host", it is "no host clearance modelled at all".
            capacity = 1.0 - _interp(hba1c, _HBA1C_IMPAIRMENT)
            consumed.add("hba1c")

        return cls(
            inflammation=inflammation,
            tissue_breakdown=tissue_breakdown,
            antimicrobial_capacity=capacity,
            consumed=frozenset(consumed),
            empty=not consumed,
        )

    def validation_targets(self) -> tuple:
        """Markers NOT spent as inputs, and therefore still honest to score the model against.

        Example: HostState.from_markers(m).validation_targets() excludes anything it consumed
        """
        return tuple(k for k in self.KNOWN if k not in self.consumed)

    def apply_to_conditions(self, cond):
        """Couplings 1 and 2: inflammation raises serum heme, breakdown deepens the anoxic niche.

        Returns a new :class:`~sim.cube.faces.FaceConditions`; the input is not mutated.
        """
        if self.empty:
            return cond
        heme = float(np.clip(cond.serum_heme + K_INFLAMMATION_HEME * self.inflammation, 0.0, 1.0))
        o2 = float(max(0.0, cond.serum_O2 - K_TISSUE_O2 * self.tissue_breakdown))
        return replace(cond, serum_heme=heme, serum_O2=o2)

    def mortality_multipliers(self, species_ids) -> dict:
        """Coupling 3: per-species clearance mortality, with keystone gingipain evasion.

        Empty when there is no antimicrobial signal, which keeps an unmeasured patient on exactly
        the engine's stock predation term.
        """
        cap = self.antimicrobial_capacity
        if cap <= 0.0:
            return {}
        full = 1.0 + K_CLEARANCE_MORTALITY * cap
        evaded = 1.0 + K_CLEARANCE_MORTALITY * cap * KEYSTONE_EVASION
        return {sid: (evaded if sid in KEYSTONE_IDS else full) for sid in species_ids}

    def summary(self) -> str:
        """One line for a UI banner.

        Example: "host: inflammation 0.55, breakdown 0.40, clearance 0.90 (from hs_crp, hba1c)"
        """
        if self.empty:
            return "host: not measured — no host coupling applied"
        return ("host: inflammation %.2f, breakdown %.2f, clearance %.2f (from %s)"
                % (self.inflammation, self.tissue_breakdown, self.antimicrobial_capacity,
                   ", ".join(sorted(self.consumed))))


def attach_host(eng, state: HostState) -> None:
    """Bind a host state's clearance term to a running engine.

    Example: attach_host(eng, HostState.from_markers(intake.host))
    """
    mult = state.mortality_multipliers(eng.species)
    eng.mortality_mult = mult or None

