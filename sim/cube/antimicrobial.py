"""A broad antimicrobial challenge — chlorhexidine, or mechanical debridement.

F4 has been BLOCKED since the harness was written, and its own note says why: "a UNIFORM
multiplicative kill leaves relative abundance exactly unchanged, so there is no order to measure."
Multiplying every species by 0.01 is not an antimicrobial, it is a change of units.

Real killing is differentiated, and the differentiators are known:

  MATRIX SHIELDING dominates. Chlorhexidine is a cation that binds the anionic EPS gel and the
  cell surfaces it meets first, so it is consumed on the way in. Depth inside the matrix is the
  strongest single predictor of survival, which is why biofilm tolerance is orders of magnitude
  above planktonic MIC for the same organisms.

  THE ENVELOPE MATTERS. Gram-positives take up more of a cationic bisbiguanide than
  gram-negatives, and Actinomyces is reported as the genus most suppressed by chlorhexidine among
  the early colonisers (Bernardi et al. 2004, PMID 15257736).

  BEING DORMANT PROTECTS. Agents that need an active target spare the un-expressing fraction. The
  engine already carries expression per voxel per species, so this needs no new state.

WHAT THE MODEL IS BEING ASKED TO REPRODUCE. Teles et al. 2012 (J Periodontal Res, PMID 21895662)
followed redevelopment after full-mouth debridement: supragingival plaque returns in a defined
order, S. mitis and Neisseria mucosa by day 1, Capnocytophaga and Eikenella over days 1-4, and the
red complex NOT rising at any point in the first 7 days. Separately, total counts reach or exceed
baseline within about 2 days (PMID 21488936) — so BIOMASS recovers long before COMPOSITION does.
That dissociation is the sharp, falsifiable claim, and it is what F4 should be testing.
"""

from __future__ import annotations

import numpy as np

from sim.cube.ecm import BIOFILM_ECM, porosity
from sim.species import BY_ID

#: How much of the kill is removed by full matrix shielding. Not 1.0 — a rinse does reach the
#: surface layers — but large, because biofilm tolerance really is orders of magnitude. [[VERIFY]]
MATRIX_PROTECTION = 0.85

#: Extra susceptibility of gram-positives to a cationic bisbiguanide, as a multiplier on the kill.
#: Ordinal, from the observation that Actinomyces is the most suppressed early coloniser. [[VERIFY]]
GRAM_SUSCEPTIBILITY = {"+": 1.35, "-": 1.0, "fungal": 0.7}

#: Fraction of the kill that requires an actively-expressing target. The dormant remainder is the
#: persister-like fraction every biofilm antimicrobial leaves behind.
ACTIVITY_DEPENDENCE = 0.6


def survival_fraction(eng, sid: str, dose: float) -> np.ndarray:
    """Per-voxel fraction of this species that SURVIVES a dose. 1.0 means untouched.

    Example: survival_fraction(eng, "Sa", 1.0) is lower near the oral face than deep in the matrix
    """
    m = eng.extra.get("ecm")
    if m is None:
        m = np.full(eng.grid.shape, BIOFILM_ECM.base_density)
    # Porosity falls as the matrix thickens, so this is high where the gel is dense.
    shielding = MATRIX_PROTECTION * (1.0 - porosity(m, BIOFILM_ECM.alpha))

    gram = BY_ID[sid].gram if sid in BY_ID else "-"
    envelope = GRAM_SUSCEPTIBILITY.get(gram, 1.0)

    n = eng.species.index(sid)
    active = np.clip(eng.expression[..., n], 0.0, 1.0)
    activity = (1.0 - ACTIVITY_DEPENDENCE) + ACTIVITY_DEPENDENCE * active

    effective = dose * envelope * activity * (1.0 - shielding)
    return np.clip(np.exp(-effective), 0.0, 1.0)


def apply_antimicrobial(eng, dose: float = 4.0) -> dict:
    """Deliver one broad antimicrobial challenge. Returns per-species survival for the record.

    ``dose`` is in units where 1.0 kills ~63% of a fully-exposed, fully-active population.

    Example: apply_antimicrobial(eng, dose=4.0) — a chlorhexidine-scale rinse
    """
    comp = eng.grid.composition
    survived = {}
    for n, sid in enumerate(eng.species):
        before = float(comp[..., n].sum())
        comp[..., n] *= survival_fraction(eng, sid, dose)
        after = float(comp[..., n].sum())
        survived[sid] = (after / before) if before > 0 else 0.0
    return survived

