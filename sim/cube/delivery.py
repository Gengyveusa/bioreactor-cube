"""Delivery scoring — does the drug REACH the target, not just bind it.

Every drug-prediction tool answers the same question: does this molecule bind. Docking, affinity
prediction, structure-based screening, de novo design — all of them stop at the binding site. None
of them says whether the compound ever arrives at the base of a 6 mm periodontal pocket through a
biofilm that evolved to keep things out.

That is the question this module answers, and it is the one the rest of the engine is unusually
well placed to answer, because the matrix, its porosity, its charge and its external boundary are
all already modelled.

THE THREE RESISTANCES IN SERIES, each measured and each independently calibratable:

  1. EXTERNAL — the bathing film. A Dirichlet face assumes an infinite well-stirred reservoir; a
     pocket is nothing of the sort. FaceConditions.oral_biot / serum_biot carry the Biot number, and
     for crevicular flow it runs 0.04-2.6, so the external layer is often rate-limiting on its own.

  2. SIZE — obstruction by the matrix mesh. This replaces a constant I invented.
     PHAGE_ECM_EXPONENT = 3.0 was a guess marked [[VERIFY]], and the verification came back
     negative: Peulen & Wilkinson 2011 (Environ Sci Technol 45:3367, PMID 21434601) measured
     dextrans, microspheres and Ag nanoparticles in a LIVE biofilm and found exp(-k*r^2) —
     obstruction scaling, Gaussian in radius — not a power of porosity. Waters & Frank 2009
     (PMID 20514136) found obstruction models beat hydrodynamic-scaling ones head to head in
     polymer hydrogels.

     The structural defect of theta^n mattered more than the wrong functional form: it welded
     size-selectivity to the same theta that governs small-solute transport, so the two could never
     be calibrated apart. Here they are separate factors, and the size term goes to 1 as r -> 0, so
     small-solute behaviour is untouched.

  3. CHARGE — electrostatic sequestration. The matrix is not a neutral sieve. eDNA is a polyanion,
     so cationic drugs are retarded and partly immobilised while neutral ones are not. This is a
     mechanism, not a fudge: chlorhexidine is a DICATION, and its notorious gap between in-vitro
     potency and in-vivo performance against established plaque is exactly what a polyanionic
     matrix predicts.

WHAT THIS IS NOT. It is not a PK model in absolute units, and it does not predict binding. It
produces a DELIVERED FRACTION — how much of what you applied reaches the depth you care about —
which is the multiplier that belongs on an affinity score and which nothing in a docking stack
supplies. Score a candidate as affinity x delivered_fraction and you are ranking on something the
rest of the field cannot rank on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from sim.cube.ecm import BIOFILM_ECM, obstruction_factor, porosity

# --------------------------------------------------------------------------- size obstruction
#: Solute radius the supplied Biot number refers to. Biot scales as 1/r through Stokes-Einstein, so
#: a quoted Bi has to be anchored to a size. 0.4 nm is a small solute like lactate or metronidazole.
BIOT_REF_RADIUS_NM = 0.4

#: Hydrodynamic radius from molecular weight, r_nm ~ 0.0515 * MW^0.392 — the standard empirical
#: relation for small molecules and globular proteins in solution. Rough, and fine for ranking.
def radius_from_mw(mw: float) -> float:
    """Hydrodynamic radius in nm from molecular weight in Da.

    Example: radius_from_mw(386.4) -> ~0.53 nm for atuzaginstat
    """
    return float(0.0515 * max(mw, 1.0) ** 0.392)


# --------------------------------------------------------------------------- charge sequestration
#: Fraction of the matrix that behaves as fixed polyanion (eDNA, and the acidic residues on
#: bacterial surfaces). Shellis & Dibdin put ~90% of plaque buffering on immobile surface charge,
#: which is the same population doing the binding here. Until the matrix is speciated into
#: glucan/eDNA/protein this is a single lumped fraction. [[VERIFY]]
POLYANION_FRACTION = 0.55

#: Retardation per unit charge per unit polyanion density. A dication is held roughly four times as
#: hard as a monocation because the interaction goes as the square of valence in a polyelectrolyte
#: brush, which is why charge enters squared below. [[VERIFY]]
CHARGE_BINDING_K = 3.0


@dataclass(frozen=True)
class DrugSpec:
    """A candidate, described by the properties that govern whether it ARRIVES.

    Deliberately not a chemistry object: molecular weight and formal charge at pocket pH are all the
    transport model needs, and both come straight out of a ChEMBL record.

    Example: DrugSpec("atuzaginstat", mw=386.4, charge=1.0)   # lysine-mimetic, cationic
    """

    name: str
    mw: float
    charge: float = 0.0          # formal charge at pocket pH; negative for anions
    radius_nm: Optional[float] = None
    note: str = ""

    @property
    def radius(self) -> float:
        return self.radius_nm if self.radius_nm is not None else radius_from_mw(self.mw)


#: A few reference compounds, for calibration and for sanity.
CHLORHEXIDINE = DrugSpec("chlorhexidine", mw=505.4, charge=2.0,
                         note="bisbiguanide DICATION — the classic in-vitro/in-vivo gap")
ATUZAGINSTAT = DrugSpec("atuzaginstat", mw=386.4, charge=1.0,
                        note="COR388, Phase 2 Kgp inhibitor, CHEMBL5095230; lysine mimetic so "
                             "cationic at pocket pH")
METRONIDAZOLE = DrugSpec("metronidazole", mw=171.2, charge=0.0,
                         note="small and NEUTRAL — the transport best case")
DOXYCYCLINE = DrugSpec("doxycycline", mw=444.4, charge=0.0,
                       note="zwitterionic overall; also a host MMP inhibitor at sub-antimicrobial dose")


# --------------------------------------------------------------------------- the model
def size_factor(m: np.ndarray, drug: DrugSpec) -> np.ndarray:
    """Obstruction factor 0..1 from solute size — the SAME law the phage layer uses.

    One transport model, not two. A virion and a drug molecule are the same physics at different
    radii, and having them disagree would have been a bug waiting to happen.
    """
    return obstruction_factor(m, drug.radius)


def retardation(m: np.ndarray, drug: DrugSpec) -> np.ndarray:
    """Retardation factor R >= 1 — transport is divided by it, so R=1 means unimpeded.

    Only CATIONS are retarded: the fixed matrix charge is negative, so anions are if anything
    excluded rather than bound, and neutrals pass.
    """
    if drug.charge <= 0:
        return np.ones_like(m)
    return 1.0 + CHARGE_BINDING_K * POLYANION_FRACTION * m * drug.charge ** 2


def local_transmissivity(eng, drug: DrugSpec) -> np.ndarray:
    """Per-voxel relative transport for this drug: porosity x size x (1/retardation).

    This is the field a delivery calculation integrates along. All three terms are separable and
    separately calibratable, which was the whole point of retiring theta^n.
    """
    m = eng.extra.get("ecm")
    if m is None:
        m = np.full(eng.grid.shape, BIOFILM_ECM.base_density)
    theta = porosity(m, BIOFILM_ECM.alpha) / BIOFILM_ECM.tortuosity ** 2
    return theta * size_factor(m, drug) / retardation(m, drug)


def delivered_fraction(eng, drug: DrugSpec, *, biot: Optional[float] = None,
                       depth_fraction: float = 0.75) -> float:
    """Fraction of an applied surface dose that reaches ``depth_fraction`` into the pocket.

    Three resistances in series, combined as a harmonic path along the depth axis — the standard
    series-resistance treatment, which correctly lets the WORST layer dominate rather than letting a
    single permeable region rescue an impermeable one.

    ``biot`` is the external boundary resistance for a SMALL solute; it is rescaled per drug by
    Stokes-Einstein. None means the well-stirred Dirichlet limit, which is the optimistic case every
    in-vitro assay reports.

    THIS IS A TRANSIENT, and that assumption is load-bearing. Retardation R slows a molecule down;
    at true steady state under continuous dosing it does NOT lower the final concentration, it only
    delays arrival. Treating R as a delivery penalty is therefore correct for a BOLUS — a 30-second
    rinse, a single irrigation — and wrong for a sustained-release fibre or a systemic drug arriving
    continuously through GCF. The rinse is the case that matters for chlorhexidine and the case this
    function models; for sustained delivery, pass charge=0 and read the size and external terms
    only.

    Example: delivered_fraction(eng, METRONIDAZOLE) > delivered_fraction(eng, CHLORHEXIDINE)
    """
    trans = local_transmissivity(eng, drug)
    nz = trans.shape[2]
    # Path from the mouth-facing face inward to the target depth.
    stop = max(1, int(round((1.0 - depth_fraction) * nz)))
    path = trans[:, :, stop:][:, :, ::-1]            # surface -> depth
    # Series resistance: total resistance is the SUM of per-layer resistances.
    layer_mean = path.reshape(-1, path.shape[2]).mean(axis=0)
    resistance = float(np.sum(1.0 / np.clip(layer_mean, 1e-9, None)))
    internal = float(path.shape[2]) / max(resistance, 1e-9)     # 0..1, harmonic mean

    if biot is None:
        external = 1.0
    else:
        # The external coefficient is NOT drug-independent. k_L = D_water/delta and Stokes-Einstein
        # gives D_water ~ 1/r, so a larger molecule has a proportionally smaller Biot number. The
        # first version of this applied one Biot to every candidate, which rescaled the whole table
        # by an identical factor and could never change a ranking — an external resistance that
        # cannot discriminate is not modelling anything.
        bi = max(float(biot), 0.0) * (BIOT_REF_RADIUS_NM / max(drug.radius, 1e-6))
        external = bi / (bi + 1.0)                   # the same C_s/C_bulk the Robin face applies
    return float(np.clip(internal * external, 0.0, 1.0))


def delivery_score(affinity_pchembl: float, delivered: float) -> float:
    """The two-factor score: potency you can actually deliver, on a pChEMBL-like scale.

    A compound is only as good as the product of what it does at the target and how much of it gets
    there. Subtracting log10(1/delivered) keeps the result in pChEMBL units, so a 10-fold delivery
    penalty costs exactly one log of potency — directly comparable to an SAR gain.

    Example: delivery_score(8.0, 0.01) -> 6.0, i.e. a 100-fold delivery loss eats two logs
    """
    d = float(np.clip(delivered, 1e-12, 1.0))
    return float(affinity_pchembl + np.log10(d))


def compare(eng, drugs, *, biot: Optional[float] = None, depth_fraction: float = 0.75) -> list:
    """Rank candidates by delivered fraction. Returns [(name, delivered, radius, R_mean), ...]."""
    m = eng.extra.get("ecm")
    if m is None:
        m = np.full(eng.grid.shape, BIOFILM_ECM.base_density)
    out = []
    for d in drugs:
        out.append((d.name, delivered_fraction(eng, d, biot=biot, depth_fraction=depth_fraction),
                    d.radius, float(retardation(m, d).mean())))
    return sorted(out, key=lambda r: -r[1])

