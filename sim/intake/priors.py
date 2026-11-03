"""Conditional priors — fill the species a panel did not measure.

A qPCR pathogen panel measures ~10 organisms. The cube simulates 18, and the ones a pathogen panel
omits are exactly the commensals and nitrate-reducers that the NO axis and the dysbiosis denominator
are built from. Filling them with a fixed population average would wash out the signal, because a
mouth with elevated red-complex is *not* a mouth with average commensals.

So the completion is CONDITIONAL: measured pathogen elevation shifts the unmeasured protective flora
down and the unmeasured late-succession flora up, around the population prior. Everything produced
here is stamped ``Provenance.PRIOR`` — it is a defensible inference, never a measurement.

The shift is deliberately gentle and monotone rather than precisely tuned. There is no dataset behind
its magnitude yet; calibrating it is exactly what the published-cohort work is for. [[VERIFY]]
"""

from __future__ import annotations

import math

from sim.species import CATALOG
from sim.cube.sample import DEFAULT_ABUNDANCE
from sim.intake.schema import Provenance, Value

# How hard a measured dysbiosis signal is allowed to move an unmeasured species, as a multiplicative
# factor bound. 2.0 means "at most doubled, at least halved". Uncalibrated. [[VERIFY]]
MAX_SHIFT = 2.0


def _protective(sp) -> bool:
    """Early colonizers and nitrate-reducers — the flora that a healthy pocket is rich in."""
    return bool(getattr(sp, "nitrate_reducer", False)) or getattr(sp, "succession", "") == "early"


def _pathogenic(sp) -> bool:
    """Late-succession and red/orange-complex organisms."""
    return (getattr(sp, "succession", "") == "late"
            or getattr(sp, "socransky", "") in ("red", "orange"))


def dysbiosis_signal(measured: dict) -> float:
    """A scalar in [-1, 1] describing how pathogen-shifted the MEASURED subset is versus prior.

    Positive means the measured pathogens came back higher than the population prior expects.
    Returns 0.0 when nothing informative was measured, which makes the completion fall back to the
    unconditioned prior — the correct behaviour, not a silent failure.

    Example: dysbiosis_signal({"Pg": 0.30}) -> a positive number (prior for Pg is 0.07)
    """
    by_id = {s.id: s for s in CATALOG}
    total = sum(v for v in measured.values() if v > 0)
    if total <= 0:
        return 0.0

    obs_path = sum(v for k, v in measured.items()
                   if v > 0 and k in by_id and _pathogenic(by_id[k])) / total
    # Prior expectation for the pathogen share *within the measured subset*, so a pathogen-only
    # panel is not automatically read as dysbiotic just because it omitted the commensals.
    prior_sub = {k: DEFAULT_ABUNDANCE.get(k, 0.0) for k in measured if measured[k] > 0}
    prior_total = sum(prior_sub.values())
    if prior_total <= 0:
        return 0.0
    exp_path = sum(v for k, v in prior_sub.items()
                   if k in by_id and _pathogenic(by_id[k])) / prior_total

    if exp_path <= 0 and obs_path <= 0:
        return 0.0
    # Log-ratio, squashed. Bounded, symmetric, and 0 when observation matches expectation.
    eps = 1e-6
    lr = math.log((obs_path + eps) / (exp_path + eps))
    return max(-1.0, min(1.0, lr / 2.0))


# A partially-covered panel cannot tell us what SHARE of the whole community it measured. We
# estimate it as the prior-expected share of those species, moved by the observed elevation, then
# clip so a panel can never claim the entire community or none of it. Uncalibrated. [[VERIFY]]
MEASURED_SHARE_BOUNDS = (0.05, 0.90)


def complete_composition(measured: dict, source: str = "conditional prior") -> dict:
    """Return a full {species_id: Value} over the 18-species CATALOG, summing to 1.0.

    Handles the unit problem that makes naive completion wrong. A qPCR panel reports absolute
    genome equivalents (~1e6); the priors are relative fractions (~0.07). Mixing them directly lets
    the measured species outweigh every prior by six orders of magnitude, so the completion collapses
    to "the panel is the whole community" and any coverage metric computed from it reads 100%.

    Instead we treat the measured values as informative about proportions *within the measured
    group* only, and separately estimate what share of the whole community that group occupies --
    because a pathogen panel genuinely does not measure that, and pretending otherwise is the
    single most misleading thing this module could do.

    Example
    -------
    >>> out = complete_composition({"Pg": Value(4.2e6, Provenance.MEASURED)})
    >>> abs(sum(v.value for v in out.values()) - 1.0) < 1e-9
    True
    >>> out["Rm"].provenance is Provenance.PRIOR
    True
    """
    by_id = {s.id: s for s in CATALOG}
    meas = {k: v for k, v in measured.items() if k in by_id and v.value > 0}
    if not meas:
        raise ValueError("no measured species map onto the catalog; nothing to condition on")

    meas_num = {k: v.value for k, v in meas.items()}
    signal = dysbiosis_signal(meas_num)

    # 1. Proportions WITHIN the measured group (scale-free, so absolute or relative both work).
    m_total = sum(meas_num.values())
    within = {k: v / m_total for k, v in meas_num.items()}

    # 2. What share of the community does the measured group plausibly occupy?
    prior_share = sum(DEFAULT_ABUNDANCE.get(k, 0.0) for k in meas)
    prior_total = sum(DEFAULT_ABUNDANCE.values()) or 1.0
    share = (prior_share / prior_total) * (MAX_SHIFT ** signal)
    lo, hi = MEASURED_SHARE_BOUNDS
    share = max(lo, min(hi, share))
    if len(meas) == len(CATALOG):          # everything was measured; no room for priors
        share = 1.0

    # 3. Unmeasured species split the remainder by conditional prior.
    unmeasured = [sp for sp in CATALOG if sp.id not in meas]
    weights: dict = {}
    for sp in unmeasured:
        base = DEFAULT_ABUNDANCE.get(sp.id, 0.01)
        if _protective(sp):
            factor = MAX_SHIFT ** (-signal)     # dysbiotic panel -> less protective flora
        elif _pathogenic(sp):
            factor = MAX_SHIFT ** (signal)      # dysbiotic panel -> more late flora
        else:
            factor = 1.0
        weights[sp.id] = base * factor
    w_total = sum(weights.values()) or 1.0

    out: dict = {}
    for k, v in meas.items():
        out[k] = Value(within[k] * share, v.provenance, v.source, "relative", v.note)
    for sp in unmeasured:
        frac = (weights[sp.id] / w_total) * (1.0 - share)
        out[sp.id] = Value(
            value=frac,
            provenance=Provenance.PRIOR,
            source=source,
            unit="relative",
            note=(f"population prior x{weights[sp.id] / max(DEFAULT_ABUNDANCE.get(sp.id, 0.01), 1e-9):.2f} "
                  f"(dysbiosis signal {signal:+.2f}); not measured. The measured group is estimated "
                  f"to be {share:.0%} of the community."),
        )

    total = sum(v.value for v in out.values())
    if total <= 0:
        raise ValueError("completed composition sums to zero; the panel carried no usable abundance")
    return {k: Value(v.value / total, v.provenance, v.source, "relative", v.note)
            for k, v in out.items()}

