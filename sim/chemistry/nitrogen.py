"""Enterosalivary nitrate → nitrite → NO reduction kinetics (pure, substrate-limited).

The nitrogen / NO axis (Layer 3). Oral nitrate-reducers respire nitrate as a
terminal electron acceptor — favored when O2 is low (facultative anaerobic
respiration) — producing nitrite; nitrite is reduced to NO, favored at low pH
(acidic protonation). Nitrite/ammonia generation buffers pH upward — the
protective, caries-suppressing coupling.

This replaces the previous flat ``NO_yield`` production (which produced NO with no
nitrate substrate): NO now genuinely depends on dietary nitrate supply and on a
living nitrate-reducer community. It is a pure function so the kinetics are
testable without the agent / model machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

# Fraction of available reducer activity that turns over per tick, and the pH
# sensitivity of nitrite→NO. Placeholders pending calibration (design §13).
_REDUCTION_RATE = 0.15
_NITRATE_KM = 0.5          # Michaelis half-saturation: below this, reduction tracks nitrate supply
_O2_SENSITIVITY = 0.2      # mildness of the O2 gate (facultative reducers work on the aerobic tongue)
_NO_BASE_FRACTION = 0.10
_NO_PH_SENSITIVITY = 0.15
_PH_BUFFER_PER_REDUCED = 0.01


@dataclass(frozen=True)
class NitrogenFlux:
    """Per-tick deltas to apply to the local chemistry fields."""

    d_nitrate: float
    d_nitrite: float
    d_no: float
    d_ph: float


def nitrite_to_no_fraction(ph: float) -> float:
    """Fraction of reduced nitrogen that becomes NO (vs staying nitrite), higher at
    low pH. Clamped to [0.05, 0.8]."""
    return min(0.8, max(0.05, _NO_BASE_FRACTION + _NO_PH_SENSITIVITY * (7.0 - ph)))


def reduce_nitrate(nitrate: float, o2: float, ph: float, capacity: float) -> NitrogenFlux:
    """Substrate-limited nitrate reduction by one reducer's activity this tick.

    ``capacity`` is the reducer's activity = ``NO_yield * growth * biomass``. Returns
    the field deltas (nitrate consumed; nitrite + NO produced; pH buffered up). No
    nitrate or no reducer activity → zero flux (the substrate gate).
    """
    if nitrate <= 0.0 or capacity <= 0.0:
        return NitrogenFlux(0.0, 0.0, 0.0, 0.0)
    o2_gate = 1.0 / (1.0 + _O2_SENSITIVITY * max(0.0, o2))   # facultative: favored anoxic, not abolished aerobic
    mm = nitrate / (_NITRATE_KM + nitrate)          # saturable in nitrate → tracks dietary supply
    reduced = min(nitrate, _REDUCTION_RATE * capacity * o2_gate * mm)
    no_frac = nitrite_to_no_fraction(ph)
    return NitrogenFlux(
        d_nitrate=-reduced,
        d_nitrite=reduced * (1.0 - no_frac),
        d_no=reduced * no_frac,
        d_ph=_PH_BUFFER_PER_REDUCED * reduced,
    )

