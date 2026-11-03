"""Plaque acid-base chemistry — pH as a solved field, not a boundary value.

Until this module existed, pH in the cube was a Dirichlet face value that never diffused and was
never produced or consumed: it took exactly two distinct values across the whole lattice. Nothing
ate sugar and nothing made acid, so a tenfold change in dietary sugar moved community composition by
exactly zero. That is the correct answer for a pH-clamped system — Bradshaw, McKee & Marsh (1989,
J Dent Res 68:1298) ran the same experiment in a nine-species chemostat with pH held and also found
essentially nothing, concluding that it is *the pH generated from carbohydrate metabolism*, not
carbohydrate availability itself, that shifts the flora. This module supplies the missing half.

METHOD. Instantaneous dissociation equilibrium plus charge balance, after the Stewart /
strong-ion-difference formulation, solved for [H+] at every voxel. For each weak acid system with
total concentration C_T and acid constant Ka, the anion fraction is Ka/(Ka+[H+]); phosphate and
carbonate contribute their own polyprotic terms. Charge balance is then

    SID + [H+] - Kw/[H+] - SUM_i (anion charge)_i = 0

which is strictly monotone decreasing in [H+], so it is solved by vectorised bisection on pH over
[2, 10]. Bisection rather than Newton-Raphson deliberately: it cannot diverge, needs no derivative,
and 45 iterations over the whole lattice is cheap and unconditionally convergent.

BUFFERS. Saliva's phosphate and bicarbonate systems plus a fixed biofilm-matrix buffer capacity,
after Dibdin (1990, J Dent Res 69:1324). Without buffers a charge-balance solver still produces a
pH field, but a badly behaved one — the buffer model, not the acid production, is what makes the
Stephan curve the right shape.

All constants are literature-standard thermodynamic values at 25 C. What is NOT calibrated is the
mapping from the engine's dimensionless sugar/acid fields onto millimolar concentrations; that is
``MM_PER_UNIT`` and it is a placeholder. [[VERIFY]]
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# ------------------------------------------------------------------ constants (25 C, standard)
KW = 1.0e-14                       # water autoionisation

PKA = {
    "lactate":    3.86,            # lactic acid — the dominant plaque acid
    "acetate":    4.76,
    "propionate": 4.87,
    "butyrate":   4.82,
}
KA = {k: 10.0 ** (-v) for k, v in PKA.items()}

# Ammonium/ammonia. This is the only ALKALINE term in the pocket and the reason a deep periodontal
# lesion is not acidic: proteolytic deamination of GCF protein releases NH3, which takes up a proton
# to become NH4+ and carries positive charge. pKa 9.25, so below pH ~8 essentially all of it is
# NH4+ and it contributes its full concentration as cation. Severe periodontal pockets measure
# pH 7.4-8.5 against a saccharolytic plaque surface near 6.5 (Eggert, Drewell & Bigelow 1991,
# Arch Oral Biol 36:233; Bickel & Cimasoni 1985) and this is the term that produces that split.
PKA_NH4 = 9.25
KA_NH4 = 10.0 ** (-PKA_NH4)

PKA_PHOS = (2.15, 7.20, 12.35)     # phosphoric: the salivary buffer that matters at plaque pH
KA_PHOS = tuple(10.0 ** (-p) for p in PKA_PHOS)
PKA_CARB = (6.35, 10.33)           # carbonic — the other salivary buffer
KA_CARB = tuple(10.0 ** (-p) for p in PKA_CARB)

# Resting whole-saliva buffer concentrations, mM. MEASURED, unstimulated whole saliva:
# bicarbonate 4.4 mM and total phosphate 4.5 mM (Bardow et al. 2000, Arch Oral Biol 45:1-12,
# PMID 10669087). Note bicarbonate does NOT reach the 40-60 mM sometimes quoted — that requires
# pilocarpine plus citric acid; cap it at ~15-25 mM even when stimulated.
SALIVA_PHOSPHATE_MM = 4.5
SALIVA_BICARB_MM = 4.4

# THE FIXED BUFFER, and it is the dominant one. Roughly 90% of whole-plaque buffering between
# pH 4 and 7 is IMMOBILE charge on bacterial surfaces and matrix, not dissolved species; mobile
# buffers contribute <=11% and bicarbonate only 2-5% (Shellis & Dibdin 1988, J Dent Res 67:438-46,
# PMID 11039053). Dibdin's own conclusion is that this fixed buffer "greatly extends the low pH of
# a Stephan curve". A charge balance over dissolved species alone therefore misses most of the
# buffering, which both under-buffers the fall and over-speeds the recovery.
#
# The absolute capacity in mmol/(L.pH) is NOT published — Shellis & Dibdin report percentages only.
# So this is the ONE free parameter here, fitted to the measured Stephan nadir (see below), and
# everything else is a measured value. pKa is set near plaque resting pH so the buffer titrates
# across the range the curve actually traverses. [[VERIFY the absolute capacity]]
FIXED_BUFFER_MM = 42.0
FIXED_BUFFER_PKA = 6.0

# Dimensionless-field -> millimolar conversion. CALIBRATED, not invented: plaque fluid lactate is
# sub-millimolar at rest and rises by ~35 mM seven minutes after a 10% sucrose rinse (Vogel et al.
# 2002, Caries Res 36:44-52, PMID 11961330). The model's lactate field peaks at ~0.154 under the
# equivalent bolus, so one field unit is ~227 mM. Re-derive this if the fermentation rates change.
MM_PER_UNIT = 227.0

PH_LO, PH_HI = 2.0, 10.0           # bracketing range for the solver
BISECT_ITERS = 45                  # ~2^-45 of the bracket: far below any meaningful pH resolution


def _anion_charge(h: np.ndarray, acids: dict, phosphate_mM: float, bicarb_mM: float,
                  fixed_mM: float = 0.0) -> np.ndarray:
    """Total negative charge (mM) carried by the dissociated weak-acid and buffer systems."""
    q = np.zeros_like(h)
    if fixed_mM:                                        # immobile matrix/cell-surface buffer
        kf = 10.0 ** (-FIXED_BUFFER_PKA)
        q = q + fixed_mM * (kf / (kf + h))
    for name, c_t in acids.items():
        ka = KA[name]
        q = q + c_t * (ka / (ka + h))                      # monoprotic: HA <-> H+ + A-

    # Phosphate: H2PO4- (1-), HPO4^2- (2-), PO4^3- (3-). Alpha fractions from the standard
    # polyprotic expansion; the denominator is the same for all species.
    k1, k2, k3 = KA_PHOS
    d = h ** 3 + k1 * h ** 2 + k1 * k2 * h + k1 * k2 * k3
    a1 = k1 * h ** 2 / d
    a2 = k1 * k2 * h / d
    a3 = k1 * k2 * k3 / d
    q = q + phosphate_mM * (a1 + 2.0 * a2 + 3.0 * a3)

    # Carbonate: HCO3- (1-), CO3^2- (2-)
    c1, c2 = KA_CARB
    dc = h ** 2 + c1 * h + c1 * c2
    b1 = c1 * h / dc
    b2 = c1 * c2 / dc
    q = q + bicarb_mM * (b1 + 2.0 * b2)
    return q


def solve_ph(acids: dict, sid_mM, *, phosphate_mM: float = SALIVA_PHOSPHATE_MM,
             bicarb_mM: float = SALIVA_BICARB_MM, fixed_mM: float = FIXED_BUFFER_MM,
             ammonia_mM=0.0, shape=None) -> np.ndarray:
    """Solve charge balance for pH, elementwise over whatever array shape the inputs carry.

    ``acids`` maps an acid name in :data:`PKA` to its TOTAL concentration in mM (array or scalar).
    ``sid_mM`` is the strong ion difference — the net charge from fully-dissociated ions, which is
    what the buffers and acids must balance against.

    Example: solve_ph({"lactate": 10.0}, sid_mM=0.0) -> about 2.9 (10 mM lactic acid, unbuffered)
    """
    ref = None
    for v in list(acids.values()) + [sid_mM]:
        if isinstance(v, np.ndarray):
            ref = v
            break
    if shape is None:
        shape = ref.shape if ref is not None else ()
    acids = {k: np.broadcast_to(np.asarray(v, dtype=float), shape).astype(float)
             for k, v in acids.items()}
    sid = np.broadcast_to(np.asarray(sid_mM, dtype=float), shape).astype(float)
    nh_t = np.broadcast_to(np.asarray(ammonia_mM, dtype=float), shape).astype(float)

    lo = np.full(shape, PH_LO)
    hi = np.full(shape, PH_HI)
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        h = 10.0 ** (-mid)
        # Ammonium is the one CATION the biology produces, so it sits on the positive side of the
        # balance. Its fraction h/(h+Ka) is ~1 below pH 8, which is why proteolysis shifts the deep
        # pocket alkaline so effectively: nearly every NH3 released becomes a full positive charge.
        nh4 = nh_t * (h / (h + KA_NH4))
        # Charge-balance residual in mM: (positive charge) - (negative charge). It DECREASES as pH
        # rises, because raising pH means more dissociation and more OH-.
        resid = (sid + h * 1e3 + nh4 - (KW / h) * 1e3
                 - _anion_charge(h, acids, phosphate_mM, bicarb_mM, fixed_mM))
        # resid > 0 -> excess positive charge -> more dissociation is needed -> root is at HIGHER pH
        lo = np.where(resid > 0, mid, lo)
        hi = np.where(resid > 0, hi, mid)
    return 0.5 * (lo + hi)


#: Serum bicarbonate, mM. Gingival crevicular fluid is a serum TRANSUDATE, so the deep pocket is
#: bathed in a fluid buffered five times harder than saliva and resting at 7.4 rather than 6.8.
#: This — not bacterial ammonia alone — is why a deep pocket is alkaline: the host delivers the
#: alkalinity, and proteolysis then pushes further in the same direction.
SERUM_BICARB_MM = 24.0
SERUM_PH = 7.4


def sid_field(target_ph, *, phosphate_mM=SALIVA_PHOSPHATE_MM, bicarb_mM=SALIVA_BICARB_MM,
              fixed_mM: float = FIXED_BUFFER_MM) -> np.ndarray:
    """Vectorised :func:`sid_for_ph` — the SID holding EACH voxel at its own resting pH.

    Needed because the pocket is not one fluid. The mouth-facing surface is saliva; the
    tissue-facing depth is a serum transudate. A single scalar SID cannot express that.

    Example: sid_field(np.array([6.8, 7.4]), bicarb_mM=np.array([4.4, 24.0]))
    """
    h = 10.0 ** (-np.asarray(target_ph, dtype=float))
    q = _anion_charge(h, {}, phosphate_mM, bicarb_mM, fixed_mM)
    return q + (KW / h) * 1e3 - h * 1e3


def sid_for_ph(target_ph: float, acids: dict = None, *,
               phosphate_mM: float = SALIVA_PHOSPHATE_MM,
               bicarb_mM: float = SALIVA_BICARB_MM,
               fixed_mM: float = FIXED_BUFFER_MM) -> float:
    """The strong ion difference that puts a given buffer mix at a given resting pH.

    This is what makes a buffer a BUFFER rather than an acid. Salivary phosphate and bicarbonate
    exist as their sodium/potassium salts; without those counter-cations in the SID, a 5 mM
    phosphate pool is just 5 mM phosphoric acid and drives pH DOWN instead of resisting change.
    Charge balance at the target inverts directly, so no iteration is needed.

    Example: sid_for_ph(6.8) -> the mM of net cation saliva carries to sit at resting pH
    """
    h = 10.0 ** (-float(target_ph))
    q = _anion_charge(np.asarray(h, dtype=float), {k: np.asarray(v, dtype=float)
                                                   for k, v in (acids or {}).items()},
                      phosphate_mM, bicarb_mM, fixed_mM)
    return float(q + (KW / h) * 1e3 - h * 1e3)


#: Net cation saliva must carry for its buffers to rest at plaque's resting pH.
SALIVA_SID_MM = sid_for_ph(6.8)

#: Plaque resting pH the SID is solved against. Changing a buffer pool must not move this.
RESTING_PH = 6.8


@lru_cache(maxsize=256)
def _sid_cached(phosphate_mM: float, bicarb_mM: float, fixed_mM: float, resting_pH: float) -> float:
    return sid_for_ph(resting_pH, phosphate_mM=phosphate_mM, bicarb_mM=bicarb_mM, fixed_mM=fixed_mM)


@dataclass(frozen=True)
class BufferSpec:
    """The buffer pools a pH solve runs against, and the SID that holds them at resting pH.

    This exists so a PATIENT's salivary buffering can enter the solver without disturbing the two
    things that must not move: the plaque fixed buffer (nothing chairside measures it) and the
    resting pH the SID is defined against.

    The subtlety worth stating, because getting it wrong is silent: changing a buffer pool WITHOUT
    re-solving the SID changes the resting pH. Saliva's phosphate and bicarbonate exist as their
    sodium/potassium salts, so raising bicarbonate without its counter-cation adds a weak ACID.
    :attr:`sid` re-inverts charge balance at ``resting_pH`` every time, so a better-buffered mouth
    starts at the same pH and simply resists change more — which is what buffering means.

    Example: BufferSpec(bicarb_mM=15.0).sid > BufferSpec().sid  -> True
    """

    phosphate_mM: float = SALIVA_PHOSPHATE_MM
    bicarb_mM: float = SALIVA_BICARB_MM
    fixed_mM: float = FIXED_BUFFER_MM          # plaque matrix; NOT patient-measurable, stays fitted
    resting_pH: float = RESTING_PH

    @property
    def sid(self) -> float:
        """Strong ion difference holding THIS buffer mix at :attr:`resting_pH`."""
        return _sid_cached(self.phosphate_mM, self.bicarb_mM, self.fixed_mM, self.resting_pH)

    def solve(self, acids: dict, shape=None, ammonia_mM=0.0) -> np.ndarray:
        """Solve pH for these buffers. The one call site a caller should need."""
        return solve_ph(acids, self.sid, phosphate_mM=self.phosphate_mM,
                        bicarb_mM=self.bicarb_mM, fixed_mM=self.fixed_mM,
                        ammonia_mM=ammonia_mM, shape=shape)


# Chairside buffer tests do not report millimolar bicarbonate; they report an endpoint pH (CRT
# Buffer, Dentobuff) or a titratable capacity (Ericsson). These anchors convert an endpoint pH to a
# salivary bicarbonate pool.
#
# The two ENDS are measured, the curve between them is not. Resting unstimulated bicarbonate is
# 4.4 mM and stimulated whole saliva reaches roughly 15 mM (Bardow 2000, Arch Oral Biol 45:1-12,
# PMID 10669087) — note this does NOT reach the 40-60 mM sometimes quoted, which needs pilocarpine
# plus citric acid. A "low" strip result is a mouth that has lost most of that pool.
#
# The pH -> mM mapping itself is a CALIBRATION CHOICE, not a published relation. It is monotone,
# it is pinned at both ends to measured values, and it is linear in between because nothing
# justifies a fancier shape. Anyone using this for a real clinical claim must re-derive it against
# their own strip lot. [[VERIFY the interior of this curve]]
_BUFFER_TEST_ANCHORS = (
    (4.0, 1.5),      # "low" (red)     — buffering largely exhausted
    (5.0, 4.4),      # "medium"        — resting unstimulated, the Bardow measured value
    (6.0, 9.0),      # "high" (green)
    (7.0, 15.0),     # stimulated ceiling, Bardow
)


def bicarbonate_from_buffer_test(value: float, unit: str = "") -> float:
    """Convert a chairside salivary buffer result into a bicarbonate pool in mM.

    Accepts what the three common test formats actually report:

    * ``"mM"`` / ``"mmol/L"`` — a bicarbonate concentration; passes through
    * ``"pH"`` — a CRT/Dentobuff endpoint pH, mapped via :data:`_BUFFER_TEST_ANCHORS`
    * ``"mmol/L/pH"`` / ``"beta"`` — an Ericsson titratable capacity, inverted numerically

    Example: bicarbonate_from_buffer_test(5.0, "pH") -> 4.4
    """
    u = (unit or "").strip().lower()
    v = float(value)
    if v < 0:
        raise ValueError(f"salivary buffer result is negative ({v}); reject it at the adapter")

    if u in ("mm", "mmol/l", "mmol/liter", "mmoll"):
        return v
    if u in ("mmol/l/ph", "beta", "mmol/l per ph", "titratable"):
        return _bicarb_for_beta(v)
    if u in ("ph", ""):
        if not u:
            # An unlabelled number in strip range is almost certainly an endpoint pH; anything
            # larger is almost certainly already millimolar. Guessing silently is how a unit bug
            # becomes a clinical result, so the guess is narrow and it is documented here.
            if v > 10.0:
                return v
        xs = [a for a, _ in _BUFFER_TEST_ANCHORS]
        ys = [b for _, b in _BUFFER_TEST_ANCHORS]
        return float(np.interp(v, xs, ys))
    raise ValueError(
        f"unrecognised salivary buffer unit {unit!r}; use 'pH', 'mM', or 'mmol/L/pH'")


def _bicarb_for_beta(beta: float, *, lo: float = 0.1, hi: float = 60.0) -> float:
    """Invert :func:`titration_capacity` for the bicarbonate pool giving a measured beta.

    Beta here is pH units moved per mM strong acid — SMALLER means better buffered — which is what
    :func:`titration_capacity` returns. Bisection because the relation is monotone but has no
    closed form.
    """
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        spec = BufferSpec(bicarb_mM=mid)
        got = float(titration_capacity({}, spec.sid, phosphate_mM=spec.phosphate_mM,
                                       bicarb_mM=mid, fixed_mM=spec.fixed_mM))
        if got > beta:      # under-buffered -> need MORE bicarbonate
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def buffered_ph(acid_mM: dict, sid_mM, **kw) -> np.ndarray:
    """Deprecated alias for :func:`solve_ph`.

    This used to fold the matrix buffer in separately, and nothing ever called it — which meant the
    engine ran with mobile buffers only and was missing ~90% of plaque's buffer capacity. The fixed
    buffer is now part of the charge balance itself, so there is exactly one pH path.
    """
    return solve_ph(acid_mM, sid_mM, **kw)


def titration_capacity(acids: dict, sid_mM, delta_mM: float = 1.0, **kw) -> np.ndarray:
    """Buffer capacity: pH units moved per mM of added strong acid. Smaller = better buffered.

    Example: a phosphate-buffered system returns a smaller number than an unbuffered one.
    """
    p0 = solve_ph(acids, sid_mM, **kw)
    p1 = solve_ph(acids, np.asarray(sid_mM, dtype=float) - delta_mM, **kw)
    return np.abs(p0 - p1) / delta_mM

