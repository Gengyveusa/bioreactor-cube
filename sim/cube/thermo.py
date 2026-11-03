"""thermo.py — the entropy-production (sigma) probe for the Cube. STANDALONE / UNWIRED.

The cube is an open, far-from-equilibrium system. Its single tissue-agnostic invariant is the
local entropy-production density (de Groot-Mazur / Prigogine form), second-law non-negative:

    sigma(x) = (1/T) [ SUM_reactions  r_j * A_j   +   SUM_species  J_i . (-grad mu_i) ]   >= 0
                       \___ reaction term ___/        \_____ transport term _____/

This module computes sigma from plain numpy fields on the cube grid. It is deliberately DECOUPLED
from the engine (imports only numpy, edits nothing) so we can validate it in isolation before
wiring it in. A tissue is ONLY a different (reaction network + boundary mu + transport map); the
calculus here is invariant -- which is the whole point of "drop the cube in any tissue".

Grid conventions (match grid.CubeGrid / ecm._diff3):
    axis 0 (x): bounded, no-flux   (enamel pellicle  <->  soft tissue)
    axis 1 (y): PERIODIC           (adjacent biofilm)
    axis 2 (z): bounded, no-flux   (oral surface z=nz-1  <->  deep z=0)

Units & honesty: model concentrations are dimensionless occupancies (~0..1), so sigma here is a
RELATIVE dissipation density that is consistent across tissues and across health states -- NOT
W/(K.m^3). Absolute units need real concentrations + reaction Gibbs energies. The transport term is
rigorous given the fields; the reaction term's affinities are placeholders [[VERIFY]].
"""

from __future__ import annotations

import numpy as np

R = 8.314          # J/mol/K, gas constant
T_BODY = 310.15    # 37 C; uniform tissue temperature assumed [[VERIFY: isothermal approximation]]
_C_FLOOR = 1e-4    # concentration floor: |grad c|^2 / c diverges as c->0 (the real anaerobic front);
                   # floor regularizes it. sigma is mildly sensitive to this near c~0 -- documented.
_EPS = 1e-30

# --------------------------------------------------------------------- calibration  [[VERIFY all]]
# Moves sigma from "relative" toward physically-scaled, and de-dominates the unbounded inflammation
# field, WITHOUT touching the engine. Per species: (s_ref, D_rel).
#   s_ref  = physiological reference scale in MODEL units. Dividing c by it makes the fields
#            comparable; the unbounded inflammation field gets a large s_ref so it stops drowning the
#            lesion species. (For 0..1 occupancy fields s_ref=1 leaves them unchanged.)
#   D_rel  = diffusivity relative to free water — small molecules ~1, large proteins/particles slow.
# These are literature-anchored PLACEHOLDERS; refine the numbers (esp. inflammation s_ref) later.
CAL = {
    "O2":           (1.0, 1.00), "sugar":   (1.0, 0.60), "lactate": (1.0, 0.70),
    "NO":           (1.0, 1.20), "nitrate": (1.0, 0.70), "nitrite": (1.0, 0.70),
    "heme":         (1.0, 0.30), "Ca":      (1.0, 0.55), "PO4":     (1.0, 0.50),
    "pH":           (1.0, 0.90),
    "inflammation": (1.0e3, 0.20),   # cytokine/mediator proxy: large s_ref de-dominates; protein-slow D
    "oxLDL":        (1.0, 0.10), "amyloid": (1.0, 0.15), "lipid":   (1.0, 0.20),
}
CAL_DEFAULT = (1.0, 0.50)


# --------------------------------------------------------------------- grid-consistent gradient
def grad_sq(f: np.ndarray) -> np.ndarray:
    """|grad f|^2 on the cube grid (unit spacing): central differences, x & z no-flux, y periodic."""
    gx = np.empty_like(f)
    gx[1:-1] = 0.5 * (f[2:] - f[:-2]); gx[0] = f[1] - f[0]; gx[-1] = f[-1] - f[-2]
    gz = np.empty_like(f)
    gz[:, :, 1:-1] = 0.5 * (f[:, :, 2:] - f[:, :, :-2])
    gz[:, :, 0] = f[:, :, 1] - f[:, :, 0]; gz[:, :, -1] = f[:, :, -1] - f[:, :, -2]
    gy = 0.5 * (np.roll(f, -1, 1) - np.roll(f, 1, 1))              # periodic in y
    return gx * gx + gy * gy + gz * gz


# --------------------------------------------------------------------- the two sigma terms
def transport_sigma(fields: dict[str, np.ndarray],
                    d_eff: dict[str, np.ndarray] | float,
                    norm: str | None = None, calibrated: bool = False) -> np.ndarray:
    """Diffusive entropy production: SUM_i D_eff,i * R * |grad c_i|^2 / c_i   (>= 0 by construction).

    Derivation: J_i = -D_eff grad c_i (Fick); force X_i = -grad(mu_i)/T with mu_i = mu_i0 + RT ln c_i,
    so X_i = -(R/c_i) grad c_i and sigma_i = J_i.X_i = D_eff R |grad c_i|^2 / c_i. T cancels. Rigorous.

    norm="per_species": divide each species' field by its own integral before summing, so a high-
    concentration supply species (O2) no longer drowns the lesion species -- a SCALE-free pattern
    view (use for "where does dissipation structurally live", not for magnitude).

    calibrated=True: apply the per-species CAL table (c -> c/s_ref, D_eff -> D_eff*D_rel) so fields
    are on comparable physiological scales and weighted by real diffusivity -- de-dominates the
    unbounded inflammation field and moves sigma toward physical units. Numbers are [[VERIFY]].
    """
    keys = list(fields)
    s = np.zeros_like(fields[keys[0]], dtype=np.float64)
    for k in keys:
        c = fields[k]
        D = d_eff[k] if isinstance(d_eff, dict) else d_eff
        if calibrated:
            s_ref, d_rel = CAL.get(k, CAL_DEFAULT)
            c = c / s_ref
            D = D * d_rel
        term = D * R * grad_sq(c) / np.clip(c, _C_FLOOR, None)
        if norm == "per_species":
            term = term / (term.sum() + _EPS)
        s += term
    return s


def delta_sigma(sigma_disease: np.ndarray, sigma_health: np.ndarray) -> np.ndarray:
    """Disease minus health. The supply-boundary common mode (same boundary conditions in both)
    cancels, leaving the LESION's thermodynamic footprint: where disease adds [+] or removes [-]
    irreversible dissipation. This is the principled fix for supply-boundary dominance."""
    return sigma_disease - sigma_health


def reaction_sigma(rates: dict[str, np.ndarray],
                   affinities: dict[str, np.ndarray | float],
                   T: float = T_BODY) -> np.ndarray:
    """Reaction entropy production: SUM_j r_j * A_j / T,  A_j = -dG_j (J/mol).  [[VERIFY affinities]]

    Spontaneous reactions (A>0) running forward (r>0) produce entropy. SCAFFOLD: affinities are
    placeholders until wired to real dG0 + activities; the SHAPE of the law is what we validate now.
    """
    keys = list(rates)
    s = np.zeros_like(rates[keys[0]], dtype=np.float64)
    for j in keys:
        A = affinities[j] if isinstance(affinities, dict) else affinities
        s += rates[j] * A
    return s / T


def total_sigma(fields, d_eff, rates=None, affinities=None, T: float = T_BODY,
                norm=None, calibrated: bool = False) -> np.ndarray:
    """Full sigma field. Transport term always; reaction term if rates/affinities supplied."""
    s = transport_sigma(fields, d_eff, norm=norm, calibrated=calibrated)
    if rates:
        s = s + reaction_sigma(rates, affinities, T)
    return s


# --------------------------------------------------------------------- readouts
def summary(sigma: np.ndarray) -> dict:
    """Scalar descriptors of a sigma field -- the candidate thermodynamic state variables."""
    m = float(sigma.mean())
    return {
        "Sigma_total": float(sigma.sum()),          # whole-cube dissipation
        "sigma_mean": m,
        "sigma_max": float(sigma.max()),
        "hotspot_frac": float((sigma > 5 * m).mean()),  # fraction of voxels in dissipative fronts
    }


def depth_profile(sigma: np.ndarray) -> np.ndarray:
    """Mean sigma per z-layer (oral surface z=nz-1 -> deep z=0): where does dissipation concentrate?"""
    return sigma.mean(axis=(0, 1))


# --------------------------------------------------------------------- wiring stub (NOT used yet)
def sigma_from_engine(eng, solutes, d_eff=None, rates=None, affinities=None):
    """[STAGE 2 — not called yet] Pull live fields off a running engine and compute sigma.

    Decoupled: no import of the engine; duck-typed at call time. When we wire it in:
        fields = {k: eng.field(k) for k in solutes}
        d_eff  = d_eff or eng.extra.get("d_eff", 1.0)   # or recompute from ecm: theta/lambda^2
    Kept here only to pin the integration contract; the engine is untouched.
    """
    fields = {k: eng.field(k) for k in solutes}
    return total_sigma(fields, d_eff if d_eff is not None else 1.0, rates, affinities)


# ===================================================================== standalone validation
def _synthetic_state(nx=8, ny=8, nz=24, *, steepness=8.0, deep_lactate=0.9, anoxic=False):
    """A toy oral-pocket state: O2 in from the oral face (z=nz-1) decaying into depth; lactate &
    H+ accumulating deep. `steepness` small = shallow gradients (healthy-ish); large = steep
    fronts. `anoxic` drives the deep O2 to ~0 (dysbiotic deep anaerobiosis)."""
    z = np.arange(nz)
    d_from_oral = (nz - 1) - z                                  # 0 at oral surface, large deep
    o2_col = 8.0 * np.exp(-d_from_oral / steepness)
    if anoxic:
        o2_col = o2_col * np.exp(-d_from_oral / 3.0)            # collapse O2 in depth
    lac_col = deep_lactate * (1.0 - np.exp(-d_from_oral / steepness))
    o2 = np.broadcast_to(o2_col, (nx, ny, nz)).astype(np.float64).copy()
    lac = np.broadcast_to(lac_col, (nx, ny, nz)).astype(np.float64).copy()
    # mild lateral structure so y/x gradients aren't exactly zero (realistic, breaks symmetry)
    ramp = np.linspace(0, 0.05, nx)[:, None, None]
    o2 += ramp
    biomass = np.broadcast_to(1.0 - np.exp(-d_from_oral / 6.0), (nx, ny, nz)).astype(np.float64).copy()
    return {"O2": o2, "lactate": lac}, biomass


def _demo():
    R_AERO = 4.8e5  # placeholder affinity for aerobic respiration, J/mol O2 [[VERIFY]]
    print("=" * 70)
    print("sigma-probe standalone validation (entropy-production density)")
    print("=" * 70)

    # ---- check 1: flat field -> ~zero transport sigma
    flat = {"O2": np.full((6, 6, 6), 5.0), "lactate": np.full((6, 6, 6), 0.3)}
    s_flat = transport_sigma(flat, d_eff=1.0)
    print(f"[1] flat field  -> Sigma_total = {s_flat.sum():.3e}   (expect ~0)")
    assert s_flat.sum() < 1e-9, "flat field must produce ~zero dissipation"

    # ---- check 2: sigma >= 0 everywhere (second law), on a real gradient state
    fields, bio = _synthetic_state(steepness=8.0)
    s = transport_sigma(fields, d_eff={"O2": 1.0, "lactate": 0.6})
    print(f"[2] gradient    -> min sigma = {s.min():.3e}   (must be >= 0)")
    assert s.min() >= -1e-12, "second law violated: sigma < 0"

    # ---- check 3: steeper gradients dissipate more
    healthy, _ = _synthetic_state(steepness=14.0)                       # shallow
    dysb,    bd = _synthetic_state(steepness=5.0, anoxic=True)          # steep + deep anoxia
    deff = {"O2": 1.0, "lactate": 0.6}
    sH = transport_sigma(healthy, deff)
    sD = transport_sigma(dysb, deff)
    print(f"[3] healthy-ish Sigma = {sH.sum():.3e}   dysbiotic Sigma = {sD.sum():.3e}")
    assert sD.sum() > sH.sum(), "steeper/anoxic state should dissipate more"

    # ---- check 4: reaction term adds where O2 is consumed (respiration ~ O2 * biomass)
    rate = {"resp": fields["O2"] * bio * 0.1}
    sT = total_sigma(fields, deff, rates=rate, affinities={"resp": R_AERO})
    print(f"[4] +reaction   -> Sigma_total = {sT.sum():.3e}  (transport-only was {s.sum():.3e})")
    assert sT.sum() > s.sum(), "reaction term must add dissipation"

    # ---- where does it concentrate? (the aerobic/anaerobic front is the dissipative hotspot)
    prof = depth_profile(sD)
    zmax = int(np.argmax(prof))
    print(f"\n[depth] dissipation peaks at z={zmax} of {len(prof)-1} "
          f"(z=nz-1 oral surface, z=0 deep) -> the redox front")
    print("        sigma(z), oral->deep:")
    bars = prof / prof.max()
    for zz in range(len(prof) - 1, -1, -1):
        print(f"        z={zz:2d} |{'#' * int(40 * bars[zz]):<40}| {prof[zz]:.3e}")

    print("\nsummary(dysbiotic):", {k: round(v, 4) for k, v in summary(sD).items()})
    print("\nALL CHECKS PASSED — probe is sound, second-law-safe, separates states. Unwired.")


if __name__ == "__main__":
    _demo()

