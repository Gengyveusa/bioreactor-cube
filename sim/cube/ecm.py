"""The extracellular space, characterised PER TISSUE — the medium of the edges (see EXTRACELLULAR-SPACE.md).

Each tissue's ECM is a different wire: the oral biofilm's EPS gel, the artery's dense fibrous collagen
that RETAINS LDL, the brain's hyaluronan/lectican gel with its measured low tortuosity that the glymphatic
system flows through. This module is the model's own characterisation (composition + parameters) and the
`ecm_metrics` calculation that turns each characterisation into transport numbers (porosity θ, effective
diffusivity D_eff = D₀·θ/λ²). Composition is well-established biology; precise numerics are marked [[VERIFY]].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# The lattice size at which the engine's rate constants were implicitly tuned. Diffusion is scaled
# relative to this so that refining the mesh resolves the same physical transport instead of
# changing it. Do not treat it as physical — it is the calibration anchor, nothing more.
DIFFUSION_REF_DIM = 10


def diffusion_iters(n: int) -> int:
    """How many stencil applications equal one step of PHYSICAL diffusion at lattice size n.

    One application of the 6-neighbour stencil moves information exactly ONE CELL, whatever that
    cell measures. So the distance diffused per timestep is dx = L/n, and refining the mesh silently
    SLOWS transport: at n=24 the shipped model diffused less than half as far per step as at n=10.
    Since diffusion length goes as sqrt(D t) and the lattice diffusivity is D ~ dx^2 per
    application, holding physical D fixed requires (n/n_ref)^2 applications.

    Example: diffusion_iters(20) -> 4   (with DIFFUSION_REF_DIM = 10)
    """
    return max(1, int(round((float(n) / DIFFUSION_REF_DIM) ** 2)))


def _stencil(f: np.ndarray) -> np.ndarray:
    """One application of the 6-neighbour stencil: x & z bounded (no-flux), y periodic."""
    xm = np.empty_like(f); xm[1:] = f[:-1]; xm[0] = f[0]
    xp = np.empty_like(f); xp[:-1] = f[1:]; xp[-1] = f[-1]
    zm = np.empty_like(f); zm[:, :, 1:] = f[:, :, :-1]; zm[:, :, 0] = f[:, :, 0]
    zp = np.empty_like(f); zp[:, :, :-1] = f[:, :, 1:]; zp[:, :, -1] = f[:, :, -1]
    return (xm + xp + np.roll(f, 1, 1) + np.roll(f, -1, 1) + zm + zp) / 6.0


def _diff3(f: np.ndarray) -> np.ndarray:
    """Resolution-independent diffusion — one timestep's worth, whatever the lattice size.

    Verified: with this scaling the coupled oral->artery readouts converge (rupture_index spread
    across dim 10-24 falls from 56.9% to 6.3%, endothelial activation 51.9% -> 4.9%, oral efflux
    34.0% -> 9.1%) at negligible cost, because the stencil is cheap next to the reaction terms.
    """
    for _ in range(diffusion_iters(f.shape[2])):
        f = _stencil(f)
    return f


@dataclass
class ECMSpec:
    """A tissue's extracellular matrix — characterised, then quantified."""
    tissue: str
    composition: str                 # what it is made of (the characterisation)
    dominant: list[str]              # dominant polymers
    base_density: float              # baseline matrix density m₀ (0..1) before cells add to it
    alpha: float                     # porosity coefficient: θ = 1/(1+α·m)
    tortuosity: float                # λ (≥1); D_eff = D₀·θ/λ²
    fluid_fraction: float            # φ_w — ECS volume fraction
    produced_by: list[str]           # which cells secrete it
    binds: dict[str, float]          # field → bind rate (matrix-sequestered information reservoir)
    enzymes: str                     # who remodels / degrades it (description)
    stiffness_kPa: float             # mechanical cue
    connectors: str                  # the physical channels of this ECS
    production_rate: float = 0.4     # how fast producers build it
    enzyme_name: str = ""            # the degrading protease (structured)
    enzyme_producers: list[str] = field(default_factory=list)   # cells that secrete it
    enzyme_rate: float = 0.0         # matrix-degradation rate
    enzyme_driver: str = "expression"  # what upregulates it: 'inflammation' | 'expression'
    grounding: str = ""              # honesty note on the numbers


#: Obstruction coefficient in D/D0 = exp(-k * m * r^2), r in nm, m the matrix density.
#:
#: This replaced a constant I invented. The phage layer used theta^3, a guess marked [[VERIFY]],
#: and the verification came back negative: Peulen & Wilkinson 2011 (Environ Sci Technol 45:3367,
#: PMID 21434601) measured dextrans, microspheres and Ag nanoparticles in a LIVE biofilm and found
#: exp(-k*r^2) — obstruction scaling, Gaussian in radius, not a power of porosity. Waters & Frank
#: 2009 (PMID 20514136) found obstruction models beat hydrodynamic-scaling ones head to head in
#: polymer hydrogels.
#:
#: The value is calibrated so a ~50 nm virion reproduces what theta^3 gave: over the matrix range
#: this engine actually occupies (m = 0.08-0.13) the two agree within 0.3%, so no phage result
#: moves. They diverge sharply in a DENSE matrix (ratio 0.51 at m=0.5, 0.11 at m=1.0), and there
#: the exponential is the measured shape and theta^3 was never anything but a guess. [[VERIFY the
#: coefficient against a real D(r) series]]
OBSTRUCTION_K = 2.2e-3

# CALIBRATION ATTEMPTED 2026-08-05 AGAINST ORAL DATA. IT FAILED, AND THE FAILURE IS INFORMATIVE.
#
# Two real measurements exist for this exact system, and I tried to fit K to them:
#   Thurnheer 2006 (Caries Res 40:163, PMID 16508276): replacing wild-type S. mutans with a gtfC-
#     mutant moved EPS volume fraction 0.51 -> 0.33 and raised the 70 kDa dextran diffusion
#     coefficient SIXTEEN-fold.
#   Thurnheer 2003 (Appl Environ Microbiol 69:1702, PMID 12620862): across 10-240 kDa, D is a
#     POWER LAW in molecular weight, and the retardation is attributed to TORTUOSITY, with
#     "selective binding of probes ... could not explain the observed extent of retardation".
#
# Three problems, in increasing order of seriousness.
#
# 1. THE FIT OVER-PREDICTS. Fitting exp(-K*phi*r^2) to the 16-fold ratio (r = 6 nm for 70 kDa
#    dextran) gives K = 0.43, which then predicts wild-type D/D0 = 3.9e-4 — a 2600-fold slowdown.
#    Thurnheer measured "markedly slower" but linear-in-time PENETRATION, and Stewart's review puts
#    biofilm De/Daq around 0.1-0.6 for small solutes. 4e-4 is exclusion, not retardation.
#
# 2. THE FORM MAY BE WRONG FOR ORAL BIOFILM. Peulen & Wilkinson found exp(-k*r^2) in a live
#    biofilm, which is why it is here. But Thurnheer's oral result is a POWER law in MW, and a
#    Gaussian in r is not a power law in r — only one of them can fit. The exponential's provenance
#    is a non-oral biofilm; the power law's is supragingival plaque, which is this system.
#    (Caveat I could not resolve: the 2003 abstract says D is "linearly proportional to the cube
#    roots of the molecular weights", i.e. D ~ MW^(+1/3), which would mean bigger diffuses FASTER.
#    Stokes-Einstein gives MW^(-1/3). PMC returns no full text for this paper, so the sign is
#    unresolved and I am not assuming it.)
#
# 3. THE UNITS DO NOT MATCH, AND THIS IS THE REAL BLOCKER. Thurnheer's phi is an EPS VOLUME
#    FRACTION, 0.33-0.51. This engine's m runs 0.078-0.133 and saturates at 1.2. If m were a volume
#    fraction it would be ~4x too low, so it is not one — it is an uncalibrated density. K therefore
#    CANNOT be transferred from phi to m without a conversion nobody has measured, and doing it
#    anyway would be a unit error wearing the costume of a calibration.
#
# CONCLUSION: K stays at the value that reproduces prior behaviour, and stays labelled as
# uncalibrated. The prerequisite for ANY transport calibration is giving m physical units — until
# the matrix density means something, no measurement can be fitted to it. That is a bigger and more
# useful piece of work than choosing between two functional forms. [[VERIFY — blocked on units]]


def obstruction_factor(m: np.ndarray, radius_nm: float) -> np.ndarray:
    """Size-selective transport penalty from matrix obstruction — 1.0 for a point solute.

    THE reason this is a separate factor rather than folded into porosity: theta^n welded
    size-selectivity to the same theta that governs small-solute transport, so the two could never
    be calibrated apart. Here a small molecule sees ~1.0 (a biofilm mesh does not sieve 0.5 nm) and
    a virion sees a real penalty, and each can be fitted to its own data.

    Example: obstruction_factor(m, 0.5) ~ 1.0 for a drug; obstruction_factor(m, 50) ~ 0.58 for phage
    """
    return np.exp(-OBSTRUCTION_K * np.asarray(m) * float(radius_nm) ** 2)


def porosity(m: np.ndarray, alpha: float) -> np.ndarray:
    return 1.0 / (1.0 + alpha * np.clip(m, 0.0, None))


def ecm_metrics(spec: ECMSpec, density: float | None = None) -> dict:
    """CALCULATE the transport consequences of a characterisation at a representative density."""
    m = spec.base_density if density is None else density
    theta = 1.0 / (1.0 + spec.alpha * m)
    d_eff = theta / spec.tortuosity ** 2                 # relative effective diffusivity (vs free water)
    return {
        "matrix_density": round(m, 3),
        "porosity_theta": round(theta, 3),
        "tortuosity_lambda": spec.tortuosity,
        "D_eff_factor": round(d_eff, 3),                 # fraction of free-water diffusivity
        "fluid_fraction": spec.fluid_fraction,
        "stiffness_kPa": spec.stiffness_kPa,
    }


def describe(spec: ECMSpec) -> str:
    """A grounded one-paragraph answer for the Q&A layer."""
    m = ecm_metrics(spec)
    binds = ", ".join(spec.binds) or "—"
    return (f"ECM here ({spec.tissue}): {spec.composition} Dominant: {', '.join(spec.dominant)}. "
            f"Computed transport — porosity θ={m['porosity_theta']}, tortuosity λ={spec.tortuosity}, "
            f"effective diffusivity {m['D_eff_factor']}× free water; stiffness ≈{spec.stiffness_kPa} kPa; "
            f"sequesters {binds}; remodelled by {spec.enzymes}. Connectors: {spec.connectors}")


# ===================================================================== per-tissue characterisations
BIOFILM_ECM = ECMSpec(
    tissue="oral biofilm",
    composition=("a hydrated EPS gel — exopolysaccharide (streptococcal Gtf glucans, glucose/mannose "
                 "homopolymers) + extracellular DNA (eDNA) + secreted proteins + lipids; ~80–90% water "
                 "threaded by water channels."),
    dominant=["glucan/EPS", "eDNA", "protein"],
    base_density=0.05, alpha=2.0, tortuosity=1.3, fluid_fraction=0.85,
    produced_by=[],                       # [] = all bacteria secrete it
    binds={"heme": 0.3},                  # heme & cations (Ca²⁺) crosslink and are stored in the matrix
    enzymes="bacterial glucanases/DNases remodel it; P. gingivalis gingipains degrade host proteins in situ",
    stiffness_kPa=0.5,
    connectors="water channels (nutrient highways) · eDNA structural cables · Ca²⁺ crosslinks",
    production_rate=0.5,
    enzyme_name="gingipains (Kgp / RgpA-B)", enzyme_producers=["Pg", "Td", "Tf"],
    enzyme_rate=0.25, enzyme_driver="expression",
    grounding="EPS/eDNA composition well-established; porosity/stiffness order-of-magnitude estimates [[VERIFY]]",
)

ARTERY_ECM = ECMSpec(
    tissue="artery / atheroma",
    composition=("dense fibrous wall ECM — fibrillar collagen I & III (the cap's tensile strength), elastin "
                 "lamellae, and proteoglycans (versican, biglycan, decorin) + hyaluronan, over a collagen-IV/"
                 "laminin basement membrane under the endothelium."),
    dominant=["collagen I/III", "elastin", "proteoglycan"],
    base_density=0.45, alpha=1.5, tortuosity=2.2, fluid_fraction=0.25,
    produced_by=["SMC"],                  # smooth muscle deposits collagen → the fibrous cap thickens
    binds={},                             # proteoglycan LDL RETENTION is realised as low D_eff trapping (below)
    enzymes="MMPs degrade the cap (→ rupture); cathepsins",
    stiffness_kPa=40.0,
    connectors="basement membrane = the endothelial gate · proteoglycan LDL-trapping · internal elastic lamina",
    production_rate=0.25,
    enzyme_name="MMP-2/9", enzyme_producers=["Foam", "Mono"],
    enzyme_rate=0.5, enzyme_driver="inflammation",       # macrophage MMPs thin the cap → rupture
    grounding="vascular ECM + proteoglycan–LDL retention well-established; numerics estimates [[VERIFY]]",
)

BRAIN_ECM = ECMSpec(
    tissue="brain / neurovascular unit",
    composition=("an unusual soft gel — almost no fibrillar collagen; a hyaluronan backbone + lecticans "
                 "(aggrecan, brevican, neurocan, versican) + tenascins/link proteins; perineuronal nets (PNNs) "
                 "around neurons; a collagen-IV/laminin/agrin/perlecan basement membrane at the BBB."),
    dominant=["hyaluronan", "lecticans (CSPGs)", "perineuronal nets"],
    base_density=0.12, alpha=1.2, tortuosity=1.6, fluid_fraction=0.20,
    produced_by=["Astro"],                # astrocytes make hyaluronan/lecticans
    binds={},                             # Aβ clearance is glymphatic (modelled separately), not matrix-bound here
    enzymes="MMPs & ADAMTS remodel lecticans/PNNs; plasmin",
    stiffness_kPa=0.5,
    connectors="glymphatic perivascular spaces (AQP4 bulk flow) = the clearance highway · PNNs · astrocyte gap junctions",
    production_rate=0.2,
    enzyme_name="MMP-9 / ADAMTS", enzyme_producers=["Micro"],
    enzyme_rate=0.4, enzyme_driver="inflammation",       # microglial MMP-9 degrades the BBB basement membrane
    grounding="brain ECS tortuosity λ≈1.6 and volume fraction ~0.2 are measured (Nicholson/Syková) [[VERIFY]]; composition established",
)


LIVER_ECM = ECMSpec(
    tissue="liver / hepatic lobule",
    composition=("the space of Disse — normally a sparse, low-density basement-membrane-like matrix "
                 "(collagen IV, laminin, fibronectin) under fenestrated sinusoidal endothelium; on hepatic "
                 "stellate-cell activation, fibrillar collagen I/III is deposited → fibrosis and, ultimately, "
                 "cirrhosis."),
    dominant=["collagen IV (normal)", "collagen I/III (fibrotic)", "laminin"],
    base_density=0.05, alpha=1.5, tortuosity=1.8, fluid_fraction=0.40,
    produced_by=["Stel"],                 # activated stellate cells deposit the fibrotic collagen
    binds={},
    enzymes="MMP-2/9 vs TIMP balance governs fibrosis progression/regression",
    stiffness_kPa=2.0,                     # soft normally; stiffens markedly with fibrosis
    connectors="fenestrated sinusoids (LSEC) · space of Disse · portal→central O2 zonation",
    production_rate=2.5,
    # in NASH the MMP/TIMP balance tips toward TIMP (net deposition) — so degradation is slow vs stellate build
    enzyme_name="MMP-2/9 (Kupffer, TIMP-restrained)", enzyme_producers=["Kup"], enzyme_rate=0.05, enzyme_driver="inflammation",
    grounding="space-of-Disse composition + stellate-driven fibrosis well-established; numerics estimates [[VERIFY]]",
)


def apply_ecm(eng, dt: float, solute_keys, spec: ECMSpec) -> tuple[np.ndarray, np.ndarray]:
    """Cells SECRETE this tissue's matrix; the matrix then GATES transport (D_eff = θ/λ²) of the named
    solutes, and SEQUESTERS its bound species (the information reservoir). Returns (matrix, porosity)."""
    grid = eng.grid
    comp = grid.composition
    m = eng.extra.get("ecm")
    if m is None:
        m = eng.extra["ecm"] = np.full(grid.shape, spec.base_density)
    # production by the characterised producer cells (∅ = all cells)
    if spec.produced_by:
        prod = sum(comp[..., eng.species.index(c)] * eng.expression[..., eng.species.index(c)]
                   for c in spec.produced_by if c in eng.species)
    else:
        prod = (comp * eng.expression).sum(axis=3)
    m += dt * spec.production_rate * prod * np.clip(1.2 - m, 0.0, None)
    # enzyme remodelling: producer cells secrete proteases that DEGRADE the matrix (gingipains cut the
    # host barrier; MMPs thin the fibrous cap / the BBB). Activity scales with its driver.
    if spec.enzyme_producers and spec.enzyme_rate > 0:
        drv = eng.extra.get("inflammation") if spec.enzyme_driver == "inflammation" else None
        E = np.zeros(grid.shape)
        for c in spec.enzyme_producers:
            if c in eng.species:
                ci = eng.species.index(c)
                E += comp[..., ci] * (drv if drv is not None else eng.expression[..., ci])
        E *= spec.enzyme_rate
        eng.extra["enzyme"] = E
        m -= dt * E * m                                            # the matrix is cut where the enzyme acts
    np.clip(m, 0.0, None, out=m)
    theta = porosity(m, spec.alpha)
    d_eff = theta / spec.tortuosity ** 2                           # the transport bottleneck
    for k in solute_keys:
        fld = eng.field(k)
        fld[:] = fld + d_eff * (_diff3(fld) - fld)
    for sp_key, rate in spec.binds.items():
        try:
            free = eng.field(sp_key)
        except KeyError:
            continue
        bk = "ecm_bound_" + sp_key
        bnd = eng.extra.get(bk)
        if bnd is None:
            bnd = eng.extra[bk] = np.zeros(grid.shape)
        flux = dt * rate * m * free - dt * 0.05 * bnd
        free -= flux
        bnd += flux
        np.clip(free, 0.0, None, out=free)
    return m, theta

