"""Apply an intake to the cube — the four insertion points.

A panel lands in four structurally different places, and conflating them is the easiest way to get
this wrong:

  1. COMPOSITION  -> ``seed_composition(grid, abundances)``   (initial condition)
  2. SALIVA       -> ``FaceConditions``                        (boundary condition, oral face)
  3. GEOMETRY     -> grid extent / voxel size, and the heme boundary (bleeding)
  4. HOST MARKERS -> carried through for psi and validation; they seed nothing

Only (1) and (2) change what the simulation computes from the first step. (3) changes what volume of
tissue the cube represents. (4) is a readout the model should be checked against, not an input.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from sim.cube.chemistry import bicarbonate_from_buffer_test
from sim.cube.faces import FaceConditions
from sim.cube.host import HostState
from sim.cube.grid import CubeGrid
from sim.cube.sample import seed_composition
from sim.intake.priors import complete_composition
from sim.intake.schema import PatientIntake, Provenance, Value

# The cube is a representative volume element, not the whole pocket. When probing depth is known we
# scale the voxel size so the lattice spans that depth; nz stays fixed so compute stays predictable.
DEFAULT_POCKET_DEPTH_MM = 3.0

# Total bacterial load the shipped relative composition corresponds to, in the units qPCR panels
# report (genome equivalents/mL of pooled subgingival sample). Subgingival load spans orders of
# magnitude between a healthy sulcus and a deep diseased pocket, and the panels do not agree on
# absolute calibration, so this is an ANCHOR chosen to sit mid-range rather than a measured
# population mean. Only the RATIO load/REFERENCE is used. [[VERIFY against a real assay's own
# reference range before any clinical reading]]
REFERENCE_TOTAL_LOAD = 1.0e6

# How far a measured load may scale initial biomass. Load enters as a bounded multiplier because
# the growth terms are not calibrated across decades of biomass: an unclamped 100x would not
# produce "a cube with 100x the bugs", it would produce a stiff system leaving the regime any of
# this was fitted in. Clamping keeps load INFLUENTIAL and keeps it HONEST about being an anchor.
LOAD_FACTOR_RANGE = (0.25, 4.0)


def recession_mm(intake: PatientIntake) -> float:
    """Attachment already lost that is NOT current pocket: ``CAL - PD``, floored at zero.

    This is the whole reason CAL earns its place next to probing depth. PD is the pocket you can
    measure today and it sets the cube's geometry. CAL is cumulative history. A 5 mm pocket with
    5 mm CAL is a site that is deep but intact; a 5 mm pocket with 8 mm CAL has already lost 3 mm
    of attachment and is a fundamentally more destroyed site presenting the same probe reading.
    Without CAL those two are indistinguishable to the model.

    Returns 0.0 when either measurement is missing, so nothing changes for a PD-only chart.

    Example: recession_mm(from_myperiopath({}, probing_depth_mm=5.0, clinical_attachment_mm=8.0))
             -> 3.0
    """
    site = intake.primary_site
    if site.clinical_attachment_mm is None or site.probing_depth_mm is None:
        return 0.0
    return max(0.0, site.clinical_attachment_mm.value - site.probing_depth_mm.value)


def to_host_state(intake: PatientIntake) -> HostState:
    """The fifth insertion point: host markers + recession -> the Layer-6 axes.

    Example: to_host_state(from_myperiopath({}, ammp8=25.0)).tissue_breakdown > 0  -> True
    """
    return HostState.from_markers(intake.host, recession_mm=recession_mm(intake))


def load_factor(intake: PatientIntake) -> float:
    """Multiplier on initial biomass implied by a measured total bacterial load.

    Returns 1.0 when load was not measured, so an intake without it behaves exactly as before.
    The schema keeps ``total_load`` separate from ``composition`` precisely because two mouths with
    identical ratios and a hundredfold difference in load are not the same mouth; this is the
    function that finally makes that true in the simulation rather than only in the docstring.

    Example: load_factor(from_myperiopath({"Pg": 1e5}, total_load=4e6)) -> 4.0
    """
    site = intake.primary_site
    if site.total_load is None or site.total_load.value <= 0:
        return 1.0
    raw = site.total_load.value / REFERENCE_TOTAL_LOAD
    lo, hi = LOAD_FACTOR_RANGE
    return float(min(hi, max(lo, raw)))


def to_abundances(intake: PatientIntake, *, complete: bool = True) -> dict:
    """Build the ``{species_id: float}`` dict ``seed_composition`` consumes.

    Measured species pass through; the rest are completed from conditional priors unless
    ``complete=False``, in which case unmeasured species are simply absent (and therefore zero,
    which is almost never what you want — the cube has no immigration, so a zero stays zero).

    Example: to_abundances(from_myperiopath({"Pg": 4.2e5}))["Rm"] > 0  -> True
    """
    site = intake.primary_site
    if complete:
        full = complete_composition(site.composition)
    else:
        total = sum(v.value for v in site.composition.values()) or 1.0
        full = {k: Value(v.value / total, v.provenance, v.source, "relative", v.note)
                for k, v in site.composition.items()}
    # Composition sets the RATIOS; measured total load sets the absolute scale. Applied here rather
    # than inside seed_composition so the spatial fitness field, and therefore seeding_losses, sees
    # exactly the biomass the patient's panel implies.
    scale = load_factor(intake)
    return {k: v.value * scale for k, v in full.items()}


def composition_provenance(intake: PatientIntake) -> dict:
    """``{species_id: Provenance}`` parallel to :func:`to_abundances`, for the viewer to render.

    This is what keeps a mostly-inferred cube from looking like a measured one.
    """
    return {k: v.provenance for k, v in complete_composition(intake.primary_site.composition).items()}


def to_face_conditions(intake: PatientIntake,
                       base: Optional[FaceConditions] = None) -> FaceConditions:
    """Map salivary chemistry and bleeding onto the six-face Dirichlet boundaries.

    Unmeasured entries keep the engine default, so a panel with no saliva data produces exactly
    today's behaviour rather than a silently different one.
    """
    b = base or FaceConditions()
    sal = intake.saliva
    site = intake.primary_site

    oral_pH = sal.pH.value if sal.pH else b.oral_pH
    oral_nitrate = sal.nitrate.value if sal.nitrate else b.oral_nitrate

    # A measured salivary buffer test replaces the BICARBONATE pool only. The plaque fixed buffer
    # (chemistry.FIXED_BUFFER_MM) is deliberately untouched: roughly 90% of plaque buffering is
    # immobile charge on bacterial surfaces and matrix (Shellis & Dibdin 1988), which no chairside
    # test measures. Putting a saliva number into that constant would be a category error - two
    # different compartments with different chemistry.
    # Salivary nitrite is the readout of the nitrate-reducing tongue flora — the step a chlorhexidine
    # rinse ablates and the one the BP literature hangs on (Kapil 2013 PMID 23183324; Vanhatalo 2018
    # PMID 29807159; Willmott 2023 PMID 37702047). The engine has modelled a nitrite field and full
    # nitrogen chemistry all along; until now the measurement simply never reached it.
    oral_nitrite = sal.nitrite.value if sal.nitrite else b.oral_nitrite

    buffers = b.buffers
    if sal.buffering_capacity is not None:
        bicarb = bicarbonate_from_buffer_test(sal.buffering_capacity.value,
                                              sal.buffering_capacity.unit)
        buffers = replace(buffers, bicarb_mM=bicarb)
    # Bleeding on probing is the clinical proxy for serum/heme ingress at the soft-tissue face.
    if site.bleeding_on_probing is not None:
        bop = max(0.0, min(1.0, site.bleeding_on_probing.value))
        serum_heme = bop * 1.0
    else:
        serum_heme = b.serum_heme

    # Host couplings land LAST so they modify the clinically-derived surfaces rather than being
    # overwritten by them: inflammation adds to whatever heme BOP already implied, and tissue
    # breakdown deepens the anoxic niche on top of the measured geometry.
    return to_host_state(intake).apply_to_conditions(FaceConditions(
        oral_O2=b.oral_O2,
        oral_pH=oral_pH,
        oral_nitrate=oral_nitrate,
        oral_nitrite=oral_nitrite,
        oral_sugar=b.oral_sugar,
        serum_pH=b.serum_pH,
        serum_O2=b.serum_O2,
        serum_heme=serum_heme,
        enamel_Ca=b.enamel_Ca,
        enamel_PO4=b.enamel_PO4,
        buffers=buffers,
    ))


def to_grid_shape(intake: PatientIntake, *, dim: int = 40) -> dict:
    """Grid construction kwargs implied by the site geometry.

    Probing depth sets the voxel edge length so the lattice spans the pocket: a 6 mm pocket at
    nz=40 gives 150 um voxels, a 3 mm pocket gives 75 um. The cube keeps a fixed voxel COUNT so
    runtime stays predictable; what changes is the volume each voxel represents.

    Example: to_grid_shape(intake, dim=40)["cell_size_um"]  -> 150.0 for a 6 mm pocket
    """
    site = intake.primary_site
    depth_mm = site.probing_depth_mm.value if site.probing_depth_mm else DEFAULT_POCKET_DEPTH_MM
    depth_mm = max(0.5, depth_mm)
    return {
        "nx": dim, "ny": dim, "nz": dim,
        "cell_size_um": (depth_mm * 1000.0) / float(dim),
    }


def seeding_losses(grid, abundances: dict) -> dict:
    """Species we asked for that the spatial fitness field refused to place. Returns {id: requested}.

    This is a real trap rather than a hypothetical one. ``seed_composition`` multiplies each
    species' abundance by an O2/depth fitness, and the anaerobe term ``clip(1.2 - O2, 0, 1.2)``
    zeroes an anaerobe anywhere O2 exceeds 1.2. On a small lattice the O2 profile has no room to
    decay -- at nz=12 the deepest voxel is still ~2.0 -- so EVERY anaerobe, the whole red complex
    included, seeds to exactly zero. The engine has no immigration, so zero is absorbing: those
    species can never return. A patient's Pg would silently vanish before step one.

    Example: seeding_losses(grid, {"Pg": 0.2}) -> {"Pg": 0.2} on a too-small lattice
    """
    lost = {}
    for sid, want in abundances.items():
        if want <= 0:
            continue
        try:
            got = float(grid.composition[..., grid.species_index(sid)].sum())
        except (KeyError, ValueError):
            continue
        if got <= 0.0:
            lost[sid] = want
    return lost


def apply_intake(intake: PatientIntake, *, dim: int = 40, seed: int = 0,
                 complete: bool = True) -> dict:
    """Build a cube from a patient. Returns the grid, the conditions, and the honesty metadata.

    The returned dict carries ``coverage`` and ``provenance`` alongside the grid deliberately: any
    caller that renders this cube has everything it needs to say how much of it is really measured,
    and no excuse for presenting an inferred community as an observed one.

    Example
    -------
    >>> out = apply_intake(from_myperiopath({"Pg": 4.2e5, "Tf": 1e5}, probing_depth_mm=6.0))
    >>> out["coverage"].degraded
    True
    """
    abundances = to_abundances(intake, complete=complete)
    grid = CubeGrid(**to_grid_shape(intake, dim=dim))
    seed_composition(grid, abundances, seed=seed)

    lost = seeding_losses(grid, abundances)
    coverage = intake.coverage()
    if lost:
        coverage.warnings.append(
            "seeding lost {} species the panel asked for ({}) - the O2/depth fitness field placed "
            "no biomass for them at dim={}. The engine has no immigration, so they can never "
            "recover. Re-run at a larger dim (40 is the shipped default).".format(
                len(lost), ", ".join(sorted(lost)), dim))

    host_state = to_host_state(intake)
    return {
        "grid": grid,
        "conditions": to_face_conditions(intake),
        "abundances": abundances,
        "provenance": composition_provenance(intake),
        "coverage": coverage,
        "seeding_losses": lost,
        "host": intake.host,
        # The normalized Layer-6 axes actually applied, the per-species clearance they imply, and
        # the markers still honest to validate against. Pass ``mortality_mult`` to the engine via
        # sim.cube.host.attach_host, or read ``host_state.validation_targets()`` before scoring.
        "host_state": host_state,
        "mortality_mult": host_state.mortality_multipliers(grid.species_ids),
        "recession_mm": recession_mm(intake),
        "intake": intake,
    }

