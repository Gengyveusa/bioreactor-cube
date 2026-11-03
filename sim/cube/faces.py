"""The surfaces module — the six faces of the Cube, where all the action happens.

Each face is a boundary condition: a well-defined, calculable surface across which flux enters or
leaves the biofilm volume. v1 applies Dirichlet boundary values per face kind; flux/transport and
the per-face influence gradients build on this.

    ENAMEL_PELLICLE  (x=0)     solid hard tissue: anoxic wall, Ca/PO4 supersaturation feeding
                              biomineralization, the acquired-pellicle adhesion substrate
    SOFT_TISSUE      (x=nx-1)  host interface: serum pH, low O2 (tissue sink), heme on bleeding,
                              and the inflow/outflow of the host-immunity + inflammatory couplings
    ORAL_CAVITY      (z=nz-1)  open mouth: saliva flow, pH, dietary nitrate, sugar, O2; metabolite
                              washout (clearance)
    ADJACENT_BIOFILM (y=0,ny-1) continuation into surrounding biofilm (periodic y axis)
    PERIODIC         (z=0)     deep unit-cell closure
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from sim.cube.chemistry import BufferSpec
from sim.cube.grid import CubeGrid


class FaceKind(Enum):
    ENAMEL_PELLICLE = "enamel_pellicle"
    SOFT_TISSUE = "soft_tissue"
    ORAL_CAVITY = "oral_cavity"
    ADJACENT_BIOFILM = "adjacent_biofilm"
    PERIODIC = "periodic"


@dataclass(frozen=True)
class Face:
    """One face of the cube: an axis (0=x, 1=y, 2=z) and a side (0 = low index, 1 = high index)."""

    kind: FaceKind
    axis: int
    side: int

    def voxel_slice(self, shape: tuple[int, int, int]) -> tuple:
        """The index tuple selecting this face's boundary voxels from a (nx, ny, nz) field."""
        idx = 0 if self.side == 0 else shape[self.axis] - 1
        sl: list = [slice(None), slice(None), slice(None)]
        sl[self.axis] = idx
        return tuple(sl)

    @property
    def label(self) -> str:
        return f"{self.kind.value}" if self.kind is not FaceKind.ADJACENT_BIOFILM else f"{self.kind.value}_{self.side}"


# The six faces of the cube — the surfaces.
FACES: tuple[Face, ...] = (
    Face(FaceKind.ENAMEL_PELLICLE,  axis=0, side=0),   # x-  hard tissue + pellicle + biomineralization
    Face(FaceKind.SOFT_TISSUE,      axis=0, side=1),   # x+  host: keratinocyte/fibroblast/blood/immune
    Face(FaceKind.ADJACENT_BIOFILM, axis=1, side=0),   # y-  biofilm continuation (periodic)
    Face(FaceKind.ADJACENT_BIOFILM, axis=1, side=1),   # y+  biofilm continuation (periodic)
    Face(FaceKind.PERIODIC,         axis=2, side=0),   # z-  deep unit-cell closure
    Face(FaceKind.ORAL_CAVITY,      axis=2, side=1),   # z+  exposed to saliva / pH / nutrients / O2
)


@dataclass(frozen=True)
class FaceConditions:
    """Boundary values applied at each surface. PLACEHOLDERS — reuse the sim config later (design §13)."""

    oral_O2: float = 8.0          # saliva is air-equilibrated
    oral_pH: float = 6.8
    oral_nitrate: float = 0.5     # enterosalivary nitrate delivery
    # Salivary NITRITE arriving at the mouth face — the OUTPUT of the tongue-dorsum nitrate
    # reductases, not a dietary input, which is why it is a separate boundary from nitrate.
    # None means "not measured": the boundary is left alone and nitrite stays whatever the cube's
    # own nitrogen chemistry makes of it. Defaulting this to a number would pin the face and turn
    # an unmeasured patient into a Dirichlet sink, which is the bug this project already fixed
    # once for the fermentation acids (see oral_clearance).
    oral_nitrite: Optional[float] = None
    oral_sugar: float = 0.0       # Stephan feed (event-driven)
    # Salivary UREA — the substrate for the only base-generating reaction in plaque. Whole saliva
    # runs 3-5 mmol/L (Sissons 1998, PMID 9438574). Expressed on the proteolytic millimolar scale,
    # so 4 mM / PEPTIDE_MM_PER_UNIT. This is what makes a Stephan curve RECOVER.
    oral_urea: float = 4.0 / 60.0
    serum_pH: float = 7.4         # soft-tissue / serum face
    serum_O2: float = 1.0         # low — tissue is a relative O2 sink
    serum_heme: float = 0.0       # rises with bleeding (BOP)
    # Gingival crevicular fluid — a serum transudate carrying protein into the pocket. This is the
    # substrate the proteolytic red complex actually eats, and it arrives from the HOST side, which
    # is why the deep pocket is a protein economy while the plaque surface is a sugar economy.
    # GCF flow rises steeply with inflammation, so this scales with the same signal as heme.
    serum_peptide: float = 0.0
    # HEMOPROTEIN, not free heme. Blood delivers haemoglobin; heme has to be prised out of it, and
    # oxyhaemoglobin must first be oxidised to methaemoglobin before the heme will come (Smalley
    # 2011, PLoS One 6:e17182, PMID 21390208). Modelling the bleeding face as a free-heme source
    # hands the red complex the very resource its proteases exist to obtain, which is why the
    # keystone could never be load-bearing. serum_heme stays for the free fraction; this is the pool
    # that has to be worked for.
    serum_hemoprotein: float = 0.0
    enamel_Ca: float = 1.0        # supersaturation driving biomineralization
    enamel_PO4: float = 1.0
    # First-order salivary clearance of fermentation acids at the mouth-facing surface, per unit
    # time. This used to be a hard Dirichlet zero — an INFINITE sink that pinned the acid pool to
    # zero at the boundary every step and drained the whole cube toward it by diffusion, so the
    # community could never out-produce its own clearance and plaque pH could not fall far.
    # Finite clearance is what lets acid accumulate faster than saliva removes it. [[VERIFY]]
    oral_clearance: float = 0.30
    # THE MASS-TRANSFER BOUNDARY — a Biot number, dimensionless, None meaning "infinite".
    #
    # A Dirichlet face asserts that an infinitely well-stirred reservoir is pressed against the
    # biofilm: whatever the community does, the surface concentration never moves. That is the
    # well-stirred limit the oral in vitro literature uses as its NEGATIVE CONTROL, and it is what
    # this model has been assuming.
    #
    # Saliva is not that. It is a ~0.1 mm film creeping at 0.8-8 mm/min, site by site (Dawes et al.,
    # PMID 2531177), and the consequences are measured: clearance half-time rises with plaque length
    # and falls with film velocity, and a slower film gives a DEEPER pH minimum (Macpherson & Dawes,
    # PMID 1918572; Sissons, PMID 1466639).
    #
    # The regime is set by the Biot number Bi = k_L * h / D_eff, external resistance over internal.
    # For a periodontal pocket, GCF flow gives k_L ~ 1e-7 to 1e-6 m/s, so Bi ~ 0.2 to 4 — the
    # external layer is comparable to or SLOWER than diffusion through the biofilm. At Bi = 0.2 the
    # steady surface concentration is about 0.17 of bulk, so a Dirichlet face overstates the driving
    # concentration roughly six-fold. That is not a correction one waves off.
    #
    # Quasi-steady flux balance at the surface voxel, k_L(C_bulk - C_s) = (D_eff/h)(C_s - C_1):
    #
    #     C_s = (Bi * C_bulk + C_1) / (Bi + 1)
    #
    # Bi -> infinity recovers C_s = C_bulk exactly, which is why None (Dirichlet) stays the default
    # and every existing result is reproduced bit for bit. Bi -> 0 gives C_s = C_1, a sealed face.
    oral_biot: Optional[float] = None
    serum_biot: Optional[float] = None
    # BULK clearance — acid removed throughout the volume rather than only at the mouth-facing
    # layer. Zero by default, because a 6 mm periodontal pocket is genuinely diffusion-isolated at
    # its base and boundary-only clearance is right for it.
    #
    # It exists because the Stephan curve is a SUPRAGINGIVAL phenomenon measured in plaque roughly
    # 100 um thick, where McNee & Geddes (1982, PMID 6818937) put the acid clearance half-time at
    # about SIX SECONDS — every part of that film is in fast diffusive contact with saliva. Since
    # diffusion time scales as L^2/D, the same acid needs ~3600x longer to leave a 6 mm pocket.
    # Testing a thin-plaque phenomenon in a deep-pocket geometry is a geometry mismatch, not a
    # transport bug, and this is the knob that lets the thin case be represented.
    bulk_clearance: float = 0.0
    # Salivary buffer pools the pH solve runs against. Patient-specific when a chairside buffer
    # test was recorded; the default reproduces the shipped literature values exactly.
    buffers: BufferSpec = BufferSpec()


_WASHOUT = ("lactate", "acetate", "propionate", "butyrate", "H2S")

#: Solutes delivered across the mouth-facing film. These are the ones a finite salivary film
#: actually rate-limits; pH is excluded because it is solved from charge balance each step rather
#: than transported, so imposing a mixed condition on it would be applying transport physics to a
#: derived quantity.
_ORAL_DELIVERED = ("O2", "nitrate", "nitrite", "sugar", "urea")
_SERUM_DELIVERED = ("O2", "heme", "peptide", "hemoprotein")


#: Free-solution diffusivity of a small solute at 37 C, m^2/s. Lactate in water is ~1.0-1.1e-9;
#: this is the reference the biofilm's relative D_eff is measured against.
D_WATER = 1.0e-9

#: Effective diffusivity inside plaque, m^2/s. McNee & Geddes 1982 (PMID 6818937) measured lactate
#: at 4.8e-6 cm^2/s = 4.8e-10 m^2/s in human dental plaque, 2-5x slower than free water.
D_PLAQUE = 4.8e-10


def biot_from_film(film_velocity_mm_min: float, *, plaque_length_mm: float = 2.0,
                   film_thickness_um: float = 100.0, biofilm_depth_mm: float = 0.5,
                   d_eff: float = D_PLAQUE) -> float:
    """Biot number for a SUPRAGINGIVAL surface bathed by a moving salivary film.

    Dawes measured the film in vivo, site by site: 0.8 mm/min at the upper anterior labial surface
    up to ~8 mm/min lingual to the lower incisors, over a film ~0.1 mm thick (PMID 2531177).

    A fluid parcel is in contact with the plaque for t = L/v, over which the concentration boundary
    layer grows as sqrt(D*t) — capped at the film thickness, since the layer cannot exceed the film
    that carries it. That gives k_L = D_water/delta, and Bi = k_L*H/D_eff.

    A result worth stating because it is easy to misread: across the WHOLE Dawes velocity range the
    boundary layer sqrt(D*t) exceeds the 0.1 mm film, so it clamps and Bi comes out ~10 regardless
    of velocity. That is not the helper ignoring its argument — it means the film is thin enough and
    slow enough that diffusion crosses it completely, so the residual resistance is set by film
    THICKNESS alone.

    Velocity still matters enormously to a Stephan curve, but through a different mechanism:
    advective removal of acid, not diffusive delivery of substrate. Delivery is this Biot number;
    removal is FaceConditions.oral_clearance. Dawes' velocity dependence lives in the second one,
    and conflating them would double-count velocity.

    Example: biot_from_film(0.8) -> ~10, so a supragingival film is NOT mass-transfer limited
    """
    if film_velocity_mm_min <= 0:
        return 0.0
    v = film_velocity_mm_min * 1e-3 / 60.0                 # mm/min -> m/s
    length = plaque_length_mm * 1e-3
    contact_s = length / v
    delta = min(film_thickness_um * 1e-6, np.sqrt(D_WATER * contact_s))
    k_l = D_WATER / max(delta, 1e-9)
    return float(k_l * (biofilm_depth_mm * 1e-3) / max(d_eff, 1e-15))


def biot_from_gcf(gcf_flow_ul_min: float, *, site_area_mm2: float = 20.0,
                  pocket_depth_mm: float = 0.5, d_eff: float = D_PLAQUE) -> float:
    """Biot number for a SUBGINGIVAL pocket, where the bathing fluid is crevicular flow.

    This is the regime that actually matters here, and it is a different one. GCF runs ~0.05-0.2
    uL/min at healthy sites and up to ~1-3 uL/min when inflamed, over a site of order 10-50 mm^2, so
    k_L = Q/A is four to five orders of magnitude BELOW the salivary-film conductance. The pocket is
    externally mass-transfer limited in a way an exposed tooth surface is not.

    Example: biot_from_gcf(0.1) -> well below 1, so the external layer is rate-limiting
    """
    if gcf_flow_ul_min <= 0:
        return 0.0
    q = gcf_flow_ul_min * 1e-9 / 60.0                      # uL/min -> m^3/s
    area = site_area_mm2 * 1e-6                            # mm^2 -> m^2
    k_l = q / max(area, 1e-12)                             # m/s
    return float(k_l * (pocket_depth_mm * 1e-3) / max(d_eff, 1e-15))


def _interior_slice(face: Face, shape) -> tuple:
    """The voxel one step INTO the volume from a face — C_1 in the flux balance."""
    idx = 1 if face.side == 0 else shape[face.axis] - 2
    idx = max(0, min(idx, shape[face.axis] - 1))
    sl: list = [slice(None), slice(None), slice(None)]
    sl[face.axis] = idx
    return tuple(sl)


def _mixed_face(grid: CubeGrid, name: str, sl, interior_sl, bulk: float, biot) -> None:
    """Apply C_s = (Bi*C_bulk + C_1)/(Bi + 1) at a face, or Dirichlet when biot is None."""
    f = grid.get_field(name)
    if biot is None:
        f[sl] = bulk
        return
    bi = max(float(biot), 0.0)
    f[sl] = (bi * bulk + f[interior_sl]) / (bi + 1.0)


def apply_face_boundaries(grid: CubeGrid, cond: FaceConditions = FaceConditions(),
                          dt: float = 0.0) -> None:
    """Set the boundary-voxel chemistry at each of the six faces (Dirichlet). This is the surfaces
    module: where flux enters/leaves the cube. ADJACENT_BIOFILM / PERIODIC faces impose no source —
    continuity is carried by the periodic/bounded neighbor stencil."""
    for face in FACES:
        sl = face.voxel_slice(grid.shape)
        if face.kind is FaceKind.ORAL_CAVITY:
            inner = _interior_slice(face, grid.shape)
            bulk = {"O2": cond.oral_O2, "nitrate": cond.oral_nitrate,
                    "sugar": cond.oral_sugar, "urea": cond.oral_urea}
            if cond.oral_nitrite is not None:
                bulk["nitrite"] = cond.oral_nitrite
            for name in _ORAL_DELIVERED:
                if name in bulk:
                    _mixed_face(grid, name, sl, inner, bulk[name], cond.oral_biot)
            # pH is SOLVED from charge balance, not transported, so a mass-transfer condition on it
            # would be applying transport physics to a derived quantity. It stays pinned.
            grid.get_field("pH")[sl] = cond.oral_pH
            for name in _WASHOUT:
                f = grid.get_field(name)
                if dt > 0.0 and cond.oral_clearance > 0.0:
                    f[sl] *= max(0.0, 1.0 - dt * cond.oral_clearance)   # finite saliva clearance
                elif dt <= 0.0:
                    f[sl] = 0.0        # relaxation/prepare passes have no clock: keep the old sink
                if dt > 0.0 and cond.bulk_clearance > 0.0:
                    f *= max(0.0, 1.0 - dt * cond.bulk_clearance)       # thin-film: all of it clears
        elif face.kind is FaceKind.SOFT_TISSUE:
            inner = _interior_slice(face, grid.shape)
            grid.get_field("pH")[sl] = cond.serum_pH
            _mixed_face(grid, "O2", sl, inner, cond.serum_O2, cond.serum_biot)
            _mixed_face(grid, "heme", sl, inner, cond.serum_heme, cond.serum_biot)
            if cond.serum_peptide > 0.0:
                _mixed_face(grid, "peptide", sl, inner, cond.serum_peptide, cond.serum_biot)
            if cond.serum_hemoprotein > 0.0:
                _mixed_face(grid, "hemoprotein", sl, inner, cond.serum_hemoprotein,
                            cond.serum_biot)
        elif face.kind is FaceKind.ENAMEL_PELLICLE:
            grid.get_field("Ca")[sl] = cond.enamel_Ca
            grid.get_field("PO4")[sl] = cond.enamel_PO4
            grid.get_field("O2")[sl] = 0.0              # solid, anoxic at the wall


def face_distance(grid: CubeGrid, face: Face) -> np.ndarray:
    """(nx, ny, nz) lattice distance from every voxel to a face — the substrate for the per-surface
    influence gradients ('where you are in relation to each surface')."""
    coords = np.indices(grid.shape)[face.axis].astype(np.float64)
    n = grid.shape[face.axis]
    return coords if face.side == 0 else (n - 1 - coords)


def face_influence(grid: CubeGrid, face: Face, decay: float = 4.0) -> np.ndarray:
    """A 0..1 influence field for a face, decaying exponentially with distance (each surface
    influences nearby voxels most). decay is in voxels."""
    return np.exp(-face_distance(grid, face) / decay)

