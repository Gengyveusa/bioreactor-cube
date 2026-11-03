"""The intake contract — one typed object everything downstream reads.

Every measurement carries its own provenance. That is the load-bearing idea in this module: a cube
seeded from a pathogen panel is mostly *inferred*, and the difference between the 8 species someone
paid to measure and the 10 we filled from a prior must survive all the way to the viewer. If it does
not, a thin panel renders as a confident patient-specific result, which is the exact failure mode
this project has worked hardest to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional


class Provenance(str, Enum):
    """Where a value came from. Ordered weakest-last so ``min`` over a set degrades correctly."""

    MEASURED = "measured"   # a real assay on this patient
    DERIVED = "derived"     # computed from this patient's measurements (e.g. relative from absolute)
    PRIOR = "prior"         # inferred from population/ecology priors, conditioned on what was measured
    DEFAULT = "default"     # a model default; nothing patient-specific informed it

    @property
    def is_patient_specific(self) -> bool:
        """True when the value reflects THIS patient rather than a population.

        Example: Provenance.MEASURED.is_patient_specific -> True
        """
        return self in (Provenance.MEASURED, Provenance.DERIVED)


@dataclass(frozen=True)
class Value:
    """A number plus how we know it. Never pass a bare float across this boundary.

    Example: Value(4.2e5, Provenance.MEASURED, source="MyPerioPath", unit="genome equivalents/mL")
    """

    value: float
    provenance: Provenance = Provenance.DEFAULT
    source: str = ""
    unit: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.value != self.value:            # NaN
            raise ValueError(f"intake Value is NaN (source={self.source!r}); reject it upstream")

    @property
    def measured(self) -> bool:
        return self.provenance is Provenance.MEASURED


@dataclass
class SiteIntake:
    """One sampled site. v1 populates exactly one ('pooled'); the list shape is the per-site seam.

    ``composition`` maps a cube species id -> Value. Units are whatever the adapter recorded; the
    absolute/relative split is preserved by keeping ``total_load`` separate rather than normalising
    it away, because two mouths with identical ratios and 100x different load are not the same mouth.
    """

    site_id: str = "pooled"
    tooth: str = ""                                  # e.g. "30-DB"; empty for a pooled sample
    probing_depth_mm: Optional[Value] = None
    clinical_attachment_mm: Optional[Value] = None
    bleeding_on_probing: Optional[Value] = None      # 0.0 / 1.0, or a site fraction for pooled
    composition: dict = field(default_factory=dict)  # species_id -> Value (absolute or relative)
    total_load: Optional[Value] = None               # total bacterial load, if the assay reports it
    composition_is_relative: bool = False

    def measured_species(self) -> list:
        """Species ids whose abundance was actually assayed on this patient."""
        return sorted(k for k, v in self.composition.items() if v.measured)


@dataclass
class Saliva:
    """Salivary chemistry. These become the oral-cavity FACE boundary, not the initial composition."""

    pH: Optional[Value] = None
    flow_rate_ml_min: Optional[Value] = None
    nitrate: Optional[Value] = None
    nitrite: Optional[Value] = None
    buffering_capacity: Optional[Value] = None


@dataclass
class HostMarkers:
    """Host inflammatory / tissue-breakdown markers. These inform psi and validation, not seeding."""

    ammp8: Optional[Value] = None          # PerioSafe aMMP-8, chairside
    hs_crp: Optional[Value] = None
    il6: Optional[Value] = None
    tnf_alpha: Optional[Value] = None
    hba1c: Optional[Value] = None          # diabetes is the big effect modifier in perio


@dataclass
class CoverageReport:
    """How much of this cube is really this patient. Shown BEFORE any result, by design."""

    n_measured: int
    n_modeled: int
    measured_fraction: float
    biomass_fraction_measured: float
    unmapped_taxa: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        """True when the run should be visibly marked as partial."""
        return bool(self.warnings) or self.measured_fraction < 0.5

    def summary(self) -> str:
        """One line fit for a UI banner.

        Example: "11/18 species measured (61% of modeled biomass) - 2 warnings"
        """
        base = (f"{self.n_measured}/{self.n_modeled} species measured "
                f"({self.biomass_fraction_measured:.0%} of modeled biomass)")
        if self.unmapped_taxa:
            base += f" - {len(self.unmapped_taxa)} reported taxa had no model node"
        if self.warnings:
            base += f" - {len(self.warnings)} warning(s)"
        return base


@dataclass
class PatientIntake:
    """Everything one patient contributes to a cube run.

    ``subject_ref`` is a caller-supplied pseudonymous key. Nothing in this package resolves it to an
    identity, writes it anywhere, or transmits it.
    """

    subject_ref: str = "unknown"
    collected_on: str = ""                              # ISO date; free text, never parsed for logic
    sites: list = field(default_factory=list)           # list[SiteIntake]
    saliva: Saliva = field(default_factory=Saliva)
    host: HostMarkers = field(default_factory=HostMarkers)
    assay: str = ""                                     # e.g. "OralDNA MyPerioPath (qPCR)"
    unmapped_taxa: list = field(default_factory=list)   # reported names with no model node
    warnings: list = field(default_factory=list)
    notes: str = ""

    @property
    def primary_site(self) -> SiteIntake:
        """The site a single-cube run represents. v1: the only site; later: the selected one."""
        if not self.sites:
            raise ValueError("PatientIntake has no sites; adapters must create at least one")
        return self.sites[0]

    def coverage(self, modeled_ids: Optional[Iterable] = None) -> CoverageReport:
        """Quantify how much of the cube is measured versus inferred.

        ``biomass_fraction_measured`` weights by the abundance actually attributed to measured
        species, which is the number that matters: measuring 4 dominant organisms tells you far
        more than measuring 4 rare ones.
        """
        from sim.intake.priors import complete_composition
        from sim.intake.taxa import modeled_species_ids

        ids = list(modeled_ids) if modeled_ids is not None else modeled_species_ids()
        site = self.primary_site
        measured = [s for s in ids if s in site.composition and site.composition[s].measured]

        # Weigh against the COMPLETED composition, not just the measured subset. Dividing measured
        # mass by measured mass always yields 100%, which would tell a clinician a pathogen-only
        # panel had covered the whole community - the exact opposite of this metric's job.
        try:
            full = complete_composition(site.composition)
        except ValueError:
            full = dict(site.composition)
        total = sum(v.value for v in full.values() if v.value > 0)
        meas_mass = sum(full[s].value for s in measured if s in full and full[s].value > 0)
        return CoverageReport(
            n_measured=len(measured),
            n_modeled=len(ids),
            measured_fraction=(len(measured) / len(ids)) if ids else 0.0,
            biomass_fraction_measured=(meas_mass / total) if total > 0 else 0.0,
            unmapped_taxa=list(self.unmapped_taxa),
            warnings=list(self.warnings),
        )

