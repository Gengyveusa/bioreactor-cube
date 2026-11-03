"""Patient intake — turn real clinical/lab measurements into a seeded cube.

The engine already exposes the seam this package targets: ``seed_composition(grid, abundances)``
takes a ``{species_id: abundance}`` dict. Everything here exists to build that dict honestly from
a real panel, and to carry the three other things a panel implies that the dict cannot hold:
the salivary boundary conditions, the pocket geometry, and the host markers.

Design commitments (operator decisions, 2026-08-01):
  * One cube per patient now, per-site later — so the schema carries a site list from day one.
  * Keep BOTH relative composition and absolute load; they are clinically different.
  * Unmeasured species are completed from priors CONDITIONED on what was measured.
  * Every value carries provenance (measured / derived / prior / default) and it is shown, not hidden.
  * Messy input runs anyway, flagged and degraded, rather than being refused.
  * Local-first: nothing here uploads, logs, or transmits. PHI stays on the machine.

Example
-------
>>> from sim.intake import from_myperiopath, apply_intake
>>> intake = from_myperiopath({"Pg": 4.2e5, "Tf": 1.1e5, "Aa": 0.0}, probing_depth_mm=6.0)
>>> intake.coverage().measured_fraction     # doctest: +SKIP
0.11
"""

from __future__ import annotations

from sim.intake.schema import (
    Provenance,
    Value,
    SiteIntake,
    Saliva,
    HostMarkers,
    PatientIntake,
    CoverageReport,
)
from sim.intake.taxa import resolve_taxon, TaxonResolution, coverage_of
from sim.intake.adapters import from_myperiopath, from_manual, from_relative_abundance
from sim.intake.priors import complete_composition
from sim.intake.apply import (
    to_abundances, to_face_conditions, to_grid_shape, apply_intake, load_factor,
    to_host_state, recession_mm,
)

__all__ = [
    "Provenance", "Value", "SiteIntake", "Saliva", "HostMarkers", "PatientIntake",
    "CoverageReport", "resolve_taxon", "TaxonResolution", "coverage_of",
    "from_myperiopath", "from_manual", "from_relative_abundance",
    "complete_composition",
    "to_abundances", "to_face_conditions", "to_grid_shape", "apply_intake", "load_factor",
    "to_host_state", "recession_mm",
]

