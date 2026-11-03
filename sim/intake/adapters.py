"""Kit adapters — one per assay. Each maps vendor names and units onto the intake contract.

Adding a new kit means adding a function here and rows to ``taxa.ALIASES``. Nothing in the engine
changes. Adapters are deliberately lenient about coverage and strict about physiology: a missing
organism is normal clinical reality, but a pH of 12 is an error, not a datum.
"""

from __future__ import annotations

from typing import Optional

from sim.intake.schema import (
    Provenance, Value, SiteIntake, Saliva, HostMarkers, PatientIntake,
)
from sim.intake.taxa import resolve_taxon

# Physiologically possible ranges. Outside these we reject the VALUE (not the run) and warn.
PLAUSIBLE = {
    "saliva_pH": (4.0, 9.0),
    "probing_depth_mm": (0.0, 15.0),
    "clinical_attachment_mm": (0.0, 20.0),
    "flow_rate_ml_min": (0.0, 10.0),
}


def _checked(name: str, raw, provenance: Provenance, source: str, unit: str,
             warnings: list) -> Optional[Value]:
    """Range-check one scalar. Returns None (and warns) when implausible, so the run continues."""
    if raw is None:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        warnings.append(f"{name}: {raw!r} is not a number - ignored")
        return None
    if v != v:
        warnings.append(f"{name}: NaN - ignored")
        return None
    lo, hi = PLAUSIBLE.get(name, (float("-inf"), float("inf")))
    if not (lo <= v <= hi):
        warnings.append(f"{name}: {v} outside plausible range [{lo}, {hi}] - ignored")
        return None
    return Value(v, provenance, source, unit)


def _ingest_counts(counts: dict, source: str, unit: str,
                   warnings: list, unmapped: list) -> dict:
    """Resolve reported taxon names to species ids, keeping the first mapping for duplicates."""
    comp: dict = {}
    for name, raw in counts.items():
        r = resolve_taxon(name)
        if not r.ok:
            unmapped.append(f"{name} ({r.reason})")
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            warnings.append(f"{name}: {raw!r} is not a number - ignored")
            continue
        if v < 0:
            warnings.append(f"{name}: negative abundance {v} - ignored")
            continue
        if r.species_id in comp:
            warnings.append(f"{name}: duplicate mapping onto {r.species_id} - kept the first")
            continue
        if not r.in_catalog:
            warnings.append(
                f"{name}: maps to {r.species_id}, which is defined but outside the default "
                "18-species catalog the grid simulates - carried but not seeded")
        comp[r.species_id] = Value(v, Provenance.MEASURED, source, unit)
    return comp


def from_myperiopath(counts: dict, *, subject_ref: str = "unknown", collected_on: str = "",
                     probing_depth_mm=None, bleeding_on_probing=None,
                     clinical_attachment_mm=None, total_load=None,
                     saliva_pH=None, salivary_nitrite=None, ammp8=None,
                     salivary_buffer=None, salivary_buffer_unit="pH",
                     hs_crp=None, il6=None, tnf_alpha=None, hba1c=None) -> PatientIntake:
    """OralDNA MyPerioPath (qPCR, dentist-ordered): 11 pathogens in genome equivalents/mL.

    Ten of the eleven have a model node; Capnocytophaga does not and is recorded as unmapped.
    The panel measures no commensals, so ``coverage()`` on the result will be low by design and the
    protective flora must be completed by :func:`sim.intake.priors.complete_composition`.

    Example: from_myperiopath({"Pg": 4.2e5, "Tf": 1.1e5}, probing_depth_mm=6.0)
    """
    warnings: list = []
    unmapped: list = []
    src = "OralDNA MyPerioPath (qPCR)"
    comp = _ingest_counts(counts, src, "genome equivalents/mL", warnings, unmapped)
    if not comp:
        warnings.append("no reported taxon mapped onto a model species")

    site = SiteIntake(
        site_id="pooled",
        probing_depth_mm=_checked("probing_depth_mm", probing_depth_mm,
                                  Provenance.MEASURED, "clinical exam", "mm", warnings),
        clinical_attachment_mm=_checked("clinical_attachment_mm", clinical_attachment_mm,
                                        Provenance.MEASURED, "clinical exam", "mm", warnings),
        bleeding_on_probing=_checked("bleeding_on_probing", bleeding_on_probing,
                                     Provenance.MEASURED, "clinical exam", "fraction", warnings),
        composition=comp,
        total_load=_checked("total_load", total_load, Provenance.MEASURED, src,
                            "genome equivalents/mL", warnings),
        composition_is_relative=False,
    )
    return PatientIntake(
        subject_ref=subject_ref, collected_on=collected_on, sites=[site],
        saliva=Saliva(
            pH=_checked("saliva_pH", saliva_pH, Provenance.MEASURED, "chairside", "pH", warnings),
            nitrite=_checked("salivary_nitrite", salivary_nitrite, Provenance.MEASURED,
                             "Berkeley Life strip", "relative", warnings),
            # The unit travels WITH the number. A buffer strip reports an endpoint pH and a
            # titration reports mmol/L/pH; a bare float is ambiguous between them and the
            # ambiguity is exactly what silently produces a wrong pH curve.
            buffering_capacity=_checked("salivary_buffer", salivary_buffer, Provenance.MEASURED,
                                        "chairside buffer test", salivary_buffer_unit, warnings),
        ),
        host=HostMarkers(
            ammp8=_checked("ammp8", ammp8, Provenance.MEASURED, "PerioSafe aMMP-8", "ng/mL", warnings),
            # These four come from a blood draw rather than the pathogen panel, but they arrive in
            # the same visit and the cube's host layer reads them together. Units matter: the
            # normalization anchors in sim.cube.host are keyed to exactly these.
            hs_crp=_checked("hs_crp", hs_crp, Provenance.MEASURED, "lab draw", "mg/L", warnings),
            il6=_checked("il6", il6, Provenance.MEASURED, "lab draw", "pg/mL", warnings),
            tnf_alpha=_checked("tnf_alpha", tnf_alpha, Provenance.MEASURED, "lab draw",
                               "pg/mL", warnings),
            hba1c=_checked("hba1c", hba1c, Provenance.MEASURED, "lab draw", "%", warnings),
        ),
        assay=src, unmapped_taxa=unmapped, warnings=warnings,
    )


def from_relative_abundance(fractions: dict, *, subject_ref: str = "unknown",
                            source: str = "NGS panel", collected_on: str = "",
                            probing_depth_mm=None, bleeding_on_probing=None,
                            total_load=None, saliva_pH=None) -> PatientIntake:
    """A shotgun/16S panel reporting relative abundance (Bristle, Viome, or a BIOM-derived table).

    Values may be fractions or percentages; anything summing to > 1.5 is treated as percent.
    """
    warnings: list = []
    unmapped: list = []
    comp = _ingest_counts(fractions, source, "relative", warnings, unmapped)
    total = sum(v.value for v in comp.values())
    if total > 1.5:
        comp = {k: Value(v.value / 100.0, v.provenance, v.source, "relative", "converted from %")
                for k, v in comp.items()}
        warnings.append("input summed to >1.5; interpreted as percentages and divided by 100")

    site = SiteIntake(
        site_id="pooled",
        probing_depth_mm=_checked("probing_depth_mm", probing_depth_mm,
                                  Provenance.MEASURED, "clinical exam", "mm", warnings),
        bleeding_on_probing=_checked("bleeding_on_probing", bleeding_on_probing,
                                     Provenance.MEASURED, "clinical exam", "fraction", warnings),
        composition=comp,
        total_load=_checked("total_load", total_load, Provenance.MEASURED, source,
                            "cells/mL", warnings),
        composition_is_relative=True,
    )
    return PatientIntake(
        subject_ref=subject_ref, collected_on=collected_on, sites=[site],
        saliva=Saliva(pH=_checked("saliva_pH", saliva_pH, Provenance.MEASURED,
                                  "chairside", "pH", warnings)),
        assay=source, unmapped_taxa=unmapped, warnings=warnings,
    )


def from_manual(composition: dict, **kw) -> PatientIntake:
    """Hand-entered composition, for testing and for the shipped baseline patients.

    Accepts species ids directly. Everything else behaves like the relative-abundance adapter.
    """
    return from_relative_abundance(composition, source=kw.pop("source", "manual entry"), **kw)

