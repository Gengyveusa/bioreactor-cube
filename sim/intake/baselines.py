"""Two shipped reference patients, so every feature can be demoed and regression-tested without
touching real patient data.

These are NOT measurements. They are hand-authored, literature-shaped compositions standing in for
"typical health" and "typical periodontitis", and every value is stamped ``Provenance.DEFAULT`` so
nothing downstream can mistake them for a real subject. Their numbers are placeholders pending the
published-cohort calibration. [[VERIFY]]
"""

from __future__ import annotations

from sim.intake.adapters import from_manual
from sim.intake.schema import PatientIntake, Provenance, Value


def _stamp(intake: PatientIntake, note: str) -> PatientIntake:
    """Re-stamp a hand-authored composition as DEFAULT, never MEASURED."""
    site = intake.primary_site
    site.composition = {
        k: Value(v.value, Provenance.DEFAULT, "shipped baseline", "relative", note)
        for k, v in site.composition.items()
    }
    intake.warnings.append("synthetic baseline patient - not a real subject")
    return intake


def healthy_baseline() -> PatientIntake:
    """Commensal-dominant, reducer-rich, shallow non-bleeding sulcus.

    Example: healthy_baseline().primary_site.probing_depth_mm.value -> 2.0
    """
    intake = from_manual(
        {"Sa": 0.18, "Sm": 0.16, "An": 0.11, "Hp": 0.08, "Rm": 0.10, "Vp": 0.09,
         "Fn": 0.09, "Pi": 0.03, "Ec": 0.03, "Pg": 0.005, "Td": 0.004, "Tf": 0.006,
         "Aa": 0.004, "Smu": 0.02, "Lb": 0.01, "Fa": 0.004, "Pm": 0.008, "Ca": 0.007},
        subject_ref="baseline-healthy", source="shipped baseline",
        probing_depth_mm=2.0, bleeding_on_probing=0.0, saliva_pH=7.0,
    )
    return _stamp(intake, "hand-authored healthy reference; not measured")


def perio_baseline() -> PatientIntake:
    """Red-complex-shifted, reducer-depleted, deep bleeding pocket."""
    intake = from_manual(
        {"Sa": 0.06, "Sm": 0.05, "An": 0.04, "Hp": 0.02, "Rm": 0.03, "Vp": 0.04,
         "Fn": 0.14, "Pi": 0.08, "Ec": 0.04, "Pg": 0.13, "Td": 0.10, "Tf": 0.09,
         "Aa": 0.03, "Smu": 0.02, "Lb": 0.01, "Fa": 0.05, "Pm": 0.05, "Ca": 0.02},
        subject_ref="baseline-perio", source="shipped baseline",
        probing_depth_mm=6.5, bleeding_on_probing=1.0, saliva_pH=6.6,
    )
    return _stamp(intake, "hand-authored periodontitis reference; not measured")


HEALTHY = healthy_baseline
PERIO = perio_baseline

