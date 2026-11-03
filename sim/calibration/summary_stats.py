"""Summary statistics for ABC distance computation.

ABC compares simulated and observed datasets via low-dimensional summaries
rather than full likelihoods. The summary set must be:
- Informative (captures features that distinguish parameter regimes)
- Low-dimensional (curse of dimensionality kills high-D ABC)
- Robust to nuisance variation (sequencing depth, sample size, batch)

The summaries below are designed for matching the simulator output (HDF5
trajectory) against real oral microbiome data (16S/shotgun composition +
clinical phenotypes).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import h5py


@dataclass
class SummaryStats:
    """Low-dimensional fingerprint of a microbiome state.

    Designed to be computable from EITHER:
    - A simulator trajectory (sim/observables.py output)
    - A real microbiome sample (16S table + clinical metadata)
    """
    # Composition fingerprint — invariant to total abundance (compositional)
    shannon_alpha: float                  # Shannon diversity
    simpson_alpha: float                  # Simpson diversity (1 - D)
    n_present: int                        # number of species with abundance > 0
    # Functional fingerprint — Layer 9 coherence axis aggregates
    pioneer_fraction: float
    nitrate_reducer_fraction: float
    red_complex_fraction: float           # the dysbiosis primary signal
    orange_complex_fraction: float
    # Topology fingerprint — realized graph properties
    fn_betweenness: float                 # bridge realization
    pg_betweenness: float                 # keystone realization
    # Chemistry fingerprint — Layer 3 outputs
    pH_mean: float
    pH_variance: float
    # Coherence axis aggregates (Layer 9)
    NO_total: float
    H2S_total: float
    # OMV flux (Layer 7) — the OVN cascade observable
    omv_flux_gingipain: float
    omv_flux_FadA: float

    def to_array(self) -> np.ndarray:
        return np.array(list(asdict(self).values()), dtype=np.float64)

    @staticmethod
    def field_names() -> list[str]:
        return list(SummaryStats.__dataclass_fields__.keys())


def compute_summary_stats(h5_path: str | Path,
                          window: tuple[int, int] | None = None) -> SummaryStats:
    """Compute the summary fingerprint from a simulator trajectory."""
    from ..species import CATALOG
    with h5py.File(h5_path, "r") as f:
        comp = f["composition"][...]   # (T, n_species, n_depths)
        ticks = f["tick"][...]
        no_total = float(f["NO"][...].mean())
        h2s_total = float(f["H2S"][...].mean())
        pH_mean = float(f["pH_mean"][...].mean())
        pH_var = float(f["pH_variance"][...].mean())
        fn_bet = float(f["Fn_betweenness"][...].max())
        pg_bet = float(f["Pg_betweenness"][...].max())

    if window is None:
        window = (len(ticks) // 2, len(ticks))  # second half = "steady state"
    comp_window = comp[window[0]:window[1]]
    avg_comp = comp_window.sum(axis=(0, 2))  # total biomass per species over window
    avg_comp = avg_comp / max(avg_comp.sum(), 1e-9)

    # Diversity
    p = avg_comp[avg_comp > 0]
    shannon = float(-(p * np.log(p)).sum()) if p.size > 0 else 0.0
    simpson = float(1.0 - (p ** 2).sum()) if p.size > 0 else 0.0
    n_present = int((avg_comp > 1e-4).sum())

    # Archetype fractions
    arch_frac: dict[str, float] = {}
    for i, s in enumerate(CATALOG):
        arch_frac[s.archetype] = arch_frac.get(s.archetype, 0.0) + avg_comp[i]
    pioneer = arch_frac.get("Pioneer", 0.0)
    nitrate = arch_frac.get("NitrateReducer", 0.0)
    keystone = arch_frac.get("Keystone", 0.0)
    bridge = arch_frac.get("Bridge", 0.0)

    # OMV flux proxies — placeholder, will be wired through HDF5 once
    # the model writes per-cargo flux. For now use Pg/Fn biomass as proxies.
    pg_idx = next((i for i, s in enumerate(CATALOG) if s.id == "Pg"), None)
    fn_idx = next((i for i, s in enumerate(CATALOG) if s.id == "Fn"), None)
    omv_gp = float(avg_comp[pg_idx]) if pg_idx is not None else 0.0
    omv_fada = float(avg_comp[fn_idx]) if fn_idx is not None else 0.0

    return SummaryStats(
        shannon_alpha=shannon,
        simpson_alpha=simpson,
        n_present=n_present,
        pioneer_fraction=pioneer,
        nitrate_reducer_fraction=nitrate,
        red_complex_fraction=keystone,
        orange_complex_fraction=bridge,
        fn_betweenness=fn_bet,
        pg_betweenness=pg_bet,
        pH_mean=pH_mean,
        pH_variance=pH_var,
        NO_total=no_total,
        H2S_total=h2s_total,
        omv_flux_gingipain=omv_gp,
        omv_flux_FadA=omv_fada,
    )


def distance(s_sim: SummaryStats, s_obs: SummaryStats,
             weights: np.ndarray | None = None) -> float:
    """Mahalanobis-like weighted Euclidean distance between summary fingerprints.

    Weights default to 1/std across a calibration cohort. For v1 we use
    component-wise normalization and unit weights.
    """
    a = s_sim.to_array()
    b = s_obs.to_array()
    diff = a - b
    if weights is None:
        weights = np.ones_like(diff)
    return float(np.sqrt(np.sum(weights * diff ** 2)))

