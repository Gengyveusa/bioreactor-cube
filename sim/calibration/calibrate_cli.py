"""Calibration entry point.

Usage:
    python -m sim.calibration.calibrate_cli demo

Runs a minimal ABC-SMC against a "healthy commensal" synthetic target.
For real calibration, plug an HMP-2 / eMBL summary into target_observed().
"""
from __future__ import annotations
import argparse
import json
from .summary_stats import SummaryStats
from .abc_smc import ABCSMC, ABCSMCConfig
from .priors import default_priors


def synthetic_healthy_target() -> SummaryStats:
    """A plausible 'healthy oral microbiome' fingerprint.

    Calibration target: tunes the simulator to reproduce this state.
    For real work, replace this with a SummaryStats computed from HMP-2 data.
    """
    return SummaryStats(
        shannon_alpha=2.3,
        simpson_alpha=0.85,
        n_present=12,
        pioneer_fraction=0.55,
        nitrate_reducer_fraction=0.18,
        red_complex_fraction=0.02,
        orange_complex_fraction=0.04,
        fn_betweenness=0.45,
        pg_betweenness=0.05,
        pH_mean=6.8,
        pH_variance=0.05,
        NO_total=10.0,
        H2S_total=0.5,
        omv_flux_gingipain=0.02,
        omv_flux_FadA=0.10,
    )


def synthetic_periodontitis_target() -> SummaryStats:
    """A plausible 'periodontitis' fingerprint — dysbiotic shift."""
    return SummaryStats(
        shannon_alpha=1.8,
        simpson_alpha=0.72,
        n_present=8,
        pioneer_fraction=0.25,
        nitrate_reducer_fraction=0.08,
        red_complex_fraction=0.18,
        orange_complex_fraction=0.15,
        fn_betweenness=0.55,
        pg_betweenness=0.20,
        pH_mean=7.2,
        pH_variance=0.20,
        NO_total=4.0,
        H2S_total=3.5,
        omv_flux_gingipain=0.30,
        omv_flux_FadA=0.25,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("target", choices=["healthy", "periodontitis", "demo"], default="demo")
    p.add_argument("--n-particles", type=int, default=20)
    p.add_argument("--n-generations", type=int, default=2)
    p.add_argument("--inner-ticks", type=int, default=300)
    args = p.parse_args()

    if args.target == "periodontitis":
        target = synthetic_periodontitis_target()
    else:
        target = synthetic_healthy_target()

    print(f"Target ({args.target}):")
    for k, v in target.to_array().__iter__():
        pass  # avoid printing the array form
    for fld in SummaryStats.field_names():
        print(f"  {fld:<30} {getattr(target, fld):.4f}")

    cfg = ABCSMCConfig(
        n_particles=args.n_particles,
        n_generations=args.n_generations,
        inner_ticks=args.inner_ticks,
    )
    abc = ABCSMC(obs=target, priors=default_priors(), cfg=cfg)
    abc.run()
    print("\nPosterior (last generation):")
    summary = abc.posterior_summary()
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()

