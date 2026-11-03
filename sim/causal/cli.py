"""Counterfactual runner CLI.

Usage:
    python -m sim.causal.cli compare --intervention gengyve
    python -m sim.causal.cli compare --intervention lytica
    python -m sim.causal.cli compare --intervention loria
    python -m sim.causal.cli compare --intervention bdello
    python -m sim.causal.cli all                              # runs all four

Compares the observational (no-intervention) vs interventional trajectory
for the bleed scenario (heme_bleed_rate=0.08 → Pg dysbiosis ignition).
Outputs the ATE table to stdout and saves a plot to output_scm/.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

from ..config import SimConfig
from .gengyve_product import (
    gengyve_ha_castor_oil, fasolati_lytica, fasolati_loria, bdellovibrio_deployment,
)
from .counterfactual import run_paired


REGISTRY = {
    "gengyve": gengyve_ha_castor_oil,
    "lytica":  fasolati_lytica,
    "loria":   fasolati_loria,
    "bdello":  bdellovibrio_deployment,
}


def make_bleed_cfg(ticks: int = 1500) -> SimConfig:
    """The OVN-ignition bleed scenario — Pg gets its heme."""
    cfg = SimConfig()
    cfg.ticks = ticks
    cfg.seed = 42
    cfg.log_every = max(10, ticks // 30)
    cfg.chemistry.heme_bleed_rate = 0.08
    cfg.agents.initial_inoculum_per_species = 8
    cfg.agents.max_agents = 5000
    cfg.agents.viability_grace_ticks = 80
    cfg.agents.division_threshold = 0.35
    return cfg


def plot_ate(results: list, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observables = [
        ("dysbiosis_end", "Dysbiosis index (end)"),
        ("dysbiosis_max", "Dysbiosis index (max)"),
        ("fn_betweenness_max", "Fn betweenness (max)"),
        ("pg_betweenness_max", "Pg betweenness (max)"),
        ("pioneer_fraction_end", "Pioneer fraction (end)"),
        ("omv_flux_proxy", "OMV flux proxy"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    interventions = [r.intervention_name for r in results]
    colors = ["#1A4036", "#C41E3A", "#7FBCA9", "#A598C9"][: len(results)]
    for ax, (key, label) in zip(axes, observables):
        obs_vals = [getattr(r.obs, key, 0.0) for r in results]
        iv_vals = [getattr(r.iv, key, 0.0) for r in results]
        x = list(range(len(interventions)))
        w = 0.35
        ax.bar([xi - w/2 for xi in x], obs_vals, width=w, color="#D5CFC0", edgecolor="#1F2A24", label="observational")
        ax.bar([xi + w/2 for xi in x], iv_vals, width=w, color=colors[: len(results)], edgecolor="#1F2A24", label="interventional")
        ax.set_xticks(x)
        ax.set_xticklabels(interventions, rotation=20, fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.tick_params(axis="y", labelsize=8)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Layer-5 SCM · Counterfactual ATE on the bleed scenario", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Plot saved: {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("compare", help="Run one intervention vs observational")
    pc.add_argument("--intervention", choices=list(REGISTRY.keys()), required=True)
    pc.add_argument("--ticks", type=int, default=1500)
    pc.add_argument("--seed", type=int, default=42)

    pa = sub.add_parser("all", help="Run all four interventions and plot")
    pa.add_argument("--ticks", type=int, default=1500)
    pa.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    out_dir = Path("output_scm")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "compare":
        cfg = make_bleed_cfg(ticks=args.ticks)
        intervention = REGISTRY[args.intervention]()
        print(f"\nRunning paired simulation: bleed vs bleed+{args.intervention}")
        print(f"  ticks={args.ticks}  seed={args.seed}")
        ate = run_paired(cfg, intervention, seed=args.seed)
        print(ate.summary())
        return 0

    if args.cmd == "all":
        results = []
        for name, fn in REGISTRY.items():
            cfg = make_bleed_cfg(ticks=args.ticks)
            print(f"\n[{name}] running paired simulation...")
            ate = run_paired(cfg, fn(), seed=args.seed)
            print(ate.summary())
            results.append(ate)
        plot_ate(results, out_dir / "ate_comparison.png")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

