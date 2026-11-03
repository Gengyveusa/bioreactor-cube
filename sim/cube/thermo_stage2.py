"""thermo_stage2.py — run the sigma-probe against a LIVE engine snapshot in each tissue.

Builds the cube for oral / artery / brain / liver via run_fused (engine UNCHANGED), snapshots the
live terrain fields + the ECM-derived effective diffusivity (theta/lambda^2), and computes the
entropy-production field sigma with sim.cube.thermo. Read-only on the engine — a driver, not an edit.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from sim.cube import thermo
from sim.cube.ecm import porosity
from sim.cube.fused import run_fused
from sim.cube.tissues import ARTERY_TISSUE, BRAIN_TISSUE, LIVER_TISSUE, ORAL_TISSUE

DIM, STEPS = 16, 60

TISSUES = [ORAL_TISSUE, ARTERY_TISSUE, BRAIN_TISSUE, LIVER_TISSUE]

# Non-solute fields (matrix density, enzyme/flow rates, mechanical, structure, baselines) are
# excluded from the chemical-potential transport term; everything else that varies is a diffusible
# solute. Auto-discovery keeps it honest and catches per-tissue species (e.g. O2 zonation in liver).
EXCLUDE = {"ecm", "enzyme", "shear", "glymph", "oxLDL0", "mineral", "eps"}


def _collect(eng):
    """Discover the varying solute fields the engine actually produced (grid chemistry + extra)."""
    cand = {k: np.asarray(v, float) for k, v in eng.grid._fields.items()}
    cand.update({k: np.asarray(v, float) for k, v in eng.extra.items()})
    return {k: v for k, v in cand.items()
            if k not in EXCLUDE and v.std() > 1e-9 and np.any(v > 0)}


def _d_eff(eng, tissue, solutes):
    """Real transport map from the live ECM: D_eff = theta/lambda^2, theta = porosity(m, alpha)."""
    spec = tissue.ecm
    m = eng.extra.get("ecm")
    if m is None:
        m = np.full(eng.grid.shape, spec.base_density)
    theta = porosity(m, spec.alpha)
    d = theta / (spec.tortuosity ** 2)
    return {k: d for k in solutes}, float(theta.mean()), spec.tortuosity


def main():
    print("=" * 80)
    print(f"Stage 2 — entropy production (sigma) across tissues   live snapshot, dim={DIM}^3, {STEPS} steps")
    print("=" * 80)
    rows = []
    for tissue in TISSUES:
        eng, _ = run_fused(tissue, dim=DIM, steps=STEPS)
        fields = _collect(eng)
        if not fields:
            print(f"\n● {tissue.name}: no varying solute fields"); continue

        d_eff, theta_mean, lam = _d_eff(eng, tissue, list(fields))
        contrib = {k: float(thermo.transport_sigma({k: fields[k]}, {k: d_eff[k]}).sum()) for k in fields}
        sigma = thermo.total_sigma(fields, d_eff)
        s = thermo.summary(sigma)
        prof = thermo.depth_profile(sigma)
        z_peak = int(np.argmax(prof))
        top = sorted(contrib.items(), key=lambda kv: -kv[1])
        ro = {k: round(v, 3) for k, v in eng.readout().items() if isinstance(v, (int, float))}
        rows.append((tissue.name, s["Sigma_total"], theta_mean, lam, top[0][0]))

        print(f"\n● {tissue.name}")
        print(f"    Sigma_total = {s['Sigma_total']:.3e}    sigma_max = {s['sigma_max']:.3e}    "
              f"hotspot_frac = {s['hotspot_frac']:.3f}")
        print(f"    ECM transport: <theta>={theta_mean:.3f}  lambda={lam}  "
              f"D_eff~{theta_mean / lam**2:.3f} of free water")
        print(f"    dissipation by species: " + "  ".join(f"{k}={v:.2e}" for k, v in top))
        print(f"    sigma peaks at z={z_peak}/{len(prof) - 1}  (lumenal/oral surface=nz-1, deep=0)")
        print(f"    psi: {ro}")

    print("\n" + "=" * 80)
    print(f"{'tissue':30}{'Sigma_total':>13}{'<theta>':>9}{'lambda':>8}   dominant dissipator")
    print("-" * 80)
    for name, S, th, lam, dom in rows:
        print(f"{name:30}{S:13.3e}{th:9.3f}{lam:8.1f}   {dom}")
    print("=" * 80)
    print("Same probe, four tissues, engine untouched: sigma is tissue-distinct in magnitude,")
    print("transport regime (theta/lambda), and which species carries the dissipation.")


if __name__ == "__main__":
    main()

