"""thermo_stage3.py — entropy production HEALTHY vs DISEASED within each tissue, supply-boundary
cancelled via delta-sigma (= sigma_diseased - sigma_healthy).

Health-state axes: oral = ORAL_HEALTHY vs ORAL_TISSUE (built-in conditions); distal tissues =
oral->systemic inflammatory influx via eng.receive(), 0 vs INFLUX. Engine UNCHANGED; a driver.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from sim.cube import thermo
from sim.cube.ecm import porosity
from sim.cube.fused import CubeEpoch4
from sim.cube.grid import CubeGrid
from sim.cube.tissues import ARTERY_TISSUE, BRAIN_TISSUE, LIVER_TISSUE, ORAL_HEALTHY, ORAL_TISSUE

DIM, STEPS, INFLUX = 16, 60, 0.15
EXCLUDE = {"ecm", "enzyme", "shear", "glymph", "oxLDL0", "mineral", "eps"}

CASES = [
    ("oral biofilm",          (ORAL_HEALTHY, 0.0),  (ORAL_TISSUE, 0.0),
     ["dysbiosis", "omv_flux", "gingipain"]),
    ("artery / atheroma",     (ARTERY_TISSUE, 0.0), (ARTERY_TISSUE, INFLUX),
     ["inflammation_load", "endothelial_activation", "rupture_index"]),
    ("brain / neurovascular", (BRAIN_TISSUE, 0.0),  (BRAIN_TISSUE, INFLUX),
     ["inflammation_load", "neuroinflammation", "bbb_integrity"]),
    ("liver / lobule",        (LIVER_TISSUE, 0.0),  (LIVER_TISSUE, INFLUX),
     ["inflammation_load", "hepatic_inflammation", "fibrosis"]),
]


def _run(tissue, influx):
    grid = CubeGrid(DIM, DIM, DIM, species_ids=tissue.cell_ids)
    eng = CubeEpoch4(grid, tissue)
    eng.seed_uniform()
    for _ in range(STEPS):
        eng.step(0.1)
        if influx:
            eng.receive(influx)
    return eng


def _discover(eng):
    cand = {k: np.asarray(v, float) for k, v in eng.grid._fields.items()}
    cand.update({k: np.asarray(v, float) for k, v in eng.extra.items()})
    return {k for k, v in cand.items() if k not in EXCLUDE and v.std() > 1e-9 and np.any(v > 0)}


def _fields(eng, keys):
    out = {}
    for k in keys:
        try:
            out[k] = np.asarray(eng.field(k), float)
        except KeyError:
            out[k] = np.zeros(eng.grid.shape)
    return out


def _deff(eng, tissue, keys):
    spec = tissue.ecm
    m = eng.extra.get("ecm", np.full(eng.grid.shape, tissue.ecm.base_density))
    d = porosity(m, spec.alpha) / (spec.tortuosity ** 2)
    return {k: d for k in keys}


def _psi(eng, want):
    r = eng.readout()
    return {k: round(float(r[k]), 3) for k in want if k in r and isinstance(r[k], (int, float))}


def main():
    print("=" * 86)
    print(f"Stage 3 — sigma HEALTHY vs DISEASED per tissue   (dim={DIM}^3, {STEPS} steps, influx={INFLUX})")
    print("delta-sigma = diseased - healthy  cancels the shared O2 supply boundary -> the lesion shows")
    print("=" * 86)

    for label, (th_h, inf_h), (th_d, inf_d), psi_keys in CASES:
        eh, ed = _run(th_h, inf_h), _run(th_d, inf_d)
        keys = sorted(_discover(eh) | _discover(ed))
        fh, fd = _fields(eh, keys), _fields(ed, keys)
        dh, dd = _deff(eh, th_h, keys), _deff(ed, th_d, keys)
        sh, sd = thermo.total_sigma(fh, dh), thermo.total_sigma(fd, dd)
        dsig = thermo.delta_sigma(sd, sh)

        per = {k: float(thermo.transport_sigma({k: fd[k]}, {k: dd[k]}).sum()
                        - thermo.transport_sigma({k: fh[k]}, {k: dh[k]}).sum()) for k in keys}
        top = sorted(per.items(), key=lambda kv: -abs(kv[1]))[:3]
        prof = dsig.mean(axis=(0, 1))
        zpk = int(np.argmax(np.abs(prof)))
        Sh, Sd = float(sh.sum()), float(sd.sum())

        print(f"\n● {label}")
        print(f"    Sigma   healthy = {Sh:.3e}    diseased = {Sd:.3e}    ratio = {Sd / max(Sh, 1e-30):.2f}x")
        print(f"    delta-Sigma (lesion footprint) = {float(dsig.sum()):+.3e}    "
              f"peaks at z={zpk}/{DIM - 1} ({'lumen/surface' if zpk > DIM // 2 else 'interior/deep'})")
        print(f"    lesion dissipation carried by: " + ", ".join(f"{k} {v:+.2e}" for k, v in top))
        print(f"    psi healthy : {_psi(eh, psi_keys)}")
        print(f"    psi diseased: {_psi(ed, psi_keys)}")

    print("\n" + "=" * 86)
    print("delta-sigma isolates the disease's thermodynamic footprint from the (shared) supply gradient.")


if __name__ == "__main__":
    main()

