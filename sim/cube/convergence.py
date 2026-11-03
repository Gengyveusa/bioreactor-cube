"""Grid-convergence harness — does a readout describe the biology, or the lattice?

A discretised model is only a model of something if its answers stop changing as the mesh is
refined. Any readout that keeps moving is reporting the grid, not the tissue, and must not be
quoted. This module measures that directly.

The failure mode it was written to catch is dimensional: mixing EXTENSIVE quantities (sums over
voxels, which scale with voxel count) with INTENSIVE ones (means, fractions, concentrations, which
do not) inside a single expression. The product is then a function of resolution.

Run it:
    python3 -c "from sim.cube.convergence import report; report()"
"""

from __future__ import annotations

import time

from sim.cube.bodymap import BodyMap
from sim.cube.tissues import ARTERY_TISSUE, ORAL_HEALTHY, ORAL_TISSUE

# Readouts to track. Marked with what they SHOULD be if the model is mesh-independent.
TRACKED = ("rupture_index", "mmp_activity", "foam_fraction", "endothelial_activation",
           "ecm_density", "inflammation_load", "plaque_density", "biomass")

DIMS = (10, 14, 18, 24, 32)


def run_coupled(dim: int, *, dysbiotic: bool = True, steps: int = 200, dt: float = 0.1,
                k: float = 2.5) -> dict:
    """One coupled oral->artery run at a given lattice size. Returns the artery readout + efflux.

    Example: run_coupled(16)["rupture_index"] -> a float
    """
    bm = BodyMap()
    bm.place("oral", ORAL_TISSUE if dysbiotic else ORAL_HEALTHY, dim=dim)
    bm.place("artery", ARTERY_TISSUE, dim=dim)
    bm.link("oral", "artery", k=k)
    for _ in range(steps):
        bm.step(dt)
    out = {kk: v for kk, v in bm.cubes["artery"].readout().items() if kk != "fields"}
    out["oral_efflux"] = round(float(bm.cubes["oral"].efflux()), 6)
    return out


def sweep(dims=DIMS, **kw) -> dict:
    """Run the same experiment across lattice sizes. Returns {dim: readout}."""
    out = {}
    for d in dims:
        t0 = time.time()
        out[d] = run_coupled(d, **kw)
        out[d]["_seconds"] = round(time.time() - t0, 1)
    return out


def variation(series: dict, key: str) -> float:
    """Relative spread of one readout across the sweep: (max-min)/|mean|. 0.0 = converged.

    Example: variation({10: {"x": 1.0}, 20: {"x": 1.0}}, "x") -> 0.0
    """
    vals = [float(r[key]) for r in series.values() if key in r]
    if not vals:
        return float("nan")
    lo, hi, mean = min(vals), max(vals), sum(vals) / len(vals)
    if abs(mean) < 1e-12:
        return 0.0 if hi - lo < 1e-12 else float("inf")
    return (hi - lo) / abs(mean)


def report(dims=DIMS, steps: int = 200) -> dict:
    """Print a convergence table for the coupled oral->artery model. Returns the raw sweep."""
    print("Grid-convergence sweep — coupled oral->artery, %d steps, biology held FIXED." % steps)
    print("Only the lattice changes. A readout that moves is reporting the mesh.\n")
    series = sweep(dims, steps=steps)

    width = max(len(k) for k in TRACKED) + 2
    header = "readout".ljust(width) + "".join(("dim%d" % d).rjust(12) for d in dims) + "   spread"
    print(header)
    print("-" * len(header))
    for key in TRACKED + ("oral_efflux",):
        row = key.ljust(width)
        for d in dims:
            v = series[d].get(key)
            row += ("%.4g" % v).rjust(12) if v is not None else "-".rjust(12)
        var = variation(series, key)
        flag = "  OK" if var < 0.10 else ("  DRIFT" if var < 0.5 else "  MESH-BOUND")
        row += "   %6.1f%%%s" % (100 * var, flag)
        print(row)
    print("\nruntime: " + ", ".join("dim%d=%ss" % (d, series[d]["_seconds"]) for d in dims))
    print("\nspread = (max-min)/mean across lattice sizes. <10% is usable; >50% means the number "
          "is a property of the grid, not the biology, and must not be published.")
    return series


def contrast(dims=DIMS, steps: int = 200) -> dict:
    """The diseased-vs-healthy contrast at each lattice size — the figure that gets quoted.

    A CONTRAST can survive mesh dependence that a raw value cannot, but only if it is checked.
    """
    print("\nDiseased vs healthy contrast across lattice sizes:\n")
    print("dim".ljust(6) + "diseased".rjust(12) + "healthy".rjust(12) + "ratio".rjust(10))
    print("-" * 40)
    out = {}
    for d in dims:
        dis = run_coupled(d, dysbiotic=True, steps=steps)["rupture_index"]
        hlt = run_coupled(d, dysbiotic=False, steps=steps)["rupture_index"]
        ratio = dis / hlt if hlt > 0 else float("inf")
        out[d] = {"diseased": dis, "healthy": hlt, "ratio": ratio}
        print(str(d).ljust(6) + ("%.3f" % dis).rjust(12) + ("%.3f" % hlt).rjust(12)
              + ("%.2fx" % ratio).rjust(10))
    return out


if __name__ == "__main__":
    report()
    contrast()

