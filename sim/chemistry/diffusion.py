"""Reaction-diffusion solver.

Explicit Euler on each chemistry field: c_{t+1} = c_t + D * laplacian(c_t).
Stable for dt * D / dx^2 <= 0.5 — at our cell size and unit-time tick, this is fine.
Source/sink terms are applied separately by agents emitting via grid.add_to_field_at().
"""
from __future__ import annotations
import numpy as np
from ..grid.base import SpatialGrid
from ..config import ChemistryConfig


def laplacian_1d(arr: np.ndarray) -> np.ndarray:
    """Discrete Laplacian with no-flux (zero-derivative) boundary conditions."""
    out = np.zeros_like(arr)
    out[1:-1] = arr[:-2] - 2 * arr[1:-1] + arr[2:]
    # Zero-flux at boundaries: reflect via ghost cell equal to adjacent cell
    out[0] = arr[1] - arr[0]
    out[-1] = arr[-2] - arr[-1]
    return out


class Diffusion1D:
    """Diffuses every chemistry field on a 1D grid each tick.

    Boundary fluxes (saliva inflow at top, heme injection at bottom) are applied
    by BoundaryConditions before this step.
    """

    def __init__(self, cfg: ChemistryConfig):
        self.cfg = cfg
        self._D = {
            "O2": cfg.D_O2,
            "pH": cfg.D_pH,
            "sugar": cfg.D_sugar,
            "heme": cfg.D_heme,
            "lactate": cfg.D_lactate,
            # NO is a sharp point source (discrete reducer cells); cap its diffusion at
            # the checkerboard-stable limit (D<=0.25 → |1-4D|<=1) to avoid Nyquist blowup.
            "NO": min(cfg.D_NO, 0.25),
            "H2S": cfg.D_H2S,
            "NH3": cfg.D_NH3,
            # secondary fermentation products — moderate diffusion
            "propionate": 0.7,
            "butyrate": 0.7,
            "acetate": 0.7,
            # nitrogen / NO axis — point-source delivery/consumption; keep D in the
            # stable range so the explicit Laplacian doesn't excite the checkerboard mode.
            "nitrate": 0.25,
            "nitrite": 0.25,
        }
        # STABILITY: the explicit forward-Euler Laplacian (arr += D*laplacian) amplifies the
        # Nyquist/checkerboard mode unless |1-4D| <= 1, i.e. D <= 0.5. Several catalogue D values
        # (O2=1.0, pH=0.6, sugar=0.8, lactate/NH3/SCFA=0.7-0.9) violated this and were oscillating
        # — producing a numerically oxygenated, acidic deep pocket. Cap all D at 0.25 (fully damps
        # the checkerboard, factor 0) so the depth gradients are physical (anoxic, near-neutral base).
        self._D = {name: min(D, 0.25) for name, D in self._D.items()}

    # Physical clamp ranges per field — keeps the model numerically sane.
    _CLAMP = {
        "O2":         (0.0, 25.0),
        "pH":         (3.0, 9.0),
        "sugar":      (0.0, 5.0),
        "heme":       (0.0, 2.0),
        "lactate":    (0.0, 3.0),
        "NO":         (0.0, 2.0),
        "H2S":        (0.0, 2.0),
        "NH3":        (0.0, 2.0),
        "propionate": (0.0, 3.0),
        "butyrate":   (0.0, 3.0),
        "acetate":    (0.0, 3.0),
        "nitrate":    (0.0, 5.0),
        "nitrite":    (0.0, 3.0),
    }

    def step(self, grid: SpatialGrid) -> None:
        for name, D in self._D.items():
            arr = grid.get_field(name)
            arr += D * laplacian_1d(arr)
            # Decay / consumption of volatile reactive species
            if name == "NO":
                arr *= 0.85     # NO is highly reactive / very short-lived → fast turnover
            elif name == "H2S":
                arr *= 0.99
            elif name in ("lactate", "propionate", "butyrate", "acetate", "NH3"):
                arr *= 0.995    # slow turnover (consumption by community + drainage)
            elif name == "sugar":
                arr *= 0.97     # rapid consumption + clearance
            elif name == "nitrite":
                arr *= 0.98     # reactive: -> NO and swallowed/absorbed (systemic)
            # pH gradual relaxation toward 6.8 (community buffering, not just saliva)
            if name == "pH":
                arr += 0.005 * (6.8 - arr)
            # Hard physical clamp
            lo, hi = self._CLAMP[name]
            np.clip(arr, lo, hi, out=arr)

