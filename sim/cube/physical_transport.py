"""Dimensioned, conservative passive diffusion on a fixed 3-D cell-centred domain.

Space: micrometres; time: seconds; D: micrometres squared / second.
Concentration retains the caller's units. This does not calibrate legacy biology.
Closed faces are zero-flux; periodic axes wrap. No reactions or boundary forcing.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class TransportDomain:
    shape: tuple[int, int, int]
    lengths_um: tuple[float, float, float]
    periodic_axes: tuple[int, ...] = (1,)

    def __post_init__(self):
        if len(self.shape) != 3 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in self.shape):
            raise ValueError('shape must contain three positive integers')
        if len(self.lengths_um) != 3 or any(not math.isfinite(x) or x <= 0 for x in self.lengths_um):
            raise ValueError('lengths_um must contain three positive finite lengths')
        if len(set(self.periodic_axes)) != len(self.periodic_axes) or any(a not in (0, 1, 2) for a in self.periodic_axes):
            raise ValueError('periodic_axes must be unique axes in 0,1,2')

    @property
    def spacing_um(self):
        return tuple(l/n for l,n in zip(self.lengths_um,self.shape))

    @property
    def voxel_volume_um3(self):
        return math.prod(self.spacing_um)


def diffuse(concentration, domain: TransportDomain, diffusivity_um2_s: float,
            duration_s: float, *, safety: float = 0.9):
    """Return (new array, receipt), with CFL-controlled explicit Euler substeps.

    The six-face flux balance is conservative and positivity preserving under
    the selected CFL bound. No post-step clipping masks loss of mass.
    Input is never mutated. Negative/non-finite initial concentrations are errors.
    """
    c=np.array(concentration,dtype=float,copy=True)
    if c.shape != domain.shape or not np.isfinite(c).all() or np.any(c < 0):
        raise ValueError('Concentration must be finite, nonnegative and match domain.shape')
    d=float(diffusivity_um2_s); t=float(duration_s)
    if not math.isfinite(d) or d < 0 or not math.isfinite(t) or t < 0:
        raise ValueError('Diffusivity and duration must be finite and nonnegative')
    if not math.isfinite(safety) or not 0 < safety <= 1:
        raise ValueError('CFL safety must be in (0,1]')
    rate=2*d*sum(1/h**2 for n,h in zip(domain.shape,domain.spacing_um) if n>1)
    bound=math.inf if rate==0 else safety/rate
    steps=0 if t==0 or d==0 or rate==0 else max(1,math.ceil(t/bound))
    if steps > 100000:
        raise ValueError('More than 100000 substeps requested; reduce duration or grid resolution')
    dt=t/steps if steps else 0.0
    mass0=float(c.sum()*domain.voxel_volume_um3)
    for _ in range(steps):
        delta=np.zeros_like(c)
        for axis,h in enumerate(domain.spacing_um):
            if domain.shape[axis]==1:
                continue
            left=np.roll(c,1,axis=axis)
            right=np.roll(c,-1,axis=axis)
            if axis not in domain.periodic_axes:
                lo=[slice(None)]*3;lo[axis]=0
                hi=[slice(None)]*3;hi[axis]=-1
                left[tuple(lo)]=c[tuple(lo)]
                right[tuple(hi)]=c[tuple(hi)]
            delta += (left-2*c+right)/h**2
        c += dt*d*delta
    if not np.isfinite(c).all() or np.any(c < -1e-14):
        raise FloatingPointError('Transport lost positivity or produced non-finite output')
    mass1=float(c.sum()*domain.voxel_volume_um3)
    return c, {
        'shape':list(domain.shape),'lengths_um':list(domain.lengths_um),
        'spacing_um':list(domain.spacing_um),'periodic_axes':list(domain.periodic_axes),
        'closed_axes':[a for a in (0,1,2) if a not in domain.periodic_axes],
        'diffusivity_um2_s':d,'elapsed_seconds':t,'substeps':steps,'substep_seconds':dt,
        'cfl_safety':safety,'mass_initial':mass0,'mass_final':mass1,
        'mass_relative_error':abs(mass1-mass0)/max(abs(mass0),1e-300),
        'mass_units':'input_concentration_units * um^3',
        'minimum':float(c.min()),'maximum':float(c.max()),
        'scope':'passive_transport_only','legacy_biology_clock':'not_calibrated',
    }

