"""Parameter priors for ABC-SMC.

Each prior specifies the range of values to explore for one tunable
simulator parameter. Priors are uniform by default — the ABC posterior
finds the regions that match data, so uninformative priors are fine.

Conventions:
    - All priors are on simulator config parameters (sim.config)
    - Each prior knows how to (a) sample a value, (b) inject the value
      into a SimConfig
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

from ..config import SimConfig


@dataclass
class ParameterPrior:
    name: str
    low: float
    high: float
    setter: Callable[[SimConfig, float], None]  # how to inject value into config

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))

    def perturb(self, value: float, scale: float, rng: np.random.Generator) -> float:
        """Gaussian perturbation kernel for SMC propagation, reflected at bounds."""
        new = value + rng.normal(0.0, scale * (self.high - self.low))
        # Reflect at boundaries
        if new < self.low:
            new = self.low + (self.low - new)
        if new > self.high:
            new = self.high - (new - self.high)
        return float(np.clip(new, self.low, self.high))


def default_priors() -> list[ParameterPrior]:
    """The first-pass calibration set.

    These are the 7 parameters most likely to drive the qualitative
    behaviour the simulator should reproduce. ABC posteriors over THESE
    are what calibrate the v1 model.
    """
    return [
        ParameterPrior(
            "saliva_O2", 4.0, 14.0,
            lambda c, v: setattr(c.chemistry, "saliva_O2", v)),
        ParameterPrior(
            "saliva_buffer", 0.01, 0.20,
            lambda c, v: setattr(c.chemistry, "saliva_buffer", v)),
        ParameterPrior(
            "D_heme", 0.1, 1.5,
            lambda c, v: setattr(c.chemistry, "D_heme", v)),
        ParameterPrior(
            "stephan_amplitude", 0.3, 2.5,
            lambda c, v: setattr(c.chemistry, "stephan_amplitude", v)),
        ParameterPrior(
            "stephan_period", 60, 720,
            lambda c, v: setattr(c.chemistry, "stephan_period", int(v))),
        ParameterPrior(
            "division_threshold", 0.15, 0.55,
            lambda c, v: setattr(c.agents, "division_threshold", v)),
        ParameterPrior(
            "viability_grace_ticks", 20, 200,
            lambda c, v: setattr(c.agents, "viability_grace_ticks", int(v))),
    ]


def sample_parameter_vector(priors: list[ParameterPrior],
                            rng: np.random.Generator) -> np.ndarray:
    return np.array([p.sample(rng) for p in priors])


def apply_parameter_vector(cfg: SimConfig, priors: list[ParameterPrior],
                           theta: np.ndarray) -> None:
    for prior, val in zip(priors, theta):
        prior.setter(cfg, val)

