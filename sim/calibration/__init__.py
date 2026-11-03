"""ABC-SMC calibration layer.

Approximate Bayesian Computation - Sequential Monte Carlo for fitting
simulator parameters against summary statistics of real oral microbiome data.

Why ABC-SMC and not MCMC:
    The simulator has no tractable likelihood. ABC lets us define a
    summary-statistic distance and a tolerance schedule, then sample from
    the implicit posterior by simulating.
    SMC adds a particle population + adaptive epsilon schedule for efficiency.
"""
from .summary_stats import SummaryStats, compute_summary_stats
from .priors import ParameterPrior, default_priors
from .abc_smc import ABCSMCConfig, ABCSMC

__all__ = [
    "SummaryStats", "compute_summary_stats",
    "ParameterPrior", "default_priors",
    "ABCSMCConfig", "ABCSMC",
]

