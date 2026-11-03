"""Counterfactual runner — paired observational + interventional simulations.

Same seed, same initial conditions, two runs:
  - observational  : the simulator runs forward with no intervention
  - interventional : the same simulator runs forward with do(intervention)

ATE (Average Treatment Effect) is the difference in observables. With
enough paired runs at different seeds, you get the distribution of
treatment effects, which is what a clinician actually needs (mean ATE
with confidence intervals).

For Phase-1 commercial conversations, a single paired run is enough to
demonstrate direction and approximate magnitude. ABC-calibrated runs
upgrade this to a real prediction with intervals.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import copy
import numpy as np
import h5py

from ..config import SimConfig
from ..model import OralBiofilmModel
from .dag import Intervention


# ---------------------------------------------------------------------------
# Per-run output bundle
# ---------------------------------------------------------------------------
@dataclass
class CounterfactualRun:
    name: str
    cfg: SimConfig
    intervention: Intervention | None
    dysbiosis_end: float
    dysbiosis_max: float
    fn_betweenness_max: float
    pg_betweenness_max: float
    pioneer_fraction_end: float
    omv_flux_proxy: float
    pH_variance_mean: float
    no_total_mean: float
    n_live_agents: int
    ticks_run: int
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class ATEResult:
    intervention_name: str
    obs: CounterfactualRun
    iv: CounterfactualRun
    # The ATE table — delta = interventional - observational
    delta: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 64,
            f"ATE · {self.intervention_name}",
            "=" * 64,
            f"{'observable':<28}  {'observ':>10}  {'interv':>10}  {'delta':>10}",
            "-" * 64,
        ]
        for k, v in self.delta.items():
            o = getattr(self.obs, k, None)
            i = getattr(self.iv, k, None)
            if o is None or i is None:
                continue
            lines.append(f"{k:<28}  {o:>10.4f}  {i:>10.4f}  {v:>+10.4f}")
        lines.append("=" * 64)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Apply per-species modifiers to a model instance (post-construction patch)
# ---------------------------------------------------------------------------
def _apply_species_modifiers(model: OralBiofilmModel, modifiers: dict[str, dict[str, float]]) -> None:
    """Attach per-species multipliers to the model.

    The agent's step() reads these via `model.species_modifiers.get(species_id, {})`
    and folds them into baseline_mortality and surface_division logic. If the
    agent doesn't know about a modifier key, it's silently ignored — gracefully
    degrades when new mechanisms are added.
    """
    # Merge modifiers per species; later writes override earlier ones (Python dicts)
    merged: dict[str, dict[str, float]] = {}
    for species_id, mods in modifiers.items():
        merged.setdefault(species_id, {}).update(mods)
    setattr(model, "species_modifiers", merged)


# ---------------------------------------------------------------------------
# Run paired simulations
# ---------------------------------------------------------------------------
def run_paired(
    base_cfg: SimConfig,
    intervention: Intervention,
    seed: int | None = None,
    apply_at_tick: int = 0,
) -> ATEResult:
    """Run observational + interventional sims with matched seed; return ATE.

    apply_at_tick controls when the intervention activates:
      - 0       : applied at t=0 (preventive scenario)
      - >0      : observational baseline established first, then intervention
                  applied at that tick (therapeutic scenario). Phase-1 supports
                  t=0 only; therapeutic-mode dispatch is a Phase-2 add-on.
    """
    if seed is not None:
        base_cfg = copy.deepcopy(base_cfg)
        base_cfg.seed = seed

    # --- Observational run
    obs_model = OralBiofilmModel(copy.deepcopy(base_cfg))
    obs_traj = obs_model.run()

    # --- Interventional run (same seed, modified config + species modifiers)
    iv_cfg = intervention.apply_to_config(base_cfg)
    iv_model = OralBiofilmModel(iv_cfg)
    if intervention.species_modifiers:
        _apply_species_modifiers(iv_model, intervention.species_modifiers)
    iv_traj = iv_model.run()

    # --- Summarize each
    def summarize(model: OralBiofilmModel, traj, label: str, iv_ref: Intervention | None) -> CounterfactualRun:
        arr = traj.to_arrays() if traj.ticks else {}
        if not arr:
            return CounterfactualRun(
                name=label,
                cfg=model.cfg,
                intervention=iv_ref,
                dysbiosis_end=0.0,
                dysbiosis_max=0.0,
                fn_betweenness_max=0.0,
                pg_betweenness_max=0.0,
                pioneer_fraction_end=0.0,
                omv_flux_proxy=0.0,
                pH_variance_mean=0.0,
                no_total_mean=0.0,
                n_live_agents=len(model.agents_alive),
                ticks_run=model.tick,
            )
        # Pioneer fraction at end
        comp = arr["composition"]  # (T, n_species, n_depths)
        # Find pioneer species indices from CATALOG
        from ..species import CATALOG
        pioneer_idx = [i for i, s in enumerate(CATALOG) if s.archetype == "Pioneer"]
        total_end = float(comp[-1].sum())
        pioneer_end = float(sum(comp[-1, i].sum() for i in pioneer_idx))
        pioneer_frac = pioneer_end / max(total_end, 1e-9)
        # OMV flux proxy: gingipain-cargo events accumulated through trajectory
        omv_proxy = float(len([e for tk, e in model.omv_log if "gingipains_Kgp" in e.cargo or "gingipains_Rgp" in e.cargo]))
        return CounterfactualRun(
            name=label,
            cfg=model.cfg,
            intervention=iv_ref,
            dysbiosis_end=float(arr["dysbiosis"][-1]),
            dysbiosis_max=float(arr["dysbiosis"].max()),
            fn_betweenness_max=float(arr["Fn_betweenness"].max()),
            pg_betweenness_max=float(arr["Pg_betweenness"].max()),
            pioneer_fraction_end=pioneer_frac,
            omv_flux_proxy=omv_proxy,
            pH_variance_mean=float(arr["pH_variance"].mean()),
            no_total_mean=float(arr["NO"].mean()),
            n_live_agents=len(model.agents_alive),
            ticks_run=model.tick,
        )

    obs_run = summarize(obs_model, obs_traj, "observational", None)
    iv_run = summarize(iv_model, iv_traj, "interventional", intervention)

    return compute_ate(obs_run, iv_run, intervention.name)


def compute_ate(obs: CounterfactualRun, iv: CounterfactualRun, name: str) -> ATEResult:
    fields = [
        "dysbiosis_end", "dysbiosis_max",
        "fn_betweenness_max", "pg_betweenness_max",
        "pioneer_fraction_end", "omv_flux_proxy",
        "pH_variance_mean", "no_total_mean",
        "n_live_agents",
    ]
    delta = {}
    for f in fields:
        o = getattr(obs, f, 0.0)
        i = getattr(iv, f, 0.0)
        delta[f] = float(i - o)
    return ATEResult(intervention_name=name, obs=obs, iv=iv, delta=delta)

