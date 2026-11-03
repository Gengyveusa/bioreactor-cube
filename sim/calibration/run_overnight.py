"""Overnight ABC-SMC calibration runner.

Wraps the single-threaded ABC-SMC scaffold with:
  • Multiprocessing parallel workers (one simulator run per CPU core)
  • Per-generation checkpointing — resume after crash, no work lost
  • Posterior dashboard at the end (corner plot, acceptance trace,
    distance trajectory, summary-stat match)
  • Sensible overnight defaults: 200 particles × 8 generations × 1500
    inner ticks. On an 8-core machine, ~8 hours wall time.

Usage
-----
    cd oral_biofilm_sim
    python -m sim.calibration.run_overnight                  # use defaults
    python -m sim.calibration.run_overnight --quick          # smoke-test config
    python -m sim.calibration.run_overnight --resume         # continue last run
    python -m sim.calibration.run_overnight --workers 16     # explicit core count
    python -m sim.calibration.run_overnight --target ref     # use the canonical reference observation
    python -m sim.calibration.run_overnight --target FILE    # use a saved SummaryStats pickle

The reference observation defaults to a long run of the simulator at
known healthy-eubiotic parameters. The ABC posterior over the priors
should concentrate around those parameters if the calibration is well-
specified — a sanity check that the inference machinery works before
we point it at real clinical data.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import SimConfig
from ..model import OralBiofilmModel
from .abc_smc import ABCSMC, ABCSMCConfig, Generation, Particle
from .priors import (
    ParameterPrior,
    apply_parameter_vector,
    default_priors,
    sample_parameter_vector,
)
from .summary_stats import SummaryStats, compute_summary_stats, distance


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------
@dataclass
class OvernightConfig:
    """Overnight defaults — tighter posteriors at the cost of wall time."""
    n_particles: int = 200          # was 30 — sharpens the posterior
    n_generations: int = 8           # was 4 — more annealing of epsilon
    inner_ticks: int = 1500          # was 400 — converges to steady state
    epsilon_quantile: float = 0.5
    perturb_scale: float = 0.1
    max_attempts_multiplier: int = 10
    base_seed: int = 42
    workers: int = max(1, (os.cpu_count() or 4) - 1)
    output_dir: str = "calibration_output_overnight"


@dataclass
class QuickConfig(OvernightConfig):
    """Smoke-test config — 5–10 minute run, sanity-check only."""
    n_particles: int = 20
    n_generations: int = 2
    inner_ticks: int = 200
    output_dir: str = "calibration_output_quick"


# ----------------------------------------------------------------------
# Reference observation
# ----------------------------------------------------------------------
def reference_observation(seed: int = 42, ticks: int = 2000) -> SummaryStats:
    """Generate the canonical reference SummaryStats from a known-healthy run.

    This is the "truth" the ABC posterior should rediscover. Used as a
    sanity check — if the posterior doesn't concentrate near the
    parameters used to produce this observation, the inference machinery
    is mis-specified somewhere.
    """
    import tempfile
    import h5py

    cfg = SimConfig()
    cfg.ticks = ticks
    cfg.seed = seed
    cfg.log_every = max(10, ticks // 40)
    cfg.chemistry.heme_bleed_rate = 0.0  # eubiotic — no bleed ignition
    cfg.agents.initial_inoculum_per_species = 8
    cfg.agents.max_agents = 5000

    model = OralBiofilmModel(cfg)
    traj = model.run()
    arr = traj.to_arrays()

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
        tmp = tf.name
    try:
        with h5py.File(tmp, "w") as f:
            for k, v in arr.items():
                f.create_dataset(k, data=v)
        return compute_summary_stats(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ----------------------------------------------------------------------
# Worker — pickled function for mp.Pool
# ----------------------------------------------------------------------
def _worker_simulate(args: tuple) -> tuple[np.ndarray, float, dict]:
    """One particle: simulate, score, return (theta, distance, summary dict).

    Module-level function (not a closure) so multiprocessing can pickle it.
    """
    import tempfile
    import h5py

    theta, particle_seed, inner_ticks, prior_specs, obs_vec = args

    # Rebuild priors from a serializable spec
    priors = [
        ParameterPrior(
            name=s["name"],
            low=s["low"],
            high=s["high"],
            setter=_lookup_setter(s["name"]),
        )
        for s in prior_specs
    ]

    cfg = SimConfig()
    cfg.ticks = inner_ticks
    cfg.seed = particle_seed
    cfg.log_every = max(10, inner_ticks // 30)
    apply_parameter_vector(cfg, priors, np.asarray(theta))

    try:
        model = OralBiofilmModel(cfg)
        traj = model.run()
        arr = traj.to_arrays()
        if not arr:
            return theta, float("inf"), {}

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
            tmp = tf.name
        try:
            with h5py.File(tmp, "w") as f:
                for k, v in arr.items():
                    f.create_dataset(k, data=v)
            s = compute_summary_stats(tmp)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

        # Compare via the simulator-side distance
        from .summary_stats import SummaryStats as _S
        obs_stats = _S(**{k: float(v) for k, v in zip(_S.field_names(), obs_vec)})
        d = distance(s, obs_stats)
        return theta, float(d), asdict(s)

    except Exception as e:
        # Particle failed — return inf distance so it's rejected
        return theta, float("inf"), {"error": str(e)[:200]}


def _lookup_setter(name: str):
    """Re-lookup a setter by parameter name (needed because lambdas don't pickle)."""
    for p in default_priors():
        if p.name == name:
            return p.setter
    raise KeyError(f"Unknown prior: {name}")


def _prior_specs(priors: list[ParameterPrior]) -> list[dict]:
    """Serializable form of the priors for pickling to workers."""
    return [{"name": p.name, "low": p.low, "high": p.high} for p in priors]


# ----------------------------------------------------------------------
# Parallel ABC-SMC
# ----------------------------------------------------------------------
class ParallelABCSMC:
    def __init__(self, obs: SummaryStats, priors: list[ParameterPrior],
                 cfg: OvernightConfig):
        self.obs = obs
        self.priors = priors
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.base_seed)
        self.history: list[Generation] = []
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self._ckpt_path = Path(cfg.output_dir) / "history.pkl"

    # ------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------
    def save(self) -> None:
        with open(self._ckpt_path, "wb") as f:
            pickle.dump({
                "history": self.history,
                "cfg": self.cfg,
                "obs": self.obs,
                "priors": _prior_specs(self.priors),
            }, f)

    def load(self) -> bool:
        if not self._ckpt_path.exists():
            return False
        with open(self._ckpt_path, "rb") as f:
            d = pickle.load(f)
        self.history = d["history"]
        self.obs = d["obs"]
        print(f"  resumed from {len(self.history)} completed generation(s)")
        return True

    # ------------------------------------------------------------
    # One generation — sample candidates in parallel
    # ------------------------------------------------------------
    def _run_generation(self, gen_idx: int, eps: float,
                        sample_one_theta) -> Generation:
        gen = Generation(index=gen_idx, epsilon=eps)
        t0 = time.time()
        n_attempts = 0
        seed_counter = self.cfg.base_seed * 1000 + gen_idx * 10000

        obs_vec = self.obs.to_array().tolist()
        specs = _prior_specs(self.priors)
        attempt_cap = self.cfg.max_attempts_multiplier * self.cfg.n_particles

        with mp.Pool(processes=self.cfg.workers) as pool:
            while len(gen.particles) < self.cfg.n_particles and n_attempts < attempt_cap:
                batch = min(
                    self.cfg.workers * 2,
                    self.cfg.n_particles - len(gen.particles),
                )
                args = []
                for _ in range(batch):
                    seed_counter += 1
                    theta = sample_one_theta()
                    args.append((
                        theta.tolist(),
                        seed_counter,
                        self.cfg.inner_ticks,
                        specs,
                        obs_vec,
                    ))

                for theta, d, summary_dict in pool.imap_unordered(_worker_simulate, args):
                    n_attempts += 1
                    if d <= eps and np.isfinite(d):
                        # Reconstruct SummaryStats if available
                        s = None
                        if summary_dict and "error" not in summary_dict:
                            try:
                                s = SummaryStats(**summary_dict)
                            except Exception:
                                s = None
                        gen.particles.append(Particle(
                            theta=np.asarray(theta),
                            distance=d,
                            weight=1.0,
                            summary=s,
                        ))

                # Progress line per batch
                print(f"    gen {gen_idx}: accepted {len(gen.particles)}/{self.cfg.n_particles} "
                      f"(attempts {n_attempts})", flush=True)

        gen.n_attempts = n_attempts
        gen.wall_time_s = time.time() - t0
        return gen

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------
    def run(self, resume: bool = False) -> list[Generation]:
        if resume:
            self.load()

        if len(self.history) >= self.cfg.n_generations:
            print("All generations already complete.")
            return self.history

        # Generation 0 (or first uncomputed): sample from prior
        if len(self.history) == 0:
            print(f"[gen 0] sampling from prior")
            sample_fn = lambda: sample_parameter_vector(self.priors, self.rng)
            gen0 = self._run_generation(0, np.inf, sample_fn)
            print(f"  accepted {len(gen0.particles)}/{gen0.n_attempts} in {gen0.wall_time_s:.1f}s")
            self.history.append(gen0)
            self.save()

        # Subsequent generations: resample + perturb
        while len(self.history) < self.cfg.n_generations:
            g = len(self.history)
            prev = self.history[-1]
            distances = np.array([p.distance for p in prev.particles])
            new_eps = float(np.quantile(distances, self.cfg.epsilon_quantile))
            print(f"[gen {g}] new epsilon = {new_eps:.4f}")

            weights = np.array([p.weight for p in prev.particles])
            weights = weights / max(weights.sum(), 1e-9)
            scale = self.cfg.perturb_scale

            def sample_fn(rng=self.rng, prev_particles=prev.particles,
                          weights=weights, priors=self.priors):
                idx = rng.choice(len(prev_particles), p=weights)
                base = prev_particles[idx]
                return np.array([
                    priors[i].perturb(base.theta[i], scale, rng)
                    for i in range(len(priors))
                ])

            gen = self._run_generation(g, new_eps, sample_fn)
            print(f"  accepted {len(gen.particles)}/{gen.n_attempts} in {gen.wall_time_s:.1f}s")
            self.history.append(gen)
            self.save()

        return self.history

    # ------------------------------------------------------------
    # Posterior summary
    # ------------------------------------------------------------
    def posterior_summary(self) -> dict:
        if not self.history:
            return {}
        last = self.history[-1]
        thetas = np.stack([p.theta for p in last.particles])
        names = [p.name for p in self.priors]
        return {
            "n_particles": len(last.particles),
            "epsilon": last.epsilon,
            "mean": dict(zip(names, thetas.mean(axis=0).tolist())),
            "std": dict(zip(names, thetas.std(axis=0).tolist())),
            "median": dict(zip(names, np.median(thetas, axis=0).tolist())),
            "q05": dict(zip(names, np.quantile(thetas, 0.05, axis=0).tolist())),
            "q95": dict(zip(names, np.quantile(thetas, 0.95, axis=0).tolist())),
            "best_distance": float(min(p.distance for p in last.particles)),
        }


# ----------------------------------------------------------------------
# Plot dashboard
# ----------------------------------------------------------------------
def plot_dashboard(history: list[Generation], priors: list[ParameterPrior],
                   out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    last = history[-1]
    thetas = np.stack([p.theta for p in last.particles])
    n_params = len(priors)
    names = [p.name for p in priors]

    # 1. Posterior marginals (one panel per parameter)
    fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(3 * (n_params + 1) // 2, 6))
    axes = axes.flatten() if n_params > 1 else [axes]
    for i, ax in enumerate(axes[:n_params]):
        ax.hist(thetas[:, i], bins=20, color="#1A4036", edgecolor="white", alpha=0.85)
        ax.axvspan(priors[i].low, priors[i].high, alpha=0.08, color="#C41E3A")
        ax.set_title(names[i], fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes[n_params:]:
        ax.set_visible(False)
    fig.suptitle("Posterior marginals (final generation)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "posterior_marginals.png", dpi=140)
    plt.close(fig)

    # 2. Acceptance rate per generation
    fig, ax = plt.subplots(figsize=(7, 4))
    accept_rates = [len(g.particles) / max(g.n_attempts, 1) for g in history]
    ax.plot(range(len(history)), accept_rates, "o-", color="#1A4036", lw=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Acceptance rate")
    ax.set_title("ABC-SMC acceptance trace")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "acceptance_trace.png", dpi=140)
    plt.close(fig)

    # 3. Epsilon trajectory (annealing schedule)
    fig, ax = plt.subplots(figsize=(7, 4))
    eps_history = [g.epsilon for g in history]
    eps_history = [v if np.isfinite(v) else None for v in eps_history]
    valid_x = [i for i, v in enumerate(eps_history) if v is not None]
    valid_y = [v for v in eps_history if v is not None]
    ax.plot(valid_x, valid_y, "o-", color="#C41E3A", lw=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("ε (acceptance threshold)")
    ax.set_title("Epsilon annealing")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "epsilon_trajectory.png", dpi=140)
    plt.close(fig)

    # 4. Best distance per generation
    fig, ax = plt.subplots(figsize=(7, 4))
    best_dists = [min(p.distance for p in g.particles) for g in history]
    mean_dists = [np.mean([p.distance for p in g.particles]) for g in history]
    ax.plot(range(len(history)), best_dists, "o-", color="#1A4036", lw=2, label="best")
    ax.plot(range(len(history)), mean_dists, "s--", color="#7FBCA9", lw=2, label="mean")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Distance to observation")
    ax.set_title("Convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "distance_convergence.png", dpi=140)
    plt.close(fig)

    print(f"  plots saved to {out_dir}/")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="Smoke-test config (20 particles × 2 gens × 200 ticks)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the last checkpoint")
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of parallel workers (default: cpu_count - 1)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Override output directory")
    ap.add_argument("--target", type=str, default="ref",
                    help='"ref" (compute reference observation) or a path to a pickled SummaryStats')
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip the plot dashboard at the end")
    args = ap.parse_args()

    cfg = QuickConfig() if args.quick else OvernightConfig()
    if args.workers is not None:
        cfg.workers = args.workers
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    # Resolve target observation
    if args.target == "ref":
        print("Generating reference observation from a known-healthy run...")
        obs = reference_observation()
    else:
        with open(args.target, "rb") as f:
            obs = pickle.load(f)
        print(f"Loaded target observation from {args.target}")

    priors = default_priors()

    print(f"ABC-SMC overnight run")
    print(f"  particles    : {cfg.n_particles}")
    print(f"  generations  : {cfg.n_generations}")
    print(f"  inner ticks  : {cfg.inner_ticks}")
    print(f"  workers      : {cfg.workers}")
    print(f"  output dir   : {cfg.output_dir}")
    print()

    runner = ParallelABCSMC(obs=obs, priors=priors, cfg=cfg)

    # Graceful Ctrl-C — save before exit
    def on_sigint(_sig, _frame):
        print("\n  caught Ctrl-C — saving checkpoint...")
        runner.save()
        sys.exit(130)
    signal.signal(signal.SIGINT, on_sigint)

    history = runner.run(resume=args.resume)

    print()
    print("Posterior summary:")
    summary = runner.posterior_summary()
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk:>24s}  {vv:.4f}" if isinstance(vv, float) else f"    {kk:>24s}  {vv}")
        else:
            print(f"  {k}: {v}")

    # Save text summary
    with open(Path(cfg.output_dir) / "posterior_summary.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    if not args.no_plot:
        print()
        print("Building plot dashboard...")
        plot_dashboard(history, priors, Path(cfg.output_dir))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

