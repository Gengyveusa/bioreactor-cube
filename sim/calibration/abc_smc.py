"""Approximate Bayesian Computation — Sequential Monte Carlo.

Toni & Stumpf (2009) scheme:

    1. At population 1, sample particles from the prior; accept if
       d(simulate(theta), obs) < epsilon_1.
    2. At population t > 1, resample particles from the previous accepted
       set weighted by w_{t-1}; perturb via a kernel; accept if
       d < epsilon_t.
    3. Epsilon schedule decreases (e.g. by quantile of accepted distances
       at each step) until tolerance threshold or budget hits.

For oral_biofilm_sim, "simulate" runs the full ABM forward. That's slow
(~1500 ticks * 5000 agents * O(species, chem) work). For real-time
calibration, the neural surrogate (Layer 2 of the architecture) replaces
the ABM inside the ABC loop. For now, the inner-loop uses a SHORT run
(default 400 ticks) and a small population so we can validate the scaffold.

The summary-statistic distance is in summary_stats.distance.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import time
import numpy as np
import h5py

from ..config import SimConfig
from ..model import OralBiofilmModel
from .priors import ParameterPrior, default_priors, sample_parameter_vector, apply_parameter_vector
from .summary_stats import SummaryStats, compute_summary_stats, distance


@dataclass
class ABCSMCConfig:
    n_particles: int = 30                # per generation
    n_generations: int = 4
    initial_epsilon: float = np.inf      # first gen accepts everything (broadest prior)
    epsilon_quantile: float = 0.5        # next gen epsilon = this quantile of prev distances
    inner_ticks: int = 400               # ticks per simulator inner-loop run
    inner_seed_per_particle: bool = True
    perturb_scale: float = 0.1
    max_attempts_per_particle: int = 10  # cap re-sampling
    base_seed: int = 42
    output_dir: str = "calibration_output"


@dataclass
class Particle:
    theta: np.ndarray
    distance: float
    weight: float = 1.0
    summary: SummaryStats | None = None


@dataclass
class Generation:
    index: int
    epsilon: float
    particles: list[Particle] = field(default_factory=list)
    n_attempts: int = 0
    wall_time_s: float = 0.0


class ABCSMC:
    def __init__(self,
                 obs: SummaryStats,
                 priors: list[ParameterPrior] | None = None,
                 cfg: ABCSMCConfig | None = None):
        self.obs = obs
        self.priors = priors or default_priors()
        self.cfg = cfg or ABCSMCConfig()
        self.rng = np.random.default_rng(self.cfg.base_seed)
        self.history: list[Generation] = []
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # inner loop: simulate one parameter vector, return distance
    # ------------------------------------------------------------
    def _simulate_and_score(self, theta: np.ndarray, particle_seed: int) -> tuple[float, SummaryStats]:
        import tempfile, os
        sim_cfg = SimConfig()
        sim_cfg.ticks = self.cfg.inner_ticks
        sim_cfg.seed = particle_seed
        sim_cfg.log_every = max(10, sim_cfg.ticks // 30)
        apply_parameter_vector(sim_cfg, self.priors, theta)
        model = OralBiofilmModel(sim_cfg)
        traj = model.run()
        arr = traj.to_arrays()
        if not arr:
            return float("inf"), SummaryStats(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        # tempfile.NamedTemporaryFile in /tmp (cross-platform safe, OS cleans it)
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
            tmp_path = tf.name
        try:
            with h5py.File(tmp_path, "w") as f:
                for k, v in arr.items():
                    f.create_dataset(k, data=v)
            s = compute_summary_stats(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        d = distance(s, self.obs)
        return d, s

    # ------------------------------------------------------------
    # Generations
    # ------------------------------------------------------------
    def _sample_initial(self, eps: float) -> Generation:
        gen = Generation(index=0, epsilon=eps)
        t0 = time.time()
        attempts = 0
        seed_counter = self.cfg.base_seed * 1000 + 1
        while len(gen.particles) < self.cfg.n_particles:
            theta = sample_parameter_vector(self.priors, self.rng)
            seed_counter += 1
            d, s = self._simulate_and_score(theta, seed_counter)
            attempts += 1
            if d <= eps:
                gen.particles.append(Particle(theta=theta, distance=d, weight=1.0, summary=s))
        gen.n_attempts = attempts
        gen.wall_time_s = time.time() - t0
        return gen

    def _resample_and_perturb(self, prev: Generation, eps: float, gen_idx: int) -> Generation:
        gen = Generation(index=gen_idx, epsilon=eps)
        t0 = time.time()
        weights = np.array([p.weight for p in prev.particles], dtype=np.float64)
        weights = weights / max(weights.sum(), 1e-9)
        attempts = 0
        seed_counter = self.cfg.base_seed * 1000 + gen_idx * 10000 + 1
        while len(gen.particles) < self.cfg.n_particles:
            idx = self.rng.choice(len(prev.particles), p=weights)
            base = prev.particles[idx]
            theta = np.array([
                self.priors[i].perturb(base.theta[i], self.cfg.perturb_scale, self.rng)
                for i in range(len(self.priors))
            ])
            seed_counter += 1
            d, s = self._simulate_and_score(theta, seed_counter)
            attempts += 1
            if d <= eps:
                gen.particles.append(Particle(theta=theta, distance=d, weight=1.0, summary=s))
            if attempts > self.cfg.max_attempts_per_particle * self.cfg.n_particles:
                # Budget exhausted — exit with what we have
                break
        gen.n_attempts = attempts
        gen.wall_time_s = time.time() - t0
        return gen

    # ------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------
    def run(self) -> list[Generation]:
        print(f"ABC-SMC: {self.cfg.n_generations} generations, "
              f"{self.cfg.n_particles} particles each, "
              f"{self.cfg.inner_ticks}-tick inner runs")
        eps = self.cfg.initial_epsilon
        print(f"[gen 0] sampling from prior, epsilon = {eps}")
        gen0 = self._sample_initial(eps)
        print(f"  accepted {len(gen0.particles)}/{gen0.n_attempts} in {gen0.wall_time_s:.1f}s")
        self.history.append(gen0)
        for g in range(1, self.cfg.n_generations):
            prev_distances = np.array([p.distance for p in self.history[-1].particles])
            new_eps = float(np.quantile(prev_distances, self.cfg.epsilon_quantile))
            print(f"[gen {g}] new epsilon = {new_eps:.4f}")
            gen = self._resample_and_perturb(self.history[-1], new_eps, g)
            print(f"  accepted {len(gen.particles)}/{gen.n_attempts} in {gen.wall_time_s:.1f}s")
            self.history.append(gen)
        return self.history

    # ------------------------------------------------------------
    # report
    # ------------------------------------------------------------
    def posterior_summary(self) -> dict:
        if not self.history:
            return {}
        last = self.history[-1]
        thetas = np.stack([p.theta for p in last.particles])
        return {
            "n_particles": len(last.particles),
            "epsilon": last.epsilon,
            "mean": dict(zip([p.name for p in self.priors], thetas.mean(axis=0))),
            "std":  dict(zip([p.name for p in self.priors], thetas.std(axis=0))),
            "median": dict(zip([p.name for p in self.priors], np.median(thetas, axis=0))),
            "best_distance": min(p.distance for p in last.particles),
        }

