# ABC-SMC overnight calibration

A long-running, parallelized Approximate Bayesian Computation — Sequential Monte Carlo run that tightens the posteriors over the simulator's seven tunable parameters by repeatedly drawing parameter vectors, simulating, and accepting those whose summary statistics match a target observation within a decreasing tolerance.

## What it does

For each of N generations:

1. Draw a batch of parameter vectors (from the prior at gen 0, or by resampling + perturbing the previous generation at gen ≥ 1).
2. For each parameter vector, run a simulator instance for `inner_ticks` ticks in a parallel worker pool.
3. Compute the summary statistics fingerprint of the resulting trajectory.
4. Compare against the target observation via the simulator-side distance function.
5. Accept particles where the distance is below the generation's epsilon threshold.
6. Save a checkpoint.

At the end, you get the joint posterior over the seven parameters, an acceptance trace, an epsilon annealing curve, and a convergence plot.

## The seven parameters under calibration

| Parameter | Prior range | Layer | What it controls |
|---|---|---|---|
| `saliva_O2` | [4, 14] | Chemistry | Atmospheric O₂ tension at the biofilm surface |
| `saliva_buffer` | [0.01, 0.20] | Chemistry | Salivary buffering capacity against pH excursions |
| `D_heme` | [0.1, 1.5] | Chemistry | Heme diffusion rate through the biofilm matrix |
| `stephan_amplitude` | [0.3, 2.5] | Chemistry | pH drop magnitude after a sugar pulse |
| `stephan_period` | [60, 720] | Chemistry | Meal-cycle period in ticks |
| `division_threshold` | [0.15, 0.55] | Agents | Fitness threshold below which agents stop dividing |
| `viability_grace_ticks` | [20, 200] | Agents | Off-envelope tolerance before agent death |

## Run it

```bash
cd oral_biofilm_sim

# Smoke test — 5–10 minutes, sanity check only
python -m sim.calibration.run_overnight --quick

# Production overnight run — ~8 hours on an 8-core machine
python -m sim.calibration.run_overnight

# Resume after a crash or Ctrl-C
python -m sim.calibration.run_overnight --resume

# Explicit worker count
python -m sim.calibration.run_overnight --workers 16

# Use a saved target observation instead of regenerating the reference
python -m sim.calibration.run_overnight --target obs.pkl
```

## What you get back

`calibration_output_overnight/` contains:

- `history.pkl` — pickled list of `Generation` objects (resume checkpoint and full audit trail)
- `posterior_summary.txt` — text dump of mean / std / median / 5th–95th percentile per parameter
- `posterior_marginals.png` — histogram per parameter, with prior bounds overlaid
- `acceptance_trace.png` — acceptance rate per generation
- `epsilon_trajectory.png` — how the tolerance shrinks
- `distance_convergence.png` — best and mean distance per generation

## Wall-time expectations

The dominant cost is the inner simulator run. At 1500 ticks with 5000 agents:

- Single inner run: ~30–90 seconds depending on the parameter vector (some explore high-density regimes that take longer per tick)
- 200 particles × 8 generations = 1600 accepted particles
- With a ~20% acceptance rate at later generations, that's ~8000 inner runs
- 8 cores → 1000 sequential inner runs per core → roughly 8–10 hours

The `--quick` config completes in 5–10 minutes and is sufficient to verify the pipeline runs end-to-end.

## What "calibrated" looks like

The reference observation is generated from a long simulator run at the *current* default parameters. If the inference machinery is well-specified, the posterior should concentrate around those defaults. A wide or multi-modal posterior on any parameter indicates that the summary statistics do not constrain that parameter — the prior survives, which is a useful negative result (it tells you which parameters need richer summary stats or richer observations).

When the posterior tightens around values *different* from the defaults, the system is telling you the current defaults don't actually reproduce the target observation — and the new mean is a better starting point for the next round of work.

## Pointing it at real data

The current target is a synthetic reference. Pointing the calibration at clinical data requires:

1. Producing a `SummaryStats` from a real 16S / shotgun composition table plus measured pH / NO / H₂S / OMV cargo where available.
2. Pickling that to `obs.pkl`.
3. Running `--target obs.pkl`.

The summary-stat compute path in `summary_stats.py` is intentionally written to be reusable for real-data inputs — the same fields (Shannon, pioneer fraction, red complex fraction, Fn betweenness, pH mean / variance, etc.) can be computed from a clinical sample using the standard bioinformatic pipeline.

## Checkpointing and crash recovery

The runner saves `history.pkl` after each completed generation. If the process dies mid-generation, the partially-completed generation is lost; everything before it is preserved. `--resume` picks up at the next uncompleted generation and continues.

Ctrl-C is handled gracefully — the runner saves a checkpoint before exit, and `--resume` will continue from where you stopped.

## Caveats

- The reference observation is itself a single noisy realization of the simulator. The "true" parameter values it was generated from are perfectly recoverable in principle but not perfectly recoverable in practice — there is irreducible simulator-side stochasticity that puts a floor on the achievable posterior tightness.
- The summary-stat distance function in `summary_stats.distance` is weighted heuristically. Re-weighting the components can dramatically change which parameters get sharp posteriors. The current weights prioritize topological + functional fingerprints (pioneer fraction, red complex fraction, Fn betweenness) over compositional details (per-species abundance).
- High-dimensional posteriors are not visualized as full corner plots in the default dashboard (would need `corner.py` dependency). Marginals only. For the joint structure, load `history.pkl` and use the notebook in `notebooks/abc_posterior_analysis.ipynb` (TODO).

