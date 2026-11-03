# The Crossing v3: information, learning and collective behavior

The build uses the project's information-primary framework:

**S(t + Δt) = Fθ(S(t), I(t), E(t), T(t), Δt; seed)**

Information is the represented substrate. Available energy permits change, terrain constrains it, and elapsed time determines the observed system state. This is a modeling contract, not a demonstrated universal physical law. `data/learning/system_contract.json` declares meanings, units, unresolved terms and scale adapters.

## Working use case

An OMV-transport research copilot compares a trained prediction with the conservative simulator, challenges alternative explanations, and ranks candidate measurements that could distinguish assumed barrier support from altered clearance or transport. It does not establish HA treatment efficacy, classify vesicles from microscopy, predict patients, or discover molecular-weight-dependent biology. No measured training cases have been ingested. The image and pathway layers remain references.

Open `/journey`, select **Learning**, press **Predict & verify**, then **Choose next measurement**. Change settings and complete a paired simulation before repeating. The selected model is a three-member neural ensemble. Its outputs are the six competing mass fractions and two bounded response indices at sixteen times. Initial conditions and null/preintervention equality are enforced. An explicit conservative correction keeps cumulative uptake and clearance monotone and redistributes the remaining mass; this correction is disclosed, not learned biology.

The portable HTML contains recorded predictions for twelve paired presets and the default assay ranking. Arbitrary valid settings and live Meta tools use the local server. Model inference, training and validation use NumPy only; no GPU, PyTorch or quantum hardware is required.

## Frozen comparison

The initial training experiment generated 512 training cases, 96 validation cases, 96 calibration cases, 128 test cases and 96 OOD cases. Each case contains both branches and all sixteen sampled times; whole cases stay in a single split. Physical parameters are sampled continuously, and HA size is explicitly excluded from predictive features. The batched data generator agrees with the authoritative tissue solver to approximately 5.6e-16 in its parity checks.

All nine trained candidates received 1,800 Adam updates with batch size 256 and the same validation schedule. Three seeds are used for each main family. Model sizes and CPU times are recorded; equal updates do not mean equal FLOPs. A polynomial ridge model provides a simpler baseline. The MPS uses seven angle feature maps, rank-nine internal bonds and a normalized squared-amplitude head. Rank-four, reversed feature ordering and a softmax-head ablation are included. Analytic gradients, serialization, Born normalization and arbitrary global-phase invariance are tested.

The initial test revealed small reversals in cumulative absorbed/cleared mass. Those results are retained in `data/learning/stage1` as retired diagnostics. A specified conservative correction was added without retraining weights; family choice used the original validation set. Fresh calibration, test and OOD draws were generated using a new frozen seed. The new choice and calibration were saved before scoring the new test. The original exposed test is not presented as an untouched validation set.

| Model | Fresh test mean absolute error | Final plaque absolute error |
|---|---:|---:|
| Neural ensemble | 0.00560 | 0.00609 |
| MPS / Born ensemble | 0.02265 | 0.01530 |
| Polynomial ridge | 0.02235 | 0.00794 |

Outputs are normalized fractions/indices. The neural mean error is 0.56 percentage points, not 99.44% biological accuracy. Its 95th-percentile case-wise maximum error is 0.08174; its mean blood-AUC absolute error is 3.49 fraction-seconds using the same sixteen-point integration. Maximum test constraint correction was 0.01362. All corrected test trajectories conserve mass and have monotone cumulative fates. Raw neural trajectories violated cumulative monotonicity in 87.5% of fresh cases, which is why the explicit constraint is necessary.

A separate calibration set defines one 90% split-conformal simultaneous envelope over both branches, all eight outputs and the sixteen sampled times. Fresh in-domain case coverage was 93.75%. OOD coverage was only 47.92%; the runtime therefore abstains outside its declared support box and supplies full-solver comparisons without calibrated bands. These bands concern exchangeable synthetic samples, not empirical biology. Ensemble standard deviation is separately labeled model disagreement. There is no quantum advantage in this benchmark.

Data, split IDs, parameters, frozen protocol, teacher hash, model checkpoint and reports are included in `data/learning`. The checkpoint SHA-256 is checked by tests. Initial training results and the later constraint revision are both retained. Reproduce in a copied project directory:

```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/train_crossing.py
OPENBLAS_NUM_THREADS=1 python3 scripts/validate_crossing_constraints.py
python3 scripts/assemble_journey.py
python3 scripts/build_journey_snapshot.py
```

The first command recreates stage-one artifacts; the second uses the retained stage-one fitted checkpoint for the released constraint benchmark. For a fully fresh reproducibility run, remove only `data/learning/stage1` in that copied directory before the second command so it snapshots the newly trained weights. Small platform-dependent floating-point changes are possible. Training requires NumPy >=1.26, as does inference.

## Choosing measurements

The planner compares three explicitly assumed, unfitted hypotheses: current barrier support, faster clearance without support, and slower transport without support. Nine candidate readouts combine blood fraction, plaque fraction or the integrity index at 320, 640 or 1200 assumed seconds. It calculates Monte Carlo mutual information under equal priors and Gaussian measurement noise with SD 0.03, divided by illustrative relative effort. Every ranking is checked with the full simulator. Model integrity must be mapped to a real assay before use. Noise, costs, intervention effects and human timing have not been calibrated. Small score differences may be Monte Carlo noise; the full receipt includes its estimated standard error.

Real validation needs paired permeability and tracer time courses, independent clearance/uptake measurements, membrane versus cargo labeling, specimen/batch provenance, and predeclared controls. Appearance alone does not identify a host matrix vesicle or intact bacterial OMV. The measurement schema in this folder describes collection fields; it is not an import-and-train claim.

## Schooling as the scalable coordination layer

Press **Schooling fish** or **Scales → Enter the schooling sandbox**. This is a running 3D local-rule demonstration with 64, 256, 1024 or 4096 agents. Orbit by dragging and zoom by scrolling. Separation, alignment and cohesion operate on local observations. Disrupt half the headings to observe reorganization; switch to aggregate groups to inspect collective state.

The controls instantiate the state contract:

| Term | Implemented meaning |
|---|---|
| Information | Local positions and headings within a selected sensing radius; zero radius removes neighbor-driven coupling. |
| Energy | A finite normalized movement budget per agent, consumed each step; zero energy yields zero motion. This is illustrative bookkeeping, not joules or metabolic calibration. |
| Terrain | Periodic unit cube, optionally containing a band with 25% mobility. |
| Time | Explicit time step, bounded by 0.05 simulation units, and a 30/60/120-unit horizon. |
| State | Positions, headings, remaining energy, consumption, time, polarization, rules and provenance; exportable as JSON. |

Spatial bins restrict candidate neighborhoods. At most 96 candidates are examined per agent per step; high-density sampling is approximate, with rotating traversal order. Update work is bounded by agent count times this cap, plus fixed-grid bookkeeping. Aggregation preserves agent counts and reduces drawing work, but every individual still updates. This is a single-machine CPU implementation, not a distributed compute deployment. Rendering costs and user-device performance are not inferred from the Node benchmark. `schooling-validation.json` records measured update times and guards.

The scale explorer includes quantum, molecule, cell, tissue, organ, organism and ecosystem. It marks what is running and what remains conceptual. The MPS is quantum-inspired math; molecule/cell layers supply evidence and aggregate hypotheses; tissue-to-blood and blood-to-plaque fluxes are implemented; organism-wide physiology is unresolved; schooling is an illustrative collective-behavior sandbox rather than a food web. No biological dynamics are borrowed from fish without an explicit validated adapter.

The existing `sim/epoch4` information-primary tissue engine remains available. It is identified as another adapter but is not numerically coupled to the Crossing or schooling solvers. The Crossing transport model has no explicit metabolic energy budget; its missing energy coupling is declared.

## Validation and limits

Run:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tests -p 'test_journey*.py' -v
node tests/check_journey_ui.cjs
node tests/check_journey_live_ui.cjs
node tests/check_schooling.cjs
```

The Python suite checks transport, learning, HTTP and agent tool contracts. Node checks cover portable and live UI control flows, including new learning and schooling controls; the model provider is stubbed. Schooling tests cover deterministic replay, information ablation, zero-energy immobility, energy accounting, terrain response, finite state, count aggregation and bounded neighbor work. The v3 changes have not yet been run against the user's live Meta session or visually inspected in their browser. The previously verified v2 session does not constitute v3 live validation.

## Primary method references

- Reynolds (1987), *Flocks, Herds, and Schools: A Distributed Behavioral Model*: https://www.red3d.com/cwr/papers/1987/boids.html
- Lakshminarayanan et al. (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*: https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html
- Stoudenmire and Schwab (2016), *Supervised Learning with Tensor Networks*: https://proceedings.neurips.cc/paper_files/paper/2016/hash/5314b9674c86e3f9d1ba25ef9bb32895-Abstract.html
- Han et al. (2018), *Unsupervised Generative Modeling Using Matrix Product States*: https://doi.org/10.1103/PhysRevX.8.031012

These motivate methods, not biological validity. The conditional MPS regression and conservative correction here are project-specific implementations.
