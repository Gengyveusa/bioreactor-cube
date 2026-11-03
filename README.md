## The Crossing: mouth to circulation

Open `/journey` on the local server, or `demo/cube-journey-recorded.html` for the portable preview. Walk or fly, compare hypothetical tissue responses, inspect two archived reference figures, and use the live Meta guide. See [The Crossing](docs/THE_CROSSING.md) for the model, demo script and verification limits.

# oral_biofilm_sim

**Hackathon workspace:** see [Cube Lab setup and scope](docs/HACKATHON_BUILD.md)
for the interactive diffusion experiment and bounded Meta research agent.

**The mechanistic core (Layer 1) of the Oral Microbiome Node Schema.**

Turns the 8-layer schema substrate from a static ontology into a running dynamical system —
the engine that the GNN bridge-asymmetry test (Layer 4), the SCM intervention engine (Layer 5),
and the criticality/early-warning detectors (Layer 6) sit on top of.

## Architectural contract: 1D now, 4D-ready

Spatial dimensionality is abstracted behind `sim.grid.base.SpatialGrid`. Agents and the chemistry
solver address the grid by `position` tuples — `(depth,)` for v1, `(x, y, depth)` for v2/v3.
Going from a depth column to a 3D voxel field is two class swaps (`SpatialGrid1D` →
`SpatialGrid3D`, `Diffusion1D` → `Diffusion3D`). Agent logic, chemistry, observables, model
orchestration: all unchanged.

Time is the ABM step counter — that's the 4th axis (xyz, time) requirement, satisfied for free.

## What the simulator solves

For each tick `t`:

1. **Chemistry update** (reaction-diffusion) — O₂, pH (via H⁺ and buffers), fermentable sugar,
   heme, lactate, NO, H₂S, NH₃ evolve under ∂c/∂t = D∇²c + sources/sinks.
2. **Boundary conditions** — saliva at the top (O₂ ≈ 8%, pH ≈ 6.8, sugar pulses per Stephan
   curve), gingival sulcus at the bottom (heme flux from bleed loop, IgG/IgA, serum proteins).
3. **Agent step** — each bacterium reads local chemistry, checks viability against its 4-field
   Layer-3 constraints, computes growth from chemistry + crossfeed neighbors, divides or dies,
   emits byproducts and (stochastically) OMVs.
4. **Observables update** — composition vector per depth, OMV cargo flux at top boundary,
   coherence indices (NO/H₂S/pH variance), dynamic topology (betweenness over the realized
   coaggregation graph at this tick).

## Falsification targets for v1 (week 2)

The simulator must qualitatively reproduce:

1. **Succession** — pioneers (Sa, Sm, An, Hp, Rm) establish first; anaerobes (Vp, Fn, Pg, Td, Tf)
   colonize as O₂ depletes with depth. Order matches Kolenbrander.
2. **Dynamic bridge emergence** — *F. nucleatum* becomes the highest-betweenness node from its
   aerotolerance + universal coaggregation properties, not as a hard-coded fact.
3. **Stephan curve** — sugar pulse → *S. mutans* bloom + local pH crash → recovery as buffering
   restores pH 6.8.
4. **OVN ignition** — heme flux at the bottom boundary (simulated bleed) → *P. gingivalis*
   bloom + OMV-flux spike at the top boundary (the systemic input).

If the simulator can't reproduce any of these, parameters are wrong, not the schema.

## Five Epoch 4 operators — where they live in the code

| Operator | Where |
|---|---|
| **ψ** (state) | `observables.composition` — community state vector per depth per tick |
| **T** (terrain) | `chemistry.fields` — O₂/pH/sugar/heme + downstream metabolites |
| **Δ** (change over time) | the ABM step itself + `observables.dynamics` rolling-window EWS |
| **E** (energy/predation) | phage layer (v2 — Layer 4 of schema) |
| **I** (information/immunity) | host immune boundary (v2 — Layer 6 of schema) |

## Install

```bash
cd oral_biofilm_sim
pip install -e .
```

## Smoke test

```bash
python -m sim.cli run --config configs/default.yaml --ticks 1000
```

Outputs trajectory HDF5 + a quick PNG of composition-vs-depth over time to `output/`.

