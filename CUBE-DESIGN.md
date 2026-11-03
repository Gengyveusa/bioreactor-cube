# The Cube — 3-D gingival-pocket biofilm engine (design + roadmap)

A representative-volume-element (RVE) voxel model of a cube of biofilm in a gingival pocket: an
analysis / simulation / treatment-recommendation generator. Each voxel ("pixel") carries a full
state and an address relatable to every other voxel; the six faces are the boundary conditions, and
**the surfaces are where the action happens.** The cube is the 3-D spatial realization of the lower
layers of the 8-layer cake (Layer 1 topology/nodes, Layer 2 terrain, Layer 3 chemistry) and rolls
up through Layers 6–7 into the systemic synthesis.

## Design decisions (locked 2026-06-21)
| # | Decision | Choice |
|---|---|---|
| 1 | Engine | **New dedicated 3-D voxel engine**, built on the existing `SpatialGrid` ABC seam |
| 2 | First step (v1) | **The surfaces / boundary module** (the six faces) — DONE |
| 3 | Time | **Both** snapshot + dynamic |
| 4 | Resolution | **Fine (~50³+)** — 125k voxels builds instantly |
| 5 | 6th face | **Periodic / unit-cell** (RVE) |
| 6 | Per-voxel state | **All** — composition + chemistry + network role + structure/mineral |
| 7 | Addressing | **Coordinate + full graph** — `index ↔ coord`, per-face distance |
| 8 | Inputs → cube | **Infer spatial distribution** from composition + ecology priors |
| 9 | AI | **Maximal from the start** — architect for GNN / DeepMind-grade models (`sim/analysis/gnn.py` exists) |
| 10 | Output | **Per-voxel 3-D state map** (addressable, visualizable); feeds the existing rollup + report |

## Geometry — the six faces (the surfaces)
A cube `nx × ny × nz`. x = enamel ↔ soft tissue; y = adjacent biofilm (periodic); z = deep ↔ oral.

| Face | Location | Kind | Boundary condition |
|---|---|---|---|
| Enamel + pellicle | x = 0 | `ENAMEL_PELLICLE` | solid anoxic wall; Ca/PO₄ supersaturation → **biomineralization**; pellicle adhesion API |
| Soft tissue | x = nx-1 | `SOFT_TISSUE` | host: serum pH 7.4, low O₂ (tissue sink), heme on bleeding; the immune / inflammatory inflow-outflow |
| Adjacent biofilm | y = 0, ny-1 | `ADJACENT_BIOFILM` | continuation into surrounding biofilm (**periodic** y) |
| Deep unit-cell | z = 0 | `PERIODIC` | unit-cell closure |
| Oral cavity | z = nz-1 | `ORAL_CAVITY` | open: saliva flow, pH, dietary nitrate, sugar, O₂; metabolite washout |

Shared edges: where two faces meet, the later-applied face wins (oral washout dominates the top edge).
`face_distance` / `face_influence` give the per-surface gradient — each surface influences nearby
voxels most, and the biofilm affects each surface differently.

## Per-voxel state (the "state calculation per pixel")
- **chemistry** — O₂, pH, NO, nitrate/nitrite, sugar, lactate, heme, H₂S, NH₃, SCFAs, **Ca, PO₄** (Layer 3)
- **composition** — biomass per species (Layer 1 nodes: frontier / connector / keystone / commensal)
- **structure** — EPS matrix density, mineralized fraction (biomineralization)
- **network role** — derived, not stored: composition × the coaggregation graph (the connectome, *in context*)
- **address** — `(i,j,k)` + linear `index`; distance to each of the six surfaces

## What v1 ships (`sim/cube/`, tested — `tests/test_cube.py`, 8 passing)
- `grid.py` — `CubeGrid` implements the `SpatialGrid` ABC (so the existing agents/chemistry/host plug
  in); fields + composition arrays; addressing (`index`/`coord`); 6-neighbor stencil (x,z bounded,
  y periodic); the O₂ gradient (high at the oral face, decaying into depth).
- `faces.py` — the **surfaces module**: `FaceKind`, the six `FACES`, `apply_face_boundaries` (Dirichlet
  flux per face), `face_distance` / `face_influence`.
- `state.py` — `read_voxel` → `VoxelState` (the addressable per-voxel state map unit).

## Reuse map (a lot was already built)
- `sim/grid/base.py` `SpatialGrid` ABC — the 1D-now/3D-later seam (built for exactly this).
- `sim/grid/grid_1d.py` `CHEMISTRY_FIELDS` — reused + Ca/PO₄ added.
- `sim/species.py` `CATALOG` — the species/nodes.
- `sim/host.py` — the soft-tissue (host immunity) couplings → the SOFT_TISSUE face.
- `sim/chemistry/{diffusion,boundaries,nitrogen}.py` — diffusion + the 1-D boundary patterns to lift to 3-D.
- `sim/analysis/gnn.py` — the GNN starting point for decision 9.
- connectome-machine: the coaggregation graph + network roles (Layer 1), and the clinical rollup
  (systemic synthesis + patient report) the cube feeds.

## Roadmap (after v1)
1. **Transport** — lift `chemistry/diffusion.py` to 3-D (anisotropic; surface-refined), with the
   face BCs as sources/sinks. Reaction–diffusion per voxel.
2. **Inputs → spatial seeding** — place taxa in 3-D from a pooled sample using ecology priors
   (anaerobes/red-complex deep + near enamel; aerobes/reducers shallow near oral O₂; bridges between).
3. **Biology** — agent growth/division, coaggregation, OMV vesiculation, biomineralization, per voxel.
4. **Snapshot + dynamic** solvers (decision 3): equilibrium solve and time integration → trajectory.
5. **GNN** (decision 9) — learn dynamics / a surrogate over the voxel + coaggregation graph.
6. **Rollup** — aggregate the cube up the cake (biofilm → oral env → systemic) into the existing
   `synthesize_systemic` + patient report; surface flux per face = what crosses to each compartment.

**Status:** mechanistic-framework foundation. Not a validated instrument — geometry/priors/effect
sizes are placeholders pending the paired-cohort calibration (see the venture data plan).

