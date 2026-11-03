"""Estimated-sample pipeline — seed the cube spatially from ecology, solve the chemistry to a
snapshot, and export the real engine state for the viewer. No hand-waving: placement comes from
each species' viability envelope + preferred depth, and the gradients emerge from the seeded biomass.

    grid = CubeGrid(40, 40, 40)
    seed_composition(grid)        # decision 8 — infer spatial distribution from ecology priors
    solve_snapshot(grid)          # decision 3 — snapshot chemistry from boundaries + reactions
    data = export_sample(grid)    # the per-voxel 3-D state map the viewer renders
"""

from __future__ import annotations

import numpy as np

from sim.cube.faces import FaceConditions, apply_face_boundaries
from sim.cube.grid import CubeGrid
from sim.cube.groups import GROUP_COLOR, GROUP_LABEL, SPECIES_GROUP
from sim.species import BY_ID, CATALOG, COAGG_EDGES

# A plausible moderately-dysbiotic pocket (relative input abundances; the spatial structure is the model).
DEFAULT_ABUNDANCE: dict[str, float] = {
    "Sa": 0.12, "Sm": 0.10, "An": 0.06, "Hp": 0.04, "Rm": 0.07, "Vp": 0.05,
    "Fn": 0.12, "Pi": 0.04, "Ec": 0.03, "Pg": 0.07, "Td": 0.05, "Tf": 0.05,
    "Aa": 0.02, "Smu": 0.03, "Lb": 0.02, "Fa": 0.02, "Pm": 0.02, "Ca": 0.02,
}


def _axis_norm(n: int) -> np.ndarray:
    return np.arange(n) / (n - 1) if n > 1 else np.zeros(n)


def _depth_field(shape: tuple[int, int, int]) -> np.ndarray:
    """0.0 at the oral face (k = nz-1) → 1.0 at the deep face (k = 0)."""
    nz = shape[2]
    df = 1.0 - _axis_norm(nz)
    return np.broadcast_to(df[np.newaxis, np.newaxis, :], shape)


def seed_composition(grid: CubeGrid, abundances: dict[str, float] | None = None, seed: int = 0) -> None:
    """Place each species' biomass where it can actually live: O2 window × preferred-depth match ×
    heme/anaerobe terms × the enamel↔soft-tissue position. (Decision 8 — inferred spatial distribution.)"""
    rng = np.random.default_rng(seed)
    shape = grid.shape
    nx = shape[0]
    depth = _depth_field(shape)
    x_norm = np.broadcast_to(_axis_norm(nx)[:, np.newaxis, np.newaxis], shape)   # 0 enamel, 1 soft tissue
    o2 = grid.get_field("O2")
    ab = abundances or DEFAULT_ABUNDANCE

    for s in CATALOG:
        a = ab.get(s.id, 0.0)
        if a <= 0:
            continue
        o2_opt = s.envelope.o2[2]
        o2_w = max(1.0, (s.envelope.o2[1] - s.envelope.o2[0]) / 2.0)
        f_o2 = np.exp(-((o2 - o2_opt) / o2_w) ** 2)
        center = 0.5 * (s.preferred_depth[0] + s.preferred_depth[1])
        f_depth = np.exp(-((depth - center) / 0.18) ** 2)
        fit = f_o2 * f_depth
        if s.envelope.heme >= 1:                       # heme-needers hug the bleeding gum + the deep zone
            fit = fit * (0.4 + 0.6 * x_norm) * (0.4 + 0.6 * depth)
        if s.o2_class == "anaerobe":
            fit = fit * np.clip(1.2 - o2, 0.0, 1.2)
        noise = np.clip(1.0 + 0.15 * rng.standard_normal(shape), 0.3, 1.7)
        grid.composition[..., grid.species_index(s.id)] = np.clip(a * fit * noise, 0.0, None)


def _diffuse(field: np.ndarray) -> np.ndarray:
    """6-neighbor average (one Jacobi diffusion step)."""
    return (np.roll(field, 1, 0) + np.roll(field, -1, 0)
            + np.roll(field, 1, 1) + np.roll(field, -1, 1)
            + np.roll(field, 1, 2) + np.roll(field, -1, 2)) / 6.0


# A bleeding, sugar-exposed pocket — so the heme and pH gradients are real and visible.
BLEEDING_POCKET = FaceConditions(serum_heme=1.0, oral_sugar=0.3)


def solve_snapshot(grid: CubeGrid, iters: int = 45, cond: FaceConditions = BLEEDING_POCKET) -> None:
    """Relax the chemistry to a snapshot: diffusion under the six-face Dirichlet boundaries, with
    reactions sourced by the seeded biomass — O2 consumed by aerobes, nitrate→nitrite→NO by reducers,
    pH shifted by acid producers, heme entering at the bleeding gum. Gradients are a consequence of
    the biology + boundaries, not painted on."""
    apply_face_boundaries(grid, cond)
    comp = grid.composition
    idx = grid.species_index

    aerobe = sum(comp[..., idx(s.id)] for s in CATALOG if s.o2_class in ("facultative", "aerotolerant"))
    reducer_no = sum(comp[..., idx(s.id)] * max(0.0, s.NO_yield) for s in CATALOG if s.nitrate_reducer)
    acid = sum(comp[..., idx(s.id)] * s.pH_effect for s in CATALOG)              # negative lowers pH

    f = grid._fields
    for _ in range(iters):
        for name in ("O2", "pH", "nitrate", "nitrite", "NO", "heme"):
            f[name][:] = _diffuse(f[name])
        f["O2"] -= 0.05 * aerobe * f["O2"]
        np.clip(f["O2"], 0.0, None, out=f["O2"])
        reduced = 0.05 * reducer_no * f["nitrate"]                              # nitrate -> nitrite
        f["nitrate"] -= reduced
        np.clip(f["nitrate"], 0.0, None, out=f["nitrate"])
        f["nitrite"] += reduced
        f["NO"] += 0.6 * reduced                                                # nitrite -> NO by reducers
        f["NO"] *= 0.96                                                         # NO decay
        f["pH"] += 0.05 * acid
        apply_face_boundaries(grid, cond)                                      # reassert the surfaces
    np.clip(f["pH"], 4.0, 8.5, out=f["pH"])


def _downsample_field(field: np.ndarray, n: int) -> dict:
    nx, ny, nz = field.shape
    sub = field[np.ix_(np.linspace(0, nx - 1, n).astype(int),
                       np.linspace(0, ny - 1, n).astype(int),
                       np.linspace(0, nz - 1, n).astype(int))]
    lo, hi = float(sub.min()), float(sub.max())
    span = hi - lo if hi > lo else 1.0
    q = np.clip((sub - lo) / span * 255.0, 0, 255).astype(int)
    return {"min": round(lo, 3), "max": round(hi, 3), "q": q.flatten().tolist()}


def export_sample(grid: CubeGrid, field_n: int = 16, n_points: int = 2000,
                  max_edges: int = 260, seed: int = 0) -> dict:
    """Serialize the engine state for the viewer: downsampled+quantized chemistry fields, biomass-
    weighted representative points (with group + species), real coaggregation edges, OMV emitters."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = grid.shape
    fields = {k: _downsample_field(grid.get_field(k), field_n) for k in ("O2", "pH", "NO", "heme")}

    comp = grid.composition
    # per-group biomass fields + the group's species, so each group is placed by its own ecology
    grp_field: dict[str, np.ndarray] = {}
    grp_species: dict[str, list[str]] = {}
    for s in CATALOG:
        g = SPECIES_GROUP.get(s.id, "other")
        arr = comp[..., grid.species_index(s.id)]
        grp_field[g] = grp_field.get(g, np.zeros(grid.shape)) + arr
        grp_species.setdefault(g, []).append(s.id)
    present = {g: float(a.sum()) for g, a in grp_field.items() if a.sum() > 0}

    # allocate points ~ sqrt(total biomass) (lifts rare groups) with a floor, so all groups show
    wsum = sum(v ** 0.5 for v in present.values())
    alloc = {g: max(60, int(n_points * (present[g] ** 0.5) / wsum)) for g in present}

    groups: list[str] = list(present.keys())
    gidx = {g: n for n, g in enumerate(groups)}
    points: list[list] = []
    by_species: dict[str, list[int]] = {}
    for g in groups:
        arr = grp_field[g].flatten()
        k = min(alloc[g], int((arr > 0).sum()))
        if k <= 0:
            continue
        chosen = rng.choice(arr.size, size=k, replace=False, p=arr / arr.sum())
        for fi in chosen:
            i, j, kk = (int(x) for x in np.unravel_index(fi, grid.shape))
            # the dominant species *within this group* at the voxel
            sid = max(grp_species[g], key=lambda sp: comp[i, j, kk, grid.species_index(sp)])
            points.append([round(i / (nx - 1), 3), round(j / (ny - 1), 3), round(kk / (nz - 1), 3),
                           gidx[g], sid])
            by_species.setdefault(sid, []).append(len(points) - 1)

    # real coaggregation edges among nearby sampled points (the connectome, in space)
    edges: list[list[int]] = []
    pa = np.array([[p[0], p[1], p[2]] for p in points])
    for a, b in COAGG_EDGES:
        ia, ib = by_species.get(a, []), by_species.get(b, [])
        if not ia or not ib:
            continue
        for u in ia[:6]:
            d = np.sum((pa[ib] - pa[u]) ** 2, axis=1)
            v = ib[int(d.argmin())]
            if d.min() < 0.05:                          # spatially adjacent only
                edges.append([u, v])
            if len(edges) >= max_edges:
                break
        if len(edges) >= max_edges:
            break

    omv = [n for n, p in enumerate(points) if groups[p[3]] == "red" and p[0] > 0.6]   # red near the gum
    return {
        "dims": [nx, ny, nz],
        "field_n": field_n,
        "fields": fields,
        "groups": groups,
        "group_color": {g: GROUP_COLOR[g] for g in groups},
        "group_label": {g: GROUP_LABEL[g] for g in groups},
        "points": points,
        "edges": edges,
        "omv": omv,
    }


def build_sample(dim: int = 40, abundances: dict[str, float] | None = None,
                 iters: int = 45, **export_kw) -> dict:
    """One call: build → seed → solve → export the cube as a JSON-able state map."""
    grid = CubeGrid(dim, dim, dim)
    seed_composition(grid, abundances)
    solve_snapshot(grid, iters)
    return export_sample(grid, **export_kw)

