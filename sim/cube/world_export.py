"""Serialize a live body-map run for the viewer (increment 5).

Runs the coupled oral→artery body-map, capturing real per-frame point clouds (sampled from each cube's
composition), ψ trajectories, the bloodstream efflux over time, and the grounded ask-anything answers.
Everything the viewer shows is REAL engine output — no JS-side fabrication.
"""

from __future__ import annotations

import numpy as np

from sim.cube import query
from sim.cube.bodymap import BodyMap
from sim.cube.groups import GROUP_COLOR, species_group
from sim.cube.tissues import ARTERY_TISSUE, ORAL_TISSUE
from sim.species import BY_ID

# Medical-grade cell colours for the artery roster (oral cells use the functional-group palette).
_ARTERY_COLORS = {"EC": "#5b9bd5", "Mono": "#e6a83c", "Foam": "#d6543f", "SMC": "#9a86c4"}
_CELL_LABEL = {"EC": "endothelium", "Mono": "monocyte/macrophage", "Foam": "foam cell",
               "SMC": "smooth muscle"}


def _cell_color(cid: str) -> str:
    if cid in _ARTERY_COLORS:
        return _ARTERY_COLORS[cid]
    return GROUP_COLOR[species_group(BY_ID[cid])]


def _sample_points(eng, n: int, rng) -> list[list]:
    """Biomass-weighted point cloud: [x, y, z, cell_index] with coords normalised to [0,1]."""
    comp = eng.grid.composition
    sh = eng.grid.shape
    tot = comp.sum(axis=3).ravel()
    s = tot.sum()
    if s <= 0:
        return []
    sel = rng.choice(tot.size, size=n, p=tot / s)
    pts = []
    for vi in sel:
        i, j, k = np.unravel_index(vi, sh)
        cc = comp[i, j, k]
        cs = cc.sum()
        if cs <= 0:
            continue
        c = int(rng.choice(len(eng.species), p=cc / cs))
        pts.append([round(float(i / (sh[0] - 1)), 3), round(float(j / (sh[1] - 1)), 3),
                    round(float(k / (sh[2] - 1)), 3), c])
    return pts


def build_world(dim: int = 26, steps: int = 160, frames: int = 6, n_points: int = 1100,
                k: float = 2.5, seed: int = 0) -> dict:
    """Run oral→artery coupled, capture `frames` snapshots, and return a serialisable world."""
    bm = BodyMap()
    bm.place("oral", ORAL_TISSUE, dim=dim)
    bm.place("artery", ARTERY_TISSUE, dim=dim)
    bm.link("oral", "artery", k=k)
    rng = np.random.default_rng(seed)
    every = max(1, steps // frames)
    caps = {name: [] for name in bm.cubes}
    psis = {name: [] for name in bm.cubes}
    efflux: list[float] = []
    times: list[float] = []
    for t in range(1, steps + 1):
        bm.step()
        if t % every == 0 or t == steps:
            for name, eng in bm.cubes.items():
                caps[name].append(_sample_points(eng, n_points, rng))
                psis[name].append({kk: v for kk, v in eng.readout().items() if kk != "fields"})
            efflux.append(round(bm.cubes["oral"].efflux(), 3))
            times.append(round(t * 0.1, 1))

    cubes = []
    for name, eng in bm.cubes.items():
        legend = [{"id": cid, "label": _CELL_LABEL.get(cid, cid), "color": _cell_color(cid)}
                  for cid in eng.species]
        cubes.append({"name": name, "tissue": eng.tissue.name, "cells": list(eng.species),
                      "legend": legend, "frames": caps[name], "psi": psis[name]})

    body = query.digest_bodymap(bm)
    qa = {name: [{"q": q, "a": query.answer(q, body["cubes"][name], body=body)}
                 for q in query.DEFAULT_QUESTIONS] for name in bm.cubes}
    return {"cubes": cubes, "efflux": efflux, "times": times, "qa": qa, "n_frames": len(efflux)}

