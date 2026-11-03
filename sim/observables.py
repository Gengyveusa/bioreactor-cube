"""Observables — the four readouts the model exposes per tick.

1. composition_by_depth : (n_species, n_depths) biomass array
2. omv_flux             : aggregate vesicle production per cargo class
3. coherence            : NO production, H2S production, pH variance, redox proxy
4. dynamic_topology     : betweenness of realized coaggregation graph this tick

These observables are what the GNN (Layer 4), SCM (Layer 5), and EWS detectors
(Layer 6) consume. Keep them numerically clean — no unit drama.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import networkx as nx

from .species import CATALOG, COAGG_EDGES, BY_ID

if TYPE_CHECKING:
    from .model import OralBiofilmModel


SPECIES_INDEX = {s.id: i for i, s in enumerate(CATALOG)}
N_SPECIES = len(CATALOG)


@dataclass
class TickObservation:
    tick: int
    composition: np.ndarray         # (N_SPECIES, N_DEPTHS)
    omv_flux: dict[str, float]      # cargo -> total quantity this tick
    NO_production: float
    H2S_production: float
    pH_mean: float
    pH_variance: float
    dysbiosis_index: float          # (red+orange biomass) / (total biomass)
    bridge_betweenness_Fn: float
    keystone_betweenness_Pg: float


@dataclass
class Trajectory:
    ticks: list[TickObservation] = field(default_factory=list)

    def to_arrays(self) -> dict:
        T = len(self.ticks)
        if T == 0:
            return {}
        out = {
            "tick": np.array([o.tick for o in self.ticks]),
            "composition": np.stack([o.composition for o in self.ticks], axis=0),
            "NO": np.array([o.NO_production for o in self.ticks]),
            "H2S": np.array([o.H2S_production for o in self.ticks]),
            "pH_mean": np.array([o.pH_mean for o in self.ticks]),
            "pH_variance": np.array([o.pH_variance for o in self.ticks]),
            "dysbiosis": np.array([o.dysbiosis_index for o in self.ticks]),
            "Fn_betweenness": np.array([o.bridge_betweenness_Fn for o in self.ticks]),
            "Pg_betweenness": np.array([o.keystone_betweenness_Pg for o in self.ticks]),
        }
        return out


def observe(model: "OralBiofilmModel") -> TickObservation:
    g = model.grid_obj
    N_depths = g.shape[0]
    comp = np.zeros((N_SPECIES, N_depths), dtype=np.float64)
    for a in model.agents_alive:
        sidx = SPECIES_INDEX[a.species.id]
        comp[sidx, a.position[0]] += a.biomass

    # OMV flux this tick
    omv_flux: dict[str, float] = {}
    for tk, em in model.omv_log:
        if tk != model.tick:
            continue
        for cargo in em.cargo:
            omv_flux[cargo] = omv_flux.get(cargo, 0.0) + em.quantity

    NO_field = g.get_field("NO")
    H2S_field = g.get_field("H2S")
    pH_field = g.get_field("pH")

    # Dysbiosis index: red + orange Socransky biomass over total
    red_orange_ids = {s.id for s in CATALOG if s.socransky in ("red", "orange")}
    total = comp.sum()
    red_orange = sum(comp[SPECIES_INDEX[sid]].sum() for sid in red_orange_ids)
    dys = float(red_orange / max(total, 1e-9))

    # Dynamic topology: realized coaggregation graph this tick
    # An edge from COAGG_EDGES is "realized" if both species have biomass > 0
    # somewhere AND they have biomass in adjacent depths (within 2 cells).
    present_depths: dict[str, set[int]] = {sid: set() for sid in BY_ID}
    for a in model.agents_alive:
        present_depths[a.species.id].add(a.position[0])

    G = nx.Graph()
    G.add_nodes_from(BY_ID.keys())
    for u, v in COAGG_EDGES:
        du, dv = present_depths.get(u, set()), present_depths.get(v, set())
        if not du or not dv:
            continue
        # any pair within 2 cells?
        if any(abs(x - y) <= 2 for x in du for y in dv):
            G.add_edge(u, v)
    bet = nx.betweenness_centrality(G) if G.number_of_edges() else {}
    fn_bet = float(bet.get("Fn", 0.0))
    pg_bet = float(bet.get("Pg", 0.0))

    return TickObservation(
        tick=model.tick,
        composition=comp,
        omv_flux=omv_flux,
        NO_production=float(NO_field.sum()),
        H2S_production=float(H2S_field.sum()),
        pH_mean=float(pH_field.mean()),
        pH_variance=float(pH_field.var()),
        dysbiosis_index=dys,
        bridge_betweenness_Fn=fn_bet,
        keystone_betweenness_Pg=pg_bet,
    )

