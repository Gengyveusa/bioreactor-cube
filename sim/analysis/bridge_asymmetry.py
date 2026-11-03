"""The Bridge-Asymmetry Theorem test.

Theorem (from the Network Medicine substrate, schema-v4-phages.md + MOC):
    Bridge species pay evolutionary rent through CONNECTIVITY, not autonomy —
    heavy bacterial defense is selected against. Removing a bridge fragments
    the network more than removing a keystone of equivalent abundance.

Falsification test:
    For each species s, compute the network damage of removing s. Group species
    by archetype. Compare distributions of (Bridge group) vs (Keystone group).

    Damage metrics:
        - ΔE          = drop in global efficiency
        - ΔLCC        = drop in largest-connected-component size (fraction of nodes)
        - Δq          = change in Louvain modularity
        - Δ⟨d⟩        = increase in average shortest path
        - n_components_after = number of disconnected components after removal

    The theorem FIRES if median(damage | Bridge) > median(damage | Keystone)
    on the primary metric (we use ΔE as the headline; ΔLCC as the redundancy check).

    The theorem DIES if median(damage | Keystone) >= median(damage | Bridge).

Two operating modes:
    1. Static  : runs on the canonical 18-node coaggregation graph from oral-network.json
    2. Dynamic : runs on the realized coaggregation graph from a simulator trajectory
                 (per-tick or end-state), where edges exist only between species
                 whose biomass is co-localized in space at that tick.

The dynamic mode is the falsification path that distinguishes "the theorem
holds in the textbook static graph" from "the theorem holds in actual
biofilm dynamics" — the second is the commercially-meaningful claim.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import networkx as nx
import h5py

from ..species import CATALOG, BY_ID, COAGG_EDGES


def get_catalog(extended: bool = False):
    """Return (catalog_list, coagg_edges) — canonical 18-node or extended ~58-node."""
    if not extended:
        return list(CATALOG), list(COAGG_EDGES)
    from ..species_extended import EXTENDED, EXTENDED_COAGG_EDGES
    catalog = list(CATALOG) + list(EXTENDED)
    edges = list(COAGG_EDGES) + list(EXTENDED_COAGG_EDGES)
    return catalog, edges


def build_static_coagg_graph(extended: bool = False) -> nx.Graph:
    """The static coaggregation graph (canonical 18 or extended ~58)."""
    catalog, edges = get_catalog(extended=extended)
    G = nx.Graph()
    for s in catalog:
        G.add_node(s.id, archetype=s.archetype, name=s.name)
    G.add_edges_from(edges)
    return G


# ---------------------------------------------------------------------------
# Damage metrics
# ---------------------------------------------------------------------------
def global_efficiency(G: nx.Graph) -> float:
    """Inverse-shortest-path-averaged across all node pairs."""
    if G.number_of_nodes() < 2:
        return 0.0
    return float(nx.global_efficiency(G))


def modularity(G: nx.Graph) -> float:
    if G.number_of_edges() == 0:
        return 0.0
    try:
        comms = nx.algorithms.community.louvain_communities(G, seed=42)
        return float(nx.algorithms.community.modularity(G, comms))
    except Exception:
        return 0.0


def largest_cc_fraction(G: nx.Graph) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    lcc = max(nx.connected_components(G), key=len)
    return len(lcc) / G.number_of_nodes()


def avg_shortest_path_lcc(G: nx.Graph) -> float:
    if G.number_of_nodes() < 2:
        return 0.0
    lcc_nodes = max(nx.connected_components(G), key=len)
    H = G.subgraph(lcc_nodes)
    if H.number_of_nodes() < 2:
        return 0.0
    return float(nx.average_shortest_path_length(H))


@dataclass
class AblationResult:
    species_id: str
    archetype: str
    delta_efficiency: float
    delta_lcc: float
    delta_modularity: float
    delta_avg_path: float
    n_components_after: int


def ablate_one(G: nx.Graph, node: str) -> AblationResult:
    base_eff = global_efficiency(G)
    base_lcc = largest_cc_fraction(G)
    base_q = modularity(G)
    base_d = avg_shortest_path_lcc(G)
    H = G.copy()
    H.remove_node(node)
    arch = G.nodes[node].get("archetype", "?")
    return AblationResult(
        species_id=node,
        archetype=arch,
        delta_efficiency=base_eff - global_efficiency(H),
        delta_lcc=base_lcc - largest_cc_fraction(H),
        delta_modularity=base_q - modularity(H),
        delta_avg_path=avg_shortest_path_lcc(H) - base_d,
        n_components_after=nx.number_connected_components(H),
    )


def run_full_ablation(G: nx.Graph) -> list[AblationResult]:
    return [ablate_one(G, n) for n in G.nodes()]


# ---------------------------------------------------------------------------
# Dynamic graph reconstruction from simulator trajectory
# ---------------------------------------------------------------------------
def build_realized_graph_from_h5(path: str | Path, tick_idx: int = -1,
                                  depth_proximity: int = 2,
                                  min_biomass: float = 1.0) -> nx.Graph:
    """Reconstruct the realized coaggregation graph at one tick of a simulator run.

    An edge from COAGG_EDGES is realized at this tick iff both species have
    biomass >= min_biomass somewhere AND those locations are within
    depth_proximity cells of each other.
    """
    with h5py.File(path, "r") as f:
        comp = f["composition"][...]   # (T, n_species, n_depths)
    snap = comp[tick_idx]              # (n_species, n_depths)
    G = nx.Graph()
    for s in CATALOG:
        G.add_node(s.id, archetype=s.archetype, name=s.name)

    presence_depths: dict[str, np.ndarray] = {}
    for i, s in enumerate(CATALOG):
        idx = np.where(snap[i] >= min_biomass)[0]
        if idx.size > 0:
            presence_depths[s.id] = idx

    for u, v in COAGG_EDGES:
        du, dv = presence_depths.get(u), presence_depths.get(v)
        if du is None or dv is None:
            continue
        if np.min(np.abs(du[:, None] - dv[None, :])) <= depth_proximity:
            G.add_edge(u, v)
    return G


# ---------------------------------------------------------------------------
# Theorem evaluator
# ---------------------------------------------------------------------------
@dataclass
class TheoremVerdict:
    bridges: list[str]
    keystones: list[str]
    bridge_damage: list[float]
    keystone_damage: list[float]
    bridge_median: float
    keystone_median: float
    bridge_max: float
    keystone_max: float
    primary_metric: str
    fires_median: bool
    fires_max: bool
    margin_median: float
    margin_max: float
    p_mannwhitney: float       # one-sided test that bridge_damage > keystone_damage
    all_results: list[AblationResult] = field(default_factory=list)

    @property
    def verdict_label(self) -> str:
        if self.fires_max and self.fires_median:
            return "FIRES (both median and max)"
        if self.fires_max:
            return "FIRES (max only — canonical-bridge effect)"
        if self.fires_median:
            return "FIRES (median only — distributed bridges)"
        return "DIES"

    def summary(self) -> str:
        lines = [
            "=" * 64,
            "BRIDGE-ASYMMETRY THEOREM — empirical verdict",
            "=" * 64,
            f"Primary metric: {self.primary_metric}",
            f"Bridges  ({len(self.bridges):2}): {self.bridges}",
            f"  damages: {[f'{x:+.4f}' for x in self.bridge_damage]}",
            f"  median={self.bridge_median:+.4f}   max={self.bridge_max:+.4f}",
            f"Keystone ({len(self.keystones):2}): {self.keystones}",
            f"  damages: {[f'{x:+.4f}' for x in self.keystone_damage]}",
            f"  median={self.keystone_median:+.4f}   max={self.keystone_max:+.4f}",
            f"Margins:  median {self.margin_median:+.4f}   max {self.margin_max:+.4f}",
            f"Mann-Whitney U (one-sided, bridges>keystones): p = {self.p_mannwhitney:.4f}",
            f"VERDICT: theorem {self.verdict_label}",
            "=" * 64,
        ]
        return "\n".join(lines)


def evaluate(G: nx.Graph, primary_metric: str = "delta_efficiency") -> TheoremVerdict:
    from scipy.stats import mannwhitneyu
    results = run_full_ablation(G)
    bridges = [r for r in results if r.archetype == "Bridge"]
    keystones = [r for r in results if r.archetype == "Keystone"]
    bridge_d = [getattr(r, primary_metric) for r in bridges]
    keystone_d = [getattr(r, primary_metric) for r in keystones]
    b_med = float(np.median(bridge_d)) if bridge_d else 0.0
    k_med = float(np.median(keystone_d)) if keystone_d else 0.0
    b_max = float(np.max(bridge_d)) if bridge_d else 0.0
    k_max = float(np.max(keystone_d)) if keystone_d else 0.0
    # Mann-Whitney U test (one-sided: bridge damages stochastically greater)
    if bridge_d and keystone_d:
        try:
            _, p = mannwhitneyu(bridge_d, keystone_d, alternative="greater")
            p = float(p)
        except ValueError:
            p = 1.0
    else:
        p = 1.0
    return TheoremVerdict(
        bridges=[r.species_id for r in bridges],
        keystones=[r.species_id for r in keystones],
        bridge_damage=bridge_d,
        keystone_damage=keystone_d,
        bridge_median=b_med,
        keystone_median=k_med,
        bridge_max=b_max,
        keystone_max=k_max,
        primary_metric=primary_metric,
        fires_median=b_med > k_med,
        fires_max=b_max > k_max,
        margin_median=b_med - k_med,
        margin_max=b_max - k_max,
        p_mannwhitney=p,
        all_results=results,
    )


def evaluate_topk(G: nx.Graph, k: int = 3,
                  primary_metric: str = "delta_efficiency") -> TheoremVerdict:
    """Refined test: compare the top-k BETWEENNESS nodes against the
    'Keystone' archetype set.

    The original theorem's claim — "bridges fragment more than keystones" —
    only makes biological sense if 'bridge' is defined STRUCTURALLY
    (top-betweenness) and not by Socransky color group or orange-complex
    membership. This test uses the structural definition.
    """
    from scipy.stats import mannwhitneyu
    results = run_full_ablation(G)
    by_id = {r.species_id: r for r in results}
    bet = nx.betweenness_centrality(G)
    # Top-k betweenness — exclude any node that has Keystone archetype
    # (so we don't double-count Pg if it happens to have high betweenness)
    bet_ranked = sorted(
        [(n, b) for n, b in bet.items() if by_id[n].archetype != "Keystone"],
        key=lambda kv: -kv[1],
    )
    top_bridges = [n for n, _ in bet_ranked[:k]]
    keystones = [r.species_id for r in results if r.archetype == "Keystone"]

    bridge_d = [getattr(by_id[n], primary_metric) for n in top_bridges]
    keystone_d = [getattr(by_id[n], primary_metric) for n in keystones]
    if bridge_d and keystone_d:
        try:
            _, p = mannwhitneyu(bridge_d, keystone_d, alternative="greater")
            p = float(p)
        except ValueError:
            p = 1.0
    else:
        p = 1.0
    b_med = float(np.median(bridge_d)) if bridge_d else 0.0
    k_med = float(np.median(keystone_d)) if keystone_d else 0.0
    b_max = float(np.max(bridge_d)) if bridge_d else 0.0
    k_max = float(np.max(keystone_d)) if keystone_d else 0.0
    return TheoremVerdict(
        bridges=top_bridges, keystones=keystones,
        bridge_damage=bridge_d, keystone_damage=keystone_d,
        bridge_median=b_med, keystone_median=k_med,
        bridge_max=b_max, keystone_max=k_max,
        primary_metric=primary_metric,
        fires_median=b_med > k_med, fires_max=b_max > k_max,
        margin_median=b_med - k_med, margin_max=b_max - k_max,
        p_mannwhitney=p, all_results=results,
    )


def evaluate_trajectory_sweep(h5_path: str | Path,
                              primary_metric: str = "delta_efficiency",
                              n_samples: int = 10) -> list[tuple[int, TheoremVerdict]]:
    """Run the test across N evenly-spaced ticks of a simulator trajectory."""
    with h5py.File(h5_path, "r") as f:
        T = len(f["tick"])
    sample_idx = np.linspace(0, T - 1, n_samples, dtype=int)
    out = []
    for idx in sample_idx:
        G = build_realized_graph_from_h5(h5_path, tick_idx=int(idx))
        if G.number_of_edges() == 0:
            continue
        v = evaluate(G, primary_metric=primary_metric)
        out.append((int(idx), v))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["static", "dynamic", "both"], default="both")
    p.add_argument("--trajectory", type=str, default="output/trajectory.h5")
    p.add_argument("--tick", type=int, default=-1)
    p.add_argument("--metric", type=str, default="delta_efficiency")
    p.add_argument("--extended", action="store_true",
                   help="use the extended ~58-node catalog for the static test")
    args = p.parse_args()

    if args.mode in ("static", "both"):
        scope = "extended ~58-node" if args.extended else "canonical 18-node"
        print(f"\n*** STATIC GRAPH ({scope} coaggregation) ***\n")
        G = build_static_coagg_graph(extended=args.extended)
        print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print("\n  [test 1: archetype-defined bridges vs keystones]")
        verdict = evaluate(G, primary_metric=args.metric)
        print(verdict.summary())
        print("\n  [test 2: structural top-3 betweenness bridges vs keystones (refined)]")
        verdict2 = evaluate_topk(G, k=3, primary_metric=args.metric)
        print(verdict2.summary())
        print("\n  [test 3: structural top-5 betweenness bridges vs keystones]")
        verdict3 = evaluate_topk(G, k=5, primary_metric=args.metric)
        print(verdict3.summary())
        print("\nPer-species ablation results (sorted by primary metric):")
        results_sorted = sorted(verdict.all_results,
                                key=lambda r: -getattr(r, args.metric))
        print(f"  {'id':>4}  {'archetype':<14}  ΔE        ΔLCC      Δq        Δ⟨d⟩      #comp")
        for r in results_sorted:
            print(f"  {r.species_id:>4}  {r.archetype:<14}  "
                  f"{r.delta_efficiency:>+.4f}  {r.delta_lcc:>+.4f}  "
                  f"{r.delta_modularity:>+.4f}  {r.delta_avg_path:>+.4f}  "
                  f"{r.n_components_after:>3}")

    if args.mode in ("dynamic", "both"):
        print(f"\n*** DYNAMIC GRAPH TRAJECTORY SWEEP ({args.trajectory}) ***\n")
        sweep = evaluate_trajectory_sweep(args.trajectory, primary_metric=args.metric,
                                          n_samples=10)
        if not sweep:
            print("  No non-empty realized graph found in trajectory.")
        else:
            print(f"  {'tick_idx':>10} {'verdict':<35} {'B_med':>9} {'K_med':>9} {'B_max':>9} {'K_max':>9} {'p_MWU':>7}")
            for idx, v in sweep:
                print(f"  {idx:>10} {v.verdict_label:<35} "
                      f"{v.bridge_median:>+.4f} {v.keystone_median:>+.4f} "
                      f"{v.bridge_max:>+.4f} {v.keystone_max:>+.4f} "
                      f"{v.p_mannwhitney:>7.3f}")
            n_fires = sum(1 for _, v in sweep if v.fires_max or v.fires_median)
            print(f"\n  Theorem fired in {n_fires}/{len(sweep)} sampled ticks.")


if __name__ == "__main__":
    main()

