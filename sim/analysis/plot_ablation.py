"""Plot the bridge-asymmetry ablation result.

Per-species network-damage bar chart, colored by archetype, with the
bridge-vs-keystone group medians and the canonical-bridge spike (Fn) annotated.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import networkx as nx
from .bridge_asymmetry import build_static_coagg_graph, evaluate, evaluate_topk, run_full_ablation


ARCHETYPE_COLORS = {
    "Pioneer":         "#d4a574",
    "NitrateReducer":  "#7fae6f",
    "Bridge":          "#e08854",
    "Keystone":        "#c84a4a",
    "Aggressive":      "#9d6ec7",
    "Cariogenic":      "#e6c66c",
    "OSCC":            "#c75a9b",
    "CrossKingdom":    "#6cb8c9",
    "Mineralizer":     "#e8d18c",
}


def plot_static_ablation(out_path: str | Path) -> None:
    G = build_static_coagg_graph()
    v = evaluate(G, primary_metric="delta_efficiency")
    results = sorted(v.all_results, key=lambda r: -r.delta_efficiency)
    ids = [r.species_id for r in results]
    damages = [r.delta_efficiency for r in results]
    colors = [ARCHETYPE_COLORS.get(r.archetype, "#888") for r in results]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(range(len(ids)), damages, color=colors, edgecolor="#222", linewidth=0.6)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=0)
    ax.set_ylabel("ΔE   (drop in global efficiency on removal)")
    ax.set_xlabel("species (sorted by network damage)")
    ax.axhline(0, color="#333", linewidth=0.8)

    # Highlight bridge and keystone medians
    bridge_med = v.bridge_median
    keystone_med = v.keystone_median
    ax.axhline(bridge_med, color="#e08854", linestyle="--", linewidth=1.2,
               label=f"Bridge median = {bridge_med:+.4f}")
    ax.axhline(keystone_med, color="#c84a4a", linestyle="--", linewidth=1.2,
               label=f"Keystone median = {keystone_med:+.4f}")

    # Annotate Fn
    fn_idx = ids.index("Fn") if "Fn" in ids else None
    if fn_idx is not None:
        ax.annotate("Fn — the canonical bridge",
                    xy=(fn_idx, damages[fn_idx]),
                    xytext=(fn_idx + 1.5, damages[fn_idx] * 0.95),
                    fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#222"))

    # Title with verdict
    fires = "FIRES" if (v.fires_max or v.fires_median) else "DIES"
    ax.set_title(f"Bridge-Asymmetry Theorem (static graph, 18 nodes) — verdict: {fires}\n"
                 f"max margin {v.margin_max:+.4f}  ·  median margin {v.margin_median:+.4f}  "
                 f"·  Mann-Whitney p = {v.p_mannwhitney:.3f}")

    # Archetype legend
    seen = set()
    handles = []
    for r in results:
        if r.archetype not in seen:
            seen.add(r.archetype)
            handles.append(plt.Rectangle((0,0),1,1, color=ARCHETYPE_COLORS.get(r.archetype, "#888"),
                                         label=r.archetype))
    arch_legend = ax.legend(handles=handles, loc="upper right", fontsize=8, title="archetype")
    ax.add_artist(arch_legend)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Wrote {out_path}")


def plot_extended_refined(out_path: str | Path) -> None:
    """The headline plot from the extended graph: structural-betweenness bridges
    vs archetype-bridges vs keystones. Three-panel.
    """
    G = build_static_coagg_graph(extended=True)
    v_arch = evaluate(G, primary_metric="delta_efficiency")
    v_top3 = evaluate_topk(G, k=3, primary_metric="delta_efficiency")
    v_top5 = evaluate_topk(G, k=5, primary_metric="delta_efficiency")
    results = run_full_ablation(G)
    bet = nx.betweenness_centrality(G)
    by_id = {r.species_id: r for r in results}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ---- panel 1: scatter, betweenness vs damage, colored by archetype ----
    ax = axes[0]
    xs, ys, cs, names = [], [], [], []
    for r in results:
        xs.append(bet[r.species_id])
        ys.append(r.delta_efficiency)
        cs.append(ARCHETYPE_COLORS.get(r.archetype, "#888"))
        names.append(r.species_id)
    ax.scatter(xs, ys, c=cs, edgecolor="#222", linewidth=0.5, s=70)
    for x, y, n in zip(xs, ys, names):
        if y > 0.01 or n == "Pg" or x > 0.1:
            ax.annotate(n, (x, y), xytext=(4, 3), textcoords="offset points",
                        fontsize=8)
    ax.set_xlabel("Betweenness centrality (structural)")
    ax.set_ylabel("ΔE — network damage on removal")
    ax.set_title("Bridge-Asymmetry — betweenness is the actual predictor\n"
                 "(not the Socransky/archetypal Bridge label)")
    ax.axhline(0, color="#333", linewidth=0.5)

    # legend
    seen = set()
    handles = []
    for r in results:
        if r.archetype not in seen:
            seen.add(r.archetype)
            handles.append(plt.Rectangle((0,0),1,1,
                          color=ARCHETYPE_COLORS.get(r.archetype, "#888"),
                          label=r.archetype))
    ax.legend(handles=handles, loc="upper left", fontsize=7, title="archetype")

    # ---- panel 2: archetype test (the FAILURE) ----
    ax = axes[1]
    arch_b = v_arch.bridge_damage
    arch_k = v_arch.keystone_damage
    ax.boxplot([arch_b, arch_k], labels=["Bridges (archetype, n=17)", "Keystones (n=7)"],
               patch_artist=True,
               boxprops=dict(facecolor="#e0885455"), medianprops=dict(color="#222"))
    ax.scatter(np.random.normal(1, 0.04, len(arch_b)), arch_b,
               color="#e08854", edgecolor="#222", s=50, alpha=0.7, zorder=3)
    ax.scatter(np.random.normal(2, 0.04, len(arch_k)), arch_k,
               color="#c84a4a", edgecolor="#222", s=50, alpha=0.7, zorder=3)
    ax.axhline(0, color="#333", linewidth=0.5)
    ax.set_ylabel("ΔE")
    ax.set_title(f"Strong theorem (archetype Bridges) — FAILS\n"
                 f"Mann-Whitney p = {v_arch.p_mannwhitney:.3f}  ·  "
                 f"median margin {v_arch.margin_median:+.4f}")

    # ---- panel 3: refined test (the SUCCESS) ----
    ax = axes[2]
    top3_b = v_top3.bridge_damage
    top5_b = v_top5.bridge_damage
    keystone_d = v_top3.keystone_damage
    ax.boxplot([top3_b, top5_b, keystone_d],
               labels=[f"Top-3 betw\n({','.join(v_top3.bridges)})",
                       f"Top-5 betw\n({','.join(v_top5.bridges)})",
                       f"Keystones (n=7)"],
               patch_artist=True,
               boxprops=dict(facecolor="#7fae6f55"), medianprops=dict(color="#222"))
    ax.scatter(np.random.normal(1, 0.04, len(top3_b)), top3_b,
               color="#7fae6f", edgecolor="#222", s=50, alpha=0.7, zorder=3)
    ax.scatter(np.random.normal(2, 0.04, len(top5_b)), top5_b,
               color="#5d8e4f", edgecolor="#222", s=50, alpha=0.7, zorder=3)
    ax.scatter(np.random.normal(3, 0.04, len(keystone_d)), keystone_d,
               color="#c84a4a", edgecolor="#222", s=50, alpha=0.7, zorder=3)
    ax.axhline(0, color="#333", linewidth=0.5)
    ax.set_ylabel("ΔE")
    ax.set_title(f"Refined theorem (structural betweenness) — FIRES\n"
                 f"top-3 p = {v_top3.p_mannwhitney:.4f}  ·  top-5 p = {v_top5.p_mannwhitney:.4f}")

    fig.suptitle(f"Bridge-Asymmetry Theorem — extended 58-node graph "
                 f"({G.number_of_edges()} edges)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    plot_static_ablation("output/bridge_asymmetry_static.png")
    plot_extended_refined("output/bridge_asymmetry_extended_refined.png")

