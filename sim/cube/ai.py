"""The AI layer for the cube — the COMPUTE half of 'maximal AI' (increment 4).

Three real components, each honestly scoped:
  1. GNN roles      — reuses the trained 2-layer GCN (sim/analysis/gnn.py, numpy+L-BFGS) over the
                      18-node coaggregation graph. Real learned node embeddings + a keystone score
                      (predicted network-damage ΔE). NOTE: the regression is weak (R²≈0.27, n=18) — the
                      embeddings are used for structure, the score is reported as a weak learned signal.
  2. Attachment     — per-voxel 'which coaggregations are REALISED here', from co-presence × learned
                      embedding affinity. The edge SET is curated biology (COAGG_EDGES); the per-voxel
                      activation and affinity weighting are computed. (This is the honest answer to
                      "are we using a GNN for connectivity yet" — yes, for activation/affinity.)
  3. Binding        — adhesin↔receptor propensity. A curated literature table + an embedding-heuristic
                      fallback, standing in for the real ESMFold + AutoDock-Vina structural pipeline
                      (not run here — flagged 'pending'). Explains the MOLECULAR basis of attachment.
"""

from __future__ import annotations

import functools

import networkx as nx
import numpy as np

from sim.analysis.bridge_asymmetry import build_static_coagg_graph
from sim.analysis.gnn import (GCN, build_feature_matrix, build_target_vector, relu,
                              sym_normalized_adjacency)
from sim.species import CATALOG, COAGG_EDGES

_IDX = {s.id: i for i, s in enumerate(CATALOG)}


@functools.lru_cache(maxsize=1)
def _trained() -> dict:
    """Train the GCN once; cache embeddings, predictions, and R²."""
    X, y, A = build_feature_matrix(), build_target_vector(), sym_normalized_adjacency()
    model = GCN(A_hat=A, hidden_dim=8)
    theta, _ = model.fit(X, y)
    yh = model.forward(theta, X)
    r2 = 1.0 - ((y - yh) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    W1, _ = model.unflatten(theta)
    emb = relu(A @ X @ W1)                       # learned node embeddings (18 × 8)
    return {"emb": emb, "pred": yh, "r2": float(r2)}


def gcn_r2() -> float:
    return _trained()["r2"]


def embedding_affinity(a: str, b: str) -> float:
    """Cosine similarity of two species in the GCN's learned embedding space (−1..1)."""
    if a not in _IDX or b not in _IDX:
        return 0.0
    H = _trained()["emb"]
    va, vb = H[_IDX[a]], H[_IDX[b]]
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))


def learn_roles() -> dict:
    """Per-species network role from the GCN keystone score + graph centrality. Real, interpretable."""
    t = _trained()
    pred = t["pred"]
    G = build_static_coagg_graph()
    btw = nx.betweenness_centrality(G)
    deg = {s.id: G.degree(s.id) for s in CATALOG}
    ks_cut = np.quantile(pred, 0.75)
    bt_cut = np.quantile(list(btw.values()), 0.75)
    roles = {}
    for i, s in enumerate(CATALOG):
        if pred[i] >= ks_cut:
            role = "keystone"
        elif btw.get(s.id, 0) >= bt_cut:
            role = "connector"
        elif s.o2_class == "aerotolerant":
            role = "frontier"
        else:
            role = "commensal"
        roles[s.id] = {"role": role, "keystone_score": round(float(pred[i]), 3),
                       "betweenness": round(float(btw.get(s.id, 0)), 3), "degree": deg[s.id]}
    return roles


def active_coaggregations(eng, top: int = 12) -> list[dict]:
    """Which curated coaggregations are REALISED in this cube — both partners co-present — ranked by
    co-presence × learned embedding affinity. Empty for non-microbial tissues (no COAGG_EDGES)."""
    comp = eng.grid.composition
    cidx = {c: n for n, c in enumerate(eng.species)}
    out = []
    for a, b in COAGG_EDGES:
        if a in cidx and b in cidx:
            copres = float(np.minimum(comp[..., cidx[a]], comp[..., cidx[b]]).sum())
            if copres > 1e-6:
                aff = max(0.0, embedding_affinity(a, b))
                out.append({"a": a, "b": b, "copresence": round(copres, 3),
                            "affinity": round(aff, 3), "weight": round(copres * (0.5 + 0.5 * aff), 3)})
    out.sort(key=lambda e: -e["weight"])
    return out[:top]


# ---- structure / docking seam (curated stand-in for ESMFold + AutoDock-Vina) --------------------
# Real adhesin↔receptor pairs from the coaggregation literature. score = relative binding propensity.
_ADHESINS: dict[tuple[str, str], tuple[str, str, float, str]] = {
    ("Fn", "Sa"): ("RadD", "streptococcal receptor polysaccharide", 0.85,
                   "F. nucleatum RadD ↔ streptococci — the central bridging adhesin"),
    ("Fn", "Pg"): ("RadD", "Pg outer-membrane (Mfa/FimA)", 0.80,
                   "Fn presents the surface late-colonizer P. gingivalis docks onto"),
    ("Pg", "Sg"): ("Mfa1 fimbriae", "S. gordonii SspB", 0.90,
                   "P. gingivalis Mfa1 ↔ SspB — the classic coadhesion pair"),
    ("Aa", "Fn"): ("RTX/adhesin", "Fn RadD", 0.70,
                   "A. actinomycetemcomitans associates with the Fn bridge"),
    ("Sm", "An"): ("antigen I/II", "Actinomyces fimbriae", 0.65,
                   "early-colonizer streptococcus ↔ Actinomyces coaggregation"),
}


def predict_binding(a: str, b: str) -> dict:
    """Adhesin↔receptor binding propensity. Curated structure where known; else an embedding heuristic.
    Flagged for the real ESMFold/Vina pipeline (not computed here)."""
    for key in ((a, b), (b, a)):
        if key in _ADHESINS:
            ad, rec, score, note = _ADHESINS[key]
            return {"a": a, "b": b, "adhesin": ad, "receptor": rec, "score": score,
                    "basis": "curated structure", "note": note}
    aff = max(0.0, embedding_affinity(a, b))
    return {"a": a, "b": b, "score": round(aff, 2), "basis": "embedding-heuristic",
            "note": "no curated structure; ESMFold + Vina docking pending"}

