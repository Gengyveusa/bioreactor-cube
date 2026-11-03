"""Minimal Graph Convolutional Network — pure NumPy.

A 2-layer GCN over the 18-node coaggregation graph, with 9-dimensional node
features extracted from the Species catalog. The GCN regresses on each species'
network-damage score (ΔE from bridge_asymmetry.evaluate). High R^2 means
"the impact of removing a species is mechanistically predictable from its
biology + neighborhood," which is the falsifiable substrate of the
Bridge-Asymmetry Theorem.

Why numpy and not torch:
    n=18 nodes. 9-dim features. ~300 trainable params. Eigen-solve in microseconds.
    GPU overhead would dominate. Numpy is the right tool here.

Architecture:
    H^(0) = X                            (18 x 9 feature matrix)
    H^(1) = relu( A_hat @ H^(0) @ W1 )   (18 x H hidden)
    y_hat = A_hat @ H^(1) @ W2           (18 x 1 prediction)
    where A_hat = D^(-1/2) (A + I) D^(-1/2)   (sym-normalized adjacency w/ self-loops)

Training: MSE loss, scipy.optimize.minimize (L-BFGS-B) on the flattened W1, W2.
Feature attribution: integrated gradients (numerically) of y_hat[Fn] wrt X[Fn].
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import networkx as nx
from scipy.optimize import minimize

from ..species import CATALOG, COAGG_EDGES
from .bridge_asymmetry import build_static_coagg_graph, evaluate


# ---------------------------------------------------------------------------
# Feature extraction — turn the 9-dim node schema into a numeric feature matrix
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "o2_optimum",         # Layer 3 chemistry
    "ph_optimum",
    "sugar_pref",
    "heme_req",
    "vesiculation_rate",  # Dim 7 OMV
    "NO_yield",           # Dim 9 coherence
    "H2S_yield",
    "pH_effect",
    "is_aerotolerant",    # Dim 3 — the bridge-enabling property
]


def build_feature_matrix() -> np.ndarray:
    """Return X of shape (n_species, n_features)."""
    X = np.zeros((len(CATALOG), len(FEATURE_NAMES)), dtype=np.float32)
    for i, s in enumerate(CATALOG):
        X[i, 0] = s.envelope.o2[2]
        X[i, 1] = s.envelope.ph[2]
        X[i, 2] = s.envelope.sugar
        X[i, 3] = s.envelope.heme
        X[i, 4] = s.vesiculation_rate
        X[i, 5] = s.NO_yield
        X[i, 6] = s.H2S_yield
        X[i, 7] = s.pH_effect
        X[i, 8] = 1.0 if s.o2_class == "aerotolerant" else 0.0
    # Z-normalize each column
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    return (X - mu) / sd


def build_target_vector() -> np.ndarray:
    """Each species' ablation impact on global efficiency."""
    G = build_static_coagg_graph()
    v = evaluate(G, primary_metric="delta_efficiency")
    # Order in CATALOG order
    by_id = {r.species_id: r.delta_efficiency for r in v.all_results}
    y = np.array([by_id[s.id] for s in CATALOG], dtype=np.float32)
    return y


# ---------------------------------------------------------------------------
# Normalized adjacency
# ---------------------------------------------------------------------------
def sym_normalized_adjacency() -> np.ndarray:
    G = build_static_coagg_graph()
    n = G.number_of_nodes()
    id_to_idx = {s.id: i for i, s in enumerate(CATALOG)}
    A = np.zeros((n, n), dtype=np.float32)
    for u, v in G.edges():
        i, j = id_to_idx[u], id_to_idx[v]
        A[i, j] = 1.0
        A[j, i] = 1.0
    A_tilde = A + np.eye(n, dtype=np.float32)
    d = A_tilde.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d + 1e-9)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt


# ---------------------------------------------------------------------------
# GCN forward
# ---------------------------------------------------------------------------
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


@dataclass
class GCN:
    A_hat: np.ndarray         # (n, n) normalized adjacency
    hidden_dim: int = 8
    n_features: int = len(FEATURE_NAMES)

    @property
    def n_params(self) -> int:
        return self.n_features * self.hidden_dim + self.hidden_dim * 1

    def unflatten(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cut = self.n_features * self.hidden_dim
        W1 = theta[:cut].reshape(self.n_features, self.hidden_dim)
        W2 = theta[cut:].reshape(self.hidden_dim, 1)
        return W1, W2

    def forward(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        W1, W2 = self.unflatten(theta)
        H1 = relu(self.A_hat @ X @ W1)
        y_hat = (self.A_hat @ H1 @ W2).ravel()
        return y_hat

    def loss(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray,
             l2: float = 1e-3) -> float:
        y_hat = self.forward(theta, X)
        mse = float(np.mean((y_hat - y) ** 2))
        return mse + l2 * float(np.sum(theta ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray,
            seed: int = 42) -> tuple[np.ndarray, float]:
        rng = np.random.default_rng(seed)
        theta0 = rng.normal(0, 0.3, size=self.n_params).astype(np.float64)
        res = minimize(
            self.loss, theta0,
            args=(X, y),
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-10},
        )
        return res.x, float(res.fun)


# ---------------------------------------------------------------------------
# Integrated-gradients feature attribution
# ---------------------------------------------------------------------------
def integrated_gradients(model: GCN, theta: np.ndarray, X: np.ndarray,
                         target_node: int, steps: int = 50,
                         eps: float = 1e-3) -> np.ndarray:
    """Attribution of features at target_node to its predicted y_hat.

    For each (alpha in 0..1), interpolate X_target between zeros and the real X,
    compute partial derivative of y_hat[target] w.r.t. each feature numerically,
    sum across the path. The result is per-feature importance.
    """
    baseline = X.copy()
    baseline[target_node] = 0.0
    attribution = np.zeros(X.shape[1], dtype=np.float64)
    for alpha in np.linspace(1.0 / steps, 1.0, steps):
        X_alpha = baseline.copy()
        X_alpha[target_node] = alpha * X[target_node]
        for k in range(X.shape[1]):
            X_pos = X_alpha.copy()
            X_neg = X_alpha.copy()
            X_pos[target_node, k] += eps
            X_neg[target_node, k] -= eps
            dy = (model.forward(theta, X_pos)[target_node]
                  - model.forward(theta, X_neg)[target_node]) / (2 * eps)
            attribution[k] += dy
    attribution *= X[target_node] / steps
    return attribution


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def plot_attribution(theta: np.ndarray, model: "GCN", X: np.ndarray,
                     target_id: str, out_path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target_idx = next(i for i, s in enumerate(CATALOG) if s.id == target_id)
    attr = integrated_gradients(model, theta, X, target_idx)
    order = np.argsort(np.abs(attr))[::-1]
    names = [FEATURE_NAMES[i] for i in order]
    vals = [attr[i] for i in order]
    colors = ["#e08854" if v >= 0 else "#6cb8c9" for v in vals]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(range(len(names)), vals, color=colors, edgecolor="#222", linewidth=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0, color="#222", linewidth=0.8)
    ax.set_xlabel("Integrated-gradients attribution to predicted ΔE")
    ax.set_title(f"What makes {target_id} a bridge? GCN feature attribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Wrote {out_path}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=str, default="Fn",
                   help="Species ID to attribute features for")
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--plot", type=str, default="output/gnn_attribution_Fn.png")
    args = p.parse_args()

    X = build_feature_matrix()
    y = build_target_vector()
    A_hat = sym_normalized_adjacency()
    model = GCN(A_hat=A_hat, hidden_dim=args.hidden)

    print(f"Training GCN: n_nodes={X.shape[0]}, n_features={X.shape[1]}, "
          f"hidden={args.hidden}, n_params={model.n_params}")
    theta, final_loss = model.fit(X, y)

    y_hat = model.forward(theta, X)
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    print(f"Final loss: {final_loss:.6f}   R^2: {r2:.4f}")
    print()
    print("Per-species predictions vs true network damage (sorted by truth):")
    order = np.argsort(-y)
    print(f"  {'id':>4}  {'archetype':<14}  {'true_dE':>9}  {'pred_dE':>9}  {'err':>8}")
    for i in order:
        s = CATALOG[i]
        print(f"  {s.id:>4}  {s.archetype:<14}  {y[i]:>+.4f}  {y_hat[i]:>+.4f}  {y_hat[i]-y[i]:>+.4f}")

    # Feature attribution for the target species
    target_idx = next(i for i, s in enumerate(CATALOG) if s.id == args.target)
    print(f"\nIntegrated-gradients attribution for {args.target}:")
    attr = integrated_gradients(model, theta, X, target_idx)
    paired = sorted(zip(FEATURE_NAMES, attr), key=lambda kv: -abs(kv[1]))
    for name, val in paired:
        bar = "+" * int(abs(val) * 200) if val > 0 else "-" * int(abs(val) * 200)
        print(f"  {name:<22} {val:>+.4f}  {bar}")

    # Plot
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    plot_attribution(theta, model, X, args.target, args.plot)


if __name__ == "__main__":
    main()

