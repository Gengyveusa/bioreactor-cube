"""The ask-anything layer — the LLM half of 'maximal AI' (increment 4).

'pow, you're a cube in the artery — tell me what's here.' Two pieces:
  state_digest(eng) — distil the LIVE engine state into structured FACTS (ψ, cells by depth, gradients,
                      GNN roles, realised coaggregations + their molecular basis, body-map efflux).
  answer(q, digest) — a GROUNDED natural-language answer. Every number it states comes from the digest;
                      it never invents simulation results. answer_llm() is the documented hook where a
                      live Claude call slots in (same digest as context) for richer phrasing.
"""

from __future__ import annotations

import numpy as np

from sim.cube import ai
from sim.cube import ecm as ecmmod

_MICRO = set(ai._IDX)


def _band_top(eng, k0: int, k1: int, n: int = 3) -> list[tuple[str, float]]:
    comp = eng.grid.composition[:, :, k0:k1, :].sum(axis=(0, 1, 2))
    order = np.argsort(-comp)
    return [(eng.species[i], round(float(comp[i]), 2)) for i in order[:n] if comp[i] > 1e-6]


def state_digest(eng, name: str = "cube") -> dict:
    """Distil the live engine into factual, citable state."""
    nz = eng.grid.shape[2]
    t = max(1, nz // 3)
    psi = {k: v for k, v in eng.readout().items() if k != "fields"}
    o2 = eng.grid.get_field("O2")
    d = {
        "name": name,
        "tissue": eng.tissue.name,
        "psi": psi,
        "zones": {"surface": _band_top(eng, 2 * t, nz), "middle": _band_top(eng, t, 2 * t),
                  "deep": _band_top(eng, 0, t)},
        "gradients": {"O2_surface": round(float(o2[:, :, -1].mean()), 2),
                      "O2_deep": round(float(o2[:, :, 0].mean()), 2)},
    }
    if set(eng.species) & _MICRO:                      # microbial tissue → GNN roles + coaggregation
        roles = ai.learn_roles()
        present = {c for z in d["zones"].values() for c, _ in z}
        d["roles_present"] = {c: roles[c] for c in present if c in roles}
        d["coaggregations"] = ai.active_coaggregations(eng, top=6)
        d["gcn_r2"] = round(ai.gcn_r2(), 3)
    if "inflammation" in eng.extra:                    # artery → systemic-arrival readout
        d["inflammation_load"] = round(float(eng.extra["inflammation"].sum()), 1)
    spec = getattr(eng.tissue, "ecm", None)            # the characterised extracellular matrix
    if spec is not None:
        density = d["psi"].get("ecm_density", spec.base_density)
        d["ecm"] = {"describe": ecmmod.describe(spec), **ecmmod.ecm_metrics(spec, density)}
    return d


def digest_bodymap(bm) -> dict:
    """Whole-body digest: each cube + the bloodstream links (efflux carried between cubes)."""
    cubes = {name: state_digest(eng, name) for name, eng in bm.cubes.items()}
    links = [{"src": s, "dst": dt, "k": k, "efflux": round(bm.cubes[s].efflux(), 3)}
             for s, dt, k in bm.edges]
    return {"cubes": cubes, "links": links}


# --------------------------------------------------------------------- prose helpers
# psi is a dict of engine scalars. It must NEVER reach a reader as a raw Python dict repr.
_PSI_LABEL = {
    "bbb_integrity": "blood-brain-barrier integrity",
    "glymphatic_clearance": "glymphatic clearance",
    "amyloid_load": "amyloid-beta load",
    "neuroinflammation": "neuroinflammation",
    "hepatocyte_function": "hepatocyte function",
    "steatosis": "steatosis (fat)",
    "fibrosis": "fibrosis",
    "hepatic_inflammation": "hepatic inflammation",
    "inflammation_load": "arriving inflammatory load",
    "mmp_activity": "MMP activity",
    "ecm_density": "ECM density",
    "ecm_porosity": "ECM porosity",
    "biomass": "total biomass",
}


def _num(v) -> str:
    """Format one psi scalar for prose. Example: _num(4701.0) -> '4,701'."""
    if v is None:
        return "-"
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    a = abs(float(v))
    if a >= 100:
        return f"{v:,.0f}"
    if a >= 1:
        return (f"{v:.2f}".rstrip("0").rstrip(".")) or "0"
    return (f"{v:.3f}".rstrip("0").rstrip(".")) or "0"


def _psi_prose(p: dict, skip: tuple = ()) -> str:
    """psi as readable prose. Example: _psi_prose({'fibrosis': 0.123}) -> 'fibrosis 0.123'."""
    return ", ".join(f"{_PSI_LABEL.get(k, k.replace('_', ' '))} {_num(v)}"
                     for k, v in p.items() if k not in skip)


def _arrival(dst: dict) -> str:
    """The destination tissue's OWN arrival sensor, so no tissue ever prints a bare None.
    Example: _arrival(brain_digest) -> 'BBB integrity 0.401 (1.0 = sealed)'."""
    dp = dst.get("psi", {}) or {}
    for key, label, tail in (("endothelial_activation", "endothelial activation", ""),
                             ("bbb_integrity", "BBB integrity", " (1.0 = sealed)"),
                             ("hepatic_inflammation", "hepatic inflammation", "")):
        if dp.get(key) is not None:
            return f"{label} {_num(dp[key])}{tail}"
    return "no arrival sensor is defined for this tissue yet"


def answer(question: str, digest: dict, body: dict | None = None) -> str:
    """A natural-language answer GROUNDED in the digest. Deterministic; cites the real numbers."""
    q = question.lower()
    p = digest["psi"]

    def cells() -> str:
        parts = [f"{zone} {', '.join(f'{c} ({b})' for c, b in items)}"
                 for zone, items in digest["zones"].items() if items]
        return "Cells by depth — " + "; ".join(parts) + "."

    def function() -> str:
        if "dysbiosis" in p:
            return (f"This is {digest['tissue']}. State (ψ): dysbiosis index {p['dysbiosis']}, "
                    f"NO output {p.get('NO')}, OMV flux {p.get('omv_flux')}, total biomass {p.get('biomass')}. "
                    f"O₂ runs {digest['gradients']['O2_surface']} at the surface to "
                    f"{digest['gradients']['O2_deep']} deep — the gradient that stratifies the community.")
        if "plaque_burden" in p:
            return (f"This is {digest['tissue']}. State (ψ): plaque burden {p['plaque_burden']}, "
                    f"foam-cell fraction {p['foam_fraction']}, endothelial activation "
                    f"{p['endothelial_activation']}, inflammatory load {p.get('inflammation_load')}. "
                    "Endothelial activation is the direct sensor of the load arriving from the blood.")
        if "bbb_integrity" in p:
            return (f"This is {digest['tissue']}. State: blood-brain-barrier integrity "
                    f"{_num(p['bbb_integrity'])} (1.0 = sealed), glymphatic clearance "
                    f"{_num(p['glymphatic_clearance'])}, amyloid-beta load {_num(p['amyloid_load'])}, "
                    f"neuroinflammation {_num(p['neuroinflammation'])}. The barrier is the sensor here - "
                    "it degrades first, and amyloid accumulates behind it.")
        if "hepatocyte_function" in p:
            return (f"This is {digest['tissue']}. State: hepatocyte function "
                    f"{_num(p['hepatocyte_function'])} (1.0 = healthy), steatosis {_num(p['steatosis'])}, "
                    f"fibrosis {_num(p['fibrosis'])}, hepatic inflammation "
                    f"{_num(p['hepatic_inflammation'])}. Steatosis is the metabolic lesion, "
                    "fibrosis the structural one; they advance together.")
        return f"This is {digest['tissue']}. State: " + _psi_prose(p) + "."

    def drivers() -> str:
        if "dysbiosis" in p:
            deep = digest["zones"]["deep"]
            lead = deep[0][0] if deep else "—"
            return (f"What's driving it: the anoxic deep zone (O₂ {digest['gradients']['O2_deep']}) plus "
                    f"heme from bleeding lets anaerobes ({lead}…) express and shed OMVs; dysbiosis index "
                    f"is {p['dysbiosis']}. Terrain gates which information expresses where.")
        if "plaque_burden" in p:
            return (f"What's driving it: inflammatory load {p.get('inflammation_load')} arriving at the "
                    f"lumen activates endothelium ({p['endothelial_activation']}), recruiting monocytes "
                    f"that take up oxLDL and become foam cells (fraction {p['foam_fraction']}).")
        if "bbb_integrity" in p:
            return (f"What's driving it: inflammatory load {_num(p.get('inflammation_load'))} arriving at the "
                    f"capillary drives microglial MMP activity ({_num(p.get('mmp_activity'))}), which degrades "
                    f"the tight junctions - BBB integrity is down to {_num(p['bbb_integrity'])}. The same "
                    f"inflammation suppresses AQP4 glymphatic clearance to {_num(p['glymphatic_clearance'])}, "
                    f"so amyloid-beta is removed slower than it is made and pools to {_num(p['amyloid_load'])}. "
                    f"Failed clearance, not extra production, is the driver.")
        if "hepatocyte_function" in p:
            return (f"What's driving it: the liver takes FIRST-PASS portal blood, so the arriving load "
                    f"{_num(p.get('inflammation_load'))} reaches Kupffer cells here before anywhere else "
                    f"(hepatic inflammation {_num(p['hepatic_inflammation'])}). Activated Kupffer cells switch "
                    f"on stellate cells in the space of Disse, which deposit collagen - fibrosis "
                    f"{_num(p['fibrosis'])} - while lipotoxic stress accumulates fat (steatosis "
                    f"{_num(p['steatosis'])}). Between them they pull hepatocyte function to "
                    f"{_num(p['hepatocyte_function'])}. One inlet, two lesions.")
        return (f"What's driving it: no tissue-specific driver model is wired for {digest['tissue']} yet, so "
                f"what the engine can state is the terrain - O2 runs {digest['gradients']['O2_surface']} at "
                f"the surface to {digest['gradients']['O2_deep']} deep - and the current state: "
                + _psi_prose(p, skip=("ecm_density", "ecm_porosity")) + ".")

    def attachment() -> str:
        co = digest.get("coaggregations")
        if not co:
            return "No microbial coaggregation network in this tissue."
        lines = []
        for e in co[:5]:
            bind = ai.predict_binding(e["a"], e["b"])
            mol = (f"{bind['adhesin']} ↔ {bind['receptor']}" if bind["basis"] == "curated structure"
                   else f"affinity {bind['score']} ({bind['basis']})")
            lines.append(f"{e['a']}–{e['b']} (realised weight {e['weight']}; {mol})")
        return ("Realised coaggregations (co-presence × GCN-learned affinity), with molecular basis — "
                + "; ".join(lines) + f". [GCN R²≈{digest.get('gcn_r2')}; binding curated/ESMFold-pending]")

    def roles() -> str:
        rp = digest.get("roles_present")
        if not rp:
            return "No GNN role analysis for this tissue."
        ks = sorted(rp.items(), key=lambda kv: -kv[1]["keystone_score"])
        return ("GCN-derived roles of cells present — "
                + "; ".join(f"{c}: {r['role']} (keystone {r['keystone_score']}, "
                            f"betweenness {r['betweenness']})" for c, r in ks)
                + f". [learned signal, weak: R²≈{digest.get('gcn_r2')}]")

    def systemic() -> str:
        if not body:
            return ("This cube in isolation. Link it into the body-map to see bloodstream propagation "
                    "(e.g. oral OMV efflux → artery).")
        out = []
        for link in body["links"]:
            src, dst = body["cubes"][link["src"]], body["cubes"][link["dst"]]
            out.append(f"{link['src']} sheds OMV efflux {link['efflux']} into the blood → {link['dst']}; "
                       f"there the arriving inflammatory load is {_num(dst.get('inflammation_load'))} "
                       f"and {_arrival(dst)}.")
        return "Systemic propagation (Layer-7, spatial): " + " ".join(out)

    if any(w in q for w in ("ecm", "matrix", "extracellular", "scaffold", "wire", "collagen",
                            "mesh", "stroma", "porosity")):
        e = digest.get("ecm")
        return e["describe"] if e else "No characterised ECM for this tissue."
    if any(w in q for w in ("attach", "bind", "coaggreg", "connect", "stick", "adhes")):
        return attachment()
    if any(w in q for w in ("role", "keystone", "important", "hub", "driver of the network")):
        return roles()
    if any(w in q for w in ("systemic", "blood", "artery", "heart", "spread", "travel", "distant")):
        return systemic()
    if any(w in q for w in ("cell", "who", "what's here", "whats here", "composition", "species")):
        return cells()
    if any(w in q for w in ("why", "driv", "cause", "because")):
        return drivers()
    if any(w in q for w in ("function", "do", "state", "psi", "ψ", "health", "disease", "how bad")):
        return function()
    return function() + " " + cells()


# Common questions used to seed the viewer's ask-anything panel (answered from the live digest).
DEFAULT_QUESTIONS = [
    "What cells are here?",
    "What is the function / state?",
    "What is driving it?",
    "What attaches to what?",
    "What is the extracellular matrix here?",
    "Does it spread systemically?",
]


def answer_llm(question: str, digest: dict, body: dict | None = None) -> str:
    """Hook for a LIVE Claude call. The digest is passed verbatim as grounding context so the model
    phrases richly but cannot invent simulation numbers. Falls back to the grounded answerer offline."""
    # An API integration would build a prompt from `digest`/`body` and call Claude here.
    return answer(question, digest, body)

