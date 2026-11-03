"""Atlas data — one polished cube per tissue, navigable from a body-map. (Front-end increment.)

Keeps the loved single-cube presentation (build_demo: real points, gradients, matruchotii playback) and
ADDS the new capabilities to every tissue: ψ readout, GNN roles / realised coaggregations / binding, and
the grounded ask-anything Q&A. The oral cube is the proven view untouched; the artery cube is the 2nd
tissue rendered the same way; the body-map ties them by the bloodstream.
"""

from __future__ import annotations

import numpy as np

from sim.cube import ai, query
from sim.cube.bodymap import BodyMap
from sim.cube.dynamics import build_demo
from sim.cube.fused import CubeEpoch4
from sim.cube.grid import CubeGrid
from sim.cube.sample import _downsample_field
from sim.cube.tissues import ARTERY_TISSUE, BRAIN_TISSUE, LIVER_TISSUE, ORAL_HEALTHY, ORAL_TISSUE

_CELL_COLOR = {"EC": "#5b9bd5", "Mono": "#e6a83c", "Foam": "#d6543f", "SMC": "#9a86c4",
               "BEC": "#5b9bd5", "Peri": "#c98a8a", "Astro": "#5fb89a", "Micro": "#e6a83c", "Neuron": "#b89cd6",
               "Hep": "#d9a441", "Kup": "#d6543f", "LSEC": "#5b9bd5", "Stel": "#9a86c4"}
_CELL_LABEL = {"EC": "endothelium", "Mono": "monocyte/macrophage", "Foam": "foam cell (lipid core)",
               "SMC": "smooth muscle", "BEC": "brain endothelium (BBB)", "Peri": "pericyte",
               "Astro": "astrocyte (glymphatic)", "Micro": "microglia", "Neuron": "neuron",
               "Hep": "hepatocyte", "Kup": "Kupffer cell", "LSEC": "sinusoidal endothelium",
               "Stel": "stellate cell (fibrosis)"}

# Artery surface info — same six-face cube, relabelled for the vessel (lumen on top, media/adventitia deep).
_ARTERY_SURFACES = {
    "enamel": {"name": "Vessel wall · intima", "col": "#cfd8e3", "rough": 0.5, "metal": 0.0,
               "html": "<b>Native surface:</b> the subendothelial intima.<br><b>Information:</b> where retained "
                       "oxLDL accumulates and foam cells form — the lipid core.<br><b>Exchange:</b> lipoprotein "
                       "retention.<br><b>Transports:</b> monocytes cross in from the lumen."},
    "gum": {"name": "Vessel wall · media", "col": "#b9a6c9", "rough": 0.7, "metal": 0.0,
            "html": "<b>Native surface:</b> the muscular media.<br><b>Information:</b> smooth-muscle cells "
                    "migrate/proliferate to form the fibrous cap.<br><b>Exchange:</b> matrix deposition."},
    "oral": {"name": "Blood lumen", "col": "#c0392b", "rough": 0.08, "metal": 0.0,
             "html": "<b>Native surface:</b> the flowing blood lumen, lined by endothelium.<br><b>Information "
                     "— the systemic INLET:</b> this is where the oral OMV / inflammatory load ARRIVES from the "
                     "bloodstream and activates the endothelium.<br><b>Exchange:</b> O₂, lipids, monocytes, the "
                     "arriving inflammatory dose.<br><b>Transports:</b> monocyte adhesion + diapedesis."},
    "deep": {"name": "Adventitia", "col": "#4a5a52", "rough": 0.95, "metal": 0.0,
             "html": "<b>Native surface:</b> the outer adventitia.<br><b>Information:</b> vasa vasorum, "
                     "outer boundary.<br><b>Transports:</b> unit-cell closure in this model."},
    "biofilm": {"name": "Vessel continuation", "col": "#5a6b80", "rough": 0.9, "metal": 0.0,
                "html": "<b>Native surface:</b> the artery continuing along its length — this cube is one "
                        "representative volume.<br><b>Transports:</b> periodic boundary."},
}

# Brain surface info — the neurovascular unit (capillary/BBB on top, parenchyma deep).
_BRAIN_SURFACES = {
    "oral": {"name": "Capillary · blood–brain barrier", "col": "#c0556a", "rough": 0.1, "metal": 0.0,
             "html": "<b>Native surface:</b> the brain capillary, sealed by the <b>BBB</b> (endothelium + tight "
                     "junctions, pericytes, astrocyte end-feet).<br><b>Information — the systemic INLET:</b> where "
                     "the oral OMV / inflammatory load ARRIVES and disrupts the barrier.<br><b>Transports:</b> "
                     "tightly gated — inflammation raises permeability (the periodontitis → brain route)."},
    "enamel": {"name": "Neuropil · neurons", "col": "#b4a7d6", "rough": 0.6, "metal": 0.0,
               "html": "<b>Native surface:</b> neuronal parenchyma — neurons &amp; synapses.<br><b>Information:</b> "
                       "viability falls as amyloid-β accumulates and microglia activate."},
    "gum": {"name": "Astrocyte domain · glymphatic", "col": "#7fc6ac", "rough": 0.6, "metal": 0.0,
            "html": "<b>Native surface:</b> astrocyte territory.<br><b>Information:</b> AQP4 end-feet drive the "
                    "<b>glymphatic</b> clearance of amyloid — <b>suppressed by inflammation</b>."},
    "deep": {"name": "Deep parenchyma", "col": "#4a5560", "rough": 0.95, "metal": 0.0,
             "html": "<b>Native surface:</b> deep tissue, farthest from the vessel — slowest clearance, where "
                     "amyloid pools."},
    "biofilm": {"name": "Brain continuation", "col": "#5a6b80", "rough": 0.9, "metal": 0.0,
                "html": "<b>Native surface:</b> the brain continuing — a representative volume; periodic boundary."},
}

# Liver surface info — a hepatic lobule (the sinusoid blood inlet on top, pericentral zone deep).
_LIVER_SURFACES = {
    "oral": {"name": "Sinusoid · blood inlet", "col": "#b5485a", "rough": 0.1, "metal": 0.0,
             "html": "<b>Native surface:</b> the fenestrated sinusoid, lined by LSEC.<br><b>Information — the "
                     "systemic INLET:</b> the liver receives FIRST-PASS portal blood, so translocating oral/gut "
                     "bacteria &amp; OMVs arrive here and inflame the lobule (the periodontitis → NAFLD route)."},
    "enamel": {"name": "Hepatocyte plate", "col": "#e2c27a", "rough": 0.6, "metal": 0.0,
               "html": "<b>Native surface:</b> the hepatocyte parenchyma — detox, synthesis, metabolism.<br>"
                       "<b>Information:</b> function falls as lipid (steatosis) accumulates (lipotoxicity)."},
    "gum": {"name": "Space of Disse · stellate", "col": "#b9a6c9", "rough": 0.6, "metal": 0.0,
            "html": "<b>Native surface:</b> the perisinusoidal space of Disse.<br><b>Information:</b> stellate "
                    "cells here ACTIVATE under inflammation and deposit fibrotic collagen → <b>fibrosis</b>."},
    "deep": {"name": "Pericentral zone (low O₂)", "col": "#4a5560", "rough": 0.95, "metal": 0.0,
             "html": "<b>Native surface:</b> the pericentral, lowest-O₂ end of the portal→central zonation."},
    "biofilm": {"name": "Lobule continuation", "col": "#5a6b80", "rough": 0.9, "metal": 0.0,
                "html": "<b>Native surface:</b> the lobule continuing — a representative volume; periodic boundary."},
}


def _present_roles(eng) -> dict:
    roles = ai.learn_roles()
    present = {eng.species[n] for n in range(len(eng.species))
               if eng.grid.composition[..., n].sum() > 0 and eng.species[n] in roles}
    return {c: roles[c] for c in present}


def _ai_block(eng) -> dict:
    """GNN roles + realised coaggregations (with molecular binding). Empty network for non-microbial."""
    co = ai.active_coaggregations(eng, top=8)
    co2 = [{"a": e["a"], "b": e["b"], "weight": e["weight"], "affinity": e["affinity"],
            "binding": ai.predict_binding(e["a"], e["b"])} for e in co]
    return {"r2": round(ai.gcn_r2(), 3) if co else None, "roles": _present_roles(eng), "coagg": co2}


def _qa(digest, body=None) -> list[dict]:
    return [{"q": q, "a": query.answer(q, digest, body=body)} for q in query.DEFAULT_QUESTIONS]


def _fused_oral(dim: int = 22, steps: int = 120) -> CubeEpoch4:
    grid = CubeGrid(dim, dim, dim, species_ids=ORAL_TISSUE.cell_ids)
    eng = CubeEpoch4(grid, ORAL_TISSUE)
    eng.seed_uniform()
    for _ in range(steps):
        eng.step()
    return eng


def _export_cells(eng, field_keys, field_n: int = 18, n_points: int = 1700, seed: int = 0) -> dict:
    """export_sample's contract for an arbitrary (non-microbial) cell roster — one group per cell type."""
    rng = np.random.default_rng(seed)
    grid = eng.grid
    nx, ny, nz = grid.shape
    comp = grid.composition
    groups = list(eng.species)
    gidx = {g: i for i, g in enumerate(groups)}
    tot = {g: float(comp[..., n].sum()) for n, g in enumerate(groups)}
    present = {g: v for g, v in tot.items() if v > 0}
    wsum = sum(v ** 0.5 for v in present.values()) or 1.0
    points: list[list] = []
    for n, g in enumerate(groups):
        if g not in present:
            continue
        arr = comp[..., n].flatten()
        k = min(max(60, int(n_points * present[g] ** 0.5 / wsum)), int((arr > 0).sum()))
        if k <= 0:
            continue
        chosen = rng.choice(arr.size, size=k, replace=False, p=arr / arr.sum())
        for fi in chosen:
            i, j, kk = (int(x) for x in np.unravel_index(fi, grid.shape))
            points.append([round(i / (nx - 1), 3), round(j / (ny - 1), 3), round(kk / (nz - 1), 3),
                           gidx[g], g])
    fields = {key: _downsample_field(eng.field(key), field_n) for key in field_keys}
    return {"dims": [nx, ny, nz], "field_n": field_n, "fields": fields, "groups": groups,
            "group_color": {g: _CELL_COLOR.get(g, "#888") for g in groups},
            "group_label": {g: _CELL_LABEL.get(g, g) for g in groups},
            "points": points, "edges": [], "omv": [n for n, p in enumerate(points) if p[4] == "Foam"]}


def _coupled(target_name, target_tissue, field_keys, info_fmt, frames: int = 6, steps: int = 160,
             n_points: int = 1500, dim: int = 24):
    """Run oral→target coupled, capturing the target cloud over time (the dose-response developing).
    Returns (final export, progression frames, body digest, target engine)."""
    bm = BodyMap()
    bm.place("oral", ORAL_TISSUE, dim=dim)
    bm.place(target_name, target_tissue, dim=dim)
    bm.link("oral", target_name, k=2.5)
    tgt = bm.cubes[target_name]
    every = max(1, steps // frames)
    caps = []
    rng = np.random.default_rng(1)
    for t in range(1, steps + 1):
        bm.step()
        if t % every == 0 or t == steps:
            ex = _export_cells(tgt, field_keys, n_points=n_points, seed=int(rng.integers(1e6)))
            caps.append({"t": round(t * 0.1, 1), "pts": ex["points"], "info": info_fmt(tgt.readout())})
    final = _export_cells(tgt, field_keys, n_points=n_points, seed=7)
    return final, caps, query.digest_bodymap(bm), tgt


_SIG_EX = {"ecm", "enzyme", "shear", "glymph", "oxLDL0", "mineral", "eps"}   # non-solute fields


def _sig_fields(eng):
    """The varying solute fields + the ECM-derived D_eff (= θ/λ²) for the σ calc. Tissue-agnostic."""
    from sim.cube.ecm import porosity
    cand = {k: np.asarray(v, float) for k, v in eng.grid._fields.items()}
    cand.update({k: np.asarray(v, float) for k, v in eng.extra.items()})
    flds = {k: v for k, v in cand.items() if k not in _SIG_EX and v.std() > 1e-9 and np.any(v > 0)}
    spec = eng.tissue.ecm
    m = eng.extra.get("ecm", np.full(eng.grid.shape, spec.base_density))
    return flds, porosity(m, spec.alpha) / (spec.tortuosity ** 2)


def _sigma_raw(eng):
    """Raw per-voxel σ = Σ_i D_eff,i·R·|∇c_i|²/c_i (not log-scaled), plus the fields + D_eff used."""
    from sim.cube import thermo
    flds, deff = _sig_fields(eng)
    sig = thermo.total_sigma(flds, {k: deff for k in flds}, calibrated=True) if flds else np.zeros(eng.grid.shape)
    return sig, flds, deff


def _sigma_field(eng):
    """Per-voxel σ, log-scaled for display — the heatmap field. See sim/cube/thermo.py."""
    return np.log1p(_sigma_raw(eng)[0])


def _sigma_meta(sig, flds, deff) -> dict:
    """Legend quantification: total dissipation, peak, and the species driving it (σ is RELATIVE)."""
    from sim.cube import thermo
    if not flds:
        return {"total": 0.0, "max": 0.0, "by_species": [], "units": "relative units, not calibrated to W/K"}
    by = sorted(((k, float(thermo.transport_sigma({k: flds[k]}, {k: deff}, calibrated=True).sum())) for k in flds),
                key=lambda kv: -kv[1])[:3]
    return {"total": float(sig.sum()), "max": float(sig.max()),
            "by_species": [[k, round(v, 2)] for k, v in by],
            "units": "relative units, not calibrated to W/K (per-species scales are literature placeholders)"}


def _downsample_symmetric(field, n):
    """_downsample_field with a symmetric ±A range so 0 maps to the centre — for diverging Δσ."""
    nx, ny, nz = field.shape
    sub = field[np.ix_(np.linspace(0, nx - 1, n).astype(int),
                       np.linspace(0, ny - 1, n).astype(int),
                       np.linspace(0, nz - 1, n).astype(int))]
    A = max(abs(float(sub.min())), abs(float(sub.max()))) or 1.0
    q = np.clip((sub + A) / (2 * A) * 255.0, 0, 255).astype(int)
    return {"min": round(-A, 3), "max": round(A, 3), "q": q.flatten().tolist(), "div": True}


def _run_plain(tissue, dim, steps):
    """A tissue run with the disease driver OFF (no oral coupling / influx) — the HEALTHY baseline."""
    grid = CubeGrid(dim, dim, dim, species_ids=tissue.cell_ids)
    eng = CubeEpoch4(grid, tissue)
    eng.seed_uniform()
    for _ in range(steps):
        eng.step()
    return eng


def _sigma_layers(eng_d, eng_h, field_n):
    """Bake diseased σ, healthy σ, and Δσ (= σ_diseased − σ_healthy, supply boundary cancels by
    subtraction → the lesion). Returns fields + grad_fields + per-view legend metas."""
    from sim.cube import thermo
    sd, fd, dd = _sigma_raw(eng_d)
    sh, fh, dh = _sigma_raw(eng_h)
    draw = sd - sh
    disp = np.sign(draw) * np.log1p(np.abs(draw))                       # sign-preserving log for display
    drv = []
    for k in set(fd) | set(fh):
        cd = float(thermo.transport_sigma({k: fd[k]}, {k: dd}, calibrated=True).sum()) if k in fd else 0.0
        ch = float(thermo.transport_sigma({k: fh[k]}, {k: dh}, calibrated=True).sum()) if k in fh else 0.0
        drv.append([k, round(cd - ch, 2)])
    drv = sorted(drv, key=lambda kv: -abs(kv[1]))[:3]
    return {
        "fields": {"sigma": _downsample_field(np.log1p(sd), field_n),
                   "sigma_h": _downsample_field(np.log1p(sh), field_n),
                   "sigma_delta": _downsample_symmetric(disp, field_n)},
        "grad_fields": [{"k": "sigma", "label": "entropy production σ — diseased (log)"},
                        {"k": "sigma_h", "label": "σ — healthy baseline (log)"},
                        {"k": "sigma_delta", "label": "Δσ — lesion (disease − health)"}],
        "sigma_meta": _sigma_meta(sd, fd, dd),
        "sigma_meta_h": _sigma_meta(sh, fh, dh),
        "sigma_delta_meta": {"added": float(np.clip(draw, 0, None).sum()),
                             "removed": float(np.clip(-draw, 0, None).sum()), "by_species": drv},
    }


def build_atlas(oral_dim: int = 40, art_dim: int = 24) -> dict:
    """Assemble the navigable atlas: oral + artery cubes (polished + ψ/AI/Q&A) and the body-map layout."""
    # ---- ORAL: the proven view (build_demo) + the new capabilities from a fused run ----
    oral = build_demo(dim=oral_dim, field_n=20, n_points=2200, max_edges=320, capture=30)
    oeng = _fused_oral()
    odig = query.state_digest(oeng, "oral")
    oral.update({
        "name": "oral", "tissue": "oral biofilm (gingival pocket)",
        "grad_fields": [{"k": "O2", "label": "oxygen (O₂)"}, {"k": "pH", "label": "acidity (pH)"},
                        {"k": "NO", "label": "nitric oxide (NO)"}, {"k": "heme", "label": "heme (blood iron)"}],
        "psi": odig["psi"], "ai": _ai_block(oeng), "qa": _qa(odig), "digest": odig,
        "demo_kind": "matruchotii", "cake_hot": [1, 2, 3, 7],
    })
    oral["fields"]["ecm"] = _downsample_field(oeng.field("ecm"), oral["field_n"])   # the EPS matrix slice
    oral["grad_fields"].append({"k": "ecm", "label": "ECM matrix (EPS)"})

    # ---- ARTERY: coupled to oral via the bloodstream; same presentation, new tissue ----
    a_final, a_caps, abody, a_eng = _coupled(
        "artery", ARTERY_TISSUE, ("O2", "oxLDL", "inflammation", "ecm"),
        lambda r: (f"endothelial activation {r['endothelial_activation']} · foam {r['foam_fraction']} · "
                   f"MMP {r['mmp_activity']} · rupture-index {r['rupture_index']}"),
        dim=art_dim)
    adig = abody["cubes"]["artery"]
    artery = dict(a_final)
    artery.update({
        "name": "artery", "tissue": "artery / atheroma",
        "grad_fields": [{"k": "O2", "label": "oxygen (O₂)"}, {"k": "oxLDL", "label": "oxidised LDL"},
                        {"k": "inflammation", "label": "inflammatory load"},
                        {"k": "ecm", "label": "ECM fibrous cap"}],
        "surfaces": _ARTERY_SURFACES,
        "psi": adig["psi"], "ai": _ai_block(a_eng), "qa": _qa(adig, body=abody), "digest": adig,
        "demo_kind": "progression", "demo": {"kind": "progression", "frames": a_caps},
        "cake_hot": [2, 3, 6, 7],
    })

    # ---- BRAIN: the other oral→systemic destination (periodontitis ↔ Alzheimer's) ----
    b_final, b_caps, bbody, b_eng = _coupled(
        "brain", BRAIN_TISSUE, ("O2", "inflammation", "amyloid", "glymph", "ecm"),
        lambda r: (f"BBB integrity {r['bbb_integrity']} · amyloid {r['amyloid_load']:.0f} · "
                   f"neuroinflammation {r['neuroinflammation']} · MMP {r['mmp_activity']}"),
        dim=art_dim)
    bdig = bbody["cubes"]["brain"]
    brain = dict(b_final)
    brain.update({
        "name": "brain", "tissue": "brain / neurovascular unit",
        "grad_fields": [{"k": "O2", "label": "oxygen (O₂)"}, {"k": "inflammation", "label": "inflammatory load"},
                        {"k": "amyloid", "label": "amyloid-β"}, {"k": "glymph", "label": "glymphatic clearance"},
                        {"k": "ecm", "label": "ECM gel (hyaluronan)"}],
        "surfaces": _BRAIN_SURFACES,
        "psi": bdig["psi"], "ai": _ai_block(b_eng), "qa": _qa(bdig, body=bbody), "digest": bdig,
        "demo_kind": "progression", "demo": {"kind": "progression", "frames": b_caps},
        "cake_hot": [2, 3, 6, 7],
    })

    # ---- LIVER: first-pass portal filter (periodontitis ↔ NAFLD/NASH → fibrosis) ----
    l_final, l_caps, lbody, l_eng = _coupled(
        "liver", LIVER_TISSUE, ("O2", "inflammation", "lipid", "ecm"),
        lambda r: (f"hepatocyte function {r['hepatocyte_function']} · steatosis {r['steatosis']} · "
                   f"fibrosis {r['fibrosis']} · hepatic inflammation {r['hepatic_inflammation']}"),
        dim=art_dim)
    ldig = lbody["cubes"]["liver"]
    liver = dict(l_final)
    liver.update({
        "name": "liver", "tissue": "liver / hepatic lobule",
        "grad_fields": [{"k": "O2", "label": "oxygen (O₂ zonation)"}, {"k": "inflammation", "label": "inflammatory load"},
                        {"k": "lipid", "label": "lipid (steatosis)"}, {"k": "ecm", "label": "ECM (fibrosis)"}],
        "surfaces": _LIVER_SURFACES,
        "psi": ldig["psi"], "ai": _ai_block(l_eng), "qa": _qa(ldig, body=lbody), "digest": ldig,
        "demo_kind": "progression", "demo": {"kind": "progression", "frames": l_caps},
        "cake_hot": [2, 3, 6, 7],
    })

    # ---- σ heatmap + HEALTHY baseline + Δσ lesion view (diseased / healthy / Δσ all selectable) ----
    art_h = _run_plain(ARTERY_TISSUE, art_dim, 160)
    brn_h = _run_plain(BRAIN_TISSUE, art_dim, 160)
    liv_h = _run_plain(LIVER_TISSUE, art_dim, 160)
    oral_h = _run_plain(ORAL_HEALTHY, 22, 120)
    for _tis, _ed, _eh in ((oral, oeng, oral_h), (artery, a_eng, art_h),
                           (brain, b_eng, brn_h), (liver, l_eng, liv_h)):
        _lay = _sigma_layers(_ed, _eh, _tis["field_n"])
        _tis["fields"].update(_lay["fields"])
        _tis["grad_fields"] += _lay["grad_fields"]
        _tis["sigma_meta"] = _lay["sigma_meta"]
        _tis["sigma_meta_h"] = _lay["sigma_meta_h"]
        _tis["sigma_delta_meta"] = _lay["sigma_delta_meta"]

    return {
        "tissues": {"oral": oral, "artery": artery, "brain": brain, "liver": liver},
        "order": ["oral", "artery", "brain", "liver"],
        "efflux": abody["links"][0]["efflux"] if abody["links"] else 0.0,
        # body-map node layout (fractions of the body SVG viewbox) — anatomically placed
        "bodymap": {
            "nodes": [
                {"id": "brain", "label": "Brain · BBB / glymphatic", "x": 0.5, "y": 0.085, "active": True},
                {"id": "oral", "label": "Mouth · oral biofilm", "x": 0.5, "y": 0.16, "active": True},
                {"id": "artery", "label": "Heart / artery · atheroma", "x": 0.55, "y": 0.26, "active": True},
                # liver = anatomical RIGHT upper abdomen → viewer's LEFT on a front-facing view, under the ribs
                {"id": "liver", "label": "Liver · hepatic lobule", "x": 0.38, "y": 0.35, "active": True},
            ],
            "links": [{"from": "oral", "to": "artery"}, {"from": "oral", "to": "brain"},
                      {"from": "oral", "to": "liver"}],
        },
    }

