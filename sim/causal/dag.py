"""Structural Causal Model — graph + intervention semantics.

A CausalGraph names the variables, the parent→child mechanisms between them,
and provides do() semantics: an Intervention is a set of overrides applied
*before* the simulator runs forward. Counterfactuals are paired runs of the
same calibrated state with and without the intervention.

This is the schema-aware version of "what does product X do" — every
mechanism in this file is grounded in a schema layer (the parents/children
trace back to the 9-dim node schema, the Layer-3 chemistry envelope, the
Layer-4 phage cocktail, or the Layer-8 exogenous bacteria deployment).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterable
import copy

from ..config import SimConfig


# ---------------------------------------------------------------------------
# Nodes — named variables in the SCM
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CausalNode:
    """One variable in the SCM.

    Names map back to schema layers:
      - chemistry.*       → Layer 3 chemistry fields and boundaries
      - agents.*          → Layer 1 species + Layer 2 succession
      - product.*         → Layer 8 exogenous interventions
      - host.*            → Layer 6 host immunity
      - omv_flux.*        → Layer 7 OMV cascade observables
      - outcome.*         → integrated readouts
    """
    name: str
    layer: str                      # schema layer this lives in (v1..v8)
    description: str = ""
    units: str = ""


# Canonical nodes — the ones we use across interventions
NODES = {
    # ----- Layer-3 chemistry boundaries (intervenable) -----
    "saliva_buffer":       CausalNode("saliva_buffer",      "v3", "Salivary buffering capacity (top boundary)", "rate/tick"),
    "saliva_O2":           CausalNode("saliva_O2",          "v3", "O2 inflow at saliva surface", "% atm"),
    "saliva_pH":           CausalNode("saliva_pH",          "v3", "Salivary pH set-point", "pH"),
    "saliva_flow":         CausalNode("saliva_flow",        "v3", "Salivary flow rate (scales surface clearance, O2 delivery, buffering)", "normalized"),
    "heme_bleed_rate":     CausalNode("heme_bleed_rate",    "v3", "Heme influx at gingival sulcus (bleed loop)", "rel/tick"),
    "stephan_amplitude":   CausalNode("stephan_amplitude",  "v3", "Magnitude of post-meal sugar pulse", "rel"),
    "saliva_nitrate":      CausalNode("saliva_nitrate",     "v3", "Enterosalivary nitrate delivery (diet-driven NO substrate)", "rel/tick"),
    # ----- Layer-1 agent properties (intervenable via per-species multipliers) -----
    "gram_pos_mortality_boost":  CausalNode("gram_pos_mortality_boost",  "v1", "Multiplier on baseline mortality for Gram+ species", "rel"),
    "gram_neg_mortality_boost":  CausalNode("gram_neg_mortality_boost",  "v1", "Multiplier on baseline mortality for Gram- species", "rel"),
    "biofilm_cohesion":          CausalNode("biofilm_cohesion",          "v1", "Effective coaggregation strength multiplier", "rel"),
    "pioneer_division_at_surface": CausalNode("pioneer_division_at_surface", "v1", "Pioneer division rate near saliva surface", "rel"),
    # ----- Layer-6 host immunity (patient-anchored) -----
    "host_inflammation":         CausalNode("host_inflammation",        "v6", "Systemic inflammatory tone (hs-CRP/IL-6/TNF-α/fibrinogen/ESR)", "normalized"),
    "host_tissue_breakdown":     CausalNode("host_tissue_breakdown",    "v6", "aMMP-8 active collagenase — local tissue destruction / pocket deepening", "normalized"),
    "host_antimicrobial_capacity": CausalNode("host_antimicrobial_capacity", "v6", "Neutrophil-mediated clearance capacity (subverted by keystone gingipains)", "normalized"),
    # ----- Layer-8 exogenous deployment -----
    "bdello_deployment":   CausalNode("bdello_deployment",  "v8", "Bdellovibrio inoculum density", "cells/tick"),
    "phage_cocktail":      CausalNode("phage_cocktail",     "v4", "Active phage cocktail composition", "set[str]"),
    # ----- Observables -----
    "dysbiosis_index":     CausalNode("dysbiosis_index",    "obs", "Red+orange biomass fraction", "fraction"),
    "omv_flux_gingipain":  CausalNode("omv_flux_gingipain", "obs", "Gingipain-bearing OMV flux at top boundary", "events/tick"),
    "fn_betweenness":      CausalNode("fn_betweenness",     "obs", "Realized betweenness of F. nucleatum", "0-1"),
    "pioneer_fraction":    CausalNode("pioneer_fraction",   "obs", "Pioneer biomass / total biomass", "fraction"),
    "ph_variance":         CausalNode("ph_variance",        "obs", "Variance of pH across depth (Stephan instability)", "pH^2"),
    "no_production":       CausalNode("no_production",      "obs", "Integrated NO production (endothelial coupling)", "rel"),
}


# ---------------------------------------------------------------------------
# Mechanisms — parent → child dependencies (declarative; for documentation)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Mechanism:
    """A parent→child causal relationship, optionally with a closed-form
    or qualitative description. For interventions we don't need analytic
    mechanism functions — the simulator IS the mechanism — but documenting
    them here keeps the SCM auditable.
    """
    parent: str
    child: str
    sign: str                        # "+", "-", "?"
    description: str


CANONICAL_MECHANISMS: list[Mechanism] = [
    # Chemistry → community
    Mechanism("saliva_buffer", "ph_variance", "-",
              "Stronger saliva buffering damps Stephan-curve excursions."),
    Mechanism("saliva_flow", "ph_variance", "-",
              "Higher flow clears surface acids and boosts effective buffering — damps Stephan excursions."),
    Mechanism("saliva_flow", "dysbiosis_index", "-",
              "Flow-driven clearance disfavors aciduric/anaerobic blooms; xerostomia (low flow) drives caries/dysbiosis."),
    Mechanism("saliva_buffer", "pioneer_fraction", "+",
              "Stable pH favors pioneer streptococci over aciduric specialists."),
    Mechanism("heme_bleed_rate", "dysbiosis_index", "+",
              "Heme availability is the rate-limiting nutrient for Pg; bleed loop drives red-complex bloom."),
    Mechanism("heme_bleed_rate", "omv_flux_gingipain", "+",
              "Pg vesiculation rises with heme; gingipain-bearing OMV flux rises."),
    Mechanism("saliva_O2", "fn_betweenness", "?",
              "O2 sets the depth at which aerotolerant Fn can bridge; non-monotone."),
    Mechanism("stephan_amplitude", "ph_variance", "+",
              "Larger sugar pulses produce deeper Stephan dips."),
    Mechanism("stephan_amplitude", "dysbiosis_index", "+",
              "Frequent acid stress selects for cariogenic attractor."),
    Mechanism("saliva_nitrate", "no_production", "+",
              "Dietary nitrate is the rate-limiting substrate for the enterosalivary NO pathway."),
    Mechanism("saliva_nitrate", "ph_variance", "-",
              "Nitrite/ammonia from nitrate reduction buffers pH upward — damps acid excursions (caries-protective)."),
    # Host immunity → community
    Mechanism("host_inflammation", "heme_bleed_rate", "+",
              "Gingival inflammation drives bleeding → heme influx at the sulcus (the OVN feed-forward)."),
    Mechanism("host_tissue_breakdown", "dysbiosis_index", "+",
              "aMMP-8 collagenolysis deepens the pocket / lowers basal O2 → expands the anaerobic niche → favors the red complex."),
    Mechanism("host_antimicrobial_capacity", "dysbiosis_index", "-",
              "Neutrophil clearance suppresses bacteria, but keystone gingipains subvert it — the red complex evades, weakening the net effect."),
    # Product → community
    Mechanism("gram_pos_mortality_boost", "pioneer_fraction", "-",
              "Castor-oil ricinoleic acid is Gram+-selective; reduces pioneer biomass."),
    Mechanism("biofilm_cohesion", "fn_betweenness", "+",
              "Reduced cohesion fragments the realized coaggregation graph."),
    Mechanism("pioneer_division_at_surface", "pioneer_fraction", "+",
              "HA viscoelastic film slows pioneer adhesion-and-division at surface."),
    Mechanism("bdello_deployment", "dysbiosis_index", "-",
              "Bdellovibrio predates Gram- species — Pg, Td, Tf, Fn — without touching commensals."),
    Mechanism("bdello_deployment", "omv_flux_gingipain", "-",
              "Removing Pg removes the dominant gingipain-OMV source."),
    Mechanism("phage_cocktail", "fn_betweenness", "-",
              "Bridge-targeted phages (ØFunu1/2, ΦSG005) collapse Fn realized betweenness."),
    # NO axis
    Mechanism("saliva_O2", "no_production", "+",
              "Higher salivary O2 supports nitrate-reducer activity; NO precursor pipeline."),
]


# ---------------------------------------------------------------------------
# Interventions — the do() operator
# ---------------------------------------------------------------------------
@dataclass
class Intervention:
    """A named intervention: a set of overrides to apply before the simulator runs.

    Two override types:
      - config_overrides: parameter changes in SimConfig (the simplest form;
        works for chemistry boundaries, agent-config tunables, etc.)
      - species_modifiers: per-species multiplicative modifiers applied at
        agent-construction time (e.g. mortality boost for Gram+ species)
    """
    name: str
    description: str
    # Direct config overrides — keyed by dotted-path like "chemistry.saliva_buffer"
    config_overrides: dict[str, float] = field(default_factory=dict)
    # Per-species modifier dict — applied by the runner via the model
    species_modifiers: dict[str, dict[str, float]] = field(default_factory=dict)
    # Documentation of which mechanisms in CANONICAL_MECHANISMS this acts on
    mechanism_targets: list[str] = field(default_factory=list)

    def apply_to_config(self, cfg: SimConfig) -> SimConfig:
        """Return a deep-copied config with overrides applied.

        Dotted paths like "chemistry.saliva_buffer" navigate the nested
        SimConfig dataclass. Original config is not mutated.
        """
        new_cfg = copy.deepcopy(cfg)
        for path, value in self.config_overrides.items():
            parts = path.split(".")
            target = new_cfg
            for p in parts[:-1]:
                target = getattr(target, p)
            setattr(target, parts[-1], value)
        return new_cfg


# ---------------------------------------------------------------------------
# CausalGraph — registry + intervention catalog
# ---------------------------------------------------------------------------
@dataclass
class CausalGraph:
    """The SCM registry. Holds nodes, declared mechanisms, and the set of
    interventions we know how to apply.
    """
    nodes: dict[str, CausalNode] = field(default_factory=lambda: dict(NODES))
    mechanisms: list[Mechanism] = field(default_factory=lambda: list(CANONICAL_MECHANISMS))
    interventions: dict[str, Intervention] = field(default_factory=dict)

    def register(self, intervention: Intervention) -> None:
        self.interventions[intervention.name] = intervention

    def parents_of(self, child: str) -> list[str]:
        return [m.parent for m in self.mechanisms if m.child == child]

    def children_of(self, parent: str) -> list[str]:
        return [m.child for m in self.mechanisms if m.parent == parent]

    def summary(self) -> str:
        lines = [
            f"CausalGraph · {len(self.nodes)} nodes · {len(self.mechanisms)} mechanisms · {len(self.interventions)} interventions",
            "",
            "Interventions:",
        ]
        for name, iv in self.interventions.items():
            lines.append(f"  - {name}: {iv.description}")
            for path, value in iv.config_overrides.items():
                lines.append(f"      cfg {path} := {value}")
            for species, mods in iv.species_modifiers.items():
                for prop, val in mods.items():
                    lines.append(f"      {species}.{prop} := {val}")
        return "\n".join(lines)

