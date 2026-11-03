"""Species catalogue — the 9-dimensional Oral Bacterial Node Schema, made executable.

Each Species wraps:
- Dim 1 Identity        : id, name, HOMD, gram
- Dim 2 Habitat         : preferred_depth (canonical from succession + site)
- Dim 3 Cell biology    : gram, oxygen_tolerance, motility
- Dim 4 Metabolism      : trophic, substrate, byproducts, nitrate_reducer, lactate_consumer
- Dim 5 Network role    : archetype, socransky, succession
- Dim 6 Disease         : commensal/pathobiont/keystone
- Dim 7 OMV biology     : vesiculation_rate, cargo
- Dim 8 Translocation   : detected_sites
- Dim 9 Coherence       : NO_production, H2S_production, pH_effect

The 4-field viability envelope (o2, ph, sugar, heme) is the Layer-3 chemistry constraint
that the agent uses every tick to decide whether it can live where it currently sits.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


Archetype = Literal[
    "Pioneer", "NitrateReducer", "Bridge", "Keystone",
    "Aggressive", "Cariogenic", "OSCC", "CrossKingdom", "Mineralizer",
]


@dataclass(frozen=True)
class Envelope:
    """4-field viability envelope from Layer 3 chemistry.

    For O2 and pH: tuple of (min, max, optimum).
    For sugar:  0=asaccharolytic, 1=low, 2=moderate, 3=high (preference, not constraint).
    For heme:   0=independent, 1=required, 2=obligate.
    """
    o2: tuple[float, float, float]
    ph: tuple[float, float, float]
    sugar: int
    heme: int


@dataclass(frozen=True)
class Species:
    id: str
    name: str
    archetype: Archetype
    gram: str                  # "+", "-", "fungal"
    o2_class: str              # "facultative", "aerotolerant", "anaerobe"
    succession: str            # "early", "middle", "late", "mid", "variable"
    #
    # VERIFIED 2026-08-04 against the coaggregation literature, after these labels were suspected of
    # disagreeing with COAGG_EDGES. Five disputed entries were checked, each with an adversarial
    # audit of its citations. The suspicion did not survive: NONE of them changed.
    #
    #   Vp  V. parvula      early  CONFIRMED (strong). A strict anaerobe AND a genuine early
    #                              coloniser — present at 4-12 h, coaggregating with S. sanguis and
    #                              Actinomyces, before the F. nucleatum bridge appears (PMID
    #                              25917902, 3178207, 21261622). "Anaerobe therefore late" is
    #                              exactly the wrong heuristic and is what raised the false alarm.
    #                              Nuance: the canonical scheme is FOUR-tier and Veillonella is
    #                              "early", not "initial" — it does not bind pellicle independently
    #                              (PMID 18805978, 20154130). sim.cube.dispersal already encodes
    #                              this correctly: only Pioneer archetypes bind pellicle, so Vp has
    #                              to find a partner like the literature says it does.
    #   Fa  F. alocis       late   CONFIRMED (moderate). No PubMed-indexed paper assigns it a
    #                              succession class, so this is an inference — but the audit found
    #                              the converging lines sound and both alternatives contradicted.
    #   Aa  A. actinomycetemcomitans  late  CONTESTED, and that is the verified state. The
    #                              Kolenbrander/NIDCR flow-cell lineage calls it a late coloniser;
    #                              other lineages call it early-colonising. A real documented split,
    #                              not an error. Left as "late" because that is the lineage this
    #                              catalog's coaggregation graph comes from.
    #   Lb  Lactobacillus   late   CHALLENGE REJECTED. The claim that the literature excludes "late"
    #                              turned out to be an artifact of the screening panel — Willcox
    #                              1993 tested lactobacilli only against streptococci and
    #                              actinomycetes, so the absence of late-coloniser partners reflects
    #                              what was never tested, not what was ruled out.
    #   Pm  P. micra        late   CHALLENGE REJECTED. "Orange complex therefore middle" imports a
    #                              temporal claim from Socransky 1998, which is cross-sectional
    #                              co-occurrence data and does not license it. Coaggregating with
    #                              F. nucleatum (PMID 10612731) is what TERTIARY colonisers do.
    #
    # The lesson worth keeping: the graph and the labels were never actually in conflict. What
    # conflicted was the graph and an intuition about oxygen tolerance.
    socransky: str             # "yellow"/"orange"/"red"/"blue"/"green"/"purple"/"—"
    envelope: Envelope
    # Dim 4 metabolism — used by agent.step() to set source/sink rates
    trophic: str               # "saccharolytic", "asaccharolytic", "proteolytic"
    primary_substrate: str     # "sugar", "lactate", "peptide", "amino_acid"
    byproducts: tuple[str, ...]  # e.g. ("lactate","H2O2") or ("propionate","H2","CO2")
    nitrate_reducer: bool
    lactate_consumer: bool
    # Dim 5 — preferred depth band (0=saliva surface, 1.0=junctional epithelium)
    preferred_depth: tuple[float, float]
    # Dim 7 — OMV biology
    vesiculation_rate: float   # relative; Pg is the reference (1.0)
    omv_cargo: tuple[str, ...] # named cargo classes
    # Dim 9 — coherence axis outputs (per-unit metabolic activity)
    NO_yield: float
    H2S_yield: float
    pH_effect: float           # per-tick local pH delta when growing (acid=negative)
    # Dim 6 — disease semantics
    role: str                  # "commensal","pathobiont","keystone","emerging","fungal"
    note: str = ""


# ---------------------------------------------------------------------------
# THE 18-SPECIES CATALOG
# values harmonized with oral-bacterial-node-schema.md and chemistry-phase-space.jsx
# ---------------------------------------------------------------------------
CATALOG: list[Species] = [
    # ------------- Archetype A — Pioneers -------------
    Species(
        id="Sa", name="S. sanguinis", archetype="Pioneer", gram="+",
        o2_class="facultative", succession="early", socransky="yellow",
        envelope=Envelope(o2=(0, 21, 8), ph=(5.5, 7.5, 6.8), sugar=3, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate", "H2O2"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.0, 0.2),
        vesiculation_rate=0.05, omv_cargo=("MV",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.005,
        role="commensal", note="H2O2 producer — chemically inhibits anaerobe ingress",
    ),
    Species(
        id="Sm", name="S. mitis/oralis", archetype="Pioneer", gram="+",
        o2_class="facultative", succession="early", socransky="yellow",
        envelope=Envelope(o2=(0, 15, 7), ph=(5.5, 7.8, 6.8), sugar=3, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate",),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.0, 0.25),
        vesiculation_rate=0.05, omv_cargo=("MV",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.003,
        role="commensal", note="most abundant pioneer; low-acid signature",
    ),
    Species(
        id="An", name="A. naeslundii", archetype="Pioneer", gram="+",
        o2_class="facultative", succession="early", socransky="blue",
        envelope=Envelope(o2=(0, 15, 5), ph=(5.0, 7.5, 6.5), sugar=2, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate", "acetate", "succinate"),
        nitrate_reducer=True, lactate_consumer=False,
        preferred_depth=(0.05, 0.3),
        vesiculation_rate=0.05, omv_cargo=("MV",),
        NO_yield=0.3, H2S_yield=0.0, pH_effect=-0.002,
        role="commensal", note="coaggregation scaffold + NO3 reducer",
    ),
    Species(
        id="Hp", name="H. parainfluenzae", archetype="Pioneer", gram="-",
        o2_class="facultative", succession="early", socransky="green",
        envelope=Envelope(o2=(2, 21, 12), ph=(6.5, 7.5, 7.0), sugar=1, heme=1),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("acetate", "CO2"),
        nitrate_reducer=True, lactate_consumer=False,
        preferred_depth=(0.0, 0.15),
        vesiculation_rate=0.08, omv_cargo=("LPS",),
        NO_yield=0.4, H2S_yield=0.0, pH_effect=-0.001,
        role="commensal", note="loss is OVN-axis warning signal",
    ),
    # ------------- Archetype B — Nitrate reducers -------------
    Species(
        id="Rm", name="R. mucilaginosa", archetype="NitrateReducer", gram="+",
        o2_class="facultative", succession="early", socransky="—",
        envelope=Envelope(o2=(5, 21, 15), ph=(6.0, 7.5, 6.8), sugar=1, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("acetate", "CO2"),
        nitrate_reducer=True, lactate_consumer=False,
        preferred_depth=(0.0, 0.1),
        vesiculation_rate=0.05, omv_cargo=("MV",),
        NO_yield=1.0, H2S_yield=0.0, pH_effect=0.0,
        role="commensal", note="PRIMARY NO precursor route — endothelial coupling proxy",
    ),
    Species(
        id="Vp", name="V. parvula", archetype="NitrateReducer", gram="-",
        o2_class="anaerobe", succession="early", socransky="purple",
        envelope=Envelope(o2=(0, 0.5, 0.1), ph=(6.0, 7.5, 6.8), sugar=0, heme=0),
        trophic="asaccharolytic", primary_substrate="lactate",
        byproducts=("propionate", "acetate", "CO2", "H2"),
        nitrate_reducer=True, lactate_consumer=True,
        preferred_depth=(0.2, 0.4),
        vesiculation_rate=0.15, omv_cargo=("LPS",),
        NO_yield=0.5, H2S_yield=0.0, pH_effect=0.002,
        role="commensal", note="cross-feeding hub — cascade victim of anti-strep moves",
    ),
    # ------------- Archetype C — Bridges -------------
    Species(
        id="Fn", name="F. nucleatum", archetype="Bridge", gram="-",
        o2_class="aerotolerant", succession="middle", socransky="orange",
        envelope=Envelope(o2=(0, 5, 1), ph=(6.5, 8.0, 7.2), sugar=1, heme=0),
        trophic="asaccharolytic", primary_substrate="amino_acid",
        byproducts=("butyrate", "propionate", "H2S", "NH3"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.3, 0.7),
        vesiculation_rate=0.5, omv_cargo=("FadA", "Fap2", "LPS"),
        NO_yield=0.0, H2S_yield=0.4, pH_effect=0.003,
        role="pathobiont", note="THE bridge — aerotolerance enables wide depth band",
    ),
    Species(
        id="Pi", name="P. intermedia", archetype="Bridge", gram="-",
        o2_class="anaerobe", succession="middle", socransky="orange",
        envelope=Envelope(o2=(0, 2, 0.3), ph=(6.5, 8.0, 7.2), sugar=1, heme=1),
        trophic="proteolytic", primary_substrate="peptide",
        byproducts=("propionate", "NH3", "H2S"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.4, 0.75),
        vesiculation_rate=0.3, omv_cargo=("LPS", "proteases"),
        NO_yield=0.0, H2S_yield=0.3, pH_effect=0.004,
        role="pathobiont", note="estradiol-responsive; bridges Fn -> Pg",
    ),
    Species(
        id="Ec", name="E. corrodens", archetype="Bridge", gram="-",
        o2_class="facultative", succession="middle", socransky="green",
        envelope=Envelope(o2=(2, 21, 10), ph=(6.5, 7.8, 7.0), sugar=0, heme=1),
        trophic="asaccharolytic", primary_substrate="peptide",
        byproducts=("acetate",),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.25, 0.5),
        vesiculation_rate=0.15, omv_cargo=("LPS",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=0.001,
        role="pathobiont", note="HACEK endocarditis pathway",
    ),
    # ------------- Archetype D — Red complex keystones -------------
    Species(
        id="Pg", name="P. gingivalis", archetype="Keystone", gram="-",
        o2_class="anaerobe", succession="late", socransky="red",
        envelope=Envelope(o2=(0, 2.0, 0), ph=(7.0, 8.5, 7.8), sugar=0, heme=2),
        trophic="proteolytic", primary_substrate="peptide",
        byproducts=("butyrate", "propionate", "NH3", "H2S"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.7, 1.0),
        vesiculation_rate=1.0,  # REFERENCE — Pg is the OMV reference species
        omv_cargo=("gingipains_Kgp", "gingipains_Rgp", "LPS_atypical", "fimbriae"),
        NO_yield=-0.5, H2S_yield=0.8, pH_effect=0.008,
        role="keystone", note="OVN cascade ignition node — the match",
    ),
    Species(
        id="Td", name="T. denticola", archetype="Keystone", gram="-",
        o2_class="anaerobe", succession="late", socransky="red",
        envelope=Envelope(o2=(0, 1.5, 0), ph=(7.0, 8.0, 7.5), sugar=0, heme=1),
        trophic="proteolytic", primary_substrate="peptide",
        byproducts=("acetate", "H2S", "NH3"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.75, 1.0),
        vesiculation_rate=0.4, omv_cargo=("dentilisin", "MSP"),
        NO_yield=0.0, H2S_yield=0.6, pH_effect=0.005,
        role="keystone", note="motile spirochete — junctional epithelium entry",
    ),
    Species(
        id="Tf", name="T. forsythia", archetype="Keystone", gram="-",
        o2_class="anaerobe", succession="late", socransky="red",
        envelope=Envelope(o2=(0, 1.5, 0), ph=(7.0, 8.0, 7.5), sugar=0, heme=1),
        trophic="proteolytic", primary_substrate="peptide",
        byproducts=("acetate", "propionate", "NH3"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.75, 1.0),
        vesiculation_rate=0.35, omv_cargo=("BspA", "KLIKK"),
        NO_yield=0.0, H2S_yield=0.3, pH_effect=0.004,
        role="keystone", note="NAM auxotroph — forced syntroph (hidden edge)",
    ),
    # ------------- Archetype E — Aggressive -------------
    Species(
        id="Aa", name="A. actinomycetemcomitans", archetype="Aggressive", gram="-",
        o2_class="facultative", succession="late", socransky="—",
        envelope=Envelope(o2=(0, 21, 5), ph=(6.5, 7.8, 7.0), sugar=2, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("acetate", "lactate"),
        nitrate_reducer=False, lactate_consumer=True,
        preferred_depth=(0.5, 0.9),
        vesiculation_rate=0.5, omv_cargo=("LtxA", "Cdt"),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.001,
        role="pathobiont", note="OMV-as-toxin-delivery vector model",
    ),
    # ------------- Archetype F — Cariogenic -------------
    Species(
        id="Smu", name="S. mutans", archetype="Cariogenic", gram="+",
        o2_class="facultative", succession="mid", socransky="—",
        envelope=Envelope(o2=(0, 21, 8), ph=(4.0, 7.0, 5.5), sugar=3, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate",),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.0, 0.2),
        vesiculation_rate=0.08, omv_cargo=("Gtfs",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.015,
        role="pathobiont", note="ACIDURIC — Stephan curve driver",
    ),
    Species(
        id="Lb", name="Lactobacillus spp.", archetype="Cariogenic", gram="+",
        o2_class="facultative", succession="late", socransky="—",
        envelope=Envelope(o2=(0, 21, 8), ph=(3.5, 6.0, 4.5), sugar=3, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate",),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.0, 0.2),
        vesiculation_rate=0.05, omv_cargo=("MV",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.020,
        role="pathobiont", note="most aciduric — deep caries dominator",
    ),
    # ------------- Archetype G — OSCC-enriched -------------
    Species(
        id="Fa", name="F. alocis", archetype="OSCC", gram="+",
        o2_class="anaerobe", succession="late", socransky="—",
        envelope=Envelope(o2=(0, 0.5, 0.1), ph=(6.5, 8.0, 7.2), sugar=1, heme=0),
        trophic="proteolytic", primary_substrate="peptide",
        byproducts=("butyrate", "NH3"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.6, 0.95),
        vesiculation_rate=0.3, omv_cargo=("MV", "proteases"),
        NO_yield=0.0, H2S_yield=0.2, pH_effect=0.003,
        role="emerging", note="emerging pathobiont; OSCC tumor enrichment",
    ),
    Species(
        id="Pm", name="P. micra", archetype="OSCC", gram="+",
        o2_class="anaerobe", succession="late", socransky="orange",
        envelope=Envelope(o2=(0, 0.5, 0.1), ph=(6.5, 8.0, 7.2), sugar=1, heme=0),
        trophic="proteolytic", primary_substrate="peptide",
        byproducts=("acetate", "NH3"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.6, 0.95),
        vesiculation_rate=0.2, omv_cargo=("MV",),
        NO_yield=0.0, H2S_yield=0.1, pH_effect=0.002,
        role="pathobiont", note="OSCC + CRC co-enriched with Fn",
    ),
    # ------------- Archetype H — Cross-kingdom -------------
    Species(
        id="Ca", name="C. albicans", archetype="CrossKingdom", gram="fungal",
        o2_class="facultative", succession="variable", socransky="—",
        envelope=Envelope(o2=(0, 21, 10), ph=(4.0, 7.5, 6.0), sugar=3, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("ethanol", "CO2"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.0, 0.4),
        vesiculation_rate=0.1, omv_cargo=("fungal_EVs",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.003,
        role="fungal", note="cross-kingdom bridge — hyphal form is the hub form",
    ),
]


BY_ID: dict[str, Species] = {s.id: s for s in CATALOG}


# ---------------------------------------------------------------------------
# COAGGREGATION + CROSSFEED — ported from build_network.py
# ---------------------------------------------------------------------------
COAGG_EDGES: list[tuple[str, str]] = [
    ("Sa","Sm"), ("Sa","An"), ("Sm","An"), ("Sa","Hp"), ("Sm","Hp"), ("An","Hp"),
    ("Sm","Rm"), ("Sa","Rm"), ("An","Rm"), ("Hp","Rm"),
    ("Sm","Vp"), ("Sa","Vp"), ("An","Vp"), ("Rm","Vp"),
    ("Fn","Sa"), ("Fn","Sm"), ("Fn","An"), ("Fn","Hp"), ("Fn","Vp"), ("Fn","Rm"),
    ("Fn","Pi"), ("Fn","Ec"),
    ("Fn","Pg"), ("Fn","Td"), ("Fn","Tf"),
    ("Fn","Fa"), ("Fn","Pm"),
    ("Fn","Ca"),
    ("Pi","Sm"), ("Pi","Pg"), ("Pi","Tf"),
    ("Ec","Sa"), ("Ec","An"), ("Ec","Aa"),
    ("Pg","Td"), ("Pg","Tf"), ("Td","Tf"),
    ("Aa","Sa"), ("Aa","An"),
    ("Smu","Sa"), ("Smu","Lb"), ("Smu","Ca"),
    ("Fa","Pg"), ("Pm","Pg"), ("Fa","Pm"),
]

# directed: producer -> consumer
CROSSFEED_EDGES: list[tuple[str, str, str]] = [
    ("Sa", "Vp", "lactate"), ("Sm", "Vp", "lactate"), ("Smu", "Vp", "lactate"),
    ("Sa", "Aa", "lactate"), ("Sm", "Aa", "lactate"),
    ("An", "Pg", "peptide_via_lysis"),
    ("Vp", "Pg", "propionate"), ("Vp", "Td", "propionate"), ("Vp", "Tf", "propionate"),
]

