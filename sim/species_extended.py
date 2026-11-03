"""Extended species catalog — ~40 additional HOMD-grade entries.

Brings the simulator from the 18-species canonical set to ~58 species
covering the full Mark Welch et al. "structured architecture" of supragingival
+ subgingival + tongue-dorsum + tonsillar biofilms.

Each archetype is now populated with enough members for the bridge-asymmetry
test to be statistically powered. The original 18 remain canonical; these
extend the niche space and unlock the Mann-Whitney U at proper n.

Sources for the relationships encoded here:
- Kolenbrander et al. — coaggregation cascade & succession
- Periasamy & Kolenbrander — Fn-as-bridge experiments
- Mark Welch et al. (2016) — "Biogeography of a human oral microbiome at the micron scale" (Hedgehog/Rotunda structures)
- Socransky complexes — the canonical color groupings
- HOMD v16.02 — taxonomic + cultivability anchors
"""
from __future__ import annotations

from .species import Species, Envelope


EXTENDED: list[Species] = [
    # ===================================================================
    # ARCHETYPE A — additional Pioneers (Streptococcus, Granulicatella, Gemella, Neisseria)
    # ===================================================================
    Species(
        id="Sgo", name="S. gordonii", archetype="Pioneer", gram="+",
        o2_class="facultative", succession="early", socransky="yellow",
        envelope=Envelope(o2=(0, 21, 8), ph=(5.5, 7.5, 6.8), sugar=3, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate", "H2O2"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.0, 0.2),
        vesiculation_rate=0.05, omv_cargo=("MV",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.004,
        role="commensal", note="model pioneer; CshA adhesin; corncob hub",
    ),
    Species(
        id="Sin", name="S. intermedius", archetype="Pioneer", gram="+",
        o2_class="facultative", succession="early", socransky="yellow",
        envelope=Envelope(o2=(0, 15, 7), ph=(5.5, 7.5, 6.8), sugar=3, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate",),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.05, 0.3),
        vesiculation_rate=0.06, omv_cargo=("MV",),
        NO_yield=0.0, H2S_yield=0.05, pH_effect=-0.003,
        role="commensal", note="anginosus group; abscess-forming when ectopic",
    ),
    Species(
        id="Sang", name="S. anginosus", archetype="Pioneer", gram="+",
        o2_class="facultative", succession="early", socransky="yellow",
        envelope=Envelope(o2=(0, 15, 5), ph=(5.5, 7.5, 6.8), sugar=2, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate",),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.05, 0.4),
        vesiculation_rate=0.06, omv_cargo=("MV",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.003,
        role="commensal", note="OSCC-associated in some cohorts",
    ),
    Species(
        id="Spa", name="S. parasanguinis", archetype="Pioneer", gram="+",
        o2_class="facultative", succession="early", socransky="yellow",
        envelope=Envelope(o2=(0, 21, 8), ph=(5.5, 7.5, 6.8), sugar=3, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate",),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.0, 0.2),
        vesiculation_rate=0.05, omv_cargo=("MV",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.004,
        role="commensal", note="long fimbriae; coaggregates broadly",
    ),
    Species(
        id="Ssa", name="S. salivarius", archetype="Pioneer", gram="+",
        o2_class="facultative", succession="early", socransky="—",
        envelope=Envelope(o2=(0, 21, 10), ph=(5.5, 7.8, 6.8), sugar=3, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate",),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.0, 0.1),
        vesiculation_rate=0.05, omv_cargo=("MV", "BLIS"),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.002,
        role="commensal", note="tongue dominant; K12 probiotic strain (BLIS)",
    ),
    Species(
        id="Gh", name="Granulicatella adiacens", archetype="Pioneer", gram="+",
        o2_class="facultative", succession="early", socransky="—",
        envelope=Envelope(o2=(0, 15, 5), ph=(6.0, 7.5, 6.8), sugar=2, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate",),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.0, 0.3),
        vesiculation_rate=0.05, omv_cargo=("MV",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.003,
        role="commensal", note="nutritionally variant strep; pyridoxal-dependent",
    ),
    Species(
        id="Gh2", name="Gemella haemolysans", archetype="Pioneer", gram="+",
        o2_class="facultative", succession="early", socransky="—",
        envelope=Envelope(o2=(0, 15, 8), ph=(6.0, 7.5, 6.8), sugar=2, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate", "acetate"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.0, 0.25),
        vesiculation_rate=0.04, omv_cargo=("MV",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.002,
        role="commensal", note="early colonizer; rare endocarditis",
    ),
    Species(
        id="Nm", name="Neisseria mucosa", archetype="Pioneer", gram="-",
        o2_class="facultative", succession="early", socransky="—",
        envelope=Envelope(o2=(5, 21, 15), ph=(6.5, 7.5, 7.0), sugar=1, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("acetate", "CO2"),
        nitrate_reducer=True, lactate_consumer=False,
        preferred_depth=(0.0, 0.1),
        vesiculation_rate=0.1, omv_cargo=("LPS",),
        NO_yield=0.6, H2S_yield=0.0, pH_effect=-0.001,
        role="commensal", note="oxygen-loving; NO precursor contributor",
    ),
    Species(
        id="Nf", name="Neisseria flavescens", archetype="Pioneer", gram="-",
        o2_class="facultative", succession="early", socransky="—",
        envelope=Envelope(o2=(8, 21, 18), ph=(6.5, 7.5, 7.0), sugar=1, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("acetate", "CO2"),
        nitrate_reducer=True, lactate_consumer=False,
        preferred_depth=(0.0, 0.08),
        vesiculation_rate=0.1, omv_cargo=("LPS",),
        NO_yield=0.5, H2S_yield=0.0, pH_effect=0.0,
        role="commensal", note="strict aerobe; saliva-surface dominant",
    ),
    # ===================================================================
    # ARCHETYPE A — additional Actinomyces (pioneer connectors)
    # ===================================================================
    Species(
        id="Ao", name="A. oris", archetype="Pioneer", gram="+",
        o2_class="facultative", succession="early", socransky="blue",
        envelope=Envelope(o2=(0, 15, 5), ph=(5.5, 7.5, 6.5), sugar=2, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate", "acetate", "succinate"),
        nitrate_reducer=True, lactate_consumer=False,
        preferred_depth=(0.05, 0.3),
        vesiculation_rate=0.05, omv_cargo=("MV",),
        NO_yield=0.35, H2S_yield=0.0, pH_effect=-0.002,
        role="commensal", note="type-2 fimbriae — THE coaggregation hub in Hedgehog structures",
    ),
    Species(
        id="Ag", name="A. gerencseriae", archetype="Pioneer", gram="+",
        o2_class="facultative", succession="early", socransky="blue",
        envelope=Envelope(o2=(0, 15, 5), ph=(5.5, 7.5, 6.5), sugar=2, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate", "acetate"),
        nitrate_reducer=True, lactate_consumer=False,
        preferred_depth=(0.05, 0.3),
        vesiculation_rate=0.05, omv_cargo=("MV",),
        NO_yield=0.3, H2S_yield=0.0, pH_effect=-0.002,
        role="commensal", note="formerly A. naeslundii genospecies 2",
    ),
    Species(
        id="Ais", name="A. israelii", archetype="Pioneer", gram="+",
        o2_class="anaerobe", succession="early", socransky="—",
        envelope=Envelope(o2=(0, 5, 1), ph=(6.0, 7.5, 6.8), sugar=2, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate", "acetate", "succinate"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.3, 0.6),
        vesiculation_rate=0.07, omv_cargo=("MV",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.001,
        role="emerging", note="cervicofacial actinomycosis when invasive",
    ),
    # ===================================================================
    # ARCHETYPE B — additional Nitrate reducers
    # ===================================================================
    Species(
        id="Rd", name="R. dentocariosa", archetype="NitrateReducer", gram="+",
        o2_class="facultative", succession="early", socransky="—",
        envelope=Envelope(o2=(5, 21, 12), ph=(6.0, 7.5, 6.8), sugar=2, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("acetate", "CO2"),
        nitrate_reducer=True, lactate_consumer=False,
        preferred_depth=(0.0, 0.15),
        vesiculation_rate=0.05, omv_cargo=("MV",),
        NO_yield=0.7, H2S_yield=0.0, pH_effect=0.0,
        role="commensal", note="secondary NO precursor pipeline",
    ),
    Species(
        id="Vd", name="V. dispar", archetype="NitrateReducer", gram="-",
        o2_class="anaerobe", succession="early", socransky="purple",
        envelope=Envelope(o2=(0, 0.5, 0.1), ph=(6.0, 7.5, 6.8), sugar=0, heme=0),
        trophic="asaccharolytic", primary_substrate="lactate",
        byproducts=("propionate", "acetate", "H2"),
        nitrate_reducer=True, lactate_consumer=True,
        preferred_depth=(0.2, 0.4),
        vesiculation_rate=0.12, omv_cargo=("LPS",),
        NO_yield=0.4, H2S_yield=0.0, pH_effect=0.002,
        role="commensal", note="streptococcal-lactate consumer; Vp sibling",
    ),
    Species(
        id="Va", name="V. atypica", archetype="NitrateReducer", gram="-",
        o2_class="anaerobe", succession="early", socransky="purple",
        envelope=Envelope(o2=(0, 0.5, 0.1), ph=(6.0, 7.5, 6.8), sugar=0, heme=0),
        trophic="asaccharolytic", primary_substrate="lactate",
        byproducts=("propionate", "acetate", "H2"),
        nitrate_reducer=True, lactate_consumer=True,
        preferred_depth=(0.2, 0.4),
        vesiculation_rate=0.12, omv_cargo=("LPS",),
        NO_yield=0.4, H2S_yield=0.0, pH_effect=0.002,
        role="commensal", note="tongue-dorsum nitrate pipeline",
    ),
    # ===================================================================
    # ARCHETYPE C — additional Bridges (Prevotella, Capnocytophaga, Selenomonas, Leptotrichia)
    # ===================================================================
    Species(
        id="Pd", name="P. denticola", archetype="Bridge", gram="-",
        o2_class="anaerobe", succession="middle", socransky="orange",
        envelope=Envelope(o2=(0, 2, 0.3), ph=(6.5, 8.0, 7.2), sugar=1, heme=1),
        trophic="proteolytic", primary_substrate="peptide",
        byproducts=("propionate", "NH3", "H2S"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.4, 0.75),
        vesiculation_rate=0.3, omv_cargo=("LPS", "proteases"),
        NO_yield=0.0, H2S_yield=0.3, pH_effect=0.003,
        role="pathobiont", note="pigmented; periodontitis-associated",
    ),
    Species(
        id="Pn", name="P. nigrescens", archetype="Bridge", gram="-",
        o2_class="anaerobe", succession="middle", socransky="orange",
        envelope=Envelope(o2=(0, 2, 0.3), ph=(6.5, 8.0, 7.2), sugar=1, heme=1),
        trophic="proteolytic", primary_substrate="peptide",
        byproducts=("propionate", "NH3", "H2S"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.4, 0.75),
        vesiculation_rate=0.3, omv_cargo=("LPS",),
        NO_yield=0.0, H2S_yield=0.3, pH_effect=0.003,
        role="pathobiont", note="frequently confused with P. intermedia",
    ),
    Species(
        id="Pmel", name="P. melaninogenica", archetype="Bridge", gram="-",
        o2_class="anaerobe", succession="middle", socransky="—",
        envelope=Envelope(o2=(0, 2, 0.5), ph=(6.5, 7.8, 7.2), sugar=1, heme=1),
        trophic="proteolytic", primary_substrate="peptide",
        byproducts=("propionate", "succinate", "NH3"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.2, 0.6),
        vesiculation_rate=0.25, omv_cargo=("LPS",),
        NO_yield=0.0, H2S_yield=0.2, pH_effect=0.002,
        role="commensal", note="tongue-dominant Prevotella; broad coaggregation",
    ),
    Species(
        id="Cg", name="C. gingivalis", archetype="Bridge", gram="-",
        o2_class="facultative", succession="middle", socransky="green",
        envelope=Envelope(o2=(1, 10, 5), ph=(6.5, 7.5, 7.0), sugar=1, heme=0),
        trophic="proteolytic", primary_substrate="peptide",
        byproducts=("acetate", "propionate"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.2, 0.5),
        vesiculation_rate=0.2, omv_cargo=("LPS", "gliding_proteins"),
        NO_yield=0.0, H2S_yield=0.1, pH_effect=0.002,
        role="commensal", note="Capnocytophaga — gliding motility; capnophile",
    ),
    Species(
        id="Co", name="C. ochracea", archetype="Bridge", gram="-",
        o2_class="facultative", succession="middle", socransky="green",
        envelope=Envelope(o2=(1, 10, 5), ph=(6.5, 7.5, 7.0), sugar=1, heme=0),
        trophic="proteolytic", primary_substrate="peptide",
        byproducts=("acetate", "propionate"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.2, 0.5),
        vesiculation_rate=0.2, omv_cargo=("LPS",),
        NO_yield=0.0, H2S_yield=0.1, pH_effect=0.002,
        role="commensal", note="Capnocytophaga ochracea; broad coaggregator",
    ),
    Species(
        id="Csp", name="C. sputigena", archetype="Bridge", gram="-",
        o2_class="facultative", succession="middle", socransky="green",
        envelope=Envelope(o2=(1, 10, 5), ph=(6.5, 7.5, 7.0), sugar=1, heme=0),
        trophic="proteolytic", primary_substrate="peptide",
        byproducts=("acetate",),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.2, 0.5),
        vesiculation_rate=0.2, omv_cargo=("LPS",),
        NO_yield=0.0, H2S_yield=0.1, pH_effect=0.002,
        role="commensal", note="Capnocytophaga sputigena",
    ),
    Species(
        id="Ss", name="Selenomonas sputigena", archetype="Bridge", gram="-",
        o2_class="anaerobe", succession="middle", socransky="orange",
        envelope=Envelope(o2=(0, 1, 0.2), ph=(6.5, 7.8, 7.0), sugar=1, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("propionate", "lactate", "acetate", "CO2"),
        nitrate_reducer=False, lactate_consumer=True,
        preferred_depth=(0.5, 0.85),
        vesiculation_rate=0.25, omv_cargo=("LPS",),
        NO_yield=0.0, H2S_yield=0.2, pH_effect=0.002,
        role="pathobiont", note="ECC-associated (Cho 2023); periodontitis bridge",
    ),
    Species(
        id="Sn", name="Selenomonas noxia", archetype="Bridge", gram="-",
        o2_class="anaerobe", succession="middle", socransky="orange",
        envelope=Envelope(o2=(0, 1, 0.2), ph=(6.5, 7.8, 7.0), sugar=1, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("propionate", "lactate", "acetate"),
        nitrate_reducer=False, lactate_consumer=True,
        preferred_depth=(0.5, 0.85),
        vesiculation_rate=0.25, omv_cargo=("LPS",),
        NO_yield=0.0, H2S_yield=0.2, pH_effect=0.002,
        role="pathobiont", note="periodontitis & gingivitis association",
    ),
    Species(
        id="Lb2", name="Leptotrichia buccalis", archetype="Bridge", gram="-",
        o2_class="aerotolerant", succession="middle", socransky="—",
        envelope=Envelope(o2=(0, 5, 1), ph=(6.5, 7.5, 7.2), sugar=2, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate", "acetate"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.2, 0.6),
        vesiculation_rate=0.2, omv_cargo=("LPS",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.001,
        role="commensal", note="filamentous; AEROTOLERANT — Fn-like bridge potential",
    ),
    Species(
        id="Lh", name="Leptotrichia hofstadii", archetype="Bridge", gram="-",
        o2_class="aerotolerant", succession="middle", socransky="—",
        envelope=Envelope(o2=(0, 5, 1), ph=(6.5, 7.5, 7.2), sugar=2, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate", "acetate"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.2, 0.6),
        vesiculation_rate=0.2, omv_cargo=("LPS",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.001,
        role="commensal", note="aerotolerant bridge candidate",
    ),
    # ===================================================================
    # ARCHETYPE C — additional Fusobacterium subspecies
    # ===================================================================
    Species(
        id="Fnp", name="F. nucleatum subsp. polymorphum", archetype="Bridge", gram="-",
        o2_class="aerotolerant", succession="middle", socransky="orange",
        envelope=Envelope(o2=(0, 5, 1), ph=(6.5, 8.0, 7.2), sugar=1, heme=0),
        trophic="asaccharolytic", primary_substrate="amino_acid",
        byproducts=("butyrate", "propionate", "H2S", "NH3"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.3, 0.7),
        vesiculation_rate=0.5, omv_cargo=("FadA", "Fap2", "LPS"),
        NO_yield=0.0, H2S_yield=0.4, pH_effect=0.003,
        role="pathobiont", note="Fn subsp polymorphum — supragingival-dominant subspecies",
    ),
    Species(
        id="Fnv", name="F. nucleatum subsp. vincentii", archetype="Bridge", gram="-",
        o2_class="aerotolerant", succession="middle", socransky="orange",
        envelope=Envelope(o2=(0, 5, 1), ph=(6.5, 8.0, 7.2), sugar=1, heme=0),
        trophic="asaccharolytic", primary_substrate="amino_acid",
        byproducts=("butyrate", "propionate", "H2S", "NH3"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.4, 0.8),
        vesiculation_rate=0.5, omv_cargo=("FadA", "Fap2"),
        NO_yield=0.0, H2S_yield=0.4, pH_effect=0.003,
        role="pathobiont", note="subgingival-dominant Fn subspecies",
    ),
    Species(
        id="Fp", name="F. periodonticum", archetype="Bridge", gram="-",
        o2_class="aerotolerant", succession="middle", socransky="orange",
        envelope=Envelope(o2=(0, 5, 1), ph=(6.5, 8.0, 7.2), sugar=1, heme=0),
        trophic="asaccharolytic", primary_substrate="amino_acid",
        byproducts=("butyrate", "propionate", "H2S"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.4, 0.8),
        vesiculation_rate=0.45, omv_cargo=("FadA-like", "LPS"),
        NO_yield=0.0, H2S_yield=0.3, pH_effect=0.003,
        role="pathobiont", note="Fn sibling species — independent bridge candidate",
    ),
    # ===================================================================
    # ARCHETYPE D — additional Keystones / red-orange complex
    # ===================================================================
    Species(
        id="Cr", name="Campylobacter rectus", archetype="Keystone", gram="-",
        o2_class="anaerobe", succession="late", socransky="orange",
        envelope=Envelope(o2=(0, 1, 0.2), ph=(6.5, 7.8, 7.2), sugar=0, heme=1),
        trophic="proteolytic", primary_substrate="peptide",
        byproducts=("acetate", "succinate", "H2S"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.5, 0.9),
        vesiculation_rate=0.3, omv_cargo=("LPS", "S-layer"),
        NO_yield=0.0, H2S_yield=0.4, pH_effect=0.003,
        role="pathobiont", note="orange-red bridge; motile; preterm-birth association",
    ),
    Species(
        id="Cgr", name="Campylobacter gracilis", archetype="Keystone", gram="-",
        o2_class="anaerobe", succession="late", socransky="orange",
        envelope=Envelope(o2=(0, 1, 0.2), ph=(6.5, 7.8, 7.2), sugar=0, heme=1),
        trophic="proteolytic", primary_substrate="peptide",
        byproducts=("acetate", "succinate"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.5, 0.9),
        vesiculation_rate=0.25, omv_cargo=("LPS",),
        NO_yield=0.0, H2S_yield=0.3, pH_effect=0.002,
        role="pathobiont", note="orange complex; gracilis is less virulent than rectus",
    ),
    Species(
        id="En", name="Eubacterium nodatum", archetype="Keystone", gram="+",
        o2_class="anaerobe", succession="late", socransky="orange",
        envelope=Envelope(o2=(0, 0.5, 0.1), ph=(6.5, 7.8, 7.0), sugar=0, heme=0),
        trophic="proteolytic", primary_substrate="peptide",
        byproducts=("butyrate", "NH3"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.6, 0.95),
        vesiculation_rate=0.2, omv_cargo=("MV",),
        NO_yield=0.0, H2S_yield=0.2, pH_effect=0.003,
        role="pathobiont", note="orange complex; periodontitis-associated",
    ),
    Species(
        id="Mt", name="Mogibacterium timidum", archetype="Keystone", gram="+",
        o2_class="anaerobe", succession="late", socransky="—",
        envelope=Envelope(o2=(0, 0.5, 0.1), ph=(6.5, 7.8, 7.0), sugar=0, heme=0),
        trophic="proteolytic", primary_substrate="peptide",
        byproducts=("butyrate", "H2S", "NH3"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.6, 0.95),
        vesiculation_rate=0.2, omv_cargo=("MV",),
        NO_yield=0.0, H2S_yield=0.4, pH_effect=0.003,
        role="emerging", note="emerging periodontitis-OSCC pathobiont",
    ),
    # ===================================================================
    # ARCHETYPE E — additional Aggressive specialists
    # ===================================================================
    Species(
        id="Aa_JP2", name="A. actinomycetemcomitans JP2", archetype="Aggressive", gram="-",
        o2_class="facultative", succession="late", socransky="—",
        envelope=Envelope(o2=(0, 21, 5), ph=(6.5, 7.8, 7.0), sugar=2, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("acetate", "lactate"),
        nitrate_reducer=False, lactate_consumer=True,
        preferred_depth=(0.5, 0.9),
        vesiculation_rate=0.7, omv_cargo=("LtxA_hyperexpressed", "Cdt"),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.001,
        role="keystone", note="JP2 clone — hyperleukotoxic; LAP",
    ),
    # ===================================================================
    # ARCHETYPE G — additional OSCC-enriched
    # ===================================================================
    Species(
        id="Pe", name="P. endodontalis", archetype="OSCC", gram="-",
        o2_class="anaerobe", succession="late", socransky="—",
        envelope=Envelope(o2=(0, 0.1, 0), ph=(7.0, 8.5, 7.6), sugar=0, heme=2),
        trophic="proteolytic", primary_substrate="peptide",
        byproducts=("butyrate", "propionate", "NH3", "H2S"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.75, 1.0),
        vesiculation_rate=0.8, omv_cargo=("LPS", "endopeptidases"),
        NO_yield=-0.3, H2S_yield=0.6, pH_effect=0.006,
        role="pathobiont", note="Pg sibling; endodontic infections",
    ),
    Species(
        id="Bg", name="Bacteroidales [G-2] sp. HMT 274", archetype="OSCC", gram="-",
        o2_class="anaerobe", succession="late", socransky="—",
        envelope=Envelope(o2=(0, 0.5, 0.1), ph=(6.5, 8.0, 7.2), sugar=0, heme=1),
        trophic="proteolytic", primary_substrate="peptide",
        byproducts=("acetate", "succinate", "propionate"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.6, 0.95),
        vesiculation_rate=0.2, omv_cargo=("MV",),
        NO_yield=0.0, H2S_yield=0.1, pH_effect=0.002,
        role="emerging", note="phylotype-only; HMT 274; OSCC tissue-enriched",
    ),
    # ===================================================================
    # ARCHETYPE F — additional Cariogenic
    # ===================================================================
    Species(
        id="Ssob", name="S. sobrinus", archetype="Cariogenic", gram="+",
        o2_class="facultative", succession="mid", socransky="—",
        envelope=Envelope(o2=(0, 21, 8), ph=(4.0, 7.0, 5.5), sugar=3, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate",),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.0, 0.2),
        vesiculation_rate=0.08, omv_cargo=("Gtfs",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.014,
        role="pathobiont", note="caries co-driver with S. mutans",
    ),
    Species(
        id="Ou", name="Olsenella uli", archetype="Cariogenic", gram="+",
        o2_class="anaerobe", succession="late", socransky="—",
        envelope=Envelope(o2=(0, 0.5, 0.1), ph=(4.0, 6.5, 5.0), sugar=3, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate", "acetate"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.2, 0.6),
        vesiculation_rate=0.05, omv_cargo=("MV",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.015,
        role="pathobiont", note="deep-caries acidogen; ECC enrichment",
    ),
    Species(
        id="Bd", name="Bifidobacterium dentium", archetype="Cariogenic", gram="+",
        o2_class="anaerobe", succession="mid", socransky="—",
        envelope=Envelope(o2=(0, 0.5, 0.1), ph=(4.5, 6.5, 5.5), sugar=3, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("lactate", "acetate"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.1, 0.4),
        vesiculation_rate=0.04, omv_cargo=("MV",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.013,
        role="pathobiont", note="caries-associated; ironic given Bifido probiotic reputation",
    ),
    # ===================================================================
    # ARCHETYPE H — additional Cross-kingdom / fungal
    # ===================================================================
    Species(
        id="Cd2", name="C. dubliniensis", archetype="CrossKingdom", gram="fungal",
        o2_class="facultative", succession="variable", socransky="—",
        envelope=Envelope(o2=(0, 21, 10), ph=(4.0, 7.5, 6.0), sugar=3, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("ethanol", "CO2"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.0, 0.4),
        vesiculation_rate=0.1, omv_cargo=("fungal_EVs",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.003,
        role="fungal", note="Candida dubliniensis — C. albicans sibling",
    ),
    Species(
        id="Cg2", name="C. glabrata", archetype="CrossKingdom", gram="fungal",
        o2_class="facultative", succession="variable", socransky="—",
        envelope=Envelope(o2=(0, 21, 10), ph=(3.5, 7.0, 5.5), sugar=3, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("ethanol", "CO2"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.0, 0.4),
        vesiculation_rate=0.08, omv_cargo=("fungal_EVs",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.004,
        role="fungal", note="azole-resistance reservoir",
    ),
    # ===================================================================
    # ARCHETYPE — Saccharibacteria / TM7 (epibiont — special case)
    # ===================================================================
    Species(
        id="Nl", name="Nanosynbacter lyticus TM7x", archetype="Bridge", gram="-",
        o2_class="aerotolerant", succession="middle", socransky="—",
        envelope=Envelope(o2=(0, 10, 2), ph=(6.0, 7.5, 6.8), sugar=0, heme=0),
        trophic="proteolytic", primary_substrate="peptide_from_host",
        byproducts=("acetate",),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.1, 0.5),
        vesiculation_rate=0.05, omv_cargo=("MV",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=0.0,
        role="emerging",
        note="OBLIGATE EPIBIONT of A. odontolyticus — single-edge dependency; Saccharibacteria",
    ),
    # ===================================================================
    # ARCHETYPE — Predator (Bdellovibrio — schema-v8 third predator class)
    # ===================================================================
    Species(
        id="Bb", name="Bdellovibrio bacteriovorus", archetype="Aggressive", gram="-",
        o2_class="facultative", succession="late", socransky="—",
        envelope=Envelope(o2=(0, 21, 10), ph=(6.0, 8.0, 7.0), sugar=0, heme=0),
        trophic="proteolytic", primary_substrate="peptide_from_prey",
        byproducts=("amino_acids",),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.0, 1.0),       # generalist depth
        vesiculation_rate=0.0, omv_cargo=(),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=0.0,
        role="emerging",
        note="OBLIGATE PREDATOR of Gram-negatives — schema-v8 third predator class",
    ),
    # ===================================================================
    # ARCHETYPE — Mineralizer (Corynebacterium — first member of the slot)
    # ===================================================================
    Species(
        id="Cma", name="C. matruchotii", archetype="Mineralizer", gram="+",
        o2_class="facultative", succession="early", socransky="—",
        envelope=Envelope(o2=(0, 21, 8), ph=(6.0, 7.5, 6.8), sugar=2, heme=0),
        trophic="saccharolytic", primary_substrate="sugar",
        byproducts=("acetate", "propionate", "CO2"),
        nitrate_reducer=False, lactate_consumer=False,
        preferred_depth=(0.0, 0.3),
        vesiculation_rate=0.05, omv_cargo=("MV",),
        NO_yield=0.0, H2S_yield=0.0, pH_effect=-0.001,
        role="commensal",
        note="hedgehog/corncob backbone filament; intracellular hydroxyapatite mineralizer (calculus nucleation) — Welch 2016",
    ),
]


# ===================================================================
# Extended coaggregation edges — Kolenbrander/Periasamy/Mark Welch-grounded
# ===================================================================
EXTENDED_COAGG_EDGES: list[tuple[str, str]] = [
    # --- New Streptococcus pioneers coaggregate with original pioneers ---
    ("Sgo", "Sa"), ("Sgo", "Sm"), ("Sgo", "An"), ("Sgo", "Hp"),
    ("Sin", "Sa"), ("Sin", "Sm"), ("Sin", "An"),
    ("Sang", "Sa"), ("Sang", "An"), ("Sang", "Fn"),
    ("Spa", "Sa"), ("Spa", "Sm"), ("Spa", "An"), ("Spa", "Hp"),
    ("Ssa", "Sm"), ("Ssa", "Sa"), ("Ssa", "Rm"),
    ("Gh", "Sm"), ("Gh", "Sa"), ("Gh", "An"),
    ("Gh2", "Sm"), ("Gh2", "Sa"),
    # Neisseria — saliva-surface
    ("Nm", "Sm"), ("Nm", "Sa"), ("Nm", "Hp"), ("Nm", "Rm"),
    ("Nf", "Sm"), ("Nf", "Hp"),
    # --- New Actinomyces ---
    ("Ao", "Sa"), ("Ao", "Sm"), ("Ao", "An"), ("Ao", "Sgo"), ("Ao", "Hp"),
    ("Ao", "Vp"), ("Ao", "Rm"), ("Ao", "Fn"),  # Ao is the corncob hub
    ("Ao", "Nl"),  # Ao is the host for Saccharibacteria TM7x
    ("Ag", "Sa"), ("Ag", "Sm"), ("Ag", "An"),
    ("Ais", "Sa"), ("Ais", "Fn"),
    # --- Rothia + Veillonella siblings ---
    ("Rd", "Sa"), ("Rd", "Sm"), ("Rd", "Rm"), ("Rd", "Vp"),
    ("Vd", "Sa"), ("Vd", "Sm"), ("Vd", "Rm"), ("Vd", "Fn"),
    ("Va", "Sm"), ("Va", "Rm"), ("Va", "Vp"),
    # --- New Prevotella bridges ---
    ("Pd", "Fn"), ("Pd", "Pg"), ("Pd", "Tf"), ("Pd", "Pi"),
    ("Pn", "Fn"), ("Pn", "Pg"), ("Pn", "Pi"),
    ("Pmel", "Fn"), ("Pmel", "Sm"), ("Pmel", "An"), ("Pmel", "Pi"),
    # --- Capnocytophaga bridges (green complex extensions) ---
    ("Cg", "Sa"), ("Cg", "An"), ("Cg", "Fn"), ("Cg", "Aa"),
    ("Co", "Sa"), ("Co", "An"), ("Co", "Fn"),
    ("Csp", "Sa"), ("Csp", "Fn"), ("Csp", "Aa"),
    # --- Selenomonas bridges ---
    ("Ss", "Fn"), ("Ss", "Smu"), ("Ss", "Pg"), ("Ss", "Cr"),
    ("Sn", "Fn"), ("Sn", "Pg"), ("Sn", "Cr"),
    # --- Leptotrichia bridges (aerotolerant — Fn-like) ---
    ("Lb2", "Sa"), ("Lb2", "Sm"), ("Lb2", "An"), ("Lb2", "Fn"), ("Lb2", "Vp"), ("Lb2", "Pg"),
    ("Lh", "Sa"), ("Lh", "Fn"), ("Lh", "Vp"),
    # --- F. nucleatum subspecies network ---
    ("Fnp", "Sa"), ("Fnp", "Sm"), ("Fnp", "An"), ("Fnp", "Pg"),
    ("Fnv", "Pg"), ("Fnv", "Td"), ("Fnv", "Tf"), ("Fnv", "Pi"),
    ("Fp",  "Sm"), ("Fp",  "Pg"), ("Fp",  "Pi"), ("Fp",  "Lb2"),
    # --- Campylobacter / Eubacterium / Mogibacterium (red-orange keystones) ---
    ("Cr", "Pg"), ("Cr", "Fn"), ("Cr", "Tf"), ("Cr", "Pi"),
    ("Cgr", "Pg"), ("Cgr", "Fn"),
    ("En", "Pg"), ("En", "Tf"), ("En", "Fn"),
    ("Mt", "Pg"), ("Mt", "Tf"), ("Mt", "Fn"),
    # --- Aa JP2 ---
    ("Aa_JP2", "Sa"), ("Aa_JP2", "An"), ("Aa_JP2", "Fn"),
    # --- Pg sibling: Pe ---
    ("Pe", "Pg"), ("Pe", "Td"), ("Pe", "Tf"), ("Pe", "Fn"),
    # --- HMT 274 ---
    ("Bg", "Fn"), ("Bg", "Pg"),
    # --- Cariogenic extensions ---
    ("Ssob", "Smu"), ("Ssob", "Sa"), ("Ssob", "Ca"),
    ("Ou", "Lb"), ("Ou", "Smu"),
    ("Bd", "Smu"), ("Bd", "Lb"),
    # --- Cross-kingdom ---
    ("Cd2", "Smu"), ("Cd2", "Fn"),
    ("Cg2", "Smu"), ("Cg2", "Lb"),
    # --- Saccharibacteria — single obligate edge ---
    # (Nl-Ao edge already added above)
    # --- Bdellovibrio — generalist predator. Modeled as "coaggregation"
    #     proxy here so the ablation test sees its connectivity. In the
    #     full simulator (Layer 4 predator dynamics) it gets its own edge type.
    ("Bb", "Fn"), ("Bb", "Pg"), ("Bb", "Pi"), ("Bb", "Aa"),
    # --- Corynebacterium matruchotii — corncob/hedgehog core (documented coaggregation) ---
    # Trimmed to direct-binding partners; Welch-2016 spatial-corona taxa
    # (Hp/Nm/Nf/Cg/Co/Csp/Rm/Lb2) are co-occurrence, not coaggregation, and are excluded.
    ("Cma", "Sa"), ("Cma", "Sm"), ("Cma", "Sgo"),  # corncob: streptococcal cocci on the filament
    ("Cma", "Fn"),                                  # F. nucleatum coaggregation bridge partner
    ("Cma", "An"), ("Cma", "Ao"),                   # Actinomyces co-filamentous framework
]

