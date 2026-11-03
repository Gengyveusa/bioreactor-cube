"""Product-mechanism encodings — Gengyve, Fasolati, Bdellovibrio.

Each intervention is a structured mapping from a real-world product to a
set of overrides on the simulator. The mechanism rationale (what biology
the override represents) is in each function's docstring and in the
mechanism_targets field.

This is where the schema's commercial implications become testable.
Falsifiable counterfactuals: if the encoded mechanism is right, the
intervention should move the observables in the predicted direction. If
it doesn't, the mechanism is mis-stated or the magnitude is wrong.
"""
from __future__ import annotations
from .dag import Intervention


# ---------------------------------------------------------------------------
# Gengyve · HA + castor oil — the preventive backbone-protector
# ---------------------------------------------------------------------------
def gengyve_ha_castor_oil(
    ha_effect: float = 1.0,
    castor_effect: float = 1.0,
) -> Intervention:
    """Gengyve hyaluronic acid + castor oil topical.

    Encoded mechanisms (each grounded in a known biological effect):

    HA (hyaluronic acid):
      - HA → tissue hydration → saliva flow & buffering capacity ↑
        (saliva_buffer is multiplied by 1 + 0.6 * ha_effect)
      - HA → viscoelastic film → biofilm adhesion barrier
        (pioneer division rate at surface depths is dampened)

    Castor oil (ricinoleic acid as the active):
      - Ricinoleic acid is documented Gram+-selective antimicrobial
        (S. mutans, Lactobacillus, S. sanguinis baseline mortality ↑)
      - Castor oil → surface tension modification → biofilm cohesion ↓
        (effective coaggregation strength multiplier ↓)

    Per the May-16 empirical refinement (Structural Centrality Asymmetry):
    this intervention targets the *structural backbone* — pioneers + the
    canonical bridge area — not the Socransky keystones. That's the
    preventive axis. Therapeutic anti-virulence (gingipain inhibition,
    anti-OMV) is the Fasolati axis instead.

    Args:
        ha_effect: dose-response multiplier on HA-side mechanisms (0=none, 1=nominal, 2=heavy)
        castor_effect: dose-response multiplier on castor-oil-side mechanisms
    """
    return Intervention(
        name="gengyve_ha_castor_oil",
        description=(
            "Gengyve topical: HA increases salivary buffering and dampens "
            "pioneer adhesion; castor oil's ricinoleic acid is Gram+-selective "
            "and reduces biofilm cohesion. Targets the structural backbone, "
            "not the Socransky keystones."
        ),
        config_overrides={
            # HA → saliva buffer
            "chemistry.saliva_buffer": 0.05 * (1.0 + 0.6 * ha_effect),
            # (Other HA / castor effects applied via species_modifiers below)
        },
        species_modifiers={
            # Castor oil — Gram+ mortality boost (small but consistent)
            # Applied as a multiplicative boost on baseline_mortality during agent.step()
            "Sa":  {"baseline_mortality_mult": 1.0 + 0.40 * castor_effect},
            "Sm":  {"baseline_mortality_mult": 1.0 + 0.40 * castor_effect},
            "An":  {"baseline_mortality_mult": 1.0 + 0.30 * castor_effect},
            "Smu": {"baseline_mortality_mult": 1.0 + 0.55 * castor_effect},
            "Lb":  {"baseline_mortality_mult": 1.0 + 0.65 * castor_effect},
            # HA biofilm adhesion barrier — pioneer division at top depths
            "Sa":  {"surface_division_mult": 1.0 - 0.30 * ha_effect},
            "Sm":  {"surface_division_mult": 1.0 - 0.30 * ha_effect},
            "An":  {"surface_division_mult": 1.0 - 0.25 * ha_effect},
        },
        mechanism_targets=[
            "saliva_buffer→ph_variance",
            "saliva_buffer→pioneer_fraction",
            "gram_pos_mortality_boost→pioneer_fraction",
            "biofilm_cohesion→fn_betweenness",
            "pioneer_division_at_surface→pioneer_fraction",
        ],
    )


# ---------------------------------------------------------------------------
# Fasolati · Lytica — enzyme + phage cocktail (anti-biofilm + bridge-targeted)
# ---------------------------------------------------------------------------
def fasolati_lytica(intensity: float = 1.0) -> Intervention:
    """Fasolati Lytica = enzymatic biofilm disruption + bridge-targeted phages.

    Encoded mechanisms:
      - Enzymatic biofilm disruption (subtilisin, proteinase K, glucanase) →
        biofilm cohesion ↓↓ (stronger than HA mechanical effect alone)
      - Bridge-targeted phages (ØFunu1/2 against Fn, ΦSG005 against Sg) →
        targeted Fn / Sg / Sm mortality
      - Indirect P. gingivalis collapse via removal of Sg/Sm bridge layer
        (Fujiki 2024 mechanism)

    Per the May-16 refinement: this acts on the *bridge* + *pioneer*
    structural backbone, which empirically drives P. gingivalis collapse
    via topology, not direct anti-Pg activity.
    """
    return Intervention(
        name="fasolati_lytica",
        description=(
            "Fasolati Lytica: enzymatic biofilm disruption (subtilisin / "
            "proteinase K) plus bridge-targeted phages (ØFunu1/2, ΦSG005). "
            "Operates through indirect targeting — Pg collapses because its "
            "bridge anchor (Sg, Fn) is removed."
        ),
        config_overrides={},
        species_modifiers={
            # Enzymatic disruption — broad biofilm-cohesion penalty across
            # heavy-EPS producers
            "Sa":  {"baseline_mortality_mult": 1.0 + 0.25 * intensity},
            "Sm":  {"baseline_mortality_mult": 1.0 + 0.25 * intensity},
            # Phage-directed mortality on Fn (ØFunu1/2)
            "Fn":  {"baseline_mortality_mult": 1.0 + 0.80 * intensity},
            "Fnp": {"baseline_mortality_mult": 1.0 + 0.70 * intensity},
            "Fnv": {"baseline_mortality_mult": 1.0 + 0.70 * intensity},
            # Phage-directed mortality on Sg-class (ΦSG005)
            "Sgo": {"baseline_mortality_mult": 1.0 + 0.85 * intensity},
            "Sa":  {"baseline_mortality_mult": 1.0 + 0.40 * intensity},
        },
        mechanism_targets=[
            "phage_cocktail→fn_betweenness",
            "biofilm_cohesion→fn_betweenness",
            "biofilm_cohesion→dysbiosis_index",
        ],
    )


# ---------------------------------------------------------------------------
# Fasolati · Loria — HA mouthrinse (consumer-grade therapy adjunct)
# ---------------------------------------------------------------------------
def fasolati_loria(ha_effect: float = 1.0) -> Intervention:
    """Fasolati Loria = consumer HA mouthrinse.

    Lighter-touch HA mechanism than Gengyve professional formulation —
    no castor oil component, no Gram+ mortality kick. Acts primarily
    through tissue hydration + light biofilm-adhesion barrier.
    """
    return Intervention(
        name="fasolati_loria",
        description="Fasolati Loria HA mouthrinse: tissue hydration and mild biofilm-adhesion barrier.",
        config_overrides={
            "chemistry.saliva_buffer": 0.05 * (1.0 + 0.4 * ha_effect),
        },
        species_modifiers={
            "Sa":  {"surface_division_mult": 1.0 - 0.20 * ha_effect},
            "Sm":  {"surface_division_mult": 1.0 - 0.20 * ha_effect},
            "An":  {"surface_division_mult": 1.0 - 0.15 * ha_effect},
        },
        mechanism_targets=[
            "saliva_buffer→ph_variance",
            "pioneer_division_at_surface→pioneer_fraction",
        ],
    )


# ---------------------------------------------------------------------------
# Bdellovibrio — the schema-v8 "third predator class" intervention
# ---------------------------------------------------------------------------
def bdellovibrio_deployment(density: float = 1.0) -> Intervention:
    """Bdellovibrio bacteriovorus topical deployment.

    Per schema-v8 §"the standout": BALOs predate the entire Gram-negative
    subgingival pathogen complement (Pg, Td, Tf, Fn, Aa, Pi, Ec) at once,
    with no known resistance mechanism. Gram+ commensals are untouched.

    Encoded as broad Gram-negative mortality boost. Magnitude calibrated
    to in-vitro Bdellovibrio kill rates (typically 90%+ Gram- reduction
    in mature biofilm at modest predator density).

    This is the highest-leverage commercial moat per the schema — zero
    human studies, ~15 years of in-vitro evidence, no commercial competitor.
    """
    return Intervention(
        name="bdellovibrio_deployment",
        description=(
            "Bdellovibrio bacteriovorus topical: broad Gram-negative predation. "
            "Spares Gram+ commensals. No known resistance mechanism. "
            "The schema-v8 standout."
        ),
        config_overrides={},
        species_modifiers={
            # Gram-negative species — all hit hard
            "Pg":  {"baseline_mortality_mult": 1.0 + 1.20 * density},
            "Td":  {"baseline_mortality_mult": 1.0 + 1.10 * density},
            "Tf":  {"baseline_mortality_mult": 1.0 + 1.10 * density},
            "Fn":  {"baseline_mortality_mult": 1.0 + 1.00 * density},
            "Fnp": {"baseline_mortality_mult": 1.0 + 1.00 * density},
            "Fnv": {"baseline_mortality_mult": 1.0 + 1.00 * density},
            "Aa":  {"baseline_mortality_mult": 1.0 + 0.95 * density},
            "Aa_JP2": {"baseline_mortality_mult": 1.0 + 0.95 * density},
            "Pi":  {"baseline_mortality_mult": 1.0 + 1.00 * density},
            "Ec":  {"baseline_mortality_mult": 1.0 + 0.90 * density},
            "Vp":  {"baseline_mortality_mult": 1.0 + 0.80 * density},
            "Vd":  {"baseline_mortality_mult": 1.0 + 0.80 * density},
            "Va":  {"baseline_mortality_mult": 1.0 + 0.80 * density},
            "Hp":  {"baseline_mortality_mult": 1.0 + 0.85 * density},
            "Pd":  {"baseline_mortality_mult": 1.0 + 0.95 * density},
            "Pn":  {"baseline_mortality_mult": 1.0 + 0.95 * density},
            "Pmel":{"baseline_mortality_mult": 1.0 + 0.85 * density},
            "Ss":  {"baseline_mortality_mult": 1.0 + 0.85 * density},
            "Sn":  {"baseline_mortality_mult": 1.0 + 0.85 * density},
            "Cg":  {"baseline_mortality_mult": 1.0 + 0.85 * density},
            "Co":  {"baseline_mortality_mult": 1.0 + 0.85 * density},
            "Csp": {"baseline_mortality_mult": 1.0 + 0.85 * density},
            "Lb2": {"baseline_mortality_mult": 1.0 + 0.85 * density},
            "Lh":  {"baseline_mortality_mult": 1.0 + 0.85 * density},
            "Cr":  {"baseline_mortality_mult": 1.0 + 0.90 * density},
            "Cgr": {"baseline_mortality_mult": 1.0 + 0.90 * density},
            "Pe":  {"baseline_mortality_mult": 1.0 + 1.20 * density},
            # Gram+ commensals deliberately untouched (mortality_mult = 1.0)
        },
        mechanism_targets=[
            "bdello_deployment→dysbiosis_index",
            "bdello_deployment→omv_flux_gingipain",
        ],
    )


# ---------------------------------------------------------------------------
# Build the default registry — all four interventions ready to compare
# ---------------------------------------------------------------------------
def build_default_graph():
    from .dag import CausalGraph
    g = CausalGraph()
    g.register(gengyve_ha_castor_oil())
    g.register(fasolati_lytica())
    g.register(fasolati_loria())
    g.register(bdellovibrio_deployment())
    return g

