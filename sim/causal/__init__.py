"""Layer-5 — Structural Causal Model + intervention engine.

Operationalizes the do() operator over the Layer-1 mechanistic simulator
so we can answer counterfactual questions:

    "If Gengyve HA + castor oil is applied to this calibrated patient
     state, what is the Average Treatment Effect on dysbiosis index,
     OMV flux, F. nucleatum betweenness, and pioneer fraction?"

This is the layer the commercial conversation happens at. The simulator
is the substrate; the SCM is the lens that turns the substrate into
intervention testing.
"""
from .dag import CausalGraph, CausalNode, Intervention, Mechanism
from .gengyve_product import (
    gengyve_ha_castor_oil,
    fasolati_lytica,
    fasolati_loria,
    bdellovibrio_deployment,
)
from .counterfactual import (
    CounterfactualRun,
    ATEResult,
    run_paired,
    compute_ate,
)

__all__ = [
    "CausalGraph", "CausalNode", "Intervention", "Mechanism",
    "gengyve_ha_castor_oil", "fasolati_lytica", "fasolati_loria", "bdellovibrio_deployment",
    "CounterfactualRun", "ATEResult", "run_paired", "compute_ate",
]

