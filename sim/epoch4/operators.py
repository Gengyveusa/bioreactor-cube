"""The Epoch-4 operators — each a single computable verb acting on the Information substrate.

    T  gate(agent, terrain)            -> expressibility 0..1   (terrain decides what CAN express)
    E  express(agent, terrain, ex, dt) -> mutates info + biomass (energy reads/transduces; predation deletes)
    I  recognize(agent)                -> persistence 0..1       (immunity = information self-recognition)

Δ (step) and ψ (readout) live in engine.py. No operator is reassigned; none crosses into another's box.
"""

from __future__ import annotations

import math

from sim.epoch4.information import Agent


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def gate(agent: Agent, env: dict[str, float]) -> float:
    """T — terrain gates which information can be expressed here-and-now (the viability window)."""
    g = agent.info.genome
    o2 = math.exp(-((env.get("O2", 0.0) - g["o2_opt"]) / g["o2_w"]) ** 2)
    ph = math.exp(-((env.get("pH", 6.8) - g["ph_opt"]) / 1.0) ** 2)
    heme = (0.35 + 0.65 * env.get("heme", 0.0)) if g.get("heme_need", 0.0) else 1.0
    return _clamp01(o2 * ph * min(1.0, heme))


def express(agent: Agent, env: dict[str, float], expressibility: float, dt: float) -> None:
    """E — energy reads the gated information: metabolism relaxes toward what terrain permits, grows
    biomass, sheds OMVs, and emits signaling. (Predation — information *deletion* — is applied in step.)"""
    reg = agent.info.regulatory
    reg["metabolism"] = _clamp01(reg.get("metabolism", 0.2) + dt * 3.0 * (expressibility - reg.get("metabolism", 0.2)))
    m = reg["metabolism"]
    agent.biomass += dt * g_growth(agent) * m * agent.biomass * (1.0 - agent.biomass)   # logistic expression
    reg["omv"] = agent.info.genome.get("vesiculation", 0.0) * m                          # OMV program ∝ activity
    for k in agent.info.signaling:
        agent.info.signaling[k] += dt * m * 0.1                                          # emit byproducts


def g_growth(agent: Agent) -> float:
    return agent.info.genome.get("growth", 0.1)


def recognize(agent: Agent) -> float:
    """I — immunity: information acting on itself (self/non-self, memory, antiphage/host defense) →
    a persistence factor. Better-defended information survives predation; poorly-defended decays."""
    return 0.5 + 0.5 * agent.info.genome.get("defense", 0.4)

