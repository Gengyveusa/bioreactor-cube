"""The Epoch-4 loop — Δ (step) and ψ (readout), running the generative chain on any Tissue.

    Information ──[ E ]──▶ ψ        gated by T, recognized by I, integrated over Δ

One tick = gate (T) → express (E) → recognize (I) → predation/decay → grow/die. The Information
substrate is the thing that moves; energy/terrain/immunity/time are the operators on it.
"""

from __future__ import annotations

from sim.epoch4.information import Agent
from sim.epoch4.operators import express, gate, recognize
from sim.epoch4.tissue import Tissue

DECAY = 0.18   # predation/clearance pressure on poorly-expressed, poorly-defended information


def step(pop: list[Agent], tissue: Tissue, dt: float = 0.1) -> None:
    """Δ — one time step of the operator loop, in order: gate → express → recognize → delete/grow."""
    for a in pop:
        if not a.alive:
            continue
        env = tissue.terrain(a.depth)
        expressibility = gate(a, env)                       # T
        express(a, env, expressibility, dt)                 # E (read/transduce + grow + emit)
        persist = recognize(a)                              # I (immunity)
        # predation = information deletion: hits the un-expressible and un-defended (E∩I)
        a.biomass -= dt * DECAY * (1.0 - expressibility) * (1.0 - persist) * a.biomass
        if a.biomass < 1e-4:
            a.alive = False


def run(tissue: Tissue, steps: int = 80, dt: float = 0.1, sample_every: int = 10):
    """Run the loop; return (final population, ψ trajectory). ψ is the tissue's own readout."""
    pop = tissue.seed()
    traj = [{"t": 0.0, **tissue.readout(pop)}]
    for t in range(1, steps + 1):
        step(pop, tissue, dt)
        if t % sample_every == 0 or t == steps:
            traj.append({"t": round(t * dt, 2), **tissue.readout(pop)})
    return pop, traj

