"""Epoch-4 engine — information-primary, place-anywhere tissue simulation.

The generative chain ``Information → (E · T · I · Δ) → ψ`` (see EPOCH-4-ONTOLOGY.md). Information is the
primary per-agent substrate; the operators act on it; the Tissue plugin makes it 'oral' / 'artery' /
'brain'. This is step 2's core — universal from the start, oral as instance #1.
"""

from sim.epoch4.engine import run, step
from sim.epoch4.information import Agent, Information
from sim.epoch4.operators import express, gate, recognize
from sim.epoch4.tissue import ORAL, Tissue

__all__ = [
    "Information",
    "Agent",
    "gate",
    "express",
    "recognize",
    "step",
    "run",
    "Tissue",
    "ORAL",
]

