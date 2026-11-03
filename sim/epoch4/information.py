"""The Information substrate (Epoch-4) — the primary, first-class, per-agent quantity.

Per the ontology contract: Information is a NOUN; it is the only thing the five operators are about.
Four faces. ``genome`` is fixed (identity/capabilities); the other three move — that movement IS the
simulation. Energy (E) reads and transduces it, Terrain (T) gates it, Immunity (I) recognizes it, Time
(Δ) integrates it, and State (ψ) reads it out. Never confuse this substrate with the I-operator (immunity).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Information:
    """The four faces of an agent's information. genome is fixed; regulatory/signaling/partners move."""

    genome: dict[str, float]                      # identity & capabilities (fixed trait vector)
    regulatory: dict[str, float] = field(default_factory=dict)   # expression 0..1 — what is ON now
    signaling: dict[str, float] = field(default_factory=dict)    # molecules emitted / sensed
    partners: tuple[str, ...] = ()                # network position — who it is wired to


@dataclass
class Agent:
    """A cell (any kind) carrying its Information at a location in the terrain. Cell-type-agnostic."""

    sid: str                                      # species/cell-type id
    depth: float                                  # 0 = tissue surface … 1 = deep (its terrain coordinate)
    info: Information
    biomass: float = 0.02
    alive: bool = True

