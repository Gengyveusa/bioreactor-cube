"""Tissue plugin — what makes a cube 'oral' (or later 'artery', 'brain'). The engine is universal;
a Tissue supplies the cell roster (each cell's Information seed), the terrain fields, and the
tissue-specific ψ function. Oral ships first; other tissues plug in behind the same interface.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from sim.epoch4.information import Agent, Information
from sim.species import CATALOG, COAGG_EDGES


@dataclass
class Tissue:
    name: str
    seed: Callable[[], list[Agent]]            # the cell roster, seeded with Information
    terrain: Callable[[float], dict]           # depth (0 surface .. 1 deep) -> environment fields
    readout: Callable[[list[Agent]], dict]     # the tissue-specific ψ (system state)


# ---------------------------------------------------------------- oral biofilm (instance #1)
# Representative moderately-dysbiotic input abundances (relative).
_ABUNDANCE = {
    "Sa": 0.12, "Sm": 0.10, "An": 0.06, "Hp": 0.04, "Rm": 0.07, "Vp": 0.05, "Fn": 0.12, "Pi": 0.04,
    "Ec": 0.03, "Pg": 0.08, "Td": 0.06, "Tf": 0.06, "Aa": 0.02, "Smu": 0.03, "Lb": 0.02, "Fa": 0.02,
    "Pm": 0.02, "Ca": 0.02,
}


def _partners(sid: str) -> tuple[str, ...]:
    return tuple(b for a, b in COAGG_EDGES if a == sid) + tuple(a for a, b in COAGG_EDGES if b == sid)


def _genome(s) -> dict[str, float]:
    return {
        "o2_opt": s.envelope.o2[2],
        "o2_w": max(1.0, (s.envelope.o2[1] - s.envelope.o2[0]) / 2.0),
        "ph_opt": s.envelope.ph[2],
        "growth": 0.16 if s.archetype == "Cariogenic" else 0.12,
        "no_yield": max(0.0, s.NO_yield),
        "vesiculation": s.vesiculation_rate,
        "defense": 0.7 if s.archetype == "Keystone" else (0.5 if s.role == "pathobiont" else 0.35),
        "heme_need": float(s.envelope.heme >= 1),
        "red": 1.0 if s.socransky == "red" else 0.0,
    }


def _oral_seed() -> list[Agent]:
    out: list[Agent] = []
    for s in CATALOG:
        center = 0.5 * (s.preferred_depth[0] + s.preferred_depth[1])
        info = Information(
            genome=_genome(s),
            regulatory={"metabolism": 0.2, "omv": 0.0, "adhesion": 0.3},
            signaling={b: 0.0 for b in s.byproducts},
            partners=_partners(s.id),
        )
        out.append(Agent(sid=s.id, depth=center, info=info, biomass=_ABUNDANCE.get(s.id, 0.02)))
    return out


def _oral_terrain(depth: float) -> dict[str, float]:
    """O2 high at the oral surface, gone deep; heme rises with depth (bleeding gum)."""
    return {"O2": 8.0 * math.exp(-depth * 3.5), "pH": 6.8, "heme": depth}


def _oral_readout(pop: list[Agent]) -> dict:
    """ψ for oral: dysbiosis (red-complex fraction), NO output, OMV systemic flux."""
    alive = [a for a in pop if a.alive]
    total = sum(a.biomass for a in alive) or 1e-9
    red = sum(a.biomass for a in alive if a.info.genome["red"] > 0)
    no = sum(a.biomass * a.info.genome["no_yield"] * a.info.regulatory.get("metabolism", 0) for a in alive)
    omv = sum(a.biomass * a.info.regulatory.get("omv", 0) for a in alive)
    return {
        "dysbiosis": round(red / total, 3),
        "NO": round(no, 3),
        "omv_flux": round(omv, 3),
        "biomass": round(total, 3),
        "n_alive": len(alive),
    }


ORAL = Tissue(name="oral biofilm", seed=_oral_seed, terrain=_oral_terrain, readout=_oral_readout)

