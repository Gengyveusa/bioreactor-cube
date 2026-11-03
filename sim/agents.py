"""BacterialAgent — the unit of biology.

Each agent represents a small population (~10^4-10^5 cells) of one species
at one grid position. Per tick:

1. Read local chemistry (O2, pH, sugar, heme, lactate)
2. Compute viability — within Envelope on all 4 fields?
3. If non-viable for grace ticks -> die
4. If viable -> compute growth fitness from:
   - chemistry match to envelope optima
   - crossfeed neighbor support (lactate consumers eat streptococcal output)
   - heme availability for heme-requiring species
   - depth-band preference
5. If fitness > division_threshold -> divide (place daughter in neighbor cell)
6. Emit byproducts to local field (lactate, propionate, NH3, H2S, NO, ...)
7. Probability of OMV emission = vesiculation_rate * (1 + stress_multiplier * stress)
   OMV cargo aggregated by observables for the OVN-flux readout
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from mesa import Agent

from .species import Species, Envelope
from .chemistry.nitrogen import reduce_nitrate

if TYPE_CHECKING:
    from .model import OralBiofilmModel


def _envelope_fitness(value: float, lo: float, hi: float, opt: float) -> float:
    """Smooth envelope fitness — 1 at optimum, ~0.3 at edges, decays past edges.

    Softer than a sharp triangle so a species doesn't instant-die just outside its band.
    """
    if value <= opt:
        if opt <= lo:
            return 1.0
        x = (value - lo) / (opt - lo)
    else:
        if hi <= opt:
            return 1.0
        x = (hi - value) / (hi - opt)
    # Map x in [0,1] -> fitness; allow soft tail outside (x<0)
    if x >= 0:
        return float(0.3 + 0.7 * x)
    # Past envelope edge: rapid decay but not zero
    return float(max(0.0, 0.3 * (1.0 + 2.0 * x)))


@dataclass
class OMVEmission:
    species_id: str
    position: tuple
    cargo: tuple[str, ...]
    quantity: float


class BacterialAgent(Agent):
    """Mesa agent representing a local micro-population of one species."""

    def __init__(self, model: "OralBiofilmModel",
                 species: Species, position: tuple):
        super().__init__(model)
        self.species = species
        self.position = position
        self.age_ticks: int = 0
        self.off_spec_ticks: int = 0
        self.alive: bool = True
        # Population size scalar — soft proxy for biomass at this location
        self.biomass: float = 1.0

    # ---------------------------------------------------------------
    # viability + fitness
    # ---------------------------------------------------------------
    def local_chemistry(self) -> dict[str, float]:
        g = self.model.grid_obj
        return {f: g.get_field_at(f, self.position)
                for f in ("O2", "pH", "sugar", "heme", "lactate",
                          "propionate", "acetate", "NH3")}

    def viability(self, chem: dict[str, float]) -> float:
        """4-field viability product. Each factor in [0,1].

        Survival is determined by the floor (max of factors >= 0.1 -> can survive);
        growth is determined by the product. We expose the product as fitness;
        Agent.step() handles survival via the off-spec counter.
        """
        e: Envelope = self.species.envelope
        f_o2 = _envelope_fitness(chem["O2"], *e.o2)
        f_ph = _envelope_fitness(chem["pH"], *e.ph)
        # Sugar: high preference (>=2) cells still survive between meals at a baseline.
        # 0.5 floor so streptococci don't starve to death waiting for Stephan curve.
        if e.sugar >= 2:
            f_sug = 0.5 + 0.5 * min(1.0, chem["sugar"] / 0.3)
        else:
            # Low/no-sugar specialists: neutral, but a tiny penalty if sugar is high
            # (acidic environment shift selects against them)
            f_sug = 1.0
        # Heme: obligate (=2) species REQUIRE heme; required (=1) prefer; independent neutral.
        if e.heme == 2:
            f_heme = min(1.0, chem["heme"] / 0.3) if chem["heme"] > 0 else 0.05
        elif e.heme == 1:
            f_heme = 0.5 + 0.5 * min(1.0, chem["heme"] / 0.5)
        else:
            f_heme = 1.0
        return float(f_o2 * f_ph * f_sug * f_heme)

    def crossfeed_support(self, chem: dict[str, float]) -> float:
        """Bonus fitness from local metabolites the species can consume."""
        if self.species.lactate_consumer and chem["lactate"] > 0.01:
            return float(min(0.4, chem["lactate"] * 0.5))
        return 0.0

    def depth_band_match(self) -> float:
        d = self.model.grid_obj.depth_fraction(self.position)
        lo, hi = self.species.preferred_depth
        if lo <= d <= hi:
            return 1.0
        # Soft falloff
        dist = min(abs(d - lo), abs(d - hi))
        return max(0.0, 1.0 - 4.0 * dist)

    def fitness(self) -> tuple[float, dict[str, float]]:
        chem = self.local_chemistry()
        v = self.viability(chem)
        xf = self.crossfeed_support(chem)
        depth = self.depth_band_match()
        return v * (0.7 + 0.3 * depth) + xf, chem

    # ---------------------------------------------------------------
    # metabolic emission
    # ---------------------------------------------------------------
    def emit(self, growth: float) -> None:
        """Push byproducts and consume substrates from local field."""
        g = self.model.grid_obj
        sp = self.species
        # Consumption
        if "sugar" in sp.primary_substrate and growth > 0:
            g.add_to_field_at("sugar", self.position, -0.02 * growth * self.biomass)
        if sp.lactate_consumer and growth > 0:
            g.add_to_field_at("lactate", self.position, -0.03 * growth * self.biomass)
        if sp.envelope.heme >= 1 and growth > 0:
            g.add_to_field_at("heme", self.position, -0.01 * growth * self.biomass)
        # O2 consumed by aerobes / facultatives
        if sp.envelope.o2[2] > 1.0:
            g.add_to_field_at("O2", self.position, -0.01 * growth * self.biomass)

        # Byproducts
        emit_amt = 0.03 * growth * self.biomass
        for bp in sp.byproducts:
            if bp == "lactate":
                g.add_to_field_at("lactate", self.position, emit_amt)
            elif bp == "propionate":
                g.add_to_field_at("propionate", self.position, emit_amt)
            elif bp == "acetate":
                g.add_to_field_at("acetate", self.position, emit_amt)
            elif bp == "butyrate":
                g.add_to_field_at("butyrate", self.position, emit_amt)
            elif bp == "NH3":
                g.add_to_field_at("NH3", self.position, emit_amt)
            elif bp == "H2S":
                g.add_to_field_at("H2S", self.position, emit_amt * sp.H2S_yield)
            elif bp == "H2O2":
                # H2O2 is locally bactericidal — model as O2 boost (rough proxy)
                g.add_to_field_at("O2", self.position, 0.005 * emit_amt)

        # Nitrogen / NO axis — substrate-limited enterosalivary reduction.
        # Nitrate-reducers consume local NO3- -> NO2- (O2-gated) -> NO (pH-gated),
        # buffering pH upward. NO now requires dietary nitrate AND a live reducer.
        if sp.nitrate_reducer and growth > 0:
            flux = reduce_nitrate(
                g.get_field_at("nitrate", self.position),
                g.get_field_at("O2", self.position),
                g.get_field_at("pH", self.position),
                sp.NO_yield * growth * self.biomass,
            )
            g.add_to_field_at("nitrate", self.position, flux.d_nitrate)
            g.add_to_field_at("nitrite", self.position, flux.d_nitrite)
            g.add_to_field_at("NO", self.position, flux.d_no)
            g.add_to_field_at("pH", self.position, flux.d_ph)

        # pH effect — depends on metabolism
        g.add_to_field_at("pH", self.position, sp.pH_effect * growth * self.biomass)

    # ---------------------------------------------------------------
    # OMV vesiculation
    # ---------------------------------------------------------------
    def maybe_emit_omv(self, fitness: float) -> OMVEmission | None:
        cfg = self.model.cfg.agents
        # Stress = 1 - fitness; stressed cells vesiculate more (well-documented for Pg)
        stress = max(0.0, 1.0 - fitness)
        rate = self.species.vesiculation_rate * cfg.omv_base_rate
        rate *= (1.0 + cfg.omv_stress_multiplier * stress)
        if self.model.rng.random() < rate * self.biomass:
            return OMVEmission(
                species_id=self.species.id,
                position=self.position,
                cargo=self.species.omv_cargo,
                quantity=1.0,
            )
        return None

    # ---------------------------------------------------------------
    # Mesa step
    # ---------------------------------------------------------------
    def step(self) -> None:
        if not self.alive:
            return
        self.age_ticks += 1
        fitness, chem = self.fitness()
        cfg = self.model.cfg.agents

        if fitness < 0.15:
            self.off_spec_ticks += 1
            if self.off_spec_ticks > cfg.viability_grace_ticks:
                self.alive = False
                return
        else:
            self.off_spec_ticks = max(0, self.off_spec_ticks - 2)

        # Emit byproducts proportional to fitness (a growth proxy)
        self.emit(growth=fitness)

        # OMV emission
        omv = self.maybe_emit_omv(fitness)
        if omv is not None:
            self.model.omv_log.append((self.model.tick, omv))

        # Per-species modifiers from Layer-5 SCM interventions (if present)
        mods = getattr(self.model, "species_modifiers", {}).get(self.species.id, {})
        mortality_mult = float(mods.get("baseline_mortality_mult", 1.0))
        surface_div_mult = float(mods.get("surface_division_mult", 1.0))

        # Stochastic baseline mortality — even fit cells turn over (gives the
        # system flux at equilibrium instead of frozen plateaus).
        # SCM modifier scales the baseline rate (e.g., Gram+-selective castor oil).
        if self.model.rng.random() < 0.0015 * mortality_mult:
            self.alive = False
            return

        # Probabilistic division: rate proportional to how far fitness exceeds threshold.
        # No more hard threshold + deterministic stasis.
        if len(self.model.agents_alive) < cfg.max_agents:
            excess = max(0.0, fitness - cfg.division_threshold)
            p_div = 0.25 * excess   # tunable; calibrated later via ABC
            # SCM modifier: HA viscoelastic film dampens pioneer division at surface
            depth_frac = self.model.grid_obj.depth_fraction(self.position)
            if depth_frac < 0.25 and surface_div_mult != 1.0:
                p_div *= surface_div_mult
            if self.model.rng.random() < p_div:
                self.model.spawn_daughter(self)

