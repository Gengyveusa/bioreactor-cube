"""Boundary conditions for the depth column.

Top boundary (saliva, depth=0):
- O2 held near saliva_O2 (8% atm by default)
- pH relaxed toward saliva_pH = 6.8 (buffering capacity)
- Sugar pulses per the Stephan curve (meal events)

Bottom boundary (gingival sulcus, depth=N-1):
- Heme flux when bleed loop is active (the OVN ignition trigger)
- (v2: IgG/IgA from host immune layer)
"""
from __future__ import annotations
import math
from ..grid.base import SpatialGrid
from ..config import ChemistryConfig

# Fraction of surface sugar/acid cleared per tick at healthy salivary flow (=1.0).
# Scaled by saliva_flow: xerostomia (flow→0) loses this clearance, so acids/sugar persist.
SALIVA_WASHOUT = 0.1


class BoundaryConditions:
    def __init__(self, cfg: ChemistryConfig):
        self.cfg = cfg
        self._last_meal_tick: int = -10**9

    def stephan_sugar_input(self, tick: int) -> float:
        """Sugar concentration injected at the top boundary this tick.

        Models meal events: amplitude pulse, exponential decay, period stephan_period.
        """
        c = self.cfg
        # Time since most recent meal
        elapsed = tick - self._last_meal_tick
        if c.stephan_period > 0 and tick - self._last_meal_tick >= c.stephan_period:
            self._last_meal_tick = tick
            elapsed = 0
        return float(c.stephan_amplitude * math.exp(-c.stephan_decay * elapsed))

    def heme_bottom_input(self, tick: int) -> float:
        """Heme flux at the gingival sulcus this tick."""
        return float(self.cfg.heme_baseline + self.cfg.heme_bleed_rate)

    def apply(self, grid: SpatialGrid, tick: int) -> None:
        c = self.cfg
        flow = c.saliva_flow
        # --- top boundary (saliva) ---
        top = next(iter(grid.positions()))
        # O2 inflow — delivery scales with salivary flow
        cur_O2 = grid.get_field_at("O2", top)
        grid.add_to_field_at("O2", top, (c.saliva_O2 - cur_O2) * 0.5 * flow)
        # pH relax toward saliva_pH — effective buffering scales with flow
        cur_pH = grid.get_field_at("pH", top)
        grid.add_to_field_at("pH", top, (c.saliva_pH - cur_pH) * c.saliva_buffer * flow)
        # Sugar pulse (meal), then flow-driven surface clearance of sugar + acid.
        # Low flow (xerostomia) loses this washout → acids/sugar linger → caries/dysbiosis.
        grid.add_to_field_at("sugar", top, self.stephan_sugar_input(tick))
        # Surface clearance (swallowing/dilution) — includes nitrate/nitrite, which are
        # continuously swallowed; this keeps steady-state nitrate ∝ dietary delivery.
        for fld in ("sugar", "lactate", "nitrate", "nitrite"):
            cur = grid.get_field_at(fld, top)
            grid.add_to_field_at(fld, top, -cur * SALIVA_WASHOUT * flow)
        # Enterosalivary nitrate delivery (diet-driven), carried by salivary flow.
        grid.add_to_field_at("nitrate", top, c.saliva_nitrate * flow)

        # --- bottom boundary (gingival sulcus) ---
        # iterate to last position
        bot = None
        for p in grid.positions():
            bot = p
        if bot is not None:
            grid.add_to_field_at("heme", bot, self.heme_bottom_input(tick))
            # Host tissue at the pocket base respires — an O2 sink that deepens the
            # anoxic niche the strict anaerobes (Pg: O2 max 0.1) need to establish.
            grid.add_to_field_at("O2", bot, -grid.get_field_at("O2", bot) * c.tissue_o2_sink)
            # Bleeding brings SERUM (pH ~7.4, buffered) into the sulcus — the diseased
            # pocket alkalinizes with bleeding, opening the niche the alkaliphilic red
            # complex (Pg: pH 7.0-8.5) needs. This closes the bleed→keystone loop.
            if c.heme_bleed_rate > 0.0:
                cur_pH = grid.get_field_at("pH", bot)
                grid.add_to_field_at("pH", bot, (c.serum_pH - cur_pH) * c.heme_bleed_rate)

