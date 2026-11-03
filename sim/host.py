"""Layer-6 host immunity couplings into the Layer-1/3 simulator (patient-anchored).

Three couplings (design §6), applied as pure transforms before the run:

1. inflammation → heme_bleed_rate (+): inflamed gingiva bleeds → heme → the Pg
   bloom. The OVN feed-forward, now anchored to the patient's systemic markers.
2. tissue_breakdown (aMMP-8) → deeper anoxic niche, modeled as reduced surface O2
   delivery (a deeper pocket gets less oxygen to its base → favors the red complex).
3. antimicrobial_capacity → bacterial clearance mortality, SUBVERTED by keystone
   gingipains: Pg/Td/Tf degrade complement/Ig and evade neutrophil killing, so the
   red complex receives only a fraction of the clearance pressure.

Couplings 1-2 fold into existing chemistry knobs (read every tick by the boundary
solver). Coupling 3 emits ``species_modifiers`` in the exact format the agent step
already consumes (``baseline_mortality_mult`` per species id, agents.py). The
dynamic version of the subversion (scaling with live Pg abundance each tick) is a
v2 refinement; v1 treats keystone evasion as constitutive, a reasonable
approximation since gingipains degrade complement constitutively.

All gains are PLACEHOLDERS pending cohort calibration (design §13).
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

from .config import SimConfig

K_INFLAMMATION_BLEED = 0.02   # heme_bleed_rate added per unit inflammation
K_TISSUE_O2 = 1.0             # saliva_O2 (% atm) removed per unit tissue_breakdown
K_CLEARANCE_MORTALITY = 0.3   # baseline-mortality multiplier added per unit capacity
KEYSTONE_EVASION = 0.2        # fraction of clearance the red complex still receives

KEYSTONE_IDS = ("Pg", "Td", "Tf")   # red-complex keystones — gingipain-mediated evasion


def apply_host_couplings(cfg: SimConfig) -> SimConfig:
    """Fold host inputs (``cfg.host``) into the chemistry boundaries. Pure — returns a
    deep copy. Couplings 1 (inflammation → bleed loop) and 2 (tissue breakdown →
    anoxia). Call once, after the PatientProfile overrides have built ``cfg``."""
    out = copy.deepcopy(cfg)
    h = out.host
    out.chemistry.heme_bleed_rate += K_INFLAMMATION_BLEED * h.inflammation
    out.chemistry.saliva_O2 = max(0.0, out.chemistry.saliva_O2 - K_TISSUE_O2 * h.tissue_breakdown)
    return out


def host_clearance_modifiers(cfg: SimConfig, species_ids: Iterable[str]) -> dict[str, dict[str, float]]:
    """Coupling 3: per-species ``baseline_mortality_mult`` from host antimicrobial
    capacity, subverted by keystone gingipains. Returns the ``species_modifiers``
    dict the agent step consumes; empty when there is no antimicrobial signal.

    Non-keystones get the full clearance pressure; the red complex evades down to a
    fraction (``KEYSTONE_EVASION``) of it.
    """
    cap = cfg.host.antimicrobial_capacity
    if cap <= 0.0:
        return {}
    full = 1.0 + K_CLEARANCE_MORTALITY * cap
    evaded = 1.0 + K_CLEARANCE_MORTALITY * cap * KEYSTONE_EVASION
    return {
        sid: {"baseline_mortality_mult": evaded if sid in KEYSTONE_IDS else full}
        for sid in species_ids
    }

