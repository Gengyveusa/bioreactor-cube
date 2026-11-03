"""Configuration loader.

Single source of truth for parameters. Everything tunable lives in a YAML file
so calibration (ABC, Bayesian) can sweep over it without touching code.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class GridConfig:
    kind: str = "1d"           # "1d" or "3d"
    depth_cells: int = 50      # number of depth bins (1mm biofilm / 20um per cell)
    xy_cells: tuple = (1, 1)   # unused for 1d; (Nx, Ny) for 3d
    cell_size_um: float = 20.0


@dataclass
class ChemistryConfig:
    # Diffusion coefficients (relative units, calibration-tunable)
    D_O2: float = 1.0
    D_pH: float = 0.6
    D_sugar: float = 0.8
    D_heme: float = 0.3
    D_lactate: float = 0.7
    D_NO: float = 1.2
    D_H2S: float = 0.9
    D_NH3: float = 0.8
    # Top boundary (saliva)
    saliva_O2: float = 8.0          # % atmospheric
    saliva_pH: float = 6.8
    saliva_buffer: float = 0.05     # how fast pH relaxes to saliva_pH per tick
    saliva_flow: float = 1.0        # normalized salivary flow; 1.0=healthy unstim (~0.3 mL/min),
                                    # <0.3=hyposalivation/xerostomia. Scales top-boundary clearance.
    saliva_nitrate: float = 0.5     # enterosalivary nitrate delivery at the surface (diet-driven,
                                    # rel/tick). 0=nitrate-poor diet, higher=nitrate-rich (e.g. beetroot).
    # Bottom boundary (gingival sulcus) — heme flux is the BLEED LOOP toggle
    heme_baseline: float = 0.0
    heme_bleed_rate: float = 0.0    # set >0 to ignite OVN cascade
    serum_pH: float = 7.4           # blood/serum pH delivered by bleeding (alkalinizes the sulcus)
    tissue_o2_sink: float = 0.0     # host tissue O2 sink at the pocket base (off by default; over-anoxia
                                    # suppresses the aerotolerant orange complex that drives dysbiosis)
    # Stephan curve (sugar pulse) parameters
    stephan_amplitude: float = 1.0
    stephan_decay: float = 0.02
    stephan_period: int = 240       # ticks between meals (e.g., 240 ticks @ 1min/tick = 4h)


@dataclass
class AgentConfig:
    initial_inoculum_per_species: int = 5
    seed_at_depths: tuple = (0, 1, 2, 3)   # pioneers seed near top
    max_agents: int = 5000
    viability_grace_ticks: int = 30        # ticks an agent can be off-spec before dying
    division_threshold: float = 0.7        # local fitness above this -> divide
    omv_base_rate: float = 0.001           # baseline vesiculation per agent per tick
    omv_stress_multiplier: float = 5.0     # stress-coupled vesiculation
    seeding: dict | None = None            # optional {species_id: relative_abundance} from a
                                           # patient microbial panel; None = uniform inoculum.


@dataclass
class HostConfig:
    """Layer-6 host immunity inputs (patient-specific, normalized; 0 = no signal).

    Set from a PatientProfile (lab-draw inflammatory markers + aMMP-8 + WBC/NLR);
    folded into the running sim by ``sim.host.apply_host_couplings`` and
    ``sim.host.host_clearance_modifiers``.
    """
    inflammation: float = 0.0            # systemic tone (hs-CRP/IL-6/TNF-α/fibrinogen/ESR)
    tissue_breakdown: float = 0.0        # aMMP-8 active collagenase (local destruction)
    antimicrobial_capacity: float = 0.0  # neutrophil-mediated clearance capacity


@dataclass
class SimConfig:
    seed: int = 42
    ticks: int = 1000
    tick_minutes: float = 1.0              # biological minutes per tick
    log_every: int = 10
    output_dir: str = "output"
    grid: GridConfig = field(default_factory=GridConfig)
    chemistry: ChemistryConfig = field(default_factory=ChemistryConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    host: HostConfig = field(default_factory=HostConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        # shallow merge sub-configs
        cfg = cls()
        for k, v in raw.items():
            if hasattr(cfg, k):
                attr = getattr(cfg, k)
                if hasattr(attr, "__dataclass_fields__") and isinstance(v, dict):
                    for kk, vv in v.items():
                        if hasattr(attr, kk):
                            setattr(attr, kk, vv)
                else:
                    setattr(cfg, k, v)
        return cfg

