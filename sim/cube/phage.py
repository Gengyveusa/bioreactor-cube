"""The third combatant — bacteriophage, as an actual population rather than a label.

Until this module, ``phage`` appeared in the atlas, the viewer, the causal DAG and the
bridge-asymmetry analysis, but nothing in the engine simulated one. The war had two sides.

WHAT A PHAGE IS HERE. A free-particle field that adsorbs to a host species, kills it after a latent
period, and bursts into more particles. Four things make that more than a predator-prey toy, and
each of them is a place where the biofilm — not the phage — decides the outcome:

  1. THE MATRIX IS THE DEFENCE. A virion is a ~100 nm particle; lactate is a small molecule. Both
     cross the same EPS gel, and the gel does not treat them alike. Phage transport is gated by
     porosity AND by :func:`sim.cube.ecm.obstruction_factor`, the measured exp(-k*m*r^2) size law
     shared with the drug-delivery model — so a mature matrix is a phage refuge even when it is
     freely permeable to nutrients. This is why phage therapy that works in a test tube stalls on
     an established biofilm.

  2. DEFENCE IS ALREADY IN THE GENOME, AND IT IS ASYMMETRIC. ``genome["defense"]`` is 0.7 for the
     keystones, 0.5 for pathobionts, 0.35 for commensals — written long before this module. The
     Bridge-Asymmetry Theorem says bridges pay their evolutionary rent in connectivity rather than
     autonomy, so heavy defence is selected against in exactly the species the network runs
     through. This module does not re-encode that. It reads the existing values, so if the theorem
     is wrong in the catalog it will be wrong here too — which is the honest arrangement.

  3. RESISTANCE EVOLVES, AND IT COSTS. Receptor loss is the commonest escape and the receptor is
     usually an adhesin. A host that stops being infectable often stops being able to coaggregate,
     which is a real fitness cost and, for a bridge, is nearly the whole job.

  4. MOST ORAL PHAGES ARE TEMPERATE. Lysogeny is the norm, not the exception, so infection here can
     integrate as prophage rather than kill — and prophage induces under stress (acid, oxidative,
     DNA damage). A phage that mostly lysogenises is a poor therapeutic and a potent gene shuttle.

THE INDIRECT ATTACK. Killing a host that a keystone DEPENDS ON collapses the keystone without a
keystone-specific phage — the Wu/Fujiki result (Appl Environ Microbiol 2024;90(10):e0095124,
PMID 39248462, >99% fall in *P. gingivalis* via phage predation of *S. gordonii*). For that to be
reproducible here rather than asserted, late colonisers must actually need their partners, which
is what :func:`coaggregation_support` supplies. It is OPT-IN: switching it on changes every
existing result, so it does not switch itself on.

ALL RATE CONSTANTS ARE ORDER-OF-MAGNITUDE. Burst sizes and lysogeny propensities are literature-
shaped; adsorption, decay and latent period are in model time units that have never been calibrated
against a clock. Relative comparisons (bridge vs keystone targeting, matrix on vs off) are the
claims this module can support. Absolute titres are not. [[VERIFY everything numeric below]]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from sim.cube.ecm import BIOFILM_ECM, _diff3, obstruction_factor, porosity
from sim.species import BY_ID, CATALOG, COAGG_EDGES

# --------------------------------------------------------------------------- physical constants
#: RETIRED. This was PHAGE_ECM_EXPONENT = 3.0 — "the crudest defensible way to say that with one
#: number", marked [[VERIFY]], and the verification came back negative. Peulen & Wilkinson 2011
#: (PMID 21434601) measured size scaling in a LIVE biofilm and found exp(-k*r^2), not a power of
#: porosity. The law now lives in ecm.obstruction_factor and is shared with the drug-delivery
#: model, so a virion and a drug molecule are the same physics at different radii rather than two
#: rules that could drift apart.
#:
#: Kept as a name so any import still resolves; nothing reads it.
PHAGE_ECM_EXPONENT = None

#: Mutation supply — the per-unit-time rate at which fresh resistant biomass appears from a
#: susceptible population even with no phage present. Without a seed, resistance starting at zero
#: can never appear, and the model would report eradication that real populations never deliver.
RESISTANCE_SEED = 2.0e-4

#: How hard phage pressure selects for the resistant fraction already present.
RESISTANCE_SELECTION = 0.8

#: Fitness cost of resistance, as a fraction of growth. Receptor loss usually means adhesin loss.
RESISTANCE_COST = 0.25

#: Extra cost paid by a species whose job IS adhesion. A bridge that stops binding is not a bridge.
RESISTANCE_COST_BRIDGE = 0.55

#: Stress that induces prophage. Acid stress is the one this engine actually resolves.
INDUCTION_PH = 5.5


# --------------------------------------------------------------------------- the phage itself
@dataclass(frozen=True)
class PhageSpec:
    """One phage population and its host range.

    Example: PhageSpec("phiFn", hosts=("Fn",), burst=50.0)
    """

    id: str
    hosts: tuple = ()
    adsorption: float = 6.0      # per unit free phage per unit susceptible biomass per unit time
    latent: float = 0.6          # model-time from infection to lysis
    burst: float = 60.0          # particles released per unit lysed biomass
    decay: float = 0.35          # free-particle loss per unit time (saliva clearance + denaturing)
    lysogeny: float = 0.30       # baseline fraction of infections that integrate instead of lysing
    induction: float = 0.05      # prophage -> lytic conversion per unit time under stress
    diffusion: float = 1.0       # multiplier on the ECM-gated stencil
    #: Hydrodynamic radius in nm. A tailed phage head runs ~30-60 nm; 50 is the calibration point
    #: at which the obstruction law reproduces what theta^3 used to give. [[VERIFY per phage]]
    radius_nm: float = 50.0
    note: str = ""

    def targets(self, sid: str) -> bool:
        return sid in self.hosts


# A small default panel spanning the three strategies the literature actually distinguishes.
# Burst sizes are literature-shaped: streptococcal siphophages run tens-to-low-hundreds, and
# anaerobe phages are typically smaller and slower. Lysogeny is high across the board because the
# oral virome is dominated by temperate phage. [[VERIFY]]
PHI_PIONEER = PhageSpec(
    "phiSt", hosts=("Sa", "Sm"), adsorption=7.0, latent=0.5, burst=80.0, decay=0.35,
    lysogeny=0.35, induction=0.06, diffusion=1.0,
    note="anti-pioneer streptococcal phage — the Wu/Fujiki analogue: hits the surface the late "
         "colonisers attach to, not the late colonisers themselves",
)
PHI_BRIDGE = PhageSpec(
    "phiFn", hosts=("Fn",), adsorption=6.0, latent=0.6, burst=50.0, decay=0.35,
    lysogeny=0.30, induction=0.05, diffusion=1.0,
    note="anti-bridge phage — the Bridge-Asymmetry play: F. nucleatum carries 14 of the 45 "
         "coaggregation edges and is only moderately defended",
)
PHI_KEYSTONE = PhageSpec(
    "phiPg", hosts=("Pg",), adsorption=4.0, latent=0.9, burst=30.0, decay=0.40,
    lysogeny=0.45, induction=0.04, diffusion=1.0,
    note="direct anti-keystone phage — the one that is hardest to isolate in reality, included so "
         "the model can be asked what it WOULD do rather than assuming it is unavailable",
)
DEFAULT_PANEL = (PHI_PIONEER, PHI_BRIDGE, PHI_KEYSTONE)


# --------------------------------------------------------------------------- coaggregation
def _coagg_partners() -> dict:
    p: dict = {s.id: set() for s in CATALOG}
    for u, v in COAGG_EDGES:
        if u in p and v in p:
            p[u].add(v)
            p[v].add(u)
    return {k: tuple(sorted(v)) for k, v in p.items()}


COAGG_PARTNERS = _coagg_partners()

#: How much a species depends on finding its coaggregation partners already present. Pioneers bind
#: the acquired pellicle directly and need nobody; the late colonisers cannot colonise clean enamel
#: and are the reason coaggregation is studied at all. Values are ordinal, not measured. [[VERIFY]]
COAGG_DEPENDENCE = {
    "Pioneer": 0.0, "NitrateReducer": 0.15, "Cariogenic": 0.1, "CrossKingdom": 0.2,
    "Bridge": 0.35, "Aggressive": 0.4, "OSCC": 0.7, "Keystone": 0.75,
}

#: Local partner biomass at which support saturates.
COAGG_HALF = 0.06


def coaggregation_support(eng, sid: str) -> np.ndarray:
    """0..1 field: how well this species' coaggregation partners are represented nearby.

    Partner biomass is smoothed by one diffusion pass because coaggregation is a contact process
    between neighbours, not a whole-cube average — a partner three hundred microns away is not a
    binding site.

    Example: coaggregation_support(eng, "Pg") is high only where Fn/Pi/Td/Tf actually are
    """
    partners = COAGG_PARTNERS.get(sid, ())
    if not partners:
        return np.ones(eng.grid.shape)
    comp = eng.grid.composition
    acc = np.zeros(eng.grid.shape)
    for p in partners:
        try:
            acc += comp[..., eng.grid.species_index(p)]
        except (KeyError, ValueError):
            continue
    return np.clip(_diff3(acc) / COAGG_HALF, 0.0, 1.0)


def coaggregation_gate(eng, sid: str) -> np.ndarray:
    """Growth multiplier from partner availability: ``(1-dep) + dep * support``.

    A species with dependence 0 is unaffected. A keystone at dependence 0.75 loses three quarters
    of its growth where its partners are absent — which is the mechanism that lets an attack on a
    bridge propagate to a keystone the phage never touches.
    """
    dep = COAGG_DEPENDENCE.get(BY_ID[sid].archetype, 0.2) if sid in BY_ID else 0.2
    if dep <= 0.0:
        return np.ones(eng.grid.shape)
    return (1.0 - dep) + dep * coaggregation_support(eng, sid)


# --------------------------------------------------------------------------- the system
@dataclass
class PhageSystem:
    """The phage populations riding on a cube, plus the state they need between ticks.

    Attach to an engine and it runs each step:

    >>> sysp = PhageSystem(panel=[PHI_BRIDGE])          # doctest: +SKIP
    >>> sysp.dose(eng, "phiFn", 0.5)                    # doctest: +SKIP
    >>> eng.phage = sysp                                # doctest: +SKIP
    """

    panel: tuple = DEFAULT_PANEL
    coaggregation: bool = False       # OPT-IN: changes every existing result when enabled
    _ready: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------ state helpers
    def _free(self, eng, pid: str) -> np.ndarray:
        return eng.extra.setdefault("phage:%s" % pid, np.zeros(eng.grid.shape))

    def _lysing(self, eng, pid: str, host: str) -> np.ndarray:
        return eng.extra.setdefault("lysing:%s:%s" % (pid, host), np.zeros(eng.grid.shape))

    def _prophage(self, eng, pid: str, host: str) -> np.ndarray:
        return eng.extra.setdefault("prophage:%s:%s" % (pid, host), np.zeros(eng.grid.shape))

    def resistant(self, eng, host: str) -> np.ndarray:
        """Resistant FRACTION of a host species, 0..1."""
        return eng.extra.setdefault("resistant:%s" % host, np.zeros(eng.grid.shape))

    def prepare(self, eng) -> None:
        """Allocate every field once so a readout before the first step does not KeyError."""
        # The flag lives on the engine (the community owns it, not the phage), but a PhageSystem
        # asking for coaggregation must still get it.
        if self.coaggregation:
            eng.coaggregation = True
        for spec in self.panel:
            self._free(eng, spec.id)
            for h in spec.hosts:
                self._lysing(eng, spec.id, h)
                self._prophage(eng, spec.id, h)
                self.resistant(eng, h)
        self._ready = True

    def dose(self, eng, pid: str, titre: float, *, where: str = "oral") -> None:
        """Apply a therapeutic dose of one phage.

        ``where='oral'`` delivers at the mouth-facing surface, which is the only route a rinse or
        an irrigant actually has. That matters: a dose that must then diffuse INWARD through the
        matrix is a very different experiment from seeding phage uniformly through the biofilm,
        and seeding uniformly is how phage therapy gets to look better on a lattice than in a
        mouth. ``where='uniform'`` is available for exactly that contrast.
        """
        if not self._ready:
            self.prepare(eng)
        f = self._free(eng, pid)
        if where == "uniform":
            f += titre
        elif where == "oral":
            f[:, :, -1] += titre * f.shape[2]      # surface flux, resolution-independent
        else:
            raise ValueError("dose(where=) must be 'oral' or 'uniform', got %r" % where)

    # ------------------------------------------------------------------ the tick
    def step(self, eng, dt: float) -> None:
        """One phage tick: transport -> adsorption -> lysogeny/lysis -> induction -> resistance."""
        if not self._ready:
            self.prepare(eng)
        comp = eng.grid.composition
        # The matrix the phage has to cross. Falls back to the ECM baseline before the oral tissue
        # has built one, so a bare grid still behaves sensibly.
        m = eng.extra.get("ecm")
        if m is None:
            m = np.full(eng.grid.shape, BIOFILM_ECM.base_density)
        # Porosity gates transport for everything; obstruction_factor adds the size selectivity a
        # ~50 nm particle sees and a small solute does not. Separable, so each can be calibrated
        # against its own data — which theta^n made impossible.
        theta_small = porosity(m, BIOFILM_ECM.alpha)
        ph = eng.grid.get_field("pH")

        for spec in self.panel:
            theta = theta_small * obstruction_factor(m, spec.radius_nm)
            free = self._free(eng, spec.id)

            # --- transport: gated hard by the matrix, then decayed
            free[:] = _diff3(free * theta * spec.diffusion) + free * (1.0 - theta * spec.diffusion)
            free *= max(0.0, 1.0 - dt * spec.decay)

            for host in spec.hosts:
                try:
                    hi = eng.grid.species_index(host)
                except (KeyError, ValueError):
                    continue
                biomass = comp[..., hi]
                res = self.resistant(eng, host)
                lysing = self._lysing(eng, spec.id, host)
                pro = self._prophage(eng, spec.id, host)
                defense = float(eng.genomes.get(host, {}).get("defense", 0.35))

                # --- adsorption. Susceptible biomass only, discounted by the genome's own defence
                # and by how much matrix stands between the particle and the cell.
                susceptible = np.clip(biomass * (1.0 - res), 0.0, None)
                contact = dt * spec.adsorption * free * susceptible * (1.0 - defense) * theta
                contact = np.minimum(contact, susceptible)      # cannot infect more than exists

                # --- lysogeny vs lysis. Temperate phage integrate more readily when hosts are
                # crowded and the phage is abundant (the classic MOI dependence), so the propensity
                # rises with local free-phage density rather than being a flat coin-flip.
                moi = free / (susceptible + 1e-9)
                lyso_frac = np.clip(spec.lysogeny * (1.0 + 0.5 * np.tanh(moi)), 0.0, 0.95)
                to_prophage = contact * lyso_frac
                to_lytic = contact - to_prophage

                comp[..., hi] = biomass - contact          # infected cells leave the growing pool
                lysing += to_lytic
                pro += to_prophage
                free -= contact                            # particles are consumed on adsorption

                # --- induction: prophage goes lytic under acid stress
                if spec.induction > 0.0:
                    stress = np.clip((INDUCTION_PH - ph) / 1.5, 0.0, 1.0)
                    induced = np.minimum(pro, dt * spec.induction * (0.1 + stress) * pro)
                    pro -= induced
                    lysing += induced

                # --- lysis: the burst
                lysed = np.minimum(lysing, dt * lysing / max(spec.latent, 1e-6))
                lysing -= lysed
                free += spec.burst * lysed

                # --- resistance, as a selection differential rather than a ratchet.
                # Receptor loss is adhesin loss, and for a bridge the adhesin IS the ecological
                # role, so bridges pay more to escape than anyone.
                arche = BY_ID[host].archetype if host in BY_ID else ""
                cost = RESISTANCE_COST_BRIDGE if arche == "Bridge" else RESISTANCE_COST
                pressure = free / (free + 0.05)
                # The resistant fraction gains where phage pressure exceeds the cost of escape and
                # LOSES where it does not. Writing this as supply-plus-selection only, with no
                # negative term, was a ratchet: with no phage anywhere the mutational supply still
                # drove resistance to fixation, which would have quietly made every long run
                # phage-proof before a phage was ever dosed.
                res += dt * (RESISTANCE_SEED * (1.0 - res)
                             + res * (1.0 - res) * (RESISTANCE_SELECTION * pressure - cost))
                np.clip(res, 0.0, 1.0, out=res)

                # The cost is paid in growth by whatever fraction is currently resistant.
                growth = float(eng.genomes.get(host, {}).get("growth", 0.12))
                comp[..., hi] -= dt * growth * cost * res * comp[..., hi]

            np.clip(free, 0.0, None, out=free)
        np.clip(comp, 0.0, None, out=comp)

    # ------------------------------------------------------------------ readout
    def readout(self, eng) -> dict:
        """Titres, infection burden, prophage carriage and resistance — all intensive."""
        out: dict = {}
        comp = eng.grid.composition
        for spec in self.panel:
            free = self._free(eng, spec.id)
            out["titre:%s" % spec.id] = round(float(free.mean()), 6)
            for host in spec.hosts:
                lys = self._lysing(eng, spec.id, host).sum()
                pro = self._prophage(eng, spec.id, host).sum()
                try:
                    alive = float(comp[..., eng.grid.species_index(host)].sum())
                except (KeyError, ValueError):
                    alive = 0.0
                denom = max(alive + float(lys) + float(pro), 1e-9)
                out["infected:%s" % host] = round(float(lys) / denom, 4)
                out["prophage:%s" % host] = round(float(pro) / denom, 4)
                out["resistant:%s" % host] = round(float(self.resistant(eng, host).mean()), 4)
        return out


def community_shares(eng) -> dict:
    """Every species' share of total biomass — the intensive view a phage experiment needs."""
    comp = eng.grid.composition
    tot = float(comp.sum()) or 1.0
    return {sid: float(comp[..., i].sum()) / tot for i, sid in enumerate(eng.species)}

