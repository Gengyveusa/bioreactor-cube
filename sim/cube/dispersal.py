"""Arrival — immigration from the salivary planktonic pool, gated by what a species can stick to.

The engine has never had dispersal. Growth is proportional to existing biomass and nothing is ever
added, so a species at exactly zero can never appear: zero is absorbing. The falsification harness
already apologises for this in ``_clean_surface``, where "from zero" has to be faked with a trace
inoculum because it is not expressible.

That gap is not cosmetic — it makes SUCCESSION untestable. Colonisation order is a statement about
who ARRIVES first. With every species seeded at once, the only thing a peak-time measurement can
report is who declines slowest, which is why F1 fails and why F12 fails even in the regime where
the late colonisers are viable: from a trace inoculum they peak at t=0 and decay from there.

WHAT GATES ARRIVAL. Not the same thing for everyone, and that asymmetry is the whole mechanism:

  PIONEERS bind the acquired pellicle directly — a salivary protein film on enamel — so they can
  colonise a freshly cleaned surface with nothing else present. Their gate is proximity to the
  hard-tissue face.

  EVERYONE ELSE cannot. Late colonisers have no pellicle adhesins; they attach to receptors
  displayed by organisms already there (Kolenbrander's coaggregation lineage). Their gate is
  :func:`~sim.cube.phage.coaggregation_support` — the same field the bridging work already uses.

So succession is not scheduled anywhere. It falls out: pioneers can land on bare enamel, bridges
can only land once pioneers are present, and the red complex can only land once the bridges are.

OFF BY DEFAULT. Immigration changes the long-run state of every existing run — an absorbing zero is
a strong constraint, and removing it lets knocked-out species creep back. Ablation experiments in
particular depend on knockout being permanent, so they must opt out (and the harness does).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.cube.phage import coaggregation_support
from sim.species import BY_ID, CATALOG

#: How fast the planktonic pool delivers biomass to an ideal attachment site, per unit time.
#: Slow relative to growth: colonisation is seeding followed by expansion, not a flood. [[VERIFY]]
IMMIGRATION_RATE = 0.004

#: Depth (in voxels) over which pellicle-binding is possible. Pioneers bind the salivary protein
#: film on the hard surface, so their opportunity is at the enamel face, not throughout the volume.
PELLICLE_DEPTH = 2.0

#: Local partner biomass at which a non-pioneer's attachment opportunity saturates. Deliberately
#: NOT the same as coaggregation's growth half-constant: attaching needs fewer receptors than
#: thriving does. [[VERIFY]]
ATTACH_HALF = 0.02

#: Archetypes that carry pellicle adhesins. Everyone else needs a partner already present.
PELLICLE_BINDERS = ("Pioneer",)


def salivary_pool() -> dict:
    """Relative planktonic abundance in saliva — what is available to arrive.

    Saliva is not the pocket: it is dominated by the mitis-group streptococci and other early
    colonisers that shed freely, while the strict anaerobes of the deep pocket are comparatively
    rare in it. Values are ordinal. [[VERIFY against a salivary 16S profile]]
    """
    weights = {"Pioneer": 1.0, "NitrateReducer": 0.6, "Cariogenic": 0.5, "CrossKingdom": 0.3,
               "Bridge": 0.35, "Aggressive": 0.2, "OSCC": 0.1, "Keystone": 0.1}
    return {s.id: weights.get(s.archetype, 0.2) for s in CATALOG}


def _pellicle_field(shape) -> np.ndarray:
    """1 at the hard-tissue face, decaying inward — where a pellicle-binder can get a hold."""
    x = np.arange(shape[0], dtype=float)
    prof = np.exp(-x / PELLICLE_DEPTH)
    return np.broadcast_to(prof[:, None, None], shape)


def attachment_gate(eng, sid: str) -> np.ndarray:
    """0..1 field: where this species can actually attach if it arrives.

    Example: attachment_gate(eng, "Sa") is high at the enamel face on a bare surface;
             attachment_gate(eng, "Pg") is ~0 there until its partners exist
    """
    arche = BY_ID[sid].archetype if sid in BY_ID else ""
    if arche in PELLICLE_BINDERS:
        return _pellicle_field(eng.grid.shape)
    partners = coaggregation_support(eng, sid)          # already smoothed to a contact process
    return partners / (partners + ATTACH_HALF)


@dataclass
class Dispersal:
    """Immigration from saliva. Attach to an engine as ``eng.dispersal``.

    Example: eng.dispersal = Dispersal()      # now a species at zero can return
    """

    rate: float = IMMIGRATION_RATE
    pool: dict = field(default_factory=salivary_pool)
    #: Species that must NOT be allowed back. Ablation experiments depend on knockout being
    #: permanent; without this, immigration silently undoes every knockout the harness performs.
    blocked: frozenset = frozenset()

    def step(self, eng, dt: float) -> None:
        comp = eng.grid.composition
        total = comp.sum(axis=3)
        headroom = np.clip(1.0 - total, 0.0, None)      # no arrival into a saturated voxel
        for n, sid in enumerate(eng.species):
            if sid in self.blocked:
                continue
            supply = self.pool.get(sid, 0.0)
            if supply <= 0.0:
                continue
            comp[..., n] += dt * self.rate * supply * attachment_gate(eng, sid) * headroom

