"""Functional groups — collapse the 18-species catalog into the display/ecology groups used by the
visualization (and any group-level analysis). Priority-ordered so every species lands in exactly one.
"""

from __future__ import annotations

from sim.species import CATALOG, Species

GROUP_COLOR = {
    "reducers": "#2bd4e8",   # nitrate-reducers — protective (NO axis)
    "red": "#f87171",        # red-complex keystones — aggressive pathogens
    "pioneers": "#6ee7a8",   # early colonizers / commensals
    "bridges": "#fbbf24",    # connectors (Fusobacterium et al.)
    "fungi": "#e879f9",      # C. albicans (cross-kingdom)
    "other": "#fb923c",      # aggressive / cariogenic / OSCC pathobionts
}
GROUP_LABEL = {
    "reducers": "nitrate-reducers (protective)",
    "red": "red-complex pathogens",
    "pioneers": "pioneers / commensals",
    "bridges": "bridges (connectors)",
    "fungi": "fungi (Candida)",
    "other": "other pathobionts",
}


def species_group(s: Species) -> str:
    if s.gram == "fungal":
        return "fungi"
    if s.socransky == "red":
        return "red"
    if s.archetype == "Bridge":
        return "bridges"
    if s.nitrate_reducer:
        return "reducers"
    if s.archetype == "Pioneer":
        return "pioneers"
    return "other"


SPECIES_GROUP: dict[str, str] = {s.id: species_group(s) for s in CATALOG}

