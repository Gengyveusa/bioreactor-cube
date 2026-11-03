"""Taxon resolution — vendor/kit names to cube species ids.

The mapping is DATA, not branching logic, so a microbiologist can audit it without reading Python.
Resolution order: exact species id, then a curated alias, then a normalised scientific-name match
against the 61 modeled species (CATALOG + EXTENDED).

A note on the ``Ca`` collision, because it will bite someone: ``Ca`` is *Candida albicans* in the
species catalog and *calcium* in the chemistry fields. They live in different namespaces inside the
engine, but a CSV column headed "Ca" is genuinely ambiguous. Resolution here is species-only: this
module never returns a chemistry field, and callers must not feed chemistry columns through it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from sim.species import CATALOG
from sim.species_extended import EXTENDED


def _all_species() -> list:
    """CATALOG (the 18 simulated by default) plus EXTENDED (43 more with defined envelopes)."""
    return list(CATALOG) + list(EXTENDED)


def modeled_species_ids(extended: bool = False) -> list:
    """Ids the cube can actually simulate. Default = the 18-species CATALOG the grid is built on.

    Example: modeled_species_ids()[:3] -> ['Sa', 'Sm', 'An']
    """
    return [s.id for s in (_all_species() if extended else CATALOG)]


# --------------------------------------------------------------------------- the mapping table
# Kit shorthand -> cube species id. Sources: OralDNA MyPerioPath panel composition; Bristle/NGS
# report names. Keys are matched case-insensitively after normalisation.
#
# MyPerioPath reports 11 organisms. Ten have a model node once EXTENDED is included; only
# Capnocytophaga has none, and it is recorded as unmapped rather than silently dropped.
ALIASES: dict = {
    # --- MyPerioPath high risk
    "aa": "Aa", "a. actinomycetemcomitans": "Aa", "aggregatibacter actinomycetemcomitans": "Aa",
    "actinobacillus actinomycetemcomitans": "Aa",
    "pg": "Pg", "p. gingivalis": "Pg", "porphyromonas gingivalis": "Pg",
    "tf": "Tf", "t. forsythia": "Tf", "tannerella forsythia": "Tf", "bacteroides forsythus": "Tf",
    "td": "Td", "t. denticola": "Td", "treponema denticola": "Td",
    # --- MyPerioPath moderate
    "en": "En", "e. nodatum": "En", "eubacterium nodatum": "En",
    "fn": "Fn", "f. nucleatum": "Fn", "fusobacterium nucleatum": "Fn",
    "fusobacterium periodonticum": "Fn", "fn/fp": "Fn",
    "pi": "Pi", "p. intermedia": "Pi", "prevotella intermedia": "Pi",
    "cr": "Cr", "c. rectus": "Cr", "campylobacter rectus": "Cr",
    "pm": "Pm", "p. micra": "Pm", "parvimonas micra": "Pm",
    "peptostreptococcus micros": "Pm", "micromonas micros": "Pm",
    # --- MyPerioPath low
    "ec": "Ec", "e. corrodens": "Ec", "eikenella corrodens": "Ec",
    # --- commensals / reducers an NGS panel adds (the dysbiosis denominator)
    "sa": "Sa", "s. sanguinis": "Sa", "streptococcus sanguinis": "Sa",
    "sm": "Sm", "s. mitis": "Sm", "streptococcus mitis": "Sm", "s. oralis": "Sm",
    "streptococcus oralis": "Sm", "s. mitis/oralis": "Sm",
    "an": "An", "a. naeslundii": "An", "actinomyces naeslundii": "An",
    "hp": "Hp", "h. parainfluenzae": "Hp", "haemophilus parainfluenzae": "Hp",
    "rm": "Rm", "r. mucilaginosa": "Rm", "rothia mucilaginosa": "Rm",
    "vp": "Vp", "v. parvula": "Vp", "veillonella parvula": "Vp",
    "smu": "Smu", "s. mutans": "Smu", "streptococcus mutans": "Smu",
    "lb": "Lb", "lactobacillus": "Lb", "lactobacillus spp.": "Lb",
    "fa": "Fa", "f. alocis": "Fa", "filifactor alocis": "Fa",
    "ca": "Ca", "c. albicans": "Ca", "candida albicans": "Ca",
}

# Reported taxa we deliberately know we cannot model, so they are reported as unmapped rather than
# quietly merged into something they are not.
KNOWN_UNMODELED: dict = {
    "capnocytophaga": "no model node (MyPerioPath low-risk group)",
    "capnocytophaga species": "no model node (MyPerioPath low-risk group)",
    "cs": "no model node (Capnocytophaga, MyPerioPath low-risk group)",
}


def _norm(name: str) -> str:
    """Lowercase, collapse whitespace, drop trailing counts/units a kit CSV may carry."""
    s = str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*\((.*?)\)\s*$", "", s)       # trailing parenthetical
    return s.strip()


@dataclass(frozen=True)
class TaxonResolution:
    """The outcome of resolving one reported taxon name.

    Example: resolve_taxon("Porphyromonas gingivalis") -> TaxonResolution('Pg', 'alias', True)
    """

    species_id: Optional[str]
    how: str                 # 'id' | 'alias' | 'name' | 'unmodeled' | 'unknown'
    in_catalog: bool         # True when the id is one of the 18 the grid simulates by default
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.species_id is not None


def resolve_taxon(name: str) -> TaxonResolution:
    """Map one reported taxon name onto a cube species id.

    Never guesses: an unrecognised name resolves to ``None`` with ``how='unknown'`` so the caller
    can record it as unmapped. Silently merging an unknown organism into a nearby one would corrupt
    the composition without leaving a trace.
    """
    raw = str(name).strip()
    catalog_ids = {s.id for s in CATALOG}
    all_by_id = {s.id: s for s in _all_species()}

    if raw in all_by_id:
        return TaxonResolution(raw, "id", raw in catalog_ids)

    key = _norm(raw)
    if key in KNOWN_UNMODELED:
        return TaxonResolution(None, "unmodeled", False, KNOWN_UNMODELED[key])
    if key in ALIASES:
        sid = ALIASES[key]
        return TaxonResolution(sid, "alias", sid in catalog_ids)

    for s in _all_species():                       # normalised scientific-name match
        if _norm(s.name) == key:
            return TaxonResolution(s.id, "name", s.id in catalog_ids)

    return TaxonResolution(None, "unknown", False, f"no model node for {raw!r}")


def coverage_of(reported: list, extended: bool = False) -> dict:
    """Summarise how a list of reported taxon names lands on the model.

    Example: coverage_of(["Pg", "Capnocytophaga"]) -> {'mapped': ['Pg'], 'unmapped': [...], ...}
    """
    mapped, unmapped, out_of_catalog = [], [], []
    for nm in reported:
        r = resolve_taxon(nm)
        if not r.ok:
            unmapped.append((nm, r.reason))
        elif r.in_catalog or extended:
            mapped.append(r.species_id)
        else:
            out_of_catalog.append((nm, r.species_id))
    return {
        "mapped": mapped,
        "unmapped": unmapped,
        "outside_default_catalog": out_of_catalog,
        "n_modeled": len(modeled_species_ids(extended)),
    }

