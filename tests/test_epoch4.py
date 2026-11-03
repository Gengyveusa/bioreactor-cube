"""Tests for the Epoch-4 engine core — information primary, operators pinned, oral instance runs.

Each operator is tested in its own box (no crisscross), and the loop is shown to produce a sensible
tissue-specific ψ. The headline behaviour: TERRAIN GATES which information expresses — the same agent
thrives in its niche and decays out of it (information-primary, terrain-gated)."""

from __future__ import annotations

from sim.epoch4 import ORAL, Agent, Information, express, gate, recognize, run
from sim.epoch4.tissue import _genome
from sim.species import BY_ID


def _agent(sid: str, depth: float = 0.5, biomass: float = 0.1) -> Agent:
    s = BY_ID[sid]
    return Agent(sid, depth, Information(genome=_genome(s), regulatory={"metabolism": 0.2},
                                         signaling={}, partners=()), biomass)


def test_information_is_the_substrate_with_four_faces() -> None:
    a = _agent("Pg")
    assert set(vars(a.info)) == {"genome", "regulatory", "signaling", "partners"}
    assert a.info.genome["o2_opt"] == BY_ID["Pg"].envelope.o2[2]   # genome carries identity


def test_T_gate_returns_expressibility_in_unit_interval() -> None:
    deep_anaerobe = _agent("Pg")            # Pg: o2_opt = 0, needs heme
    assert 0.0 <= gate(deep_anaerobe, {"O2": 0.0, "pH": 7.6, "heme": 1.0}) <= 1.0


def test_terrain_gates_information_the_headline_law() -> None:
    """Same agent, two terrains: a deep anaerobe expresses deep (anoxic, heme) and is shut down shallow."""
    pg = _agent("Pg")
    deep = gate(pg, {"O2": 0.05, "pH": 7.6, "heme": 1.0})
    shallow = gate(pg, {"O2": 8.0, "pH": 6.8, "heme": 0.0})
    assert deep > shallow                    # terrain decides whether the information can express
    # and a surface aerobe is the opposite
    sa = _agent("Sa")
    assert gate(sa, {"O2": 8.0, "pH": 6.8, "heme": 0.0}) > gate(sa, {"O2": 0.0, "pH": 7.6, "heme": 1.0})


def test_E_express_reads_information_into_biomass_and_metabolism() -> None:
    a = _agent("Sa", biomass=0.1)
    b0, m0 = a.biomass, a.info.regulatory["metabolism"]
    express(a, {"O2": 8.0, "pH": 6.8, "heme": 0.0}, expressibility=0.9, dt=0.2)
    assert a.info.regulatory["metabolism"] > m0     # expression relaxes toward what terrain permits
    assert a.biomass > b0                            # energy grows biomass


def test_I_recognize_is_higher_for_better_defended_information() -> None:
    keystone = _agent("Pg")        # defense 0.7
    commensal = _agent("Sa")       # defense 0.35
    assert recognize(keystone) > recognize(commensal)


def test_oral_loop_runs_and_produces_tissue_specific_psi() -> None:
    pop, traj = run(ORAL, steps=80, dt=0.1)
    last = traj[-1]
    assert {"dysbiosis", "NO", "omv_flux", "biomass", "n_alive"} <= set(last)
    assert 0.0 <= last["dysbiosis"] <= 1.0
    assert last["NO"] >= 0.0 and last["omv_flux"] >= 0.0
    assert last["n_alive"] > 0                       # a community persists
    assert len(traj) >= 2                            # a trajectory over time (Δ)


def test_information_moves_over_time() -> None:
    """The substrate is the thing that moves: expression states change across the run."""
    pop, _ = run(ORAL, steps=60, dt=0.1)
    mets = [a.info.regulatory["metabolism"] for a in pop if a.alive]
    assert any(m > 0.25 for m in mets)               # some information got expressed up from the 0.2 seed

