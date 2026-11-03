"""Tests for the cube AI layer — GNN roles/attachment, binding seam, and grounded Q&A.

The Q&A is grounded: every figure an answer states must come from the live engine digest, so the test
checks that real numbers (e.g. the actual dysbiosis index) appear in the prose."""

from __future__ import annotations

from sim.cube import ai, query
from sim.cube.bodymap import BodyMap
from sim.cube.fused import run_fused
from sim.cube.tissues import ARTERY_TISSUE, ORAL_TISSUE


def test_gcn_trains_and_gives_roles() -> None:
    roles = ai.learn_roles()
    assert "Fn" in roles and roles["Fn"]["role"] in {"keystone", "connector", "frontier", "commensal"}
    assert -1.0 <= ai.embedding_affinity("Fn", "Pg") <= 1.0
    assert 0.0 <= ai.gcn_r2() <= 1.0                          # real (weak) regression score


def test_active_coaggregations_are_realised_and_ranked() -> None:
    eng, _ = run_fused(ORAL_TISSUE, dim=18, steps=80, sample_every=80)
    co = ai.active_coaggregations(eng, top=6)
    assert co and all(e["copresence"] > 0 for e in co)        # only co-present edges
    assert co == sorted(co, key=lambda e: -e["weight"])       # ranked by learned weight


def test_binding_curated_and_fallback() -> None:
    fn_sa = ai.predict_binding("Fn", "Sa")
    assert fn_sa["basis"] == "curated structure" and "RadD" in fn_sa["adhesin"]
    unknown = ai.predict_binding("Lb", "Td")
    assert unknown["basis"] == "embedding-heuristic" and "pending" in unknown["note"]


def test_digest_and_grounded_answer_cite_real_numbers() -> None:
    eng, _ = run_fused(ORAL_TISSUE, dim=18, steps=90, sample_every=90)
    d = query.state_digest(eng, "oral")
    assert d["psi"]["dysbiosis"] >= 0.0 and "roles_present" in d
    # the function answer must literally contain the live dysbiosis figure (grounded, not invented)
    ans = query.answer("what is the function?", d)
    assert str(d["psi"]["dysbiosis"]) in ans
    # the attachment answer names a real coaggregation partner
    att = query.answer("what attaches to what?", d)
    assert any(e["a"] in att and e["b"] in att for e in d["coaggregations"])


def test_systemic_answer_uses_bloodstream_link() -> None:
    bm = BodyMap()
    bm.place("oral", ORAL_TISSUE, dim=18)
    bm.place("artery", ARTERY_TISSUE, dim=18)
    bm.link("oral", "artery", k=2.5)
    bm.run(80, sample_every=80)
    body = query.digest_bodymap(bm)
    ans = query.answer("does it spread systemically?", body["cubes"]["oral"], body=body)
    assert "oral" in ans and "artery" in ans and "efflux" in ans.lower()

