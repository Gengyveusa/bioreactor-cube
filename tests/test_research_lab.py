import json
import numpy as np
import pytest
from sim.cube.research_lab import run_experiment, validate_plan, digest

@pytest.mark.parametrize('boundary',['closed','periodic'])
def test_conservation_and_reproducible_output(boundary):
    a=run_experiment({'grid':8,'boundary':boundary})
    b=run_experiment({'grid':8,'boundary':boundary})
    assert a==b
    assert all(c['passed'] for c in a['checks'])
    assert a['metrics']['variance_after']<a['metrics']['variance_before']
    assert a['provenance']['final_sha256']==digest(a['final'])
    json.dumps(a,allow_nan=False)

@pytest.mark.parametrize('args',[{'seconds':0},{'diffusion_um2_s':0}])
def test_identity_when_no_diffusion(args):
    r=run_experiment({'grid':8,**args})
    assert r['initial']==r['final']

@pytest.mark.parametrize('args',[
    {'seconds':float('nan')},{'seconds':float('inf')},{'seconds':-1},{'seconds':11},
    {'seconds':10**400},{'diffusion_um2_s':10**400},
    {'grid':True},{'grid':1000},{'grid':16.5},{'seed':-1}, {'boundary':'open'},
    {'diffusion_um2_s':-2},{'diffusion_um2_s':'480'},{'shell':'rm -rf'},[],None])
def test_reject_invalid_inputs(args):
    with pytest.raises(ValueError): validate_plan(args)

def test_seed_changes_initial_field():
    a=run_experiment({'grid':8,'seed':1});b=run_experiment({'grid':8,'seed':2})
    assert a['provenance']['initial_sha256']!=b['provenance']['initial_sha256']

def test_changed_diffusivity_preserves_initial_field():
    a=run_experiment({'grid':8,'diffusion_um2_s':200})
    b=run_experiment({'grid':8,'diffusion_um2_s':1000})
    assert a['initial']==b['initial']
    assert b['metrics']['variance_after']<a['metrics']['variance_after']
