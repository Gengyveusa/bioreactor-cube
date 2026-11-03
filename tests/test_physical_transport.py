import numpy as np
import pytest
from sim.cube.physical_transport import TransportDomain, diffuse

@pytest.mark.parametrize('periodic',[(),(1,),(0,1,2)])
def test_mass_and_positivity_without_clipping(periodic):
    domain=TransportDomain((6,7,8),(60.,140.,80.),periodic)
    c=np.zeros(domain.shape);c[0,0,0]=1
    final,r=diffuse(c,domain,480,2)
    assert final.min()>=0
    assert r['mass_relative_error']<1e-12
    assert r['substeps']>1
    assert c.sum()==1 and c[0,0,0]==1

def test_uniform_solution_is_invariant_on_anisotropic_grid():
    d=TransportDomain((4,5,6),(20.,50.,120.))
    c=np.full(d.shape,2.5)
    out,r=diffuse(c,d,480,1)
    np.testing.assert_array_equal(c,out)
    assert r['elapsed_seconds']==1

def test_zero_duration_and_zero_diffusivity_do_not_evolve():
    d=TransportDomain((4,4,4),(40.,40.,40.))
    c=np.arange(64,dtype=float).reshape(d.shape)
    for diffusion,time in [(0,5),(480,0)]:
        out,r=diffuse(c,d,diffusion,time)
        np.testing.assert_array_equal(out,c)
        assert r['substeps']==0

def test_singleton_axis_and_periodic_wrap():
    d=TransportDomain((1,5,1),(10.,50.,10.))
    c=np.zeros(d.shape);c[0,0,0]=1
    out,r=diffuse(c,d,1,.1)
    assert out[0,-1,0]>0 and out[0,1,0]>0
    assert abs(out.sum()-1)<1e-12

@pytest.mark.parametrize('value',[-1,np.nan,np.inf])
def test_invalid_diffusivity_rejected(value):
    d=TransportDomain((2,2,2),(20.,20.,20.))
    with pytest.raises(ValueError):diffuse(np.ones(d.shape),d,value,1)

def test_negative_and_nonfinite_concentrations_rejected():
    d=TransportDomain((2,2,2),(20.,20.,20.))
    for value in [-1,np.nan,np.inf]:
        c=np.ones(d.shape);c[0,0,0]=value
        with pytest.raises(ValueError):diffuse(c,d,1,1)

def test_fixed_domain_refinement_against_analytic_solution():
    errors=[]
    for n in [8,16,32]:
        d=TransportDomain((n,1,1),(100.,10.,10.),())
        x=(np.arange(n)+.5)*100/n
        initial=(1+.2*np.cos(np.pi*x/100))[:,None,None]
        exact=(1+.2*np.cos(np.pi*x/100)*np.exp(-480*(np.pi/100)**2))[:,None,None]
        final,_=diffuse(initial,d,480,1,safety=.5)
        errors.append(np.sqrt(np.mean((final-exact)**2)))
    assert errors[0]>errors[1]>errors[2]
    assert errors[-1]<1e-3

@pytest.mark.parametrize('shape,lengths', [((0,2,2),(1,1,1)),((2.5,2,2),(1,1,1)),((2,2,2),(0,1,1)),((2,2,2),(1,np.nan,1))])
def test_invalid_geometry_rejected(shape,lengths):
    with pytest.raises(ValueError):TransportDomain(shape,lengths)

def test_snapshot_transport_preserves_source_and_its_scope(tmp_path):
    import hashlib,json
    from sim.cube.transport_snapshot import run
    source=tmp_path/'source';source.mkdir()
    field=np.zeros((4,4,4));field[1,1,1]=1
    np.savez_compressed(source/'cube-fields.npz',lactate=field)
    raw=(source/'cube-fields.npz').read_bytes()
    manifest={'shape':[4,4,4],'cell_size_um':20.,'upstream_commit':'test',
              'artifacts':{'cube-fields.npz':hashlib.sha256(raw).hexdigest()}}
    (source/'manifest.json').write_text(json.dumps(manifest))
    receipt=run(source,tmp_path/'result',1.)
    assert receipt['elapsed_seconds']==1.
    assert receipt['reactions']=='paused'
    assert not receipt['whole_cube_physical_clock_enabled']
    assert receipt['mass_relative_error']<1e-12
    assert (source/'cube-fields.npz').read_bytes()==raw

def test_snapshot_rejects_changed_source_archive(tmp_path):
    import json
    from sim.cube.transport_snapshot import run
    source=tmp_path/'source';source.mkdir()
    np.savez_compressed(source/'cube-fields.npz',lactate=np.ones((4,4,4)))
    (source/'manifest.json').write_text(json.dumps({'artifacts':{'cube-fields.npz':'wrong'}}))
    with pytest.raises(ValueError,match='recorded hash'):run(source,tmp_path/'result',1.)

