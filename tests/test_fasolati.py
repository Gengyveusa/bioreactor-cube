import numpy as np
import pytest
from sim.cube.grid import CubeGrid
from sim.cube.fasolati import load_panel, read_panel

def test_registry_has_115_unique_targets():
    panel=load_panel()
    assert len(panel)==115
    assert len({x['name'] for x in panel})==115

def test_unsupported_is_null_not_zero():
    records=read_panel(CubeGrid(4,4,4))
    missing=[r for r in records if r['state']=='not_implemented']
    assert len(missing)==108
    assert all(r['value'] is None and r['units'] is None for r in missing)
    assert all(not r['sensor_measured'] and r['evidence_class']=='hypothesis' for r in records)

def test_zero_is_preserved_without_promoting_it_to_a_measurement():
    r=next(r for r in read_panel(CubeGrid(4,4,4)) if r['name']=='L-lactate')
    assert r['value']==0
    assert r['state']=='modeled_proxy'
    assert r['validation_status']=='unconfirmed'

def test_voxel_and_volume_mean_differ_and_reading_does_not_mutate():
    g=CubeGrid(4,4,4)
    g.get_field('lactate')[1,2,3]=64
    before=g.get_field('lactate').copy()
    avg=next(r for r in read_panel(g) if r['name']=='L-lactate')
    voxel=next(r for r in read_panel(g,(1,2,3)) if r['name']=='L-lactate')
    assert avg['value']==1 and voxel['value']==64
    assert voxel['support']['position']==[1,2,3]
    np.testing.assert_array_equal(g.get_field('lactate'),before)

@pytest.mark.parametrize('pos',[(-1,0,0),(4,0,0),(1,2),(1.5,0,0)])
def test_invalid_coordinates_rejected(pos):
    with pytest.raises(ValueError): read_panel(CubeGrid(4,4,4),pos)

def test_nonfinite_model_field_rejected():
    g=CubeGrid(4,4,4)
    g.get_field('urea')[0,0,0]=np.nan
    with pytest.raises(ValueError): read_panel(g)

