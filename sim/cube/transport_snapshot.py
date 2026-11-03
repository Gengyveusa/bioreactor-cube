"""Passive lactate diffusion from an exported Cube state; biological reactions paused.

python -m sim.cube.transport_snapshot output/fasolati-50 --seconds 1 --out output/passive-001
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sim.cube.physical_transport import TransportDomain,diffuse
from sim.cube.transport_benchmark import SOURCE

def run(source,out,seconds=1.):
    source=Path(source);root=Path(out)
    m=json.loads((source/'manifest.json').read_text())
    raw=(source/'cube-fields.npz').read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if digest != m['artifacts']['cube-fields.npz']:
        raise ValueError('Source voxel archive does not match its recorded hash')
    with np.load(source/'cube-fields.npz') as a: initial=a['lactate']
    if list(initial.shape)!=m['shape']:
        raise ValueError('Source lactate geometry does not match the manifest')
    lengths=tuple(n*m['cell_size_um'] for n in initial.shape)
    domain=TransportDomain(initial.shape,lengths)
    final,receipt=diffuse(initial,domain,480.,seconds)
    root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(root/'passive-lactate.npz',initial=initial,final=final)
    receipt.update({
        'source_archive_sha256':digest,'source_manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
        'source_upstream_commit':m['upstream_commit'],'field':'lactate',
        'concentration_units':'legacy_model_units; no concentration calibration',
        'geometry_status':'nominal spacing inherited from source; not specimen calibrated',
        'boundaries':'closed x,z; periodic y; original face forcing disabled',
        'reactions':'paused','whole_cube_physical_clock_enabled':False,
        'diffusivity_source':SOURCE,
        'limits':['Passive tracer experiment on a frozen Cube field',
                  'Not a live-biofilm diffusion measurement',
                  'Not a clinical clearance or Gengyve-response prediction'],
        'code_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ['transport_snapshot.py','physical_transport.py']},
        'output_sha256':hashlib.sha256((root/'passive-lactate.npz').read_bytes()).hexdigest(),
    })
    (root/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source')
    p.add_argument('--out',default='output/passive-001')
    p.add_argument('--seconds',type=float,default=1.)
    a=p.parse_args();print(json.dumps(run(a.source,a.out,a.seconds),indent=2))

