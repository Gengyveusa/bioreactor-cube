"""Bounded, reproducible numerical experiments over the existing Cube solver."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from sim.cube.physical_transport import TransportDomain, diffuse

BASELINE = '45d5415e5aa148553756c9a737f27b36643eecd4'
DEFAULTS = dict(diffusion_um2_s=480.0, seconds=1.0, boundary='closed', grid=16, seed=7)
LIMITATIONS = [
    'Synthetic initial field in arbitrary concentration units; no live sensor measurements.',
    'Passive diffusion only. Reactions, flow, binding and living tissue dynamics are excluded.',
    'Numerical verification is not biological validation or evidence of Gengyve efficacy.',
    'Sensor calibration and physical experiments, including bioprinting where useful, are future validation steps.',
]

def inspect_baseline():
    return {'baseline': BASELINE, 'registry_targets': 115, 'model_proxies': 7,
            'unimplemented_targets': 108, 'current_field': 'synthetic lactate-like tracer',
            'limits': {'grid': [8,32], 'diffusion_um2_s':[0,2000], 'seconds':[0,10],
                       'boundary':['closed','periodic'], 'max_substeps':2000},
            'limitations': LIMITATIONS, 'prior_whole_engine_scoreboard':
            {'pass':8, 'fail':7, 'blocked':1, 'status':'historical; not rerun by this lab'}}

def validate_plan(args):
    if not isinstance(args, dict) or set(args)-set(DEFAULTS):
        raise ValueError('Expected only diffusion_um2_s, seconds, boundary, grid and seed')
    p = {**DEFAULTS, **args}
    for k, upper in [('diffusion_um2_s',2000),('seconds',10)]:
        v=p[k]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not 0<=v<=upper or not math.isfinite(v):
            raise ValueError(f'{k} must be a finite number from 0 to {upper}')
        p[k]=float(v)
    for k, low, high in [('grid',8,32),('seed',0,2147483647)]:
        if type(p[k]) is not int or not low<=p[k]<=high:
            raise ValueError(f'{k} must be an integer from {low} to {high}')
    if p['boundary'] not in ('closed','periodic'):
        raise ValueError('boundary must be closed or periodic')
    steps=math.ceil(p['seconds']*6*p['diffusion_um2_s']/(200/p['grid'])**2/.9)
    if steps>2000:
        raise ValueError('Experiment exceeds 2000 substep compute budget')
    return p

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def pack(field):
    return {'shape':list(field.shape), 'values':field.ravel().tolist()}

def run_experiment(args):
    p=validate_plan(args); n=p['grid']
    xyz=np.meshgrid(*[(np.arange(n)+.5)/n]*3,indexing='ij')
    # Seed determines the position of a synthetic pulse. Same seed gives same field.
    center=np.random.default_rng(p['seed']).uniform(.3,.7,3)
    initial=.08+np.exp(-sum((v-c)**2 for v,c in zip(xyz,center))/.025)
    domain=TransportDomain((n,n,n),(200.,200.,200.),(0,1,2) if p['boundary']=='periodic' else ())
    final,receipt=diffuse(initial,domain,p['diffusion_um2_s'],p['seconds'])
    metrics={'mass_before':receipt['mass_initial'],'mass_after':receipt['mass_final'],
        'relative_mass_error':receipt['mass_relative_error'],'variance_before':float(initial.var()),
        'variance_after':float(final.var()),'min_after':float(final.min()),'max_after':float(final.max())}
    checks=[{'name':'Mass conserved','passed':metrics['relative_mass_error']<=1e-12,'detail':'Relative error ≤ 1e-12'},
            {'name':'Nonnegative field','passed':bool(np.all(final>=0)),'detail':'Every final voxel ≥ 0'},
            {'name':'Variance does not increase','passed':metrics['variance_after']<=metrics['variance_before']+1e-14,
             'detail':'Diffusion should smooth a closed or periodic field'}]
    fields={'initial':pack(initial),'final':pack(final)}
    code={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
          for name in ('research_lab.py','physical_transport.py')}
    provenance={'baseline':BASELINE,'code_sha256':code,'initial_sha256':digest(fields['initial']),
       'final_sha256':digest(fields['final']),'plan_sha256':digest(p),'numpy_version':np.__version__,
       'concentration_units':'arbitrary simulation units','mass_units':'arbitrary concentration units * um^3',
       'lengths_um':[200,200,200],'source':'generated synthetic pulse',
       'coefficient':'User-selected illustrative parameter; not sensor calibration',
       'scope':'numerical passive transport'}
    rid=digest({'parameters':p,'provenance':provenance})[:16]
    return {'run_id':rid,'mode':'deterministic','parameters':p,'metrics':metrics,**fields,
       'checks':checks,'provenance':provenance,'limitations':LIMITATIONS,
       'receipts':[{'tool':'validate_experiment','result':{'accepted':True,'parameters':p}},
                   {'tool':'run_transport','result':receipt},
                   {'tool':'inspect_verification','result':checks}]}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='output/cube-lab/run.json')
    parser.add_argument('--seconds',type=float,default=1.)
    options=parser.parse_args()
    result=run_experiment({'seconds':options.seconds})
    dest=Path(options.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,allow_nan=False))
    print(json.dumps({'run_id':result['run_id'],'checks':result['checks'],'out':str(dest)}))
