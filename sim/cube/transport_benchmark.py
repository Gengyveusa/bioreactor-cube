"""Fixed-domain numerical benchmark; criteria are declared before execution.

python -m sim.cube.transport_benchmark --out output/transport-001
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sim.cube.physical_transport import TransportDomain, diffuse

CRITERIA={'mass_relative_error_max':1e-12,'finest_rmse_max':1e-3,
          'errors_decrease_with_refinement':True,'beats_well_mixed_baseline':True}
SOURCE={
 'url':'https://pubmed.ncbi.nlm.nih.gov/6818937/',
 'title':'Diffusion of sugars and acids in human dental plaque in vitro',
 'reported_sodium_lactate_cm2_s':4.8e-6,'converted_um2_s':480.0,
 'setting':'37 C, glutaraldehyde-fixed human dental plaque; abstract-level verification',
 'limit':'Illustrative transport coefficient for this experiment, not calibration of living Fasolati biofilm or reaction kinetics',
}

def run(out):
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    plan={'criteria':CRITERIA,'dimensions':[10,20,40],'lengths_um':[200,180,160],
          'duration_s':0.8,'diffusivity_um2_s':480.0,
          'initial':'1+0.2*cos(pi*x/Lx)+0.1*cos(2*pi*y/Ly)',
          'boundary_conditions':'zero flux x,z; periodic y; cell-centred finite volumes',
          'baseline':'spatially uniform concentration equal to initial volume mean',
          'source':SOURCE,'concentration_units':'normalized tracer units',
          'scope':'numerical verification, no biological efficacy or sensor validation'}
    encoded=json.dumps(plan,sort_keys=True).encode()
    (root/'benchmark-plan.json').write_bytes(encoded)
    rows=[]
    for n in plan['dimensions']:
        domain=TransportDomain((n,n,n),tuple(plan['lengths_um']))
        x,y,z=np.meshgrid(*[(np.arange(n)+.5)*h for h in domain.spacing_um],indexing='ij')
        mx=.2*np.cos(np.pi*x/200)
        my=.1*np.cos(2*np.pi*y/180)
        initial=1+mx+my
        exact=1+mx*np.exp(-480*(np.pi/200)**2*.8)+my*np.exp(-480*(2*np.pi/180)**2*.8)
        final,receipt=diffuse(initial,domain,480,.8)
        rows.append({'dim':n,'rmse':float(np.sqrt(np.mean((final-exact)**2))),
                     'baseline_rmse':float(np.sqrt(np.mean((1-exact)**2))),**receipt})
        if n==40:
            np.savez_compressed(root/'transport-fields.npz',initial=initial,final=final,exact=exact)
    checks={
      'mass_conserved':all(r['mass_relative_error']<=CRITERIA['mass_relative_error_max'] for r in rows),
      'nonnegative':all(r['minimum']>=0 for r in rows),
      'finest_error_bounded':rows[-1]['rmse']<=CRITERIA['finest_rmse_max'],
      'refinement_reduces_error':all(a['rmse']>b['rmse'] for a,b in zip(rows,rows[1:])),
      'beats_well_mixed_baseline':all(r['rmse']<r['baseline_rmse'] for r in rows),
    }
    result={'plan_sha256':hashlib.sha256(encoded).hexdigest(),'results':rows,'checks':checks,
            'passed':all(checks.values()),'whole_cube_physical_clock_enabled':False,
            'code_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in ['transport_benchmark.py','physical_transport.py']}}
    (root/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))
    if not result['passed']: raise SystemExit('Transport benchmark failed')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='output/transport-001')
    run(p.parse_args().out)

