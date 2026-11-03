"""Illustrative, dimensionless mouth-to-circulation counterfactual model.

This is a separate hypothesis demonstrator, not a calibrated extension of the
200 micrometer diffusion solver. No product efficacy is encoded or asserted.
"""
import hashlib
import json
import math

REGIONS = ('oral', 'barrier', 'tissue', 'circulation')
REGION_INFO = {
    'oral': 'Oral surface: the initial synthetic tracer reservoir. Decorative bacteria are illustrative.',
    'barrier': 'Epithelial interface: model integrity controls tracer transfer into tissue.',
    'tissue': 'Subepithelial tissue: a single well-mixed model compartment, illustrated as matrix and cells.',
    'circulation': 'Circulation: a receiving compartment with first-order clearance; illustrated red cells do not set the kinetics.',
}
SCOPE = ('Illustrative three-compartment hypothesis model. All quantities and time are dimensionless; '
         'geometry is conceptual, not microscopy reconstruction. Repair strength is an assumed parameter, '
         'not a drug dose or measured treatment effect. No sensor data or clinical prediction.')
VERSION = 'journey-1.0'

def parameters(data):
    if not isinstance(data, dict) or set(data) - {'strength', 'start'}:
        raise ValueError('Expected strength and start only.')
    strength, start = data.get('strength', .7), data.get('start', 20)
    if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Assumed repair strength must be from 0 to 1.')
    if type(start) is not int or not 0 <= start <= 80:
        raise ValueError('Intervention start must be an integer from 0 to 80.')
    return {'strength': float(strength), 'start': start}

def _branch(strength, start):
    oral, tissue, blood, cleared, integrity, inflammation = 1., 0., 0., 0., .65, .35
    cumulative = 0.
    frames = []
    for t in range(121):
        frames.append(dict(step=t, oral=oral, tissue=tissue, blood=blood, cleared=cleared,
                           integrity=integrity, inflammation=inflammation, cumulative=cumulative))
        # Twenty small explicit updates per model step; use previous-state fluxes.
        for _ in range(20):
            dt = .05
            repair = strength if t >= start else 0.
            permeability = .008 + .06 * (1. - integrity)
            into_tissue = permeability * oral * dt
            into_blood = .04 * tissue * dt
            removal = .015 * blood * dt
            di = (.16*tissue*(1.-inflammation) - .025*inflammation) * dt
            db = (.012*(1.-integrity) - .04*inflammation*integrity
                  + .045*repair*(1.-integrity)) * dt
            oral -= into_tissue
            tissue += into_tissue - into_blood
            blood += into_blood - removal
            cleared += removal
            cumulative += into_blood
            integrity = min(1., max(0., integrity + db))
            inflammation = min(1., max(0., inflammation + di))
    return frames

def simulate(data=None):
    p = parameters({} if data is None else data)
    baseline = _branch(0., p['start'])
    intervention = _branch(p['strength'], p['start'])
    frames = baseline + intervention
    mass_error = max(abs(f['oral']+f['tissue']+f['blood']+f['cleared']-1.) for f in frames)
    nonnegative = all(min(f[k] for k in ('oral','tissue','blood','cleared')) >= 0 for f in frames)
    same_before = all(baseline[t] == intervention[t] for t in range(p['start']+1))
    run_id = hashlib.sha256(json.dumps({'version':VERSION, **p},sort_keys=True).encode()).hexdigest()[:16]
    return {'version': VERSION, 'run_id': run_id, 'parameters': p, 'scope': SCOPE,
            'baseline': baseline, 'intervention': intervention,
            'checks': {'mass_error': mass_error, 'mass_conserved': mass_error < 1e-12,
                       'nonnegative': nonnegative, 'identical_before_intervention': same_before},
            'microscopy': {'status': 'source_not_verified', 'used_in_geometry': False}}

def context(data):
    if not isinstance(data, dict) or set(data) != {'parameters','region','step','branch','movement'}:
        raise ValueError('Invalid journey context.')
    if data['region'] not in REGIONS or data['branch'] not in ('baseline','intervention'):
        raise ValueError('Invalid region or branch.')
    if data['movement'] not in ('walk','fly') or type(data['step']) is not int or not 0 <= data['step'] <= 120:
        raise ValueError('Invalid movement or step.')
    run = simulate(data['parameters'])
    t = data['step']
    return run, {'region': data['region'], 'description': REGION_INFO[data['region']],
                 'step': t, 'branch': data['branch'], 'movement': data['movement'],
                 'baseline': run['baseline'][t], 'intervention': run['intervention'][t],
                 'run_id': run['run_id'], 'parameters': run['parameters'], 'scope': SCOPE,
                 'checks': run['checks'], 'microscopy': run['microscopy']}
