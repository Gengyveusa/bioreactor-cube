"""The Crossing v2: a defined OMV, spatial transport and falsifiable HA hypotheses."""
import hashlib,json,math
from pathlib import Path
from functools import lru_cache
from sim.cube.journey_spatial import branch,metrics,diffusivity,STEP_SECONDS

VERSION='journey-2.0'
REGIONS=('oral','barrier','tissue','circulation','plaque')
REGION_INFO={
 'oral':'Oral reservoir releases one unit of ensemble probability for a defined P. gingivalis OMV exemplar.',
 'barrier':'Prescribed influx depends on a bounded integrity index and an assumed HA-CD44 repair coupling.',
 'tissue':'Twenty tissue voxels resolve diffusion, directed transport and competing uptake over an assumed 100 µm path.',
 'circulation':'A receiving blood pool competes between clearance and delivery to a pre-existing plaque.',
 'plaque':'Pre-existing carotid plaque with assumed capture and release. No plaque growth or clinical risk is predicted.'}
SCOPE=('Exploratory hypothesis model, not a clinical prediction. The transport clock assumes 10 seconds per step; '
 'biological rates, tissue geometry, flow and HA response are uncalibrated. A hero vesicle follows a narrative '
 'route conditioned on arrival; computed population fractions include competing fates. The spatial solver '
 'does not reconstruct a patient artery or the whole circulatory system. HA size is recorded as context; '
 'no efficacy-size relationship is imposed. The literature pathway map is broader than the kinetic model.')
DEFAULTS={'strength':.7,'start':20,'diameter_nm':168.2,'ha_kda':1000,'cd44':True,
          'coupling':1.,'hindrance':.25,'velocity':1.,'clearance':1.,'plaque_capture':1.}
LIMITS={'strength':(0,1),'diameter_nm':(50,300),'ha_kda':(1,2000),'coupling':(0,1),
        'hindrance':(.05,1),'velocity':(.1,2),'clearance':(.25,2),'plaque_capture':(0,2)}
def parameters(data):
    if not isinstance(data,dict) or set(data)-set(DEFAULTS):raise ValueError('Unknown journey parameter')
    p={**DEFAULTS,**data}
    for k,(lo,hi) in LIMITS.items():
        if type(p[k]) not in (int,float) or not math.isfinite(p[k]) or not lo<=p[k]<=hi:raise ValueError('Invalid '+k)
        p[k]=float(p[k])
    if type(p['start']) is not int or not 0<=p['start']<=80:raise ValueError('Invalid start')
    if type(p['cd44']) is not bool:raise ValueError('CD44 must be boolean')
    return p

@lru_cache(maxsize=64)
def _cached(encoded):
    p=json.loads(encoded)
    baseline=branch(p,False);intervention=branch(p,True)
    frames=baseline+intervention
    keys=('oral','tissue','blood','plaque','cleared','uptake')
    mass_error=max(abs(sum(f[k] for k in keys)-1) for f in frames)
    run_id=hashlib.sha256((VERSION+encoded).encode()).hexdigest()[:16]
    return {'version':VERSION,'run_id':run_id,'parameters':p,'scope':SCOPE,
      'baseline':baseline,'intervention':intervention,
      'metrics':{'baseline':metrics(baseline),'intervention':metrics(intervention)},
      'physics':{'free_diffusivity_um2_s':diffusivity(p['diameter_nm']),
         'effective_diffusivity_um2_s':diffusivity(p['diameter_nm'])*p['hindrance'],
         'temperature_K':310,'viscosity_mPa_s':1,'step_seconds':STEP_SECONDS,
         'tissue_length_um':100,'voxel_width_um':5,'velocity_um_s':p['velocity'],
         'source':'sim.cube.physical_transport.diffuse','status':'assumed medium and geometry'},
      'checks':{'mass_error':mass_error,'mass_conserved':mass_error<1e-10,
         'nonnegative':all(min(f[k] for k in keys)>=-1e-14 for f in frames),
         'identical_before_intervention':baseline[:p['start']+1]==intervention[:p['start']+1]},
      'microscopy':{'status':'published carotid source plus unverified oral archive',
        'used_in_geometry':'carotid 2D reference plane and manually traced outline; inferred extrusion depth',
        'not_measured':['patient-specific 3D geometry','OMV cargo copy number','oral HA efficacy']}}

def simulate(data=None):
    # Return a copy so agent actions cannot poison cached runs.
    return json.loads(json.dumps(_cached(json.dumps(parameters({} if data is None else data),sort_keys=True))))

def challenge(data=None):
    p=parameters({} if data is None else data);base=simulate(p)
    rows=[]
    variants=[('No HA coupling','coupling',0),('CD44 unavailable','cd44',False),
       ('Slow clearance','clearance',.25),('Fast clearance','clearance',2),
       ('Low tissue mobility','velocity',.1),('High tissue mobility','velocity',2),
       ('Weak plaque capture','plaque_capture',.25),('Strong plaque capture','plaque_capture',2)]
    for label,key,value in variants:
        r=simulate({**p,key:value});a,b=r['metrics']['baseline'],r['metrics']['intervention']
        rows.append({'label':label,'parameter':key,'value':value,'run_id':r['run_id'],
            'blood_auc_delta':b['blood_auc']-a['blood_auc'],
            'plaque_retained_delta':b['plaque_retained']-a['plaque_retained'],
            'final_inflammation_delta':b['final_inflammation']-a['final_inflammation'],
            'checks':r['checks']})
    return {'run_id':base['run_id'],'method':'One parameter at a time; scenario range, not a confidence interval.',
      'rows':rows,'questions':[
       'Does the benefit vanish when the assumed HA-to-barrier coupling is zero?',
       'Does the conclusion hold for cumulative exposure and plaque retention, not just the last frame?',
       'Could uptake, clearance or an altered transport time explain the result?',
       'Can an intact OMV be distinguished from free dye or released cargo at the destination?'],
      'measurements':['Label both OMV membrane and cargo; include free-dye, disrupted-vesicle and vehicle controls.',
       'Measure size distribution, zeta potential, cargo activity and endotoxin in the actual preparation.',
       'Measure epithelial permeability and HA molecular-weight distribution with CD44 perturbation.',
       'Quantify time-resolved tissue, blood and plaque signal with uptake and clearance controls.']}

def context(data):
    expected={'parameters','region','step','branch','movement'}
    if not isinstance(data,dict) or set(data)!=expected:raise ValueError('Invalid journey context')
    if data['region'] not in REGIONS or data['branch'] not in ('baseline','intervention'):raise ValueError('Invalid region')
    if data['movement'] not in ('walk','fly') or type(data['step']) is not int or not 0<=data['step']<=120:raise ValueError('Invalid time/movement')
    run=simulate(data['parameters']);t=data['step']
    return run,{'region':data['region'],'description':REGION_INFO[data['region']],
      'step':t,'branch':data['branch'],'movement':data['movement'],
      'baseline':run['baseline'][t],'intervention':run['intervention'][t],
      'run_id':run['run_id'],'parameters':run['parameters'],'scope':SCOPE,
      'checks':run['checks'],'metrics':run['metrics'],'physics':run['physics'],'microscopy':run['microscopy']}

def evidence():
    return json.loads((Path(__file__).resolve().parents[2]/'data/journey/evidence.json').read_text())
