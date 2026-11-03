"""Read-only learning receipts and assay planning, always checked against the solver."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from sim.cube.journey import parameters,simulate
from sim.cube.learning_models import Network,features,OUTPUTS,ridge_predict,impose_initial,constrain_trajectory
D=Path(__file__).resolve().parents[2]/'data/learning'
STEPS=np.arange(0,121,8)
@lru_cache(maxsize=1)
def assets():
    c=json.loads((D/'checkpoint.json').read_text());r=json.loads((D/'benchmark.json').read_text())
    return c,r,{k:Network.restore(v) for k,v in c['models'].items()}

def summary():
    c,r,_=assets()
    return {k:r[k] for k in ('version','provenance','split_counts','selected','acceptance_passed','acceptance_scope','validation_mae','training','calibration_radius','results','protocol_sha256','checkpoint_sha256')}

def predict(p):
    c,r,models=assets();p=parameters(p);name=c['selected'];out={};spreads={}
    for branch in ('baseline','intervention'):
        x=features(p,branch=='intervention',STEPS)
        predictions=[ridge_predict(np.array(c['ridge']),x)] if name=='ridge' else [models[f'{name}_{s}'].forward(x) for s in c['protocol']['seeds']]
        predictions=np.array([impose_initial(y,STEPS) for y in predictions]);out[branch]=constrain_trajectory(predictions.mean(0))[0];spreads[branch]=predictions.std(0)
    reasons=[k for k,(lo,hi) in c['support'].items() if not lo<=p[k]<=hi]
    if not r['acceptance_passed']:reasons.append('benchmark accuracy gate failed')
    # Cumulative absorbed/cleared mass must never decrease. Reject a violating trajectory.
    if any((np.diff(y[:,[4,5]],axis=0)<-1e-8).any() for y in out.values()):reasons.append('predicted cumulative fate decreases')
    return out,spreads,reasons

def inspect(data):
    p=parameters(data);reference=simulate(p);c,r,_=assets();out,spread,reasons=predict(p)
    q=c['calibration_radius'];rows={}
    for b,y in out.items():
        rows[b]={}
        for j,k in enumerate(OUTPUTS):
            rows[b][k]={'prediction':float(y[-1,j]),'solver':reference[b][-1][k],
               'ensemble_sd':float(spread[b][-1,j]),
               'synthetic_90pct_band':[max(0,float(y[-1,j]-q)),min(1,float(y[-1,j]+q))] if not reasons else None}
        rows[b]['blood_auc']={'prediction':float(np.sum((y[1:,2]+y[:-1,2])*.5*np.diff(STEPS*10))),
            'solver':float(sum((reference[b][int(a)]['blood']+reference[b][int(z)]['blood'])*.5*(z-a)*10 for a,z in zip(STEPS,STEPS[1:]))),
            'unit':'fraction × seconds; 16-point trapezoidal integration, no calibrated AUC interval'}
    return dict(run_id=reference['run_id'],version=r['version'],parameters=p,provenance=r['provenance'],selected=c['selected'],
      status='abstain — use full solver' if reasons else 'synthetic in-domain comparison',abstention_reasons=reasons,
      constraint='Explicit conservative correction enforces monotone cumulative uptake/clearance and mass conservation; this is not learned biology. Ensemble SD describes uncorrected model disagreement.',
      calibration_scope='Joint 90% split-conformal envelope over 8 outputs, both branches and 16 sampled times. Synthetic exchangeability only; not biological confidence.',
      predicted_frames={b:[dict(step=int(t),**dict(zip(OUTPUTS,v.tolist()))) for t,v in zip(STEPS,y)] for b,y in out.items()},
      final=rows,scope='Surrogate learns the assumed simulator. HA molecular weight is excluded. No measured outcomes were used. Ensemble SD is disagreement, not a confidence interval.')

def information_gain(means,noise,seed):
    """Monte Carlo mutual information for equiprobable Gaussian assay hypotheses."""
    rng=np.random.default_rng(seed);means=np.asarray(means);z=means[:,None]+rng.normal(0,noise,(len(means),1000))
    ll=-.5*((z[:,:,None]-means[None,None,:])/noise)**2
    maxll=ll.max(2);mixture=maxll+np.log(np.exp(ll-maxll[:,:,None]).mean(2))
    own=-.5*((z-means[:,None])/noise)**2;v=own-mixture
    return max(0,float(v.mean())),float(v.std()/np.sqrt(v.size))

def plan(data):
    p=parameters(data);hypotheses=[('Barrier-support hypothesis',p),
      ('Faster-clearance alternative',{**p,'strength':0,'clearance':min(2,p['clearance']*1.5)}),
      ('Slower-transport alternative',{**p,'strength':0,'velocity':max(.1,p['velocity']*.65)})]
    simulations=[simulate(h) for _,h in hypotheses];surrogates=[predict(h) for _,h in hypotheses];candidates=[]
    for step in (32,64,120):
        for key,cost in [('blood',1.),('integrity',1.5),('plaque',3.)]:
            means=[s['intervention'][step][key] for s in simulations];j=OUTPUTS.index(key);i=int(step//8)
            approx=[s[0]['intervention'][i,j] for s in surrogates]
            eig,se=information_gain(means,.03,20261030+step+j);aeig,_=information_gain(approx,.03,20261030+step+j)
            candidates.append(dict(measurement=key,seconds=step*10,noise_sd=.03,relative_cost=cost,
              expected_information_nats=eig,monte_carlo_se=se,score=eig/cost,
              surrogate_information_nats=aeig,solver_hypothesis_means=means,
              surrogate_usable=all(not s[2] for s in surrogates)))
    candidates.sort(key=lambda a:a['score'],reverse=True)
    return dict(run_id=simulations[0]['run_id'],provenance='Hypothetical assay design; every ranking confirmed with the full simulator.',
      hypotheses=[dict(name=n,parameters=h,run_id=s['run_id'],prior=1/3) for (n,h),s in zip(hypotheses,simulations)],
      assumptions='Three illustrative, unfitted hypotheses; equal priors; independent Gaussian measurement noise SD 0.03 in normalized units; costs are illustrative relative effort. Integrity is a model index requiring an experimental measurement mapping. Timing is simulation time, not established human kinetics.',
      method='Mutual information / relative cost, Monte Carlo (3000 draws per assay). High scores separate these assumed hypotheses; they do not establish biological causality.',
      candidates=candidates,measurements_needed=['Paired barrier permeability and blood-associated label time courses','Separate vesicle membrane and cargo labels; distinguish free label and intact vesicles','Independent tissue/plaque uptake and clearance measurements with controls'])
