"""Prospective second-stage benchmark. Initial test retired after discovering a constraint defect.

Reuses frozen fitted weights, selects on the original validation partition,
calibrates and tests on entirely fresh parameter draws. Does not refit weights.
"""
import sys,json,hashlib,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sim.cube.learning_models import Network,impose_initial,constrain_trajectory,ridge_predict
from scripts.train_crossing import stats,data,STEPS
import scripts.train_crossing as training
D=ROOT/'data/learning'
def save(n,v):(D/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
    old=D/'stage1';old.mkdir(exist_ok=True)
    for n in ['checkpoint.json','benchmark.json','protocol.json','dataset_manifest.json']:
        if not (old/n).exists():shutil.copy2(D/n,old/n)
    c=json.loads((old/'checkpoint.json').read_text());initial=json.loads((old/'benchmark.json').read_text())
    p=c['protocol'];p.update(version='crossing-learning-1.1',revision='The first test exposed cumulative-fate reversals. Its results are retired diagnostics. Fixed conservative postprocessing is evaluated on fresh calibration/test/OOD draws; no weight retraining.',
       prospective_seed=20261031,constraint='Monotone cumulative uptake/clearance, bounded by available mass; proportional redistribution of remaining mass. Report raw violations and correction magnitude.',
       selection='Choose family by original validation MAE after fixed postprocessing; freeze selection and calibration before opening fresh test.')
    p['acceptance']['cumulative_monotonicity_violation_fraction_max']=0
    save('protocol.json',p) # Freeze before generating the new test.
    training.rng=np.random.default_rng(p['prospective_seed'])
    fresh={s:training.sample(p['splits'][s],s=='ood') for s in ('calibration','test','ood')}
    original=json.loads((old/'dataset_manifest.json').read_text());original['parameters'].update(fresh)
    dataset=np.load(D/'synthetic_journeys.npz');sets={s:(dataset[s+'_x'],dataset[s+'_y']) for s in ('train','validation')}
    sets.update({s:data(ps) for s,ps in fresh.items()})
    models={n:Network.restore(v) for n,v in c['models'].items()}
    def raw(name,x):
        flat=x.reshape(-1,7)
        if name=='ridge':y=ridge_predict(np.array(c['ridge']),flat)
        elif name.endswith('_ensemble'):
            kind=name.split('_')[0];y=np.mean([models[f'{kind}_{s}'].forward(flat) for s in p['seeds']],0)
        else:y=models[name].forward(flat)
        return impose_initial(y,np.tile(STEPS,len(x)*2)).reshape(len(x),2,len(STEPS),8)
    def corrected(y):
        z=y.copy();maxcor=0
        for i in range(len(y)):
            for b in (0,1):z[i,b],delta=constrain_trajectory(y[i,b]);maxcor=max(maxcor,delta)
        return z,maxcor
    names=list(models)+['mlp_ensemble','mps_ensemble','ridge']
    xv,yv=sets['validation'];validation={n:float(np.abs(corrected(raw(n,xv))[0]-yv).mean()) for n in names[-3:]}
    selected=min(validation,key=validation.get);xc,yc=sets['calibration'];cal,_=corrected(raw(selected,xc))
    errors=np.abs(cal-yc).max((1,2,3));q=float(np.sort(errors)[int(np.ceil((len(errors)+1)*.9))-1])
    c.update(protocol=p,selected=selected.replace('_ensemble',''),validation_mae={k.replace('_ensemble',''):v for k,v in validation.items()},calibration_radius=q,constraints=True)
    save('checkpoint.json',c) # Choice, weights, constraints and interval frozen before scoring test.
    results={}
    for s in ('test','ood'):
        x,y=sets[s];rows={}
        for n in names:
            z=raw(n,x);zz,correction=corrected(z);row=stats(y,zz)
            row.update(max_constraint_correction=correction,raw_cumulative_monotonicity_violation_fraction=stats(y,z)['cumulative_monotonicity_violation_fraction']);rows[n]=row
        z,_=corrected(raw(selected,x));rows['selected_90pct_envelope_coverage']=float((np.abs(z-y).max((1,2,3))<=q).mean());results[s]=rows
    r=results['test'][selected];g=p['acceptance'];passed=r['mae']<=g['mean_absolute_error_max'] and r['p95_trajectory_max_error']<=g['p95_trajectory_max_error_max'] and r['mass_error']<=g['mass_error_max'] and r['cumulative_monotonicity_violation_fraction']==0
    np.savez_compressed(D/'synthetic_journeys.npz',steps=STEPS,**{s+'_'+k:a for s,xy in sets.items() for k,a in zip(('x','y'),xy)})
    original.update(ids={s:[hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()[:20] for v in ps] for s,ps in original['parameters'].items()},protocol_sha256=sha(D/'protocol.json'),dataset_sha256=sha(D/'synthetic_journeys.npz'),retired_test_ids=json.loads((old/'dataset_manifest.json').read_text())['ids']['test'])
    save('dataset_manifest.json',original)
    report={**initial,'version':p['version'],'selected':c['selected'],'acceptance_passed':passed,'validation_mae':c['validation_mae'],'calibration_radius':q,'results':results,
      'revision':p['revision'],'constraint':p['constraint'],'protocol_sha256':sha(D/'protocol.json'),'checkpoint_sha256':sha(D/'checkpoint.json'),'dataset_sha256':sha(D/'synthetic_journeys.npz')}
    save('benchmark.json',report);print(json.dumps({'selected':selected,'test':r,'coverage':results['test']['selected_90pct_envelope_coverage'],'gate':passed},indent=2))
