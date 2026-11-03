"""Reproduce the frozen synthetic benchmark: OPENBLAS_NUM_THREADS=1 python3 scripts/train_crossing.py"""
import sys,json,time,hashlib,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sim.cube.journey import parameters
from sim.cube.journey_spatial import branch
from sim.cube.learning_teacher import generate
from sim.cube.learning_models import Network,features,OUTPUTS,FEATURES,poly,ridge_predict,impose_initial
D=ROOT/'data/learning';P=json.loads((D/'protocol.json').read_text());STEPS=np.arange(0,121,8)
def save(name,data): (D/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def sample(n,ood=False):
    out=[]
    for i in range(n):
        out.append(parameters(dict(strength=float(rng.uniform(0,1)),start=int(rng.integers(0,61)),
          diameter_nm=float(rng.uniform(80,250)),hindrance=float(rng.uniform(.1,.8)),
          velocity=float(rng.uniform(.1,.29) if ood and i%2 else rng.uniform(1.81,2) if ood else rng.uniform(.3,1.8)),
          clearance=float(rng.uniform(.4,1.8)),plaque_capture=float(rng.uniform(.1,1.8)),
          coupling=float(rng.uniform(0,1)),cd44=bool(rng.random()>.12),ha_kda=float(rng.uniform(1,2000)))))
    return out

def data(ps):
    y=generate(ps,tuple(STEPS));x=np.array([[features(p,b,STEPS) for b in (False,True)] for p in ps]);return x,y

def predict(m,x):return impose_initial(m.forward(x.reshape(-1,7)),np.tile(STEPS,len(x)*2)).reshape(len(x),2,len(STEPS),8)
def stats(y,z):
    e=np.abs(z-y)
    auc=lambda a:np.sum((a[:,:,1:,2]+a[:,:,:-1,2])*.5*np.diff(STEPS*10),axis=2)
    return {'mae':float(e.mean()),'per_output_mae':dict(zip(OUTPUTS,e.mean((0,1,2)).tolist())),
      'p95_trajectory_max_error':float(np.quantile(e.max((1,2,3)),.95)),
      'mass_error':float(np.abs(z[...,:6].sum(-1)-1).max()),'nonnegative':bool((z[...,:6]>=0).all()),
      'blood_auc_mae_fraction_seconds':float(np.abs(auc(y)-auc(z)).mean()),
      'final_plaque_mae':float(e[:,:,-1,3].mean()),
      'cumulative_monotonicity_violation_fraction':float((np.diff(z[...,[4,5]],axis=2)<-1e-8).any((1,2,3)).mean())}

if __name__=='__main__':
    started=time.perf_counter();rng=np.random.default_rng(P['seed'])
    ps={s:sample(n,s=='ood') for s,n in P['splits'].items()}
    # Verify vectorization before fitting, including domain corners and exact zero repair.
    verify=[parameters({}),parameters(dict(diameter_nm=50,hindrance=1,velocity=2)),parameters(dict(velocity=.1,cd44=False))]
    actual=generate(verify,tuple(STEPS));error=0.
    for i,p in enumerate(verify):
        for b in (0,1):
            ref=np.array([[f[k] for k in OUTPUTS] for f in branch(p,bool(b))])[STEPS]
            error=max(error,float(np.abs(ref-actual[i,b]).max()))
    if error>1e-11:raise RuntimeError(f'Teacher mismatch {error}')
    print('Teacher parity',error,flush=True)
    sets={s:data(v) for s,v in ps.items()}
    ids={s:[hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:20] for p in v] for s,v in ps.items()}
    np.savez_compressed(D/'synthetic_journeys.npz',steps=STEPS,**{s+'_'+k:a for s,xy in sets.items() for k,a in zip(('x','y'),xy)})
    save('dataset_manifest.json',dict(parameters=ps,ids=ids,provenance=P['provenance'],outputs=OUTPUTS,features=FEATURES,teacher_parity_max_error=error,protocol_sha256=sha(D/'protocol.json'),dataset_sha256=sha(D/'synthetic_journeys.npz'),teacher_sha256=sha(ROOT/'sim/cube/journey_spatial.py')))
    xt,yt=[a.reshape(-1,a.shape[-1]) for a in sets['train']];xv,yv=sets['validation'];models={};training={}
    configs=[(f'{kind}_{seed}',kind,seed,9,None,True) for kind in ('mlp','mps') for seed in P['seeds']]
    configs += [('mps_rank4','mps',11,4,None,True),('mps_reverse','mps',11,9,list(reversed(range(7))),True),('mps_softmax','mps',11,9,None,False)]
    for name,kind,seed,rank,order,born in configs:
        t0=time.perf_counter();m=Network(kind,seed,rank,order,born);rr=np.random.default_rng(seed)
        mm=[np.zeros_like(w) for w in m.w];vv=[w.copy() for w in mm];best=float('inf');trace=[]
        for step in range(1,P['training_steps']+1):
            ix=rr.integers(0,len(xt),P['batch_size']);loss,grad=m.gradient(xt[ix],yt[ix])
            norm=np.sqrt(sum(np.sum(g*g) for g in grad));scale=min(1,1/(norm+1e-12))
            for j,g in enumerate(grad):
                g=g*scale;mm[j]=.9*mm[j]+.1*g;vv[j]=.999*vv[j]+.001*g*g
                m.w[j]-=P['learning_rate']*(mm[j]/(1-.9**step))/(np.sqrt(vv[j]/(1-.999**step))+1e-8)
            if step%P['validation_every']==0:
                err=float(np.abs(predict(m,xv)-yv).mean());trace.append([step,err])
                if err<best:best=err;checkpoint=m.serial();selected_step=step
        models[name]=checkpoint;training[name]=dict(seconds=time.perf_counter()-t0,parameters=sum(w.size for w in m.w),validation_mae=best,selected_step=selected_step,trace=trace)
        print(name,'val MAE',round(best,5),'seconds',round(training[name]['seconds'],1),flush=True)
    a=poly(xt);ridge=np.linalg.solve(a.T@a+np.eye(a.shape[1])*.01,a.T@yt)
    def ridge_out(x):return impose_initial(ridge_predict(ridge,x.reshape(-1,7)),np.tile(STEPS,len(x)*2)).reshape(len(x),2,len(STEPS),8)
    family={}
    def ensemble(kind,x):return np.mean([predict(Network.restore(models[f'{kind}_{seed}']),x) for seed in P['seeds']],axis=0)
    for kind in ('mlp','mps'):family[kind]=float(np.abs(ensemble(kind,xv)-yv).mean())
    family['ridge']=float(np.abs(ridge_out(xv)-yv).mean());chosen=min(family,key=family.get)
    xc,yc=sets['calibration'];cal=ensemble(chosen,xc) if chosen!='ridge' else ridge_out(xc)
    errors=np.abs(cal-yc).max((1,2,3));k=min(len(errors),int(np.ceil((len(errors)+1)*.9)));q=float(np.sort(errors)[k-1])
    checkpoint=dict(protocol=P,models=models,ridge=ridge.tolist(),selected=chosen,validation_mae=family,calibration_radius=q,
      support={'diameter_nm':[80,250],'hindrance':[.1,.8],'velocity':[.3,1.8],'clearance':[.4,1.8],'plaque_capture':[.1,1.8],'start':[0,60]})
    save('checkpoint.json',checkpoint) # Frozen before test scoring.
    results={}
    for split in ('test','ood'):
        x,y=sets[split];r={name:stats(y,predict(Network.restore(m),x)) for name,m in models.items()}
        for kind in ('mlp','mps'):r[kind+'_ensemble']=stats(y,ensemble(kind,x))
        r['ridge']=stats(y,ridge_out(x));z=ensemble(chosen,x) if chosen!='ridge' else ridge_out(x)
        r['selected_90pct_envelope_coverage']=float((np.abs(z-y).max((1,2,3))<=q).mean());results[split]=r
    selected=results['test'][chosen+'_ensemble' if chosen!='ridge' else 'ridge'];g=P['acceptance']
    accepted=selected['mae']<=g['mean_absolute_error_max'] and selected['p95_trajectory_max_error']<=g['p95_trajectory_max_error_max'] and selected['mass_error']<=g['mass_error_max']
    save('benchmark.json',dict(version=P['version'],provenance=P['provenance'],split_counts=P['splits'],teacher_parity_max_error=error,
      selected=chosen,acceptance_passed=accepted,acceptance_scope='Synthetic in-domain accuracy gate only; always compare with full solver. OOD abstention is mandatory.',
      validation_mae=family,training=training,calibration_radius=q,results=results,
      protocol_sha256=sha(D/'protocol.json'),checkpoint_sha256=sha(D/'checkpoint.json'),dataset_sha256=sha(D/'synthetic_journeys.npz'),
      environment={'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform()},total_seconds=time.perf_counter()-started))
    print('Selected',chosen,'test',selected,'gate',accepted,'calibration radius',q,flush=True)
