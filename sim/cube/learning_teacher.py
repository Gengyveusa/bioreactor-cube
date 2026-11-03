"""Batched equivalent of journey_spatial.branch; verified against the reference.

Mass rather than density is stored on the uniform grid. No learned biology.
"""
import numpy as np
from sim.cube.journey_spatial import diffusivity

def generate(parameters,steps=tuple(range(0,121,8))):
    n=len(parameters);p={k:np.repeat([v[k] for v in parameters],2) for k in parameters[0]}
    count=n*2;treated=np.tile([0,1],n);c=np.zeros((count,20));oral=np.ones(count)
    blood=np.zeros(count);plaque=blood.copy();cleared=blood.copy();uptake=blood.copy()
    b=np.full(count,.65);infl=np.full(count,.35);d=diffusivity(p['diameter_nm'])*p['hindrance']
    result=[];dt=.05;fraction=p['velocity']*.1;diff=d*.5/25
    for t in range(121):
        if t in steps:result.append(np.column_stack([oral,c.sum(1),blood,plaque,uptake,cleared,b,infl]))
        if t==120:break
        repair=p['strength']*p['coupling']*p['cd44']*treated*(t>=p['start'])
        for _ in range(20):
            entry=(.008+.06*(1-b))*oral*dt;oral-=entry;c[:,0]+=entry
            flux=(c[:,1:]-c[:,:-1])*diff[:,None]
            c[:,:-1]+=flux;c[:,1:]-=flux
            out=c*fraction[:,None];c-=out;c[:,1:]+=out[:,:-1]
            capture=c*(.003*dt);c-=capture;uptake+=capture.sum(1)
            cb=.015*p['clearance']*blood*dt;tp=.006*p['plaque_capture']*blood*dt;fp=.002*plaque*dt
            blood+=out[:,-1]-cb-tp+fp;plaque+=tp-fp;cleared+=cb
            di=(.16*c.sum(1)*(1-infl)-.025*infl)*dt
            db=(.012*(1-b)-.04*infl*b+.045*repair*(1-b))*dt
            infl+=di;b+=db
    return np.stack(result,axis=1).reshape(n,2,len(steps),8)
