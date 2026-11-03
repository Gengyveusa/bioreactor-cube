"""NumPy-only neural and matrix-product-state surrogates; classical CPU training.

The Born head normalizes squared real amplitudes. It represents competing fate
fractions, not epistemic uncertainty or quantum behavior of biological vesicles.
"""
import numpy as np
from sim.cube.journey_spatial import diffusivity

OUTPUTS=['oral','tissue','blood','plaque','uptake','cleared','integrity','inflammation']
FEATURES=['log_effective_diffusivity','velocity','clearance','plaque_capture','time','active_repair','elapsed_repair']

def features(p, treated, steps):
    t=np.asarray(steps,dtype=float)
    repair=p['strength']*p['coupling']*p['cd44'] if treated else 0.
    active=(t>p['start'])*repair
    # Same numerical inputs for both branches before repair, or when repair is zero.
    return np.column_stack([np.full_like(t,(np.log(diffusivity(p['diameter_nm'])*p['hindrance'])+3)/6),
        np.full_like(t,p['velocity']/2), np.full_like(t,p['clearance']/2),
        np.full_like(t,p['plaque_capture']/2), t/120,active,
        np.where(active>0,np.maximum(t-p['start'],0)/120,0)])

def head(z, born=False):
    if born:
        a=z[:,:6]**2+1e-10; p=a/a.sum(axis=1,keepdims=True)
    else:
        a=np.exp(z[:,:6]-z[:,:6].max(axis=1,keepdims=True));p=a/a.sum(axis=1,keepdims=True)
    q=1/(1+np.exp(-np.clip(z[:,6:],-40,40)))
    return np.concatenate([p,q],axis=1)

def head_grad(z,y,g,born=False):
    p=y[:,:6];v=g[:,:6];center=v-(v*p).sum(axis=1,keepdims=True)
    if born:d=2*z[:,:6]*center/(z[:,:6]**2+1e-10).sum(axis=1,keepdims=True)
    else:d=p*center
    return np.concatenate([d,g[:,6:]*y[:,6:]*(1-y[:,6:])],axis=1)

class Network:
    def __init__(self,kind='mlp',seed=1,rank=9,order=None,born=True):
        self.kind=kind;self.order=list(range(7)) if order is None else list(order)
        self.born=bool(born and kind=='mps');rng=np.random.default_rng(seed)
        if kind=='mlp':
            self.w=[rng.normal(0,np.sqrt(1/a),(a,b)) for a,b in [(7,24),(24,24),(24,8)]]
            self.w += [np.zeros(24),np.zeros(24),np.zeros(8)]
        else:
            dims=[1]+[rank]*6+[8];self.w=[]
            for i in range(7):
                c=rng.normal(0,.09,(dims[i],2,dims[i+1]))
                for j in range(min(dims[i],dims[i+1])):c[j,:,j]+=.65
                self.w.append(c)
            self.w.append(np.array([.4]*6+[0.,0.]))
    def forward(self,x,cache=False):
        if self.kind=='mlp':
            a=np.tanh(x@self.w[0]+self.w[3]);b=np.tanh(a@self.w[1]+self.w[4]);z=b@self.w[2]+self.w[5];memo=(x,a,b)
        else:
            u=x[:,self.order];phi=np.stack([np.cos(np.pi*u/2),np.sin(np.pi*u/2)],axis=2)
            states=[np.ones((len(x),1))]
            for i,c in enumerate(self.w[:-1]):states.append(np.einsum('bi,bj,ijk->bk',states[-1],phi[:,i],c,optimize=True))
            z=states[-1]+self.w[-1];memo=(phi,states)
        y=head(z,self.born)
        return (y,(z,memo)) if cache else y
    def gradient(self,x,target):
        y,(z,m)=self.forward(x,True);loss=np.mean((y-target)**2);g=head_grad(z,y,2*(y-target)/y.size,self.born)
        if self.kind=='mlp':
            x,a,b=m;g2=b.T@g;gb2=g.sum(0);db=(g@self.w[2].T)*(1-b*b)
            g1=a.T@db;gb1=db.sum(0);da=(db@self.w[1].T)*(1-a*a)
            grad=[x.T@da,g1,g2,da.sum(0),gb1,gb2]
        else:
            phi,states=m;grad=[None]*8;grad[-1]=g.sum(0)
            for i in reversed(range(7)):
                grad[i]=np.einsum('bi,bj,bk->ijk',states[i],phi[:,i],g,optimize=True)
                g=np.einsum('bj,ijk,bk->bi',phi[:,i],self.w[i],g,optimize=True)
        return float(loss),grad
    def serial(self):return dict(kind=self.kind,order=self.order,born=self.born,weights=[w.tolist() for w in self.w])
    @classmethod
    def restore(cls,d):
        m=cls(d['kind'],order=d['order'],born=d['born']);m.w=[np.asarray(w) for w in d['weights']];return m

def poly(x):
    return np.column_stack([np.ones(len(x)),x]+[x[:,i]*x[:,j] for i in range(7) for j in range(i,7)])

def ridge_predict(w,x):
    y=poly(x)@w;p=np.maximum(y[:,:6],0);p=(p+1e-10)/(p+1e-10).sum(axis=1,keepdims=True)
    return np.column_stack([p,np.clip(y[:,6:],0,1)])

def impose_initial(y,steps):
    y=y.copy();y[np.asarray(steps)==0]=[1,0,0,0,0,0,.65,.35];return y

def constrain_trajectory(y):
    """Explicit conservative postprocessor, not learned dynamics.

Keep cumulative uptake/clearance monotone, allocating any increments from the
remaining mass budget; redistribute remaining mass proportionally among four
mobile/retained pools. Return maximum correction for transparent receipts.
"""
    z=y.copy();previous=np.zeros(2)
    for i in range(len(z)):
        wanted=np.maximum(z[i,4:6],previous);inc=wanted-previous
        budget=max(0.,1-previous.sum())
        if inc.sum()>budget:inc*=budget/inc.sum()
        z[i,4:6]=previous+inc;previous=z[i,4:6].copy()
        remaining=max(0.,1-previous.sum());mobile=z[i,:4].sum()
        z[i,:4]=z[i,:4]*remaining/mobile if mobile>0 else [remaining,0,0,0]
    return z,float(np.abs(z-y).max())
