"""Small tanh neural network with explicit NumPy reverse-mode derivatives."""
import numpy as np
from scipy.optimize import minimize
from mechanics import energy_gradient,basis

SHAPES=[(2,24),(24,),(24,24),(24,),(24,12),(12,)]

def unpack(w):
    out=[]; k=0
    for shape in SHAPES:
        n=int(np.prod(shape)); out.append(w[k:k+n].reshape(shape)); k+=n
    return out

def initialize(seed):
    rng=np.random.default_rng(seed); a=[]
    for sh in SHAPES:
        a.append(rng.normal(size=sh)*(.35/np.sqrt(sh[0])) if len(sh)==2 else np.zeros(sh))
    return np.concatenate([v.ravel() for v in a])

def forward(w,x):
    a,b,c,d,e,f=unpack(w); h=np.tanh(x@a+b); j=np.tanh(h@c+d)
    return j@e+f,(x,h,j)

def backward(w,cache,g):
    a,b,c,d,e,f=unpack(w); x,h,j=cache
    ge=j.T@g; gf=g.sum(0); gj=(g@e.T)*(1-j*j)
    gc=h.T@gj; gd=gj.sum(0); gh=(gj@c.T)*(1-h*h)
    ga=x.T@gh; gb=gh.sum(0)
    return np.concatenate([z.ravel() for z in [ga,gb,gc,gd,ge,gf]])

def features(p):
    return p/np.array([1.5,.4])

def predict(w,p,n=64):
    c,_=forward(w,features(np.asarray(p))); return c@basis(n).T

def train(seed,kind,labels,targets,physical,maxiter=1800):
    B=basis(); p=np.vstack([labels,physical]); x=features(p); nl=len(labels)
    # Separate means keep the two terms independent of sample count.
    def objective(w):
        c,cache=forward(w,x); t=c@B.T; grad=np.zeros_like(t); loss=0.
        if kind in ('data','hybrid'):
            err=t[:nl]-targets; weight=1. if kind=='data' else .1
            loss+=weight*np.mean(err**2); grad[:nl]+=weight*2*err/err.size
        if kind in ('energy','hybrid'):
            energy,g=energy_gradient(t[nl:],p[nl:,0],p[nl:,1])
            loss+=energy.mean(); grad[nl:]+=g/len(physical)
        return loss,backward(w,cache,grad@B)
    history=[]
    result=minimize(objective,initialize(seed),jac=True,method='L-BFGS-B',callback=lambda w:history.append(float(objective(w)[0])),options={'maxiter':maxiter,'maxcor':30,'ftol':1e-11,'gtol':1e-7,'maxls':40})
    return result.x,{'iterations':result.nit,'converged':bool(result.success),'message':str(result.message),'loss':float(result.fun),'gradient_inf':float(np.max(np.abs(result.jac))),'history':history}
