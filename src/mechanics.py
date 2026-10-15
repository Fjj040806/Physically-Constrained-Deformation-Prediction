"""Dimensionless planar inextensible Euler elastica; conservative tip load."""
import numpy as np
from scipy.optimize import minimize


def energy_gradient(theta, q, alpha):
    """theta: (batch,n) free nodal rotations; clamped theta(0)=0.
    EI(s)=1+alpha*s. Trapezoidal centerline integration, midpoint bending.
    """
    theta=np.atleast_2d(theta); n=theta.shape[1]; h=1/n
    q=np.asarray(q).reshape(-1,1); alpha=np.asarray(alpha).reshape(-1,1)
    t=np.column_stack([np.zeros(len(theta)),theta])
    ei=1+alpha*((np.arange(n)+.5)/n)[None,:]
    d=np.diff(t,axis=1)
    weights=np.ones(n+1)*h; weights[[0,-1]]*=.5
    e=.5*np.sum(ei*d*d/h,axis=1)-q[:,0]*np.sum(np.sin(t)*weights,axis=1)
    z=ei*d/h
    g=z.copy(); g[:,:-1]-=z[:,1:]
    g-=q*np.cos(theta)*weights[None,1:]
    return e,g


def solve(q,alpha,n=256):
    s=np.linspace(0,1,n+1)
    initial=q*(s[1:]-.5*s[1:]**2)/(1+alpha/2)
    def fun(t):
        e,g=energy_gradient(t,[q],[alpha]); return e[0],g[0]
    res=minimize(fun,initial,jac=True,method='L-BFGS-B',options={'maxiter':3000,'ftol':1e-14,'gtol':1e-8,'maxcor':30})
    if not res.success and np.max(np.abs(res.jac))>2e-5:
        raise RuntimeError(res.message)
    return res.x


def centerline(theta):
    theta=np.atleast_2d(theta); t=np.column_stack([np.zeros(len(theta)),theta]); h=1/theta.shape[1]
    x=np.column_stack([np.zeros(len(theta)),np.cumsum(.5*h*(np.cos(t[:,:-1])+np.cos(t[:,1:])),axis=1)])
    y=np.column_stack([np.zeros(len(theta)),np.cumsum(.5*h*(np.sin(t[:,:-1])+np.sin(t[:,1:])),axis=1)])
    return np.stack([x,y],axis=-1)


def basis(n=64,m=12):
    s=np.linspace(0,1,n+1)[1:]
    # theta(0)=0 exactly; free-end moment is learned through the variational loss.
    return np.sin(np.pi*s[:,None]*(np.arange(m)[None,:]+.5)) / (2*np.arange(m)[None,:]+1)**2
