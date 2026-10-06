"""Explicit real-frequency cutoff limit for identical upper/lower fluids only.

No incident grazing wave. At each grazing order, continuity in the outgoing
branch imposes Sg*u=0 and r_g=-t_g. The common-pressure nullspace in the raw
admittance system is removed by this derived limit, not by damping.
"""
import numpy as np
from scipy.linalg import solve
from acoustics import orders

def identical_fluid_limit(D,S,omega,Q,fluid,a,grazing_indices):
    D=np.asarray(D,complex);S=np.asarray(S,complex);a=np.asarray(a,complex);Q=np.asarray(Q,float)
    H=len(a);ids=np.asarray(grazing_indices,dtype=int)
    if len(ids)==0 or len(np.unique(ids))!=len(ids) or np.any((ids<0)|(ids>=H)):raise ValueError('Explicit unique grazing indices required')
    wave2=(omega/fluid['c'])**2;norm2=np.sum(Q*Q,axis=1);tol=16*np.finfo(float).eps*(wave2+norm2)
    mask=np.zeros(H,bool);mask[ids]=True
    if np.any(abs(wave2-norm2[mask])>tol[mask]) or np.any(abs(wave2-norm2[~mask])<=tol[~mask]):raise ValueError('Grazing set inconsistent with exact cutoff')
    if np.any(a[mask]!=0):raise ValueError('No incident grazing field in this limit')
    q,Y,_=orders(omega,Q,**fluid);regular=~mask;C=S[ids]
    if np.any((Y[regular].real==0)&(abs(a[regular])>0)):raise ValueError('Incident waves must be open')
    Z=1/Y[regular];Sr=S[regular]
    L=D-2j*omega*Sr.conj().T@(Z[:,None]*Sr)
    A=np.block([[L,-2*C.conj().T],[C,np.zeros((len(ids),len(ids)),complex)]])
    F=np.r_[2*S.conj().T@a,np.zeros(len(ids))];x=solve(A,F);u=x[:len(D)];rg=x[len(D):]
    r=np.zeros(H,complex);t=r.copy();r[regular]=a[regular]+1j*omega*Z*(Sr@u);t[regular]=-1j*omega*Z*(Sr@u)
    r[ids]=rg;t[ids]=-rg
    # Zero normal flux at exact grazing is the analytic limit, not clipping.
    pi=np.zeros(H);pr=pi.copy();pt=pi.copy()
    pi[regular]=.5*Y[regular].real*abs(a[regular])**2
    pr[regular]=.5*Y[regular].real*abs(r[regular])**2;pt[regular]=.5*Y[regular].real*abs(t[regular])**2
    if pi.sum()<=0:raise ValueError('No incoming power')
    return dict(u=u,r=r,t=t,Pin=pi,PR=pr,PT=pt,R=float(pr.sum()/pi.sum()),T=float(pt.sum()/pi.sum()),
        residual=float(np.linalg.norm(A@x-F)/(np.linalg.norm(A)*np.linalg.norm(x)+np.linalg.norm(F))),
        constraint_residual=float(np.linalg.norm(C@u)/max(np.linalg.norm(u),1e-30)),grazing_indices=ids)
