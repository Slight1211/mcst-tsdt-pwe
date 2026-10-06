"""Real-positive-frequency outgoing half spaces, exp(-i omega t)."""
import numpy as np
from scipy.linalg import solve

def orders(omega,Q,rho,c):
    if not np.isfinite(omega) or omega<=0 or rho<=0 or c<=0:raise ValueError('Invalid fluid/frequency')
    z=(omega/c)**2-np.sum(np.asarray(Q)**2,axis=1)
    q=np.sqrt(z.astype(complex));Y=q/(rho*omega)
    return q,Y,z>0

def reject_unresolved_cutoff(omega,Q,fluids):
    """Reject radicands indistinguishable from zero at floating-point precision.

    This is a classification bound, not a q/Z regularization: no value is replaced.
    """
    norm2=np.sum(np.asarray(Q,dtype=float)**2,axis=1)
    for f in fluids:
        wave2=(omega/f['c'])**2
        roundoff=16*np.finfo(float).eps*(wave2+norm2)
        if np.any(abs(wave2-norm2)<=roundoff):
            raise ValueError('Rayleigh cutoff unresolved at floating-point precision; limiting analysis required')

def fold_incidence(omega,c,theta,phi=0):
    if not 0<=theta<np.pi/2:raise ValueError('Grazing incidence excluded')
    kp=omega/c*np.sin(theta)*np.array([np.cos(phi),np.sin(phi)])
    inc=np.floor((kp+np.pi)/(2*np.pi)).astype(int)
    return kp,kp-2*np.pi*inc,inc

def check_open_channels(omega,k,N,fluids):
    # For square truncation, omitted orders have at least one |index|>=N+1.
    k=np.asarray(k,dtype=float)
    if k.shape!=(2,) or not np.all(np.isfinite(k)):raise ValueError('k must be a finite two-vector')
    if not isinstance(N,(int,np.integer)) or N<0:raise ValueError('N must be a nonnegative integer')
    if not np.all(abs(k)<=np.pi+1e-12):raise ValueError('k must be folded')
    minimum_omitted=min(2*np.pi*(N+1)-abs(k[0]),2*np.pi*(N+1)-abs(k[1]))
    if any(omega/f['c']>=minimum_omitted for f in fluids):raise ValueError('Possible omitted open/grazing channel; increase N')

def scatter(D,S,omega,Q,minus,plus,a,form='impedance',a_top=None):
    """r/t are bottom/top outgoing amplitudes, also for top incidence.

    With simultaneous incidence R/T denote bottom/top output power fractions,
    not reflection/transmission coefficients of one incident beam.
    """
    D=np.asarray(D,complex);S=np.asarray(S,complex);a=np.asarray(a,complex)
    b=np.zeros_like(a) if a_top is None else np.asarray(a_top,complex)
    if b.shape!=a.shape:raise ValueError('Incident arrays must have equal shapes')
    qm,Ym,_=orders(omega,Q,**minus);qp,Yp,_=orders(omega,Q,**plus)
    reject_unresolved_cutoff(omega,Q,[minus,plus])
    if np.any(qm==0) or np.any(qp==0):raise ValueError('Exact grazing threshold: explicit limiting analysis required')
    n,H=D.shape[0],len(a)
    if form=='impedance':
        Zm,Zp=1/Ym,1/Yp
        L=D-1j*omega*(S.conj().T@((Zm+Zp)[:,None]*S));F=2*S.conj().T@(a-b)
        u=solve(L,F);r=a+1j*omega*Zm*(S@u);t=b-1j*omega*Zp*(S@u)
        x=u
    elif form=='admittance':
        L=np.block([[D,-S.conj().T,S.conj().T],[1j*omega*S,-np.diag(Ym),np.zeros((H,H))],[1j*omega*S,np.zeros((H,H)),np.diag(Yp)]])
        F=np.r_[S.conj().T@(a-b),-Ym*a,Yp*b];x=solve(L,F);u,r,t=x[:n],x[n:n+H],x[n+H:]
    else:raise ValueError('Unknown formulation')
    pi_bottom=.5*Ym.real*abs(a)**2;pi_top=.5*Yp.real*abs(b)**2
    pi=pi_bottom+pi_top;pr=.5*Ym.real*abs(r)**2;pt=.5*Yp.real*abs(t)**2
    if pi.sum()<=0 or np.any((Ym.real==0)&(abs(a)>0)) or np.any((Yp.real==0)&(abs(b)>0)):raise ValueError('Incident field must carry propagating power')
    R=pr.sum()/pi.sum();T=pt.sum()/pi.sum()
    res=np.linalg.norm(L@x-F)/(np.linalg.norm(L)*np.linalg.norm(x)+np.linalg.norm(F))
    return dict(u=u,r=r,t=t,power_in=pi,power_in_bottom=pi_bottom,power_in_top=pi_top,power_R=pr,power_T=pt,R=float(R),T=float(T),power_error=float(R+T-1),residual=float(res),condition=float(np.linalg.cond(L)),q_minus=qm,q_plus=qp)

def pressure(z,h,omega,Q,minus,plus,a,r,t,xy,a_top=None):
    """Reconstruct outside plate only; reference amplitudes at physical faces."""
    phase=np.exp(1j*np.asarray(xy)@np.asarray(Q).T)
    if z<=-h/2:
        q,_,_=orders(omega,Q,**minus);coef=a*np.exp(1j*q*(z+h/2))+r*np.exp(-1j*q*(z+h/2))
    elif z>=h/2:
        q,_,_=orders(omega,Q,**plus)
        b=np.zeros_like(a,dtype=complex) if a_top is None else np.asarray(a_top,complex)
        coef=t*np.exp(1j*q*(z-h/2))+b*np.exp(-1j*q*(z-h/2))
    else:raise ValueError('Pressure reconstruction only in half spaces')
    return phase@coef
