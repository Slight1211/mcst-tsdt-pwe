"""3D classical solid FE with independent upper/lower Fourier acoustic traces.

Sparse dry factorization plus exact finite-dimensional Schur elimination; no
modal reduction, plate ansatz, damping or forced matrix symmetrization.
"""
import numpy as np
from scipy.sparse.linalg import splu,norm as sparse_norm
from scipy.linalg import solve
from skfem import FacetBasis,ElementVector,ElementTetP2,LinearForm,asm
from classical_solid_fem import bloch_map
from acoustics import orders,reject_unresolved_cutoff

def face_trace(mesh,Q,side,h=.1,intorder=8):
    facets=mesh.facets_satisfying(lambda x:np.isclose(x[2],side*h/2,atol=1e-9,rtol=0))
    fb=FacetBasis(mesh,ElementVector(ElementTetP2()),facets=facets,intorder=intorder)
    trace=[]
    for q in Q:
        @LinearForm(dtype=complex)
        def form(v,w):return np.exp(-1j*(q[0]*w.x[0]+q[1]*w.x[1]))*v[2]
        trace.append(asm(form,fb))
    return np.asarray(trace)

def response(mesh,basis,K,M,omega,k,Q,fluid,a,intorder=8,h=.1):
    reject_unresolved_cutoff(omega,Q,[fluid,fluid]);q,Y,_=orders(omega,Q,**fluid)
    if np.any((Y.real==0)&(abs(a)>0)):raise ValueError('Incident field must carry power')
    P=bloch_map(basis,k);D=(P.conj().T@(K-omega**2*M)@P).tocsc()
    traces=[(P.T@face_trace(mesh,Q,side,h,intorder).T).T for side in [-1,1]]
    Sb,St=traces;B=np.vstack(traces);Z=np.r_[1/Y,1/Y]
    F=2*Sb.conj().T@a;lu=splu(D);DF=lu.solve(F);DB=lu.solve(B.conj().T)
    A=np.eye(len(Z))-1j*omega*(B@DB)*Z[None,:]
    velocity_displacement=solve(A,B@DF)
    u=DF+1j*omega*DB@(Z*velocity_displacement)
    ub,ut=Sb@u,St@u;r=a+1j*omega/Y*ub;t=-1j*omega/Y*ut
    pi=.5*Y.real*abs(a)**2;pr=.5*Y.real*abs(r)**2;pt=.5*Y.real*abs(t)**2
    force=1j*omega*B.conj().T@(Z*(B@u));res=D@u-force-F
    denom=sparse_norm(D)*np.linalg.norm(u)+np.linalg.norm(force)+np.linalg.norm(F)
    return dict(u=P@u,r=r,t=t,Pin=pi,PR=pr,PT=pt,R=float(pr.sum()/pi.sum()),T=float(pt.sum()/pi.sum()),
        residual=float(np.linalg.norm(res)/denom),trace_consistency=float(np.linalg.norm(B@u-velocity_displacement)/max(np.linalg.norm(velocity_displacement),1e-30)),
        reduced_dofs=P.shape[1],q=q)
