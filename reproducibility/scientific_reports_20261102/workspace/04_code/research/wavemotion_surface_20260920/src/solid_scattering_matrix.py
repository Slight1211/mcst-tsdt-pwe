"""All-open-port classical 3D FE scattering, unit incoming power per column."""
import numpy as np
from scipy.sparse.linalg import splu,norm as sparse_norm
from scipy.linalg import solve
from classical_solid_fem import bloch_map
from solid_acoustic_fem import face_trace
from acoustics import orders,reject_unresolved_cutoff

def scattering_matrix(mesh,basis,K,M,omega,k,Q,fluid,intorder=8,h=.1):
    reject_unresolved_cutoff(omega,Q,[fluid,fluid]);q,Y,_=orders(omega,Q,**fluid);opened=np.flatnonzero(Y.real>0)
    P=bloch_map(basis,k);D=(P.conj().T@(K-omega**2*M)@P).tocsc()
    Sb,St=[(P.T@face_trace(mesh,Q,side,h,intorder).T).T for side in [-1,1]]
    H=len(Q);ports=[(side,int(j)) for side in [0,1] for j in opened]
    ab=np.zeros((H,len(ports)),complex);at=ab.copy()
    for column,(side,j) in enumerate(ports):(ab if side==0 else at)[j,column]=np.sqrt(2/Y[j].real)
    B=np.vstack([Sb,St]);Z=np.r_[1/Y,1/Y];F=2*(Sb.conj().T@ab-St.conj().T@at)
    lu=splu(D);DF=lu.solve(F);DB=lu.solve(B.conj().T);A=np.eye(2*H)-1j*omega*(B@DB)*Z[None,:]
    trace=solve(A,B@DF);u=DF+1j*omega*DB@(Z[:,None]*trace)
    r=ab+1j*omega/Y[:,None]*(Sb@u);t=at-1j*omega/Y[:,None]*(St@u)
    matrix=np.asarray([np.sqrt(Y[j].real/2)*(r if side==0 else t)[j] for side,j in ports])
    force=1j*omega*B.conj().T@(Z[:,None]*(B@u));err=D@u-force-F
    residual=np.linalg.norm(err)/(sparse_norm(D)*np.linalg.norm(u)+np.linalg.norm(force)+np.linalg.norm(F))
    return dict(scattering=matrix,ports=ports,r=r,t=t,u=P@u,residual=float(residual),
        unitary_error=float(np.linalg.norm(matrix.conj().T@matrix-np.eye(len(ports)))/np.sqrt(len(ports))),
        column_power_error=float(np.max(abs(np.sum(abs(matrix)**2,axis=0)-1))))
